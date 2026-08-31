"""Deterministic document-to-model-token materialization."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from ppmoe.config import DatasetSpec
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.provenance import atomic_write_json, sha256_file


class Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    row_index: int
    url: str | None
    text_sha256: str
    token_count: int
    separator_before: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    dataset: DatasetSpec
    tokenizer_repo_id: str
    tokenizer_revision: str
    ordering: str
    shuffle: bool
    target_tokens: int
    total_tokens: int
    context_length: int
    context_count: int
    final_context_length: int
    start_row_index: int
    next_row_index: int
    documents: tuple[DocumentRecord, ...]
    input_ids_sha256: str
    attention_mask_sha256: str
    manifest_sha256: str


def _manifest_payload(manifest: CorpusManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload.pop("manifest_sha256")
    return payload


def materialize_token_stream(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Tokenizer,
    *,
    dataset: DatasetSpec,
    tokenizer_repo_id: str,
    tokenizer_revision: str,
    target_tokens: int | None,
    context_length: int,
    output_dir: Path,
    start_row_index: int = 0,
    excluded_document_ids: frozenset[str] = frozenset(),
) -> CorpusManifest:
    if target_tokens is not None and target_tokens <= 0:
        raise ValueError("target_tokens must be positive when provided")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if start_row_index < 0:
        raise ValueError("start_row_index must be non-negative")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer requires eos_token_id")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    token_ids: list[int] = []
    documents: list[DocumentRecord] = []
    seen_ids: set[str] = set()
    next_row_index = start_row_index
    for row_index, row in enumerate(rows):
        if target_tokens is not None and len(token_ids) >= target_tokens:
            break
        if row_index < start_row_index:
            continue
        next_row_index = row_index + 1
        raw_id = row.get("id")
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError(f"row {row_index} requires a non-empty document id")
        if raw_id in seen_ids:
            raise ValueError(f"duplicate document id: {raw_id}")
        seen_ids.add(raw_id)
        if raw_id in excluded_document_ids:
            raise ValueError(f"excluded document id encountered: {raw_id}")
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"row {row_index} requires text")
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue

        remaining = None if target_tokens is None else target_tokens - len(token_ids)
        separator_before = bool(documents)
        if separator_before:
            token_ids.append(int(tokenizer.eos_token_id))
            if remaining is not None:
                remaining -= 1
        consumed = len(encoded) if remaining is None else max(0, min(len(encoded), remaining))
        token_ids.extend(int(token) for token in encoded[:consumed])
        documents.append(
            DocumentRecord(
                id=raw_id,
                row_index=row_index,
                url=str(row["url"]) if row.get("url") is not None else None,
                text_sha256=sha256(text.encode("utf-8")).hexdigest(),
                token_count=consumed,
                separator_before=separator_before,
                truncated=consumed < len(encoded),
            )
        )
        if target_tokens is not None and len(token_ids) == target_tokens:
            break

    if not token_ids:
        raise ValueError("empty token stream")
    if target_tokens is not None and len(token_ids) != target_tokens:
        raise ValueError(
            f"stream ended at {len(token_ids)} tokens before target {target_tokens}"
        )

    total_tokens = len(token_ids)
    context_count = (total_tokens + context_length - 1) // context_length
    padded_tokens = context_count * context_length
    input_ids = torch.full(
        (context_count, context_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    flattened_ids = input_ids.view(-1)
    flattened_mask = attention_mask.view(-1)
    flattened_ids[:total_tokens] = torch.tensor(token_ids, dtype=torch.long)
    flattened_mask[:total_tokens] = 1

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input_ids.pt"
    mask_path = output_dir / "attention_mask.pt"
    torch.save(input_ids, input_path)
    torch.save(attention_mask, mask_path)
    provisional = CorpusManifest(
        dataset=dataset,
        tokenizer_repo_id=tokenizer_repo_id,
        tokenizer_revision=tokenizer_revision,
        ordering="repository_order",
        shuffle=False,
        target_tokens=total_tokens,
        total_tokens=total_tokens,
        context_length=context_length,
        context_count=context_count,
        final_context_length=(total_tokens - 1) % context_length + 1,
        start_row_index=start_row_index,
        next_row_index=next_row_index,
        documents=tuple(documents),
        input_ids_sha256=sha256_file(input_path),
        attention_mask_sha256=sha256_file(mask_path),
        manifest_sha256="",
    )
    manifest_hash = atomic_write_json(
        output_dir / "manifest.json",
        _manifest_payload(provisional),
    )
    return CorpusManifest(
        **{
            **asdict(provisional),
            "dataset": dataset,
            "documents": tuple(documents),
            "manifest_sha256": manifest_hash,
        }
    )


def load_token_tensors(
    output_dir: Path,
    manifest: CorpusManifest,
) -> tuple[Tensor, Tensor]:
    input_path = output_dir / "input_ids.pt"
    mask_path = output_dir / "attention_mask.pt"
    if sha256_file(input_path) != manifest.input_ids_sha256:
        raise ValueError("input_ids hash does not match manifest")
    if sha256_file(mask_path) != manifest.attention_mask_sha256:
        raise ValueError("attention_mask hash does not match manifest")
    input_ids = torch.load(input_path, map_location="cpu", weights_only=True)
    attention_mask = torch.load(mask_path, map_location="cpu", weights_only=True)
    expected_shape = (manifest.context_count, manifest.context_length)
    if tuple(input_ids.shape) != expected_shape:
        raise ValueError("input_ids shape does not match manifest")
    if tuple(attention_mask.shape) != expected_shape:
        raise ValueError("attention_mask shape does not match manifest")
    if int(attention_mask.sum()) != manifest.total_tokens:
        raise ValueError("attention_mask token count does not match manifest")
    return input_ids, attention_mask


def assert_disjoint_document_ids(
    first: CorpusManifest,
    second: CorpusManifest,
) -> None:
    first_ids = {record.id for record in first.documents}
    second_ids = {record.id for record in second.documents}
    overlap = first_ids & second_ids
    if overlap:
        raise ValueError(f"document id overlap detected: {sorted(overlap)}")


def with_stable_row_ids(
    rows: Iterable[Mapping[str, Any]],
    prefix: str,
) -> Iterable[Mapping[str, Any]]:
    """Yield copies with deterministic IDs only where a source omits one."""

    if not prefix:
        raise ValueError("row-id prefix must be non-empty")
    for row_index, row in enumerate(rows):
        stable = dict(row)
        if not isinstance(stable.get("id"), str) or not stable["id"]:
            stable["id"] = f"{prefix}:{row_index}"
        yield stable


def iter_evaluation_batches(
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    microbatch_size: int,
) -> Iterable[EvaluationBatch]:
    """Yield stable, contiguous evaluation batches without altering tensors."""

    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("token tensors must be two-dimensional")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    for batch_index, start in enumerate(range(0, input_ids.shape[0], microbatch_size)):
        stop = min(start + microbatch_size, input_ids.shape[0])
        yield EvaluationBatch(
            index=batch_index,
            input_ids=input_ids[start:stop],
            attention_mask=attention_mask[start:stop],
        )
