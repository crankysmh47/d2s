"""Deterministic, cacheless evaluation for recurrence V0.

The evaluator deliberately keeps generation and retention scoring small and
explicit.  In particular, every generated token is obtained by a fresh
full-prefix forward and every retention chunk is scored independently, so a
chunk boundary can never create an accidental loss term.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
import numbers
from pathlib import Path
import time

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import RenderedExample, RetentionChunk, parse_answer
from .provenance import canonical_json, sha256_file, sha256_payload, write_immutable_json


_REQUIRED_EXAMPLE_FIELDS = frozenset(
    {
        "example_id",
        "family",
        "prompt_hash",
        "token_ids",
        "completion_token_ids",
        "completion_text",
        "parsed_value",
        "parse_error",
        "correct",
        "reason_code",
        "latency_seconds",
        "arm",
        "seed",
        "split",
        "mode",
        "signatures",
    }
)
_REQUIRED_RETENTION_FIELDS = frozenset(
    {
        "chunk_index",
        "token_count",
        "scored_tokens",
        "nll_sum",
        "mean_nll",
        "source_document_ids",
        "latency_seconds",
        "arm",
        "seed",
        "mode",
        "signatures",
        "failure",
    }
)
_EXAMPLE_REASONS = frozenset(
    {
        "correct",
        "wrong_value",
        "invalid_format",
        "empty_completion",
        "missing_eos_at_cap",
        "decode_failure",
        "execution_error",
    }
)
_MODES = frozenset({"single", "repeat", "damped"})


class GenerationDeadlineExceeded(TimeoutError):
    """Optional generation deadline, with tokens produced before expiry."""

    def __init__(self, message: str, partial_result: Mapping[str, object]) -> None:
        super().__init__(message)
        self.partial_result = dict(partial_result)


def generate_answer(
    model: object,
    input_ids: Tensor,
    mode: str,
    eos_token_id: int,
    max_new_tokens: int = 32,
    *,
    stop_on_eos: bool = True,
    check_deadline: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Greedily decode by recomputing the entire prefix at every step.

    The model is intentionally called with only ``(prefix, mode)``.  This is
    both the public recurrence contract and a guard against accidentally
    introducing a KV-cache into the reference evaluator.
    """
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [1, T]")
    if input_ids.dtype != torch.long or input_ids.shape[0] != 1:
        raise ValueError("input_ids must be one long sequence")
    if input_ids.shape[1] < 1 or input_ids.shape[1] > 512:
        raise ValueError("input length must satisfy 1 <= T <= 512")
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("eos_token_id must be a non-negative integer")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(stop_on_eos, bool):
        raise TypeError("stop_on_eos must be a boolean")
    if check_deadline is not None and not callable(check_deadline):
        raise TypeError("check_deadline must be callable or None")
    if input_ids.shape[1] + max_new_tokens > 512:
        raise ValueError("generation would exceed the 512-token sequence limit")

    prefix = input_ids.detach().clone()
    generated: list[int] = []
    stop_reason = "length"
    started = time.perf_counter()
    _synchronize_cuda()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if check_deadline is not None:
                    try:
                        check_deadline()
                    except TimeoutError as error:
                        partial = {
                            "token_ids": list(generated),
                            "completion_token_ids": list(generated),
                            "stop_reason": "length",
                            "generated_token_count": len(generated),
                            "latency_seconds": max(0.0, time.perf_counter() - started),
                        }
                        raise GenerationDeadlineExceeded(str(error), partial) from error
                # Do not pass use_cache, past_key_values, or any other cache
                # argument.  The wrapper rejects those arguments explicitly.
                output = model(prefix, mode)
                logits = _extract_logits(output)
                if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != prefix.shape[1]:
                    raise ValueError("model must return logits with shape [1, T, vocabulary]")
                if logits.shape[-1] <= eos_token_id:
                    raise ValueError("EOS token is outside the model vocabulary")
                next_token = int(torch.argmax(logits[0, -1], dim=-1).item())
                generated.append(next_token)
                prefix = torch.cat(
                    (prefix, torch.tensor([[next_token]], dtype=torch.long, device=prefix.device)),
                    dim=1,
                )
                if stop_on_eos and next_token == eos_token_id:
                    stop_reason = "eos"
                    break
    finally:
        _synchronize_cuda()
    return {
        "token_ids": generated,
        "completion_token_ids": generated[:-1] if stop_on_eos and generated and generated[-1] == eos_token_id else generated,
        "stop_reason": stop_reason,
        "generated_token_count": len(generated),
        "latency_seconds": max(0.0, time.perf_counter() - started),
    }


def evaluate_examples(
    model: object,
    examples: Sequence[RenderedExample],
    mode: str,
    output_path: Path,
    *,
    arm: str | None = None,
    seed: int | None = None,
    signatures: Mapping[str, str] | None = None,
    eos_token_id: int | None = None,
    max_new_tokens: int = 32,
) -> dict[str, object]:
    """Score a sealed exact-answer sequence and persist one raw record at a time."""
    rows = _validate_examples(examples)
    arm_value, seed_value, signature_value = _resolve_metadata(
        model, mode, arm, seed, signatures
    )
    eos = _resolve_eos(model, eos_token_id)
    if eos is None:
        return _incomplete_result(Path(output_path), "eos_token_id is not available")
    expected_ids = [row.example.id for row in rows]
    expected_rows = {row.example.id: row for row in rows}
    record_path = Path(output_path)
    existing, load_error = _load_jsonl(record_path)
    if load_error is not None:
        return _incomplete_result(record_path, f"malformed score JSONL: {load_error}")
    reconciliation_error = _reconcile_ids(existing, expected_ids, "example_id")
    if reconciliation_error is not None:
        return _incomplete_result(record_path, reconciliation_error)
    reconciliation_error = _validate_example_records(
        existing, expected_rows, mode, arm_value, seed_value, signature_value, eos, max_new_tokens
    )
    if reconciliation_error is not None:
        return _incomplete_result(record_path, reconciliation_error)

    known = {str(record["example_id"]) for record in existing}

    for rendered in rows:
        identifier = rendered.example.id
        if identifier in known:
            continue
        record = _score_example(
            model,
            rendered,
            mode,
            eos,
            max_new_tokens,
            arm_value,
            seed_value,
            signature_value,
        )
        _append_jsonl(record_path, record)
        existing.append(record)
        known.add(identifier)

    return _finalize_examples(
        record_path,
        existing,
        expected_ids,
        rows,
        mode,
        arm_value,
        seed_value,
        signature_value,
        eos,
        max_new_tokens,
    )


def evaluate_retention(
    model: object,
    chunks: Sequence[RetentionChunk],
    mode: str,
    output_path: Path,
    *,
    arm: str | None = None,
    seed: int | None = None,
    signatures: Mapping[str, str] | None = None,
    split: str = "retention",
) -> dict[str, object]:
    """Score each frozen retention chunk independently with shifted causal NLL."""
    frozen_chunks = _validate_chunks(chunks)
    arm_value, seed_value, signature_value = _resolve_metadata(
        model, mode, arm, seed, signatures
    )
    expected_indices = [chunk.index for chunk in frozen_chunks]
    expected_chunks = {chunk.index: chunk for chunk in frozen_chunks}
    record_path = Path(output_path)
    existing, load_error = _load_jsonl(record_path)
    if load_error is not None:
        return _incomplete_result(record_path, f"malformed retention JSONL: {load_error}")
    reconciliation_error = _reconcile_ids(existing, expected_indices, "chunk_index")
    if reconciliation_error is not None:
        return _incomplete_result(record_path, reconciliation_error)
    if split not in {"pilot", "dev", "test", "retention"}:
        raise ValueError("retention evaluation split is invalid")
    reconciliation_error = _validate_retention_records(
        existing, expected_chunks, mode, arm_value, seed_value, signature_value, split
    )
    if reconciliation_error is not None:
        return _incomplete_result(record_path, reconciliation_error)

    known = {int(record["chunk_index"]) for record in existing}
    for chunk in frozen_chunks:
        if chunk.index in known:
            continue
        record = _score_chunk(model, chunk, mode, arm_value, seed_value, signature_value, split)
        _append_jsonl(record_path, record)
        existing.append(record)
        known.add(chunk.index)

    return _finalize_retention(record_path, existing, expected_indices, frozen_chunks, mode, arm_value, seed_value, signature_value, split)


def _score_example(
    model: object,
    rendered: RenderedExample,
    mode: str,
    eos_token_id: int,
    max_new_tokens: int,
    arm: object,
    seed: object,
    signatures: object,
) -> dict[str, object]:
    started = time.perf_counter()
    base: dict[str, object] = {
        "example_id": rendered.example.id,
        "family": rendered.example.family,
        "prompt_hash": sha256_payload({"prompt_input_ids": list(rendered.prompt_input_ids)}),
        "token_ids": [],
        "completion_token_ids": [],
        "completion_text": None,
        "parsed_value": None,
        "parse_error": None,
        "correct": None,
        "reason_code": "execution_error",
        "latency_seconds": 0.0,
        "arm": arm,
        "seed": seed,
        "split": rendered.example.split,
        "mode": mode,
        "signatures": signatures,
    }
    try:
        input_tensor = torch.tensor([rendered.prompt_input_ids], dtype=torch.long, device=_model_device(model))
        generated = generate_answer(model, input_tensor, mode, eos_token_id, max_new_tokens)
        token_ids = list(generated["token_ids"])
        completion_ids = list(generated["completion_token_ids"])
        base["token_ids"] = token_ids
        base["completion_token_ids"] = completion_ids
        stop_reason = str(generated["stop_reason"])
        if stop_reason == "length":
            base["reason_code"] = "missing_eos_at_cap"
            base["correct"] = False
            return _finish_record(base, started)
        if not completion_ids:
            base["reason_code"] = "empty_completion"
            base["correct"] = False
            return _finish_record(base, started)
        try:
            text = _decode_completion(model, completion_ids)
        except Exception as error:
            base["reason_code"] = "decode_failure"
            base["parse_error"] = str(error)
            return _finish_record(base, started)
        base["completion_text"] = text
        try:
            parsed = parse_answer(text, rendered.example)
            base["parsed_value"] = parsed
        except Exception as error:
            base["reason_code"] = "invalid_format"
            base["parse_error"] = str(error)
            base["correct"] = False
            return _finish_record(base, started)
        expected = parse_answer(rendered.example.answer, rendered.example)
        base["correct"] = parsed == expected
        base["reason_code"] = "correct" if base["correct"] else "wrong_value"
    except Exception as error:
        base["reason_code"] = "execution_error"
        base["parse_error"] = str(error)
    return _finish_record(base, started)


def _score_chunk(
    model: object,
    chunk: RetentionChunk,
    mode: str,
    arm: object,
    seed: object,
    signatures: object,
    split: str = "retention",
) -> dict[str, object]:
    started = time.perf_counter()
    base: dict[str, object] = {
        "chunk_index": chunk.index,
        "token_count": len(chunk.input_ids),
        "scored_tokens": chunk.scored_tokens,
        "nll_sum": None,
        "mean_nll": None,
        "source_document_ids": list(chunk.source_document_ids),
        "latency_seconds": 0.0,
        "arm": arm,
        "seed": seed,
        "mode": mode,
        "signatures": signatures,
        "split": split,
        "families": ["retention"],
        "failure": None,
    }
    try:
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in chunk.input_ids):
            raise ValueError("retention token ID must be a non-negative integer")
        input_ids = torch.tensor([chunk.input_ids], dtype=torch.long, device=_model_device(model))
        _synchronize_cuda()
        with torch.inference_mode():
            logits = _extract_logits(model(input_ids, mode))
        _synchronize_cuda()
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(chunk.input_ids):
            raise ValueError("model must return retention logits with shape [1, T, vocabulary]")
        targets = input_ids[:, 1:]
        prediction_logits = logits[:, :-1, :]
        if (
            targets.numel() == 0
            or bool(input_ids.lt(0).any())
            or int(input_ids.max().item()) >= prediction_logits.shape[-1]
            or int(targets.min().item()) < 0
            or int(targets.max().item()) >= prediction_logits.shape[-1]
        ):
            raise ValueError("retention target token is outside model vocabulary")
        dtype_logits = prediction_logits if prediction_logits.dtype in {torch.float32, torch.float64} else prediction_logits.float()
        nll = -F.log_softmax(dtype_logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        if nll.numel() != chunk.scored_tokens:
            raise ValueError("retention gathered token count does not match chunk.scored_tokens")
        if not bool(torch.isfinite(nll).all()):
            raise ValueError("nonfinite retention NLL")
        nll_sum = float(nll.double().sum().item())
        base["nll_sum"] = nll_sum
        base["mean_nll"] = nll_sum / int(nll.numel())
    except Exception as error:
        base["failure"] = str(error)
    return _finish_record(base, started)


def _finalize_examples(
    path: Path,
    records: list[dict[str, object]],
    expected_ids: list[str],
    examples: Sequence[RenderedExample],
    mode: str,
    arm: object,
    seed: object,
    signatures: object,
    eos_token_id: int,
    max_new_tokens: int,
) -> dict[str, object]:
    expected_rows = {row.example.id: row for row in examples}
    error = _reconcile_ids(records, expected_ids, "example_id") or _validate_example_records(
        records, expected_rows, mode, arm, seed, signatures, eos_token_id, max_new_tokens
    )
    if error is not None:
        return _incomplete_result(path, error)
    failures = [record for record in records if record.get("reason_code") in {"execution_error", "decode_failure"}]
    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": "exact_scores",
        "record_count": len(records),
        "expected_count": len(expected_ids),
        "correct": sum(record.get("correct") is True for record in records),
        "incorrect": sum(record.get("correct") is False for record in records),
        "failures": len(failures),
        "complete": not failures,
        "ordered_ids_sha256": sha256_payload({"ids": expected_ids}),
        "mode": mode,
        "arm": arm,
        "seed": seed,
        "signatures": signatures,
        "families": sorted({row["family"] for row in records}),
        "split": next(iter({str(row["split"]) for row in records})),
    }
    if failures:
        summary["complete"] = False
        summary["failure_reason"] = "execution or decoding failure"
        return {**summary, "records_path": str(path), "reconciliation_error": summary["failure_reason"]}
    summary_path = _summary_path(path)
    summary["records_path"] = str(path)
    summary["records_sha256"] = sha256_file(path)
    write_immutable_json(summary_path, summary)
    return {**summary, "records_path": str(path), "summary_path": str(summary_path)}


def _finalize_retention(
    path: Path,
    records: list[dict[str, object]],
    expected_indices: list[int],
    chunks: Sequence[RetentionChunk],
    mode: str,
    arm: object,
    seed: object,
    signatures: object,
    split: str = "retention",
) -> dict[str, object]:
    expected_chunks = {chunk.index: chunk for chunk in chunks}
    error = _reconcile_ids(records, expected_indices, "chunk_index") or _validate_retention_records(
        records, expected_chunks, mode, arm, seed, signatures, split
    )
    if error is not None:
        return _incomplete_result(path, error)
    failures = [record for record in records if record.get("failure")]
    if failures:
        return _incomplete_result(path, "one or more retention chunks failed")
    scored_tokens = sum(int(record["scored_tokens"]) for record in records)
    nll_sum = sum(float(record["nll_sum"]) for record in records)
    mean_nll = nll_sum / scored_tokens if scored_tokens else float("nan")
    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": "retention_scores",
        "record_count": len(records),
        "expected_count": len(expected_indices),
        "scored_tokens": scored_tokens,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "perplexity": float(torch.exp(torch.tensor(mean_nll, dtype=torch.float64)).item()),
        "complete": True,
        "failures": 0,
        "ordered_indices_sha256": sha256_payload({"indices": expected_indices}),
        "mode": mode,
        "arm": arm,
        "seed": seed,
        "signatures": signatures,
        "families": ["retention"],
        "split": split,
    }
    summary_path = _summary_path(path)
    summary["records_path"] = str(path)
    summary["records_sha256"] = sha256_file(path)
    write_immutable_json(summary_path, summary)
    return {**summary, "records_path": str(path), "summary_path": str(summary_path)}


def _validate_examples(examples: Sequence[RenderedExample]) -> list[RenderedExample]:
    rows = list(examples)
    if not rows:
        raise ValueError("exact evaluation requires at least one example")
    if any(not isinstance(row, RenderedExample) for row in rows):
        raise TypeError("examples must contain RenderedExample values")
    ids = [row.example.id for row in rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(set(ids)) != len(ids):
        raise ValueError("examples require unique non-empty IDs")
    return rows


def _resolve_metadata(
    model: object,
    mode: str,
    arm: str | None,
    seed: int | None,
    signatures: Mapping[str, str] | None,
) -> tuple[object, object, dict[str, str]]:
    """Normalize one provenance authority and reject stale wrapper metadata."""
    if mode not in _MODES:
        raise ValueError(f"unsupported evaluation mode: {mode}")
    model_mode = getattr(model, "mode", None)
    if model_mode is not None and model_mode != mode:
        raise ValueError("model mode conflicts with explicit mode")

    model_arm = getattr(model, "arm", None)
    if arm is not None:
        _validate_arm(arm)
        if model_arm is not None and model_arm != arm:
            raise ValueError("model arm conflicts with explicit arm")
        arm_value: object = arm
    elif model_arm is not None:
        _validate_arm(model_arm)
        arm_value = model_arm
    else:
        arm_value = "unknown"

    model_seed = getattr(model, "seed", None)
    if seed is not None:
        _validate_seed(seed)
        if model_seed is not None and model_seed != seed:
            raise ValueError("model seed conflicts with explicit seed")
        seed_value: object = seed
    elif model_seed is not None:
        _validate_seed(model_seed)
        seed_value = model_seed
    else:
        seed_value = None

    explicit_signatures = _normalize_signatures(signatures) if signatures is not None else None
    model_signatures_raw = getattr(model, "signatures", None)
    model_signatures = (
        _normalize_signatures(model_signatures_raw)
        if model_signatures_raw is not None
        else None
    )
    if explicit_signatures is not None:
        if model_signatures is not None and model_signatures != explicit_signatures:
            raise ValueError("model signatures conflict with explicit signatures")
        signature_value = explicit_signatures
    elif model_signatures is not None:
        signature_value = model_signatures
    else:
        signature_value = {}
    return arm_value, seed_value, signature_value


def _validate_arm(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("arm must be a non-empty string")


def _validate_seed(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("seed must be a non-negative integer")


def _normalize_signatures(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("signatures must be a mapping")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("signature names must be non-empty strings")
        if not isinstance(item, str) or not item:
            raise ValueError(f"signature {key!r} must be a non-empty string")
        normalized[key] = item
    return dict(sorted(normalized.items()))


def _validate_example_records(
    records: Sequence[Mapping[str, object]],
    expected_rows: Mapping[str, RenderedExample],
    mode: str,
    arm: object,
    seed: object,
    signatures: Mapping[str, str],
    eos_token_id: int,
    max_new_tokens: int,
) -> str | None:
    for record in records:
        identifier = record.get("example_id")
        if not isinstance(identifier, str) or identifier not in expected_rows:
            return f"unexpected example_id: {identifier}"
        missing = sorted(_REQUIRED_EXAMPLE_FIELDS.difference(record))
        if missing:
            return f"record {identifier} is missing fields: {', '.join(missing)}"
        expected = expected_rows[identifier]
        if record.get("family") != expected.example.family:
            return f"family mismatch for {identifier}"
        if record.get("prompt_hash") != sha256_payload({"prompt_input_ids": list(expected.prompt_input_ids)}):
            return f"prompt hash mismatch for {identifier}"
        if record.get("split") != expected.example.split:
            return f"split mismatch for {identifier}"
        if record.get("mode") != mode:
            return f"mode mismatch for {identifier}"
        if record.get("arm") != arm:
            return f"arm mismatch for {identifier}"
        if record.get("seed") != seed:
            return f"seed mismatch for {identifier}"
        if record.get("signatures") != dict(signatures):
            return f"signatures mismatch for {identifier}"
        if not _finite_nonnegative(record.get("latency_seconds")):
            return f"latency is invalid for {identifier}"
        token_ids = record.get("token_ids")
        completion_ids = record.get("completion_token_ids")
        if not _token_id_list(token_ids) or not _token_id_list(completion_ids):
            return f"token IDs are invalid for {identifier}"
        expected_completion = (
            token_ids[:-1] if token_ids and token_ids[-1] == eos_token_id else token_ids
        )
        if completion_ids != expected_completion:
            return f"completion token IDs are inconsistent for {identifier}"
        reason = record.get("reason_code")
        if not isinstance(reason, str) or reason not in _EXAMPLE_REASONS:
            return f"reason code is invalid for {identifier}"
        if not _validate_example_semantics(record, expected, reason, max_new_tokens, eos_token_id):
            return f"reason/correctness semantics are invalid for {identifier}"
    return None


def _validate_example_semantics(
    record: Mapping[str, object],
    rendered: RenderedExample,
    reason: str,
    max_new_tokens: int,
    eos_token_id: int,
) -> bool:
    correct = record.get("correct")
    parsed = record.get("parsed_value")
    parse_error = record.get("parse_error")
    text = record.get("completion_text")
    token_ids = record.get("token_ids")
    completion_ids = record.get("completion_token_ids")
    if correct is not True and correct is not False and correct is not None:
        return False
    if parse_error is not None and (not isinstance(parse_error, str) or not parse_error):
        return False
    if reason == "correct":
        if correct is not True or not isinstance(text, str) or parse_error is not None or parsed is None:
            return False
        try:
            return parsed == parse_answer(text, rendered.example) == parse_answer(rendered.example.answer, rendered.example)
        except Exception:
            return False
    if reason == "wrong_value":
        if correct is not False or not isinstance(text, str) or parse_error is not None or parsed is None:
            return False
        try:
            return parsed == parse_answer(text, rendered.example) and parsed != parse_answer(rendered.example.answer, rendered.example)
        except Exception:
            return False
    if reason == "invalid_format":
        if correct is not False or not isinstance(text, str) or parsed is not None or not parse_error:
            return False
        try:
            parse_answer(text, rendered.example)
        except Exception:
            return True
        return False
    if reason == "empty_completion":
        return correct is False and parsed is None and parse_error is None and text is None and token_ids == [eos_token_id] and completion_ids == []
    if reason == "missing_eos_at_cap":
        return correct is False and parsed is None and parse_error is None and text is None and isinstance(token_ids, list) and len(token_ids) == max_new_tokens and eos_token_id not in token_ids and completion_ids == token_ids
    if reason == "decode_failure":
        return correct is None and parsed is None and text is None and bool(parse_error) and isinstance(completion_ids, list) and bool(completion_ids)
    if reason == "execution_error":
        return correct is None and parsed is None and text is None and bool(parse_error)
    return False


def _validate_retention_records(
    records: Sequence[Mapping[str, object]],
    expected_chunks: Mapping[int, RetentionChunk],
    mode: str,
    arm: object,
    seed: object,
    signatures: Mapping[str, str],
    split: str = "retention",
) -> str | None:
    for record in records:
        index = record.get("chunk_index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in expected_chunks:
            return f"unexpected chunk_index: {index}"
        missing = sorted(_REQUIRED_RETENTION_FIELDS.difference(record))
        if missing:
            return f"record {index} is missing fields: {', '.join(missing)}"
        expected = expected_chunks[index]
        if record.get("token_count") != len(expected.input_ids):
            return f"token_count mismatch for chunk {index}"
        if record.get("scored_tokens") != expected.scored_tokens:
            return f"scored_tokens mismatch for chunk {index}"
        if record.get("source_document_ids") != list(expected.source_document_ids):
            return f"source document mismatch for chunk {index}"
        if record.get("mode") != mode:
            return f"mode mismatch for chunk {index}"
        if record.get("arm") != arm:
            return f"arm mismatch for chunk {index}"
        if record.get("seed") != seed:
            return f"seed mismatch for chunk {index}"
        if record.get("signatures") != dict(signatures):
            return f"signatures mismatch for chunk {index}"
        if record.get("split", "retention") != split:
            return f"split mismatch for chunk {index}"
        if not _finite_nonnegative(record.get("latency_seconds")):
            return f"latency is invalid for chunk {index}"
        failure = record.get("failure")
        if failure is not None:
            if not isinstance(failure, str) or not failure:
                return f"failure field is invalid for chunk {index}"
            if record.get("nll_sum") is not None or record.get("mean_nll") is not None:
                return f"failed chunk {index} contains NLL values"
            continue
        nll_sum = record.get("nll_sum")
        mean_nll = record.get("mean_nll")
        if not _finite_nonnegative(nll_sum) or not _finite_nonnegative(mean_nll):
            return f"NLL is invalid for chunk {index}"
        if expected.scored_tokens <= 0 or not math.isclose(
            float(mean_nll), float(nll_sum) / expected.scored_tokens, rel_tol=1e-10, abs_tol=1e-12
        ):
            return f"NLL mean mismatch for chunk {index}"
    return None


def _finite_nonnegative(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0.0


def _token_id_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value)


def _validate_chunks(chunks: Sequence[RetentionChunk]) -> list[RetentionChunk]:
    rows = list(chunks)
    if not rows:
        raise ValueError("retention evaluation requires at least one chunk")
    if any(not isinstance(row, RetentionChunk) for row in rows):
        raise TypeError("chunks must contain RetentionChunk values")
    indices = [row.index for row in rows]
    if indices != list(range(len(indices))):
        raise ValueError("retention chunks must be ordered from index zero")
    return rows


def _extract_logits(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    if isinstance(output, Mapping) and isinstance(output.get("logits"), Tensor):
        return output["logits"]  # type: ignore[return-value]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("model output does not contain logits")


def _decode_completion(model: object, token_ids: Sequence[int]) -> str:
    candidates = [getattr(model, "tokenizer", None), getattr(getattr(model, "stock_model", None), "tokenizer", None)]
    for candidate in candidates:
        decoder = getattr(candidate, "decode", None)
        if callable(decoder):
            try:
                value = decoder(list(token_ids), skip_special_tokens=True)
            except TypeError:
                value = decoder(list(token_ids))
            if not isinstance(value, str):
                raise TypeError("tokenizer decode must return text")
            return value
    decoder = getattr(model, "decode", None)
    if callable(decoder):
        value = decoder(list(token_ids))
        if not isinstance(value, str):
            raise TypeError("model decode must return text")
        return value
    raise RuntimeError("no tokenizer decoder is available")


def _model_device(model: object) -> torch.device:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except (StopIteration, TypeError):
            pass
    return torch.device("cpu")


def _resolve_eos(model: object, explicit: int | None) -> int | None:
    value = explicit
    if value is None:
        value = getattr(getattr(model, "stock_model", None), "config", None)
        value = getattr(value, "eos_token_id", None)
    if value is None:
        value = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _metadata(model: object, name: str, fallback: object) -> object:
    value = getattr(model, name, fallback)
    if name == "signatures" and isinstance(value, Mapping):
        return dict(value)
    return value


def _finish_record(record: dict[str, object], started: float) -> dict[str, object]:
    record["latency_seconds"] = max(float(record.get("latency_seconds", 0.0)), max(0.0, time.perf_counter() - started))
    return record


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(record)
    needs_separator = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("ab") as stream:
        if needs_separator:
            stream.write(b"\n")
        stream.write(encoded)
        stream.flush()
        import os

        os.fsync(stream.fileno())


def _load_jsonl(path: Path) -> tuple[list[dict[str, object]], str | None]:
    if not path.exists():
        return [], None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        return [], str(error)
    records: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            return [], f"line {number}: {error}"
        if not isinstance(value, dict):
            return [], f"line {number}: record is not an object"
        records.append(value)
    return records, None


def _reconcile_ids(records: Sequence[Mapping[str, object]], expected: Sequence[object], field: str) -> str | None:
    actual: list[object] = []
    for record in records:
        value = record.get(field)
        if value in actual:
            return f"duplicate {field}: {value}"
        actual.append(value)
    unexpected = [value for value in actual if value not in expected]
    if unexpected:
        return f"unexpected {field}: {unexpected[0]}"
    if actual != list(expected[: len(actual)]):
        return f"out-of-order {field} records"
    return None


def _validate_record_schema(records: Sequence[Mapping[str, object]], required: frozenset[str]) -> str | None:
    for record in records:
        missing = sorted(required.difference(record))
        if missing:
            return f"record is missing fields: {', '.join(missing)}"
    return None


def _incomplete_result(path: Path, error: str) -> dict[str, object]:
    return {"complete": False, "records_path": str(path), "reconciliation_error": error}


def _summary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.summary.json")


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
