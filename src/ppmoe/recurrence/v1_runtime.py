"""Local-only F0 runtime and durable exact-answer scoring for recurrence V1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import torch

from .answer_v1 import score_completion
from .data import Example, render_example
from .evaluate import GenerationDeadlineExceeded, generate_answer
from .gates import summarize_exact_scores
from .provenance import canonical_json


class ScoreExecutionError(RuntimeError):
    """Generation failed after preserving any raw rows already produced."""

    def __init__(self, message: str, *, row_count: int, raw_sha256: str, cuda_active_seconds: float, deadline_exceeded: bool = False):
        super().__init__(message)
        self.row_count = row_count
        self.raw_sha256 = raw_sha256
        self.cuda_active_seconds = cuda_active_seconds
        self.deadline_exceeded = deadline_exceeded


class V1DeadlineExceeded(TimeoutError):
    """Registered two-hour calibration wall safety deadline was reached."""


def load_v1_model(snapshot: Path, revision: str) -> tuple[object, object]:
    """Load the pinned tokenizer and Qwen model without network access."""
    if not torch.cuda.is_available():
        raise RuntimeError("V1 calibration requires CUDA")
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    from .model import RecurrentQwen3

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), revision=revision, local_files_only=True
    )
    stock = Qwen3ForCausalLM.from_pretrained(
        str(snapshot), revision=revision, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True,
    ).to("cuda")
    model = RecurrentQwen3(stock, layer_start=12, layer_stop=16).to("cuda").eval()
    return model, tokenizer


def score_v1_examples(
    model: object,
    tokenizer: object,
    examples: Sequence[Example],
    mode: str,
    output_path: Path,
    signatures: Mapping[str, str],
    *,
    max_output_tokens: int = 256,
    max_total_tokens: int = 512,
    deadline: float | None = None,
) -> dict[str, object]:
    """Score examples once, append+fsync each raw row, then validate the complete ID set."""
    if mode != "single":
        raise ValueError("Task 3 V1 scoring supports only mode='single'")
    if not examples:
        raise ValueError("score examples cannot be empty")
    if max_output_tokens != 256 or max_total_tokens != 512:
        raise ValueError("V1 calibration token limits are frozen at 256/512")
    expected_ids = [row.id for row in examples]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate example IDs in score request")
    path = Path(output_path)
    _assert_no_reparse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    row_count = 0
    stream = os.fdopen(fd, "wb")
    active_start = _cuda_event()
    active_end = _cuda_event()
    if active_start is not None:
        active_start.record()
    try:
        device = _model_device(model)
        eos_id = _eos_id(tokenizer)
        for example in examples:
            _check_deadline(deadline)
            rendered = render_example(tokenizer, example)
            if len(rendered.prompt_input_ids) + max_output_tokens > max_total_tokens:
                raise ValueError(f"prompt plus output allowance exceeds V1 token limit for {example.id}")
            input_ids = torch.tensor([rendered.prompt_input_ids], dtype=torch.long, device=device)

            def check_deadline() -> None:
                _check_deadline(deadline)

            try:
                result = generate_answer(
                    model, input_ids, mode, eos_id, max_output_tokens,
                    check_deadline=check_deadline if deadline is not None else None,
                )
            except GenerationDeadlineExceeded as error:
                raise V1DeadlineExceeded(str(error)) from error
            token_ids = list(result["token_ids"])
            completion_ids = list(result["completion_token_ids"])
            completion_text = _decode(tokenizer, completion_ids)
            score = score_completion(example, completion_text, str(result["stop_reason"]))
            normalized = _normalize_score(score)
            record: dict[str, object] = {
                "schema_version": 1,
                "example_id": example.id,
                "family": example.family,
                "split": example.split,
                "mode": mode,
                "prompt_token_count": len(rendered.prompt_input_ids),
                "completion_token_ids": completion_ids,
                "token_ids": token_ids,
                "completion_text": completion_text,
                "generated_token_count": int(result["generated_token_count"]),
                "stop_reason": str(result["stop_reason"]),
                "latency_seconds": float(result.get("latency_seconds", 0.0)),
                "correct": bool(normalized["correct"]),
                "reason_code": normalized["reason_code"],
                "failure_detail": normalized["failure_detail"],
                "parsed_value": normalized.get("parsed_value"),
                "evidence_span": normalized.get("evidence_span"),
                "explanation_status": normalized.get("explanation_status"),
                "explanation_detail": normalized.get("explanation_detail"),
                "scorer_fields": normalized,
                "signatures": dict(signatures),
            }
            stream.write(canonical_json(record))
            stream.flush()
            os.fsync(stream.fileno())
            row_count += 1
    except BaseException as error:
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        active_seconds = _finish_cuda_events(active_end, active_start)
        digest = _sha256(path)
        raise ScoreExecutionError(
            f"scoring stopped after {row_count}/{len(examples)} rows: {type(error).__name__}: {error}",
            row_count=row_count,
            raw_sha256=digest,
            cuda_active_seconds=active_seconds,
            deadline_exceeded=isinstance(error, V1DeadlineExceeded),
        ) from error
    stream.close()
    active_seconds = _finish_cuda_events(active_end, active_start)
    try:
        rows = _read_jsonl(path)
        _validate_exact_rows(rows, expected_ids)
        summary = summarize_exact_scores(rows)
    except BaseException as error:
        raise ScoreExecutionError(
            f"completed raw score file failed validation: {type(error).__name__}: {error}",
            row_count=row_count,
            raw_sha256=_sha256(path),
            cuda_active_seconds=active_seconds,
        ) from error
    return {
        "validated_count": row_count,
        "raw_sha256": _sha256(path),
        "summary": summary,
        "cuda_active_seconds": active_seconds,
    }


def _normalize_score(score: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(score)
    reason = normalized.get("reason_code")
    if normalized.get("correct") is not True and reason not in {"wrong_value", "invalid_format", "ambiguous"}:
        normalized["failure_detail"] = normalized.get("failure_detail") or str(reason or "ungradable")
        normalized["reason_code"] = "invalid_format"
    elif reason == "ambiguous":
        normalized["failure_detail"] = normalized.get("failure_detail") or "ambiguous final answer"
        normalized["reason_code"] = "invalid_format"
    return normalized


def _validate_exact_rows(rows: Sequence[Mapping[str, object]], expected_ids: Sequence[str]) -> None:
    actual = [row.get("example_id") for row in rows]
    if len(actual) != len(expected_ids) or actual != list(expected_ids):
        raise ValueError("score rows must contain each expected ID exactly once in order")
    for row in rows:
        if row.get("correct") not in (True, False):
            raise ValueError("score row correctness must be boolean")
        if not isinstance(row.get("family"), str) or not isinstance(row.get("reason_code"), str):
            raise ValueError("score row family and normalized reason are required")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"score row {line_number} is not an object")
            rows.append(row)
    return rows


def _model_device(model: object) -> str:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return str(next(parameters()).device)
        except StopIteration:
            pass
    return "cuda" if torch.cuda.is_available() and getattr(model, "device", None) == "cuda" else "cpu"


def _eos_id(tokenizer: object) -> int:
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("tokenizer must provide a non-negative integer EOS token ID")
    return value


def _decode(tokenizer: object, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must provide decode")
    return str(decode(token_ids, skip_special_tokens=True))


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise V1DeadlineExceeded("V1 calibration wall safety deadline reached")


def _cuda_event() -> object | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.Event(enable_timing=True)


def _finish_cuda_events(end: object | None, start: object | None) -> float:
    if end is None or start is None:
        return 0.0
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_reparse(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError(f"symlink or reparse point is not allowed: {current}")
