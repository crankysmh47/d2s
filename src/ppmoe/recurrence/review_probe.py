"""Bounded, development-only longer-output diagnostic for recurrence F0."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import time
import tomllib

import torch

from .data import Example, FAMILIES, render_example
from .evaluate import GenerationDeadlineExceeded, generate_answer
from .provenance import canonical_json, sha256_file, sha256_payload


_REQUIRED_CONFIG_KEYS = frozenset({
    "source_run", "source_revision", "output_dir", "max_output_tokens",
    "max_total_tokens", "examples_per_family", "seed", "deadline_seconds", "families",
})
_PINNED_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
_MAX_TOTAL_TOKENS = 512


@dataclass(frozen=True, slots=True)
class ReviewProbeConfig:
    source_run: str
    source_revision: str
    output_dir: str
    max_output_tokens: int
    max_total_tokens: int
    examples_per_family: int
    seed: int
    deadline_seconds: int
    families: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProbeDependencies:
    """Small injection seam for the CPU-only end-to-end test."""

    tokenizer_loader: Callable[..., object]
    model_loader: Callable[..., object]
    model_factory: Callable[[object, ReviewProbeConfig], object]
    device: str
    hardware_probe: Callable[[], Mapping[str, object]]
    clock: Callable[[], float] = time.perf_counter
    synchronize: Callable[[], None] = lambda: None
    progress: Callable[[str], None] = print


def load_config(path: Path) -> ReviewProbeConfig:
    """Load the exact diagnostic schema and enforce its frozen resource bounds."""
    if not isinstance(path, Path):
        raise TypeError("config path must be a Path")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("review probe config is missing or invalid TOML") from error
    if set(raw) != {"probe"} or not isinstance(raw["probe"], dict):
        raise ValueError("review probe config must contain exactly one [probe] table")
    values = raw["probe"]
    if set(values) != _REQUIRED_CONFIG_KEYS:
        raise ValueError("review probe config keys do not match the strict schema")
    for key in ("source_run", "source_revision", "output_dir"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"probe.{key} must be a non-empty string")
    for key in (
        "max_output_tokens", "max_total_tokens", "examples_per_family", "seed", "deadline_seconds"
    ):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"probe.{key} must be an integer")
    families = values["families"]
    if not isinstance(families, list) or any(not isinstance(family, str) for family in families):
        raise ValueError("probe.families must be a string array")
    config = ReviewProbeConfig(
        source_run=values["source_run"],
        source_revision=values["source_revision"].lower(),
        output_dir=values["output_dir"],
        max_output_tokens=values["max_output_tokens"],
        max_total_tokens=values["max_total_tokens"],
        examples_per_family=values["examples_per_family"],
        seed=values["seed"],
        deadline_seconds=values["deadline_seconds"],
        families=tuple(families),
    )
    if config.source_revision != _PINNED_REVISION:
        raise ValueError("probe source revision does not match the pinned V0 snapshot")
    if config.max_output_tokens < 1 or config.max_output_tokens > config.max_total_tokens:
        raise ValueError("probe token bounds require 1 <= max_output_tokens <= max_total_tokens")
    if config.max_total_tokens > _MAX_TOTAL_TOKENS:
        raise ValueError("probe total token bound cannot exceed 512")
    if config.examples_per_family < 1:
        raise ValueError("probe examples_per_family must be positive")
    if config.seed < 0:
        raise ValueError("probe seed must be non-negative")
    if not 1 <= config.deadline_seconds <= 3600:
        raise ValueError("probe deadline_seconds must be between 1 and 3600")
    if config.families != FAMILIES:
        raise ValueError("probe family coverage must exactly match the three registered families")
    return config


def select_balanced_examples(
    rows: Sequence[Mapping[str, object]], config: ReviewProbeConfig
) -> list[Mapping[str, object]]:
    """Select a stable, balanced dev subset without consulting test data."""
    selected: list[Mapping[str, object]] = []
    for family_index, family in enumerate(config.families):
        candidates = [
            row for row in rows
            if row.get("split") == "dev" and row.get("family") == family
        ]
        identifiers = [row.get("id") for row in candidates]
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError(f"dev {family} rows require non-empty string IDs")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"dev {family} rows contain duplicate IDs")
        if len(candidates) < config.examples_per_family:
            raise ValueError(f"dev data does not cover requested family {family}")
        candidates.sort(key=lambda row: str(row["id"]))
        family_random = random.Random(config.seed + family_index)
        selected.extend(family_random.sample(candidates, config.examples_per_family))
    return selected


def run_probe(
    config_path: Path, *, dependencies: ProbeDependencies | None = None
) -> Mapping[str, object]:
    """Run the isolated F0 output diagnostic and persist immutable provenance."""
    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parents[3]
    source_root = _resolve_config_path(config.source_run, project_root)
    output_root = _resolve_config_path(config.output_dir, project_root)
    _assert_no_reparse(source_root)
    _assert_no_reparse(output_root)
    _assert_no_reparse(output_root.parent)
    if output_root.exists():
        raise FileExistsError(f"review probe output already exists: {output_root}")
    if _is_inside(output_root, source_root):
        raise ValueError("review probe output must be separate from its V0 source run")

    source, data_rows, hashes, snapshot = _load_inputs(
        source_root, config, config_path, project_root
    )
    selected = select_balanced_examples(data_rows, config)
    if dependencies is None:
        dependencies = _default_dependencies()
    deadline_start = dependencies.clock()
    deadline = deadline_start + config.deadline_seconds

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    answers_path = output_root / "answers.jsonl"
    summary_path = output_root / "summary.json"
    encoded_ids = [str(row["id"]) for row in selected]
    base_summary: dict[str, object] = {
        "schema_version": 1,
        "output_dir": str(output_root),
        "status": "running",
        "config_sha256": hashes["config"],
        "source_sha256": hashes["source"],
        "data_sha256": hashes["data"],
        "dev_sha256": hashes["dev"],
        "code_sha256": hashes["code"],
        "code_files": hashes["code_files"],
        "selected_example_ids": encoded_ids,
        "selected_count": len(selected),
        "seed": config.seed,
        "families": list(config.families),
        "mode": "single",
        "arm": "F0",
        "split": "dev",
        "max_output_tokens": config.max_output_tokens,
        "max_total_tokens": config.max_total_tokens,
        "deadline_seconds": config.deadline_seconds,
        "started_monotonic": deadline_start,
        "source_snapshot": str(snapshot),
    }
    _write_json(summary_path, base_summary)
    records: list[dict[str, object]] = []
    started = dependencies.clock()
    try:
        tokenizer = dependencies.tokenizer_loader(
            str(snapshot), revision=config.source_revision, local_files_only=True
        )
        stock = dependencies.model_loader(
            str(snapshot), revision=config.source_revision, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True,
        )
        model = dependencies.model_factory(stock, config)
        evaluate_mode = getattr(model, "eval", None)
        if not callable(evaluate_mode):
            raise TypeError("review probe model must provide eval()")
        evaluate_mode()
        if dependencies.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        hardware_start = dict(dependencies.hardware_probe())
        _append_progress(
            dependencies.progress,
            f"PROBE_START examples={len(selected)} output_tokens={config.max_output_tokens} "
            f"device={hardware_start.get('device', 'unknown')}",
        )
        for index, row in enumerate(selected, 1):
            if dependencies.clock() >= deadline:
                raise TimeoutError("review probe deadline reached between examples")
            example = Example(
                id=str(row["id"]), family=str(row["family"]), split="dev",
                structure_id=str(row["structure_id"]), prompt=str(row["prompt"]),
                answer=str(row["answer"]), latent_spec=row["latent_spec"],
            )
            rendered = render_example(tokenizer, example)
            if len(rendered.prompt_input_ids) + config.max_output_tokens > config.max_total_tokens:
                raise ValueError(
                    f"selected prompt {example.id} plus output allowance exceeds total token bound"
                )
            record = _generate_record(
                model, tokenizer, rendered, config, dependencies, deadline
            )
            _append_jsonl(answers_path, record)
            records.append(record)
            _append_progress(
                dependencies.progress,
                f"PROBE_ROW {index}/{len(selected)} id={example.id} "
                f"tokens={record['generated_token_count']} stop={record['stop_reason']}",
            )
            if record["reason_code"] == "deadline_exceeded":
                raise TimeoutError("review probe deadline reached during generation")
        elapsed = max(0.0, dependencies.clock() - started)
        hardware_end = dict(dependencies.hardware_probe())
        completed = {
            **base_summary,
            "status": "complete",
            "record_count": len(records),
            "elapsed_seconds": elapsed,
            "hardware": {"start": hardware_start, "end": hardware_end},
            "generated_tokens": sum(int(record["generated_token_count"]) for record in records),
            "stop_reasons": _count_reasons(records),
            "answers_sha256": sha256_file(answers_path),
        }
        _export_human_review(records, output_root, config.seed)
        _write_json(summary_path, completed)
        _append_progress(dependencies.progress, f"PROBE_COMPLETE records={len(records)} elapsed={elapsed:.3f}s")
        return completed
    except BaseException as error:
        failed = {
            **base_summary,
            "status": "failed",
            "record_count": len(records),
            "elapsed_seconds": max(0.0, dependencies.clock() - started),
            "failure_type": type(error).__name__,
            "failure": str(error),
            "answers_sha256": sha256_file(answers_path) if answers_path.exists() else None,
        }
        if records:
            try:
                _export_human_review(records, output_root, config.seed)
            except BaseException as export_error:
                failed["review_export_failure"] = f"{type(export_error).__name__}: {export_error}"
        _write_json(summary_path, failed)
        _append_progress(dependencies.progress, f"PROBE_FAILED type={type(error).__name__}: {error}")
        raise


def _generate_record(
    model: object,
    tokenizer: object,
    rendered: object,
    config: ReviewProbeConfig,
    dependencies: ProbeDependencies,
    deadline: float,
) -> dict[str, object]:
    started = dependencies.clock()
    input_ids = torch.tensor(
        [rendered.prompt_input_ids], dtype=torch.long, device=dependencies.device
    )

    def check_deadline() -> None:
        if dependencies.clock() >= deadline:
            raise TimeoutError("review probe deadline reached during generation")

    dependencies.synchronize()
    try:
        result = generate_answer(
            model, input_ids, "single", _eos_id(tokenizer), config.max_output_tokens,
            check_deadline=check_deadline,
        )
    except GenerationDeadlineExceeded as error:
        result = error.partial_result
        text = _decode(tokenizer, list(result["token_ids"]))
        record = _record_from_result(rendered, result, text, started, dependencies)
        record["reason_code"] = "deadline_exceeded"
        return record
    dependencies.synchronize()
    token_ids = list(result["token_ids"])
    text = _decode(tokenizer, token_ids)
    return _record_from_result(rendered, result, text, started, dependencies)


def _record_from_result(
    rendered: object,
    result: Mapping[str, object],
    text: str,
    started: float,
    dependencies: ProbeDependencies,
) -> dict[str, object]:
    return {
        "example_id": rendered.example.id,
        "family": rendered.example.family,
        "split": "dev",
        "prompt": rendered.example.prompt,
        "answer": rendered.example.answer,
        "prompt_token_count": len(rendered.prompt_input_ids),
        "completion_token_ids": list(result["completion_token_ids"]),
        "token_ids": list(result["token_ids"]),
        "completion_text": text,
        "generated_token_count": int(result["generated_token_count"]),
        "stop_reason": str(result["stop_reason"]),
        "reason_code": "missing_eos_at_cap" if result["stop_reason"] == "length" else "generated",
        "latency_seconds": max(0.0, dependencies.clock() - started),
        "mode": "single",
        "arm": "F0",
        "seed": None,
    }


def _load_inputs(
    run_root: Path,
    config: ReviewProbeConfig,
    config_path: Path,
    project_root: Path,
) -> tuple[Mapping[str, object], list[Mapping[str, object]], dict[str, object], Path]:
    _assert_no_reparse(run_root)
    if not run_root.is_dir():
        raise ValueError("source run must be a directory")
    source_stage = _read_json(run_root / "stages" / "source.json")
    data_stage = _read_json(run_root / "stages" / "data.json")
    source_manifest_bytes = _read_bytes(run_root / "source.json")
    source = _parse_json(source_manifest_bytes, "source manifest")
    if source_stage.get("status") != "passed" or data_stage.get("status") != "passed":
        raise ValueError("source run source/data stages must both be passed")
    source_payload = _mapping(source_stage.get("payload"), "source stage payload")
    data_payload = _mapping(data_stage.get("payload"), "data stage payload")
    source_signatures = _mapping(source_stage.get("signatures"), "source stage signatures")
    data_signatures = _mapping(data_stage.get("signatures"), "data stage signatures")
    if source_payload.get("revision") != config.source_revision or source.get("revision") != config.source_revision:
        raise ValueError("source run revision does not match pinned configuration")
    source_hash = hashlib.sha256(source_manifest_bytes).hexdigest()
    if source_payload.get("source_manifest_sha256") != source_hash or source_signatures.get("source") != source_hash:
        raise ValueError("source manifest hash does not match source stage evidence")
    if data_signatures.get("source") != source_hash:
        raise ValueError("data stage source signature does not match source manifest")
    if source.get("repo_id") != "Qwen/Qwen3-1.7B" or source.get("model_class") != "Qwen3ForCausalLM":
        raise ValueError("source manifest is not the pinned Qwen3 source")
    relative_snapshot = source_payload.get("snapshot_relative_path")
    if not isinstance(relative_snapshot, str) or Path(relative_snapshot).is_absolute() or ".." in Path(relative_snapshot).parts:
        raise ValueError("source snapshot path is invalid")
    snapshot = run_root / relative_snapshot
    _verify_snapshot_files(snapshot, source)
    tokenizer_revision = source_payload.get("tokenizer_revision")
    tokenizer_hash = source_payload.get("tokenizer_sha256")
    if tokenizer_revision != config.source_revision:
        raise ValueError("source tokenizer revision does not match pinned model revision")
    tokenizer_files = [
        entry for entry in source["files"]
        if isinstance(entry, Mapping) and entry.get("relative_path") == "tokenizer.json"
    ]
    if len(tokenizer_files) != 1 or tokenizer_files[0].get("sha256") != tokenizer_hash:
        raise ValueError("source tokenizer hash does not match pinned model file inventory")

    manifest_path = run_root / "data" / "manifest.json"
    manifest_bytes = _read_bytes(manifest_path)
    data_manifest = _parse_json(manifest_bytes, "data manifest")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if data_payload.get("manifest_sha256") != manifest_hash:
        raise ValueError("data manifest hash does not match data stage evidence")
    data_hash = data_signatures.get("data")
    if not isinstance(data_hash, str) or len(data_hash) != 64:
        raise ValueError("data stage signature is invalid")
    split_hashes = _mapping(data_manifest.get("split_sha256"), "data split hashes")
    dev_path = run_root / "data" / "dev.jsonl"
    dev_bytes = _read_bytes(dev_path)
    dev_hash = hashlib.sha256(dev_bytes).hexdigest()
    if split_hashes.get("dev") != dev_hash:
        raise ValueError("development data hash does not match data manifest")
    rows = _parse_dev_rows(dev_bytes)
    code_paths = (
        project_root / "src" / "ppmoe" / "recurrence" / "review_probe.py",
        project_root / "src" / "ppmoe" / "recurrence" / "evaluate.py",
        project_root / "src" / "ppmoe" / "recurrence" / "data.py",
        project_root / "src" / "ppmoe" / "recurrence" / "model.py",
        project_root / "src" / "ppmoe" / "recurrence" / "human_review.py",
        project_root / "src" / "ppmoe" / "recurrence" / "provenance.py",
        project_root / "scripts" / "run_recurrence_review.py",
    )
    missing_code = [path for path in code_paths if not path.is_file()]
    if missing_code:
        raise ValueError(f"review probe code provenance is missing: {missing_code[0]}")
    code_files = {
        path.relative_to(project_root).as_posix(): sha256_file(path)
        for path in code_paths
    }
    hashes: dict[str, object] = {
        "config": sha256_file(config_path),
        "source": source_hash,
        "data": data_hash,
        "dev": dev_hash,
        "code_files": code_files,
        "code": sha256_payload(code_files),
    }
    return source, rows, hashes, snapshot.resolve(strict=True)


def _verify_snapshot_files(snapshot: Path, source: Mapping[str, object]) -> None:
    _assert_no_reparse(snapshot)
    if not snapshot.is_dir() or snapshot.name != _PINNED_REVISION:
        raise ValueError("pinned model snapshot is missing or invalid")
    entries = source.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source manifest has no pinned model file inventory")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("source model file inventory is invalid")
        relative = entry.get("relative_path")
        expected_size = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("source model file path is invalid")
        path = snapshot / Path(relative)
        _assert_no_reparse(path)
        if not path.is_file() or isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ValueError(f"pinned source file is missing: {relative}")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise ValueError(f"pinned source file hash mismatch: {relative}")


def _parse_dev_rows(payload: bytes) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    try:
        for line_number, line in enumerate(payload.splitlines(), 1):
            row = json.loads(line)
            if not isinstance(row, Mapping) or row.get("split") != "dev":
                raise ValueError(f"development data row {line_number} has an invalid split")
            required = {"id", "family", "split", "structure_id", "prompt", "answer", "latent_spec"}
            if set(row) != required:
                raise ValueError(f"development data row {line_number} has an invalid schema")
            if row["family"] not in FAMILIES or not isinstance(row["latent_spec"], Mapping):
                raise ValueError(f"development data row {line_number} has invalid family/spec")
            rows.append(row)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("development data JSONL is invalid") from error
    if not rows:
        raise ValueError("development data is empty")
    return rows


def _default_dependencies() -> ProbeDependencies:
    if not torch.cuda.is_available():
        raise RuntimeError("the recurrence review probe requires a CUDA device")
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from .model import RecurrentQwen3

    device = "cuda"

    def load_stock(snapshot: str, **kwargs: object) -> object:
        stock = Qwen3ForCausalLM.from_pretrained(snapshot, **kwargs)
        return stock.to(device)

    def factory(stock: object, _config: ReviewProbeConfig) -> object:
        wrapper = RecurrentQwen3(stock, layer_start=12, layer_stop=16)
        return wrapper.to(device)

    def hardware() -> Mapping[str, object]:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return {
            "device": torch.cuda.get_device_name(index),
            "index": index,
            "total_vram_bytes": props.total_memory,
            "allocated_vram_bytes": torch.cuda.memory_allocated(index),
            "reserved_vram_bytes": torch.cuda.memory_reserved(index),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(index),
        }

    return ProbeDependencies(
        tokenizer_loader=AutoTokenizer.from_pretrained,
        model_loader=load_stock,
        model_factory=factory,
        device=device,
        hardware_probe=hardware,
        synchronize=torch.cuda.synchronize,
    )


def _resolve_config_path(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).absolute()


def _read_bytes(path: Path) -> bytes:
    _assert_no_reparse(path)
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"required recurrence review input is unreadable: {path}") from error


def _read_json(path: Path) -> Mapping[str, object]:
    return _parse_json(_read_bytes(path), path.name)


def _parse_json(payload: bytes, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _assert_no_reparse(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            raise ValueError(f"symlink or reparse point is not allowed: {current}")


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = canonical_json(payload)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_json(payload))
        stream.flush()
        os.fsync(stream.fileno())


def _export_human_review(records: Sequence[Mapping[str, object]], output_root: Path, seed: int) -> None:
    if not records:
        return
    from .human_review import export_review

    export_review(records, output_root / "human-review", seed=seed)


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


def _count_reasons(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        reason = str(row["stop_reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _append_progress(progress: Callable[[str], None], message: str) -> None:
    progress(message)
