"""Immutable prepare/calibrate/report stages for the separate recurrence V1 study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import time
from typing import Any

import torch

from .config_v1 import V1Config, load_v1_config
from .data import Example
from .data_v1 import FAMILIES, freeze_v1_split, make_pilot, make_selected_splits
from .gates import is_informative_family, summarize_exact_scores
from .provenance import canonical_json, sha256_file, sha256_payload
from .v1_runtime import ScoreExecutionError, load_v1_model, score_v1_examples


_FROZEN_SCORER_SHA256 = "2f138e6fe106c81dd03712d0a49a3a916cec2596a4acfa8f4e111f6463bab719"
_STAGES = {"prepare", "calibrate", "report"}
_TERMINAL = {"stopped_scientific", "stopped_resource", "failed_environment", "passed"}


class ResourceLimitReached(RuntimeError):
    """A frozen V1 compute or artifact budget has been reached."""


class MeasuredPhaseFailure(RuntimeError):
    """Failed CUDA phase carrying time accrued before its exception."""

    def __init__(self, cause: BaseException, cuda_active_seconds: float):
        super().__init__(str(cause))
        self.cuda_active_seconds = cuda_active_seconds


def run_v1_stage(stage: str, config_path: Path, run_root: Path | None = None) -> dict[str, object]:
    """Run exactly one V1 stage. Later study stages remain deliberately unavailable."""
    if stage not in _STAGES:
        raise ValueError(f"unsupported V1 stage {stage!r}; implemented stages are prepare, calibrate, report")
    config_path = Path(config_path).resolve(strict=True)
    config = load_v1_config(config_path)
    project_root = Path(__file__).resolve().parents[3]
    if stage == "prepare":
        prepared = _verify_source(config, config_path, project_root)
        run_id = _derive_run_id(prepared["config_bytes"], prepared["signatures"])
        expected_root = _resolve(config.data.output_root, project_root) / run_id
        if run_root is not None and Path(run_root).absolute() != expected_root.absolute():
            raise ValueError("prepare run root must equal configured output root plus deterministic run ID")
        return _prepare(expected_root, run_id, config_path, config, prepared)
    if run_root is None:
        raise ValueError(f"{stage} requires a run root")
    root = Path(run_root).absolute()
    _assert_no_reparse(root)
    if stage == "calibrate":
        try:
            return _calibrate(root, config, config_path, project_root)
        except Exception as error:
            calibration_path = root / "stages" / "calibrate.json"
            if calibration_path.exists():
                raise
            prepare = _read_stage(root / "stages" / "prepare.json", "prepare")
            prepare_payload = _mapping(prepare.get("payload"), "prepare payload")
            payload = {
                "run_id": prepare_payload.get("run_id"), "status": "failed_environment",
                "failure": f"preflight {type(error).__name__}: {error}",
                "signatures": dict(_mapping(prepare_payload.get("signatures"), "prepared signatures")),
                "resources": {"calibration_cuda_active_seconds": _ledger_seconds(root), "artifact_bytes": _tree_bytes(root)},
                "runtime_code_files": _runtime_code_files(project_root),
                "next_stage": "report",
            }
            return _finish_calibration(root, payload, "failed_environment")
    return _report(root, config, config_path)


def _derive_run_id(config_bytes: bytes, signatures: Mapping[str, str]) -> str:
    return sha256_payload({"config_bytes_sha256": hashlib.sha256(config_bytes).hexdigest(), "signatures": dict(signatures)})


def _verify_source(config: V1Config, config_path: Path, project_root: Path) -> dict[str, Any]:
    source_root = _resolve(config.source_run, project_root)
    _assert_no_reparse(source_root)
    if not source_root.is_dir():
        raise ValueError("V0 source run is missing")
    source_record = _read_json(source_root / "stages" / "source.json")
    data_record = _read_json(source_root / "stages" / "data.json")
    if source_record.get("stage") != "source" or source_record.get("status") != "passed":
        raise ValueError("V0 source stage must be passed")
    if data_record.get("stage") != "data" or data_record.get("status") != "passed":
        raise ValueError("V0 data stage must be passed")
    source_manifest_bytes = _read_bytes(source_root / "source.json")
    source = _parse_json(source_manifest_bytes, "V0 source manifest")
    source_hash = hashlib.sha256(source_manifest_bytes).hexdigest()
    source_payload = _mapping(source_record.get("payload"), "V0 source payload")
    source_signatures = _mapping(source_record.get("signatures"), "V0 source signatures")
    data_payload = _mapping(data_record.get("payload"), "V0 data payload")
    data_signatures = _mapping(data_record.get("signatures"), "V0 data signatures")
    if (
        source.get("repo_id") != config.model_repo_id
        or source.get("model_class") != "Qwen3ForCausalLM"
        or source.get("revision") != config.model_revision
        or source_payload.get("revision") != config.model_revision
        or source_payload.get("source_manifest_sha256") != source_hash
        or source_signatures.get("source") != source_hash
        or data_signatures.get("source") != source_hash
    ):
        raise ValueError("V0 source stage does not bind the pinned Qwen source manifest")
    relative_snapshot = source_payload.get("snapshot_relative_path")
    if not isinstance(relative_snapshot, str) or Path(relative_snapshot).is_absolute() or ".." in Path(relative_snapshot).parts:
        raise ValueError("pinned source snapshot path is invalid")
    snapshot = source_root / relative_snapshot
    if snapshot.name != config.model_revision:
        raise ValueError("pinned model snapshot directory name does not match the immutable revision")
    _verify_snapshot(snapshot, source)
    tokenizer_hash = source_payload.get("tokenizer_sha256")
    if source_payload.get("tokenizer_revision") != config.model_revision:
        raise ValueError("pinned tokenizer revision mismatch")
    inventory = source.get("files")
    tokenizer_entries = [row for row in inventory if isinstance(row, Mapping) and row.get("relative_path") == "tokenizer.json"] if isinstance(inventory, list) else []
    if len(tokenizer_entries) != 1 or tokenizer_entries[0].get("sha256") != tokenizer_hash:
        raise ValueError("tokenizer hash is not bound by the pinned source inventory")

    manifest_bytes = _read_bytes(source_root / "data" / "manifest.json")
    manifest = _parse_json(manifest_bytes, "V0 data manifest")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if data_payload.get("manifest_sha256") != manifest_hash or data_payload.get("data_hash") != data_signatures.get("data"):
        raise ValueError("V0 data stage does not bind its manifest and data signature")
    counts = _mapping(manifest.get("counts"), "V0 split counts")
    if counts.get("train") != 8000 or counts.get("dev") != 600:
        raise ValueError("V0 train/dev source counts differ from the registered 8,000/600")
    split_hashes = _mapping(manifest.get("split_sha256"), "V0 split hashes")
    train_path, dev_path = source_root / "data" / "train.jsonl", source_root / "data" / "dev.jsonl"
    train_bytes, dev_bytes = _read_bytes(train_path), _read_bytes(dev_path)
    train_hash, dev_hash = hashlib.sha256(train_bytes).hexdigest(), hashlib.sha256(dev_bytes).hexdigest()
    if split_hashes.get("train") != train_hash or split_hashes.get("dev") != dev_hash:
        raise ValueError("V0 train/dev split raw hash differs from its data manifest")
    train_rows = _parse_rows(train_bytes, "train", 8000)
    dev_rows = _parse_rows(dev_bytes, "dev", 600)
    forbidden = {str(row["id"]) for row in train_rows} | {str(row["id"]) for row in dev_rows}
    if len(forbidden) != 8600:
        raise ValueError("V0 train/dev source IDs are not unique")
    if sha256_payload({"ids": sorted(row["id"] for row in train_rows)}) != _mapping(manifest.get("split_id_sha256"), "V0 split ID hashes").get("train"):
        raise ValueError("V0 training IDs do not match manifest split ID hash")
    if sha256_payload({"ids": sorted(row["id"] for row in dev_rows)}) != _mapping(manifest.get("split_id_sha256"), "V0 split ID hashes").get("dev"):
        raise ValueError("V0 development IDs do not match manifest split ID hash")

    config_bytes = config_path.read_bytes()
    scorer_path = project_root / "src" / "ppmoe" / "recurrence" / "answer_v1.py"
    generator_path = project_root / "src" / "ppmoe" / "recurrence" / "data_v1.py"
    scorer_hash, generator_hash = sha256_file(scorer_path), sha256_file(generator_path)
    if scorer_hash != _FROZEN_SCORER_SHA256:
        raise ValueError("V1 scorer SHA256 differs from the frozen specification")
    signatures = {
        "source": source_hash,
        "model": sha256_payload({"revision": config.model_revision, "files": inventory}),
        "tokenizer": str(tokenizer_hash),
        "train": train_hash,
        "dev": dev_hash,
        "scorer": scorer_hash,
        "generator": generator_hash,
    }
    return {
        "config_bytes": config_bytes,
        "signatures": signatures,
        "source": source,
        "source_root": source_root,
        "snapshot": snapshot.resolve(strict=True),
        "forbidden_ids": forbidden,
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "manifest_sha256": manifest_hash,
    }


def _verify_snapshot(snapshot: Path, source: Mapping[str, object]) -> None:
    _assert_no_reparse(snapshot)
    if not snapshot.is_dir():
        raise ValueError("pinned local model snapshot is missing")
    entries = source.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source file inventory is missing")
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("source file inventory entry is invalid")
        relative, expected_size, expected_hash = entry.get("relative_path"), entry.get("bytes"), entry.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("source file inventory path is invalid")
        if relative in seen_paths:
            raise ValueError("source file inventory paths must be unique")
        seen_paths.add(relative)
        target = snapshot / Path(relative)
        _assert_no_reparse(target)
        if not target.is_file() or isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ValueError(f"pinned model file missing: {relative}")
        if target.stat().st_size != expected_size or sha256_file(target) != expected_hash:
            raise ValueError(f"pinned model file hash mismatch: {relative}")


def _prepare(root: Path, run_id: str, config_path: Path, config: V1Config, inputs: Mapping[str, Any]) -> dict[str, object]:
    _assert_no_reparse(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=False, exist_ok=False)
    signatures = dict(inputs["signatures"])
    payload: dict[str, object] = {
        "run_id": run_id,
        "run_root": str(root),
        "config_sha256": hashlib.sha256(inputs["config_bytes"]).hexdigest(),
        "source_root": str(inputs["source_root"]),
        "snapshot_relative_path": str(Path(inputs["snapshot"]).relative_to(inputs["source_root"])),
        "source_manifest_sha256": signatures["source"],
        "data_manifest_sha256": inputs["manifest_sha256"],
        "signatures": signatures,
        "v0_train_count": inputs["train_count"],
        "v0_dev_count": inputs["dev_count"],
        "v0_train_dev_forbidden_id_count": len(inputs["forbidden_ids"]),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        for directory in (root / "stages", root / "data", root / "scores", root / "resources"):
            _assert_no_reparse(directory)
            directory.mkdir()
        _append_ledger(root, {"event": "prepare", "cumulative_cuda_active_seconds": 0.0, "artifact_bytes": 0})
        record = _stage_record("prepare", "passed", payload, {})
        _write_new(root / "stages" / "prepare.json", record)
    except BaseException as error:
        record = _stage_record("prepare", "failed_environment", {**payload, "failure": f"{type(error).__name__}: {error}"}, {})
        _write_new(root / "stages" / "prepare.json", record)
        return {"run_id": run_id, "status": "failed_environment", "next_stage": "report", "run_root": str(root), "failure": str(error)}
    return {"run_id": run_id, "status": "passed", "next_stage": "calibrate", "run_root": str(root), "stage_sha256": record["stage_sha256"]}


def _calibrate(root: Path, config: V1Config, config_path: Path, project_root: Path) -> dict[str, object]:
    prepare_path = root / "stages" / "prepare.json"
    prepare_record = _read_stage(prepare_path, "prepare")
    if prepare_record.get("status") != "passed":
        raise ValueError("calibrate requires a passed prepare stage")
    if (root / "stages" / "calibrate.json").exists():
        raise FileExistsError("calibration stage already exists; retries require a new run identity")
    payload = _mapping(prepare_record.get("payload"), "prepare payload")
    if payload.get("config_sha256") != sha256_file(config_path):
        raise ValueError("config hash changed after prepare")
    fresh = _verify_source(config, config_path, project_root)
    frozen = _mapping(payload.get("signatures"), "prepared signatures")
    if dict(frozen) != fresh["signatures"]:
        raise ValueError("source, tokenizer, data, scorer, or generator hash changed after prepare")
    signatures = dict(frozen)
    runtime_code_files = _runtime_code_files(project_root)
    runtime_code_sha256 = sha256_payload(runtime_code_files)
    signatures["runtime_code"] = runtime_code_sha256
    snapshot = Path(str(fresh["snapshot"]))
    if Path(str(payload.get("source_root"))).resolve(strict=True) != Path(fresh["source_root"]).resolve(strict=True):
        raise ValueError("source run path changed after prepare")
    deadline = time.monotonic() + config.resources.maximum_calibration_cuda_seconds
    cumulative = _ledger_seconds(root)
    score_manifest: dict[str, object] = {}
    pilot_table: list[dict[str, object]] = []
    dev_summary: dict[str, object] | None = None
    dev_split_evidence: dict[str, object] | None = None
    active_score_path: Path | None = None
    sealed_meta: dict[str, object] | None = None
    status = "passed"
    failure: str | None = None
    try:
        _budget_check(root, cumulative, config)
        _append_ledger(root, {"event": "before_model_load", "cumulative_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root)})
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, tokenizer, model_load_seconds = _load_calibration_runtime_measured(snapshot, config.model_revision)
        cumulative += model_load_seconds
        _append_ledger(root, {"event": "model_loaded", "phase_cuda_active_seconds": model_load_seconds, "cumulative_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root)})
        forbidden = set(fresh["forbidden_ids"])
        pilot_examples: dict[tuple[str, int], tuple[Example, ...]] = {}
        for family in FAMILIES:
            for candidate in (1, 2):
                _budget_check(root, cumulative, config)
                _append_ledger(root, {"event": "before_pilot", "family": family, "candidate": candidate, "cumulative_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root)})
                examples = make_pilot(config, family, candidate, forbidden)
                pilot_examples[(family, candidate)] = examples
                forbidden.update(row.id for row in examples)
                split_path = root / "data" / f"pilot-{family}-c{candidate}.jsonl"
                split_manifest = freeze_v1_split(
                    split_path, examples, tokenizer, revision=config.model_revision,
                    tokenizer_sha256=str(signatures["tokenizer"]),
                )
                _verify_v1_split(split_path, str(split_manifest["sha256"]), len(examples))
                score_path = root / "scores" / f"pilot-{family}-c{candidate}.jsonl"
                active_score_path = score_path
                score_result = score_v1_examples(
                    model, tokenizer, examples, "single", score_path, signatures,
                    max_output_tokens=256, max_total_tokens=512, deadline=deadline,
                )
                cumulative += float(score_result["cuda_active_seconds"])
                _append_ledger(root, {"event": "raw_score_file_written", "path": str(score_path.relative_to(root)), "row_count": score_result["validated_count"], "raw_sha256": score_result["raw_sha256"], "phase_cuda_active_seconds": score_result["cuda_active_seconds"], "cumulative_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root)})
                summary = _validate_score_file(score_path, examples, score_result)
                score_manifest[f"pilot-{family}-c{candidate}"] = {
                    "split_sha256": split_manifest["sha256"],
                    "split_manifest_sha256": sha256_file(Path(str(split_path) + ".manifest.json")),
                    "split_count": len(examples),
                    "score_sha256": score_result["raw_sha256"],
                    "score_count": score_result["validated_count"],
                    "summary": summary,
                }
                _budget_check(root, cumulative, config)
                table_row = _pilot_summary(family, candidate, summary)
                pilot_table.append(table_row)
        choices, fallback = _select_pilot_difficulties(pilot_table)
        if len(choices) < 2:
            status = "stopped_scientific"
            result_payload: dict[str, object] = {
                "run_id": payload["run_id"], "status": status, "pilot_table": pilot_table,
                "pilot_scores": score_manifest, "eligible_families": sorted(choices),
                "selected_difficulty": {}, "dev": None, "sealed": None,
                "cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root),
                "signatures": signatures, "next_stage": "report",
                "runtime_code_files": runtime_code_files,
                "resources": {"calibration_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root), "cuda": _cuda_resource_snapshot()},
            }
            return _finish_calibration(root, result_payload, status)
        for row in pilot_table:
            row["selected"] = row["candidate"] == choices[str(row["family"])]
            row["selection_kind"] = "selected_first_eligible" if row["family"] not in fallback and row["selected"] else "prespecified_candidate_1_fallback" if row["family"] in fallback and row["selected"] else "not_selected"
        _budget_check(root, cumulative, config)
        _append_ledger(root, {"event": "before_development", "cumulative_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root)})
        splits = make_selected_splits(config, choices, forbidden)
        dev_examples = splits["dev"]
        sealed_examples = splits["sealed"]
        dev_path = root / "data" / "dev.jsonl"
        dev_manifest = freeze_v1_split(dev_path, dev_examples, tokenizer, revision=config.model_revision, tokenizer_sha256=str(signatures["tokenizer"]))
        _verify_v1_split(dev_path, str(dev_manifest["sha256"]), len(dev_examples))
        sealed_path = root / "data" / "sealed.jsonl"
        sealed_manifest = freeze_v1_split(sealed_path, sealed_examples, tokenizer, revision=config.model_revision, tokenizer_sha256=str(signatures["tokenizer"]))
        sealed_meta = {"sha256": sealed_manifest["sha256"], "count": len(sealed_examples)}
        dev_split_evidence = {"sha256": dev_manifest["sha256"], "count": len(dev_examples)}
        dev_score_path = root / "scores" / "dev-f0.jsonl"
        active_score_path = dev_score_path
        dev_result = score_v1_examples(model, tokenizer, dev_examples, "single", dev_score_path, signatures, max_output_tokens=256, max_total_tokens=512, deadline=deadline)
        cumulative += float(dev_result["cuda_active_seconds"])
        _append_ledger(root, {"event": "raw_score_file_written", "path": str(dev_score_path.relative_to(root)), "row_count": dev_result["validated_count"], "raw_sha256": dev_result["raw_sha256"], "phase_cuda_active_seconds": dev_result["cuda_active_seconds"], "cumulative_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root)})
        dev_summary = _validate_score_file(dev_score_path, dev_examples, dev_result)
        score_manifest["dev-f0"] = {"score_sha256": dev_result["raw_sha256"], "score_count": dev_result["validated_count"], "summary": dev_summary}
        _budget_check(root, cumulative, config)
        selected_gate = _development_gate(dev_summary)
        review = _export_blinded_review(dev_score_path, dev_examples, root)
        status = "passed" if selected_gate["passes"] else "stopped_scientific"
        result_payload = {
            "run_id": payload["run_id"], "status": status, "pilot_table": pilot_table,
            "pilot_scores": score_manifest, "eligible_families": sorted(row for row in FAMILIES if row not in fallback),
            "selected_difficulty": choices, "fallback_families": fallback,
            "dev": {"split_sha256": dev_manifest["sha256"], "split_manifest_sha256": sha256_file(Path(str(dev_path) + ".manifest.json")), "split_count": len(dev_examples), "scores": dev_summary, "gate": selected_gate},
            "sealed": sealed_meta,
            "resources": {"calibration_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root), "cuda": _cuda_resource_snapshot(), "wall_safety_deadline_seconds": config.resources.maximum_calibration_cuda_seconds, "maximum_calibration_cuda_seconds": config.resources.maximum_calibration_cuda_seconds, "maximum_study_cuda_seconds": config.resources.maximum_study_cuda_seconds, "maximum_artifact_bytes": config.resources.maximum_artifact_bytes},
            "human_review": review, "signatures": signatures,
            "runtime_code_files": runtime_code_files,
            "next_stage": "frozen" if status == "passed" else "report",
        }
        return _finish_calibration(root, result_payload, status)
    except ResourceLimitReached as error:
        status, failure = "stopped_resource", str(error)
    except Exception as error:
        status, failure = "failed_environment", f"{type(error).__name__}: {error}"
        if isinstance(error, ScoreExecutionError):
            cumulative += float(error.cuda_active_seconds)
            score_manifest["partial_failure"] = {
                "path": str(active_score_path.relative_to(root)) if active_score_path is not None else None,
                "row_count": error.row_count, "raw_sha256": error.raw_sha256,
            }
            if error.deadline_exceeded:
                status = "stopped_resource"
        if isinstance(error, MeasuredPhaseFailure):
            cumulative += float(error.cuda_active_seconds)
        if active_score_path is not None and active_score_path.is_file() and "partial_failure" not in score_manifest:
            score_manifest["partial_failure"] = {
                "path": str(active_score_path.relative_to(root)),
                "row_count": _raw_line_count(active_score_path),
                "raw_sha256": sha256_file(active_score_path),
            }
    _append_ledger(root, {"event": "calibration_failed", "phase_cuda_active_seconds": 0.0, "cumulative_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root), "status": status, "failure": failure})
    result_payload = {
        "run_id": payload.get("run_id"), "status": status, "failure": failure,
        "pilot_table": pilot_table, "pilot_scores": score_manifest,
        "dev": None if dev_summary is None and dev_split_evidence is None else {**(dev_split_evidence or {}), "scores": dev_summary},
        "sealed": sealed_meta, "resources": {"calibration_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root), "cuda": _cuda_resource_snapshot()},
        "signatures": signatures, "next_stage": "report",
        "runtime_code_files": _runtime_code_files(project_root),
    }
    return _finish_calibration(root, result_payload, status)


def _load_calibration_runtime(snapshot: Path, revision: str) -> tuple[object, object]:
    return load_v1_model(snapshot, revision)


def _runtime_code_files(project_root: Path) -> dict[str, str]:
    paths = (
        "src/ppmoe/recurrence/study_v1.py",
        "src/ppmoe/recurrence/v1_runtime.py",
        "src/ppmoe/recurrence/model.py",
        "src/ppmoe/recurrence/evaluate.py",
        "src/ppmoe/recurrence/data.py",
        "src/ppmoe/recurrence/human_review.py",
        "src/ppmoe/recurrence/gates.py",
    )
    output = {relative: sha256_file(project_root / Path(relative)) for relative in paths}
    output["aggregate_sha256"] = sha256_payload(output)
    return output


def _cuda_resource_snapshot() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False}
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "device": torch.cuda.get_device_name(index),
        "index": index,
        "total_vram_bytes": int(properties.total_memory),
        "allocated_vram_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_vram_bytes": int(torch.cuda.memory_reserved(index)),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def _verify_calibration_runtime_code(root: Path, project_root: Path) -> None:
    calibration = _read_stage(root / "stages" / "calibrate.json", "calibrate")
    payload = _mapping(calibration.get("payload"), "calibration payload")
    expected = _mapping(payload.get("runtime_code_files"), "calibration runtime code signatures")
    current = _runtime_code_files(project_root)
    if dict(expected) != current:
        raise ValueError("calibration runtime code hashes changed")


def _load_calibration_runtime_measured(snapshot: Path, revision: str) -> tuple[object, object, float]:
    if not torch.cuda.is_available():
        model, tokenizer = _load_calibration_runtime(snapshot, revision)
        return model, tokenizer, 0.0
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        model, tokenizer = _load_calibration_runtime(snapshot, revision)
    except BaseException as error:
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end)) / 1000.0
        raise MeasuredPhaseFailure(error, elapsed) from error
    end.record()
    end.synchronize()
    return model, tokenizer, float(start.elapsed_time(end)) / 1000.0


def _pilot_summary(family: str, candidate: int, summary: Mapping[str, object]) -> dict[str, object]:
    families = _mapping(summary.get("families"), "pilot family summary")
    family_summary = _mapping(families.get(family), "pilot family result")
    accuracy, invalid_rate = float(family_summary["accuracy"]), float(family_summary["invalid_rate"])
    return {
        "family": family, "candidate": candidate,
        "record_count": int(summary["record_count"]), "correct": int(summary["correct"]),
        "invalid_format": int(summary["invalid_format"]), "accuracy": accuracy,
        "invalid_rate": invalid_rate, "eligible": is_informative_family(accuracy, invalid_rate),
    }


def _validate_score_file(path: Path, examples: Sequence[Example], result: Mapping[str, object]) -> dict[str, object]:
    raw_hash = sha256_file(path)
    if raw_hash != result.get("raw_sha256") or result.get("validated_count") != len(examples):
        raise ValueError("score file hash/count does not match runtime validation")
    rows = _read_jsonl(path)
    expected_ids = [row.id for row in examples]
    actual_ids = [row.get("example_id") for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(expected_ids):
        raise ValueError("score file IDs must match the frozen ordered split exactly once")
    summary = summarize_exact_scores(rows)
    if summary != result.get("summary"):
        raise ValueError("score summary does not match raw score records")
    return summary


def _development_gate(summary: Mapping[str, object]) -> dict[str, object]:
    families = _mapping(summary.get("families"), "development family summaries")
    family_set = set(families) == set(FAMILIES)
    info = [family for family, value in families.items() if isinstance(value, Mapping) and is_informative_family(float(value["accuracy"]), float(value["invalid_rate"]))]
    macro = float(summary.get("macro_accuracy", -1))
    invalid = float(summary.get("global_invalid_rate", 1))
    return {
        "family_summaries": dict(families), "macro_accuracy": macro,
        "global_invalid_rate": invalid, "informative_families": sorted(info),
        "macro_pass": 0.05 < macro < 0.95, "invalid_pass": invalid <= 0.10,
        "family_set_pass": family_set, "minimum_informative_families_pass": len(info) >= 2,
        "passes": family_set and 0.05 < macro < 0.95 and invalid <= 0.10 and len(info) >= 2,
    }


def _select_pilot_difficulties(pilot_table: Sequence[Mapping[str, object]]) -> tuple[dict[str, int], list[str]]:
    choices: dict[str, int] = {}
    for family in FAMILIES:
        eligible = [row for row in pilot_table if row.get("family") == family and row.get("eligible") is True]
        eligible.sort(key=lambda row: int(row["candidate"]))
        if eligible:
            choices[family] = int(eligible[0]["candidate"])
    fallback = [family for family in FAMILIES if family not in choices]
    if len(choices) >= 2:
        for family in fallback:
            choices[family] = 1
    return choices, fallback


def _pilot_selection_status(choices: Mapping[str, int]) -> str:
    return "stopped_scientific" if len(choices) < 2 else "continue"


def _verify_v1_split(path: Path, expected_sha256: str, expected_count: int) -> list[dict[str, object]]:
    """Check stage-bound bytes before parsing; sidecars cannot authenticate themselves."""
    _assert_no_reparse(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("split raw hash differs from immutable stage record")
    raw = _read_bytes(path)
    if len(raw.splitlines()) != expected_count:
        raise ValueError("split raw count differs from immutable stage record")
    rows = _read_jsonl(path)
    identifiers = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) for identifier in identifiers) or len(set(identifiers)) != expected_count:
        raise ValueError("split IDs must be unique and complete")
    return rows


def _export_blinded_review(score_path: Path, examples: Sequence[Example], root: Path) -> dict[str, object]:
    from .human_review import export_review

    rows = _read_jsonl(score_path)
    by_id = {example.id: example for example in examples}
    sampled: list[dict[str, object]] = []
    rng = random.Random(9158)
    for family in FAMILIES:
        members = [row for row in rows if row.get("family") == family]
        if len(members) < 10:
            raise ValueError("development set has fewer than ten examples in a family")
        sampled.extend(rng.sample(members, 10))
    records = []
    for row in sampled:
        example = by_id[str(row["example_id"])]
        records.append({
            **row, "arm": "F0", "split": "dev", "prompt": example.prompt,
            "answer": example.answer,
        })
    output = export_review(records, root / "human-review", seed=9158)
    return {
        "sample_count": len(records), "seed": 9158,
        "review_sha256": sha256_file(Path(output["review_path"])),
        "manifest_sha256": sha256_file(Path(output["manifest_path"])),
        "review_path": str(output["review_path"].relative_to(root)),
        "manifest_path": str(output["manifest_path"].relative_to(root)),
        "reviewer": "unassigned; parent-run Luna audit pending",
    }


def _finish_calibration(root: Path, payload: Mapping[str, object], status: str) -> dict[str, object]:
    if status not in _TERMINAL:
        raise ValueError(f"invalid calibration status {status}")
    prepare = _read_stage(root / "stages" / "prepare.json", "prepare")
    record = _stage_record("calibrate", status, dict(payload), {"prepare": str(prepare["stage_sha256"])})
    _write_new(root / "stages" / "calibrate.json", record)
    result = dict(payload)
    result.update({"run_id": payload.get("run_id"), "status": status, "next_stage": payload.get("next_stage", "report"), "run_root": str(root), "stage_sha256": record["stage_sha256"]})
    return result


def _report(root: Path, config: V1Config, config_path: Path) -> dict[str, object]:
    if (root / "stages" / "report.json").exists():
        record = _read_stage(root / "stages" / "report.json", "report")
        prepare = _read_stage(root / "stages" / "prepare.json", "prepare")
        calibration = _read_stage(root / "stages" / "calibrate.json", "calibrate")
        prerequisites = _mapping(record.get("prerequisite_stage_sha256"), "report prerequisites")
        if prerequisites != {"prepare": prepare.get("stage_sha256"), "calibrate": calibration.get("stage_sha256")}:
            raise ValueError("report prerequisite stage hashes changed")
        return {**_mapping(record.get("payload"), "report payload"), "stage_sha256": record["stage_sha256"], "run_root": str(root)}
    prepare = _read_stage(root / "stages" / "prepare.json", "prepare")
    calibration = _read_stage(root / "stages" / "calibrate.json", "calibrate")
    if calibration.get("status") not in _TERMINAL:
        raise ValueError("report requires a terminal calibration record")
    if calibration.get("status") == "passed":
        raise ValueError("calibration passed; final report is reserved until later study stages complete")
    prepared_payload = _mapping(prepare.get("payload"), "prepare payload")
    if prepared_payload.get("config_sha256") != sha256_file(config_path):
        raise ValueError("report config hash differs from prepared config")
    cal_payload = _mapping(calibration.get("payload"), "calibration payload")
    payload = {
        "run_id": prepared_payload.get("run_id"), "status": calibration.get("status"),
        "prepare_stage_sha256": prepare["stage_sha256"], "calibration_stage_sha256": calibration["stage_sha256"],
        "calibration": dict(cal_payload), "next_stage": "complete" if calibration.get("status") in {"stopped_scientific", "stopped_resource", "failed_environment"} else "frozen",
    }
    record = _stage_record("report", str(calibration.get("status")), payload, {"prepare": str(prepare["stage_sha256"]), "calibrate": str(calibration["stage_sha256"])})
    _write_new(root / "stages" / "report.json", record)
    return {**payload, "stage_sha256": record["stage_sha256"], "run_root": str(root)}


def _budget_check(root: Path, cumulative: float, config: V1Config) -> None:
    artifacts = _tree_bytes(root)
    if cumulative >= config.resources.maximum_calibration_cuda_seconds:
        raise ResourceLimitReached("calibration CUDA-active time cap reached")
    if cumulative >= config.resources.maximum_study_cuda_seconds:
        raise ResourceLimitReached("whole-study CUDA-active time cap reached")
    if artifacts >= config.resources.maximum_artifact_bytes:
        raise ResourceLimitReached("new artifact byte cap reached")


def _append_ledger(root: Path, item: Mapping[str, object]) -> None:
    path = root / "resources" / "ledger.jsonl"
    _assert_no_reparse(path)
    encoded = canonical_json(dict(item))
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _ledger_seconds(root: Path) -> float:
    path = root / "resources" / "ledger.jsonl"
    if not path.exists():
        return 0.0
    rows = _read_jsonl(path)
    return max((float(row.get("cumulative_cuda_active_seconds", 0.0)) for row in rows), default=0.0)


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        _assert_no_reparse(path)
        if path.is_file():
            total += path.stat().st_size
    return total


def _stage_record(stage: str, status: str, payload: Mapping[str, object], prerequisites: Mapping[str, str]) -> dict[str, object]:
    base = {"schema_version": 1, "stage": stage, "status": status, "prerequisite_stage_sha256": dict(prerequisites), "payload": dict(payload)}
    return {**base, "stage_sha256": sha256_payload(base)}


def _read_stage(path: Path, expected_stage: str) -> dict[str, object]:
    record = _read_json(path)
    if record.get("stage") != expected_stage or record.get("schema_version") != 1:
        raise ValueError(f"{expected_stage} stage record has an unsupported schema")
    digest = record.get("stage_sha256")
    unsigned = {key: value for key, value in record.items() if key != "stage_sha256"}
    if not isinstance(digest, str) or sha256_payload(unsigned) != digest:
        raise ValueError(f"{expected_stage} stage hash mismatch")
    return record


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    _assert_no_reparse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical_json(dict(payload)))
        stream.flush()
        os.fsync(stream.fileno())


def _parse_rows(raw: bytes, split: str, expected_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"V0 {split} row {number} is invalid JSON") from error
        if not isinstance(value, dict) or value.get("split") != split:
            raise ValueError(f"V0 {split} row {number} has invalid schema or split")
        identifier = value.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"V0 {split} IDs must be non-empty and unique")
        seen.add(identifier)
        rows.append(value)
    if len(rows) != expected_count:
        raise ValueError(f"V0 {split} count is {len(rows)}, expected {expected_count}")
    return rows


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).absolute()


def _read_bytes(path: Path) -> bytes:
    _assert_no_reparse(path)
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"required V1 source input is unreadable: {path}") from error


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(_read_bytes(path))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _parse_json(raw: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
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
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError(f"symlink or reparse point is not allowed: {current}")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(row)
    return rows


def _raw_line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)
