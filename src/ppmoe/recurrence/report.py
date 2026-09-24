"""Strict, evidence-derived reporting for the recurrence experiment.

The report builder is intentionally a reader of persisted evidence.  It does
not run evaluation, load model weights, or turn absent measurements into zeroes.
Every aggregate that is presented in a report is tied to a raw JSONL source
whose bytes are hashed into ``report-sources.json``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .gates import _CUDA_LEDGER_FORMAT, _CUDA_LEDGER_GENESIS, _cuda_cost_entry, _entry_integrity, _ledger_anchor_path, _ledger_anchor_payload, summarize_exact_scores
from .event_identity import task9_failure_event_id
from .provenance import canonical_json, sha256_file, sha256_payload, write_immutable_json


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_STAGE_STATUSES = {"passed", "stopped_scientific", "stopped_budget", "failed_environment"}
_REPORT_OUTPUTS = {"report.json", "report.md", "report-sources.json"}
_STAGE_ORDER = ("source", "data", "smoke", "pilot", "frozen", "discover", "confirm", "report")
# F1/F2 are valid frozen-stage dev arms. Test-sealed summaries remain bound
# separately to the fixed F0/A1/A2/A3 matrix below.
_ARMS = {"F0", "F1", "F2", "A1", "A2", "A3"}
_EVENT_KINDS = {"stage_attempt", "stage_complete", "training_started", "training_progress", "training_checkpoint", "training_completed", "training_failure", "failed_environment"}
_EVENT_OUTCOMES = {"fresh_process_required", "failed_environment", "pilot_cleanup_failure", "passed", "completed", "protocol_deviation", "error", "failure"}


def build_report(run_root: Path) -> dict[str, object]:
    """Validate a run tree and write immutable JSON/Markdown evidence reports."""
    raw_root = Path(run_root).absolute()
    _reject_reparse_path(raw_root, raw_root)
    root = raw_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("report run root must be a materialized directory")
    inputs = _read_inputs(root)
    inventory = {str(item["path"]): item for item in inputs}
    stages = _load_stages(root, inputs, inventory)
    signatures = _signatures(stages)
    source_manifest = _load_source_manifest(root, inputs, stages, inventory)
    events = _load_events(root, inventory)
    exact = _collect_summaries(root, inputs, kind="exact_scores", inventory=inventory, stages=stages)
    retention = _collect_summaries(root, inputs, kind="retention_scores", inventory=inventory, stages=stages)
    _validate_evaluation_authorization(exact, retention, stages)
    _validate_ci_metadata(exact["aggregates"] + retention["aggregates"])
    decision = _decision(stages, inputs)
    if decision["status"] == "passed":
        _require_success_evidence(exact, retention, stages, root, inventory)
    failures = _failure_history(events, exact, retention, stages)
    resources = _resource_evidence(root, inputs, inventory, stages)
    provenance = _provenance(stages, source_manifest, signatures)
    evaluations = {
        "exact_scores": exact,
        "retention": retention,
        "timing": _section_from_payload(stages, ("timing", "training_time", "a3")),
        "training": _section_from_payload(stages, ("training", "training_manifest", "steps")),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "run_root": str(root),
        "provenance": provenance,
        "signatures": signatures,
        "source_manifest": source_manifest or {"status": "not_evaluated"},
        "stages": stages,
        "evaluations": evaluations,
        # These stable top-level names are deliberate: downstream reviewers
        # should not have to infer whether an absent value means zero.
        "examples": _examples_section(exact),
        "families": _families_section(exact),
        "transitions": _transitions_section(exact),
        "accuracy_confidence_intervals": _ci_section(exact),
        "retention_metrics": _retention_section(retention),
        "training_metrics": evaluations["training"],
        "timing_metrics": evaluations["timing"],
        "parameters": _section_from_payload(stages, ("parameters", "parameter_count")),
        "layer_evaluations": _section_from_payload(stages, ("layer_evaluations", "layer_calls")),
        "resources": resources,
        "event_history": events,
        "failure_history": failures,
        "protocol_deviations": _protocol_deviations(stages, events),
        "decision": decision,
        "conclusion": _conclusion(decision),
        "limitations": _limitations(),
        "next_action": _next_action(decision),
    }
    manifest = {"schema_version": 1, "files": inputs}
    manifest_path = root / "report-sources.json"
    write_immutable_json(manifest_path, manifest)
    report["source_manifest_sha256"] = sha256_file(manifest_path)
    # This digest covers the report payload without its self-describing field.
    report["report_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    write_immutable_json(root / "report.json", report)
    markdown = _render_markdown(report)
    _write_immutable_text(root / "report.md", markdown)
    return report


def _read_inputs(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted((entry for entry in root.rglob("*") if entry.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"report evidence may not be a symbolic link: {path}")
        if relative in _REPORT_OUTPUTS or relative == "stages/report.json":
            continue
        _reject_reparse_path(root, path)
        data, before, after = _descriptor_read(path, root)
        if before.st_size != after.st_size or before.st_ino != after.st_ino or len(data) != after.st_size:
            raise ValueError(f"input changed during inventory: {path}")
        records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return records


def _load_json(path: Path, *, root: Path, inventory: Mapping[str, Mapping[str, object]] | None = None) -> dict[str, object]:
    _reject_reparse_path(root, path)
    try:
        data = _read_verified(path, root, inventory)
        value = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    _inside(root, path)
    _finite_tree(value, path)
    return value


def _load_stages(root: Path, inputs: list[dict[str, object]], inventory: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    stages_dir = root / "stages"
    if not stages_dir.is_dir():
        raise ValueError("report requires a stages directory")
    stages: dict[str, dict[str, object]] = {}
    for path in sorted(stages_dir.glob("*.json")):
        name = path.stem
        if name == "report":
            continue
        if name not in _STAGE_ORDER:
            raise ValueError(f"unknown stage record: {path}")
        record = _load_json(path, root=root, inventory=inventory)
        required = {"stage", "status", "signatures", "payload", "attempts"}
        if set(record) != required:
            raise ValueError(f"invalid stage record: {path}")
        if record["stage"] != name or record["status"] not in _STAGE_STATUSES:
            raise ValueError(f"invalid stage status: {path}")
        if not isinstance(record["payload"], dict) or not isinstance(record["signatures"], dict) or isinstance(record["attempts"], bool) or not isinstance(record["attempts"], int) or record["attempts"] < 1:
            raise ValueError(f"invalid stage payload/signatures: {path}")
        _validate_signatures(record["signatures"], allow_empty=name == "source")
        if name == "source":
            source_signatures = record["signatures"]
            if (not _SHA256.fullmatch(str(source_signatures["source"]))
                    or source_signatures["data"] not in ("",)
                    or not _SHA256.fullmatch(str(source_signatures["config"]))
                    or not _SHA256.fullmatch(str(source_signatures["code"]))):
                raise ValueError("source stage must finalize source only and bind config/code")
        if name in stages:
            raise ValueError(f"duplicate stage record: {name}")
        stages[name] = record
    if not stages:
        raise ValueError("report requires at least one stage record")
    ordered = {name: stages[name] for name in _STAGE_ORDER if name in stages}
    _validate_stage_dag(ordered)
    return ordered | {
        name: value for name, value in stages.items() if name not in _STAGE_ORDER
    }


def _validate_stage_dag(stages: Mapping[str, Mapping[str, object]]) -> None:
    names = list(stages)
    if names != sorted(names, key=_STAGE_ORDER.index):
        raise ValueError("stage records are out of registered order")
    for name, predecessor in ((name, _STAGE_ORDER[_STAGE_ORDER.index(name) - 1]) for name in names if name not in {"source", "report"}):
        if predecessor not in stages:
            raise ValueError(f"stage {name} is missing predecessor {predecessor}")
        previous = stages[predecessor]
        if previous.get("status") != "passed":
            raise ValueError(f"stage {name} follows non-passing predecessor {predecessor}")
        current_signatures = stages[name].get("signatures")
        previous_signatures = previous.get("signatures")
        if not isinstance(current_signatures, Mapping) or not isinstance(previous_signatures, Mapping):
            raise ValueError(f"stage {name} signatures are malformed")
        for key in ("source", "data", "config", "code"):
            if current_signatures[key] == previous_signatures[key]:
                continue
            if name == "data" and key == "data" and previous_signatures[key] == "" and _SHA256.fullmatch(str(current_signatures[key])):
                continue
            raise ValueError(f"stage {name} signature mismatch with {predecessor}")


def _reject_reparse_path(root: Path, path: Path) -> None:
    """Reject symlink/junction/reparse components before resolving a path."""
    root = Path(root).absolute()
    path = Path(path).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"report evidence escapes trusted root: {path}") from error
    current = root
    components = [root]
    for component in relative.parts:
        current = current / component
        components.append(current)
    for current in components:
        try:
            # lstat is deliberately used here: stat follows junctions and
            # reparse points, which would make the trusted-root boundary
            # meaningless on Windows.
            info = current.lstat()
            attributes = int(getattr(info, "st_file_attributes", 0))
        except OSError as error:
            raise ValueError(f"report evidence cannot be inspected: {current}") from error
        if current.is_symlink() or getattr(info, "st_reparse_tag", 0):
            raise ValueError(f"report evidence may not contain a symbolic link/reparse point: {current}")
        if attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT on Windows
            raise ValueError(f"report evidence may not contain a reparse point: {current}")


def _read_verified(path: Path, root: Path, inventory: Mapping[str, Mapping[str, object]] | None) -> bytes:
    _reject_reparse_path(root, path)
    relative = path.relative_to(root).as_posix()
    expected = inventory.get(relative) if inventory is not None else None
    if expected is None:
        raise ValueError(f"input was not present in the trusted inventory: {relative}")
    data, before, after = _descriptor_read(path, root)
    digest = hashlib.sha256(data).hexdigest()
    if before.st_size != after.st_size or before.st_ino != after.st_ino or len(data) != int(expected["bytes"]) or digest != expected["sha256"]:
        raise ValueError(f"input changed after inventory: {relative}")
    return data


def _descriptor_read(path: Path, trusted_root: Path | None = None) -> tuple[bytes, os.stat_result, os.stat_result]:
    """Read one immutable descriptor/handle, never rereading through a path."""
    flags = os.O_RDONLY
    parent_fds: list[int] = []
    parent_handles: list[object] = []
    parent_before: list[os.stat_result] = []
    parent_paths: list[Path] = []
    if trusted_root is not None:
        root = Path(trusted_root).absolute()
        absolute = Path(path).absolute()
        _reject_reparse_path(root, absolute)
        try:
            relative_parent = absolute.parent.relative_to(root)
        except ValueError as error:
            raise ValueError(f"evidence escapes trusted root: {path}") from error
        parent_paths = [root]
        current = root
        for component in relative_parent.parts:
            current = current / component
            parent_paths.append(current)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            if parent_paths:
                parent_fds.append(os.open(parent_paths[0], flags | getattr(os, "O_DIRECTORY", 0)))
                parent_before.append(os.fstat(parent_fds[-1]))
                for component in relative_parent.parts:
                    descriptor_parent = parent_fds[-1]
                    parent_fds.append(os.open(component, flags | getattr(os, "O_DIRECTORY", 0), dir_fd=descriptor_parent))
                    parent_before.append(os.fstat(parent_fds[-1]))
                descriptor = os.open(absolute.name, flags, dir_fd=parent_fds[-1])
            else:
                descriptor = os.open(path, flags)
        except OSError as error:
            raise ValueError(f"evidence cannot be opened without following links: {path}") from error
    else:
        if parent_paths:
            parent_before = [parent.lstat() for parent in parent_paths]
        import ctypes
        import msvcrt
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        class _UnicodeString(ctypes.Structure):
            _fields_ = [("Length", ctypes.c_ushort), ("MaximumLength", ctypes.c_ushort), ("Buffer", ctypes.c_wchar_p)]
        class _ObjectAttributes(ctypes.Structure):
            _fields_ = [("Length", ctypes.c_ulong), ("RootDirectory", ctypes.c_void_p), ("ObjectName", ctypes.c_void_p), ("Attributes", ctypes.c_ulong), ("SecurityDescriptor", ctypes.c_void_p), ("SecurityQualityOfService", ctypes.c_void_p)]
        class _IoStatusBlock(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_long), ("Information", ctypes.c_size_t)]
        ntdll.RtlDosPathNameToNtPathName_U.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(_UnicodeString), ctypes.POINTER(ctypes.c_wchar_p), ctypes.c_void_p]
        ntdll.RtlDosPathNameToNtPathName_U.restype = ctypes.c_ubyte
        ntdll.RtlFreeUnicodeString.argtypes = [ctypes.POINTER(_UnicodeString)]
        ntdll.RtlFreeUnicodeString.restype = None
        ntdll.RtlInitUnicodeString.argtypes = [ctypes.POINTER(_UnicodeString), ctypes.c_wchar_p]
        ntdll.RtlInitUnicodeString.restype = None
        ntdll.NtOpenFile.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong, ctypes.POINTER(_ObjectAttributes), ctypes.POINTER(_IoStatusBlock), ctypes.c_ulong, ctypes.c_ulong]
        ntdll.NtOpenFile.restype = ctypes.c_long
        def _open_nt(name: str, root_handle: object | None, *, absolute: bool) -> object:
            allocated = _UnicodeString()
            file_part = ctypes.c_wchar_p()
            if absolute:
                if not ntdll.RtlDosPathNameToNtPathName_U(name, ctypes.byref(allocated), ctypes.byref(file_part), None):
                    raise ValueError(f"trusted evidence path cannot be converted to an NT path: {name}")
                name_ref = ctypes.byref(allocated)
            else:
                keep_name = ctypes.c_wchar_p(name)
                name_ref = _UnicodeString()
                ntdll.RtlInitUnicodeString(ctypes.byref(name_ref), keep_name)
            object_name = ctypes.cast(name_ref, ctypes.c_void_p) if absolute else ctypes.cast(ctypes.byref(name_ref), ctypes.c_void_p)
            attributes = _ObjectAttributes(ctypes.sizeof(_ObjectAttributes), root_handle, object_name, 0x40, None, None)
            io_status = _IoStatusBlock()
            opened = ctypes.c_void_p()
            desired_access = 0x100081 if root_handle is not None and not absolute else 0x100001
            status = ntdll.NtOpenFile(ctypes.byref(opened), desired_access, ctypes.byref(attributes), ctypes.byref(io_status), 0x7, 0x20 | 0x00200000)
            if absolute:
                ntdll.RtlFreeUnicodeString(ctypes.byref(allocated))
            if status < 0 or opened.value in (None, ctypes.c_void_p(-1).value):
                raise ValueError(f"evidence component cannot be opened relative to trusted handle: {name}")
            return opened.value
        try:
            for index, parent in enumerate(parent_paths):
                if index == 0:
                    parent_handles.append(_open_nt(str(parent), None, absolute=True))
                else:
                    parent_handles.append(_open_nt(parent.name, parent_handles[-1], absolute=False))
        except (OSError, ValueError) as error:
            for parent_handle in reversed(parent_handles):
                kernel32.CloseHandle(parent_handle)
            raise ValueError(str(error)) from error
        if not parent_handles:
            raise ValueError(f"trusted parent handle chain is unavailable: {path}")
        for parent, original in zip(parent_paths, parent_before, strict=True):
            current_parent = parent.lstat()
            if (current_parent.st_ino, getattr(current_parent, "st_file_attributes", 0)) != (original.st_ino, getattr(original, "st_file_attributes", 0)):
                raise ValueError(f"trusted parent directory changed before evidence open: {path}")
        try:
            opened = _open_nt(absolute.name, parent_handles[-1], absolute=False)
        except ValueError:
            for parent_handle in reversed(parent_handles):
                kernel32.CloseHandle(parent_handle)
            raise ValueError(f"evidence cannot be opened relative to trusted parent handle: {path}")
        try:
            descriptor = msvcrt.open_osfhandle(opened, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        except Exception:
            kernel32.CloseHandle(opened)
            for parent_handle in reversed(parent_handles):
                kernel32.CloseHandle(parent_handle)
            raise
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if parent_fds:
            for descriptor_parent, original in zip(parent_fds, parent_before, strict=True):
                current_parent = os.fstat(descriptor_parent)
                if (current_parent.st_dev, current_parent.st_ino) != (original.st_dev, original.st_ino):
                    raise ValueError(f"trusted parent directory changed during evidence read: {path}")
        elif parent_paths:
            for parent, original in zip(parent_paths, parent_before, strict=True):
                current_parent = parent.lstat()
                if (current_parent.st_ino, getattr(current_parent, "st_file_attributes", 0)) != (original.st_ino, getattr(original, "st_file_attributes", 0)):
                    raise ValueError(f"trusted parent directory changed during evidence read: {path}")
        return b"".join(chunks), before, after
    finally:
        os.close(descriptor)
        for descriptor_parent in reversed(parent_fds):
            os.close(descriptor_parent)
        if parent_handles:
            for parent_handle in reversed(parent_handles):
                kernel32.CloseHandle(parent_handle)


def _signatures(stages: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    candidates = [record["signatures"] for record in stages.values() if isinstance(record.get("signatures"), Mapping)]
    if not candidates:
        return {"source": "", "data": "", "config": "", "code": ""}
    first = dict(candidates[-1])
    _validate_signatures(first, allow_empty=True)
    for candidate in candidates:
        current = dict(candidate)
        _validate_signatures(current, allow_empty=True)
        for key in first:
            if first[key] and current[key] and first[key] != current[key]:
                raise ValueError(f"stage signature mismatch for {key}")
            if not first[key] and current[key]:
                first[key] = current[key]
    return first


def _validate_signatures(value: object, *, allow_empty: bool) -> None:
    if not isinstance(value, Mapping) or set(value) != {"source", "data", "config", "code"}:
        raise ValueError("report signatures must contain source/data/config/code")
    for key, digest in value.items():
        if not isinstance(digest, str) or (not allow_empty and not _SHA256.fullmatch(digest)):
            raise ValueError(f"invalid {key} signature")
        if digest and not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid {key} signature")


def _load_source_manifest(root: Path, inputs: list[dict[str, object]], stages: Mapping[str, Mapping[str, object]], inventory: Mapping[str, Mapping[str, object]]) -> dict[str, object] | None:
    candidates = [root / name for name in ("source-manifest.json", "source_manifest.json", "source.json")]
    payloads = []
    for path in candidates:
        if path.is_file():
            payloads.append(_load_json(path, root=root, inventory=inventory))
    if not payloads:
        for record in stages.values():
            payload = record.get("payload")
            if isinstance(payload, Mapping):
                for key in ("source_manifest", "source_manifest_path"):
                    value = payload.get(key)
                    if isinstance(value, str):
                        path = _resolve_evidence_path(root, value)
                        if path.is_file():
                            payloads.append(_load_json(path, root=root, inventory=inventory))
    if not payloads:
        return None
    manifest = payloads[0]
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision.lower()):
        raise ValueError("source manifest has no immutable revision")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("source manifest has no file inventory")
    manifest_paths: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("relative_path"), str) or not _SHA256.fullmatch(str(item.get("sha256"))):
            raise ValueError("source manifest contains an invalid file hash")
        relative = str(item["relative_path"])
        if relative in manifest_paths or isinstance(item.get("bytes"), bool) or not isinstance(item.get("bytes"), int) or item["bytes"] < 0:
            raise ValueError("source manifest contains duplicate or invalid file sizes")
        manifest_paths.add(relative)
        path = _resolve_evidence_path(root, relative)
        if not path.is_file():
            raise ValueError(f"source manifest path is missing: {relative}")
        expected = inventory.get(path.relative_to(root).as_posix())
        if expected is None:
            raise ValueError(f"source manifest path is outside the inventory: {relative}")
        data = _read_verified(path, root, inventory)
        if item["bytes"] != len(data) or str(item.get("sha256")) != hashlib.sha256(data).hexdigest():
            raise ValueError(f"source manifest hash/size mismatch: {relative}")
    return manifest


def _collect_summaries(root: Path, inputs: list[dict[str, object]], *, kind: str, inventory: Mapping[str, Mapping[str, object]], stages: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    aggregates: list[dict[str, object]] = []
    for item in inputs:
        relative = str(item["path"])
        if not relative.endswith(".json"):
            continue
        path = root / relative
        try:
            summary = _load_json(path, root=root, inventory=inventory)
        except ValueError:
            raise
        if summary.get("kind") != kind:
            continue
        complete = summary.get("complete") is True
        if summary.get("complete") is not True:
            continue
        raw_ref = summary.get("records_path") if kind == "exact_scores" else summary.get("records_path")
        declared_hash = next((summary.get(key) for key in ("records_sha256", "raw_sha256", "records_hash", "raw_hash") if key in summary), None)
        if complete and not isinstance(raw_ref, str):
            raise ValueError(f"aggregate {relative} has no raw score/chunk path")
        if complete and (not isinstance(declared_hash, str) or not _SHA256.fullmatch(declared_hash)):
            raise ValueError(f"aggregate {relative} has no declared raw score/chunk hash")
        _validate_summary_schema(summary, kind, stages, relative)
        aggregate = dict(summary)
        aggregate["summary_path"] = relative
        if isinstance(raw_ref, str):
            raw_path = _resolve_evidence_path(root, raw_ref)
            if not raw_path.is_file():
                raise ValueError(f"aggregate {relative} raw score/chunk file is missing")
            raw_rows = _read_jsonl(raw_path, root, inventory)
            raw_hash = hashlib.sha256(_read_verified(raw_path, root, inventory)).hexdigest()
            aggregate["raw_path"] = raw_path.relative_to(root).as_posix()
            aggregate["raw_sha256"] = raw_hash
            for key in ("records_sha256", "raw_sha256", "records_hash", "raw_hash"):
                if key in summary and summary[key] != raw_hash:
                    raise ValueError(f"aggregate {relative} raw score/chunk hash mismatch")
            _validate_raw(kind, raw_rows, summary, relative)
            if kind == "exact_scores" and complete:
                derived = summarize_exact_scores(raw_rows)
                aggregate["family_summary"] = derived["families"]
                aggregate["macro_accuracy"] = derived["macro_accuracy"]
                aggregate["global_invalid_rate"] = derived["global_invalid_rate"]
                aggregate["transitions"] = derived.get("transitions", {})
                aggregate["record_ids"] = [str(row["example_id"]) for row in raw_rows]
                aggregate["record_families"] = {str(row["example_id"]): str(row["family"]) for row in raw_rows}
                splits = {row.get("split") for row in raw_rows}
                if len(splits) != 1 or next(iter(splits)) not in {"train", "dev", "test"}:
                    raise ValueError(f"aggregate {relative} has an invalid evaluation split")
                aggregate["split"] = next(iter(splits))
                if summary["split"] != aggregate["split"]:
                    raise ValueError(f"aggregate {relative} split does not match raw rows")
                if "ordered_ids_sha256" in summary and summary["ordered_ids_sha256"] != sha256_payload({"ids": aggregate["record_ids"]}):
                    raise ValueError(f"aggregate {relative} ordered ID hash mismatch")
            if kind == "retention_scores" and complete:
                splits = {row.get("split") for row in raw_rows}
                if len(splits) != 1 or next(iter(splits)) not in {"pilot", "dev", "test", "retention"}:
                    raise ValueError(f"aggregate {relative} has an invalid retention split")
                aggregate["split"] = next(iter(splits))
                if summary["split"] != aggregate["split"]:
                    raise ValueError(f"aggregate {relative} split does not match raw rows")
                raw_nll = sum(float(row["nll_sum"]) for row in raw_rows)
                raw_tokens = sum(int(row["scored_tokens"]) for row in raw_rows)
                declared_nll = _finite_number(summary.get("nll_sum"), "nll_sum")
                declared_tokens = _positive_int(summary.get("scored_tokens"), "scored_tokens")
                if not math.isclose(raw_nll, declared_nll, rel_tol=1e-10, abs_tol=1e-12) or raw_tokens != declared_tokens:
                    raise ValueError(f"aggregate {relative} retention NLL/token totals do not match raw chunks")
                aggregate["nll_sum"] = declared_nll
                aggregate["scored_tokens"] = declared_tokens
                aggregate["chunk_indices"] = [int(row["chunk_index"]) for row in raw_rows]
                aggregate["chunk_tokens"] = {int(row["chunk_index"]): int(row["scored_tokens"]) for row in raw_rows}
                if "ordered_indices_sha256" in summary and summary["ordered_indices_sha256"] != sha256_payload({"indices": aggregate["chunk_indices"]}):
                    raise ValueError(f"aggregate {relative} ordered chunk hash mismatch")
        aggregates.append(aggregate)
    return {
        "status": "evaluated" if aggregates else "not_evaluated",
        "aggregates": aggregates,
        "examples_scored": sum(int(item.get("record_count", 0)) for item in aggregates if kind == "exact_scores"),
        "failures": sum(int(item.get("failures", 0)) for item in aggregates if isinstance(item.get("failures", 0), int)),
    }


def _validate_summary_schema(summary: Mapping[str, object], kind: str, stages: Mapping[str, Mapping[str, object]], relative: str) -> None:
    required = {"schema_version", "kind", "record_count", "expected_count", "complete", "records_path", "records_sha256", "arm", "seed", "failures", "signatures", "families", "split"}
    if not required.issubset(summary) or summary.get("schema_version") != 1 or summary.get("kind") != kind or summary.get("complete") is not True:
        raise ValueError(f"summary {relative} has an invalid schema")
    common = required | {"mode", "correct", "incorrect", "ordered_ids_sha256", "failure_reason"}
    allowed = common | {"split"} | ({"nll_sum", "mean_nll", "scored_tokens", "perplexity", "ordered_indices_sha256"} if kind == "retention_scores" else set())
    if set(summary) - allowed:
        raise ValueError(f"summary {relative} contains unknown fields")
    arm = summary.get("arm")
    seed = summary.get("seed")
    if not isinstance(arm, str) or arm.upper() not in _ARMS or isinstance(seed, bool) or not isinstance(seed, int) or seed not in {47, 48}:
        raise ValueError(f"summary {relative} has invalid arm/seed")
    families = summary.get("families")
    if kind == "exact_scores" and (not isinstance(families, list) or set(families) != {"expressions", "lists", "graphs"}):
        raise ValueError(f"summary {relative} has invalid registered families")
    if kind == "retention_scores" and (not isinstance(families, list) or set(families) != {"retention"}):
        raise ValueError(f"summary {relative} has invalid retention families")
    signatures = summary.get("signatures")
    _validate_signatures(signatures, allow_empty=False)
    split = summary.get("split")
    if split not in {"train", "dev", "test", "pilot", "retention"}:
        raise ValueError(f"summary {relative} has an invalid split")
    upper_arm = arm.upper()
    producer = (
        "confirm" if split == "test"
        else "frozen" if upper_arm in {"F0", "F1", "F2"}
        else "discover" if upper_arm in {"A1", "A2", "A3"} and seed == 47
        else "confirm"
    )
    stage = stages.get(producer)
    if (
        not stage
        or stage.get("status") not in {"passed", "stopped_scientific"}
        or stage.get("signatures") != signatures
    ):
        raise ValueError(f"summary {relative} signature is not bound to authorized {producer} stage")


def _validate_raw(kind: str, rows: list[dict[str, object]], summary: Mapping[str, object], relative: str) -> None:
    expected = summary.get("expected_count")
    record_count = summary.get("record_count")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (expected, record_count)) or record_count != expected or len(rows) != expected:
        raise ValueError(f"aggregate {relative} count does not match raw evidence")
    failures = summary.get("failures")
    if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
        raise ValueError(f"aggregate {relative} has no complete failure count")
    if kind == "exact_scores":
        ids: set[str] = set()
        for row in rows:
            identifier = row.get("example_id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError(f"aggregate {relative} has duplicate/missing example IDs")
            if not isinstance(row.get("correct"), bool) or row.get("family") not in {"expressions", "lists", "graphs"}:
                raise ValueError(f"aggregate {relative} has invalid raw score")
            ids.add(identifier)
        if "correct" in summary and summary["correct"] != sum(row["correct"] is True for row in rows):
            raise ValueError(f"aggregate {relative} correct count mismatch")
        raw_failures = sum(row.get("reason_code") in {"execution_error", "decode_failure"} for row in rows)
        if failures != raw_failures:
            raise ValueError(f"aggregate {relative} failure count mismatch")
    else:
        indices: list[int] = []
        for row in rows:
            if isinstance(row.get("chunk_index"), bool) or not isinstance(row.get("chunk_index"), int):
                raise ValueError(f"aggregate {relative} has invalid chunk index")
            _positive_int(row.get("scored_tokens"), "scored_tokens")
            if _finite_number(row.get("nll_sum"), "nll_sum") < 0.0:
                raise ValueError(f"aggregate {relative} NLL must be non-negative")
            indices.append(int(row["chunk_index"]))
        if len(indices) != len(set(indices)):
            raise ValueError(f"aggregate {relative} has duplicate chunk indices")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"aggregate {relative} chunk indices are not a complete ordered universe")
        raw_failures = sum(row.get("failure") is not None for row in rows)
        if failures != raw_failures:
            raise ValueError(f"aggregate {relative} failure count mismatch")


def _validate_ci_metadata(aggregates: list[Mapping[str, object]]) -> None:
    for aggregate in aggregates:
        has_ci = any(key in aggregate for key in ("ci_low", "ci_high", "ppl_ratio_ci_low", "ppl_ratio_ci_high"))
        if not has_ci:
            continue
        if aggregate.get("replicates") != 10000 or aggregate.get("seed") not in {9047, 9157} or not isinstance(aggregate.get("bootstrap"), list) or len(aggregate["bootstrap"]) != 10000:
            raise ValueError("confidence interval is missing bootstrap metadata")
        for key in ("ci_low", "ci_high", "ppl_ratio_ci_low", "ppl_ratio_ci_high"):
            if key in aggregate:
                _finite_number(aggregate[key], key)
        if isinstance(aggregate.get("bootstrap"), list):
            values = np.asarray([_finite_number(value, "bootstrap value") for value in aggregate["bootstrap"]], dtype=np.float64)
            low = float(np.percentile(values, 2.5))
            high = float(np.percentile(values, 97.5))
            for key, expected in (("ci_low", low), ("ppl_ratio_ci_low", low), ("ci_high", high), ("ppl_ratio_ci_high", high)):
                if key in aggregate and not math.isclose(float(aggregate[key]), expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("confidence interval does not reconcile with bootstrap vector")


def _validate_evaluation_authorization(
    exact: Mapping[str, object], retention: Mapping[str, object], stages: Mapping[str, Mapping[str, object]]
) -> None:
    """Do not accept score files that could only have been produced by an
    unrecorded downstream stage."""
    completed = {name for name, record in stages.items() if record.get("status") == "passed"}
    for section in (exact, retention):
        for aggregate in section.get("aggregates", []):
            if not isinstance(aggregate, Mapping) or aggregate.get("complete") is not True:
                continue
            arm = str(aggregate.get("arm", "")).upper()
            seed = aggregate.get("seed")
            if arm in {"A1", "A2", "A3"} and not ({"discover", "confirm"} & completed):
                raise ValueError("evaluation is not authorized by a discovery/confirmation stage")
            if seed == 47 and arm in {"A1", "A2", "A3"} and "discover" not in completed:
                raise ValueError("discovery-seed evaluation is not authorized")
            if seed == 48 and "confirm" not in completed:
                raise ValueError("confirmation-seed evaluation is not authorized")


def _decision(stages: Mapping[str, Mapping[str, object]], inputs: list[dict[str, object]]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    # Gate decisions are produced only by G3/G4/G5 stages.  Source/data and
    # infrastructure payloads are provenance, never scientific authority.
    for name in ("confirm", "discover", "frozen"):
        record = stages.get(name)
        if not record:
            continue
        if record.get("status") not in _STAGE_STATUSES:
            raise ValueError(f"invalid gate stage status: {name}")
        if record.get("status") != "passed":
            payload = record.get("payload")
            reason = payload.get("reason") if isinstance(payload, Mapping) else None
            if reason is None and isinstance(payload, Mapping) and isinstance(payload.get("decision"), Mapping):
                reason = payload["decision"].get("reason")
            candidates.append({"status": record["status"], "reason": reason})
            break
        payload = record.get("payload")
        if isinstance(payload, Mapping):
            for key in ("decision", "gate_decision", "outcome"):
                value = payload.get(key)
                if isinstance(value, Mapping) and value.get("status") in (_STAGE_STATUSES | {"retention_uncertain", "timing_inconclusive"}):
                    candidates.append(dict(value))
        if candidates:
            break
    if not candidates:
        for name in _STAGE_ORDER[::-1]:
            record = stages.get(name)
            if record and record.get("status") in _STAGE_STATUSES - {"passed"}:
                candidates.append({"status": record["status"], "reason": record.get("payload", {}).get("reason") if isinstance(record.get("payload"), Mapping) else None})
                break
    decision = candidates[0] if candidates else {"status": "failed_environment", "reason": "no terminal decision evidence"}
    if decision.get("status") in {"retention_uncertain", "timing_inconclusive"}:
        decision["reason"] = decision["status"]
        decision["status"] = "stopped_scientific"
    if decision["status"] not in _STAGE_STATUSES:
        raise ValueError("invalid report decision status")
    return decision


def _require_success_evidence(exact: Mapping[str, object], retention: Mapping[str, object], stages: Mapping[str, Mapping[str, object]], root: Path, inventory: Mapping[str, Mapping[str, object]]) -> None:
    aggregates = [item for item in exact.get("aggregates", []) if isinstance(item, Mapping) and item.get("complete") is True and item.get("split") == "test"]
    by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for item in aggregates:
        arm = str(item.get("arm", "")).upper()
        seed = item.get("seed")
        if arm not in _ARMS or isinstance(seed, bool) or not isinstance(seed, int) or (arm, seed) in by_key:
            raise ValueError("success report has duplicate or unknown arm/seed evidence")
        by_key[(arm, seed)] = item
    required = {(arm, seed) for arm in ("A1", "A2", "A3") for seed in (47, 48)} | {("F0", 47)}
    if set(by_key) != required:
        raise ValueError("success report is missing a registered arm/seed/control")
    family_sets = []
    id_sets = []
    for key, item in sorted(by_key.items()):
        families = item.get("family_summary")
        ids = item.get("record_ids")
        row_families = item.get("record_families")
        if not isinstance(families, Mapping) or set(families) != {"expressions", "lists", "graphs"}:
            raise ValueError("success report does not contain all three registered families")
        if any(not isinstance(value, Mapping) or not _is_finite_number(value.get("accuracy")) or not _is_finite_number(value.get("invalid_rate")) for value in families.values()):
            raise ValueError("success report contains invalid family evidence")
        if not isinstance(ids, list) or not ids or not isinstance(row_families, Mapping) or set(row_families) != set(ids):
            raise ValueError("success report has incomplete paired ID/family evidence")
        family_sets.append(set(families))
        id_sets.append(tuple(ids))
    if any(value != family_sets[0] for value in family_sets) or any(value != id_sets[0] for value in id_sets):
        raise ValueError("success report arm/seed ID or family universes do not match")
    retention_rows = [
        item for item in retention.get("aggregates", [])
        if isinstance(item, Mapping)
        and item.get("complete") is True
        and (
            (item.get("split") == "dev" and item.get("seed") == 47)
            or (item.get("split") == "test" and item.get("seed") == 48)
            or (str(item.get("arm", "")).upper() == "F0" and item.get("split") in {"dev", "retention"})
        )
    ]
    retention_map = {(str(item.get("arm", "")).upper(), item.get("seed")): item for item in retention_rows}
    if len(retention_map) != len(retention_rows):
        raise ValueError("success report has duplicate retention arm/seed evidence")
    required_retention = {
        (arm, seed) for arm in ("A1", "A2", "A3") for seed in (47, 48)
    } | {("F0", 47)}
    if set(retention_map) != required_retention:
        raise ValueError("success report is missing the registered retention matrix")
    for seed in (47, 48):
        candidate = retention_map.get(("A2", seed))
        baseline = retention_map.get(("F0", 47))
        if candidate is None or baseline is None or candidate.get("chunk_indices") != baseline.get("chunk_indices") or candidate.get("chunk_tokens") != baseline.get("chunk_tokens"):
            raise ValueError("success report is missing paired retention F0/candidate universes")
        for arm in ("A1", "A2", "A3"):
            item = retention_map[(arm, seed)]
            if item.get("chunk_indices") != baseline.get("chunk_indices") or item.get("chunk_tokens") != baseline.get("chunk_tokens"):
                raise ValueError("success report retention universes do not match")
        accuracy_universe = _family_universe(by_key)
        retention_universe = _retention_universe(retention_map)
        if not _has_registered_bootstrap(root, inventory, stages, 9047, "accuracy", accuracy_universe, by_key) or not _has_registered_bootstrap(root, inventory, stages, 9157, "retention", retention_universe, retention_map):
            raise ValueError("success report is missing registered bootstrap/gate evidence")
    if not _has_timing_gate(stages):
        raise ValueError("success report is missing A3 timing-control evidence")
    if any(int(item.get("failures", 0) or 0) != 0 for item in aggregates + retention_rows):
        raise ValueError("success report contains incomplete evaluation failures")


def _family_universe(aggregates: Mapping[tuple[str, int], Mapping[str, object]]) -> dict[str, list[str]]:
    reference = next(iter(aggregates.values()))
    mapping = reference.get("record_families")
    if not isinstance(mapping, Mapping):
        return {}
    result = {family: sorted(str(identifier) for identifier, value in mapping.items() if value == family) for family in ("expressions", "lists", "graphs")}
    return result if all(result.values()) else {}


def _retention_universe(aggregates: Mapping[tuple[str, object], Mapping[str, object]]) -> dict[str, list[str]]:
    reference = next((value for key, value in aggregates.items() if key[0] == "F0"), None)
    if not isinstance(reference, Mapping) or not isinstance(reference.get("chunk_indices"), list):
        return {}
    return {"retention": [str(index) for index in reference["chunk_indices"]]}


def _has_registered_bootstrap(root: Path, inventory: Mapping[str, Mapping[str, object]], stages: Mapping[str, Mapping[str, object]], seed: int, label: str, expected_universe: Mapping[str, list[str]], aggregates: Mapping[tuple[object, object], Mapping[str, object]]) -> bool:
    producers = {"discover", "confirm"} if seed == 9047 else {"confirm"} if seed == 9157 else set()
    for stage_name, record in stages.items():
        if producers and stage_name not in producers:
            continue
        if record.get("status") != "passed":
            continue
        payload = record.get("payload")
        for value in _walk_mappings(payload):
            if value.get("seed") == seed and value.get("replicates") == 10000 and (label in " ".join(str(key).lower() for key in value) or isinstance(value.get("bootstrap"), list) or isinstance(value.get("bootstrap_sha256"), str)):
                if isinstance(value.get("bootstrap"), list) and len(value["bootstrap"]) != 10000:
                    continue
                if not _SHA256.fullmatch(str(value.get("bootstrap_sha256", ""))):
                    continue
                paired_keys = [key for key in value if "paired" in str(key).lower() or "sample_indices" in str(key).lower()]
                if not isinstance(value.get("bootstrap"), list) and not paired_keys:
                    continue
                if paired_keys and not any(isinstance(value[key], list) or _SHA256.fullmatch(str(value[key])) for key in paired_keys):
                    continue
                paired_path = value.get("paired_sample_indices_path") or value.get("sample_indices_path")
                if not isinstance(paired_path, str):
                    continue
                paired = _read_bootstrap_matrix(root, inventory, paired_path, value.get("paired_sample_indices_sha256") or value.get("sample_indices_sha256"), expected_universe, registered_seed=seed)
                if paired is None:
                    continue
                comparisons = value.get("comparisons")
                expected_comparisons = {"A2-F0", "A2-A1", "A2-A3"} if label == "accuracy" else {"A2-F0"}
                if not isinstance(comparisons, Mapping):
                    continue
                if label == "retention":
                    normalized: dict[str, Mapping[str, object]] = {}
                    for key, item in comparisons.items():
                        if not isinstance(item, Mapping):
                            continue
                        item_seed = item.get("seed")
                        if isinstance(item_seed, bool) or not isinstance(item_seed, int) or item_seed not in {47, 48}:
                            continue
                        if key in {"47", "48"}:
                            canonical = f"A2-F0@{item_seed}"
                        elif key in {"A2-F0@47", "A2-F0@48"}:
                            canonical = str(key)
                        else:
                            continue
                        if canonical in normalized:
                            continue
                        normalized[canonical] = item
                    comparisons = normalized
                    expected_comparisons = {"A2-F0@47", "A2-F0@48"}
                if set(comparisons) != expected_comparisons:
                    continue
                if any(not isinstance(item, Mapping) or not isinstance(item.get("vector_path"), str) or not _SHA256.fullmatch(str(item.get("vector_sha256"))) or item.get("matrix_sha256") != value.get("paired_sample_indices_sha256") and item.get("matrix_sha256") != value.get("sample_indices_sha256") or (item.get("evaluation_seeds") != [47, 48] if label == "accuracy" else item.get("evaluation_seeds") not in (None, [item.get("seed")] )) or item.get("passed") is not True for item in comparisons.values()):
                    continue
                if label == "retention" and any(item.get("seed") != int(name.rsplit("@", 1)[1]) for name, item in comparisons.items()):
                    continue
                if label == "accuracy" and (len({item["vector_path"] for item in comparisons.values()}) != 3 or len({item["vector_sha256"] for item in comparisons.values()}) != 3):
                    continue
                if label == "retention" and (len({item["vector_path"] for item in comparisons.values()}) != 2 or len({item["vector_sha256"] for item in comparisons.values()}) != 2):
                    continue
                derived = _derive_bootstrap_comparisons(root, inventory, label, expected_universe, paired, comparisons, aggregates)
                if derived is None:
                    continue
                observed_points = _derive_observed_comparison_points(root, inventory, label, expected_universe, aggregates)
                if observed_points is None or set(observed_points) != set(derived):
                    continue
                for name, item in comparisons.items():
                    vector = _read_bootstrap_vector(root, inventory, str(item["vector_path"]), item["vector_sha256"])
                    if vector is None:
                        break
                    bootstrap_values = np.asarray([float(value) for value in vector], dtype=np.float64)
                    expected_values = derived[name]
                    if len(expected_values) != len(bootstrap_values) or not np.allclose(bootstrap_values, expected_values, rtol=1e-12, atol=1e-12):
                        break
                    low = float(np.percentile(expected_values, 2.5))
                    high = float(np.percentile(expected_values, 97.5))
                    if not _is_finite_number(item.get("point")) or not _is_finite_number(item.get("ci_low")) or not _is_finite_number(item.get("ci_high")):
                        break
                    if not math.isclose(float(item["point"]), float(observed_points[name]), rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(float(item["ci_low"]), low, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(float(item["ci_high"]), high, rel_tol=1e-12, abs_tol=1e-12):
                        break
                    if label == "accuracy" and (float(item["point"]) < 0.02 or float(item["ci_low"]) <= 0.0):
                        break
                    if label == "retention" and (float(item["point"]) > 1.02 or float(item["ci_high"]) > 1.02):
                        break
                else:
                    return True
    return False


def _read_bootstrap_vector(root: Path, inventory: Mapping[str, Mapping[str, object]], relative: str, declared_hash: object, *, require_numeric: bool = True) -> list[object] | None:
    try:
        path = _resolve_evidence_path(root, relative)
        data = _read_verified(path, root, inventory)
    except (ValueError, OSError):
        return None
    if not isinstance(declared_hash, str) or not _SHA256.fullmatch(declared_hash) or hashlib.sha256(data).hexdigest() != declared_hash:
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    values = payload.get("values") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or len(values) != 10000:
        return None
    if require_numeric and any(not _is_finite_number(value) for value in values):
        return None
    return values


def _read_bootstrap_matrix(root: Path, inventory: Mapping[str, Mapping[str, object]], relative: str, declared_hash: object, expected_universe: Mapping[str, list[str]], *, registered_seed: int | None = None) -> dict[str, list[list[int]]] | None:
    try:
        path = _resolve_evidence_path(root, relative)
        data = _read_verified(path, root, inventory)
    except (ValueError, OSError):
        return None
    if not isinstance(declared_hash, str) or not _SHA256.fullmatch(declared_hash) or hashlib.sha256(data).hexdigest() != declared_hash:
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    families = payload.get("families") if isinstance(payload, Mapping) else None
    # Evidence JSON is canonicalized with sorted object keys.  The registered
    # family order is therefore defined by ``expected_universe`` below (and by
    # the RNG stream), not by incidental serialized object-key order.
    if not isinstance(families, Mapping) or set(families) != set(expected_universe):
        return None
    result: dict[str, list[list[int]]] = {}
    for family, expected_ids in expected_universe.items():
        item = families.get(family)
        if not isinstance(item, Mapping) or set(item) != {"ids", "indices"} or item.get("ids") != expected_ids or not isinstance(item.get("indices"), list) or len(item["indices"]) != 10000:
            return None
        rows: list[list[int]] = []
        for row in item["indices"]:
            if not isinstance(row, list) or len(row) != len(expected_ids) or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(expected_ids) for index in row):
                return None
            rows.append(row)
        result[family] = rows
    if registered_seed is not None:
        if isinstance(registered_seed, bool) or not isinstance(registered_seed, int) or registered_seed not in {9047, 9157}:
            return None
        rng = np.random.default_rng(registered_seed)
        for family, expected_ids in expected_universe.items():
            expected = rng.integers(0, len(expected_ids), size=(10000, len(expected_ids))).tolist()
            if result[family] != expected:
                return None
    return result


def _aggregate_rows(root: Path, inventory: Mapping[str, Mapping[str, object]], aggregate: Mapping[str, object]) -> list[dict[str, object]] | None:
    raw_path = aggregate.get("raw_path")
    if not isinstance(raw_path, str):
        return None
    try:
        return _read_jsonl(_resolve_evidence_path(root, raw_path), root, inventory)
    except ValueError:
        return None


def _derive_observed_comparison_points(
    root: Path,
    inventory: Mapping[str, Mapping[str, object]],
    label: str,
    expected_universe: Mapping[str, list[str]],
    aggregates: Mapping[tuple[object, object], Mapping[str, object]],
) -> dict[str, float] | None:
    """Recompute producer points from the persisted raw evaluation rows.

    The producer point is an observed statistic, not the mean of a bootstrap
    vector.  Keeping this derivation separate from vector reconciliation makes
    it impossible for a shape-valid or biased bootstrap file to define the
    reported point estimate.
    """
    seeds = (47, 48)
    if label == "accuracy":
        rows: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
        for key, aggregate in aggregates.items():
            arm, seed = str(key[0]).upper(), key[1]
            if arm not in _ARMS or seed not in seeds:
                continue
            raw = _aggregate_rows(root, inventory, aggregate)
            if raw is None:
                return None
            keyed: dict[str, dict[str, object]] = {}
            for item in raw:
                identifier = item.get("example_id")
                if not isinstance(identifier, str) or identifier in keyed:
                    return None
                keyed[identifier] = item
            rows[(arm, int(seed))] = keyed
        required = {(arm, seed) for arm in ("A1", "A2", "A3") for seed in seeds} | {("F0", 47)}
        if set(rows) != required:
            return None
        points: dict[str, float] = {}
        for name, reference in (("A2-F0", "F0"), ("A2-A1", "A1"), ("A2-A3", "A3")):
            family_means: list[float] = []
            for family, identifiers in expected_universe.items():
                differences: list[float] = []
                for identifier in identifiers:
                    paired: list[float] = []
                    for seed in seeds:
                        candidate = rows[("A2", seed)].get(identifier)
                        baseline = rows[(reference, 47 if reference == "F0" else seed)].get(identifier)
                        if (
                            not isinstance(candidate, Mapping)
                            or not isinstance(baseline, Mapping)
                            or not isinstance(candidate.get("correct"), bool)
                            or not isinstance(baseline.get("correct"), bool)
                        ):
                            return None
                        paired.append(float(candidate["correct"]) - float(baseline["correct"]))
                    differences.append(float(np.mean(paired)))
                if not differences:
                    return None
                family_means.append(float(np.mean(differences)))
            if len(family_means) != len(expected_universe) or not family_means or not all(math.isfinite(value) for value in family_means):
                return None
            points[name] = float(np.mean(family_means))
        return points
    if label == "retention":
        rows: dict[tuple[str, int], dict[int, Mapping[str, object]]] = {}
        for key, aggregate in aggregates.items():
            arm, seed = str(key[0]).upper(), key[1]
            if arm not in {"F0", "A2"} or seed not in seeds:
                continue
            raw = _aggregate_rows(root, inventory, aggregate)
            if raw is None:
                return None
            keyed: dict[int, Mapping[str, object]] = {}
            for item in raw:
                index = item.get("chunk_index")
                if isinstance(index, bool) or not isinstance(index, int) or index in keyed:
                    return None
                keyed[index] = item
            rows[(arm, int(seed))] = keyed
        if set(rows) != {("F0", 47), ("A2", 47), ("A2", 48)}:
            return None
        identifiers = expected_universe.get("retention")
        if not identifiers:
            return None
        points: dict[str, float] = {}
        for seed in seeds:
            candidate = [rows[("A2", seed)].get(int(index)) for index in identifiers]
            baseline = [rows[("F0", 47)].get(int(index)) for index in identifiers]
            if any(not isinstance(item, Mapping) for item in candidate + baseline):
                return None
            candidate_tokens = sum(int(item["scored_tokens"]) for item in candidate)
            baseline_tokens = sum(int(item["scored_tokens"]) for item in baseline)
            if candidate_tokens <= 0 or baseline_tokens <= 0:
                return None
            candidate_nll = sum(float(item["nll_sum"]) for item in candidate)
            baseline_nll = sum(float(item["nll_sum"]) for item in baseline)
            if not all(math.isfinite(value) for value in (candidate_nll, baseline_nll)):
                return None
            point = math.exp(candidate_nll / candidate_tokens - baseline_nll / baseline_tokens)
            if not math.isfinite(point) or point <= 0.0:
                return None
            points[f"A2-F0@{seed}"] = float(point)
        return points
    return None


def _derive_bootstrap_comparisons(root: Path, inventory: Mapping[str, Mapping[str, object]], label: str, expected_universe: Mapping[str, list[str]], matrix: Mapping[str, list[list[int]]], comparisons: Mapping[str, object], aggregates: Mapping[tuple[object, object], Mapping[str, object]]) -> dict[str, list[float]] | None:
    seeds = (47, 48)
    if label == "accuracy":
        rows: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
        for key, aggregate in aggregates.items():
            arm, seed = str(key[0]).upper(), key[1]
            if arm not in _ARMS or seed not in seeds:
                continue
            raw = _aggregate_rows(root, inventory, aggregate)
            if raw is None:
                return None
            rows[(arm, int(seed))] = {str(item["example_id"]): item for item in raw}
        required = {(arm, seed) for arm in ("A1", "A2", "A3") for seed in seeds} | {("F0", 47)}
        if set(rows) != required:
            return None
        vectors: dict[str, list[float]] = {}
        for name, reference in (("A2-F0", "F0"), ("A2-A1", "A1"), ("A2-A3", "A3")):
            family_values: dict[str, np.ndarray] = {}
            for family, identifiers in expected_universe.items():
                values = []
                for identifier in identifiers:
                    differences = []
                    for seed in seeds:
                        candidate = rows[("A2", seed)].get(identifier)
                        baseline = rows[(reference, 47 if reference == "F0" else seed)].get(identifier)
                        if not isinstance(candidate, Mapping) or not isinstance(baseline, Mapping) or not isinstance(candidate.get("correct"), bool) or not isinstance(baseline.get("correct"), bool):
                            return None
                        differences.append(float(candidate["correct"]) - float(baseline["correct"]))
                    values.append(float(np.mean(differences)))
                family_values[family] = np.asarray(values, dtype=np.float64)
            for seed in seeds:
                seed_family_means = [float(np.mean([float(rows[("A2", seed)][identifier]["correct"]) - float(rows[(reference, 47 if reference == "F0" else seed)][identifier]["correct"]) for identifier in identifiers])) for identifiers in expected_universe.values()]
                if float(np.mean(seed_family_means)) <= 0.0:
                    return None
            if any(float(values.mean()) < -0.02 for values in family_values.values()):
                return None
            vectors[name] = [float(np.mean([float(family_values[family][matrix[family][row_number]].mean()) for family in expected_universe])) for row_number in range(10000)]
        return vectors
    if label == "retention":
        rows: dict[tuple[str, int], dict[int, dict[str, object]]] = {}
        for key, aggregate in aggregates.items():
            arm, seed = str(key[0]).upper(), key[1]
            if arm not in {"F0", "A2"} or seed not in seeds:
                continue
            raw = _aggregate_rows(root, inventory, aggregate)
            if raw is None:
                return None
            rows[(arm, int(seed))] = {int(item["chunk_index"]): item for item in raw}
        if set(rows) != {("F0", 47), ("A2", 47), ("A2", 48)}:
            return None
        for seed in seeds:
            candidate_all = [rows[("A2", seed)][int(identifier)] for identifier in expected_universe["retention"]]
            baseline_all = [rows[("F0", 47)][int(identifier)] for identifier in expected_universe["retention"]]
            candidate_tokens = sum(int(item["scored_tokens"]) for item in candidate_all)
            baseline_tokens = sum(int(item["scored_tokens"]) for item in baseline_all)
            if candidate_tokens <= 0 or baseline_tokens <= 0:
                return None
            full_ratio = math.exp(sum(float(item["nll_sum"]) for item in candidate_all) / candidate_tokens - sum(float(item["nll_sum"]) for item in baseline_all) / baseline_tokens)
            if full_ratio > 1.02:
                return None
        vectors: dict[str, list[float]] = {}
        for seed in seeds:
            ratios: list[float] = []
            for row_number in range(10000):
                indices = matrix["retention"][row_number]
                candidate = [rows[("A2", seed)][int(expected_universe["retention"][index])] for index in indices]
                baseline = [rows[("F0", 47)][int(expected_universe["retention"][index])] for index in indices]
                candidate_tokens = sum(int(item["scored_tokens"]) for item in candidate)
                baseline_tokens = sum(int(item["scored_tokens"]) for item in baseline)
                if candidate_tokens <= 0 or baseline_tokens <= 0:
                    return None
                seed_ratio = math.exp(sum(float(item["nll_sum"]) for item in candidate) / candidate_tokens - sum(float(item["nll_sum"]) for item in baseline) / baseline_tokens)
                if seed_ratio > 1.02:
                    return None
                ratios.append(float(seed_ratio))
            vectors[f"A2-F0@{seed}"] = ratios
        return vectors
    return None


def _has_timing_gate(stages: Mapping[str, Mapping[str, object]]) -> bool:
    valid_seeds: set[int] = set()
    for stage_name, record in stages.items():
        if stage_name not in {"discover", "confirm"} or record.get("status") != "passed":
            continue
        expected_seed = 47 if stage_name == "discover" else 48
        for value in _walk_mappings(record.get("payload")):
            # A boolean assertion is not timing evidence.  Require persisted,
            # finite, positive realized samples and derive the ratio here.
            a2 = value.get("a2_training_times") or value.get("a2_realized_training_times")
            a3 = value.get("a3_training_times") or value.get("a3_realized_training_times")
            realized = value.get("realized_training_times") or value.get("realized_training_seconds")
            if isinstance(realized, Mapping):
                a2 = a2 or realized.get("a2") or realized.get("A2")
                a3 = a3 or realized.get("a3") or realized.get("A3")
            if a2 is None and a3 is None:
                a2 = value.get("a2_realized_training_time_seconds") or value.get("a2_training_time_seconds")
                a3 = value.get("a3_realized_training_time_seconds") or value.get("a3_training_time_seconds")
            def _positive_samples(samples: object) -> list[float]:
                if isinstance(samples, (int, float)) and not isinstance(samples, bool):
                    samples = [samples]
                if not isinstance(samples, list) or not samples:
                    return []
                result = [float(item) for item in samples if _is_finite_number(item) and float(item) > 0.0]
                return result if len(result) == len(samples) else []
            a2_values = _positive_samples(a2)
            a3_values = _positive_samples(a3)
            metadata = value.get("metadata")
            pilot_shape = (
                len(a2_values) == 20 and len(a3_values) == 20
                and value.get("sample_count") == 20
                and isinstance(metadata, Mapping)
                and metadata.get("warmup_samples") == 5
            )
            production_shape = (
                len(a2_values) == 1 and len(a3_values) == 1
                and value.get("sample_count") == 1
                and isinstance(metadata, Mapping)
                and metadata.get("measurement") == "complete_production_training"
            )
            if (not (pilot_shape or production_shape)
                    or metadata.get("units") != "seconds" or metadata.get("synchronized") is not True
                    or metadata.get("arms") != ["A2", "A3"]):
                continue
            ratio = float(np.median(a3_values)) / float(np.median(a2_values))
            if math.isfinite(ratio) and 0.90 <= ratio <= 1.10:
                seed = value.get("seed")
                if seed == expected_seed:
                    valid_seeds.add(seed)
    return valid_seeds == {47, 48}


def _walk_mappings(value: object) -> list[Mapping[str, object]]:
    found: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        found.append(value)
        for child in value.values():
            found.extend(_walk_mappings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_mappings(child))
    return found


def _failure_history(events: list[dict[str, object]], exact: Mapping[str, object], retention: Mapping[str, object], stages: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    failures = [event for event in events if event.get("kind") == "training_failure" or event.get("cuda_process_fault") is True or any(token in str(event.get(key, "")).lower() for key in ("kind", "outcome", "status") for token in ("failed", "failure", "fault", "fresh_process", "error"))]
    for section in (exact, retention):
        for aggregate in section.get("aggregates", []):
            if isinstance(aggregate, Mapping) and int(aggregate.get("failures", 0) or 0) > 0:
                failures.append({"kind": "evaluation_failure", "summary_path": aggregate.get("summary_path"), "count": aggregate.get("failures")})
    for name, record in stages.items():
        if record.get("status") == "failed_environment" and not any(item.get("stage") == name for item in failures):
            failures.append({"kind": "stage_failure", "stage": name, "reason": record.get("payload", {}).get("reason") if isinstance(record.get("payload"), Mapping) else None})
    return failures


def _resource_evidence(root: Path, inputs: list[dict[str, object]], inventory: Mapping[str, Mapping[str, object]], stages: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    ledger = root / "cuda-cost.jsonl"
    result: dict[str, object] = {"status": "not_evaluated", "cuda": "not_evaluated", "disk_bytes": sum(int(item["bytes"]) for item in inputs)}
    if ledger.is_file():
        rows = _read_jsonl(ledger, root, inventory)
        _validate_cuda_ledger(root, ledger, rows, inventory)
        result["cuda"] = {"status": "evaluated", "entries": rows, "cumulative_seconds": sum(float(row.get("elapsed_seconds", 0.0)) for row in rows if isinstance(row.get("elapsed_seconds"), (int, float)))}
        result["status"] = "evaluated"
    return result


def _validate_cuda_ledger(root: Path, path: Path, rows: list[dict[str, object]], inventory: Mapping[str, Mapping[str, object]]) -> None:
    previous = _CUDA_LEDGER_GENESIS
    cumulative = 0.0
    for line_number, row in enumerate(rows, 1):
        entry = _cuda_cost_entry(row, line_number)
        if entry.format != _CUDA_LEDGER_FORMAT or entry.previous_integrity != previous:
            raise ValueError("CUDA ledger chain is invalid")
        if entry.integrity != _entry_integrity(entry) or not math.isclose(entry.cumulative_seconds, cumulative + entry.elapsed_seconds, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("CUDA ledger chain arithmetic/integrity is invalid")
        anchor = _ledger_anchor_path(path, entry.index)
        if not anchor.is_file():
            raise ValueError("CUDA ledger anchor is missing")
        anchor_payload = _load_json(anchor, root=root, inventory=inventory)
        if anchor_payload != _ledger_anchor_payload(entry):
            raise ValueError("CUDA ledger anchor does not reconcile")
        previous = entry.integrity
        cumulative = entry.cumulative_seconds
    pending = path.with_name(path.name + ".pending.json")
    if pending.exists():
        raise ValueError("CUDA ledger contains an unresolved pending transaction")


def _provenance(stages: Mapping[str, Mapping[str, object]], manifest: Mapping[str, object] | None, signatures: Mapping[str, str]) -> dict[str, object]:
    source = stages.get("source", {}).get("payload", {}) if isinstance(stages.get("source", {}).get("payload", {}), Mapping) else {}
    return {
        "branch": source.get("branch", "unknown"),
        "protocol_revision": source.get("protocol_revision", manifest.get("protocol_revision") if manifest else "unknown"),
        "model_revision": manifest.get("revision", "not_evaluated") if manifest else "not_evaluated",
        "signatures": dict(signatures),
        "mixed_revision": bool(manifest and manifest.get("recovery_revision")),
        "mixed_revision_note": "source manifest records recovery provenance" if manifest and manifest.get("recovery_revision") else "none recorded",
    }


def _section_from_payload(stages: Mapping[str, Mapping[str, object]], keys: tuple[str, ...]) -> dict[str, object]:
    values: dict[str, object] = {}
    for record in stages.values():
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        for present, value in payload.items():
            if any(key == present or key in str(present).lower() for key in keys):
                values[str(present)] = value
    return values or {"status": "not_evaluated"}


def _examples_section(exact: Mapping[str, object]) -> dict[str, object]:
    return {"scored": exact.get("examples_scored", "not_evaluated"), "failed": exact.get("failures", "not_evaluated")}


def _families_section(exact: Mapping[str, object]) -> dict[str, object]:
    families: dict[str, object] = {}
    for aggregate in exact.get("aggregates", []):
        if isinstance(aggregate, Mapping) and isinstance(aggregate.get("family_summary"), Mapping):
            families[str(aggregate.get("arm", "unknown")) + "/" + str(aggregate.get("seed", "unknown"))] = aggregate["family_summary"]
    return families or {"status": "not_evaluated"}


def _transitions_section(exact: Mapping[str, object]) -> object:
    values = [item.get("transitions") for item in exact.get("aggregates", []) if isinstance(item, Mapping) and item.get("transitions")]
    return values or "not_evaluated"


def _ci_section(exact: Mapping[str, object]) -> object:
    values = [item for item in exact.get("aggregates", []) if isinstance(item, Mapping) and any(key in item for key in ("ci_low", "ci_high"))]
    return values or "not_evaluated"


def _retention_section(retention: Mapping[str, object]) -> object:
    values = [item for item in retention.get("aggregates", []) if isinstance(item, Mapping)]
    return values or "not_evaluated"


def _protocol_deviations(stages: Mapping[str, Mapping[str, object]], events: list[dict[str, object]]) -> list[object]:
    values: list[object] = []
    for record in stages.values():
        payload = record.get("payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("protocol_deviations"), list):
            values.extend(payload["protocol_deviations"])
    values.extend(event for event in events if event.get("outcome") == "protocol_deviation")
    return values


def _conclusion(decision: Mapping[str, object]) -> str:
    status = decision.get("status")
    reason = decision.get("reason")
    if status == "passed":
        base = "recurrent LoRA passed this fixed two-seed, three-family protocol."
    elif status == "stopped_scientific" and reason == "retention_uncertain":
        base = "task quality may have passed, but the registered retention bound was not established."
    elif status == "stopped_scientific" and reason == "timing_inconclusive":
        base = "task quality may have passed, but the timing-controlled comparison was inconclusive."
    elif status == "stopped_scientific":
        base = "recurrent LoRA did not pass the registered quality gate."
    elif status == "stopped_budget":
        base = "no scientific conclusion; the registered study did not fit the resource cap."
    else:
        base = "no scientific conclusion; execution evidence is incomplete because of an environment failure."
    return base + " Synthetic-task success is not general intelligence; recurrence is not sparse efficiency; uncached latency is not serving performance; and two fixed seeds do not establish population-wide seed robustness."


def _limitations() -> list[str]:
    return ["one checkpoint", "one loop window", "synthetic exact-answer tasks", "two fixed seeds", "uncached inference", "no sparse-efficiency or general-intelligence claim"]


def _next_action(decision: Mapping[str, object]) -> str:
    if decision.get("status") == "passed":
        return "terminal: no further registered experiment is authorized"
    if decision.get("status") in {"stopped_scientific", "stopped_budget", "failed_environment"}:
        return "terminal: report/closeout only; do not retune or start downstream work"
    return "not_evaluated"


def _render_markdown(report: Mapping[str, object]) -> str:
    decision = report["decision"]
    fields = ("run_root", "provenance", "signatures", "source_manifest", "stages", "evaluations", "examples", "families", "transitions", "accuracy_confidence_intervals", "retention_metrics", "training_metrics", "timing_metrics", "parameters", "layer_evaluations", "resources", "event_history", "failure_history", "protocol_deviations", "decision", "conclusion", "limitations", "next_action", "source_manifest_sha256", "report_sha256")
    lines = ["# Recurrence V0 report", "", f"- Decision: `{decision['status']}`", f"- Conclusion: {report['conclusion']}", f"- Next action: {report['next_action']}", "", "## Evidence", ""]
    for field in fields:
        lines.extend((f"### {field}", "", f"```json\n{json.dumps(report[field], sort_keys=True, indent=2)}\n```", ""))
    lines.extend(("## Limitations", ""))
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "## Missing work", "", "Absent downstream evaluations are recorded as `not_evaluated`; absence is never interpreted as zero accuracy.", ""])
    return "\n".join(lines)


def _load_events(root: Path, inventory: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    path = root / "events.jsonl"
    rows = _read_jsonl(path, root, inventory) if path.is_file() else []
    last_index: int | None = None
    for row in rows:
        if set(row) - {"index", "timestamp", "kind", "stage", "attempt", "payload", "outcome", "arm", "seed", "update", "cursor", "message", "error_type", "cuda_process_fault", "chargeable_cuda_seconds", "requires_fresh_process", "retry_status", "event_id", "checkpoint_hash", "checkpoint_sha256", "checkpoint_path", "checkpoint_manifest_digest", "failure_event_indices", "failure_event_ids", "attempts", "completion_status"}:
            raise ValueError("event record contains unknown fields")
        if isinstance(row.get("index"), bool) or not isinstance(row.get("index"), int) or row["index"] < 0:
            raise ValueError("event index is required and invalid")
        if not isinstance(row.get("timestamp"), str):
            raise ValueError("event timestamp is required and invalid")
        try:
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("event timestamp is required and invalid") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError("event timestamp must be UTC")
        if not isinstance(row.get("kind"), str) or row["kind"] not in _EVENT_KINDS:
            raise ValueError("event kind is required and not registered")
        kind = row["kind"]
        allowed_by_kind = {
            "stage_attempt": {"index", "timestamp", "kind", "stage", "attempt", "payload", "outcome"},
            "stage_complete": {"index", "timestamp", "kind", "stage", "attempt", "payload", "outcome"},
            "training_started": {"index", "timestamp", "kind", "arm", "seed", "attempt", "update"},
            "training_progress": {"index", "timestamp", "kind", "arm", "seed", "attempt", "update", "cursor"},
            "training_checkpoint": {"index", "timestamp", "kind", "arm", "seed", "attempt", "update", "checkpoint_hash", "checkpoint_path", "checkpoint_manifest_digest"},
            "training_completed": {"index", "timestamp", "kind", "arm", "seed", "attempt", "update", "cursor"},
            "training_failure": {"index", "timestamp", "kind", "stage", "arm", "seed", "attempt", "update", "message", "error_type", "cuda_process_fault", "chargeable_cuda_seconds", "requires_fresh_process", "retry_status", "checkpoint_hash", "checkpoint_sha256", "checkpoint_path", "checkpoint_manifest_digest", "event_id", "payload"},
            "failed_environment": {"index", "timestamp", "kind", "stage", "arm", "seed", "attempts", "failure_event_indices", "failure_event_ids", "completion_status", "requires_fresh_process", "payload"},
        }
        if set(row) - allowed_by_kind[kind]:
            raise ValueError(f"event {kind} contains fields outside its registered schema")
        required_by_kind = {
            "stage_attempt": {"stage", "attempt", "payload", "outcome"},
            "stage_complete": {"stage", "attempt", "payload", "outcome"},
            "training_started": {"arm", "seed", "attempt", "update"},
            "training_progress": {"arm", "seed", "attempt", "update", "cursor"},
            "training_checkpoint": {"arm", "seed", "attempt", "update", "checkpoint_hash", "checkpoint_path", "checkpoint_manifest_digest"},
            "training_completed": {"arm", "seed", "attempt", "update", "cursor"},
            "training_failure": {"stage", "arm", "seed", "attempt", "update", "event_id", "payload", "message", "error_type", "cuda_process_fault"},
            "failed_environment": {"stage", "arm", "seed", "attempts", "failure_event_indices", "failure_event_ids", "completion_status", "payload"},
        }
        if not required_by_kind[kind].issubset(row):
            raise ValueError(f"event {kind} is missing required fields")
        if "stage" in row and (not isinstance(row["stage"], str) or row["stage"] not in _STAGE_ORDER):
            raise ValueError("event stage is not registered")
        if "arm" in row and (not isinstance(row["arm"], str) or row["arm"].upper() not in _ARMS):
            raise ValueError("event arm is not registered")
        if kind in {"training_started", "training_progress", "training_checkpoint", "training_completed"} and row["arm"].upper() not in _ARMS:
            raise ValueError("training event arm is not registered")
        if "seed" in row and (isinstance(row["seed"], bool) or not isinstance(row["seed"], int) or row["seed"] not in {47, 48}):
            raise ValueError("event seed is invalid")
        if kind in {"training_started", "training_progress", "training_checkpoint", "training_completed"} and (isinstance(row["seed"], bool) or not isinstance(row["seed"], int) or row["seed"] not in {47, 48}):
            raise ValueError("training event seed is invalid")
        if "attempt" in row and (isinstance(row["attempt"], bool) or not isinstance(row["attempt"], int) or row["attempt"] < 1):
            raise ValueError("event attempt is invalid")
        for field in ("update", "cursor"):
            if field in row and (isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0):
                raise ValueError(f"event {field} is invalid")
        if kind in {"training_started", "training_progress", "training_checkpoint", "training_completed"} and any(field not in row or isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0 for field in ("update",)):
            raise ValueError("training event update is invalid")
        for field in ("checkpoint_hash", "checkpoint_sha256", "checkpoint_manifest_digest"):
            if field in row and (not isinstance(row[field], str) or not _SHA256.fullmatch(row[field])):
                raise ValueError(f"event {field} is invalid")
        if "checkpoint_path" in row:
            checkpoint_path = Path(row["checkpoint_path"]) if isinstance(row["checkpoint_path"], str) else None
            if checkpoint_path is None or not row["checkpoint_path"] or checkpoint_path.is_absolute() or ".." in checkpoint_path.parts:
                raise ValueError("event checkpoint path is invalid")
        for field in ("message", "error_type", "completion_status"):
            if field in row and not isinstance(row[field], str):
                raise ValueError(f"event {field} is invalid")
        if "event_id" in row and (not isinstance(row["event_id"], str) or not _SHA256.fullmatch(row["event_id"])):
            raise ValueError("event id is invalid")
        if "chargeable_cuda_seconds" in row and (isinstance(row["chargeable_cuda_seconds"], bool) or not isinstance(row["chargeable_cuda_seconds"], (int, float)) or not math.isfinite(float(row["chargeable_cuda_seconds"])) or float(row["chargeable_cuda_seconds"]) < 0):
            raise ValueError("event chargeable CUDA seconds are invalid")
        if "requires_fresh_process" in row and not isinstance(row["requires_fresh_process"], bool):
            raise ValueError("event fresh-process flag is invalid")
        if "retry_status" in row and row["retry_status"] not in {"fresh_process_required", "terminal"}:
            raise ValueError("event retry status is invalid")
        if "attempts" in row and (not isinstance(row["attempts"], list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in row["attempts"])):
            raise ValueError("event attempts are invalid")
        if "failure_event_indices" in row and (not isinstance(row["failure_event_indices"], list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in row["failure_event_indices"])):
            raise ValueError("event failure indexes are invalid")
        if "failure_event_ids" in row and (not isinstance(row["failure_event_ids"], list) or any(not isinstance(item, str) or not _SHA256.fullmatch(item) for item in row["failure_event_ids"])):
            raise ValueError("event failure IDs are invalid")
        if kind == "failed_environment" and (not row.get("attempts") or not row.get("failure_event_indices") or not row.get("failure_event_ids") or row.get("completion_status") != "failed_environment"):
            raise ValueError("failed-environment event is incomplete")
        if "outcome" in row and (not isinstance(row["outcome"], str) or row["outcome"] not in _EVENT_OUTCOMES):
            raise ValueError("event outcome is not registered")
        if "payload" in row and not isinstance(row["payload"], Mapping):
            raise ValueError("event payload is invalid")
        if kind == "training_failure" and (row["payload"].get("message") != row["message"]):
            raise ValueError("training failure payload does not match message")
        if kind == "failed_environment" and any(row["payload"].get(field) != row[field] for field in ("attempts", "failure_event_indices", "failure_event_ids")):
            raise ValueError("failed-environment payload does not match history")
        failure = row.get("kind") in {"training_failure", "failed_environment"} or str(row.get("outcome", "")).lower() in {"failed_environment", "failure", "error", "pilot_cleanup_failure"}
        if failure and not (isinstance(row.get("message"), str) or isinstance(row.get("payload"), Mapping)):
            raise ValueError("failure event has no payload")
        if row.get("kind") == "training_failure":
            if row.get("stage") not in _STAGE_ORDER or not isinstance(row.get("arm"), str) or row["arm"].upper() not in _ARMS or isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int) or row["seed"] not in {47, 48} or isinstance(row.get("update"), bool) or not isinstance(row.get("update"), int) or row["update"] < 0:
                raise ValueError("training failure stage/identity is invalid")
            if not isinstance(row.get("payload"), Mapping) or not isinstance(row.get("event_id"), str) or not _SHA256.fullmatch(row["event_id"]):
                raise ValueError("training failure is missing event identity/payload")
            if not isinstance(row.get("cuda_process_fault"), bool):
                raise ValueError("training failure CUDA flag is invalid")
            if row["event_id"] != task9_failure_event_id(row):
                raise ValueError("training failure event identity does not match canonical Task-9 fields")
        index = row["index"]
        if last_index is not None and index <= last_index:
            raise ValueError("event index is invalid or non-monotonic")
        last_index = index
    by_index = {row["index"]: row for row in rows}
    for row in rows:
        if row["kind"] != "failed_environment":
            continue
        referenced = row["failure_event_indices"]
        if any(index >= row["index"] or index not in by_index or by_index[index]["kind"] != "training_failure" for index in referenced):
            raise ValueError("failed-environment event does not reference complete failure history")
        referenced_attempts = [by_index[index]["attempt"] for index in referenced]
        referenced_ids = [by_index[index]["event_id"] for index in referenced]
        if row["attempts"] != referenced_attempts or row["failure_event_ids"] != referenced_ids:
            raise ValueError("failed-environment attempts/IDs do not match failure history")
        for index in referenced:
            failure = by_index[index]
            if any(row.get(field) != failure.get(field) for field in ("stage", "arm", "seed")):
                raise ValueError("failed-environment event crosses failure stage/arm/seed identity")
    return rows


def _read_jsonl(path: Path, root: Path, inventory: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    data = _read_verified(path, root, inventory)
    rows: list[dict[str, object]] = []
    try:
        lines = data.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid JSONL evidence: {path}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL evidence line {number}: {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL evidence line is not an object: {path}:{number}")
        _finite_tree(value, path)
        rows.append(value)
    return rows


def _resolve_evidence_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    raw = candidate if candidate.is_absolute() else root / candidate
    _reject_reparse_path(root, raw)
    path = raw.resolve()
    _inside(root, path)
    _reject_reparse_path(root, path)
    return path


def _inside(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"report evidence escapes run root: {path}") from error


def _finite_tree(value: object, path: Path) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite report evidence: {path}")
    if isinstance(value, Mapping):
        for child in value.values():
            _finite_tree(child, path)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child, path)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _write_immutable_text(path: Path, value: str) -> None:
    encoded = value.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"immutable evidence already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
