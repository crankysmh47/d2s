"""Detached, fail-closed stage supervision for registered executions."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from .execution import (
    ACTIVE_RUNTIME_SECONDS,
    ActiveJobContext,
    active_elapsed_seconds,
    append_event,
    artifact_baseline_bytes,
    initial_active_seconds,
    process_creation_utc,
    read_events,
    register_execution,
    validate_execution,
    validate_job_intervals,
    execution_stage_signature,
)
from .study import StageError, execution_stage_files, safe_output_path
from .windows_job import direct_venv_launch


_TERMINAL_STATES = {"completed", "stopped_environment", "stopped_scientific"}
_EXECUTION_STOP_STATES = {"stopped_environment", "stopped_scientific"}
_STATUS_LIMIT = 256 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical(dict(value)) + b"\n"
    if len(body) > _STATUS_LIMIT:
        raise ValueError("bounded execution metadata size exceeded")
    if immutable:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_status(root: Path, status: Mapping[str, Any]) -> None:
    record = dict(status)
    if "costs" not in record:
        try:
            manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
            record.update(_resource_projection(root, manifest))
        except Exception as exc:
            record.update(_resource_projection(root, {}, measurement_error=str(exc)))
    record["updated_utc"] = _utc_now()
    _atomic_json(root / "status.json", record)
    lines = [
        "# Distillation execution status", "",
        f"- State: `{record.get('state', 'unknown')}`",
        f"- Stage: `{record.get('stage') or 'none'}`",
        f"- Runtime used: `{record.get('runtime_used_seconds', 0)}` seconds",
        f"- Runtime remaining: `{record.get('runtime_remaining_seconds', 0)}` seconds",
        f"- Stop reason: {record.get('reason') or 'none'}",
        f"- Diagnostic: {record.get('diagnostic') or 'none'}",
        f"- CUDA: {_format_cost(record.get('costs', {}).get('cuda_seconds'))}",
        f"- Download: {_format_cost(record.get('costs', {}).get('download_bytes'))}",
        f"- Inherited costs: `{json.dumps(record.get('inherited_costs', {}), sort_keys=True)}`",
        f"- Artifact usage: `{json.dumps(record.get('artifact_usage', {}), sort_keys=True)}`",
        f"- Resource caps: `{json.dumps(record.get('resource_caps', {}), sort_keys=True)}`",
        f"- Cost provenance: `{json.dumps(record.get('cost_provenance', {}), sort_keys=True)}`",
        f"- Review packet: `{', '.join(record.get('review_packet_paths', [])) or 'none'}`",
        f"- Next action: {record.get('next_action') or 'inspect status and diagnostics'}",
        f"- Logs: `{root / 'logs'}`",
    ]
    body = ("\n".join(lines) + "\n").encode("utf-8")
    if len(body) > _STATUS_LIMIT:
        raise ValueError("bounded status markdown size exceeded")
    temporary = root / f".status.md.{os.getpid()}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "status.md")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validated_state(root: Path, *, repo_root: Path | None = None, config_path: Path | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = root / "execution.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    if config_path is None:
        config_path = Path(manifest["config_path"])
    checked_manifest = validate_execution(root, repo_root=repo_root, config_path=config_path)
    jobs = validate_job_intervals(root)
    return checked_manifest, jobs


def _record_payload(root: Path, record_name: str) -> dict[str, Any] | None:
    path = root / record_name
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{record_name} is not a JSON object")
    if record_name == "terminal.json":
        digest = value.get("record_sha256")
        unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
        if digest != _sha(unsigned) or value.get("execution_id") != root.name:
            raise ValueError("terminal record identity or digest is invalid")
    return value


def _format_cost(value: Mapping[str, Any] | None) -> str:
    if not value or value.get("total") is None:
        return "unavailable/incomplete"
    state = "complete" if value.get("complete") else "incomplete"
    return f"{value['total']} ({state})"


def _resource_projection(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any] | None = None,
    artifact_authenticated: bool = False,
    artifact_sha256: str | None = None,
    measurement_error: str | None = None,
) -> dict[str, Any]:
    inherited = manifest.get("inherited_costs", {}) if isinstance(manifest, Mapping) else {}
    inherited_cuda = inherited.get("cuda_seconds") if isinstance(inherited, Mapping) else None
    inherited_download = inherited.get("download_bytes") if isinstance(inherited, Mapping) else None
    ledger = artifact.get("resource_ledger") if artifact_authenticated and artifact else None
    ledger = ledger if isinstance(ledger, Mapping) else {}
    values: dict[str, int | None] = {}
    cost_fields_valid = artifact_authenticated
    for field in ("cuda_seconds", "download_bytes"):
        value = ledger.get(field)
        valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        if field == "cuda_seconds" and isinstance(inherited_cuda, int) and not isinstance(inherited_cuda, bool):
            valid = valid and value >= inherited_cuda if valid else False
        if field == "download_bytes" and isinstance(inherited_download, int) and not isinstance(inherited_download, bool):
            valid = valid and value >= inherited_download if valid else False
        values[field] = int(value) if valid else None
        cost_fields_valid = cost_fields_valid and valid
    policy = manifest.get("policy", {}) if isinstance(manifest, Mapping) else {}
    caps = {
        "active_runtime_seconds": policy.get("active_runtime_seconds"),
        "cuda_seconds": policy.get("cuda_seconds"),
        "download_bytes": policy.get("download_bytes"),
        "artifact_bytes": policy.get("artifact_bytes"),
        "artifact_terminal_reserve_bytes": policy.get("artifact_terminal_reserve_bytes"),
        "memory_reserve_bytes": policy.get("memory_reserve_bytes"),
    }
    try:
        inherited_artifact = artifact_baseline_bytes(manifest)
        measured_new = _artifact_bytes(root)
        measurement_complete = measurement_error is None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        inherited_artifact = 0
        measured_new = None
        measurement_complete = False
        measurement_error = measurement_error or str(exc)
    total_artifact = inherited_artifact + measured_new if measured_new is not None else None
    now = _utc_now()
    return {
        "costs": {
            "complete": bool(cost_fields_valid),
            "source": "authenticated settled calibration stage artifact" if cost_fields_valid else "unavailable; no authenticated complete stage ledger",
            "cuda_seconds": {"total": values["cuda_seconds"], "complete": values["cuda_seconds"] is not None},
            "download_bytes": {"total": values["download_bytes"], "complete": values["download_bytes"] is not None},
        },
        "inherited_costs": {
            "cuda_seconds": inherited_cuda,
            "download_bytes": inherited_download,
            "complete": type(inherited_cuda) is int and type(inherited_download) is int,
            "provenance": "immutable execution.json inherited_costs",
        },
        "artifact_usage": {
            "new_subtree_bytes": measured_new,
            "inherited_artifact_baseline_bytes": inherited_artifact,
            "total_bytes": total_artifact,
            "cap_bytes": caps["artifact_bytes"],
            "complete": measurement_complete,
            "measured_at_utc": now,
            "provenance": "measured execution-root subtree plus immutable execution.json artifact_baseline",
            "diagnostic": measurement_error,
        },
        "resource_caps": caps,
        "cost_provenance": {
            "execution_id": manifest.get("execution_id"),
            "source_run_id": manifest.get("source_run_id"),
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "policy_sha256": _sha(policy) if policy else None,
            "stage_artifact_sha256": artifact_sha256 if artifact_authenticated else None,
            "measured_at_utc": now,
        },
        "review_packet_paths": [],
    }


def _authenticated_stage_artifact(
    root: Path,
    manifest: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    result_event: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Authenticate the saved producer artifact against its lifecycle digest and nested signature."""
    digest = result_event.get("artifact_sha256")
    stage = result_event.get("stage")
    attempt_id = result_event.get("attempt_id")
    if not isinstance(digest, str) or not isinstance(stage, str) or not isinstance(attempt_id, str):
        return None, "stage result has no authenticated artifact digest or identity"
    path = root / "stages" / f"{stage}.json"
    try:
        body = path.read_bytes()
        if sha256(body).hexdigest() != digest:
            return None, "authenticated stage artifact changed after settlement"
        artifact = json.loads(body)
        if not isinstance(artifact, dict) or body != _canonical(artifact) + b"\n":
            return None, "authenticated stage artifact is not canonical JSON"
        if (
            artifact.get("execution_id") != manifest.get("execution_id")
            or artifact.get("stage") != stage
            or artifact.get("status") != result_event.get("state")
        ):
            return None, "authenticated stage artifact execution/stage identity mismatch"
        signature = artifact.get("signature")
        if not isinstance(signature, dict):
            return None, "authenticated stage artifact lacks nested producer signature"
        code_signatures = signature.get("code_signatures")
        if not isinstance(code_signatures, Mapping) or any(
            not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in code_signatures.values()
        ):
            return None, "authenticated stage artifact has malformed nested code signatures"
        signature_digest = signature.get("stage_signature")
        unsigned_signature = {key: item for key, item in signature.items() if key != "stage_signature"}
        job_state = jobs.get(attempt_id)
        start = job_state.get("start") if isinstance(job_state, Mapping) else None
        predecessor = signature.get("predecessors", {})
        if (
            not isinstance(signature_digest, str)
            or _sha(unsigned_signature) != signature_digest
            or artifact.get("stage_signature") != signature_digest
            or signature.get("stage") != stage
            or signature.get("source_run_id") != manifest.get("source_run_id")
            or not isinstance(predecessor, Mapping)
            or predecessor.get("prepare_manifest_sha256") != manifest.get("source_manifest_sha256")
            or predecessor.get("prepare_stage_sha256") != manifest.get("source_prepare_sha256")
            or not isinstance(start, Mapping)
            or start.get("stage") != stage
            or signature.get("execution_stage_signature") != start.get("stage_signature")
            or signature.get("execution_policy_sha256") != _sha(manifest.get("policy", {}))
            or artifact.get("reason") != result_event.get("reason")
        ):
            return None, "stage artifact nested signature digest/context is invalid"
        resource_caps = artifact.get("resource_caps")
        policy = manifest.get("policy", {})
        if not isinstance(resource_caps, Mapping) or any(
            resource_caps.get(result_key) != policy.get(policy_key)
            for result_key, policy_key in (
                ("cuda_seconds", "cuda_seconds"),
                ("wall_seconds", "active_runtime_seconds"),
                ("download_bytes", "download_bytes"),
                ("artifact_bytes", "artifact_bytes"),
            )
        ):
            return None, "stage artifact resource caps do not match immutable execution policy"
        return artifact, None
    except (OSError, ValueError, TypeError) as exc:
        return None, f"authenticated stage artifact cannot be read: {exc}"


def _safe_review_packet(root: Path, artifact: Mapping[str, Any]) -> str:
    relative = artifact.get("review_packet")
    digest = artifact.get("review_packet_sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("review packet path is missing or unsafe")
    if not isinstance(digest, str):
        raise ValueError("review packet digest is missing")
    path = safe_output_path(root, root / relative)
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("review packet is missing or changed")
    return path.relative_to(root).as_posix()


def _next_action(state: str, *, accounting_valid: bool = True) -> str:
    if not accounting_valid:
        return "Preserve the stop and inspect lifecycle/accounting diagnostics; do not resume compute."
    if state == "registered":
        return "Await root authorization; the only implemented production stage is calibrate."
    if state == "running":
        return "Monitor this execution's status and logs; do not start another worker."
    if state == "awaiting_review":
        return "Review the authenticated public packet; do not launch another stage before review gates."
    if state == "completed":
        return "Preserve this stage result; any later stage requires its own reviewed adapter and prerequisites."
    if state == "stopped_scientific":
        return "Preserve the scientific stop and evidence; no further compute is allowed."
    if state == "stopped_environment":
        return "Preserve the environment stop and inspect diagnostics; no automatic retry is allowed."
    return "Inspect authenticated status, costs, and diagnostics before taking action."


def _base_status(root: Path, manifest: Mapping[str, Any], jobs: Mapping[str, Mapping[str, Any]], *, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    settled = [
        state["settled_active_seconds"] for state in jobs.values()
        if state.get("settled_active_seconds") is not None
    ]
    initial_active = initial_active_seconds(manifest)
    used = settled[-1] if settled else initial_active
    current = next((state for state in jobs.values() if state.get("settlement") is None), None)
    current_owner_valid = True
    if current is not None:
        start = current["start"]
        context = ActiveJobContext(
            execution_id=str(start["execution_id"]), attempt_id=str(start["attempt_id"]),
            stage=str(start["stage"]), prior_active_seconds=int(start["prior_active_seconds"]),
            started_monotonic_ns=int(start["started_monotonic_ns"]),
            supervisor_pid=int(start["supervisor_pid"]),
            supervisor_created_utc=str(start["supervisor_created_utc"]),
            stage_signature=str(start["stage_signature"]),
        )
        try:
            owner_record = json.loads((root / "supervisor-owner.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            owner_record = {}
        current_owner_valid = (
            process_creation_utc(context.supervisor_pid) == context.supervisor_created_utc
            and owner_record.get("pid") == context.supervisor_pid
            and owner_record.get("created_utc") == context.supervisor_created_utc
            and owner_record.get("attempt_id") == context.attempt_id
        )
        try:
            current_bytes = (root / "current-job.json").read_bytes()
            current_record = json.loads(current_bytes)
            current_owner_valid = current_owner_valid and (
                current_bytes == _canonical(current_record) + b"\n"
                and current_record.get("payload") == start
                and current_record.get("payload_sha256") == _sha(start)
            )
        except (OSError, ValueError, TypeError):
            current_owner_valid = False
        if not current_owner_valid:
            used = limit = int(manifest["policy"]["active_runtime_seconds"])
        else:
            used = active_elapsed_seconds(context)
    limit = int(manifest["policy"]["active_runtime_seconds"])
    last_event = events[-1]["payload"] if events else {}
    result_envelope = next((
        item for item in reversed(events)
        if item["payload"].get("type") == "attempt_result"
    ), None)
    result_event = result_envelope["payload"] if result_envelope is not None else None
    artifact: dict[str, Any] | None = None
    artifact_diagnostic: str | None = None
    artifact_authenticated = False
    review_packet_paths: list[str] = []
    if result_event and result_event.get("artifact_sha256"):
        artifact, artifact_diagnostic = _authenticated_stage_artifact(root, manifest, jobs, result_event)
        artifact_authenticated = artifact is not None
        if artifact is not None and artifact.get("status") == "awaiting_review":
            try:
                review_packet_paths = [_safe_review_packet(root, artifact)]
            except (OSError, ValueError, TypeError) as exc:
                artifact_diagnostic = f"authenticated review handoff is invalid: {exc}"
                artifact_authenticated = False
    terminal = _record_payload(root, "terminal.json")
    if terminal:
        state_name = terminal.get("state", "stopped_environment")
        reason = terminal.get("reason")
    elif current is not None:
        start = current["start"]
        if not current_owner_valid:
            state_name, reason = "stopped_environment", "supervisor exited with an unsettled active interval; accounting is uncertain"
        else:
            state_name, reason = "running", None
    else:
        if result_event:
            state_name = result_event.get("state", "stopped_environment")
            reason = result_event.get("reason")
            if result_event.get("artifact_sha256") and artifact_diagnostic:
                state_name, reason = "stopped_environment", artifact_diagnostic
        elif last_event.get("type") == "registered":
            state_name, reason = "registered", None
        elif any(item["payload"].get("type") == "job_settled" for item in events):
            state_name, reason = "stopped_environment", "settled job lacks an authenticated stage outcome"
        else:
            state_name, reason = "registered", None
    owner_path = root / "supervisor-owner.json"
    if not terminal and owner_path.is_file():
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            owner_live = process_creation_utc(int(owner["pid"])) == owner.get("created_utc")
        except (OSError, ValueError, TypeError, KeyError):
            owner_live = False
        if not owner_live and current is None:
            if state_name in {"awaiting_review", "completed"}:
                status_lease_stale = True
            else:
                state_name, reason = "stopped_environment", "stale supervisor lease requires fail-closed review"
                status_lease_stale = True
        else:
            status_lease_stale = False
    else:
        status_lease_stale = False
    status = {
        "execution_id": manifest["execution_id"], "state": state_name,
        "stage": current["start"].get("stage") if current is not None else (result_event or last_event).get("stage"),
        "attempt_id": current["start"].get("attempt_id") if current is not None else (result_event or last_event).get("attempt_id"),
        "runtime_used_seconds": min(used, limit),
        "runtime_remaining_seconds": max(0, limit - used),
        "reason": reason, "logs_path": str(root / "logs"),
        "events_path": str(root / "events.jsonl"), "terminal_path": str(root / "terminal.json"),
        "status_json_path": str(root / "status.json"), "status_md_path": str(root / "status.md"),
        "snapshot_stale": True,
    }
    status.update(_resource_projection(
        root, manifest, artifact=artifact, artifact_authenticated=artifact_authenticated,
        artifact_sha256=result_event.get("artifact_sha256") if result_event else None,
    ))
    status["review_packet_paths"] = review_packet_paths
    if artifact_diagnostic:
        status["diagnostic"] = artifact_diagnostic
    status["cost_provenance"].update({
        "attempt_id": result_event.get("attempt_id") if result_event else None,
        "result_event_digest": result_envelope.get("payload_sha256") if result_envelope else None,
        "result_state": result_event.get("state") if result_event else None,
    })
    if state_name == "stopped_environment" and current is not None and reason and "uncertain" in reason:
        status["accounting_valid"] = False
    status["next_action"] = _next_action(status["state"], accounting_valid=status.get("accounting_valid", True))
    if status_lease_stale:
        status["supervisor_lease_stale"] = True
    return status


def execution_status(
    execution_root: Path,
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct status from immutable manifest and authenticated events."""
    root = Path(execution_root).resolve(strict=True)
    try:
        manifest, jobs = _validated_state(root, repo_root=repo_root, config_path=config_path)
        events = read_events(root)
        status = _base_status(root, manifest, jobs, events=events)
    except Exception as exc:
        # A corrupt journal must remain visible, while mutable snapshots never imply success.
        try:
            manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
            execution_id = manifest.get("execution_id")
            runtime_limit = int(manifest.get("policy", {}).get("active_runtime_seconds", ACTIVE_RUNTIME_SECONDS))
        except Exception:
            execution_id, runtime_limit = None, ACTIVE_RUNTIME_SECONDS
        status = {
            "execution_id": execution_id, "state": "stopped_environment",
            "runtime_used_seconds": runtime_limit, "runtime_remaining_seconds": 0,
            "reason": "lifecycle/accounting validation failed",
            "diagnostic": str(exc), "accounting_valid": False,
            "logs_path": str(root / "logs"), "events_path": str(root / "events.jsonl"),
            "terminal_path": str(root / "terminal.json"), "status_json_path": str(root / "status.json"),
            "status_md_path": str(root / "status.md"), "snapshot_stale": True,
        }
        status.update(_resource_projection(root, manifest if "manifest" in locals() else {}, measurement_error=str(exc)))
        status["next_action"] = _next_action("stopped_environment", accounting_valid=False)
        status["diagnostic"] = str(exc)
    try:
        convenience = _record_payload(root, "status.json")
    except (OSError, ValueError, TypeError):
        convenience = None
    if convenience is not None and convenience.get("state") == status.get("state"):
        status["snapshot_stale"] = False
        if status.get("state") == "running":
            try:
                updated = datetime.fromisoformat(str(convenience["updated_utc"]).replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - updated).total_seconds()
                status["snapshot_stale"] = age > 10
            except (KeyError, ValueError, TypeError):
                status["snapshot_stale"] = True
    _write_status(root, status)
    return status


def _artifact_bytes(root: Path) -> int:
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs:
            path = base / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"execution output contains a link: {path}")
        for name in files:
            path = base / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"execution output contains a link: {path}")
            total += path.stat().st_size
    return total


def _write_current_job(root: Path, payload: Mapping[str, Any]) -> None:
    _atomic_json(root / "current-job.json", {"payload": dict(payload), "payload_sha256": _sha(payload)})


def _write_terminal(root: Path, record: Mapping[str, Any]) -> None:
    payload = dict(record)
    payload["record_sha256"] = _sha(payload)
    _atomic_json(root / "terminal.json", payload, immutable=True)


def _acquire_lease(root: Path, attempt_id: str) -> dict[str, Any]:
    pid = os.getpid()
    created = process_creation_utc(pid)
    if created is None:
        raise RuntimeError("could not prove supervisor process identity")
    owner = {"pid": pid, "created_utc": created, "attempt_id": attempt_id}
    path = root / "supervisor-owner.json"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("an unreadable supervisor lease exists; refusing takeover") from exc
        if process_creation_utc(previous.get("pid")) == previous.get("created_utc"):
            raise RuntimeError("another live supervisor owns this execution")
        raise RuntimeError("stale supervisor lease requires fail-closed terminal review")
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(owner) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return owner


def _release_lease(root: Path, owner: Mapping[str, Any]) -> None:
    path = root / "supervisor-owner.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if existing == dict(owner):
        path.unlink()


def _artifact_baseline(manifest: Mapping[str, Any]) -> int:
    return artifact_baseline_bytes(manifest)


def _command(repo_root: Path, config_path: Path, manifest: Mapping[str, Any], execution_root: Path, attempt_id: str) -> list[str]:
    python = repo_root / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        python = Path(sys.executable)
    return [
        str(python), str(repo_root / "scripts" / "run_distillation_v1.py"),
        "calibrate", "--config", str(config_path),
        "--run-root", str(manifest["source_root"]),
        "--execution-root", str(execution_root), "--attempt-id", attempt_id,
    ]


def _save_terminal_state(root: Path, owner: Mapping[str, Any], attempt_id: str, state: str, reason: str | None, *, runtime_used: int, stage: str | None = None, artifact_sha256: str | None = None) -> None:
    record = {
        "execution_id": root.name, "attempt_id": attempt_id, "stage": stage,
        "state": state, "reason": reason, "runtime_used_seconds": runtime_used,
        "runtime_remaining_seconds": max(0, ACTIVE_RUNTIME_SECONDS - runtime_used),
        "artifact_sha256": artifact_sha256, "supervisor_pid": owner.get("pid"),
        "supervisor_created_utc": owner.get("created_utc"), "ended_utc": _utc_now(),
    }
    _write_terminal(root, record)
    append_event(root, {"type": "execution_stopped", **{key: value for key, value in record.items() if key not in {"record_sha256"}}})
    _write_status(root, record)


def _write_ack(root: Path, attempt_id: str, owner: Mapping[str, Any], state: str, reason: str | None = None) -> None:
    _atomic_json(root / "launches" / f"{attempt_id}.json", {
        "attempt_id": attempt_id, "supervisor_pid": owner["pid"],
        "supervisor_created_utc": owner["created_utc"], "state": state, "reason": reason,
    })


def supervise_stage(
    execution_root: Path,
    stage: str,
    *,
    repo_root: Path,
    config_path: Path,
    _job_factory: Callable[[], Any] | None = None,
    _argv_builder: Callable[[Path, Mapping[str, Any], str], Sequence[str]] | None = None,
    _clock_ns: Callable[[], int] = time.monotonic_ns,
    _sleep: Callable[[float], None] = time.sleep,
    _attempt_id: str | None = None,
    _expected_supervisor: Mapping[str, Any] | None = None,
) -> int:
    """Internal foreground supervisor; underscore dependencies are CPU-test only."""
    if os.name != "nt" and _job_factory is None:
        raise OSError("distillation background execution requires Windows Job Objects")
    if stage not in {"calibrate"}:
        # Validate registry spelling, then refuse every unimplemented stage.
        execution_stage_files(stage)
        raise StageError(f"execution stage {stage} is registered but not implemented")
    attempt_id = _attempt_id or uuid4().hex
    root = Path(execution_root).resolve(strict=True)
    owner: dict[str, Any] | None = None
    job: Any | None = None
    start_payload: dict[str, Any] | None = None
    settlement_written = False
    try:
        manifest = validate_execution(root, repo_root=repo_root, config_path=config_path)
        intervals = validate_job_intervals(root)
        terminal = _record_payload(root, "terminal.json")
        if terminal and terminal.get("state") in _EXECUTION_STOP_STATES:
            raise RuntimeError(f"execution is terminal: {terminal.get('state')}")
        if any(state["settlement"] is None for state in intervals.values()):
            raise RuntimeError("execution has an unsettled prior job; recovery is fail-closed")
        prior_attempts = [
            event["payload"] for event in read_events(root)
            if event["payload"].get("type") == "attempt_result"
        ]
        settled_attempts = {
            event["payload"].get("attempt_id") for event in read_events(root)
            if event["payload"].get("type") == "job_settled"
        }
        outcome_attempts = {item.get("attempt_id") for item in prior_attempts}
        if settled_attempts - outcome_attempts:
            raise RuntimeError("a prior settled interval lacks its authenticated stage outcome")
        if any(item.get("stage") == stage for item in prior_attempts):
            raise RuntimeError(f"stage {stage} already has a settled attempt; review its saved outcome")
        prior = max(
            (int(state["settled_active_seconds"]) for state in intervals.values()),
            default=initial_active_seconds(manifest),
        )
        if prior >= ACTIVE_RUNTIME_SECONDS:
            raise RuntimeError("cumulative active runtime budget is exhausted")
        owner = _acquire_lease(root, attempt_id)
        if _expected_supervisor and (
            owner["pid"] != _expected_supervisor.get("pid")
            or owner["created_utc"] != _expected_supervisor.get("created_utc")
        ):
            raise RuntimeError("supervisor identity differs from launcher's spawned process")
        signature = execution_stage_signature(root, stage, repo_root=repo_root)
        (root / "logs").mkdir(exist_ok=True)
        stdout_path = root / "logs" / f"{attempt_id}.stdout.log"
        stderr_path = root / "logs" / f"{attempt_id}.stderr.log"
        append_event(root, {
            "type": "job_reserved", "execution_id": root.name,
            "attempt_id": attempt_id, "stage": stage, "prior_active_seconds": prior,
            "stage_signature": signature, "supervisor_pid": owner["pid"],
            "supervisor_created_utc": owner["created_utc"],
        })
        if _job_factory is None:
            from .windows_job import WindowsJob
            job_factory = WindowsJob
        else:
            job_factory = _job_factory
        job = job_factory()
        if _argv_builder is None:
            argv = _command(Path(repo_root), Path(config_path), manifest, root, attempt_id)
        else:
            argv = list(_argv_builder(Path(repo_root), manifest, attempt_id))
        process = job.start(
            argv, cwd=Path(repo_root), stdout_path=stdout_path, stderr_path=stderr_path,
        )
        start_ns = _clock_ns()
        start_payload = {
            "type": "job_started", "execution_id": root.name,
            "attempt_id": attempt_id, "stage": stage,
            "prior_active_seconds": prior, "started_monotonic_ns": start_ns,
            "supervisor_pid": owner["pid"], "supervisor_created_utc": owner["created_utc"],
            "stage_pid": process["pid"], "stage_created_utc": process["created_utc"],
            "stage_signature": signature,
        }
        _write_current_job(root, start_payload)
        append_event(root, start_payload)
        # The exact PID/creation identity is durable and authenticated before execution starts.
        job.resume()
        _write_ack(root, attempt_id, owner, "running")
        prior_context = ActiveJobContext(
            root.name, attempt_id, stage, prior, start_ns, owner["pid"], owner["created_utc"], signature,
        )
        limit = int(manifest["policy"]["active_runtime_seconds"])
        inherited = _artifact_baseline(manifest)
        next_heartbeat = _clock_ns()
        reason: str | None = None
        returncode: int | None = None
        while True:
            now_ns = _clock_ns()
            used = active_elapsed_seconds(prior_context, now_ns=now_ns)
            try:
                generated_bytes = _artifact_bytes(root)
            except Exception as exc:
                reason = f"artifact inventory failed: {exc}"
                job.terminate_and_wait(10)
                break
            if inherited + generated_bytes > int(manifest["policy"]["artifact_bytes"]) - 1024 * 1024:
                reason = "projected artifacts, logs, journals, and status exceeded the registered cap"
                job.terminate_and_wait(10)
                break
            if used >= limit:
                reason = "cumulative active runtime budget exhausted"
                job.terminate_and_wait(10)
                break
            returncode = job.poll()
            if returncode is not None:
                if getattr(job, "active_processes", 0) == 0:
                    break
            if now_ns >= next_heartbeat:
                _write_status(root, {
                    "execution_id": root.name, "attempt_id": attempt_id,
                    "stage": stage, "state": "running", "runtime_used_seconds": used,
                    "runtime_remaining_seconds": max(0, limit - used),
                    "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                    "snapshot_stale": False,
                })
                next_heartbeat = now_ns + 5_000_000_000
            _sleep(1.0)
        if getattr(job, "active_processes", 0) != 0:
            reason = reason or "owned process tree exit could not be confirmed"
            _write_status(root, {
                "execution_id": root.name, "attempt_id": attempt_id,
                "stage": stage, "state": "stopped_environment", "reason": reason,
                "runtime_used_seconds": limit, "runtime_remaining_seconds": 0,
                "accounting_valid": False,
            })
            _write_ack(root, attempt_id, owner, "stopped_environment", reason)
            return 2
        ended_ns = _clock_ns()
        settled = prior + max(0, (ended_ns - start_ns + 999_999_999) // 1_000_000_000)
        settlement = {
            key: start_payload[key] for key in (
                "execution_id", "attempt_id", "stage", "prior_active_seconds",
                "started_monotonic_ns", "supervisor_pid", "supervisor_created_utc",
                "stage_pid", "stage_created_utc", "stage_signature",
            )
        }
        append_event(root, {
            "type": "job_settled", **settlement,
            "ended_monotonic_ns": ended_ns, "settled_active_seconds": settled,
        })
        settlement_written = True
        try:
            (root / "current-job.json").unlink()
        except FileNotFoundError:
            pass
        artifact_path = root / "stages" / f"{stage}.json"
        artifact_digest = None
        stage_state = "stopped_environment"
        stage_reason = reason
        if returncode == 0 and artifact_path.is_file():
            artifact_bytes = artifact_path.read_bytes()
            artifact = json.loads(artifact_bytes)
            if artifact_bytes != _canonical(artifact) + b"\n":
                stage_reason = "stage result is not canonical JSON"
            else:
                candidate_digest = sha256(artifact_bytes).hexdigest()
                candidate_event = {
                    "attempt_id": attempt_id, "stage": stage,
                    "state": artifact.get("status"), "reason": artifact.get("reason"),
                    "artifact_sha256": candidate_digest,
                }
                checked_artifact, signature_error = _authenticated_stage_artifact(
                    root, manifest, {attempt_id: {"start": start_payload}}, candidate_event,
                )
                if checked_artifact is None or artifact.get("status") not in {
                    "awaiting_review", "completed", "stopped_environment", "stopped_scientific",
                }:
                    stage_reason = signature_error or "zero exit lacked an authenticated expected stage result"
                else:
                    try:
                        if artifact["status"] == "awaiting_review":
                            _safe_review_packet(root, checked_artifact)
                    except (OSError, ValueError, TypeError) as exc:
                        stage_reason = f"review packet handoff is invalid: {exc}"
                    else:
                        stage_state = str(artifact["status"])
                        stage_reason = artifact.get("reason")
                        artifact_digest = candidate_digest
        elif stage_reason is None:
            stage_reason = "stage exited without its expected signed result" if returncode == 0 else f"stage process exited with code {returncode}"
        if settled >= limit and stage_state not in {"stopped_environment", "stopped_scientific"}:
            stage_state, stage_reason = "stopped_environment", "cumulative active runtime budget exhausted"
        append_event(root, {
            "type": "attempt_result", "execution_id": root.name,
            "attempt_id": attempt_id, "stage": stage, "state": stage_state,
            "reason": stage_reason, "artifact_sha256": artifact_digest,
        })
        if stage_state == "awaiting_review":
            final_record = {
                "execution_id": root.name, "attempt_id": attempt_id, "stage": stage,
                "state": "awaiting_review", "reason": stage_reason,
                "runtime_used_seconds": settled, "runtime_remaining_seconds": max(0, limit - settled),
                "artifact_sha256": artifact_digest,
                "review_packet_paths": [artifact["review_packet"]],
            }
            append_event(root, {"type": "stage_stopped", **final_record})
        elif stage_state in _EXECUTION_STOP_STATES:
            _save_terminal_state(root, owner, attempt_id, stage_state, stage_reason,
                                 runtime_used=settled, stage=stage, artifact_sha256=artifact_digest)
        elif stage_state == "completed":
            pass
        else:
            stage_state = "stopped_environment"
            stage_reason = stage_reason or "stage produced an unrecognized outcome"
            _save_terminal_state(root, owner, attempt_id, stage_state, stage_reason,
                                 runtime_used=settled, stage=stage, artifact_sha256=artifact_digest)
        projected_status = execution_status(root, repo_root=repo_root, config_path=config_path)
        projected_status["snapshot_stale"] = False
        _write_status(root, projected_status)
        _write_ack(root, attempt_id, owner, stage_state, stage_reason)
        job.close()
        _release_lease(root, owner)
        return 0 if stage_state in {"awaiting_review", "completed", "stopped_scientific"} else 2
    except BaseException as exc:
        cleanup_confirmed = job is None
        if job is not None:
            try:
                cleanup_confirmed = bool(job.terminate_and_wait(10))
            except Exception:
                cleanup_confirmed = False
            if cleanup_confirmed and start_payload is not None and not settlement_written:
                try:
                    ended_ns = _clock_ns()
                    prior_seconds = int(start_payload["prior_active_seconds"])
                    start_ns = int(start_payload["started_monotonic_ns"])
                    settlement = {
                        key: start_payload[key] for key in (
                            "execution_id", "attempt_id", "stage", "prior_active_seconds",
                            "started_monotonic_ns", "supervisor_pid", "supervisor_created_utc",
                            "stage_pid", "stage_created_utc", "stage_signature",
                        )
                    }
                    settled = prior_seconds + max(0, (ended_ns - start_ns + 999_999_999) // 1_000_000_000)
                    append_event(root, {"type": "job_settled", **settlement,
                                        "ended_monotonic_ns": ended_ns,
                                        "settled_active_seconds": settled})
                    settlement_written = True
                    try:
                        (root / "current-job.json").unlink()
                    except FileNotFoundError:
                        pass
                    append_event(root, {
                        "type": "attempt_result", "execution_id": root.name,
                        "attempt_id": attempt_id, "stage": stage,
                        "state": "stopped_environment", "reason": str(exc),
                    })
                    if owner is not None and not (root / "terminal.json").exists():
                        _save_terminal_state(root, owner, attempt_id, "stopped_environment", str(exc),
                                             runtime_used=settled, stage=stage)
                except Exception:
                    cleanup_confirmed = False
            try:
                job.close()
            except Exception:
                cleanup_confirmed = False
        if owner is not None:
            if not cleanup_confirmed:
                try:
                    _write_status(root, {
                        "execution_id": root.name, "attempt_id": attempt_id,
                        "stage": stage, "state": "stopped_environment",
                        "reason": "owned process cleanup could not be confirmed; accounting is uncertain",
                        "runtime_used_seconds": ACTIVE_RUNTIME_SECONDS,
                        "runtime_remaining_seconds": 0, "accounting_valid": False,
                    })
                except Exception:
                    pass
            try:
                _write_ack(root, attempt_id, owner, "stopped_environment", str(exc))
            except Exception:
                pass
        if cleanup_confirmed and not (root / "current-job.json").exists():
                _release_lease(root, owner)
        else:
            try:
                pid = os.getpid()
                created = process_creation_utc(pid)
                if created is not None:
                    _write_ack(root, attempt_id, {"pid": pid, "created_utc": created}, "refused", str(exc))
            except Exception:
                pass
        raise


def launch_stage(
    execution_root: Path,
    stage: str,
    *,
    repo_root: Path,
    config_path: Path,
    _supervisor_argv_builder: Callable[[Path, str, Path, Path, str], Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Spawn a detached supervisor and return only its own acknowledged ownership."""
    if os.name != "nt":
        raise OSError("detached production launch requires Windows")
    if stage != "calibrate":
        execution_stage_files(stage)
        raise StageError(f"execution stage {stage} is registered but not implemented")
    root = Path(execution_root).resolve(strict=True)
    validate_execution(root, repo_root=repo_root, config_path=config_path)
    intervals = validate_job_intervals(root)
    terminal = _record_payload(root, "terminal.json")
    events = read_events(root)
    settled_attempts = {
        event["payload"].get("attempt_id") for event in events
        if event["payload"].get("type") == "job_settled"
    }
    outcome_attempts = {
        event["payload"].get("attempt_id") for event in events
        if event["payload"].get("type") == "attempt_result"
    }
    if terminal and terminal.get("state") in _EXECUTION_STOP_STATES:
        raise RuntimeError(f"execution is terminal: {terminal.get('state')}")
    if settled_attempts - outcome_attempts:
        raise RuntimeError("a settled stage lacks its authenticated outcome")
    if any(job["settlement"] is None for job in intervals.values()):
        raise RuntimeError("execution is terminal or has an unsettled job")
    prior_results = [event["payload"] for event in events if event["payload"].get("type") == "attempt_result"]
    if any(item.get("stage") == stage for item in prior_results):
        raise RuntimeError(f"stage {stage} already has a settled attempt; review its saved outcome")
    attempt_id = uuid4().hex
    python = Path(repo_root) / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        python = Path(sys.executable)
    bootstrap = root / "logs" / f"{attempt_id}.supervisor.log"
    bootstrap.parent.mkdir(parents=True, exist_ok=True)
    if _supervisor_argv_builder is None:
        command = [
            str(python), str(Path(repo_root) / "scripts" / "run_distillation_background.py"),
            "run", stage, "--config", str(config_path), "--execution-root", str(root),
            "--repo-root", str(repo_root), "--attempt-id", attempt_id,
        ]
    else:
        command = list(_supervisor_argv_builder(root, stage, Path(config_path), Path(repo_root), attempt_id))
    command, executable, environment = direct_venv_launch(command)
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    with bootstrap.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command, cwd=repo_root, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=stream, close_fds=True, creationflags=flags,
            executable=executable, env=environment,
        )
    created = process_creation_utc(process.pid)
    if created is None:
        raise RuntimeError("launcher could not prove its spawned supervisor identity")
    expected = {"pid": process.pid, "created_utc": created}
    deadline = time.monotonic() + 10
    ack_path = root / "launches" / f"{attempt_id}.json"
    while time.monotonic() < deadline:
        if ack_path.is_file():
            try:
                ack_text = ack_path.read_text(encoding="utf-8")
            except (FileNotFoundError, PermissionError):
                pass
            else:
                ack = json.loads(ack_text)
                if ack.get("attempt_id") != attempt_id or ack.get("supervisor_pid") != expected["pid"] or ack.get("supervisor_created_utc") != expected["created_utc"]:
                    raise RuntimeError("launch acknowledgment does not match this launcher’s supervisor")
                state = ack.get("state")
                if state == "running" and process_creation_utc(process.pid) == created:
                    return {"state": "running", "execution_root": str(root), "attempt_id": attempt_id,
                            "supervisor_pid": process.pid, "supervisor_created_utc": created,
                            "status_path": str(root / "status.md"), "logs_path": str(root / "logs")}
                if state in _TERMINAL_STATES or state == "awaiting_review":
                    return {"state": state, "execution_root": str(root), "attempt_id": attempt_id,
                            "supervisor_pid": process.pid, "supervisor_created_utc": created,
                            "reason": ack.get("reason"), "status_path": str(root / "status.md"),
                            "logs_path": str(root / "logs")}
                if state == "refused":
                    raise RuntimeError(ack.get("reason") or "another supervisor owns this execution")
        if process.poll() is not None and (root / "terminal.json").is_file():
            terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
            if terminal.get("attempt_id") == attempt_id:
                return {"state": terminal.get("state"), "execution_root": str(root),
                        "attempt_id": attempt_id, "reason": terminal.get("reason"),
                        "status_path": str(root / "status.md"), "logs_path": str(root / "logs")}
        time.sleep(0.05)
    raise RuntimeError(f"supervisor did not acknowledge owned startup within 10 seconds; see {bootstrap}")


def register_execution_cli(source_root: Path, executions_root: Path, *, repo_root: Path, config_path: Path) -> Path:
    root = register_execution(source_root, executions_root, repo_root=repo_root, config_path=config_path)
    manifest = validate_execution(root, repo_root=repo_root, config_path=config_path)
    append_event(root, {
        "type": "registered", "execution_id": manifest["execution_id"],
        "created_utc": manifest["created_utc"],
    })
    _write_status(root, {
        "execution_id": manifest["execution_id"], "state": "registered",
        "runtime_used_seconds": 0, "runtime_remaining_seconds": ACTIVE_RUNTIME_SECONDS,
        "reason": None,
    })
    return root


def register_recovery_cli(
    previous_execution_root: Path,
    executions_root: Path,
    *, repo_root: Path,
    config_path: Path,
) -> Path:
    from .recovery import register_recovery

    root = register_recovery(
        previous_execution_root, executions_root,
        repo_root=repo_root, config_path=config_path,
    )
    manifest = validate_execution(root, repo_root=repo_root, config_path=config_path)
    append_event(root, {
        "type": "registered", "execution_id": manifest["execution_id"],
        "created_utc": manifest["created_utc"], "recovery": True,
    })
    initial = initial_active_seconds(manifest)
    _write_status(root, {
        "execution_id": manifest["execution_id"], "state": "registered",
        "runtime_used_seconds": initial,
        "runtime_remaining_seconds": max(0, ACTIVE_RUNTIME_SECONDS - initial),
        "reason": None,
    })
    return root


def register_repair_cli(
    previous_execution_root: Path,
    executions_root: Path,
    *, repo_root: Path,
    config_path: Path,
) -> Path:
    from .repair_recovery import register_repair

    root = register_repair(
        previous_execution_root, executions_root,
        repo_root=repo_root, config_path=config_path,
    )
    manifest = validate_execution(root, repo_root=repo_root, config_path=config_path)
    append_event(root, {
        "type": "registered", "execution_id": manifest["execution_id"],
        "created_utc": manifest["created_utc"], "repair": True,
    })
    initial = initial_active_seconds(manifest)
    _write_status(root, {
        "execution_id": manifest["execution_id"], "state": "registered",
        "runtime_used_seconds": initial,
        "runtime_remaining_seconds": max(0, ACTIVE_RUNTIME_SECONDS - initial),
        "reason": None,
    })
    return root
