"""One-hop, evidence-pinned recovery registration for the stopped Task 2 execution."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .execution import _canonical, _checked_path, _inventory, _sha_file


PREDECESSOR_ID = "fc901bb0766a9c79fb205302d220fe727a99ebbd0b315e6ee4ad4e66e6733634"
RECOVERY_PROTOCOL = "distillation-task2-one-hop-recovery-v1"
RECOVERY_REVISION = 1
STATUS_FILES = ("status.json", "status.md")
REVIEWED_IMMUTABLE_FILES: dict[str, str] = {
    "execution.json": "400eb64e940c8ba9e9ed4de7e3135573c1614fbff87e5e349b02e3641b432931",
    "events.jsonl": "14c440a1dcd2eb31e4f6d0f6d99533c4eb5422e61faae9e4f8a80cc89445d5be",
    "terminal.json": "e81f942480cd605e3e5bcc2612404456c587d4840a0dad6321240865828269aa",
    "stages/calibrate.json": "e8ee6708573b9ba10df1677befa5044c5b8d84e7c9f42d86db446a93f11411f1",
    "calibration/raw/256-768-student.jsonl.attempts.jsonl": "e7ecddc7dc198250a81d837be5efc78a51f6bfc537054521ced2344f14b341bc",
    "stages/calibrate-reservations.jsonl": "31a4342e00a7ea8b011d2d7b0c51756e15c25f10c4af1c4f80e6ad6af6831079",
    "calibration/model-inventory/teacher.json": "5bde59ee552551a8c81266a79019e62081accb99c614254ea508b0bd17dae74a",
    "calibration/model-inventory/student.json": "b771e48e0eca0812c4a2271fe565cdecd2970bc5b2f0875bbfc27df7bd015566",
    "launches/dd45bbdd2ecf45cb8d8fcc653132ee0a.json": "4d84ad1ef2a115074389dbbd1f2991ad2db8cf6ef5a553f87826f77a8e5a96fe",
    "logs/dd45bbdd2ecf45cb8d8fcc653132ee0a.stderr.log": "7a94e8679ed0fbabfc9d2df058470b2c83879875bcfd30ebfa7209e8d2ea2912",
    "logs/dd45bbdd2ecf45cb8d8fcc653132ee0a.stdout.log": "4272c8c0f2386c86914ab884a93c997687239dc7ef36b6ed10649187d3ff613e",
    "logs/dd45bbdd2ecf45cb8d8fcc653132ee0a.supervisor.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "stages/calibrate-attempts.jsonl": "49ee83ee44d35f3b680faaf08a0edb3748332c148e0312f8e118edbe90713b8e",
}
_EXPECTED_STATUS_BYTES = 4482
_EXPECTED_ACTIVE_SECONDS = 31
_EXPECTED_COSTS = {"cuda_seconds": 1199, "download_bytes": 8066717862}
_EXPECTED_TEACHER_RESERVATION_BYTES = 18427


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _expected_inventory(root: Path, *, frozen_status_bytes: int | None = None) -> tuple[list[dict[str, Any]], int, int]:
    all_files, _ = _inventory(root)
    observed_status: dict[str, int] = {}
    retained: list[dict[str, Any]] = []
    for row in all_files:
        path = row["path"]
        if path in STATUS_FILES:
            observed_status[path] = row["size_bytes"]
        else:
            retained.append(row)
    if set(observed_status) != set(STATUS_FILES):
        raise ValueError("predecessor must contain exactly the two reviewed convenience status files")
    status_bytes = sum(observed_status.values())
    if frozen_status_bytes is None and status_bytes != _EXPECTED_STATUS_BYTES:
        raise ValueError("predecessor convenience status byte reserve differs from reviewed evidence")
    if frozen_status_bytes is not None:
        status_bytes = frozen_status_bytes
    if len(retained) != len(REVIEWED_IMMUTABLE_FILES):
        raise ValueError("predecessor immutable inventory has unexpected or missing files")
    actual = {row["path"]: row["sha256"] for row in retained}
    if actual != REVIEWED_IMMUTABLE_FILES:
        raise ValueError("predecessor immutable inventory differs from reviewed pinned evidence")
    return retained, sum(row["size_bytes"] for row in retained), status_bytes


def _check_no_live_owner(previous: Path, terminal: Mapping[str, Any], events: list[dict[str, Any]]) -> None:
    from .execution import process_creation_utc

    owners: list[tuple[Any, Any]] = []
    for key in ("supervisor_pid", "stage_pid"):
        pid = terminal.get(key)
        created = terminal.get("supervisor_created_utc" if key == "supervisor_pid" else "stage_created_utc")
        if pid is not None:
            owners.append((pid, created))
    for event in events:
        payload = event["payload"]
        if payload.get("type") in {"job_started", "job_settled"}:
            owners.extend(((payload.get("supervisor_pid"), payload.get("supervisor_created_utc")),
                           (payload.get("stage_pid"), payload.get("stage_created_utc"))))
    if any(type(pid) is int and process_creation_utc(pid) == created for pid, created in owners):
        raise ValueError("predecessor still has a live process owner")
    current_job = previous / "current-job.json"
    if current_job.exists():
        value = json.loads(current_job.read_text(encoding="utf-8"))
        payload = value.get("payload", {})
        if type(payload.get("stage_pid")) is int and process_creation_utc(payload["stage_pid"]) == payload.get("stage_created_utc"):
            raise ValueError("predecessor current job still has a live stage owner")


def _verify_predecessor(
    previous: Path, *, repo_root: Path, config_path: Path,
    frozen_status_bytes: int | None = None,
) -> dict[str, Any]:
    from .execution import read_events, validate_execution, validate_job_intervals

    if previous.name != PREDECESSOR_ID:
        raise ValueError("only the pinned stopped predecessor is eligible for one-hop recovery")
    if _sha_file(previous / "execution.json") != REVIEWED_IMMUTABLE_FILES["execution.json"]:
        raise ValueError("predecessor execution identity differs from reviewed pin")
    manifest = validate_execution(previous, repo_root=repo_root, config_path=config_path)
    if manifest.get("schema_version") != 1 or manifest.get("execution_id") != PREDECESSOR_ID:
        raise ValueError("recovery predecessor must be the pinned legacy schema 1 execution")
    inventory, immutable_bytes, status_bytes = _expected_inventory(
        previous, frozen_status_bytes=frozen_status_bytes,
    )
    events = read_events(previous)
    jobs = validate_job_intervals(previous)
    if len(jobs) != 1:
        raise ValueError("predecessor must contain exactly one settled calibration interval")
    job = next(iter(jobs.values()))
    if job["settlement"] is None or job["settled_active_seconds"] != _EXPECTED_ACTIVE_SECONDS:
        raise ValueError("predecessor active interval does not settle at the reviewed 31 seconds")
    terminal_path = previous / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("execution_id") != PREDECESSOR_ID or terminal.get("state") != "stopped_environment":
        raise ValueError("predecessor terminal state is not the reviewed environment stop")
    if terminal.get("runtime_used_seconds") != _EXPECTED_ACTIVE_SECONDS or terminal.get("runtime_remaining_seconds") != 172769:
        raise ValueError("predecessor terminal clock differs from the settled lifecycle")
    if terminal.get("reason") != "authenticated stage artifact execution/stage identity mismatch":
        raise ValueError("predecessor terminal cause differs from the pinned producer failure")
    attempt_results = [event["payload"] for event in events if event["payload"].get("type") == "attempt_result"]
    if len(attempt_results) != 1 or attempt_results[0].get("state") != "stopped_environment" or attempt_results[0].get("artifact_sha256") is not None:
        raise ValueError("predecessor must retain exactly one sealed failed producer outcome without an artifact")
    artifact = json.loads((previous / "stages" / "calibrate.json").read_text(encoding="utf-8"))
    if artifact.get("execution_id") != PREDECESSOR_ID or artifact.get("status") != "stopped_environment":
        raise ValueError("predecessor stage artifact is not the pinned failed producer record")
    if artifact.get("signature", {}).get("execution_stage_signature") != "d78060dc7839b906b45fec7231ec2f30e851ae5c8ad58d72d906aca10cc14199":
        raise ValueError("predecessor stage artifact lacks the original stored producer signature")
    producer_signature = artifact.get("signature", {})
    for role in ("teacher", "student"):
        inventory_path = previous / "calibration" / "model-inventory" / f"{role}.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        canonical_inventory = json.dumps(inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        expected_digest = sha256(canonical_inventory).hexdigest()
        if producer_signature.get("model_inventory_sha256", {}).get(role) != expected_digest:
            raise ValueError(f"predecessor {role} inventory does not reconcile with the stored producer signature")
    attempts_path = previous / "calibration" / "raw" / "256-768-student.jsonl.attempts.jsonl"
    attempts = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines()]
    failures = [row for row in attempts if row.get("status") == "worker_load_failed" and row.get("role") == "student"]
    if len(failures) != 3 or any(row.get("error_type") != "RuntimeError" for row in failures):
        raise ValueError("predecessor does not retain exactly three failed student loads")
    raw_answers = [path for path in (previous / "calibration" / "raw").glob("*.jsonl") if not path.name.endswith(".attempts.jsonl")]
    if raw_answers or artifact.get("attempts", [{}])[0].get("raw_rows") != []:
        raise ValueError("predecessor contains completed or raw answer rows")
    reservation_rows = [json.loads(line) for line in (previous / "stages" / "calibrate-reservations.jsonl").read_text(encoding="utf-8").splitlines()]
    teacher_reservations = [row for row in reservation_rows if row.get("event_type") == "download_reservation"]
    teacher_completions = [row for row in reservation_rows if row.get("event_type") == "download_complete"]
    if (len(teacher_reservations) != 1 or len(teacher_completions) != 1
            or teacher_reservations[0].get("role") != "teacher"
            or teacher_reservations[0].get("reserved_bytes") != _EXPECTED_TEACHER_RESERVATION_BYTES
            or teacher_reservations[0].get("missing_files") != [".gitattributes", "README.md"]
            or teacher_reservations[0].get("status") != "reserved"
            or teacher_completions[0].get("role") != "teacher"
            or teacher_completions[0].get("status") != "completed"
            or teacher_reservations[0].get("repo_id") != teacher_completions[0].get("repo_id")
            or teacher_reservations[0].get("revision") != teacher_completions[0].get("revision")):
        raise ValueError("predecessor teacher metadata reservation/completion does not reconcile")
    if artifact.get("resource_ledger", {}).get("cuda_seconds") != 1190 or artifact.get("resource_ledger", {}).get("download_bytes") != _EXPECTED_COSTS["download_bytes"]:
        raise ValueError("predecessor producer costs do not reconcile with retained reservations")
    _check_no_live_owner(previous, terminal, events)
    return {
        "manifest": manifest, "events": events, "inventory": inventory,
        "immutable_bytes": immutable_bytes, "status_bytes": status_bytes,
        "active_seconds": _EXPECTED_ACTIVE_SECONDS, "costs": dict(_EXPECTED_COSTS),
    }


def _reservation_path(previous: Path) -> Path:
    return previous.parent / f"recovery-authorization-{PREDECESSOR_ID}.json"


def _load_authorization(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("recovery authorization reservation is malformed or oversized")
    value = json.loads(raw)
    if raw != _canonical(value) + b"\n":
        raise ValueError("recovery authorization reservation is not canonical")
    return value, raw


def register_recovery(
    previous_execution_root: Path,
    executions_root: Path,
    *,
    repo_root: Path,
    config_path: Path,
) -> Path:
    """Create the single exclusive schema 2 execution linked to the reviewed stop."""
    previous = _checked_path(Path(previous_execution_root)).resolve(strict=True)
    executions = _checked_path(Path(executions_root)).resolve(strict=False)
    repo = _checked_path(Path(repo_root)).resolve(strict=True)
    config = _checked_path(Path(config_path)).resolve(strict=True)
    if executions != previous.parent:
        raise ValueError("recovery executions root must be the predecessor's parent")
    prior = _verify_predecessor(previous, repo_root=repo, config_path=config)
    source = Path(prior["manifest"]["source_root"]).resolve(strict=True)
    if source == previous or source == executions or source.is_relative_to(executions) or executions.is_relative_to(source):
        raise ValueError("recovery source, predecessor, and new execution roots must be disjoint")
    authorization_path = _reservation_path(previous)
    executions.mkdir(parents=True, exist_ok=True)
    _checked_path(executions)
    if list(executions.glob("recovery-authorization-*.json")):
        raise FileExistsError("a recovery authorization reservation already exists in the execution namespace")
    authorization = {
        "predecessor_execution_id": PREDECESSOR_ID,
        "predecessor_execution_sha256": REVIEWED_IMMUTABLE_FILES["execution.json"],
        "protocol": RECOVERY_PROTOCOL,
        "revision": RECOVERY_REVISION,
        "created_utc": _utc_now(),
    }
    body = _canonical(authorization) + b"\n"
    try:
        descriptor = os.open(authorization_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise FileExistsError("recovery authorization already exists; the one-hop slot is consumed") from None
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    authorization_hash = _sha_file(authorization_path)
    original = prior["manifest"]
    baseline = dict(original["artifact_baseline"])
    baseline.update({
        "predecessor_immutable_files": prior["inventory"],
        "predecessor_immutable_bytes": prior["immutable_bytes"],
        "predecessor_status_reserve_bytes": prior["status_bytes"],
        "authorization_reservation": {
            "path": str(authorization_path), "sha256": authorization_hash, "size_bytes": len(body),
        },
    })
    manifest: dict[str, Any] = {
        key: value for key, value in original.items()
        if key not in {"execution_id", "schema_version", "created_utc", "inherited_costs", "artifact_baseline"}
    }
    manifest.update({
        "schema_version": 2,
        "created_utc": authorization["created_utc"],
        "inherited_active_seconds": prior["active_seconds"],
        "inherited_costs": prior["costs"],
        "artifact_baseline": baseline,
        "recovery": {
            "protocol": RECOVERY_PROTOCOL, "revision": RECOVERY_REVISION,
            "one_hop": True, "predecessor_root": str(previous),
            "predecessor_execution_id": PREDECESSOR_ID,
            "predecessor_execution_sha256": REVIEWED_IMMUTABLE_FILES["execution.json"],
            "pinned_files": dict(REVIEWED_IMMUTABLE_FILES),
            "predecessor_inventory": prior["inventory"],
            "predecessor_status_reserve_bytes": prior["status_bytes"],
            "inherited_active_seconds": prior["active_seconds"],
            "inherited_costs": dict(prior["costs"]),
            "accounting_rationale": {
                "cuda_seconds": "1168 feasibility seconds plus the full 31 second settled failed-job upper bound; this replaces the producer-reported 22 second charge rather than refunding 9 seconds",
                "producer_reported_cuda_seconds": 1190,
                "download_bytes": "8066699435 prior bytes plus the completed 18427 byte teacher metadata reservation",
            },
            "authorization_reservation": dict(baseline["authorization_reservation"]),
        },
    })
    from .execution import _sha
    manifest["execution_id"] = _sha(manifest)
    root = executions / manifest["execution_id"]
    root.mkdir()
    try:
        descriptor = os.open(root / "execution.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(manifest) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        (root / "events.jsonl").touch(exist_ok=False)
        (root / "logs").mkdir()
        (root / "stages").mkdir()
    except BaseException:
        raise
    return root


def validate_recovery_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path, config_path: Path,
) -> None:
    """Authenticate the fixed recovery pins, predecessor inventory, and reservation."""
    recovery = manifest.get("recovery")
    if manifest.get("schema_version") != 2 or not isinstance(recovery, Mapping):
        raise ValueError("unsupported recovery manifest schema")
    if (recovery.get("protocol") != RECOVERY_PROTOCOL or recovery.get("revision") != RECOVERY_REVISION
            or recovery.get("one_hop") is not True or recovery.get("predecessor_execution_id") != PREDECESSOR_ID
            or recovery.get("pinned_files") != REVIEWED_IMMUTABLE_FILES):
        raise ValueError("recovery manifest differs from the reviewed one-hop authorization")
    seconds = manifest.get("inherited_active_seconds")
    costs = manifest.get("inherited_costs")
    if type(seconds) is not int or seconds != _EXPECTED_ACTIVE_SECONDS or costs != _EXPECTED_COSTS:
        raise ValueError("recovery inherited clock/cost baseline differs from reviewed predecessor evidence")
    if recovery.get("inherited_active_seconds") != seconds or recovery.get("inherited_costs") != costs:
        raise ValueError("recovery manifest duplicates inconsistent inherited accounting")
    if recovery.get("accounting_rationale") != {
        "cuda_seconds": "1168 feasibility seconds plus the full 31 second settled failed-job upper bound; this replaces the producer-reported 22 second charge rather than refunding 9 seconds",
        "producer_reported_cuda_seconds": 1190,
        "download_bytes": "8066699435 prior bytes plus the completed 18427 byte teacher metadata reservation",
    }:
        raise ValueError("recovery carried-cost rationale differs from reviewed accounting")
    previous = _checked_path(Path(str(recovery.get("predecessor_root")))).resolve(strict=True)
    if previous.name != PREDECESSOR_ID:
        raise ValueError("recovery predecessor path does not name the pinned execution")
    frozen_status_bytes = recovery.get("predecessor_status_reserve_bytes")
    if type(frozen_status_bytes) is not int or frozen_status_bytes < 0:
        raise ValueError("recovery status reserve is invalid")
    verified = _verify_predecessor(
        previous, repo_root=Path(repo_root), config_path=Path(config_path),
        frozen_status_bytes=frozen_status_bytes,
    )
    if recovery.get("predecessor_inventory") != verified["inventory"]:
        raise ValueError("recovery predecessor inventory changed")
    original = verified["manifest"]
    for field in (
        "source_root", "source_run_id", "source_manifest_sha256", "source_prepare_sha256",
        "source_terminal_sha256", "config_path", "config_sha256", "lock_sha256",
        "runtime_identity", "policy",
    ):
        if manifest.get(field) != original.get(field):
            raise ValueError(f"recovery execution {field} differs from its pinned predecessor")
    baseline = manifest.get("artifact_baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("recovery artifact baseline is missing")
    for field in ("source_files", "source_bytes", "feasibility_files"):
        if baseline.get(field) != original.get("artifact_baseline", {}).get(field):
            raise ValueError("recovery original artifact baseline differs from the predecessor")
    for field, value in (
        ("predecessor_immutable_files", verified["inventory"]),
        ("predecessor_immutable_bytes", verified["immutable_bytes"]),
        ("predecessor_status_reserve_bytes", verified["status_bytes"]),
    ):
        if baseline.get(field) != value:
            raise ValueError("recovery artifact baseline differs from immutable predecessor inventory")
    reservation = baseline.get("authorization_reservation")
    if not isinstance(reservation, Mapping) or recovery.get("authorization_reservation") != reservation:
        raise ValueError("recovery authorization reservation context is missing or changed")
    path = _checked_path(Path(str(reservation.get("path")))).resolve(strict=True)
    if path != _reservation_path(previous).resolve(strict=True):
        raise ValueError("recovery authorization reservation path is not the fixed predecessor slot")
    authorization, raw = _load_authorization(path)
    if sorted(path.parent.glob("recovery-authorization-*.json")) != [path]:
        raise ValueError("duplicate recovery authorization reservations exist")
    if (_sha_file(path) != reservation.get("sha256") or len(raw) != reservation.get("size_bytes")
            or authorization.get("predecessor_execution_id") != PREDECESSOR_ID
            or authorization.get("predecessor_execution_sha256") != REVIEWED_IMMUTABLE_FILES["execution.json"]
            or authorization.get("protocol") != RECOVERY_PROTOCOL or authorization.get("revision") != RECOVERY_REVISION
            or authorization.get("created_utc") != manifest.get("created_utc")):
        raise ValueError("recovery authorization reservation changed or binds another protocol")
