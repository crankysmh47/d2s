"""Single-use, evidence-pinned repair authorization for the stopped schema 2 run."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .execution import _canonical, _checked_path, _inventory, _sha, _sha_file


PREDECESSOR_ID = "b0c082cba66a37f6f49bacc4bee182102b926ac5ac3af4c0a657823213e1802f"
REPAIR_PROTOCOL = "distillation-production-path-repair-v1"
REPAIR_REVISION = 1
STATUS_FILES = ("status.json", "status.md")
REVIEWED_IMMUTABLE_FILES: dict[str, tuple[str, int]] = {
    "execution.json": ("4f8f0071edea9ec9e60c7ee2b1f92265a7485d2c9973179f67964abe65468715", 13411),
    "events.jsonl": ("31da70086f7b84deebc73d81264d8fd774cc7a4d4f01eccce35d260d78a16868", 3619),
    "terminal.json": ("6ae485ccb972bcd7648a6ab67b2d0da9b43901ff6fc4727678f377f6c086fd3d", 612),
    "stages/calibrate.json": ("4c3fb2996c9e91b26b1187efa38ad9ca39846ed8cc8f8965fa30509760d993bd", 4136),
    "stages/calibrate-reservations.jsonl": ("7c4b29a8329b4e695df00f0fa560be3b88efea41c023c8af32bfc6393e51089f", 3474),
    "stages/calibrate-attempts.jsonl": ("4adda447dee77b56c35757e50a2a215bbf677c5d9b642eb419cf963359bd8b0d", 136),
    "calibration/raw/256-768-student.jsonl.attempts.jsonl": ("e6f7b50678c363ef6da873c6827ab7d7eed24a0535adae3da30c808138b87378", 2980),
    "calibration/model-inventory/student.json": ("b771e48e0eca0812c4a2271fe565cdecd2970bc5b2f0875bbfc27df7bd015566", 2790),
    "calibration/model-inventory/teacher.json": ("5bde59ee552551a8c81266a79019e62081accb99c614254ea508b0bd17dae74a", 3072),
    "launches/216e170f91e1458aa74d679e457a159e.json": ("bfde45b207f6835d9df0e618f56b3bcc408e2de27c3397c0b817c09870500dcd", 239),
    "logs/216e170f91e1458aa74d679e457a159e.stderr.log": ("e9c55510876a5d8de68e610d43696cb6bc7e3383f3806dcb8338d8370a148a91", 3160),
    "logs/216e170f91e1458aa74d679e457a159e.stdout.log": ("4adcde87f24effd6ee8798c324a4ef9cee2aac5138fc02398a57c2bd618d0c56", 4641),
    "logs/216e170f91e1458aa74d679e457a159e.supervisor.log": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0),
}
_EXPECTED_STATUS_BYTES = 4744
_EXPECTED_ACTIVE_SECONDS = 60
_EXPECTED_COSTS = {"cuda_seconds": 1228, "download_bytes": 8066717862}
_EXPECTED_PREVIOUS_COSTS = {"cuda_seconds": 1199, "download_bytes": 8066717862}
_EXPECTED_ATTEMPT_ID = "216e170f91e1458aa74d679e457a159e"
_EXPECTED_AUTHENTICATED_BASELINE_BYTES = 2047868


def _matches_exact_costs(value: Any, expected: Mapping[str, int]) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(expected)
        and all(type(value[name]) is int and value[name] >= 0 and value[name] == amount
                for name, amount in expected.items())
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _expected_inventory(root: Path) -> tuple[list[dict[str, Any]], int, int]:
    rows, _ = _inventory(root)
    statuses: dict[str, int] = {}
    retained: list[dict[str, Any]] = []
    for row in rows:
        if row["path"] in STATUS_FILES:
            statuses[row["path"]] = row["size_bytes"]
        else:
            retained.append(row)
    if set(statuses) != set(STATUS_FILES) or sum(statuses.values()) != _EXPECTED_STATUS_BYTES:
        raise ValueError("schema 2 predecessor status reserve differs from reviewed evidence")
    expected = {
        path: {"sha256": digest, "size_bytes": size}
        for path, (digest, size) in REVIEWED_IMMUTABLE_FILES.items()
    }
    actual = {row["path"]: {"sha256": row["sha256"], "size_bytes": row["size_bytes"]} for row in retained}
    if actual != expected:
        raise ValueError("schema 2 predecessor immutable inventory differs from the reviewed 13-file pin set")
    return retained, sum(row["size_bytes"] for row in retained), sum(statuses.values())


def _reservation_path(previous: Path) -> Path:
    return previous.parent / f"repair-authorization-{PREDECESSOR_ID}.json"


def _load_authorization(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("repair authorization reservation is malformed or oversized")
    value = json.loads(raw)
    if raw != _canonical(value) + b"\n":
        raise ValueError("repair authorization reservation is not canonical")
    return value, raw


def _verify_predecessor(previous: Path, *, repo_root: Path, config_path: Path) -> dict[str, Any]:
    from .execution import process_creation_utc, read_events, validate_execution, validate_job_intervals

    if previous.name != PREDECESSOR_ID:
        raise ValueError("only the pinned stopped schema 2 execution is eligible for repair")
    manifest = validate_execution(previous, repo_root=repo_root, config_path=config_path)
    if manifest.get("schema_version") != 2 or manifest.get("execution_id") != PREDECESSOR_ID:
        raise ValueError("repair predecessor must be the pinned schema 2 execution")
    if manifest.get("inherited_active_seconds") != 31 or manifest.get("inherited_costs") != _EXPECTED_PREVIOUS_COSTS:
        raise ValueError("repair predecessor inherited accounting differs from authenticated schema 2 evidence")

    inventory, immutable_bytes, status_bytes = _expected_inventory(previous)
    if immutable_bytes != 42270 or status_bytes != _EXPECTED_STATUS_BYTES:
        raise ValueError("repair predecessor immutable/status byte totals differ from reviewed evidence")
    events = read_events(previous)
    jobs = validate_job_intervals(previous)
    if len(jobs) != 1:
        raise ValueError("repair predecessor must contain one settled calibration interval")
    job = next(iter(jobs.values()))
    if (job["settlement"] is None or job["prior_active_seconds"] != 31
            or job["settled_active_seconds"] != _EXPECTED_ACTIVE_SECONDS):
        raise ValueError("repair predecessor interval does not settle at cumulative 60 seconds")

    terminal = json.loads((previous / "terminal.json").read_text(encoding="utf-8"))
    if (terminal.get("execution_id") != PREDECESSOR_ID or terminal.get("attempt_id") != _EXPECTED_ATTEMPT_ID
            or terminal.get("state") != "stopped_environment" or terminal.get("runtime_used_seconds") != 60
            or terminal.get("runtime_remaining_seconds") != 172740
            or terminal.get("reason") != "bounded worker retries were exhausted; partial raw rows are preserved"):
        raise ValueError("repair predecessor terminal stop differs from the reviewed exhausted-worker result")
    owner_records = []
    for payload in (event["payload"] for event in events):
        if payload.get("type") in {"job_started", "job_settled"}:
            owner_records.extend((
                (payload.get("supervisor_pid"), payload.get("supervisor_created_utc")),
                (payload.get("stage_pid"), payload.get("stage_created_utc")),
            ))
    owner_records.append((terminal.get("supervisor_pid"), terminal.get("supervisor_created_utc")))
    owner_records.append((terminal.get("stage_pid"), terminal.get("stage_created_utc")))
    if any(type(pid) is int and process_creation_utc(pid) == created for pid, created in owner_records):
        raise ValueError("repair predecessor still has a live process owner")
    for lock_name in ("current-job.json", "supervisor-owner.json", "stage-owner.json"):
        if (previous / lock_name).exists():
            raise ValueError(f"repair predecessor has a live-owner/lock record: {lock_name}")

    stage = json.loads((previous / "stages/calibrate.json").read_text(encoding="utf-8"))
    if (stage.get("status") != "stopped_environment" or stage.get("execution_id") != PREDECESSOR_ID
            or stage.get("reason") != terminal.get("reason") or len(stage.get("attempts", [])) != 1):
        raise ValueError("repair predecessor calibration stage is not the reviewed stopped result")
    attempt = stage["attempts"][0]
    if attempt.get("status") != "stopped_environment" or attempt.get("raw_rows") != []:
        raise ValueError("repair predecessor contains answers or a different exhausted-worker attempt")
    raw_attempts = [
        json.loads(line)
        for line in (previous / "calibration/raw/256-768-student.jsonl.attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    load_failures = [row for row in raw_attempts if row.get("status") == "worker_load_failed"]
    if (len(load_failures) != 3 or any(
        row.get("role") != "student" or row.get("error_type") != "RuntimeError"
        or row.get("error_message") != "isolated model worker failed: ValueError: model device placement is unavailable for offload validation"
        for row in load_failures
    )):
        raise ValueError("repair predecessor does not retain exactly three reviewed placement failures")
    reservation_rows = [
        json.loads(line)
        for line in (previous / "stages/calibrate-reservations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if any(row.get("event_type") in {"download_reservation", "download_complete"} for row in reservation_rows):
        raise ValueError("repair predecessor records an unreviewed download during the failed repair attempt")
    # Preserve this known producer-accounting defect as evidence. It is not used as a refund.
    producer_ledger = stage.get("resource_ledger", {})
    if producer_ledger.get("cuda_seconds") != 1189 or producer_ledger.get("download_bytes") != 8066699435:
        raise ValueError("repair predecessor producer ledger no longer matches its pinned discrepancy")
    signature = stage.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("repair predecessor lacks the original producer signature")
    signature_body = {key: value for key, value in signature.items() if key != "stage_signature"}
    if signature.get("stage_signature") != sha256(_canonical(signature_body)).hexdigest():
        raise ValueError("repair predecessor stored producer signature is not canonically authenticated")
    if (signature.get("originating_git_commit") != "d6bb515e735626dd7f897b5c8392afd65aea0b92"
            or signature.get("stage") != "calibrate"
            or signature.get("execution_policy_sha256") != sha256(_canonical(manifest["policy"])).hexdigest()
            or signature.get("predecessors") != {
                "prepare_manifest_sha256": manifest["source_manifest_sha256"],
                "prepare_stage_sha256": manifest["source_prepare_sha256"],
            }):
        raise ValueError("repair predecessor stored producer signature has changed origin or inputs")

    reservation = manifest.get("artifact_baseline", {}).get("authorization_reservation")
    if not isinstance(reservation, Mapping):
        raise ValueError("schema 2 predecessor lacks its authenticated first authorization")
    if reservation.get("size_bytes") != 303:
        raise ValueError("schema 2 predecessor first authorization size differs from pinned evidence")
    baseline_bytes = (
        int(manifest["artifact_baseline"]["source_bytes"])
        + sum(row["size_bytes"] for row in manifest["artifact_baseline"]["feasibility_files"])
        + int(manifest["artifact_baseline"]["predecessor_immutable_bytes"])
        + int(manifest["artifact_baseline"]["predecessor_status_reserve_bytes"])
        + int(reservation["size_bytes"])
    )
    if baseline_bytes != _EXPECTED_AUTHENTICATED_BASELINE_BYTES:
        raise ValueError("schema 2 predecessor authenticated artifact baseline differs from reviewed 2047868 bytes")
    return {
        "manifest": manifest,
        "events": events,
        "inventory": inventory,
        "immutable_bytes": immutable_bytes,
        "status_bytes": status_bytes,
        "active_seconds": _EXPECTED_ACTIVE_SECONDS,
        "costs": dict(_EXPECTED_COSTS),
    }


def register_repair(
    previous_execution_root: Path,
    executions_root: Path,
    *,
    repo_root: Path,
    config_path: Path,
) -> Path:
    """Consume the one reviewed repair slot and create its immutable schema 3 execution."""
    previous = _checked_path(Path(previous_execution_root)).resolve(strict=True)
    executions = _checked_path(Path(executions_root)).resolve(strict=False)
    repo = _checked_path(Path(repo_root)).resolve(strict=True)
    config = _checked_path(Path(config_path)).resolve(strict=True)
    if executions != previous.parent:
        raise ValueError("repair executions root must be the predecessor's parent")
    prior = _verify_predecessor(previous, repo_root=repo, config_path=config)
    source = Path(prior["manifest"]["source_root"]).resolve(strict=True)
    if source == previous or source == executions or source.is_relative_to(executions) or executions.is_relative_to(source):
        raise ValueError("repair source, predecessor, and execution roots must be disjoint")

    slot = _reservation_path(previous)
    executions.mkdir(parents=True, exist_ok=True)
    _checked_path(executions)
    if list(executions.glob("repair-authorization-*.json")):
        raise FileExistsError("a repair authorization reservation already exists; the single-use slot is consumed")
    created_utc = _utc_now()
    authorization = {
        "predecessor_execution_id": PREDECESSOR_ID,
        "predecessor_execution_sha256": REVIEWED_IMMUTABLE_FILES["execution.json"][0],
        "protocol": REPAIR_PROTOCOL,
        "revision": REPAIR_REVISION,
        "created_utc": created_utc,
    }
    body = _canonical(authorization) + b"\n"
    try:
        descriptor = os.open(slot, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise FileExistsError("repair authorization already exists; the single-use slot is consumed") from None
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())

    original = prior["manifest"]
    baseline = dict(original["artifact_baseline"])
    repair_reservation = {"path": str(slot), "sha256": _sha_file(slot), "size_bytes": len(body)}
    baseline.update({
        "repair_predecessor_immutable_files": prior["inventory"],
        "repair_predecessor_immutable_bytes": prior["immutable_bytes"],
        "repair_predecessor_status_reserve_bytes": prior["status_bytes"],
        "repair_authorization_reservation": repair_reservation,
    })
    manifest = {
        key: value for key, value in original.items()
        if key not in {"execution_id", "schema_version", "created_utc", "inherited_active_seconds", "inherited_costs", "artifact_baseline"}
    }
    manifest.update({
        "schema_version": 3,
        "created_utc": created_utc,
        "inherited_active_seconds": prior["active_seconds"],
        "inherited_costs": prior["costs"],
        "artifact_baseline": baseline,
        "repair": {
            "protocol": REPAIR_PROTOCOL,
            "revision": REPAIR_REVISION,
            "predecessor_root": str(previous),
            "predecessor_execution_id": PREDECESSOR_ID,
            "predecessor_execution_sha256": REVIEWED_IMMUTABLE_FILES["execution.json"][0],
            "pinned_files": {path: {"sha256": digest, "size_bytes": size} for path, (digest, size) in REVIEWED_IMMUTABLE_FILES.items()},
            "predecessor_inventory": prior["inventory"],
            "predecessor_producer_signature": json.loads(
                (previous / "stages/calibrate.json").read_text(encoding="utf-8")
            )["signature"],
            "predecessor_immutable_bytes": prior["immutable_bytes"],
            "predecessor_status_reserve_bytes": prior["status_bytes"],
            "inherited_active_seconds": prior["active_seconds"],
            "inherited_costs": dict(prior["costs"]),
            "accounting_rationale": {
                "cuda_seconds": "1199 authenticated schema 2 seconds plus the full 29 second settled repair interval; producer-reported 1189 remains pinned diagnostic evidence and is never refunded or substituted",
                "download_bytes": "Retain the authenticated schema 2 total of 8066717862 bytes; the predecessor producer ledger value 8066699435 is a known pinned discrepancy, not a basis for refund or re-accounting",
            },
            "authorization_reservation": repair_reservation,
        },
    })
    manifest["execution_id"] = _sha(manifest)
    root = executions / manifest["execution_id"]
    root.mkdir()
    descriptor = os.open(root / "execution.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(manifest) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    (root / "events.jsonl").touch(exist_ok=False)
    (root / "logs").mkdir()
    (root / "stages").mkdir()
    return root


def validate_repair_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path, config_path: Path,
) -> None:
    """Revalidate the pinned schema 2 predecessor and its one-use repair reservation."""
    repair = manifest.get("repair")
    if (manifest.get("schema_version") != 3 or not isinstance(repair, Mapping)
            or repair.get("protocol") != REPAIR_PROTOCOL or repair.get("revision") != REPAIR_REVISION
            or repair.get("predecessor_execution_id") != PREDECESSOR_ID):
        raise ValueError("unsupported or unreviewed repair manifest schema")
    expected_pins = {path: {"sha256": digest, "size_bytes": size} for path, (digest, size) in REVIEWED_IMMUTABLE_FILES.items()}
    if repair.get("pinned_files") != expected_pins:
        raise ValueError("repair manifest pin set differs from the reviewed 13-file predecessor")
    previous = _checked_path(Path(str(repair.get("predecessor_root")))).resolve(strict=True)
    if previous.name != PREDECESSOR_ID:
        raise ValueError("repair predecessor path does not name the pinned schema 2 execution")
    verified = _verify_predecessor(previous, repo_root=Path(repo_root), config_path=Path(config_path))
    if (repair.get("predecessor_inventory") != verified["inventory"]
            or repair.get("predecessor_producer_signature") != json.loads(
                (previous / "stages/calibrate.json").read_text(encoding="utf-8")
            ).get("signature")
            or repair.get("predecessor_immutable_bytes") != verified["immutable_bytes"]
            or repair.get("predecessor_status_reserve_bytes") != verified["status_bytes"]
            or type(repair.get("inherited_active_seconds")) is not int
            or repair.get("inherited_active_seconds") != _EXPECTED_ACTIVE_SECONDS
            or type(manifest.get("inherited_active_seconds")) is not int
            or manifest.get("inherited_active_seconds") != _EXPECTED_ACTIVE_SECONDS
            or not _matches_exact_costs(repair.get("inherited_costs"), _EXPECTED_COSTS)
            or not _matches_exact_costs(manifest.get("inherited_costs"), _EXPECTED_COSTS)):
        raise ValueError("repair predecessor inventory or carried accounting changed")
    if repair.get("accounting_rationale") != {
        "cuda_seconds": "1199 authenticated schema 2 seconds plus the full 29 second settled repair interval; producer-reported 1189 remains pinned diagnostic evidence and is never refunded or substituted",
        "download_bytes": "Retain the authenticated schema 2 total of 8066717862 bytes; the predecessor producer ledger value 8066699435 is a known pinned discrepancy, not a basis for refund or re-accounting",
    }:
        raise ValueError("repair cumulative cost rationale changed")
    original = verified["manifest"]
    for field in (
        "source_root", "source_run_id", "source_manifest_sha256", "source_prepare_sha256",
        "source_terminal_sha256", "config_path", "config_sha256", "lock_sha256", "runtime_identity", "policy", "recovery",
    ):
        if manifest.get(field) != original.get(field):
            raise ValueError(f"repair execution {field} differs from the pinned schema 2 predecessor")
    baseline = manifest.get("artifact_baseline")
    old_baseline = original.get("artifact_baseline", {})
    if not isinstance(baseline, Mapping):
        raise ValueError("repair artifact baseline is missing")
    for field, value in old_baseline.items():
        if baseline.get(field) != value:
            raise ValueError("repair original schema 2 artifact baseline was changed")
    for field, value in (
        ("repair_predecessor_immutable_files", verified["inventory"]),
        ("repair_predecessor_immutable_bytes", verified["immutable_bytes"]),
        ("repair_predecessor_status_reserve_bytes", verified["status_bytes"]),
    ):
        if baseline.get(field) != value:
            raise ValueError("repair artifact baseline differs from pinned predecessor evidence")
    reservation = baseline.get("repair_authorization_reservation")
    if not isinstance(reservation, Mapping) or repair.get("authorization_reservation") != reservation:
        raise ValueError("repair authorization reservation context is missing or changed")
    path = _checked_path(Path(str(reservation.get("path")))).resolve(strict=True)
    if path != _reservation_path(previous).resolve(strict=True):
        raise ValueError("repair authorization path is not the fixed single-use slot")
    authorization, raw = _load_authorization(path)
    if sorted(path.parent.glob("repair-authorization-*.json")) != [path]:
        raise ValueError("duplicate repair authorization reservations exist")
    if (_sha_file(path) != reservation.get("sha256") or len(raw) != reservation.get("size_bytes")
            or authorization.get("predecessor_execution_id") != PREDECESSOR_ID
            or authorization.get("predecessor_execution_sha256") != REVIEWED_IMMUTABLE_FILES["execution.json"][0]
            or authorization.get("protocol") != REPAIR_PROTOCOL or authorization.get("revision") != REPAIR_REVISION
            or authorization.get("created_utc") != manifest.get("created_utc")):
        raise ValueError("repair authorization reservation changed or binds another protocol")
