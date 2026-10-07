"""Single-use tokenizer-contract recovery authorization for the frozen Task 2 stop."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .execution import _canonical, _checked_path, _inventory, _sha, _sha_file


PREDECESSOR_ID = "3116768bd90444328e1ec0e4f8b518a18c238ed74e643d24ae0a69affd0683eb"
TOKENIZER_RECOVERY_PROTOCOL = "distillation-tokenizer-contract-recovery-v1"
TOKENIZER_RECOVERY_REVISION = 1
STATUS_FILES = ("status.json", "status.md")
REVIEWED_IMMUTABLE_FILES: dict[str, tuple[str, int]] = {
    "calibration/model-inventory/student.json": ("b771e48e0eca0812c4a2271fe565cdecd2970bc5b2f0875bbfc27df7bd015566", 2790),
    "calibration/model-inventory/teacher.json": ("5bde59ee552551a8c81266a79019e62081accb99c614254ea508b0bd17dae74a", 3072),
    "calibration/raw/256-768-student.jsonl.attempts.jsonl": ("2f9aed0ab7e108b10d2faae2391b7db7ea926b04bfd562597eb934126bb8c6fe", 4562),
    "events.jsonl": ("c1f3c1b92573c16179de21ac9f460c0d07f463bc02f5bd99c5244d4223d3c64e", 3616),
    "execution.json": ("475d289ad8e4d8bc6618e46d6fe55df138f4ec394f643de7d45291cf310eae06", 22960),
    "launches/41c657fa3727456e812988a0cf54d4c9.json": ("d0e2bb2023058441104de4d3834d785004ca928d522afcf293bb524bcc6a4ebe", 238),
    "logs/41c657fa3727456e812988a0cf54d4c9.stderr.log": ("7816a030fdc56592b773294910ab32b262a5e7590a6b579049004de5b7268b5f", 3024),
    "logs/41c657fa3727456e812988a0cf54d4c9.stdout.log": ("6e475d0be935d9336a375e4abe49d133f7eba0d2086d85e38e59b9efc7ef46d3", 5054),
    "logs/41c657fa3727456e812988a0cf54d4c9.supervisor.log": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0),
    "stages/calibrate-attempts.jsonl": ("ccfddf0501d90524b0bef7bb0135cff337a20739ae4d2d4e59353a2ce0a30209", 136),
    "stages/calibrate-reservations.jsonl": ("eb30c4b0012170df1b0968d1d834c6bdb6fdba4ecffd31ccb50fe7ba273ac510", 3473),
    "stages/calibrate.json": ("2d5bcfdd5a2b3d2ec2dc15ec5aedf6a7ba438ed0f236c824941b765d0a00d6e4", 4138),
    "terminal.json": ("33cd54d85fb2d115257db4b54dba9826829e857a2c5ba34555dae6bfb06c9c2a", 611),
}
EXPECTED_PREDECESSOR_EXECUTION_SHA256 = "475d289ad8e4d8bc6618e46d6fe55df138f4ec394f643de7d45291cf310eae06"
EXPECTED_ORIGINATING_COMMIT = "da8a15819f92f6b8c28673dc89473019f0f7fe31"
EXPECTED_ACTIVE_SECONDS = 91
EXPECTED_COSTS = {"cuda_seconds": 1250, "download_bytes": 8066717862}
EXPECTED_STATUS_SIZES = {"status.json": 2776, "status.md": 1960}
EXPECTED_IMMUTABLE_BYTES = 53674
EXPECTED_STATUS_BYTES = 4736
EXPECTED_ATTEMPT_ID = "41c657fa3727456e812988a0cf54d4c9"
EXPECTED_STOP_REASON = "bounded worker retries were exhausted; partial raw rows are preserved"
_TOP_LEVEL_EXCLUDED = {
    "execution_id", "schema_version", "created_utc", "inherited_active_seconds",
    "inherited_costs", "artifact_baseline", "tokenizer_recovery",
}
_CONTEXT_KEYS = {
    "protocol", "revision", "predecessor_root", "predecessor_execution_id",
    "predecessor_execution_sha256", "pinned_files", "predecessor_inventory",
    "predecessor_status_reserve_bytes", "predecessor_producer_signature",
    "inherited_active_seconds", "inherited_costs", "accounting_rationale",
    "authorization_reservation",
}
_BASELINE_ADDITIONS = {
    "tokenizer_predecessor_immutable_files", "tokenizer_predecessor_immutable_bytes",
    "tokenizer_predecessor_status_reserve_bytes", "tokenizer_authorization_reservation",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _same_exact(left: Any, right: Any) -> bool:
    """Compare JSON-shaped values without Python's bool/int/float equivalences."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_exact(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_exact(a, b) for a, b in zip(left, right))
    return left == right


def _pinned_files() -> dict[str, dict[str, Any]]:
    return {
        path: {"sha256": digest, "size_bytes": size}
        for path, (digest, size) in REVIEWED_IMMUTABLE_FILES.items()
    }


def _status_sizes(previous: Path) -> tuple[dict[str, int], int]:
    sizes = {name: (previous / name).stat().st_size for name in STATUS_FILES if (previous / name).is_file()}
    if sizes != EXPECTED_STATUS_SIZES:
        raise ValueError("tokenizer predecessor status reserve differs from reviewed byte sizes")
    return sizes, sum(sizes.values())


def _inventory_without_status(previous: Path) -> tuple[list[dict[str, Any]], int, int]:
    rows, _total = _inventory(previous)
    observed: dict[str, tuple[str, int]] = {}
    retained: list[dict[str, Any]] = []
    for row in rows:
        path = row["path"]
        if path in STATUS_FILES:
            observed[path] = (row["sha256"], row["size_bytes"])
        else:
            retained.append(row)
    _status, status_bytes = _status_sizes(previous)
    expected = {
        path: (digest, size) for path, (digest, size) in REVIEWED_IMMUTABLE_FILES.items()
    }
    actual = {row["path"]: (row["sha256"], row["size_bytes"]) for row in retained}
    if actual != expected or observed != {
        "status.json": (observed.get("status.json", ("", -1))[0], 2776),
        "status.md": (observed.get("status.md", ("", -1))[0], 1960),
    }:
        raise ValueError("tokenizer predecessor immutable inventory differs from reviewed pins")
    immutable_bytes = sum(row["size_bytes"] for row in retained)
    if immutable_bytes != EXPECTED_IMMUTABLE_BYTES or status_bytes != EXPECTED_STATUS_BYTES:
        raise ValueError("tokenizer predecessor artifact byte reserve differs from reviewed evidence")
    return retained, immutable_bytes, status_bytes


def _json_lines(path: Path) -> list[dict[str, Any]]:
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError(f"tokenizer predecessor journal has a truncated tail: {path.name}")
    try:
        rows = [json.loads(line) for line in content.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"tokenizer predecessor journal is malformed: {path.name}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"tokenizer predecessor journal has a non-object row: {path.name}")
    return rows


def _check_no_live_owner(previous: Path, terminal: Mapping[str, Any], events: list[dict[str, Any]]) -> None:
    from .execution import process_creation_utc

    identities: list[tuple[Any, Any]] = []
    for key, created_key in (("supervisor_pid", "supervisor_created_utc"), ("stage_pid", "stage_created_utc")):
        if terminal.get(key) is not None:
            identities.append((terminal.get(key), terminal.get(created_key)))
    for event in events:
        payload = event["payload"]
        if payload.get("type") in {"job_started", "job_settled"}:
            identities.extend((payload.get(pid_key), payload.get(created_key)) for pid_key, created_key in (
                ("supervisor_pid", "supervisor_created_utc"), ("stage_pid", "stage_created_utc"),
            ))
    if any(type(pid) is int and process_creation_utc(pid) == created for pid, created in identities):
        raise ValueError("tokenizer predecessor still has a live process owner")
    current_job = previous / "current-job.json"
    if current_job.exists():
        try:
            payload = json.loads(current_job.read_text(encoding="utf-8")).get("payload", {})
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("tokenizer predecessor retains malformed current-job owner evidence") from exc
        if type(payload.get("stage_pid")) is int and process_creation_utc(payload["stage_pid"]) == payload.get("stage_created_utc"):
            raise ValueError("tokenizer predecessor still has a live current-job owner")
        raise ValueError("tokenizer predecessor retains current-job ownership evidence")


def _verify_predecessor(previous: Path, *, repo_root: Path, config_path: Path) -> dict[str, Any]:
    from .execution import read_events, validate_execution, validate_job_intervals

    if previous.name != PREDECESSOR_ID:
        raise ValueError("only the pinned schema 3 tokenizer predecessor is eligible")
    if _sha_file(previous / "execution.json") != EXPECTED_PREDECESSOR_EXECUTION_SHA256:
        raise ValueError("tokenizer predecessor schema 3 execution pin changed")
    manifest = validate_execution(previous, repo_root=repo_root, config_path=config_path)
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 3 or manifest.get("execution_id") != PREDECESSOR_ID:
        raise ValueError("tokenizer recovery requires the exact schema 3 predecessor")
    inventory, immutable_bytes, status_bytes = _inventory_without_status(previous)
    events = read_events(previous)
    jobs = validate_job_intervals(previous)
    if len(jobs) != 1:
        raise ValueError("tokenizer predecessor must have exactly one settled job interval")
    job = next(iter(jobs.values()))
    if (
        job.get("start", {}).get("attempt_id") != EXPECTED_ATTEMPT_ID
        or job.get("prior_active_seconds") != 60
        or job.get("settled_active_seconds") != EXPECTED_ACTIVE_SECONDS
        or job.get("settlement") is None
    ):
        raise ValueError("tokenizer predecessor interval does not settle at the reviewed 91 seconds")

    terminal = json.loads((previous / "terminal.json").read_text(encoding="utf-8"))
    if (
        terminal.get("execution_id") != PREDECESSOR_ID
        or terminal.get("attempt_id") != EXPECTED_ATTEMPT_ID
        or terminal.get("state") != "stopped_environment"
        or terminal.get("reason") != EXPECTED_STOP_REASON
        or terminal.get("runtime_used_seconds") != EXPECTED_ACTIVE_SECONDS
        or terminal.get("runtime_remaining_seconds") != 172709
    ):
        raise ValueError("tokenizer predecessor terminal is not the reviewed retry-exhaustion stop")

    stage = json.loads((previous / "stages" / "calibrate.json").read_text(encoding="utf-8"))
    stage_ledger = stage.get("resource_ledger", {})
    initial = json.loads((previous / "stages" / "calibrate-reservations.jsonl").read_text(encoding="utf-8").splitlines()[0])
    producer_signature = stage.get("signature")
    if (
        stage.get("execution_id") != PREDECESSOR_ID
        or stage.get("originating_git_commit") != EXPECTED_ORIGINATING_COMMIT
        or stage.get("status") != "stopped_environment"
        or stage.get("reason") != EXPECTED_STOP_REASON
        or stage_ledger.get("cuda_seconds") != EXPECTED_COSTS["cuda_seconds"]
        or stage_ledger.get("download_bytes") != EXPECTED_COSTS["download_bytes"]
        or stage.get("effective_eos_token_ids") != [151643, 151645]
        or initial.get("event_type") != "reservation"
        or initial.get("initial_cuda_seconds") != 1228
        or initial.get("initial_download_bytes") != EXPECTED_COSTS["download_bytes"]
        or initial.get("download_bytes") != 0
        or not isinstance(producer_signature, Mapping)
        or producer_signature.get("originating_git_commit") != EXPECTED_ORIGINATING_COMMIT
        or producer_signature.get("stage_signature") != _sha({key: value for key, value in producer_signature.items() if key != "stage_signature"})
    ):
        raise ValueError("tokenizer predecessor producer signature or resource ledger differs from pinned evidence")

    attempt_rows = _json_lines(previous / "calibration/raw/256-768-student.jsonl.attempts.jsonl")
    worker_started = [row for row in attempt_rows if row.get("status") == "worker_started"]
    failed = [row for row in attempt_rows if row.get("status") == "failed"]
    if (
        len(worker_started) != 3 or len(failed) != 3
        or any(row.get("role") != "student" or row.get("error_type") != "RuntimeError" for row in failed)
        or any("chat template did not return token IDs" not in row.get("error_message", "") for row in failed)
        or [row.get("attempt_id") for row in worker_started] != [row.get("attempt_id") for row in failed]
    ):
        raise ValueError("tokenizer predecessor does not retain exactly three failed token-ID worker attempts")
    stdout = (previous / "logs" / f"{EXPECTED_ATTEMPT_ID}.stdout.log").read_text(encoding="utf-8")
    if stdout.count('"event":"model_placement"') != 3:
        raise ValueError("tokenizer predecessor stdout lacks the three reviewed placement records")
    raw_dir = previous / "calibration" / "raw"
    if any(path.is_file() for path in raw_dir.glob("*.jsonl") if not path.name.endswith(".attempts.jsonl")):
        raise ValueError("tokenizer predecessor contains calibration answer rows")
    reservation_rows = _json_lines(previous / "stages/calibrate-reservations.jsonl")
    if len(reservation_rows) != 2 or any(row.get("event_type") in {"download_reservation", "download_complete"} for row in reservation_rows):
        raise ValueError("tokenizer predecessor contains a new download reservation or completion")
    _check_no_live_owner(previous, terminal, events)
    return {
        "manifest": manifest, "events": events, "inventory": inventory,
        "immutable_bytes": immutable_bytes, "status_bytes": status_bytes,
        "producer_signature": dict(producer_signature),
    }


def _authorization_path(previous: Path) -> Path:
    return previous.parent / f"tokenizer-authorization-{PREDECESSOR_ID}.json"


def _load_authorization(path: Path, *, predecessor_root: Path) -> tuple[dict[str, Any], bytes]:
    expected_path = _authorization_path(predecessor_root)
    if _checked_path(path).resolve(strict=True) != expected_path.resolve(strict=False):
        raise ValueError("tokenizer authorization reservation path differs from its fixed predecessor slot")
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("tokenizer authorization reservation is malformed or oversized")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("tokenizer authorization reservation is malformed") from exc
    expected_keys = {
        "predecessor_execution_id", "predecessor_execution_sha256", "protocol", "revision", "created_utc",
    }
    if not isinstance(value, dict) or set(value) != expected_keys or raw != _canonical(value) + b"\n":
        raise ValueError("tokenizer authorization reservation is not exact canonical JSON")
    if (
        value.get("predecessor_execution_id") != PREDECESSOR_ID
        or value.get("predecessor_execution_sha256") != EXPECTED_PREDECESSOR_EXECUTION_SHA256
        or value.get("protocol") != TOKENIZER_RECOVERY_PROTOCOL
        or type(value.get("revision")) is not int
        or value.get("revision") != TOKENIZER_RECOVERY_REVISION
        or not isinstance(value.get("created_utc"), str)
    ):
        raise ValueError("tokenizer authorization reservation differs from reviewed pins")
    try:
        parsed = datetime.fromisoformat(value["created_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("tokenizer authorization created time is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("tokenizer authorization created time must be UTC")
    return value, raw


def register_tokenizer_recovery(
    previous_execution_root: Path,
    executions_root: Path,
    *,
    repo_root: Path,
    config_path: Path,
) -> Path:
    """Create the one schema 4 child authorized by the exact tokenizer stop."""
    previous = _checked_path(Path(previous_execution_root)).resolve(strict=True)
    executions = _checked_path(Path(executions_root)).resolve(strict=False)
    repo = _checked_path(Path(repo_root)).resolve(strict=True)
    config = _checked_path(Path(config_path)).resolve(strict=True)
    if executions != previous.parent:
        raise ValueError("tokenizer executions root must be the predecessor's parent")
    prior = _verify_predecessor(previous, repo_root=repo, config_path=config)
    source = Path(prior["manifest"]["source_root"]).resolve(strict=True)
    if source == previous or source == executions or source.is_relative_to(executions) or executions.is_relative_to(source):
        raise ValueError("tokenizer source, predecessor, and new execution roots must be disjoint")
    existing_slots = list(executions.glob("tokenizer-authorization-*.json"))
    if existing_slots:
        raise FileExistsError("tokenizer authorization slot already exists; it remains consumed")
    executions.mkdir(parents=True, exist_ok=True)
    _checked_path(executions)
    authorization_path = _authorization_path(previous)
    authorization = {
        "predecessor_execution_id": PREDECESSOR_ID,
        "predecessor_execution_sha256": EXPECTED_PREDECESSOR_EXECUTION_SHA256,
        "protocol": TOKENIZER_RECOVERY_PROTOCOL,
        "revision": TOKENIZER_RECOVERY_REVISION,
        "created_utc": _utc_now(),
    }
    body = _canonical(authorization) + b"\n"
    try:
        descriptor = os.open(authorization_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise FileExistsError("tokenizer authorization already exists; the one-use slot is consumed") from None
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    authorization_record = {
        "path": str(authorization_path), "sha256": sha256(body).hexdigest(), "size_bytes": len(body),
    }
    original = prior["manifest"]
    original_baseline = dict(original["artifact_baseline"])
    baseline = dict(original_baseline)
    baseline.update({
        "tokenizer_predecessor_immutable_files": prior["inventory"],
        "tokenizer_predecessor_immutable_bytes": prior["immutable_bytes"],
        "tokenizer_predecessor_status_reserve_bytes": prior["status_bytes"],
        "tokenizer_authorization_reservation": authorization_record,
    })
    accounting_rationale = {
        "active_seconds": "60 authenticated predecessor seconds plus the full 31 second settled interval equals 91; no time is refunded",
        "cuda_seconds": "Carry the authenticated producer ledger value 1250 without adding the already settled interval again",
        "download_bytes": "Carry the authenticated initial reservation and producer ledger value 8066717862 without adjustment",
    }
    context = {
        "protocol": TOKENIZER_RECOVERY_PROTOCOL,
        "revision": TOKENIZER_RECOVERY_REVISION,
        "predecessor_root": str(previous),
        "predecessor_execution_id": PREDECESSOR_ID,
        "predecessor_execution_sha256": EXPECTED_PREDECESSOR_EXECUTION_SHA256,
        "pinned_files": _pinned_files(),
        "predecessor_inventory": prior["inventory"],
        "predecessor_status_reserve_bytes": prior["status_bytes"],
        "predecessor_producer_signature": prior["producer_signature"],
        "inherited_active_seconds": EXPECTED_ACTIVE_SECONDS,
        "inherited_costs": dict(EXPECTED_COSTS),
        "accounting_rationale": accounting_rationale,
        "authorization_reservation": authorization_record,
    }
    manifest: dict[str, Any] = {
        key: value for key, value in original.items() if key not in _TOP_LEVEL_EXCLUDED
    }
    manifest.update({
        "schema_version": 4,
        "created_utc": authorization["created_utc"],
        "inherited_active_seconds": EXPECTED_ACTIVE_SECONDS,
        "inherited_costs": dict(EXPECTED_COSTS),
        "artifact_baseline": baseline,
        "tokenizer_recovery": context,
    })
    manifest["execution_id"] = _sha(manifest)
    root = executions / manifest["execution_id"]
    try:
        root.mkdir()
    except FileExistsError:
        raise FileExistsError(f"tokenizer recovery execution already registered: {manifest['execution_id']}") from None
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
        # Preserve the authorization and partial directory; the slot is one-use.
        raise
    return root


def validate_tokenizer_recovery_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path, config_path: Path,
) -> None:
    """Authenticate the closed schema 4 projection and its single-use reservation."""
    context = manifest.get("tokenizer_recovery")
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 4 or not isinstance(context, Mapping):
        raise ValueError("unsupported tokenizer recovery manifest schema")
    if set(context) != _CONTEXT_KEYS:
        raise ValueError("tokenizer recovery context has unknown or missing fields")
    previous_value = context.get("predecessor_root")
    if not isinstance(previous_value, str):
        raise ValueError("tokenizer recovery predecessor root is invalid")
    previous = _checked_path(Path(previous_value)).resolve(strict=True)
    verified = _verify_predecessor(previous, repo_root=Path(repo_root), config_path=Path(config_path))
    original = verified["manifest"]
    if (
        context.get("protocol") != TOKENIZER_RECOVERY_PROTOCOL
        or type(context.get("revision")) is not int
        or context.get("revision") != TOKENIZER_RECOVERY_REVISION
        or context.get("predecessor_execution_id") != PREDECESSOR_ID
        or context.get("predecessor_execution_sha256") != EXPECTED_PREDECESSOR_EXECUTION_SHA256
        or _sha_file(previous / "execution.json") != context.get("predecessor_execution_sha256")
        or not _same_exact(context.get("pinned_files"), _pinned_files())
        or not _same_exact(context.get("predecessor_inventory"), verified["inventory"])
        or type(context.get("predecessor_status_reserve_bytes")) is not int
        or context.get("predecessor_status_reserve_bytes") != EXPECTED_STATUS_BYTES
        or not _same_exact(context.get("predecessor_producer_signature"), verified["producer_signature"])
    ):
        raise ValueError("tokenizer recovery context differs from the exact predecessor authorization")

    seconds = manifest.get("inherited_active_seconds")
    costs = manifest.get("inherited_costs")
    if type(seconds) is not int or seconds != EXPECTED_ACTIVE_SECONDS:
        raise ValueError("tokenizer recovery inherited clock differs from reviewed evidence")
    if (
        not isinstance(costs, Mapping) or set(costs) != {"cuda_seconds", "download_bytes"}
        or any(type(value) is not int for value in costs.values())
        or dict(costs) != EXPECTED_COSTS
    ):
        raise ValueError("tokenizer recovery inherited costs differ from reviewed evidence")
    context_seconds = context.get("inherited_active_seconds")
    context_costs = context.get("inherited_costs")
    if type(context_seconds) is not int or context_seconds != seconds:
        raise ValueError("tokenizer recovery context clock differs from top-level clock")
    if (
        not isinstance(context_costs, Mapping) or set(context_costs) != {"cuda_seconds", "download_bytes"}
        or any(type(value) is not int for value in context_costs.values())
        or dict(context_costs) != dict(costs)
    ):
        raise ValueError("tokenizer recovery context costs differ from top-level costs")
    expected_rationale = {
        "active_seconds": "60 authenticated predecessor seconds plus the full 31 second settled interval equals 91; no time is refunded",
        "cuda_seconds": "Carry the authenticated producer ledger value 1250 without adding the already settled interval again",
        "download_bytes": "Carry the authenticated initial reservation and producer ledger value 8066717862 without adjustment",
    }
    if not _same_exact(context.get("accounting_rationale"), expected_rationale):
        raise ValueError("tokenizer recovery accounting rationale changed")

    reservation = context.get("authorization_reservation")
    baseline = manifest.get("artifact_baseline")
    if not isinstance(reservation, Mapping) or set(reservation) != {"path", "sha256", "size_bytes"}:
        raise ValueError("tokenizer authorization reservation context is malformed")
    if type(reservation.get("size_bytes")) is not int or reservation.get("size_bytes") <= 0:
        raise ValueError("tokenizer authorization reservation size is invalid")
    if not isinstance(baseline, Mapping) or set(baseline) != set(original["artifact_baseline"]) | _BASELINE_ADDITIONS:
        raise ValueError("tokenizer artifact baseline has unknown or missing fields")
    for key, value in original["artifact_baseline"].items():
        if not _same_exact(baseline.get(key), value):
            raise ValueError(f"tokenizer recovery changed inherited artifact baseline field {key}")
    if (
        not _same_exact(baseline.get("tokenizer_predecessor_immutable_files"), verified["inventory"])
        or type(baseline.get("tokenizer_predecessor_immutable_bytes")) is not int
        or baseline.get("tokenizer_predecessor_immutable_bytes") != EXPECTED_IMMUTABLE_BYTES
        or type(baseline.get("tokenizer_predecessor_status_reserve_bytes")) is not int
        or baseline.get("tokenizer_predecessor_status_reserve_bytes") != EXPECTED_STATUS_BYTES
        or not _same_exact(baseline.get("tokenizer_authorization_reservation"), dict(reservation))
    ):
        raise ValueError("tokenizer artifact baseline differs from predecessor inventory or authorization")

    authorization_path = _authorization_path(previous)
    slots = list(previous.parent.glob("tokenizer-authorization-*.json"))
    if len(slots) != 1 or _checked_path(slots[0]).resolve(strict=True) != authorization_path.resolve(strict=False):
        raise ValueError("tokenizer authorization namespace must contain exactly the fixed predecessor slot")
    authorization, raw = _load_authorization(authorization_path, predecessor_root=previous)
    expected_reservation = {
        "path": str(authorization_path), "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw),
    }
    if not _same_exact(dict(reservation), expected_reservation) or manifest.get("created_utc") != authorization.get("created_utc"):
        raise ValueError("tokenizer authorization path, hash, bytes, or created time differs")
    if type(manifest.get("created_utc")) is not str:
        raise ValueError("tokenizer manifest created time is invalid")

    expected_top_keys = (set(original) - _TOP_LEVEL_EXCLUDED) | _TOP_LEVEL_EXCLUDED
    if set(manifest) != expected_top_keys:
        raise ValueError("tokenizer schema 4 manifest has unknown or missing fields")
    for key, value in original.items():
        if key not in _TOP_LEVEL_EXCLUDED and not _same_exact(manifest.get(key), value):
            raise ValueError(f"tokenizer passthrough field {key} differs from authenticated predecessor")
    if not _same_exact(manifest.get("recovery"), original.get("recovery")) or not _same_exact(manifest.get("repair"), original.get("repair")):
        raise ValueError("tokenizer recovery rewrote original recovery or repair ancestry")
    if manifest.get("execution_id") != _sha({key: value for key, value in manifest.items() if key != "execution_id"}):
        raise ValueError("tokenizer recovery execution ID does not authenticate the exact manifest")
