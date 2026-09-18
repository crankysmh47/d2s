"""Durable local checkpoints and completion evidence for recurrence training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .data import RenderedExample
from .event_identity import task9_failure_event_id
from .lora import adapter_sha256, adapter_state_dict
from .provenance import append_event, canonical_json, sha256_file, sha256_payload, write_immutable_json


_SCHEMA_VERSION = 1
_CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
_REQUIRED_SIGNATURES = (
    "source",
    "data",
    "config",
    "code",
    "arm",
    "seed",
    "initialization_hash",
)
_LATEST_NAME = "latest.pt"
_PREVIOUS_NAME = "previous.pt"
_ARMS = frozenset(("a1", "a2", "a3"))
_ADAPTED_MODES = {"a1": "single", "a2": "repeat", "a3": "single"}
_BASE_LEARNING_RATE = 1e-4
_WARMUP_UPDATES = 20
_ACCUMULATION_STEPS = 8
_REGISTERED_TRAIN_EXAMPLES = 8_000
_INITIALIZATION_SCHEMA_VERSION = 1


class FreshProcessRequired(RuntimeError):
    """A CUDA/process fault ended one attempt and requires a new process."""


class FailedEnvironment(RuntimeError):
    """The registered one-retry budget was exhausted by repeated faults."""


@dataclass(frozen=True, slots=True)
class TrainingState:
    """The last completely committed optimizer update in a training attempt."""

    update: int
    cursor: int
    elapsed_cuda_seconds: float
    attempt: int

    def __post_init__(self) -> None:
        for name, value in (("update", self.update), ("cursor", self.cursor), ("attempt", self.attempt)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.attempt < 1:
            raise ValueError("attempt must be at least one")
        if self.cursor != self.update * _ACCUMULATION_STEPS:
            raise ValueError("cursor must equal update * accumulation steps")
        if not isinstance(self.elapsed_cuda_seconds, (int, float)) or isinstance(
            self.elapsed_cuda_seconds, bool
        ) or not math.isfinite(float(self.elapsed_cuda_seconds)) or self.elapsed_cuda_seconds < 0:
            raise ValueError("elapsed_cuda_seconds must be a finite non-negative number")


def save_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    state: TrainingState,
    signatures: Mapping[str, str],
) -> Path:
    """Atomically publish a locally trusted, fully resumable checkpoint.

    The old ``latest.pt`` remains live until the newly fsynced checkpoint can
    replace it.  Its old bytes are retained as ``previous.pt`` before the
    pointer changes, so a failed final rename never destroys recovery state.
    """
    root = _checkpoint_directory(directory)
    normalized_signatures = _normalize_signatures(signatures)
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    if not isinstance(state, TrainingState):
        raise TypeError("state must be a TrainingState")

    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "adapter_state": _adapter_payload(model),
        "optimizer_state": optimizer.state_dict(),
        # The V0 schedule is calculated from the complete update number.  Task
        # 9 consumes this explicit state rather than inferring it from events.
        "scheduler_state": {"last_completed_update": state.update},
        "rng_state": _capture_rng_state(),
        "state": asdict(state),
        "initialization_hash": normalized_signatures["initialization_hash"],
        "signatures": normalized_signatures,
    }

    latest = root / _LATEST_NAME
    temporary = _write_torch_temporary(root, payload)
    try:
        checkpoint_hash = sha256_file(temporary)
        _write_checkpoint_sidecar(root, checkpoint_hash, normalized_signatures)
        _write_checkpoint_manifest(root, checkpoint_hash, payload, normalized_signatures)
        if latest.exists():
            _retain_previous(latest, root / _PREVIOUS_NAME)
        os.replace(temporary, latest)
    finally:
        temporary.unlink(missing_ok=True)
    return latest


def load_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    signatures: Mapping[str, str],
) -> TrainingState:
    """Restore only a complete compatible local checkpoint.

    Signature, hash, schema, adapter-shape, and state validation complete
    before any model, optimizer, or RNG mutation.  ``previous.pt`` is used
    only when an interrupted publication left no ``latest.pt`` pointer.
    """
    root = _checkpoint_directory(directory)
    normalized_signatures = _normalize_signatures(signatures)
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")

    path = root / _LATEST_NAME
    if not path.is_file():
        path = root / _PREVIOUS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no complete local checkpoint in {root}")
    _require_direct_child(root, path)

    checkpoint_hash = sha256_file(path)
    _validate_checkpoint_sidecar(root, checkpoint_hash, normalized_signatures)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # torch exposes several version-dependent errors.
        raise ValueError("could not load trusted local checkpoint") from error
    _validate_payload(payload, model, optimizer, normalized_signatures)

    assert isinstance(payload, dict)
    adapter_payload = payload["adapter_state"]
    optimizer_payload = payload["optimizer_state"]
    rng_payload = payload["rng_state"]
    state_payload = payload["state"]
    assert isinstance(adapter_payload, Mapping)
    assert isinstance(optimizer_payload, Mapping)
    assert isinstance(rng_payload, Mapping)
    assert isinstance(state_payload, Mapping)

    # `_validate_payload` has established an exact name/shape/dtype match, so
    # copies cannot partially alter the adapter set.  Optimizer/RNG restore is
    # deliberately after every compatibility and corruption check.
    with torch.no_grad():
        for name, target in adapter_state_dict(model).items():
            saved = adapter_payload[name]
            assert isinstance(saved, Tensor)
            target.copy_(saved.to(device=target.device, dtype=target.dtype))
    optimizer.load_state_dict(dict(optimizer_payload))
    _restore_rng_state(rng_payload)
    return _training_state_from_payload(state_payload)


def validate_checkpoint_artifact(
    directory: Path,
    signatures: Mapping[str, str],
    *,
    expected_attempt: int,
    expected_update: int,
    expected_arm: str | None = None,
    expected_seed: int | None = None,
    expected_adapter_schema: Mapping[str, object] | None = None,
    checkpoint_bytes: bytes | None = None,
) -> dict[str, object]:
    """Validate the immutable part of a resumable checkpoint before dispatch.

    The stage child still calls :func:`load_checkpoint` with its concrete model
    and optimizer.  This parent-side validator deliberately performs every
    model-independent check first: the direct-child pointer, SHA sidecar,
    complete payload schema, signatures, optimizer/scheduler/RNG containers,
    and durable update/attempt identity.  It returns the exact bytes digest
    bound into the retry request.
    """
    root = _checkpoint_directory(directory)
    # The orchestration boundary has the four predecessor/control hashes;
    # Task-9's concrete arm/seed/initialization hashes are authenticated by
    # the checkpoint payload and are carried by the child implementation.
    if set(signatures) == set(_REQUIRED_SIGNATURES):
        normalized = _normalize_signatures(signatures)
    elif set(signatures) == {"source", "data", "config", "code"} and all(isinstance(value, str) and value for value in signatures.values()):
        normalized = dict(sorted(signatures.items()))
    else:
        raise ValueError("checkpoint control signatures are incomplete")
    if isinstance(expected_attempt, bool) or not isinstance(expected_attempt, int) or expected_attempt < 1:
        raise ValueError("expected checkpoint attempt must be a positive integer")
    if isinstance(expected_update, bool) or not isinstance(expected_update, int) or expected_update < 0:
        raise ValueError("expected checkpoint update must be a non-negative integer")
    path = root / _LATEST_NAME
    if not path.is_file():
        path = root / _PREVIOUS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no complete local checkpoint in {root}")
    _require_direct_child(root, path)
    if checkpoint_bytes is not None and not isinstance(checkpoint_bytes, bytes):
        raise TypeError("checkpoint_bytes must be bytes")
    checkpoint_hash = sha256(checkpoint_bytes).hexdigest() if checkpoint_bytes is not None else sha256_file(path)
    sidecar = root / f"checkpoint-{checkpoint_hash}.json"
    _require_direct_child(root, sidecar)
    try:
        sidecar_bytes = sidecar.read_bytes()
        metadata = json.loads(sidecar_bytes.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint SHA-256 sidecar is corrupt") from error
    if not isinstance(metadata, dict) or canonical_json(metadata) != sidecar_bytes or metadata.get("schema_version") != _SCHEMA_VERSION or metadata.get("checkpoint_sha256") != checkpoint_hash or not isinstance(metadata.get("signatures"), Mapping):
        raise ValueError("checkpoint SHA-256 sidecar is corrupt")
    saved_sidecar_signatures = metadata["signatures"]
    if set(saved_sidecar_signatures) != set(_REQUIRED_SIGNATURES) or any(not isinstance(value, str) or not value for value in saved_sidecar_signatures.values()) or any(saved_sidecar_signatures.get(key) != value for key, value in normalized.items()):
        raise ValueError("checkpoint signature mismatch")
    try:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False) if checkpoint_bytes is not None else torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("could not load trusted local checkpoint") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "adapter_state", "optimizer_state", "scheduler_state",
        "rng_state", "state", "initialization_hash", "signatures",
    } or payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("checkpoint payload schema is corrupt")
    saved_signatures = payload.get("signatures")
    if not isinstance(saved_signatures, Mapping) or set(saved_signatures) != set(_REQUIRED_SIGNATURES) or any(not isinstance(value, str) or not value for value in saved_signatures.values()) or any(saved_signatures.get(key) != value for key, value in normalized.items()):
        raise ValueError("checkpoint signature mismatch")
    _validate_checkpoint_manifest(root, checkpoint_hash, payload, saved_signatures)
    if not isinstance(payload.get("initialization_hash"), str) or not payload["initialization_hash"] or payload["initialization_hash"] != saved_signatures["initialization_hash"]:
        raise ValueError("checkpoint initialization_hash mismatch")
    if expected_arm is not None and saved_signatures.get("arm") != expected_arm:
        raise ValueError("checkpoint arm does not match durable failure event")
    if expected_seed is not None and saved_signatures.get("seed") != str(expected_seed):
        raise ValueError("checkpoint seed does not match durable failure event")
    state = _training_state_from_payload(payload.get("state"))
    if state.attempt != expected_attempt or state.update != expected_update:
        raise ValueError("checkpoint state does not match durable failure event")
    scheduler = payload.get("scheduler_state")
    if not isinstance(scheduler, Mapping) or set(scheduler) != {"last_completed_update"} or scheduler.get("last_completed_update") != state.update:
        raise ValueError("checkpoint scheduler/update state is corrupt")
    adapters = payload.get("adapter_state")
    if not isinstance(adapters, Mapping) or not adapters or any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in adapters.items()):
        raise ValueError("checkpoint adapter state is corrupt")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping) or set(optimizer_state) != {"state", "param_groups"} or not isinstance(optimizer_state.get("param_groups"), list) or not optimizer_state["param_groups"] or not isinstance(optimizer_state.get("state"), Mapping):
        raise ValueError("checkpoint optimizer state is corrupt")
    for group in optimizer_state["param_groups"]:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list) or any(isinstance(identifier, bool) or not isinstance(identifier, int) for identifier in group["params"]):
            raise ValueError("checkpoint optimizer state is corrupt")
    if expected_adapter_schema is not None:
        expected_parameters = expected_adapter_schema.get("parameters")
        actual_parameters = {
            str(name): {"shape": [int(dimension) for dimension in value.shape], "dtype": str(value.dtype)}
            for name, value in adapters.items()
            if isinstance(value, Tensor)
        }
        if not isinstance(expected_parameters, Mapping) or actual_parameters != dict(expected_parameters):
            raise ValueError("checkpoint adapter schema does not match registered model")
        expected_initialization = expected_adapter_schema.get("initialization_hash")
        if not isinstance(expected_initialization, str) or expected_initialization != saved_signatures.get("initialization_hash"):
            raise ValueError("checkpoint initialization identity does not match registered model")
    _validate_rng_state(payload.get("rng_state"))
    assert isinstance(payload, Mapping)
    manifest_digest = _checkpoint_manifest_digest(root, checkpoint_hash, payload)
    return {
        "path": str(path),
        "checkpoint_sha256": checkpoint_hash,
        "sidecar_sha256": sha256_file(sidecar),
        "attempt": state.attempt,
        "update": state.update,
        "manifest_digest": manifest_digest,
        "signatures": dict(sorted(saved_signatures.items())),
    }


def write_completed_training_manifest(path: Path, payload: Mapping[str, object]) -> str:
    """Write an immutable completion record that explicitly retains event history."""
    if not isinstance(payload, Mapping):
        raise TypeError("training manifest payload must be a mapping")
    event_indices = _event_indices(payload, "event_indices")
    failure_indices = _event_indices(payload, "failure_event_indices")
    if not set(failure_indices).issubset(event_indices):
        raise ValueError("failure_event_indices must be included in event_indices")
    return write_immutable_json(path, dict(payload))


def capture_initialization(path: Path, model: nn.Module, seed: int) -> str:
    """Capture one immutable per-seed adapter initialization snapshot."""
    if not isinstance(path, Path):
        raise TypeError("initialization snapshot path must be a Path")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    state = _adapter_payload(model)
    initialization_hash = _adapter_state_sha256(state)
    payload = {
        "schema_version": _INITIALIZATION_SCHEMA_VERSION,
        "seed": seed,
        "initialization_hash": initialization_hash,
        "adapter_state": state,
    }
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            existing = torch.load(destination, map_location="cpu", weights_only=False)
        except Exception as error:
            raise ValueError("initialization snapshot is corrupt") from error
        if not _initialization_payload_equal(existing, payload):
            raise FileExistsError("initialization snapshot is immutable")
        return initialization_hash
    temporary = _write_torch_temporary(destination.parent, payload)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return initialization_hash


def load_initialization(path: Path, model: nn.Module, *, expected_seed: int | None = None) -> str:
    """Load an immutable adapter snapshot after complete no-mutation validation."""
    if not isinstance(path, Path):
        raise TypeError("initialization snapshot path must be a Path")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if expected_seed is not None and (
        isinstance(expected_seed, bool) or not isinstance(expected_seed, int) or expected_seed < 0
    ):
        raise ValueError("expected_seed must be a non-negative integer")
    try:
        payload = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("initialization snapshot is corrupt") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "seed",
        "initialization_hash",
        "adapter_state",
    }:
        raise ValueError("initialization snapshot is corrupt")
    if payload["schema_version"] != _INITIALIZATION_SCHEMA_VERSION:
        raise ValueError("initialization snapshot is corrupt")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("initialization snapshot seed is corrupt")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("initialization snapshot seed mismatch")
    saved = payload["adapter_state"]
    if not isinstance(saved, Mapping):
        raise ValueError("initialization snapshot adapter state is corrupt")
    expected = adapter_state_dict(model)
    if set(saved) != set(expected):
        raise ValueError("initialization snapshot adapter state is corrupt")
    normalized: dict[str, Tensor] = {}
    for name, target in expected.items():
        tensor = saved[name]
        if not isinstance(tensor, Tensor) or tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError("initialization snapshot adapter state is corrupt")
        normalized[name] = tensor.detach().to(device="cpu").contiguous()
    initialization_hash = payload["initialization_hash"]
    if not isinstance(initialization_hash, str) or initialization_hash != _adapter_state_sha256(normalized):
        raise ValueError("initialization snapshot hash mismatch")
    with torch.no_grad():
        for name, target in expected.items():
            target.copy_(normalized[name].to(device=target.device, dtype=target.dtype))
    return initialization_hash


def _enable_deterministic_algorithms() -> None:
    """Require deterministic kernels for both fresh and resumed processes."""
    # The reference experiment prefers a loud failure to a silently different
    # numerical path.  These flags are harmless for the CPU fixtures.
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_deterministic(seed: int) -> None:
    """Set every RNG used by the training process to the registered seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _enable_deterministic_algorithms()


def _make_adapter_optimizer(model: nn.Module) -> torch.optim.AdamW:
    """Build the registered optimizer over adapters only."""
    parameters = list(adapter_state_dict(model).values())
    if not parameters:
        raise ValueError("model has no trainable LoRA adapters")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if set(map(id, trainable)) != set(map(id, parameters)):
        raise ValueError("only registered adapter parameters may be trainable")
    return torch.optim.AdamW(
        parameters,
        lr=_BASE_LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )


def _learning_rate_for_update(update: int) -> float:
    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise ValueError("update must be a positive integer")
    return _BASE_LEARNING_RATE * min(update, _WARMUP_UPDATES) / _WARMUP_UPDATES


def _example_ids(examples: Sequence[RenderedExample]) -> list[str]:
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence of RenderedExample values")
    identifiers: list[str] = []
    for example in examples:
        if not isinstance(example, RenderedExample):
            raise TypeError("examples must contain RenderedExample values")
        identifier = example.example.id
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("training example IDs must be non-empty strings")
        identifiers.append(identifier)
    if not identifiers:
        raise ValueError("training examples may not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("training example IDs must be unique")
    return identifiers


def _presentation_indices(
    examples: Sequence[RenderedExample], arm: str, seed: int, updates: int
) -> list[int]:
    """Return ``updates * 8`` IDs from one seeded stream.

    A1/A2 expose prefixes of their one seeded epoch for resumable partial
    targets.  A3 uses the same permutation as a non-reshuffled cycle.
    """
    identifiers = _example_ids(examples)
    if arm not in _ARMS:
        raise ValueError(f"unknown training arm: {arm}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("updates must be a non-negative integer")
    presentations = updates * _ACCUMULATION_STEPS
    order = list(range(len(identifiers)))
    random.Random(seed).shuffle(order)
    if arm in {"a1", "a2"} and presentations > len(order):
        raise ValueError("A1/A2 may not present an example more than once")
    if presentations == 0:
        return []
    if arm == "a3":
        return [order[index % len(order)] for index in range(presentations)]
    return order[:presentations]


def _ordered_id_hash(examples: Sequence[RenderedExample], seed: int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    identifiers = _example_ids(examples)
    order = list(range(len(identifiers)))
    random.Random(seed).shuffle(order)
    return sha256_payload({"ordered_ids": [identifiers[index] for index in order]})


def _event_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("training event history is corrupt")
            records.append(payload)
    return records


def _is_cuda_process_fault(error: BaseException) -> bool:
    if isinstance(error, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cuda",
            "cublas",
            "cudnn",
            "device-side assert",
            "out of memory",
            "nccl",
        )
    )


def _fault_records(records: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        record
        for record in records
        if record.get("kind") == "training_failure"
        and record.get("cuda_process_fault") is True
    ]


def _terminal_environment_record(records: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    for record in records:
        if record.get("kind") == "failed_environment":
            return record
    return None


def _training_manifest(
    *,
    model: nn.Module,
    examples: Sequence[RenderedExample],
    arm: str,
    seed: int,
    requested_updates: int,
    state: TrainingState,
    signatures: Mapping[str, str],
    checkpoint_dir: Path,
    records: Sequence[Mapping[str, object]],
    synchronized_cuda_seconds: float,
) -> dict[str, object]:
    checkpoint_hashes = list(
        dict.fromkeys(
            str(record["checkpoint_hash"])
            for record in records
            if record.get("kind") == "training_checkpoint"
            and isinstance(record.get("checkpoint_hash"), str)
        )
    )
    if not checkpoint_hashes:
        checkpoint_hashes = sorted(
            path.stem.removeprefix("checkpoint-")
            for path in checkpoint_dir.glob("checkpoint-*.json")
            if path.is_file()
        )
    attempts = sorted(
        {
            int(record["attempt"])
            for record in records
            if isinstance(record.get("attempt"), int)
            and not isinstance(record.get("attempt"), bool)
        }
    )
    if not attempts:
        attempts = [state.attempt]
    failures = [dict(record) for record in records if record.get("kind") == "training_failure"]
    if torch.cuda.is_available():
        peak_allocated = max(torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count()))
        peak_reserved = max(torch.cuda.max_memory_reserved(index) for index in range(torch.cuda.device_count()))
    else:
        peak_allocated = peak_reserved = 0
    return {
        "arm": arm,
        "seed": seed,
        "requested_updates": requested_updates,
        "completed_updates": state.update,
        "cursor": state.cursor,
        "unique_example_count": len(_example_ids(examples)),
        "total_example_presentations": state.cursor,
        "ordered_id_sha256": _ordered_id_hash(examples, seed),
        "initialization_hash": signatures["initialization_hash"],
        "final_adapter_hash": adapter_sha256(model),
        "trainable_parameter_count": sum(parameter.numel() for parameter in adapter_state_dict(model).values()),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "synchronized_cuda_seconds": synchronized_cuda_seconds,
        "checkpoint_hashes": checkpoint_hashes,
        "attempts": attempts,
        "failures": failures,
        "completion_status": "completed" if state.update >= requested_updates else "incomplete",
        "determinism_notes": [],
        "event_indices": [
            record["index"]
            for record in records
            if isinstance(record.get("index"), int) and not isinstance(record.get("index"), bool)
        ],
                        "failure_event_indices": [
            record["index"]
            for record in records
            if record.get("kind") == "training_failure"
            and isinstance(record.get("index"), int)
            and not isinstance(record.get("index"), bool)
        ],
    }


def train_arm(
    model: nn.Module,
    examples: Sequence[RenderedExample],
    arm: str,
    seed: int,
    steps: int,
    checkpoint_dir: Path,
    event_path: Path,
    signatures: Mapping[str, str],
    stage: str = "pilot",
) -> dict[str, object]:
    """Train one registered arm with atomic complete-update checkpoints.

    ``steps`` is the target cumulative update count.  Supplying a target above
    a saved checkpoint resumes from its committed cursor; no partial update is
    ever represented as progress.
    """
    if arm not in _ARMS:
        raise ValueError(f"unknown training arm: {arm}")
    if stage not in {"smoke", "pilot", "frozen", "discover", "confirm"}:
        raise ValueError(f"unknown training stage: {stage}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    normalized_signatures = _normalize_signatures(signatures)
    identifiers = _example_ids(examples)
    if len(identifiers) != _REGISTERED_TRAIN_EXAMPLES:
        raise ValueError(
            "training pool must contain exactly "
            f"{_REGISTERED_TRAIN_EXAMPLES:,} unique examples"
        )
    if normalized_signatures["arm"] != arm or normalized_signatures["seed"] != str(seed):
        raise ValueError("training arm/seed does not match checkpoint signatures")
    if adapter_sha256(model) != normalized_signatures["initialization_hash"]:
        raise ValueError("model adapter initialization does not match signatures")
    required_presentations = steps * _ACCUMULATION_STEPS
    if arm in {"a1", "a2"} and required_presentations > len(identifiers):
        raise ValueError("A1/A2 may not present an example more than once")

    checkpoint_root = _checkpoint_directory(checkpoint_dir)
    event_path = event_path.resolve()
    records = _event_records(event_path)
    terminal = _terminal_environment_record(records)
    if terminal is not None:
        raise FailedEnvironment("training stage is terminal: failed_environment")
    prior_faults = _fault_records(records)
    optimizer = _make_adapter_optimizer(model)
    _enable_deterministic_algorithms()
    checkpoint_exists = (checkpoint_root / _LATEST_NAME).is_file() or (
        checkpoint_root / _PREVIOUS_NAME
    ).is_file()
    if checkpoint_exists:
        state = load_checkpoint(checkpoint_root, model, optimizer, normalized_signatures)
        if state.update > steps:
            raise ValueError("requested target update precedes saved checkpoint")
        previous_attempts = [
            record.get("attempt")
            for record in records
            if isinstance(record.get("attempt"), int)
            and not isinstance(record.get("attempt"), bool)
        ]
        durable_attempt = max(previous_attempts, default=state.attempt)
        attempt = max(state.attempt + 1, int(durable_attempt) + (1 if prior_faults else 0))
        # The checkpoint restored the exact RNG stream.  Do not reseed here.
    else:
        _seed_deterministic(seed)
        state = TrainingState(update=0, cursor=0, elapsed_cuda_seconds=0.0, attempt=1)
        durable_attempt = max(
            (
                int(record["attempt"])
                for record in records
                if isinstance(record.get("attempt"), int)
                and not isinstance(record.get("attempt"), bool)
            ),
            default=0,
        )
        attempt = durable_attempt + 1 if records else 1

    append_event(
        event_path,
        {"kind": "training_started", "arm": arm, "seed": seed, "attempt": attempt, "update": state.update},
    )
    records = _event_records(event_path)
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    permutation = _presentation_indices(examples, arm, seed, steps)
    mode = _ADAPTED_MODES[arm]
    elapsed_cuda = state.elapsed_cuda_seconds
    try:
        while state.update < steps:
            update_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for offset in range(_ACCUMULATION_STEPS):
                cursor = state.cursor + offset
                index = permutation[cursor % len(permutation)] if arm == "a3" else permutation[cursor]
                rendered = examples[index]
                device = next(iter(adapter_state_dict(model).values())).device
                input_ids = torch.tensor([rendered.train_input_ids], dtype=torch.long, device=device)
                labels = torch.tensor([rendered.labels], dtype=torch.long, device=device)
                loss = model.answer_loss(input_ids, labels, mode)
                if loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("non-finite training loss")
                (loss / _ACCUMULATION_STEPS).backward()
            adapters = list(adapter_state_dict(model).values())
            gradients = [parameter for parameter in adapters if parameter.grad is not None]
            if not gradients or any(not bool(torch.isfinite(parameter.grad).all()) for parameter in gradients):
                raise FloatingPointError("non-finite or missing adapter gradient")
            clipped_norm = torch.nn.utils.clip_grad_norm_(adapters, 1.0)
            if not bool(torch.isfinite(clipped_norm).all()) or any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in adapters
            ):
                raise FloatingPointError("non-finite clipped adapter gradient")
            completed_update = state.update + 1
            optimizer.param_groups[0]["lr"] = _learning_rate_for_update(completed_update)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                elapsed_cuda += time.perf_counter() - update_started
            committed = TrainingState(
                update=completed_update,
                cursor=state.cursor + _ACCUMULATION_STEPS,
                elapsed_cuda_seconds=elapsed_cuda,
                attempt=attempt,
            )
            # This is the commit boundary: only a fully stepped update is
            # serialized, and resume starts at its cursor.
            if committed.update % 50 == 0 or committed.update == steps:
                checkpoint = save_checkpoint(
                    checkpoint_root, model, optimizer, committed, normalized_signatures
                )
                checkpoint_artifact = validate_checkpoint_artifact(
                    checkpoint_root,
                    normalized_signatures,
                    expected_attempt=committed.attempt,
                    expected_update=committed.update,
                )
                checkpoint_relative = checkpoint.resolve().relative_to(event_path.parent.resolve()).as_posix()
                append_event(
                    event_path,
                    {
                        "kind": "training_checkpoint",
                        "arm": arm,
                        "seed": seed,
                        "attempt": attempt,
                        "update": committed.update,
                        "checkpoint_hash": sha256_file(checkpoint),
                        "checkpoint_path": checkpoint_relative,
                        "checkpoint_manifest_digest": checkpoint_artifact["manifest_digest"],
                    },
                )
            state = committed
            if state.update % 10 == 0 or state.update == steps:
                append_event(
                    event_path,
                    {
                        "kind": "training_progress",
                        "arm": arm,
                        "seed": seed,
                        "attempt": attempt,
                        "update": state.update,
                        "cursor": state.cursor,
                    },
                )
        append_event(
            event_path,
            {
                "kind": "training_completed",
                "arm": arm,
                "seed": seed,
                "attempt": attempt,
                "update": state.update,
                "cursor": state.cursor,
            },
        )
    except Exception as error:
        if _is_cuda_process_fault(error):
            chargeable_seconds = max(0.0, time.perf_counter() - update_started)
            failure_payload: dict[str, object] = {
                "kind": "training_failure",
                "stage": stage,
                "arm": arm,
                "seed": seed,
                "attempt": attempt,
                "update": state.update,
                "message": str(error),
                "error_type": type(error).__name__,
                "cuda_process_fault": True,
                "chargeable_cuda_seconds": chargeable_seconds,
                "requires_fresh_process": not prior_faults,
                "retry_status": "fresh_process_required" if not prior_faults else "terminal",
                "payload": {"message": str(error)},
            }
            checkpoint_evidence = _checkpoint_event_evidence(
                checkpoint_root, event_path, normalized_signatures, state
            )
            if checkpoint_evidence is not None:
                failure_payload.update(checkpoint_evidence)
            failure_payload["event_id"] = _task9_failure_event_id(failure_payload)
            failure_index = append_event(
                event_path,
                failure_payload,
            )
            if prior_faults:
                append_event(
                    event_path,
                    {
                    "kind": "failed_environment",
                    "stage": stage,
                    "arm": arm,
                    "seed": seed,
                    "attempts": [
                        int(record["attempt"])
                        for record in [*prior_faults, {"attempt": attempt}]
                        if isinstance(record.get("attempt"), int)
                        and not isinstance(record.get("attempt"), bool)
                    ],
                    "failure_event_indices": [
                        int(record["index"])
                        for record in [*prior_faults, {"index": failure_index}]
                        if isinstance(record.get("index"), int)
                        and not isinstance(record.get("index"), bool)
                        ],
                    "failure_event_ids": [
                        str(record["event_id"])
                        for record in [*prior_faults, failure_payload]
                        if isinstance(record.get("event_id"), str)
                    ],
                        "completion_status": "failed_environment",
                        "requires_fresh_process": False,
                        "payload": {
                            "attempts": [
                                int(record["attempt"])
                                for record in [*prior_faults, {"attempt": attempt}]
                                if isinstance(record.get("attempt"), int)
                                and not isinstance(record.get("attempt"), bool)
                            ],
                        "failure_event_indices": [
                                int(record["index"])
                                for record in [*prior_faults, {"index": failure_index}]
                                if isinstance(record.get("index"), int)
                                and not isinstance(record.get("index"), bool)
                            ],
                            "failure_event_ids": [
                                str(record["event_id"])
                                for record in [*prior_faults, failure_payload]
                                if isinstance(record.get("event_id"), str)
                            ],
                        },
                    },
                )
                raise FailedEnvironment(
                    "second CUDA/process fault; stage is terminal failed_environment"
                ) from error
            raise FreshProcessRequired(
                "CUDA/process fault; terminate this process and retry once in a fresh process"
            ) from error
        append_event(
            event_path,
            {
                "kind": "training_failure",
                "stage": stage,
                "arm": arm,
                "seed": seed,
                "attempt": attempt,
                "update": state.update,
                "message": str(error),
                "error_type": type(error).__name__,
                "cuda_process_fault": False,
                "event_id": _task9_failure_event_id({
                    "kind": "training_failure", "stage": stage, "arm": arm,
                    "seed": seed, "attempt": attempt, "update": state.update,
                }),
                "payload": {"message": str(error)},
            },
        )
        raise
    return _training_manifest(
        model=model,
        examples=examples,
        arm=arm,
        seed=seed,
        requested_updates=steps,
        state=state,
        signatures=normalized_signatures,
        checkpoint_dir=checkpoint_root,
        records=_event_records(event_path),
        synchronized_cuda_seconds=elapsed_cuda,
    )


def _checkpoint_directory(directory: Path) -> Path:
    if not isinstance(directory, Path):
        raise TypeError("checkpoint directory must be a Path")
    return directory.resolve()


def _checkpoint_event_evidence(
    checkpoint_root: Path,
    event_path: Path,
    signatures: Mapping[str, str],
    state: TrainingState,
) -> dict[str, object] | None:
    latest = checkpoint_root / _LATEST_NAME
    if not latest.is_file():
        return None
    try:
        artifact = validate_checkpoint_artifact(
            checkpoint_root,
            signatures,
            expected_attempt=state.attempt,
            expected_update=state.update,
        )
        relative = latest.resolve().relative_to(event_path.parent.resolve()).as_posix()
    except (OSError, ValueError):
        return None
    if not relative.startswith("checkpoints/"):
        return None
    return {
        "checkpoint_path": relative,
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "checkpoint_manifest_digest": artifact["manifest_digest"],
    }


def _task9_failure_event_id(event: Mapping[str, object]) -> str:
    return task9_failure_event_id(event)


def _normalize_signatures(signatures: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(signatures, Mapping):
        raise TypeError("signatures must be a mapping")
    normalized: dict[str, str] = {}
    for name, value in signatures.items():
        if not isinstance(name, str) or not name:
            raise ValueError("signature names must be non-empty strings")
        if not isinstance(value, str) or not value:
            raise ValueError(f"signature {name!r} must be a non-empty string")
        normalized[name] = value
    unknown = sorted(set(normalized) - set(_REQUIRED_SIGNATURES))
    if unknown:
        raise ValueError(f"unknown checkpoint signatures: {', '.join(unknown)}")
    missing = [name for name in _REQUIRED_SIGNATURES if name not in normalized]
    if missing:
        raise ValueError(f"missing checkpoint signatures: {', '.join(missing)}")
    return dict(sorted(normalized.items()))


def _adapter_payload(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in adapter_state_dict(model).items()
    }


def registered_adapter_schema(model: nn.Module, initialization_hash: str) -> dict[str, object]:
    """Return the concrete LoRA schema a retry is authorized to load."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(initialization_hash, str) or not initialization_hash:
        raise ValueError("initialization_hash must be a non-empty string")
    return {
        "parameters": {
            name: {"shape": [int(dimension) for dimension in parameter.shape], "dtype": str(parameter.dtype)}
            for name, parameter in adapter_state_dict(model).items()
        },
        "initialization_hash": initialization_hash,
    }


def _adapter_state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = sha256()
    for name, tensor in sorted(state.items()):
        contiguous = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(dimension) for dimension in contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _initialization_payload_equal(actual: object, expected: Mapping[str, object]) -> bool:
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        return False
    for key in ("schema_version", "seed", "initialization_hash"):
        if actual.get(key) != expected.get(key):
            return False
    actual_state = actual.get("adapter_state")
    expected_state = expected.get("adapter_state")
    if not isinstance(actual_state, Mapping) or not isinstance(expected_state, Mapping):
        return False
    if set(actual_state) != set(expected_state):
        return False
    return all(
        isinstance(actual_state[name], Tensor)
        and isinstance(expected_state[name], Tensor)
        and torch.equal(actual_state[name], expected_state[name])
        for name in expected_state
    )


def _capture_rng_state() -> dict[str, object]:
    cuda_state: tuple[Tensor, ...]
    if torch.cuda.is_available():
        cuda_state = tuple(torch.cuda.get_rng_state_all())
    else:
        cuda_state = ()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
    }


def _write_torch_temporary(root: Path, payload: Mapping[str, object]) -> Path:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=root, prefix=".latest.", suffix=".pt.tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _retain_previous(latest: Path, previous: Path) -> None:
    """Publish a durable old-pointer copy while leaving ``latest`` intact."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=latest.parent, prefix=".previous.", suffix=".pt.tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        try:
            os.link(latest, temporary)
        except OSError:
            # Filesystems without hard links retain the same bytes through a
            # fsynced sibling copy; latest is still untouched until final swap.
            with latest.open("rb") as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
        os.replace(temporary, previous)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checkpoint_sidecar(
    root: Path, checkpoint_hash: str, signatures: Mapping[str, str]
) -> None:
    write_immutable_json(
        root / f"checkpoint-{checkpoint_hash}.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint_hash,
            "signatures": dict(signatures),
        },
    )


def _tensor_manifest(value: Tensor) -> dict[str, object]:
    if not isinstance(value, Tensor):
        raise ValueError("checkpoint tensor manifest is corrupt")
    finite = bool(torch.isfinite(value).all())
    return {"shape": [int(dimension) for dimension in value.shape], "dtype": str(value.dtype), "finite": finite}


def _manifest_value(value: object) -> object:
    if isinstance(value, Tensor):
        descriptor = _tensor_manifest(value)
        if value.numel() == 1:
            descriptor["value"] = value.detach().to(device="cpu").item()
        return descriptor
    if isinstance(value, Mapping):
        return {str(key): _manifest_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_manifest_value(item) for item in value]
    if isinstance(value, list):
        return [_manifest_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("checkpoint manifest contains an unsupported value")


def _checkpoint_manifest_payload(
    checkpoint_hash: str, payload: Mapping[str, object], signatures: Mapping[str, str]
) -> dict[str, object]:
    adapters = payload.get("adapter_state")
    optimizer_state = payload.get("optimizer_state")
    state = payload.get("state")
    if not isinstance(adapters, Mapping) or not isinstance(optimizer_state, Mapping) or not isinstance(state, Mapping):
        raise ValueError("checkpoint manifest cannot describe malformed payload")
    return {
        "schema_version": _CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_hash,
        "signatures": dict(sorted(signatures.items())),
        "state": dict(state),
        "scheduler_state": _manifest_value(payload.get("scheduler_state")),
        "adapter_schema": {
            str(name): {"shape": [int(dimension) for dimension in value.shape], "dtype": str(value.dtype)}
            for name, value in adapters.items() if isinstance(value, Tensor)
        },
        "adapter_manifest": {
            str(name): _tensor_manifest(value) for name, value in adapters.items()
        },
        "optimizer_manifest": _manifest_value(optimizer_state),
    }


def _write_checkpoint_manifest(
    root: Path, checkpoint_hash: str, payload: Mapping[str, object], signatures: Mapping[str, str]
) -> None:
    write_immutable_json(
        root / f"manifest-{checkpoint_hash}.json",
        _checkpoint_manifest_payload(checkpoint_hash, payload, signatures),
    )


def _checkpoint_manifest_digest(
    root: Path, checkpoint_hash: str, payload: Mapping[str, object]
) -> str:
    manifest = root / f"manifest-{checkpoint_hash}.json"
    return sha256_payload({
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": sha256_file(manifest),
        "payload_signatures": dict(sorted(payload["signatures"].items())),
        "state": dict(payload["state"]),
    })


def _validate_checkpoint_manifest(
    root: Path, checkpoint_hash: str, payload: object, signatures: Mapping[str, str]
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is corrupt")
    path = root / f"manifest-{checkpoint_hash}.json"
    _require_direct_child(root, path)
    try:
        raw = path.read_bytes()
        metadata = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint manifest is corrupt") from error
    if not isinstance(metadata, dict) or canonical_json(metadata) != raw:
        raise ValueError("checkpoint manifest is corrupt")
    expected = _checkpoint_manifest_payload(checkpoint_hash, payload, signatures)
    if metadata != expected:
        raise ValueError("checkpoint manifest does not match payload")
    adapters = payload.get("adapter_state")
    if not isinstance(adapters, Mapping) or not adapters or any(not isinstance(name, str) for name in adapters):
        raise ValueError("checkpoint adapter state is corrupt")
    for tensor in adapters.values():
        if not isinstance(tensor, Tensor) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("checkpoint adapter state is corrupt")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint training state is corrupt")
    update = state.get("update")
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("checkpoint training state is corrupt")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer state is corrupt")
    params = [nn.Parameter(tensor.detach().to(device="cpu").clone()) for tensor in adapters.values()]
    trial = torch.optim.AdamW(params, lr=_BASE_LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    _validate_optimizer_state(optimizer_state, trial, update)


def _validate_checkpoint_sidecar(
    root: Path, checkpoint_hash: str, signatures: Mapping[str, str]
) -> None:
    sidecar = root / f"checkpoint-{checkpoint_hash}.json"
    _require_direct_child(root, sidecar)
    if not sidecar.is_file():
        raise ValueError("checkpoint SHA-256 sidecar is missing")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint SHA-256 sidecar is corrupt") from error
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version",
        "checkpoint_sha256",
        "signatures",
    }:
        raise ValueError("checkpoint SHA-256 sidecar is corrupt")
    if metadata["schema_version"] != _SCHEMA_VERSION or metadata["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("checkpoint SHA-256 sidecar does not match checkpoint")
    saved_signatures = metadata["signatures"]
    if not isinstance(saved_signatures, Mapping):
        raise ValueError("checkpoint SHA-256 sidecar signatures are corrupt")
    mismatches = _signature_mismatches(saved_signatures, signatures)
    if mismatches:
        raise ValueError(f"checkpoint signature mismatch: {', '.join(mismatches)}")


def _validate_payload(
    payload: object,
    model: nn.Module,
    optimizer: Optimizer,
    signatures: Mapping[str, str],
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is corrupt")
    expected_keys = {
        "schema_version",
        "adapter_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "state",
        "initialization_hash",
        "signatures",
    }
    if set(payload) != expected_keys or payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("checkpoint payload schema is corrupt")
    saved_signatures = payload["signatures"]
    if not isinstance(saved_signatures, Mapping):
        raise ValueError("checkpoint signatures are corrupt")
    mismatches = _signature_mismatches(saved_signatures, signatures)
    if mismatches:
        raise ValueError(f"checkpoint signature mismatch: {', '.join(mismatches)}")
    expected_initialization = signatures.get("initialization_hash")
    if expected_initialization is not None and payload["initialization_hash"] != expected_initialization:
        raise ValueError("checkpoint initialization_hash mismatch")

    adapters = payload["adapter_state"]
    expected_adapters = adapter_state_dict(model)
    if not isinstance(adapters, Mapping) or set(adapters) != set(expected_adapters):
        raise ValueError("checkpoint adapter state is corrupt")
    for name, target in expected_adapters.items():
        saved = adapters[name]
        if not isinstance(saved, Tensor) or saved.shape != target.shape or saved.dtype != target.dtype:
            raise ValueError(f"checkpoint adapter state is corrupt for {name}")
    state = _training_state_from_payload(payload["state"])
    _validate_optimizer_state(payload["optimizer_state"], optimizer, state.update)
    scheduler = payload["scheduler_state"]
    if not isinstance(scheduler, Mapping) or set(scheduler) != {"last_completed_update"}:
        raise ValueError("checkpoint scheduler state is corrupt")
    completed_update = scheduler["last_completed_update"]
    if isinstance(completed_update, bool) or not isinstance(completed_update, int):
        raise ValueError("checkpoint scheduler state is corrupt")
    if completed_update != state.update:
        raise ValueError("checkpoint scheduler/update state is corrupt")
    _validate_rng_state(payload["rng_state"])


def _training_state_from_payload(payload: object) -> TrainingState:
    if not isinstance(payload, Mapping) or set(payload) != {
        "update",
        "cursor",
        "elapsed_cuda_seconds",
        "attempt",
    }:
        raise ValueError("checkpoint training state is corrupt")
    try:
        return TrainingState(
            update=payload["update"],
            cursor=payload["cursor"],
            elapsed_cuda_seconds=payload["elapsed_cuda_seconds"],
            attempt=payload["attempt"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint training state is corrupt") from error


def _validate_rng_state(payload: object) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("checkpoint RNG state is corrupt")
    python_state = payload["python"]
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("checkpoint Python RNG state is corrupt") from error
    numpy_state = payload["numpy"]
    try:
        np.random.RandomState().set_state(numpy_state)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("checkpoint NumPy RNG state is corrupt") from error

    cpu_state = payload["torch_cpu"]
    if (
        not isinstance(cpu_state, Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.device.type != "cpu"
    ):
        raise ValueError("checkpoint CPU RNG state is corrupt")
    try:
        torch.Generator(device="cpu").set_state(cpu_state)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint CPU RNG state is corrupt") from error

    cuda_states = payload["torch_cuda"]
    if not isinstance(cuda_states, tuple):
        raise ValueError("checkpoint CUDA RNG state is corrupt")
    expected_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != expected_cuda_devices:
        raise ValueError("checkpoint CUDA RNG state is corrupt")
    for index, cuda_state in enumerate(cuda_states):
        if (
            not isinstance(cuda_state, Tensor)
            or cuda_state.dtype != torch.uint8
            or cuda_state.device.type != "cpu"
        ):
            raise ValueError("checkpoint CUDA RNG state is corrupt")
        try:
            torch.Generator(device=torch.device("cuda", index)).set_state(cuda_state)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError("checkpoint CUDA RNG state is corrupt") from error


def _validate_optimizer_state(payload: object, optimizer: Optimizer, update: int) -> None:
    """Exercise restore validation against a deep-copied, never-live optimizer."""
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint optimizer state is corrupt")
    groups = payload.get("param_groups")
    states = payload.get("state")
    if not isinstance(groups, list) or not isinstance(states, Mapping):
        raise ValueError("checkpoint optimizer state is corrupt")
    live_groups = optimizer.param_groups
    if len(groups) != len(live_groups):
        raise ValueError("checkpoint optimizer state is corrupt")

    parameter_by_saved_id: dict[int, nn.Parameter] = {}
    for saved_group, live_group in zip(groups, live_groups, strict=True):
        if not isinstance(saved_group, Mapping):
            raise ValueError("checkpoint optimizer state is corrupt")
        saved_parameters = saved_group.get("params")
        live_parameters = live_group.get("params")
        if not isinstance(saved_parameters, list) or not isinstance(live_parameters, list):
            raise ValueError("checkpoint optimizer state is corrupt")
        if len(saved_parameters) != len(live_parameters):
            raise ValueError("checkpoint optimizer state is corrupt")
        if any(
            isinstance(identifier, bool) or not isinstance(identifier, int)
            for identifier in saved_parameters
        ) or len(set(saved_parameters)) != len(saved_parameters):
            raise ValueError("checkpoint optimizer state is corrupt")
        if not all(isinstance(parameter, nn.Parameter) for parameter in live_parameters):
            raise ValueError("checkpoint optimizer state is corrupt")
        if set(saved_group) != set(live_group):
            raise ValueError("checkpoint optimizer state is corrupt")
        expected_learning_rate = (
            _BASE_LEARNING_RATE if update == 0 else _learning_rate_for_update(update)
        )
        for field, expected in live_group.items():
            if field == "params":
                continue
            if field == "lr":
                saved_learning_rate = saved_group.get(field)
                if (
                    isinstance(saved_learning_rate, bool)
                    or not isinstance(saved_learning_rate, (int, float))
                    or not math.isfinite(float(saved_learning_rate))
                    or float(saved_learning_rate) != expected_learning_rate
                ):
                    raise ValueError("checkpoint optimizer learning rate is corrupt")
            else:
                _validate_optimizer_group_value(saved_group[field], expected)
        parameter_by_saved_id.update(zip(saved_parameters, live_parameters, strict=True))

    if any(
        isinstance(identifier, bool) or not isinstance(identifier, int)
        for identifier in states
    ) or not set(states).issubset(parameter_by_saved_id):
        raise ValueError("checkpoint optimizer state is corrupt")
    saved_state_ids = set(states)
    if update > 0 and saved_state_ids != set(parameter_by_saved_id):
        raise ValueError("checkpoint optimizer state is incomplete")
    for identifier, state in states.items():
        _validate_adamw_parameter_state(
            state,
            parameter_by_saved_id[identifier],
            expected_step=update,
        )

    try:
        # Torch's own loader validates version-specific optimizer invariants;
        # copy it first so neither failure nor success mutates the live object.
        trial_optimizer = copy.deepcopy(optimizer)
        trial_optimizer.load_state_dict(copy.deepcopy(dict(payload)))
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise ValueError("checkpoint optimizer state is corrupt") from error


def _validate_optimizer_group_value(value: object, expected: object) -> None:
    if isinstance(expected, bool):
        if not isinstance(value, bool):
            raise ValueError("checkpoint optimizer state is corrupt")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value != expected
        ):
            raise ValueError("checkpoint optimizer state is corrupt")
    elif isinstance(expected, tuple):
        if not isinstance(value, tuple) or len(value) != len(expected):
            raise ValueError("checkpoint optimizer state is corrupt")
        for item, expected_item in zip(value, expected, strict=True):
            _validate_optimizer_group_value(item, expected_item)
    elif value != expected:
        raise ValueError("checkpoint optimizer state is corrupt")


def _validate_adamw_parameter_state(
    state: object, parameter: nn.Parameter, *, expected_step: int
) -> None:
    if not isinstance(state, Mapping) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("checkpoint optimizer state is corrupt")
    step = state["step"]
    if isinstance(step, Tensor):
        if (
            step.numel() != 1
            or step.dtype == torch.bool
            or not (
                step.dtype.is_floating_point
                or step.dtype
                in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            )
            or not bool(torch.isfinite(step).all())
        ):
            raise ValueError("checkpoint optimizer state is corrupt")
        step_value = float(step.detach().to(device="cpu").item())
    elif isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(float(step)):
        raise ValueError("checkpoint optimizer state is corrupt")
    else:
        step_value = float(step)
    if not step_value.is_integer() or int(step_value) != expected_step:
        raise ValueError("checkpoint optimizer step does not match completed update")
    for field in ("exp_avg", "exp_avg_sq"):
        value = state[field]
        if (
            not isinstance(value, Tensor)
            or value.shape != parameter.shape
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("checkpoint optimizer state is corrupt")


def _restore_rng_state(payload: Mapping[str, object]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    cpu_state = payload["torch_cpu"]
    assert isinstance(cpu_state, Tensor)
    torch.set_rng_state(cpu_state)
    cuda_states = payload["torch_cuda"]
    assert isinstance(cuda_states, tuple)
    if cuda_states:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(cuda_states):
            raise ValueError("checkpoint CUDA RNG device state is incompatible")
        torch.cuda.set_rng_state_all(list(cuda_states))
    elif torch.cuda.is_available():
        raise ValueError("checkpoint has no CUDA RNG state for the active CUDA environment")


def _event_indices(payload: Mapping[str, object], field: str) -> tuple[int, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in value
    ):
        raise ValueError(f"{field} must be a list of non-negative event indices")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} may not contain duplicate event indices")
    return tuple(value)


def _signature_mismatches(
    saved_signatures: Mapping[object, object], expected_signatures: Mapping[str, str]
) -> list[str]:
    return sorted(
        str(key)
        for key in set(saved_signatures) | set(expected_signatures)
        if saved_signatures.get(key) != expected_signatures.get(key)
    )


def _require_direct_child(root: Path, path: Path) -> None:
    if path.resolve().parent != root:
        raise ValueError("checkpoint path is outside the trusted local checkpoint directory")
