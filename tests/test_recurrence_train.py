from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from ppmoe.recurrence import train
from ppmoe.recurrence.data import Example, RenderedExample
from ppmoe.recurrence.lora import LoRALinear, adapter_sha256, adapter_state_dict
from ppmoe.recurrence.provenance import append_event, sha256_file, write_immutable_json
from ppmoe.recurrence.train import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
    train_arm,
    validate_checkpoint_artifact,
    write_completed_training_manifest,
)


class _AdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = LoRALinear(nn.Linear(2, 2, bias=False), rank=1, alpha=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class _TrainingFixture(nn.Module):
    """CPU-only answer-loss fixture exercising the real LoRA parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = LoRALinear(nn.Linear(2, 2, bias=False), rank=1, alpha=2)
        self.seen_ids: list[int] = []

    def answer_loss(
        self, input_ids: torch.Tensor, labels: torch.Tensor, mode: str
    ) -> torch.Tensor:
        if mode not in {"single", "repeat"}:
            raise ValueError("unexpected recurrence mode")
        example_id = int(input_ids[0, 0])
        self.seen_ids.append(example_id)
        features = torch.tensor(
            [[float(example_id % 7) / 7.0, float((example_id * 3) % 11) / 11.0]],
            dtype=torch.float32,
        )
        target = torch.tensor([[0.125, -0.25]], dtype=torch.float32)
        return (self.projection(features) - target).square().mean()


class _CudaFaultFixture(_TrainingFixture):
    def answer_loss(
        self, input_ids: torch.Tensor, labels: torch.Tensor, mode: str
    ) -> torch.Tensor:
        raise RuntimeError("CUDA out of memory in deterministic test fixture")


class _CudaFaultAfterOneFixture(_TrainingFixture):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def answer_loss(
        self, input_ids: torch.Tensor, labels: torch.Tensor, mode: str
    ) -> torch.Tensor:
        if self.calls >= 8:
            raise RuntimeError("CUDA out of memory after one committed update")
        self.calls += 1
        return super().answer_loss(input_ids, labels, mode)


def _make_training_fixture(seed: int = 917) -> _TrainingFixture:
    torch.manual_seed(seed)
    return _TrainingFixture()


def _training_examples(count: int) -> list[RenderedExample]:
    rows: list[RenderedExample] = []
    for index in range(count):
        example = Example(
            id=f"train-{index:04d}",
            family="expressions",
            split="train",
            structure_id=f"shape-{index}",
            prompt=f"example {index}",
            answer=str(index),
            latent_spec={},
        )
        rows.append(
            RenderedExample(
                example=example,
                prompt_input_ids=(index,),
                train_input_ids=(index, index + 1),
                labels=(-100, index + 1),
            )
        )
    return rows


def _training_signatures(model: nn.Module, arm: str, seed: int) -> dict[str, str]:
    return {
        "source": "source-v1",
        "data": "data-v1",
        "config": "config-v1",
        "code": "code-v1",
        "arm": arm,
        "seed": str(seed),
        "initialization_hash": adapter_sha256(model),
    }


def _make_model_and_optimizer() -> tuple[_AdapterModel, torch.optim.AdamW]:
    torch.manual_seed(917)
    model = _AdapterModel()
    optimizer = torch.optim.AdamW(
        [model.projection.a, model.projection.b],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    return model, optimizer


def _step(model: _AdapterModel, optimizer: torch.optim.AdamW) -> None:
    optimizer.zero_grad(set_to_none=True)
    model(torch.tensor([[1.0, -2.0]])).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step = int(next(iter(optimizer.state.values()))["step"])
    optimizer.param_groups[0]["lr"] = train._learning_rate_for_update(step)


def _signatures(model: _AdapterModel) -> dict[str, str]:
    return {
        "source": "source-v1",
        "data": "data-v1",
        "config": "config-v1",
        "code": "code-v1",
        "arm": "a2",
        "seed": "47",
        "initialization_hash": adapter_sha256(model),
    }


def _state(update: int) -> TrainingState:
    return TrainingState(
        update=update,
        cursor=update * 8,
        elapsed_cuda_seconds=float(update) / 10,
        attempt=1,
    )


def _assert_tree_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for member, expected_member in zip(actual, expected, strict=True):
            _assert_tree_equal(member, expected_member)
    else:
        assert actual == expected


def _live_recovery_state(
    model: _AdapterModel, optimizer: torch.optim.AdamW
) -> dict[str, object]:
    return {
        "adapters": {
            name: tensor.detach().clone() for name, tensor in adapter_state_dict(model).items()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    }


def _assert_live_recovery_state(
    model: _AdapterModel, optimizer: torch.optim.AdamW, expected: dict[str, object]
) -> None:
    _assert_tree_equal(
        {name: tensor.detach() for name, tensor in adapter_state_dict(model).items()},
        expected["adapters"],
    )
    _assert_tree_equal(optimizer.state_dict(), expected["optimizer"])
    _assert_tree_equal(random.getstate(), expected["python"])
    _assert_tree_equal(np.random.get_state(), expected["numpy"])
    _assert_tree_equal(torch.get_rng_state(), expected["torch_cpu"])
    current_cuda = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    _assert_tree_equal(current_cuda, expected["torch_cuda"])


def _rewrite_checkpoint_with_valid_sidecar(
    directory: Path, signatures: dict[str, str], mutate: Any
) -> None:
    checkpoint = directory / "latest.pt"
    payload = torch.load(checkpoint, weights_only=False)
    mutate(payload)
    torch.save(payload, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    write_immutable_json(
        directory / f"checkpoint-{checkpoint_hash}.json",
        {
            "schema_version": 1,
            "checkpoint_sha256": checkpoint_hash,
            "signatures": signatures,
        },
    )


def test_checkpoint_persists_complete_recovery_state_and_preserves_previous(tmp_path: Path) -> None:
    """Catches a checkpoint missing state needed for exact post-step resume."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    first_adapter_hash = adapter_sha256(model)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)

    first_payload = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert set(first_payload) == {
        "schema_version",
        "adapter_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "state",
        "initialization_hash",
        "signatures",
    }
    assert set(first_payload["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert first_payload["scheduler_state"] == {"last_completed_update": 1}
    assert first_payload["state"] == {
        "update": 1,
        "cursor": 8,
        "elapsed_cuda_seconds": 0.1,
        "attempt": 1,
    }
    assert first_payload["initialization_hash"] == signatures["initialization_hash"]
    assert first_payload["signatures"] == signatures
    assert set(first_payload["adapter_state"]) == set(adapter_state_dict(model))
    _assert_tree_equal(first_payload["optimizer_state"], optimizer.state_dict())

    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(2), signatures)

    assert (tmp_path / "previous.pt").is_file()
    previous = torch.load(tmp_path / "previous.pt", weights_only=False)
    assert previous["state"]["update"] == 1
    assert adapter_sha256(model) != first_adapter_hash
    sidecars = list(tmp_path.glob("checkpoint-*.json"))
    assert len(sidecars) == 2
    for sidecar in sidecars:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        assert manifest["checkpoint_sha256"] == sidecar.stem.removeprefix("checkpoint-")
        assert manifest["signatures"] == signatures


def test_parent_checkpoint_artifact_validator_binds_manifest_and_failure_state(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)
    artifact = validate_checkpoint_artifact(
        tmp_path, signatures, expected_attempt=1, expected_update=1
    )
    assert artifact["attempt"] == 1
    assert artifact["update"] == 1
    assert len(artifact["manifest_digest"]) == 64
    (tmp_path / "latest.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sidecar|checkpoint"):
        validate_checkpoint_artifact(tmp_path, signatures, expected_attempt=1, expected_update=1)


@pytest.mark.parametrize("corruption", ["adapter", "optimizer", "step", "group"])
def test_parent_checkpoint_validator_rejects_manifest_bound_trajectory_corruption(
    tmp_path: Path, corruption: str
) -> None:
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)

    def mutate(payload: dict[str, object]) -> None:
        if corruption == "adapter":
            adapters = payload["adapter_state"]
            assert isinstance(adapters, dict)
            first = next(iter(adapters))
            adapters[first] = adapters[first].reshape(-1)
        elif corruption == "optimizer":
            payload["optimizer_state"] = {"state": {}, "param_groups": []}
        elif corruption == "step":
            states = payload["optimizer_state"]["state"]
            first = next(iter(states.values()))
            first["step"] = torch.tensor(0.0)
        else:
            payload["optimizer_state"]["param_groups"][0]["lr"] = 0.5

    _rewrite_checkpoint_with_valid_sidecar(tmp_path, signatures, mutate)
    with pytest.raises(ValueError, match="manifest|optimizer|adapter"):
        validate_checkpoint_artifact(tmp_path, signatures, expected_attempt=1, expected_update=1)


@pytest.mark.parametrize("mutation", ["renamed", "float64"])
def test_parent_checkpoint_validator_requires_external_registered_adapter_schema(
    tmp_path: Path, mutation: str
) -> None:
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    expected_schema = {
        "parameters": {
            name: {"shape": [int(dimension) for dimension in value.shape], "dtype": str(value.dtype)}
            for name, value in adapter_state_dict(model).items()
        },
        "initialization_hash": signatures["initialization_hash"],
    }
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)
    payload = torch.load(tmp_path / "latest.pt", weights_only=False)
    adapters = payload["adapter_state"]
    if mutation == "renamed":
        payload["adapter_state"] = {f"evil.{name}": value for name, value in adapters.items()}
    else:
        first = next(iter(adapters))
        adapters[first] = adapters[first].to(dtype=torch.float64)
    temporary = train._write_torch_temporary(tmp_path, payload)
    checkpoint_hash = sha256_file(temporary)
    train._write_checkpoint_sidecar(tmp_path, checkpoint_hash, signatures)
    train._write_checkpoint_manifest(tmp_path, checkpoint_hash, payload, signatures)
    os.replace(temporary, tmp_path / "latest.pt")
    with pytest.raises(ValueError, match="schema|registered"):
        validate_checkpoint_artifact(
            tmp_path, signatures, expected_attempt=1, expected_update=1,
            expected_adapter_schema=expected_schema,
        )


def test_failed_latest_replacement_leaves_complete_checkpoint_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a failed rename destroying the last known complete checkpoint."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)
    original_hash = sha256_file(tmp_path / "latest.pt")

    _step(model, optimizer)
    real_replace = train.os.replace

    def fail_only_latest(source: object, destination: object) -> None:
        if Path(destination) == tmp_path / "latest.pt":
            raise OSError("injected final rename failure")
        real_replace(source, destination)

    monkeypatch.setattr(train.os, "replace", fail_only_latest)
    with pytest.raises(OSError, match="injected final rename failure"):
        save_checkpoint(tmp_path, model, optimizer, _state(2), signatures)

    assert sha256_file(tmp_path / "latest.pt") == original_hash
    resumed_model, resumed_optimizer = _make_model_and_optimizer()
    assert load_checkpoint(tmp_path, resumed_model, resumed_optimizer, signatures) == _state(1)


def test_checkpoint_rejects_missing_initialization_hash(tmp_path: Path) -> None:
    """Catches recording a post-update adapter hash as if it were initialization evidence."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    signatures.pop("initialization_hash")

    with pytest.raises(ValueError, match="initialization_hash"):
        save_checkpoint(tmp_path, model, optimizer, _state(0), signatures)


@pytest.mark.parametrize("changed", ["source", "data", "config", "code", "arm", "seed"])
def test_signature_mismatch_rejects_before_mutating_model_or_optimizer(
    tmp_path: Path, changed: str
) -> None:
    """Catches resuming an incompatible arm and silently corrupting its state."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)

    target_model, target_optimizer = _make_model_and_optimizer()
    target_adapters = {
        name: tensor.detach().clone() for name, tensor in adapter_state_dict(target_model).items()
    }
    target_optimizer_state = copy.deepcopy(target_optimizer.state_dict())
    incompatible = dict(signatures)
    incompatible[changed] = f"different-{changed}"

    with pytest.raises(ValueError, match=changed):
        load_checkpoint(tmp_path, target_model, target_optimizer, incompatible)

    for name, expected in target_adapters.items():
        assert torch.equal(adapter_state_dict(target_model)[name], expected)
    _assert_tree_equal(target_optimizer.state_dict(), target_optimizer_state)


def test_corrupt_checkpoint_is_rejected_before_state_restore(tmp_path: Path) -> None:
    """Catches accepting a modified checkpoint whose immutable hash no longer agrees."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)
    (tmp_path / "latest.pt").write_bytes((tmp_path / "latest.pt").read_bytes() + b"corrupt")

    target_model, target_optimizer = _make_model_and_optimizer()
    with pytest.raises(ValueError, match="SHA-256|corrupt"):
        load_checkpoint(tmp_path, target_model, target_optimizer, signatures)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("optimizer", "optimizer"),
        ("python_rng", "RNG"),
        ("numpy_rng", "RNG"),
        ("cpu_rng", "RNG"),
        ("cuda_rng", "CUDA RNG"),
        ("scheduler", "scheduler"),
    ],
)
def test_sidecar_valid_malformed_recovery_payload_leaves_all_live_state_unchanged(
    tmp_path: Path, corruption: str, message: str
) -> None:
    """Catches a late checkpoint validation failure after any live restore mutation."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)

    def mutate(payload: dict[str, object]) -> None:
        if corruption == "optimizer":
            optimizer_payload = payload["optimizer_state"]
            assert isinstance(optimizer_payload, dict)
            optimizer_payload["param_groups"] = []
        elif corruption == "python_rng":
            rng_payload = payload["rng_state"]
            assert isinstance(rng_payload, dict)
            rng_payload["python"] = ("not-a-python-rng-state",)
        elif corruption == "numpy_rng":
            rng_payload = payload["rng_state"]
            assert isinstance(rng_payload, dict)
            rng_payload["numpy"] = ("not-a-numpy-rng-state",)
        elif corruption == "cpu_rng":
            rng_payload = payload["rng_state"]
            assert isinstance(rng_payload, dict)
            rng_payload["torch_cpu"] = torch.tensor([1], dtype=torch.uint8)
        elif corruption == "cuda_rng":
            rng_payload = payload["rng_state"]
            assert isinstance(rng_payload, dict)
            count = torch.cuda.device_count() + 1
            rng_payload["torch_cuda"] = tuple(
                torch.tensor([1], dtype=torch.uint8) for _ in range(count)
            )
        else:
            scheduler_payload = payload["scheduler_state"]
            assert isinstance(scheduler_payload, dict)
            scheduler_payload["last_completed_update"] = True

    _rewrite_checkpoint_with_valid_sidecar(tmp_path, signatures, mutate)
    target_model, target_optimizer = _make_model_and_optimizer()
    expected = _live_recovery_state(target_model, target_optimizer)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(tmp_path, target_model, target_optimizer, signatures)

    _assert_live_recovery_state(target_model, target_optimizer, expected)


def test_sidecar_valid_inconsistent_cursor_is_rejected_before_live_restore(
    tmp_path: Path,
) -> None:
    """Catches a committed update replaying the wrong presentation cursor."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)

    def mutate(payload: dict[str, object]) -> None:
        state = payload["state"]
        scheduler = payload["scheduler_state"]
        assert isinstance(state, dict)
        assert isinstance(scheduler, dict)
        state["update"] = 2
        state["cursor"] = 8
        scheduler["last_completed_update"] = 2

    _rewrite_checkpoint_with_valid_sidecar(tmp_path, signatures, mutate)
    target_model, target_optimizer = _make_model_and_optimizer()
    expected = _live_recovery_state(target_model, target_optimizer)
    with pytest.raises(ValueError, match="cursor|training state"):
        load_checkpoint(tmp_path, target_model, target_optimizer, signatures)
    _assert_live_recovery_state(target_model, target_optimizer, expected)


def test_warmup_checkpoint_lr_is_validated_against_saved_update(
    tmp_path: Path,
) -> None:
    """Catches rejecting a legitimate scheduled LR during warmup resume."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    assert optimizer.param_groups[0]["lr"] == train._learning_rate_for_update(1)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)

    target_model, target_optimizer = _make_model_and_optimizer()
    assert load_checkpoint(tmp_path, target_model, target_optimizer, signatures) == _state(1)
    assert target_optimizer.param_groups[0]["lr"] == train._learning_rate_for_update(1)


def test_sidecar_valid_tampered_warmup_lr_is_rejected_before_live_restore(
    tmp_path: Path,
) -> None:
    """Catches accepting a checkpoint whose LR does not match its update."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)

    def mutate(payload: dict[str, object]) -> None:
        optimizer_payload = payload["optimizer_state"]
        assert isinstance(optimizer_payload, dict)
        groups = optimizer_payload["param_groups"]
        assert isinstance(groups, list)
        groups[0]["lr"] = 0.0001

    _rewrite_checkpoint_with_valid_sidecar(tmp_path, signatures, mutate)
    target_model, target_optimizer = _make_model_and_optimizer()
    expected = _live_recovery_state(target_model, target_optimizer)
    with pytest.raises(ValueError, match="learning rate|optimizer"):
        load_checkpoint(tmp_path, target_model, target_optimizer, signatures)
    _assert_live_recovery_state(target_model, target_optimizer, expected)


@pytest.mark.parametrize("corruption", ["missing_one", "missing_all", "wrong_step"])
def test_sidecar_valid_optimizer_trajectory_corruption_is_rejected_before_restore(
    tmp_path: Path, corruption: str
) -> None:
    """Catches optimizer-state omissions or counters changing resume math."""
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)

    def mutate(payload: dict[str, object]) -> None:
        optimizer_payload = payload["optimizer_state"]
        assert isinstance(optimizer_payload, dict)
        states = optimizer_payload["state"]
        assert isinstance(states, dict)
        if corruption == "missing_all":
            states.clear()
        elif corruption == "missing_one":
            states.pop(next(iter(states)))
        else:
            state = next(iter(states.values()))
            assert isinstance(state, dict)
            state["step"] = 0

    _rewrite_checkpoint_with_valid_sidecar(tmp_path, signatures, mutate)
    target_model, target_optimizer = _make_model_and_optimizer()
    expected = _live_recovery_state(target_model, target_optimizer)
    with pytest.raises(ValueError, match="optimizer|step"):
        load_checkpoint(tmp_path, target_model, target_optimizer, signatures)
    _assert_live_recovery_state(target_model, target_optimizer, expected)


def test_load_restores_rng_after_validating_checkpoint_state(tmp_path: Path) -> None:
    """Catches a resumed run taking a different random path despite a valid checkpoint."""
    random.seed(47)
    np.random.seed(47)
    torch.manual_seed(47)
    model, optimizer = _make_model_and_optimizer()
    signatures = _signatures(model)
    _step(model, optimizer)
    save_checkpoint(tmp_path, model, optimizer, _state(1), signatures)
    payload = torch.load(tmp_path / "latest.pt", weights_only=False)
    random.random()
    np.random.random()
    torch.rand(1)

    target_model, target_optimizer = _make_model_and_optimizer()
    load_checkpoint(tmp_path, target_model, target_optimizer, signatures)

    assert random.getstate() == payload["rng_state"]["python"]
    np_state = np.random.get_state()
    assert np_state[0] == payload["rng_state"]["numpy"][0]
    assert np.array_equal(np_state[1], payload["rng_state"]["numpy"][1])
    assert np_state[2:] == payload["rng_state"]["numpy"][2:]
    assert torch.equal(torch.get_rng_state(), payload["rng_state"]["torch_cpu"])


def test_shared_initialization_snapshots_are_immutable_and_cross_arm_reusable(
    tmp_path: Path,
) -> None:
    """Catches pilot mutations or per-arm initialization drift."""
    seed47 = _make_training_fixture(917)
    seed48 = _make_training_fixture(918)
    path47 = tmp_path / "seed-47.pt"
    path48 = tmp_path / "seed-48.pt"
    hash47 = train.capture_initialization(path47, seed47, 47)
    hash48 = train.capture_initialization(path48, seed48, 48)
    assert hash47 == adapter_sha256(seed47)
    assert hash48 == adapter_sha256(seed48)
    assert hash47 != hash48

    for arm in ("a1", "a2", "a3"):
        target = _make_training_fixture(917)
        with torch.no_grad():
            for parameter in adapter_state_dict(target).values():
                parameter.add_(1.0)
        assert adapter_sha256(target) != hash47
        assert train.load_initialization(path47, target, expected_seed=47) == hash47
        assert adapter_sha256(target) == hash47

    with pytest.raises(FileExistsError, match="immutable"):
        train.capture_initialization(path47, _make_training_fixture(918), 47)


def test_failure_history_stays_append_only_when_completed_manifest_is_written(tmp_path: Path) -> None:
    """Catches a successful retry erasing the failure evidence it resolved."""
    event_path = tmp_path / "events.jsonl"
    failed_index = append_event(
        event_path,
        {"kind": "training_failure", "attempt": 1, "message": "CUDA fault"},
    )
    completed_index = append_event(
        event_path,
        {"kind": "training_completed", "attempt": 2, "update": 1000},
    )
    manifest_path = tmp_path / "training-complete.json"
    write_completed_training_manifest(
        manifest_path,
        {
            "status": "passed",
            "attempts": [1, 2],
            "failure_event_indices": [failed_index],
            "event_indices": [failed_index, completed_index],
        },
    )

    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [event["attempt"] for event in events] == [1, 2]
    assert events[0]["message"] == "CUDA fault"
    assert manifest["event_indices"] == [0, 1]
    assert manifest["failure_event_indices"] == [0]
    with pytest.raises(FileExistsError, match="immutable"):
        write_completed_training_manifest(
            manifest_path,
            {"status": "passed", "event_indices": [], "failure_event_indices": []},
        )


def test_registered_adapter_optimizer_and_warmup_schedule() -> None:
    """Catches dense weights or a non-protocol AdamW/warmup schedule entering training."""
    model = _make_training_fixture()

    optimizer = train._make_adapter_optimizer(model)

    assert tuple(optimizer.param_groups[0]["params"]) == tuple(adapter_state_dict(model).values())
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert [train._learning_rate_for_update(update) for update in (1, 10, 20, 21)] == [
        5e-6,
        5e-5,
        1e-4,
        1e-4,
    ]


def test_train_arm_accumulates_exactly_eight_mean_losses_in_seeded_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches stepping early, using a token-weighted loss, or reshuffling A1 data."""
    examples = _training_examples(8)
    monkeypatch.setattr(train, "_REGISTERED_TRAIN_EXAMPLES", len(examples))
    seed = 47
    trained = _make_training_fixture()
    signatures = _training_signatures(trained, "a1", seed)

    result = train_arm(
        trained,
        examples,
        "a1",
        seed,
        1,
        tmp_path / "trained",
        tmp_path / "trained-events.jsonl",
        signatures,
    )

    reference = _make_training_fixture()
    reference_optimizer = torch.optim.AdamW(
        list(adapter_state_dict(reference).values()),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    expected_order = list(range(8))
    random.Random(seed).shuffle(expected_order)
    for index in expected_order:
        rendered = examples[index]
        reference.answer_loss(
            torch.tensor([rendered.train_input_ids], dtype=torch.long),
            torch.tensor([rendered.labels], dtype=torch.long),
            "single",
        ).div(8).backward()
    torch.nn.utils.clip_grad_norm_(list(adapter_state_dict(reference).values()), 1.0)
    reference_optimizer.param_groups[0]["lr"] = 5e-6
    reference_optimizer.step()
    reference_optimizer.zero_grad(set_to_none=True)

    assert trained.seen_ids == expected_order
    for name, expected in adapter_state_dict(reference).items():
        assert torch.equal(adapter_state_dict(trained)[name], expected)
    assert result["completed_updates"] == 1
    assert result["unique_example_count"] == 8
    assert result["total_example_presentations"] == 8
    assert result["ordered_id_sha256"] == train._ordered_id_hash(examples, seed)
    assert result["completion_status"] == "completed"


@pytest.mark.parametrize(
    ("arm", "example_count", "steps"),
    [("a1", 7, 1), ("a2", 8, 2), ("a4", 8, 1)],
)
def test_train_arm_rejects_non_protocol_accumulation_or_arm(
    tmp_path: Path, arm: str, example_count: int, steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches A1/A2 overshoot or an unregistered training condition."""
    model = _make_training_fixture()
    monkeypatch.setattr(train, "_REGISTERED_TRAIN_EXAMPLES", example_count)
    with pytest.raises(ValueError):
        train_arm(
            model,
            _training_examples(example_count),
            arm,
            47,
            steps,
            tmp_path / "checkpoints",
            tmp_path / "events.jsonl",
            _training_signatures(model, arm, 47),
        )


def test_train_arm_rejects_non_registered_production_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches silently training a prefix of a non-8,000-example pool."""
    monkeypatch.setattr(train, "_REGISTERED_TRAIN_EXAMPLES", 8_000)
    model = _make_training_fixture()
    examples = _training_examples(8)
    with pytest.raises(ValueError, match="8,000"):
        train_arm(
            model,
            examples,
            "a1",
            47,
            1,
            tmp_path / "checkpoints",
            tmp_path / "events.jsonl",
            _training_signatures(model, "a1", 47),
        )


def test_a3_reuses_one_seeded_order_and_cycles_without_reshuffling() -> None:
    """Catches A3 drawing a second permutation or an example outside its training pool."""
    examples = _training_examples(8_000)
    order = train._presentation_indices(examples, "a3", 47, 1_001)
    expected_first_epoch = list(range(8_000))
    random.Random(47).shuffle(expected_first_epoch)

    assert len(order) == 8_008
    assert order[:8_000] == expected_first_epoch
    assert order[8_000:] == expected_first_epoch[:8]
    assert set(order) == set(range(8_000))


def test_train_arm_resume_is_bitwise_equivalent_to_uninterrupted_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches replaying a committed update or losing optimizer/RNG state on resume."""
    examples = _training_examples(32)
    monkeypatch.setattr(train, "_REGISTERED_TRAIN_EXAMPLES", len(examples))
    seed = 47

    uninterrupted = _make_training_fixture()
    uninterrupted_signatures = _training_signatures(uninterrupted, "a2", seed)
    uninterrupted_result = train_arm(
        uninterrupted,
        examples,
        "a2",
        seed,
        4,
        tmp_path / "uninterrupted",
        tmp_path / "uninterrupted-events.jsonl",
        uninterrupted_signatures,
    )
    uninterrupted_payload = torch.load(
        tmp_path / "uninterrupted" / "latest.pt", weights_only=False
    )

    partial = _make_training_fixture()
    partial_signatures = _training_signatures(partial, "a2", seed)
    train_arm(
        partial,
        examples,
        "a2",
        seed,
        2,
        tmp_path / "resumed",
        tmp_path / "resumed-events.jsonl",
        partial_signatures,
    )
    resumed = _make_training_fixture()
    resumed_result = train_arm(
        resumed,
        examples,
        "a2",
        seed,
        4,
        tmp_path / "resumed",
        tmp_path / "resumed-events.jsonl",
        _training_signatures(resumed, "a2", seed),
    )
    resumed_payload = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=False)

    assert uninterrupted_result["completed_updates"] == resumed_result["completed_updates"] == 4
    assert uninterrupted_result["cursor"] == resumed_result["cursor"] == 32
    _assert_tree_equal(uninterrupted_payload["adapter_state"], resumed_payload["adapter_state"])
    _assert_tree_equal(uninterrupted_payload["optimizer_state"], resumed_payload["optimizer_state"])
    _assert_tree_equal(uninterrupted_payload["rng_state"], resumed_payload["rng_state"])


def test_train_arm_manifest_records_all_recoverable_training_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a completed arm omitting evidence needed by later timing and gate stages."""
    model = _make_training_fixture()
    monkeypatch.setattr(train, "_REGISTERED_TRAIN_EXAMPLES", 8)
    result = train_arm(
        model,
        _training_examples(8),
        "a3",
        48,
        1,
        tmp_path / "checkpoints",
        tmp_path / "events.jsonl",
        _training_signatures(model, "a3", 48),
    )

    assert set(result) >= {
        "arm",
        "seed",
        "requested_updates",
        "completed_updates",
        "cursor",
        "unique_example_count",
        "total_example_presentations",
        "ordered_id_sha256",
        "initialization_hash",
        "final_adapter_hash",
        "trainable_parameter_count",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "synchronized_cuda_seconds",
        "checkpoint_hashes",
        "attempts",
        "failures",
        "completion_status",
    }
    assert result["arm"] == "a3"
    assert result["seed"] == 48
    assert result["checkpoint_hashes"] == [sha256_file(tmp_path / "checkpoints" / "latest.pt")]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["kind"] == "training_completed"
    assert events[-1]["update"] == 1


def test_cuda_fault_requires_one_fresh_process_then_records_terminal_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches repeated CUDA faults being retried forever or in-process."""
    examples = _training_examples(8)
    monkeypatch.setattr(train, "_REGISTERED_TRAIN_EXAMPLES", len(examples))
    event_path = tmp_path / "events.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"

    first = _CudaFaultFixture()
    first_signatures = _training_signatures(first, "a1", 47)
    with pytest.raises(train.FreshProcessRequired, match="fresh process"):
        train_arm(
            first,
            examples,
            "a1",
            47,
            1,
            checkpoint_dir,
            event_path,
            first_signatures,
        )

    second = _CudaFaultFixture()
    second_signatures = _training_signatures(second, "a1", 47)
    with pytest.raises(train.FailedEnvironment, match="failed_environment"):
        train_arm(
            second,
            examples,
            "a1",
            47,
            1,
            checkpoint_dir,
            event_path,
            second_signatures,
        )

    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    failures = [event for event in events if event["kind"] == "training_failure"]
    assert [event["attempt"] for event in failures] == [1, 2]
    assert failures[0]["requires_fresh_process"] is True
    assert failures[1]["retry_status"] == "terminal"
    assert events[-1]["kind"] == "failed_environment"
    assert events[-1]["completion_status"] == "failed_environment"

    third = _CudaFaultFixture()
    with pytest.raises(train.FailedEnvironment, match="terminal"):
        train_arm(
            third,
            examples,
            "a1",
            47,
            1,
            checkpoint_dir,
            event_path,
            _training_signatures(third, "a1", 47),
        )
    final_events = event_path.read_text(encoding="utf-8").splitlines()
    assert len(final_events) == len(events)


def test_cuda_fault_cost_counts_only_current_uncommitted_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches cumulative committed time being charged again on a fault."""
    examples = _training_examples(8)
    monkeypatch.setattr(train, "_REGISTERED_TRAIN_EXAMPLES", len(examples))
    # Keep an explicit attempt-start timestamp as well as update-1 start,
    # update-2 start, and the fault boundary.  CPU execution does not consume
    # the attempt-start value, but the four-value sequence also matches the
    # CUDA-timing path.
    clock = iter((0.0, 10.0, 20.0, 23.0))
    monkeypatch.setattr(train.time, "perf_counter", lambda: next(clock))
    model = _CudaFaultAfterOneFixture()
    with pytest.raises(train.FreshProcessRequired):
        train_arm(
            model,
            examples,
            "a3",
            47,
            2,
            tmp_path / "checkpoints",
            tmp_path / "events.jsonl",
            _training_signatures(model, "a3", 47),
        )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    failure = next(event for event in events if event["kind"] == "training_failure")
    assert failure["update"] == 1
    assert failure["chargeable_cuda_seconds"] == 23.0 - 20.0
