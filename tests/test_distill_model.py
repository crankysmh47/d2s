from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from ppmoe.distill.model import (
    GenerationBudgetError,
    build_prompt_ids,
    generate_one,
    generation_stop_reason,
    pinned_eos_token_ids,
    verify_student_is_frozen,
    candidate_limits,
    validate_cuda_memory,
    validate_cuda_placement,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return [10, 11, 12]


class FakeParam:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, *, trainable: bool = False) -> None:
        self._parameters = {"base": FakeParam(trainable)}

    def named_parameters(self):
        return self._parameters.items()

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return {"text": "Final answer: 7", "stop_reason": "eos", "generated_tokens": 4}


@dataclass(frozen=True)
class FakePromptConfig:
    template: str = "Solve: {question}"
    thinking: bool = False


def test_prompt_uses_chat_template_once_without_duplicate_special_tokens() -> None:
    tokenizer = FakeTokenizer()
    ids = build_prompt_ids(tokenizer, FakePromptConfig(), "2+5?")
    assert ids == [10, 11, 12]
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0]["tokenize"] is True
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[0]["enable_thinking"] is False


def test_prompt_overflow_is_rejected_before_generation() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    ids = build_prompt_ids(tokenizer, FakePromptConfig(), "2+5?")
    with pytest.raises(GenerationBudgetError, match="prompt.*budget"):
        generate_one(model, ids, max_prompt_tokens=2, max_new_tokens=3, candidate_total=5)
    assert not hasattr(model, "generation_kwargs")


def test_generation_uses_greedy_settings_and_no_prompt_truncation() -> None:
    model = FakeModel()
    result = generate_one(model, [10, 11], max_prompt_tokens=2, max_new_tokens=4, candidate_total=6)
    assert result.text == "Final answer: 7"
    assert model.generation_kwargs["do_sample"] is False
    assert model.generation_kwargs["use_cache"] is True
    assert model.generation_kwargs["max_new_tokens"] == 4
    assert "truncation" not in model.generation_kwargs


def test_candidate_pair_means_new_tokens_then_total_and_derives_prompt_capacity() -> None:
    assert candidate_limits((256, 768)) == (512, 256, 768)
    assert candidate_limits((512, 1024)) == (512, 512, 1024)
    assert candidate_limits((768, 1536)) == (768, 768, 1536)


def test_memory_cap_checks_allocated_and_reserved_peaks() -> None:
    validate_cuda_memory(allocated=10, reserved=20, memory_cap=20)
    with pytest.raises(ValueError, match="allocated"):
        validate_cuda_memory(allocated=21, reserved=22, memory_cap=20)
    with pytest.raises(ValueError, match="reserved"):
        validate_cuda_memory(allocated=10, reserved=21, memory_cap=20)


def test_model_placement_rejects_cpu_or_disk_offload() -> None:
    class Placed:
        hf_device_map = {"model.embed_tokens": 0, "lm_head": "cpu"}

    with pytest.raises(ValueError, match="offload"):
        validate_cuda_placement(Placed())


def test_student_base_must_have_no_trainable_parameters() -> None:
    verify_student_is_frozen(FakeModel())
    with pytest.raises(ValueError, match="trainable"):
        verify_student_is_frozen(FakeModel(trainable=True))


def test_effective_alternate_eos_wins_at_exact_generation_budget_boundary() -> None:
    assert generation_stop_reason([10, 11, 99], [2, 99], max_new_tokens=3) == "eos"
    assert generation_stop_reason([10, 11, 12], [2, 99], max_new_tokens=3) == "length"
    assert generation_stop_reason([99], 99, max_new_tokens=3) == "eos"


def test_pinned_generation_config_accepts_multiple_effective_eos_ids(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [9, 3, 9]}), encoding="utf-8")
    assert pinned_eos_token_ids(tmp_path) == (3, 9)


def test_snapshot_authentication_checks_hub_git_blob_sha_for_non_lfs_files(tmp_path: Path) -> None:
    import hashlib
    from types import SimpleNamespace
    from ppmoe.distill.model import authenticate_model_snapshot

    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    data = b"one"
    (snapshot / "config.json").write_bytes(data)
    blob = hashlib.sha1(b"blob 3\0one").hexdigest()
    metadata = SimpleNamespace(
        sha=revision,
        siblings=[SimpleNamespace(rfilename="config.json", size=3, blob_id=blob, lfs=None)],
    )
    assert authenticate_model_snapshot("repo", revision, snapshot, metadata=metadata)["files"][0]["path"] == "config.json"
    (snapshot / "config.json").write_bytes(b"two")
    with pytest.raises(ValueError, match="blob"):
        authenticate_model_snapshot("repo", revision, snapshot, metadata=metadata)


@pytest.mark.parametrize(("alive_after_terminate", "error_type"), [(False, RuntimeError), (True, "ownership")])
def test_startup_timeout_terminates_owned_child_or_reports_uncertain_ownership(monkeypatch, alive_after_terminate, error_type) -> None:
    import ppmoe.distill.model as model_module

    callbacks = []
    class Queue:
        def get(self, timeout): raise TimeoutError
        def close(self): pass
    class Process:
        pid = 321
        def __init__(self, *args, **kwargs):
            self.alive = True
            self.terminated = False

        def start(self): pass

        def join(self, timeout):
            if self.terminated and not alive_after_terminate:
                self.alive = False

        def is_alive(self): return self.alive
        def terminate(self): self.terminated = True
    class Context:
        def Queue(self): return Queue()
        def Process(self, *args, **kwargs):
            self.process = Process(*args, **kwargs)
            return self.process
    context = Context()
    monkeypatch.setattr(model_module.multiprocessing, "get_context", lambda name: context)
    args = dict(role="student", config=object(), repo_root=Path("."), snapshot_path=Path("."),
                tokenizer_snapshot_path=Path("."), eos_token_ids=(2,), timeout=0.01,
                child_started=callbacks.append)
    if error_type == "ownership":
        with pytest.raises(model_module.WorkerOwnershipError):
            model_module.IsolatedModelWorker(**args)
    else:
        with pytest.raises(RuntimeError, match="startup timed out"):
            model_module.IsolatedModelWorker(**args)
    assert callbacks == [321]
    assert context.process.terminated
