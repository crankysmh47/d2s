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
        self.encoded = [10, 11, 12]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        from transformers.tokenization_utils_base import BatchEncoding

        if kwargs.get("return_dict", True):
            return BatchEncoding({"input_ids": self.encoded})
        return self.encoded


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
    assert tokenizer.calls[0]["return_dict"] is False


@pytest.mark.parametrize(
    "encoded",
    [
        [],
        {"input_ids": [1, 2]},
        "12",
        [True],
        [-1],
        [1.0],
        ["1"],
        [[1], [2]],
        [1, [2]],
    ],
    ids=["empty", "mapping", "string", "bool", "negative", "float", "numeric-string", "two-batches", "nested-invalid"],
)
def test_prompt_rejects_malformed_token_id_outputs(encoded) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.encoded = encoded
    with pytest.raises(ValueError):
        build_prompt_ids(tokenizer, FakePromptConfig(), "2+5?")


@pytest.mark.parametrize(
    "encoded",
    [
        pytest.param([10, 11, 12], id="plain-list"),
        pytest.param([[10, 11, 12]], id="singleton-list-batch"),
        pytest.param(__import__("torch").tensor([10, 11, 12]), id="tensor"),
        pytest.param(__import__("torch").tensor([[10, 11, 12]]), id="tensor-singleton-batch"),
        pytest.param(__import__("numpy").array([10, 11, 12], dtype="int64"), id="numpy-integer-ids"),
    ],
)
def test_prompt_accepts_integer_id_sequence_after_supported_normalization(encoded) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.encoded = encoded
    assert build_prompt_ids(tokenizer, FakePromptConfig(), "2+5?") == [10, 11, 12]


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


def test_model_placement_accepts_actual_cuda0_tensors_without_hf_map() -> None:
    from types import SimpleNamespace

    class Placed:
        def named_parameters(self):
            return [("weight", SimpleNamespace(device="cuda:0", quant_state=None))]

        def named_buffers(self):
            return [("buffer", SimpleNamespace(device="cuda:0"))]

    assert validate_cuda_placement(Placed()) == {
        "parameter_count": 1, "buffer_count": 1, "quant_state_tensor_count": 0,
        "devices": ["cuda:0"],
    }


@pytest.mark.parametrize("device", ["cpu", "meta", None, "cuda:1"])
def test_model_placement_rejects_any_non_cuda0_actual_tensor(device) -> None:
    from types import SimpleNamespace

    class Placed:
        hf_device_map = {"model": 0}
        def named_parameters(self):
            return [("weight", SimpleNamespace(device="cuda:0", quant_state=None))]

        def named_buffers(self):
            return [("buffer", SimpleNamespace(device=device))]

    with pytest.raises(ValueError, match="CUDA device 0"):
        validate_cuda_placement(Placed())


def test_model_placement_rejects_nested_quant_state_tensor_on_other_device() -> None:
    from types import SimpleNamespace

    class Placed:
        def named_parameters(self):
            quant_state = SimpleNamespace(absmax=SimpleNamespace(device="cuda:0"))
            quant_state.state2 = SimpleNamespace(code=SimpleNamespace(device="cpu"))
            quant_state.offset = SimpleNamespace(device="cuda:0")
            return [("weight", SimpleNamespace(device="cuda:0", quant_state=quant_state))]

        def named_buffers(self):
            return []

    with pytest.raises(ValueError, match="CUDA device 0"):
        validate_cuda_placement(Placed())


def test_model_placement_requires_parameters_and_rejects_misleading_maps() -> None:
    from types import SimpleNamespace

    class Empty:
        hf_device_map = {"model": 0}
        def named_parameters(self): return []
        def named_buffers(self): return []

    with pytest.raises(ValueError, match="parameter"):
        validate_cuda_placement(Empty())

    class Misleading:
        hf_device_map = {"model": "cuda:1"}
        def named_parameters(self): return [("weight", SimpleNamespace(device="cuda:0"))]
        def named_buffers(self): return []

    with pytest.raises(ValueError, match="device map"):
        validate_cuda_placement(Misleading())


@pytest.mark.parametrize("device_map", [{"model": True}, {"model": "somewhere"}, {}, [0]])
def test_model_placement_rejects_malformed_or_unknown_present_device_map(device_map) -> None:
    from types import SimpleNamespace

    class Placed:
        hf_device_map = device_map
        def named_parameters(self): return [("weight", SimpleNamespace(device="cuda:0"))]
        def named_buffers(self): return []

    with pytest.raises(ValueError, match="device map"):
        validate_cuda_placement(Placed())


def test_load_worker_emits_bounded_placement_summary_after_validation(monkeypatch, tmp_path: Path, capsys) -> None:
    import sys
    import types
    from types import SimpleNamespace
    from ppmoe.distill.model import load_hf_worker

    class FakeTorch:
        bfloat16 = object()
        class cuda:
            @staticmethod
            def get_device_properties(_index): return SimpleNamespace(total_memory=16 * 1024**3)
            @staticmethod
            def max_memory_allocated(): return 10
            @staticmethod
            def max_memory_reserved(): return 20

    class Model:
        hf_device_map = {"model": 0}
        generation_config = SimpleNamespace(eos_token_id=2)
        training = False
        def eval(self): pass
        def named_parameters(self): return [("weight", SimpleNamespace(device="cuda:0", requires_grad=False, quant_state=None))]
        def named_buffers(self): return [("buffer", SimpleNamespace(device="cuda:0"))]

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs): return Model()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs): return FakeTokenizer()

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = AutoModel
    transformers.AutoTokenizer = AutoTokenizer
    transformers.BitsAndBytesConfig = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    teacher_dir = tmp_path / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/teacher-pin"
    teacher_dir.mkdir(parents=True)
    config = SimpleNamespace(
        teacher=SimpleNamespace(repo_id="teacher", revision="teacher-pin"),
        resources=SimpleNamespace(vram_reserve_bytes=2 * 1024**3),
        prompt=FakePromptConfig(),
    )

    load_hf_worker("teacher", config, repo_root=tmp_path, snapshot_path=teacher_dir)
    emitted = json.loads(capsys.readouterr().out.strip())
    assert emitted == {
        "event": "model_placement", "role": "teacher", "parameter_count": 1,
        "buffer_count": 1, "quant_state_tensor_count": 0, "devices": ["cuda:0"],
    }


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


def test_lfs_snapshot_uses_materialized_sha256_not_pointer_git_blob_id(tmp_path: Path) -> None:
    import hashlib
    from types import SimpleNamespace
    from ppmoe.distill.model import authenticate_model_snapshot

    revision = "b" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    weight = b"weight-payload"
    (snapshot / "weights.safetensors").write_bytes(weight)
    metadata = SimpleNamespace(
        sha=revision,
        siblings=[SimpleNamespace(
            rfilename="weights.safetensors", size=len(weight), blob_id="different-lfs-pointer-blob",
            lfs=SimpleNamespace(sha256=hashlib.sha256(weight).hexdigest()),
        )],
    )
    inventory = authenticate_model_snapshot("repo", revision, snapshot, metadata=metadata)
    assert inventory["files"][0]["sha256"] == hashlib.sha256(weight).hexdigest()
    (snapshot / "weights.safetensors").write_bytes(b"other-payload!")
    with pytest.raises(ValueError, match="LFS content hash"):
        authenticate_model_snapshot("repo", revision, snapshot, metadata=metadata)


def test_snapshot_auth_ignores_only_explicit_local_cache_metadata(tmp_path: Path) -> None:
    import hashlib
    from types import SimpleNamespace
    from ppmoe.distill.model import authenticate_model_snapshot

    revision = "e" * 40
    snapshot = tmp_path / revision
    (snapshot / ".cache" / "metadata").mkdir(parents=True)
    content = b"{}"
    (snapshot / "config.json").write_bytes(content)
    (snapshot / ".cache" / "metadata" / "download.json").write_text("local-only", encoding="utf-8")
    metadata = SimpleNamespace(
        sha=revision,
        siblings=[SimpleNamespace(rfilename="config.json", size=2, blob_id=hashlib.sha1(b"blob 2\0{}").hexdigest(), lfs=None)],
    )
    inventory = authenticate_model_snapshot(
        "repo", revision, snapshot, metadata=metadata, excluded_metadata_subtree=".cache",
    )
    assert [entry["path"] for entry in inventory["files"]] == ["config.json"]
    (snapshot / "rogue.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory"):
        authenticate_model_snapshot(
            "repo", revision, snapshot, metadata=metadata, excluded_metadata_subtree=".cache",
        )
    (snapshot / "rogue.bin").unlink()
    (snapshot / "nested" / ".cache").mkdir(parents=True)
    (snapshot / "nested" / ".cache" / "not-snapshot-metadata").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        authenticate_model_snapshot(
            "repo", revision, snapshot, metadata=metadata, excluded_metadata_subtree=".cache",
        )


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
