import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from ppmoe.config import Phase1Config
from ppmoe.adapters.qwen35_runtime import validate_loaded_qwen35
from ppmoe.experiments.phase1_v0 import (
    Phase1Dependencies,
    StagePrerequisiteError,
    run_phase1,
)
import ppmoe.experiments.phase1_v0 as phase1_v0


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "phase1-v0.toml"


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [1] * int(text)


def _dependencies() -> Phase1Dependencies:
    rows = [{"id": f"fine-{index}", "text": "100"} for index in range(2_100)]
    return Phase1Dependencies(
        runtime_probe=lambda: {
            "branch": "exp",
            "cuda_available": True,
            "bf16_available": True,
            "free_disk_bytes": 100 * 1024**3,
            "python": "test",
            "torch": "test",
            "transformers": "test",
            "datasets": "test",
        },
        remote_probe=lambda config: {
            "model_revision": config.model.revision,
            "fineweb_revision": config.fineweb.revision,
            "wikitext_revision": config.wikitext.revision,
            "expected_weight_bytes": 123,
            "expected_repository_bytes": 456,
        },
        enforce_locked_model_dimensions=False,
        tokenizer_loader=lambda *_args, **_kwargs: TinyTokenizer(),
        fineweb_loader=lambda _spec: list(rows),
        wikitext_loader=lambda _spec: [{"text": "2"}, {"text": "3"}],
    )


def test_preflight_is_read_only_and_never_loads_models(tmp_path: Path) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()
    dependencies.model_pair_loader = lambda _config: pytest.fail("model load is forbidden")

    result = run_phase1(
        config,
        "preflight",
        resume=False,
        artifact_root=tmp_path,
        microbatch_size=2,
        dependencies=dependencies,
    )

    assert result["status"] == "complete"
    assert result["expected_weight_bytes"] == 123
    assert result["expected_repository_bytes"] == 456
    saved = json.loads((tmp_path / "stages" / "preflight.json").read_text())
    assert saved["status"] == "complete"
    assert saved["expected_weight_bytes"] == 123
    assert saved["expected_repository_bytes"] == 456


def test_datasets_requires_preflight_and_materializes_locked_counts(tmp_path: Path) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()

    with pytest.raises(StagePrerequisiteError, match="preflight"):
        run_phase1(
            config, "datasets", resume=False, artifact_root=tmp_path,
            microbatch_size=7, dependencies=dependencies,
        )
    run_phase1(
        config, "preflight", resume=False, artifact_root=tmp_path,
        microbatch_size=7, dependencies=dependencies,
    )
    result = run_phase1(
        config, "datasets", resume=False, artifact_root=tmp_path,
        microbatch_size=7, dependencies=dependencies,
    )

    assert result["status"] == "complete"
    assert result["counts"]["calibration"] == 100_000
    assert result["counts"]["fineweb_heldout"] == 100_000
    assert result["counts"]["wikitext"] == 6
    assert result["overlap"] == []


def test_prerequisite_signature_mismatch_is_rejected(tmp_path: Path) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()
    run_phase1(
        config, "preflight", resume=False, artifact_root=tmp_path,
        microbatch_size=1, dependencies=dependencies,
    )
    path = tmp_path / "stages" / "preflight.json"
    state = json.loads(path.read_text())
    state["config_signature"] = "wrong"
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(StagePrerequisiteError, match="signature"):
        run_phase1(
            config, "datasets", resume=False, artifact_root=tmp_path,
            microbatch_size=1, dependencies=dependencies,
        )


def test_m0_kill_switch_and_resume_idempotence(tmp_path: Path) -> None:
    from copy import deepcopy

    from tests.helpers import tiny_qwen35

    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()
    original_fineweb_loader = dependencies.fineweb_loader
    fineweb_calls = 0

    def count_fineweb(spec):
        nonlocal fineweb_calls
        fineweb_calls += 1
        assert original_fineweb_loader is not None
        return original_fineweb_loader(spec)

    dependencies.fineweb_loader = count_fineweb
    run_phase1(config, "preflight", False, tmp_path, 5, dependencies=dependencies)
    first = run_phase1(config, "datasets", False, tmp_path, 5, dependencies=dependencies)
    resumed = run_phase1(config, "datasets", True, tmp_path, 5, dependencies=dependencies)
    assert resumed == first
    assert fineweb_calls == 2

    def pair_loader(_config):
        torch.manual_seed(317)
        teacher = tiny_qwen35()
        return SimpleNamespace(teacher=teacher, student=deepcopy(teacher))

    dependencies.model_pair_loader = pair_loader
    dependencies.m0_evaluator = lambda *_args: SimpleNamespace(passed=False)
    with pytest.raises(Exception, match="M0 gate failed"):
        run_phase1(config, "m0", False, tmp_path, 5, dependencies=dependencies)


def test_execution_microbatch_isolates_m0_local_and_rollout_checkpoints(tmp_path: Path) -> None:
    from copy import deepcopy

    from tests.helpers import tiny_qwen35

    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()
    checkpoints: dict[str, list[tuple[Path, str]]] = {"m0": [], "local": [], "rollout": []}

    def mark_attempt(kind: str, checkpoint_dir: Path, run_signature: str) -> None:
        checkpoints[kind].append((checkpoint_dir, run_signature))
        marker = checkpoint_dir / "completed-batch-0"
        if marker.exists():
            pytest.fail(f"{kind} reused a completed batch from another attempt")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("completed", encoding="utf-8")

    def pair_loader(_config):
        torch.manual_seed(619)
        teacher = tiny_qwen35()
        return SimpleNamespace(teacher=teacher, student=deepcopy(teacher))

    def m0_evaluator(_teacher, _student, _batches, checkpoint_dir):
        mark_attempt("m0", checkpoint_dir, checkpoint_dir.name)
        return SimpleNamespace(passed=True)

    def local_runner(_teacher, _student, _batches, *, checkpoint_dir, run_signature, **_kwargs):
        mark_attempt("local", checkpoint_dir, run_signature)
        return SimpleNamespace(
            complete=True,
            direct_check_count=1,
            all_expert_check_count=1,
            layer_k={
                (layer, k): SimpleNamespace(
                    normalized_mse=5.8e-6,
                    squared_error_sum=5.8e-6,
                    target_squared_norm_sum=1.0,
                    token_count=1,
                )
                for layer in range(2)
                for k in config.k_values
            },
        )

    def rollout_runner(_teacher, _student, _batches, *, checkpoint_dir, run_signature, **_kwargs):
        mark_attempt("rollout", checkpoint_dir, run_signature)
        return SimpleNamespace(complete=True)

    dependencies.model_pair_loader = pair_loader
    dependencies.m0_evaluator = m0_evaluator
    dependencies.local_runner = local_runner
    dependencies.rollout_runner = rollout_runner
    run_phase1(config, "preflight", False, tmp_path, 4, dependencies=dependencies)
    run_phase1(config, "datasets", False, tmp_path, 4, dependencies=dependencies)

    first_m0 = run_phase1(config, "m0", False, tmp_path, 4, dependencies=dependencies)
    second_m0 = run_phase1(config, "m0", False, tmp_path, 2, dependencies=dependencies)
    assert first_m0["execution_microbatch_size"] == 4
    assert second_m0["execution_microbatch_size"] == 2
    assert first_m0["input_signature"] != second_m0["input_signature"]
    assert checkpoints["m0"][0] != checkpoints["m0"][2]
    assert (checkpoints["m0"][0][0] / "completed-batch-0").exists()

    run_phase1(config, "local-smoke", False, tmp_path, 4, dependencies=dependencies)
    first_local = run_phase1(config, "local", False, tmp_path, 4, dependencies=dependencies)
    second_local = run_phase1(config, "local", False, tmp_path, 2, dependencies=dependencies)
    assert first_local["execution_microbatch_size"] == 4
    assert second_local["execution_microbatch_size"] == 2
    assert checkpoints["local"][-2] != checkpoints["local"][-1]
    assert (checkpoints["local"][-2][0] / "completed-batch-0").exists()

    first_rollout = run_phase1(config, "rollout", False, tmp_path, 4, dependencies=dependencies)
    second_rollout = run_phase1(config, "rollout", False, tmp_path, 2, dependencies=dependencies)
    assert first_rollout["execution_microbatch_size"] == 4
    assert second_rollout["execution_microbatch_size"] == 2
    assert all(
        first != second
        for first, second in zip(checkpoints["rollout"][:10], checkpoints["rollout"][10:], strict=True)
    )
    assert all((checkpoint_dir / "completed-batch-0").exists() for checkpoint_dir, _ in checkpoints["rollout"][:10])


def test_cli_returns_nonzero_for_stage_failure(monkeypatch) -> None:
    from scripts import run_phase1_v0

    config = Phase1Config.load(CONFIG_PATH)
    monkeypatch.setattr(run_phase1_v0.Phase1Config, "load", lambda _path: config)
    monkeypatch.setattr(
        run_phase1_v0,
        "run_phase1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("failed gate")),
    )

    assert run_phase1_v0.main(["--stage", "preflight"]) == 1


def test_preflight_rejects_zero_weight_byte_estimate(tmp_path: Path) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()
    dependencies.remote_probe = lambda value: {
        "model_revision": value.model.revision,
        "fineweb_revision": value.fineweb.revision,
        "wikitext_revision": value.wikitext.revision,
        "expected_weight_bytes": 0,
        "expected_repository_bytes": 123,
    }

    with pytest.raises(Exception, match="expected weight byte"):
        run_phase1(config, "preflight", False, tmp_path, 1, dependencies=dependencies)


def test_preflight_rejects_repository_smaller_than_weight_files(tmp_path: Path) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()
    dependencies.remote_probe = lambda value: {
        "model_revision": value.model.revision,
        "fineweb_revision": value.fineweb.revision,
        "wikitext_revision": value.wikitext.revision,
        "expected_weight_bytes": 124,
        "expected_repository_bytes": 123,
    }

    with pytest.raises(Exception, match="repository byte"):
        run_phase1(config, "preflight", False, tmp_path, 1, dependencies=dependencies)


def test_default_remote_probe_requests_file_metadata(monkeypatch) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeApi:
        def model_info(self, *args, **kwargs):
            calls.append(("model", kwargs))
            return SimpleNamespace(
                sha=config.model.revision,
                siblings=[
                    SimpleNamespace(rfilename=".gitattributes", size=3),
                    SimpleNamespace(rfilename="config.json", size=5),
                    SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=7),
                    SimpleNamespace(rfilename="model-00002-of-00002.safetensors", size=11),
                    SimpleNamespace(rfilename="model.safetensors.index.json", size=13),
                ],
            )

        def dataset_info(self, repo_id, **kwargs):
            calls.append((repo_id, kwargs))
            revision = config.fineweb.revision if repo_id == config.fineweb.repo_id else config.wikitext.revision
            return SimpleNamespace(sha=revision)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    result = phase1_v0._default_remote_probe(config)

    assert calls[0] == ("model", {"revision": config.model.revision, "files_metadata": True})
    assert result["expected_weight_bytes"] == 18
    assert result["expected_repository_bytes"] == 39


def test_load_stream_rejects_corrupt_manifest_and_m0_uses_signature_checkpoint(tmp_path: Path) -> None:
    from copy import deepcopy

    from tests.helpers import tiny_qwen35

    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()
    run_phase1(config, "preflight", False, tmp_path, 8, dependencies=dependencies)
    datasets = run_phase1(config, "datasets", False, tmp_path, 8, dependencies=dependencies)
    reference = datasets["streams"]["wikitext"]
    manifest_path = Path(reference["path"]) / "manifest.json"
    manifest_path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(Exception, match="manifest"):
        phase1_v0._load_stream(reference)
    run_phase1(config, "datasets", False, tmp_path, 8, dependencies=dependencies)

    calls: list[Path] = []
    def pair_loader(_config):
        torch.manual_seed(419)
        teacher = tiny_qwen35()
        return SimpleNamespace(teacher=teacher, student=deepcopy(teacher))

    dependencies.model_pair_loader = pair_loader
    dependencies.m0_evaluator = lambda *_args: calls.append(_args[3]) or SimpleNamespace(passed=True)
    m0 = run_phase1(config, "m0", False, tmp_path, 8, dependencies=dependencies)
    conversion = json.loads((tmp_path / "manifests" / "conversion.json").read_text())
    assert conversion["converted_layers"] == 2
    assert len(conversion["records"]) == 2
    assert len(calls) == 2
    assert calls[0].parent.name == "wikitext"
    assert calls[0].name != calls[1].name
    assert m0["conversion_file_sha256"] == phase1_v0.sha256_file(tmp_path / "manifests" / "conversion.json")
    model_manifest = json.loads((tmp_path / "manifests" / "model.json").read_text())
    assert m0["model_file_sha256"] == phase1_v0.sha256_file(tmp_path / "manifests" / "model.json")
    assert m0["output_hashes"]["model"] == m0["model_file_sha256"]
    assert model_manifest["teacher"]["model_class"] == "Qwen3_5ForConditionalGeneration"
    assert model_manifest["student"]["parameter_count"] > 0
    assert model_manifest["cuda"] == {
        "free_cuda_bytes": None,
        "peak_cuda_bytes": 0,
        "total_cuda_bytes": None,
    }
    model_manifest["cuda"]["peak_cuda_bytes"] = 1
    phase1_v0.atomic_write_json(tmp_path / "manifests" / "model.json", model_manifest)
    dependencies.local_runner = lambda *_args, **_kwargs: pytest.fail("model manifest guard must run first")
    with pytest.raises(Exception, match="model manifest file hash"):
        run_phase1(config, "local-smoke", False, tmp_path, 8, dependencies=dependencies)


def test_model_manifest_verification_ignores_fresh_cuda_evidence_but_rejects_identity_change(tmp_path: Path) -> None:
    from copy import deepcopy

    from tests.helpers import tiny_qwen35

    teacher = tiny_qwen35(dtype=torch.bfloat16)
    student = deepcopy(teacher)
    report = validate_loaded_qwen35(
        teacher,
        expected_dtype=torch.bfloat16,
        expected_num_hidden_layers=2,
        expected_hidden_size=16,
        expected_intermediate_size=32,
    )
    m0_pair = SimpleNamespace(
        teacher=teacher,
        student=student,
        teacher_report=report,
        student_report=report,
        free_cuda_bytes=5 * 1024**3,
        total_cuda_bytes=8 * 1024**3,
        peak_cuda_bytes=2 * 1024**3,
    )
    model_hash = phase1_v0._persist_model_manifest(tmp_path, m0_pair)
    m0 = {"model_file_sha256": model_hash}
    fresh_pair = SimpleNamespace(
        teacher=teacher,
        student=student,
        teacher_report=report,
        student_report=report,
        free_cuda_bytes=6 * 1024**3,
        total_cuda_bytes=10 * 1024**3,
        peak_cuda_bytes=3 * 1024**3,
    )

    phase1_v0._verify_model_manifest(tmp_path, m0, fresh_pair)

    fresh_pair.student_report = replace(report, parameter_count=report.parameter_count + 1)
    with pytest.raises(Exception, match="identity differs"):
        phase1_v0._verify_model_manifest(tmp_path, m0, fresh_pair)


def test_m0_stops_before_evaluation_when_conversion_probe_fails(monkeypatch, tmp_path: Path) -> None:
    from copy import deepcopy

    from tests.helpers import tiny_qwen35

    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()
    dependencies.model_pair_loader = lambda _config: SimpleNamespace(
        teacher=tiny_qwen35(), student=deepcopy(tiny_qwen35())
    )
    dependencies.m0_evaluator = lambda *_args: pytest.fail("M0 evaluator must not run")
    monkeypatch.setattr(
        phase1_v0,
        "convert_qwen35_text_ffns",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("conversion probe failed")),
    )
    run_phase1(config, "preflight", False, tmp_path, 1, dependencies=dependencies)
    run_phase1(config, "datasets", False, tmp_path, 1, dependencies=dependencies)

    with pytest.raises(RuntimeError, match="conversion probe failed"):
        run_phase1(config, "m0", False, tmp_path, 1, dependencies=dependencies)


def test_weight_stages_use_current_disk_and_rollout_loads_one_pair(tmp_path: Path) -> None:
    from copy import deepcopy

    from tests.helpers import tiny_qwen35

    config = Phase1Config.load(CONFIG_PATH)
    dependencies = _dependencies()
    run_phase1(config, "preflight", False, tmp_path, 9, dependencies=dependencies)
    run_phase1(config, "datasets", False, tmp_path, 9, dependencies=dependencies)
    dependencies.disk_free_probe = lambda: 0
    dependencies.model_pair_loader = lambda _config: pytest.fail("disk guard must run first")
    with pytest.raises(Exception, match="50 GiB"):
        run_phase1(config, "m0", False, tmp_path, 9, dependencies=dependencies)

    dependencies.disk_free_probe = lambda: 100 * 1024**3
    pair_calls = 0

    def pair_loader(_config):
        nonlocal pair_calls
        pair_calls += 1
        torch.manual_seed(521)
        teacher = tiny_qwen35()
        return SimpleNamespace(teacher=teacher, student=deepcopy(teacher))

    dependencies.model_pair_loader = pair_loader
    dependencies.m0_evaluator = lambda *_args: SimpleNamespace(passed=True)
    run_phase1(config, "m0", False, tmp_path, 9, dependencies=dependencies)
    pair_calls = 0
    config_signature = phase1_v0._config_signature(config)
    phase1_v0._write_stage(tmp_path, "local", config_signature, "local", {})
    dependencies.rollout_runner = lambda *_args, **_kwargs: SimpleNamespace(complete=True)
    run_phase1(config, "rollout", False, tmp_path, 9, dependencies=dependencies)

    assert pair_calls == 1
