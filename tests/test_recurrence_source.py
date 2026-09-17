"""Offline contract tests for immutable recurrence model provenance."""

import json
from pathlib import Path

import pytest

from ppmoe.recurrence.config import RecurrenceConfig
from ppmoe.recurrence.provenance import (
    _runtime_code_files,
    build_source_manifest,
    compute_run_id,
    resolve_model_revision,
    sha256_payload,
    validate_qwen3_config,
    validate_revision,
)


REGISTERED_CONFIG = Path("configs/recurrence-v0.toml")
REVISION = "a" * 40


class FakeApi:
    """A local metadata boundary with the shape used by HfApi.model_info."""

    def __init__(self, sha: str) -> None:
        self.sha = sha

    def model_info(self, repo_id: str, revision: str) -> object:
        if (repo_id, revision) != ("Qwen/Qwen3-1.7B", "main"):
            raise AssertionError("resolver sent unexpected model metadata request")
        return type("ModelInfo", (), {"sha": self.sha})()


def official_qwen3_config() -> dict[str, object]:
    """Hand-checked source metadata expected from the registered Qwen3 model."""
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 28,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "use_sliding_window": False,
        "vocab_size": 151936,
        "embedding_vocab_size": 151936,
    }


def write_snapshot(root: Path, revision: str = REVISION) -> Path:
    """Create a small local Hugging Face-shaped snapshot, with no network access."""
    snapshot = root / "models--Qwen--Qwen3-1.7B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(official_qwen3_config()), encoding="utf-8"
    )
    (snapshot / "generation_config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"test-weight-shard")
    (snapshot / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    return snapshot


def test_resolver_returns_full_commit_sha() -> None:
    """Catches a source resolver passing a mutable branch into later loaders."""
    assert resolve_model_revision(
        "Qwen/Qwen3-1.7B", "main", api=FakeApi(sha=REVISION)
    ) == REVISION


@pytest.mark.parametrize("value", ["main", "refs/pr/1", "abc123", "g" * 40])
def test_revision_rejects_mutable_or_invalid_values(value: str) -> None:
    """Catches source manifests accepting a ref that can move after the run."""
    with pytest.raises(ValueError, match="full immutable"):
        validate_revision(value)


def test_revision_accepts_uppercase_full_commit_sha() -> None:
    """Catches valid immutable commits being rejected only for hex letter case."""
    validate_revision("A" * 40)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architectures", ["OtherForCausalLM"]),
        ("num_hidden_layers", 27),
        ("hidden_size", 1024),
        ("intermediate_size", 4096),
        ("num_attention_heads", 8),
        ("num_key_value_heads", 4),
        ("head_dim", 64),
        ("use_sliding_window", True),
        ("embedding_vocab_size", 151935),
    ],
)
def test_qwen3_config_rejects_each_architecture_mutation(
    field: str, value: object
) -> None:
    """Catches a model whose loader-visible architecture differs from V0's contract."""
    config = official_qwen3_config()
    config[field] = value

    with pytest.raises(ValueError, match=field):
        validate_qwen3_config(config)


def test_qwen3_config_accepts_the_registered_architecture() -> None:
    """Catches an over-strict validator rejecting the intended model source."""
    validate_qwen3_config(official_qwen3_config())


def test_source_manifest_hashes_sorted_snapshot_and_writes_immutable_record(
    tmp_path: Path,
) -> None:
    """Catches an unhashed, unsorted, or mutable model source record."""
    snapshot = write_snapshot(tmp_path)
    output = tmp_path / "artifacts" / "source.json"

    manifest = build_source_manifest(
        RecurrenceConfig.load(REGISTERED_CONFIG), snapshot, output
    )

    assert manifest["schema_version"] == 1
    assert manifest["repo_id"] == "Qwen/Qwen3-1.7B"
    assert manifest["revision"] == REVISION
    assert manifest["model_class"] == "Qwen3ForCausalLM"
    assert [entry["relative_path"] for entry in manifest["files"]] == sorted(
        entry["relative_path"] for entry in manifest["files"]
    )
    assert {entry["relative_path"] for entry in manifest["files"]} == {
        "LICENSE",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert manifest["license_relative_path"] == "LICENSE"
    assert len(manifest["config_sha256"]) == len(manifest["code_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == manifest


def test_source_manifest_rejects_snapshot_path_without_resolved_revision(
    tmp_path: Path,
) -> None:
    """Catches a local cache directory being claimed as a different resolved commit."""
    snapshot = write_snapshot(tmp_path)
    invalid_snapshot = tmp_path / "plain-snapshot"
    snapshot.rename(invalid_snapshot)

    with pytest.raises(ValueError, match="revision"):
        build_source_manifest(
            RecurrenceConfig.load(REGISTERED_CONFIG), invalid_snapshot, tmp_path / "source.json"
        )


@pytest.mark.parametrize(
    ("removed_file", "expected_name"),
    [
        ("config.json", "config.json"),
        ("generation_config.json", "generation_config.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("tokenizer.json", "tokenizer data"),
        ("model.safetensors", "safetensors"),
    ],
)
def test_source_manifest_rejects_each_required_snapshot_inventory_gap(
    tmp_path: Path, removed_file: str, expected_name: str
) -> None:
    """Catches G0 certifying a snapshot that cannot instantiate the pinned model."""
    snapshot = write_snapshot(tmp_path)
    (snapshot / removed_file).unlink()

    with pytest.raises(ValueError, match=expected_name):
        build_source_manifest(
            RecurrenceConfig.load(REGISTERED_CONFIG), snapshot, tmp_path / "source.json"
        )


def test_source_manifest_rejects_minimal_config_and_license_only_snapshot(
    tmp_path: Path,
) -> None:
    """Catches a valid architecture record being mistaken for downloadable model source."""
    snapshot = write_snapshot(tmp_path)
    for relative_path in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ):
        (snapshot / relative_path).unlink()

    with pytest.raises(ValueError, match="generation_config.json"):
        build_source_manifest(
            RecurrenceConfig.load(REGISTERED_CONFIG), snapshot, tmp_path / "source.json"
        )


def test_source_manifest_requires_shard_index_for_multiple_weight_shards(
    tmp_path: Path,
) -> None:
    """Catches a multi-file weight set without the mapping needed to load its tensors."""
    snapshot = write_snapshot(tmp_path)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second")

    with pytest.raises(ValueError, match=r"model\.safetensors\.index\.json"):
        build_source_manifest(
            RecurrenceConfig.load(REGISTERED_CONFIG), snapshot, tmp_path / "source.json"
        )


def test_source_manifest_rejects_symlinked_snapshot_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a cache alias producing duplicate or falsified source-file paths."""
    snapshot = write_snapshot(tmp_path)
    alias = snapshot / "model-alias.safetensors"
    try:
        alias.symlink_to(snapshot / "model.safetensors")
    except OSError:
        # Windows can deny symlink creation without Developer Mode. Exercise the
        # same Path boundary with a regular alias marked as a symbolic link.
        alias.write_bytes(b"alias")
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == alias or original_is_symlink(path),
        )

    with pytest.raises(ValueError, match="symbolic link"):
        build_source_manifest(
            RecurrenceConfig.load(REGISTERED_CONFIG), snapshot, tmp_path / "source.json"
        )


def test_runtime_code_signature_includes_future_recurrence_entrypoint(
    tmp_path: Path,
) -> None:
    """Catches a changed stage dispatcher leaving the runtime-code signature unchanged."""
    project_root = tmp_path / "project"
    package_file = project_root / "src" / "ppmoe" / "runtime.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("VALUE = 1\n", encoding="utf-8")
    (project_root / "uv.lock").write_text("lock-v1\n", encoding="utf-8")
    entrypoint = project_root / "scripts" / "run_recurrence_v0.py"
    entrypoint.parent.mkdir()
    entrypoint.write_text("STAGE = 'source'\n", encoding="utf-8")

    before = _runtime_code_files(project_root)
    entrypoint.write_text("STAGE = 'smoke'\n", encoding="utf-8")
    after = _runtime_code_files(project_root)

    assert "scripts/run_recurrence_v0.py" in {
        entry.relative_path for entry in before
    }
    assert sha256_payload({entry.relative_path: entry.sha256 for entry in before}) != sha256_payload(
        {entry.relative_path: entry.sha256 for entry in after}
    )


def test_run_id_is_stable_and_changes_for_each_signed_input() -> None:
    """Catches run roots being reused after any source, data, config, or code drift."""
    source = "a" * 64
    data = "b" * 64
    config = "c" * 64
    code = "d" * 64
    run_id = compute_run_id(source, data, config, code)

    assert run_id == compute_run_id(source, data, config, code)
    assert len(run_id) == 12
    assert run_id != compute_run_id("e" * 64, data, config, code)
    assert run_id != compute_run_id(source, "e" * 64, config, code)
    assert run_id != compute_run_id(source, data, "e" * 64, code)
    assert run_id != compute_run_id(source, data, config, "e" * 64)


def test_run_id_rejects_a_recovery_revision_as_a_code_hash() -> None:
    """Catches a 40-hex recovery revision impersonating the 64-hex code evidence hash."""
    with pytest.raises(ValueError, match="code_hash"):
        compute_run_id("a" * 64, "b" * 64, "c" * 64, REVISION)


def test_source_manifest_records_recovery_revision_separately(tmp_path: Path) -> None:
    """Catches recovery provenance being folded into the original runtime-code hash."""
    snapshot = write_snapshot(tmp_path)

    original = build_source_manifest(
        RecurrenceConfig.load(REGISTERED_CONFIG), snapshot, tmp_path / "original.json"
    )
    recovered = build_source_manifest(
        RecurrenceConfig.load(REGISTERED_CONFIG), snapshot, tmp_path / "recovered.json",
        recovery_revision="b" * 40,
    )

    assert recovered["protocol_revision"] == original["protocol_revision"]
    assert recovered["code_sha256"] == original["code_sha256"]
    assert recovered["recovery_revision"] == "b" * 40
