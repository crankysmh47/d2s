from dataclasses import asdict
import json
import shutil
from pathlib import Path

import pytest

from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.experiments.phase1a5_provenance import (
    SourceVerificationError,
    phase1a5_signature,
    verify_phase1a_sources,
)
from ppmoe.provenance import sha256_file


REPOSITORY_ROOT = Path(__file__).parents[1]
PHASE1_ROOT = REPOSITORY_ROOT / "artifacts" / "phase1-v0"
CONFIG = Phase1A5Config.load(REPOSITORY_ROOT / "configs" / "phase1-v0_5.toml")
EXPECTED_STAGE_HASHES = {
    "local": "90382EA91C74B1850FB7AEF81C6F4705DD625A8559EF0FA725BD4511FB2D8341",
    "rollout": "0F415AF16EBFFB6231041605C0A72C4A3505C6DB1F5FF832D6126B8B339E334E",
    "m0": "B9BE2ECFE2B73A906682B652D65CE1C9DB094E0D23ECC825AB00E677BD3A40D0",
}


def copy_required_phase1_files(source: Path, destination: Path) -> None:
    for relative in (
        "stages/datasets.json",
        "stages/local.json",
        "stages/rollout.json",
        "stages/m0.json",
        "manifests/conversion.json",
        "manifests/model.json",
        "tokens/calibration/manifest.json",
        "tokens/fineweb-heldout/manifest.json",
        "tokens/wikitext/manifest.json",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


def clean_revision(_: Path) -> dict[str, object]:
    return {"commit": "a" * 40, "dirty": False}


def test_source_verifier_pins_all_phase1a_stage_hashes(tmp_path: Path) -> None:
    copy_required_phase1_files(PHASE1_ROOT, tmp_path)

    manifest = verify_phase1a_sources(
        tmp_path, tmp_path / "v0_5", CONFIG, code_revision_provider=clean_revision
    )

    assert manifest["source_root"] == str(tmp_path.resolve())
    assert manifest["stage_sha256"] == EXPECTED_STAGE_HASHES
    assert manifest["code_revision"] == {"commit": "a" * 40, "dirty": False}
    assert manifest["solver_version"] == CONFIG.solver_version
    assert set(manifest["dataset_manifest_sha256"]) == {
        "calibration",
        "fineweb-heldout",
        "wikitext",
    }
    assert set(manifest["model_manifest_sha256"]) == {"conversion", "model"}
    saved = tmp_path / "v0_5" / "source-manifest.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == manifest
    assert sha256_file(saved)


def test_source_verifier_rejects_mutated_phase1a_stage(tmp_path: Path) -> None:
    copy_required_phase1_files(PHASE1_ROOT, tmp_path)
    with (tmp_path / "stages/local.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(SourceVerificationError, match="local SHA-256"):
        verify_phase1a_sources(
            tmp_path, tmp_path / "v0_5", CONFIG, code_revision_provider=clean_revision
        )


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("tokens/calibration/manifest.json", "calibration manifest SHA-256"),
        ("tokens/fineweb-heldout/manifest.json", "fineweb-heldout manifest SHA-256"),
        ("tokens/wikitext/manifest.json", "wikitext manifest SHA-256"),
        ("manifests/conversion.json", "conversion manifest SHA-256"),
        ("manifests/model.json", "model manifest SHA-256"),
    ],
)
def test_source_verifier_rejects_mutated_pinned_evidence(
    tmp_path: Path, relative_path: str, message: str
) -> None:
    copy_required_phase1_files(PHASE1_ROOT, tmp_path)
    with (tmp_path / relative_path).open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(SourceVerificationError, match=message):
        verify_phase1a_sources(
            tmp_path, tmp_path / "v0_5", CONFIG, code_revision_provider=clean_revision
        )


def test_source_verifier_refuses_dirty_production_revision(tmp_path: Path) -> None:
    copy_required_phase1_files(PHASE1_ROOT, tmp_path)

    with pytest.raises(SourceVerificationError, match="dirty source tree"):
        verify_phase1a_sources(
            tmp_path,
            tmp_path / "v0_5",
            CONFIG,
            code_revision_provider=lambda _: {"commit": "b" * 40, "dirty": True},
        )


def _config_variant() -> Phase1A5Config:
    values = asdict(CONFIG)
    values["tau_g"] = 1e-13
    variant = object.__new__(Phase1A5Config)
    for field, value in values.items():
        object.__setattr__(variant, field, value)
    return variant


def test_phase1a5_signature_is_stable_and_binds_each_execution_identity_input(
    tmp_path: Path,
) -> None:
    copy_required_phase1_files(PHASE1_ROOT, tmp_path)
    source_manifest = verify_phase1a_sources(
        tmp_path, tmp_path / "v0_5", CONFIG, code_revision_provider=clean_revision
    )
    first = phase1a5_signature(CONFIG, source_manifest, "solver-v1", "c" * 40, 2)

    assert first == phase1a5_signature(
        CONFIG, source_manifest, "solver-v1", "c" * 40, 2
    )
    timestamp_changed = dict(source_manifest)
    timestamp_changed["created_at"] = "2099-01-01T00:00:00+00:00"
    assert first == phase1a5_signature(
        CONFIG, timestamp_changed, "solver-v1", "c" * 40, 2
    )
    assert first != phase1a5_signature(
        _config_variant(), source_manifest, "solver-v1", "c" * 40, 2
    )
    source_hash_changed = dict(source_manifest)
    source_hash_changed["stage_sha256"] = dict(EXPECTED_STAGE_HASHES, local="0" * 64)
    assert first != phase1a5_signature(
        CONFIG, source_hash_changed, "solver-v1", "c" * 40, 2
    )
    assert first != phase1a5_signature(
        CONFIG, source_manifest, "solver-v2", "c" * 40, 2
    )
    assert first != phase1a5_signature(
        CONFIG, source_manifest, "solver-v1", "d" * 40, 2
    )
    assert first != phase1a5_signature(
        CONFIG, source_manifest, "solver-v1", "c" * 40, 1
    )
