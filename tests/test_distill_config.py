from __future__ import annotations

from pathlib import Path

import pytest

from ppmoe.distill.config import canonical_config, load_config


CONFIG = Path(__file__).parents[1] / "configs" / "distillation-v1.toml"


def test_registered_config_loads_frozen_sources_training_and_partitions() -> None:
    config = load_config(CONFIG)
    canonical = canonical_config(config)

    assert config.student.repo_id == "Qwen/Qwen3-1.7B"
    assert config.student.revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert config.teacher.repo_id == "Qwen/Qwen3-4B"
    assert config.teacher.revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert config.gsm8k.revision == "740312add88f781978c0658806c59bc2815b9866"
    assert config.svamp.revision == "78e727689e1c1bebfc4be39c446898e8e10b0518"
    assert config.training.updates == 300
    assert config.training.seeds == (47, 48)
    assert config.resources.vram_reserve_bytes == 2147483648
    assert (
        config.partition.gsm_train.calibration,
        config.partition.gsm_train.pilot,
        config.partition.gsm_train.development,
        config.partition.gsm_train.pool,
    ) == (60, 120, 200, 1200)
    assert canonical["prompt"]["template"].startswith("Solve the following problem.")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("updates = 300", "updates = 301", "registered"),
        ("revision = \"740312add88f781978c0658806c59bc2815b9866\"", 'revision = "main"', "revision"),
        ("seeds = [47, 48]", "seeds = [47, true]", "type"),
        ("dropout = 0.0", "dropout = 0", "type"),
    ],
)
def test_rejects_drifted_or_malformed_registered_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    changed = text.replace(old, new, 1)
    assert changed != text
    path = tmp_path / "bad.toml"
    path.write_text(changed, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_rejects_unknown_config_keys(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8") + "\n[unregistered]\nmagic = 1\n"
    path = tmp_path / "unknown.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
