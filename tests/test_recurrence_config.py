from pathlib import Path

import pytest

from ppmoe.recurrence.config import RecurrenceConfig


REGISTERED_CONFIG = Path("configs/recurrence-v0.toml")


def write_toml_copy(tmp_path: Path, changes: dict[str, dict[str, object]]) -> Path:
    """Copy the registered configuration and replace or add only requested keys."""
    lines = REGISTERED_CONFIG.read_text(encoding="utf-8").splitlines()
    current_section: str | None = None
    replacements = {section: dict(values) for section, values in changes.items()}
    output: list[str] = []

    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            if current_section in replacements:
                output.extend(
                    f"{key} = {toml_literal(value)}"
                    for key, value in replacements.pop(current_section).items()
                )
            current_section = line[1:-1]
            output.append(line)
            continue
        key = line.split("=", maxsplit=1)[0].strip()
        if current_section in replacements and key in replacements[current_section]:
            output.append(f"{key} = {toml_literal(replacements[current_section].pop(key))}")
        else:
            output.append(line)

    if current_section in replacements:
        output.extend(
            f"{key} = {toml_literal(value)}"
            for key, value in replacements.pop(current_section).items()
        )
    for section, values in replacements.items():
        output.append(f"[{section}]")
        output.extend(f"{key} = {toml_literal(value)}" for key, value in values.items())

    destination = tmp_path / "recurrence-v0.toml"
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    return destination


def toml_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def test_registered_config_loads() -> None:
    """Catches drift in the controls consumed by recurrence training and gates."""
    config = RecurrenceConfig.load(REGISTERED_CONFIG)

    assert config.model.repo_id == "Qwen/Qwen3-1.7B"
    assert config.loop.layer_start == 12
    assert config.loop.layer_stop == 16
    assert config.lora.rank == 8
    assert config.statistics.retention_bootstrap_seed == 9157
    assert config.statistics.minimum_informative_families == 2


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    """Catches an unregistered knob silently changing the scientific protocol."""
    path = write_toml_copy(tmp_path, {"loop": {"layer_search": True}})

    with pytest.raises(ValueError, match="unknown.*layer_search"):
        RecurrenceConfig.load(path)


def test_registered_constant_cannot_be_overridden(tmp_path: Path) -> None:
    """Catches a changed LoRA rank being accepted as a V0 experiment."""
    path = write_toml_copy(tmp_path, {"lora": {"rank": 16}})

    with pytest.raises(ValueError, match="lora.rank"):
        RecurrenceConfig.load(path)


def test_boolean_is_rejected_for_integer_control(tmp_path: Path) -> None:
    """Catches TOML's bool-as-int ambiguity at a fixed update count boundary."""
    path = write_toml_copy(tmp_path, {"optimizer": {"updates": True}})

    with pytest.raises(ValueError, match="optimizer.updates"):
        RecurrenceConfig.load(path)


def test_scientific_payload_retains_resource_limits_but_excludes_artifact_path() -> None:
    """Catches a changed registered budget escaping the scientific signature."""
    payload = RecurrenceConfig.load(REGISTERED_CONFIG).scientific_payload()

    assert payload["resources"] == {
        "maximum_cuda_seconds": 86400.0,
        "maximum_download_bytes": 8 * 1024**3,
        "maximum_artifact_bytes": 20 * 1024**3,
        "reserved_vram_bytes": 2 * 1024**3,
    }
    assert "artifact_root" not in payload["resources"]
    assert payload["statistics"]["bootstrap_replicates"] == 10000


def test_total_sequence_limit_is_distinct_from_generation_cap(tmp_path: Path) -> None:
    """Catches the 512-token total input cap being mistaken for output length."""
    config = RecurrenceConfig.load(REGISTERED_CONFIG)

    assert config.data.max_sequence_tokens == 512
    assert config.decoding.max_new_tokens == 32
    assert not hasattr(config.data, "maximum_generation_tokens")

    path = write_toml_copy(tmp_path, {"data": {"maximum_generation_tokens": 512}})
    with pytest.raises(ValueError, match="unknown.*maximum_generation_tokens"):
        RecurrenceConfig.load(path)
