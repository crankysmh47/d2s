from pathlib import Path

import pytest

from ppmoe.experiments.phase1a5_config import Phase1A5Config


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "phase1-v0_5.toml"


def mutate_toml(source: Path, destination: Path, field: str, value: object) -> Path:
    replacement = repr(value).replace("'", '"')
    text = source.read_text(encoding="utf-8")
    lines = [
        f"{field} = {replacement}" if line.startswith(f"{field} =") else line
        for line in text.splitlines()
    ]
    path = destination / "phase1-v0_5.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_phase1a5_config_locks_oracle_grid_and_numerics() -> None:
    config = Phase1A5Config.load(CONFIG_PATH)

    assert config.k_values == (4, 6)
    assert config.rho_values == (10.0, 1.0, 0.1, 0.01, 0.001)
    assert config.tau_g == 1e-12
    assert config.epsilon_zero == 1e-15
    assert config.coefficient_bounds == (0.0, 2.0)
    assert config.local_improvement_gate == 0.30
    assert config.practical_negative_rate_max == 0.05
    assert config.practical_above_two_rate_max == 0.01
    assert config.bounded_runtime_seconds_max == 7200.0
    assert config.bounded_peak_cuda_bytes_max == 12 * 1024**3


@pytest.mark.parametrize(
    ("field", "value"), [("tau_g", 1e-8), ("k_values", [4, 5, 6])]
)
def test_phase1a5_config_rejects_protocol_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    path = mutate_toml(CONFIG_PATH, tmp_path, field, value)

    with pytest.raises(ValueError, match="locked Phase 1A.5"):
        Phase1A5Config.load(path)
