"""Locked configuration for the Phase 1A.5 reconstruction oracle."""

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from ppmoe.config import Phase1Config


@dataclass(frozen=True, slots=True)
class Phase1A5Config:
    name: str
    artifact_root: Path
    source_artifact_root: Path
    k_values: tuple[int, ...]
    rho_values: tuple[float, ...]
    tau_g: float
    epsilon_zero: float
    coefficient_bounds: tuple[float, float]
    local_improvement_gate: float
    practical_negative_rate_max: float
    practical_above_two_rate_max: float
    bounded_runtime_seconds_max: float
    bounded_peak_cuda_bytes_max: int
    solver_version: str

    def __post_init__(self) -> None:
        expected = {
            "name": "qwen35_0p8b_ppmoe_sparse_reconstruction_oracle_v0_5",
            "artifact_root": Path("artifacts/phase1-v0_5"),
            "source_artifact_root": Path("artifacts/phase1-v0"),
            "k_values": (4, 6),
            "rho_values": (10.0, 1.0, 0.1, 0.01, 0.001),
            "tau_g": 1e-12,
            "epsilon_zero": 1e-15,
            "coefficient_bounds": (0.0, 2.0),
            "local_improvement_gate": 0.30,
            "practical_negative_rate_max": 0.05,
            "practical_above_two_rate_max": 0.01,
            "bounded_runtime_seconds_max": 7200.0,
            "bounded_peak_cuda_bytes_max": 12 * 1024**3,
            "solver_version": "phase1a5-oracle-v1",
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"locked Phase 1A.5 protocol rejects {field}")

    @property
    def phase1_config(self) -> Phase1Config:
        return Phase1Config.load(Path(__file__).parents[3] / "configs" / "phase1-v0.toml")

    @classmethod
    def load(cls, path: Path) -> "Phase1A5Config":
        # Phase 1A.5 intentionally owns only oracle controls; the model and
        # data protocol remains the already-locked Phase 1 configuration.
        Phase1Config.load(Path(__file__).parents[3] / "configs" / "phase1-v0.toml")
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        experiment = values.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError("locked Phase 1A.5 configuration requires [experiment]")
        try:
            return cls(
                name=str(experiment["name"]),
                artifact_root=Path(str(experiment["artifact_root"])),
                source_artifact_root=Path(str(experiment["source_artifact_root"])),
                k_values=tuple(int(value) for value in experiment["k_values"]),
                rho_values=tuple(float(value) for value in experiment["rho_values"]),
                tau_g=float(experiment["tau_g"]),
                epsilon_zero=float(experiment["epsilon_zero"]),
                coefficient_bounds=tuple(
                    float(value) for value in experiment["coefficient_bounds"]
                ),
                local_improvement_gate=float(experiment["local_improvement_gate"]),
                practical_negative_rate_max=float(
                    experiment["practical_negative_rate_max"]
                ),
                practical_above_two_rate_max=float(
                    experiment["practical_above_two_rate_max"]
                ),
                bounded_runtime_seconds_max=float(
                    experiment["bounded_runtime_seconds_max"]
                ),
                bounded_peak_cuda_bytes_max=int(
                    experiment["bounded_peak_cuda_bytes_max"]
                ),
                solver_version=str(experiment["solver_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ValueError) and "locked Phase 1A.5" in str(error):
                raise
            raise ValueError("locked Phase 1A.5 configuration is malformed") from error
