"""Locked Phase 1 experiment configuration."""

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    repo_id: str
    revision: str


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    repo_id: str
    name: str
    split: str
    revision: str
    streaming: bool


@dataclass(frozen=True, slots=True)
class Phase1Config:
    name: str
    artifact_root: Path
    dtype: str
    context_length: int
    calibration_tokens: int
    heldout_tokens: int
    smoke_tokens: int
    k_values: tuple[int, ...]
    num_experts: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    seed_base: int
    model: ModelSpec
    fineweb: DatasetSpec
    wikitext: DatasetSpec

    def __post_init__(self) -> None:
        if self.name != "qwen35_0p8b_ppmoe_oracle_v0":
            raise ValueError("experiment name does not match the locked protocol")
        if self.model.repo_id != "Qwen/Qwen3.5-0.8B-Base":
            raise ValueError("model repository does not match the locked protocol")
        if self.model.revision != MODEL_REVISION:
            raise ValueError("model revision does not match the locked protocol")
        if self.fineweb != DatasetSpec(
            repo_id="HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            revision=FINEWEB_REVISION,
            streaming=True,
        ):
            raise ValueError("FineWeb dataset does not match the locked protocol")
        if self.wikitext != DatasetSpec(
            repo_id="Salesforce/wikitext",
            name="wikitext-103-raw-v1",
            split="test",
            revision=WIKITEXT_REVISION,
            streaming=False,
        ):
            raise ValueError("WikiText dataset does not match the locked protocol")
        if self.k_values != (8, 6, 4, 3, 2):
            raise ValueError("k_values must equal (8, 6, 4, 3, 2)")
        if self.num_experts != 8:
            raise ValueError("num_experts must equal 8")
        if self.hidden_size != 1_024:
            raise ValueError("hidden_size must equal 1024")
        if self.num_hidden_layers != 24:
            raise ValueError("num_hidden_layers must equal 24")
        if self.intermediate_size <= 0 or self.intermediate_size % self.num_experts:
            raise ValueError("intermediate_size must be positive and divisible by experts")
        if self.intermediate_size != 3_584:
            raise ValueError("intermediate_size must equal 3584")
        if self.calibration_tokens <= 0 or self.heldout_tokens <= 0:
            raise ValueError("token targets must be positive")
        if self.calibration_tokens != 100_000 or self.heldout_tokens != 100_000:
            raise ValueError("token targets must equal the locked 100000-token values")
        if self.smoke_tokens != 2_048:
            raise ValueError("smoke_tokens must equal 2048")
        if self.context_length != 512:
            raise ValueError("context_length must equal 512")
        if self.dtype != "bfloat16":
            raise ValueError("dtype must equal bfloat16")
        if self.seed_base != 20_260_830:
            raise ValueError("seed_base does not match the locked protocol")
        if (
            self.artifact_root.is_absolute()
            or ".." in self.artifact_root.parts
            or self.artifact_root.parts[:1] != ("artifacts",)
        ):
            raise ValueError("artifact_root must stay beneath artifacts/")

    @property
    def expert_width(self) -> int:
        return self.intermediate_size // self.num_experts

    @classmethod
    def load(cls, path: Path) -> "Phase1Config":
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        experiment = _required_table(values, "experiment")
        return cls(
            name=str(experiment["name"]),
            artifact_root=Path(str(experiment["artifact_root"])),
            dtype=str(experiment["dtype"]),
            context_length=int(experiment["context_length"]),
            calibration_tokens=int(experiment["calibration_tokens"]),
            heldout_tokens=int(experiment["heldout_tokens"]),
            smoke_tokens=int(experiment["smoke_tokens"]),
            k_values=tuple(int(value) for value in experiment["k_values"]),
            num_experts=int(experiment["num_experts"]),
            hidden_size=int(experiment["hidden_size"]),
            num_hidden_layers=int(experiment["num_hidden_layers"]),
            intermediate_size=int(experiment["intermediate_size"]),
            seed_base=int(experiment["seed_base"]),
            model=_load_model(_required_table(values, "model")),
            fineweb=_load_dataset(_required_table(values, "fineweb")),
            wikitext=_load_dataset(_required_table(values, "wikitext")),
        )


def _required_table(values: dict[str, Any], name: str) -> dict[str, Any]:
    table = values.get(name)
    if not isinstance(table, dict):
        raise ValueError(f"configuration requires [{name}]")
    return table


def _load_model(values: dict[str, Any]) -> ModelSpec:
    return ModelSpec(repo_id=str(values["repo_id"]), revision=str(values["revision"]))


def _load_dataset(values: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        repo_id=str(values["repo_id"]),
        name=str(values["name"]),
        split=str(values["split"]),
        revision=str(values["revision"]),
        streaming=bool(values["streaming"]),
    )
