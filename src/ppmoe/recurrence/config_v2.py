"""Strict, independent configuration for the pretrained recurrence V2 study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import tomllib
from types import MappingProxyType
from typing import Mapping


_SCHEMA: dict[str, object] = {
    "source": {"run", "v1_pilot_root"},
    "model": {"repo_id", "revision"},
    "inference": {
        "dtype", "attention_implementation", "layer_start", "layer_stop", "use_cache",
        "enable_thinking", "greedy", "max_new_tokens", "max_sequence_tokens",
    },
    "data": {
        "output_root", "pilot_per_candidate", "development_per_family", "sealed_per_family",
        "maximum_attempts_per_example", "maximum_absolute_expression_answer", "graph_edge_probability",
        "candidates", "seeds",
    },
    "lora": {"rank", "alpha", "dropout", "seeds", "target_modules"},
    "optimizer": {
        "name", "updates", "microbatch_size", "accumulation_steps", "warmup_updates", "learning_rate",
        "beta1", "beta2", "epsilon", "weight_decay", "max_grad_norm",
    },
    "resources": {
        "maximum_calibration_cuda_seconds", "maximum_calibration_wall_seconds",
        "maximum_study_cuda_seconds", "maximum_artifact_bytes",
    },
}
_SUBSCHEMAS = {
    "data.candidates": {"expression_leaf_counts", "list_item_counts", "list_operation_counts", "graph_vertex_counts"},
    "data.seeds": {"pilot_expressions", "pilot_lists", "pilot_graphs", "development", "sealed"},
}
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_TARGET_MODULES = ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_V2_PILOT_ROOT = "artifacts/recurrence-v1/4f1ddbedf7b4cbd11d73699d579a3e299a049aa8fb2b036e9124842d695eb2ca"


@dataclass(frozen=True, slots=True)
class V2DataConfig:
    output_root: str
    pilot_per_candidate: int
    development_per_family: int
    sealed_per_family: int
    maximum_attempts_per_example: int
    maximum_absolute_expression_answer: int
    graph_edge_probability: float
    expression_leaf_counts: tuple[int, int]
    list_item_counts: tuple[int, int]
    list_operation_counts: tuple[int, int]
    graph_vertex_counts: tuple[int, int]
    pilot_seeds: Mapping[str, tuple[int, int]]
    development_seeds: Mapping[str, int]
    sealed_seeds: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class V2LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    seeds: tuple[int, int]
    target_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class V2OptimizerConfig:
    name: str
    updates: int
    microbatch_size: int
    accumulation_steps: int
    warmup_updates: int
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    max_grad_norm: float


@dataclass(frozen=True, slots=True)
class V2ResourceConfig:
    maximum_calibration_cuda_seconds: int
    maximum_calibration_wall_seconds: int
    maximum_study_cuda_seconds: int
    maximum_artifact_bytes: int


@dataclass(frozen=True, slots=True)
class V2Config:
    source_run: str
    v1_pilot_root: str
    model_repo_id: str
    model_revision: str
    dtype: str
    attention_implementation: str
    layer_start: int
    layer_stop: int
    use_cache: bool
    enable_thinking: bool
    greedy: bool
    max_new_tokens: int
    max_sequence_tokens: int
    data: V2DataConfig
    lora: V2LoRAConfig
    optimizer: V2OptimizerConfig
    resources: V2ResourceConfig


def load_v2_config(path: Path) -> V2Config:
    """Load a complete V2 configuration and reject any deviation from its protocol."""
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    _exact_keys(raw, _SCHEMA)
    for section_name in ("source", "model", "inference", "lora", "optimizer", "resources"):
        _exact_keys(_table(raw, section_name), _SCHEMA[section_name], section_name + ".")
    data = _table(raw, "data")
    _exact_keys(data, _SCHEMA["data"], "data.")
    for dotted, expected in _SUBSCHEMAS.items():
        section: Mapping[str, object] = raw
        for component in dotted.split("."):
            value = section.get(component)
            if not isinstance(value, Mapping):
                raise ValueError(f"configuration section {dotted} must be a table")
            section = value
        _exact_keys(section, expected, dotted + ".")

    source, model = _table(raw, "source"), _table(raw, "model")
    inference = _table(raw, "inference")
    candidate, seeds = _table(data, "candidates"), _table(data, "seeds")
    lora, optimizer, resources = _table(raw, "lora"), _table(raw, "optimizer"), _table(raw, "resources")
    source_run = _string(source, "run")
    pilot_root = _string(source, "v1_pilot_root")
    output_root = _string(data, "output_root")
    for value, name in ((source_run, "source.run"), (pilot_root, "source.v1_pilot_root"),
                        (output_root, "data.output_root")):
        _valid_relative_path(value, name)
    if len({source_run, pilot_root, output_root}) != 3:
        raise ValueError("source, V1 pilot, and output paths must be distinct")
    if source_run != "artifacts/recurrence-v0/129d39646933":
        raise ValueError("source run differs from the registered V2 source")
    if pilot_root != _V2_PILOT_ROOT:
        raise ValueError("V1 pilot root differs from the registered V1 run")
    if output_root != "artifacts/recurrence-v2":
        raise ValueError("output root differs from the registered V2 output root")

    revision = _string(model, "revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("model.revision must be an immutable 40-character revision")
    if revision != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e":
        raise ValueError("model.revision differs from the registered local snapshot")
    if _string(model, "repo_id") != "Qwen/Qwen3-1.7B":
        raise ValueError("model.repo_id differs from the registered source model")
    dtype, attention = _string(inference, "dtype"), _string(inference, "attention_implementation")
    start, stop = _integer(inference, "layer_start"), _integer(inference, "layer_stop")
    if (dtype, attention, start, stop) != ("bfloat16", "eager", 12, 16):
        raise ValueError("inference settings differ from the registered V2 protocol")
    cache, thinking, greedy = (_boolean(inference, key) for key in ("use_cache", "enable_thinking", "greedy"))
    output_tokens = _integer(inference, "max_new_tokens")
    sequence_tokens = _integer(inference, "max_sequence_tokens")
    if cache or thinking or not greedy:
        raise ValueError("decoding flags differ from the registered V2 protocol")
    if (output_tokens, sequence_tokens) != (256, 512):
        raise ValueError("token budgets differ from the registered V2 protocol")

    pilot_seeds = MappingProxyType({
        family: _pair(seeds, key)
        for family, key in (("expressions", "pilot_expressions"), ("lists", "pilot_lists"), ("graphs", "pilot_graphs"))
    })
    development_seeds = MappingProxyType(dict(zip(
        ("expressions", "lists", "graphs"), _triple(seeds, "development"), strict=True
    )))
    sealed_seeds = MappingProxyType(dict(zip(
        ("expressions", "lists", "graphs"), _triple(seeds, "sealed"), strict=True
    )))
    data_config = V2DataConfig(
        output_root, _integer(data, "pilot_per_candidate"), _integer(data, "development_per_family"),
        _integer(data, "sealed_per_family"), _integer(data, "maximum_attempts_per_example"),
        _integer(data, "maximum_absolute_expression_answer"), _number(data, "graph_edge_probability"),
        _pair(candidate, "expression_leaf_counts"), _pair(candidate, "list_item_counts"),
        _pair(candidate, "list_operation_counts"), _pair(candidate, "graph_vertex_counts"),
        pilot_seeds, development_seeds, sealed_seeds,
    )
    lora_config = V2LoRAConfig(
        _integer(lora, "rank"), _integer(lora, "alpha"), _number(lora, "dropout"),
        _pair(lora, "seeds"), _strings(lora, "target_modules"),
    )
    optimizer_config = V2OptimizerConfig(
        _string(optimizer, "name"), _integer(optimizer, "updates"), _integer(optimizer, "microbatch_size"),
        _integer(optimizer, "accumulation_steps"), _integer(optimizer, "warmup_updates"),
        _number(optimizer, "learning_rate"), _number(optimizer, "beta1"), _number(optimizer, "beta2"),
        _number(optimizer, "epsilon"), _number(optimizer, "weight_decay"), _number(optimizer, "max_grad_norm"),
    )
    resource_config = V2ResourceConfig(
        _integer(resources, "maximum_calibration_cuda_seconds"),
        _integer(resources, "maximum_calibration_wall_seconds"),
        _integer(resources, "maximum_study_cuda_seconds"), _integer(resources, "maximum_artifact_bytes"),
    )
    if (
        data_config.pilot_per_candidate, data_config.development_per_family, data_config.sealed_per_family,
        data_config.maximum_attempts_per_example, data_config.maximum_absolute_expression_answer,
        data_config.graph_edge_probability, data_config.expression_leaf_counts, data_config.list_item_counts,
        data_config.list_operation_counts, data_config.graph_vertex_counts,
    ) != (30, 100, 200, 100, 9999, 0.2, (6, 7), (4, 5), (1, 2), (7, 8)):
        raise ValueError("data candidate grid differs from the registered V2 protocol")
    if pilot_seeds != {"expressions": (721001, 721002), "lists": (722001, 722002), "graphs": (723001, 723002)}:
        raise ValueError("pilot seeds differ from the registered V2 protocol")
    if development_seeds != {"expressions": 731001, "lists": 731002, "graphs": 731003}:
        raise ValueError("development seeds differ from the registered V2 protocol")
    if sealed_seeds != {"expressions": 741001, "lists": 741002, "graphs": 741003}:
        raise ValueError("sealed seeds differ from the registered V2 protocol")
    if (lora_config.rank, lora_config.alpha, lora_config.dropout, lora_config.seeds,
        lora_config.target_modules) != (8, 16, 0.0, (47, 48), _TARGET_MODULES):
        raise ValueError("LoRA settings differ from the registered target configuration")
    if (
        optimizer_config.name, optimizer_config.updates, optimizer_config.warmup_updates,
        optimizer_config.learning_rate, optimizer_config.beta1, optimizer_config.beta2, optimizer_config.epsilon,
        optimizer_config.weight_decay, optimizer_config.max_grad_norm, optimizer_config.microbatch_size,
        optimizer_config.accumulation_steps,
    ) != ("AdamW", 1000, 20, 1e-4, 0.9, 0.95, 1e-8, 0.0, 1.0, 1, 8):
        raise ValueError("optimizer hyperparameters differ from the registered protocol")
    if (
        resource_config.maximum_calibration_cuda_seconds, resource_config.maximum_calibration_wall_seconds,
        resource_config.maximum_study_cuda_seconds, resource_config.maximum_artifact_bytes,
    ) != (7200, 7200, 86400, 20 * 1024**3):
        raise ValueError("resource limits or calibration wall budget differ from the registered V2 protocol")
    return V2Config(
        source_run, pilot_root, "Qwen/Qwen3-1.7B", revision, dtype, attention, start, stop, cache, thinking,
        greedy, output_tokens, sequence_tokens, data_config, lora_config, optimizer_config, resource_config,
    )


def _exact_keys(raw: Mapping[str, object], schema: set[str] | dict[str, object], prefix: str = "") -> None:
    unknown, missing = set(raw) - set(schema), set(schema) - set(raw)
    if unknown:
        raise ValueError(f"unknown configuration key {prefix}{sorted(unknown)[0]}")
    if missing:
        raise ValueError(f"missing configuration key {prefix}{sorted(missing)[0]}")


def _table(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {key} must be a table")
    return value


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(raw: Mapping[str, object], key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _boolean(raw: Mapping[str, object], key: str) -> bool:
    value = raw[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _string(raw: Mapping[str, object], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _strings(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = raw[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a string array")
    return tuple(value)


def _integer_array(raw: Mapping[str, object], key: str, length: int) -> tuple[int, ...]:
    value = raw[key]
    if (
        not isinstance(value, list) or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{key} must contain {length} integers")
    return tuple(value)


def _pair(raw: Mapping[str, object], key: str) -> tuple[int, int]:
    return _integer_array(raw, key, 2)  # type: ignore[return-value]


def _triple(raw: Mapping[str, object], key: str) -> tuple[int, int, int]:
    return _integer_array(raw, key, 3)  # type: ignore[return-value]


def _valid_relative_path(value: str, name: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"{name} must be a safe relative POSIX path")
