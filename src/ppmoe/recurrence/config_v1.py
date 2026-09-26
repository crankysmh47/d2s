"""Strict, independent configuration for the pretrained recurrence V1 study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import tomllib
from types import MappingProxyType
from typing import Mapping


_SCHEMA: dict[str, object] = {
    "source": {"run"},
    "model": {"repo_id", "revision"},
    "inference": {
        "dtype", "attention_implementation", "layer_start", "layer_stop", "use_cache",
        "enable_thinking", "greedy", "max_new_tokens", "max_sequence_tokens",
    },
    "data": {"output_root", "pilot_per_candidate", "development_per_family", "sealed_per_family",
             "maximum_attempts_per_example", "maximum_absolute_expression_answer", "graph_edge_probability",
             "candidates", "seeds"},
    "lora": {"rank", "alpha", "dropout", "seeds", "target_modules"},
    "optimizer": {"name", "updates", "microbatch_size", "accumulation_steps", "warmup_updates", "learning_rate",
                  "beta1", "beta2", "epsilon", "weight_decay", "max_grad_norm"},
    "resources": {"maximum_calibration_cuda_seconds", "maximum_study_cuda_seconds", "maximum_artifact_bytes"},
}
_SUBSCHEMAS = {
    "data.candidates": {"expression_leaf_counts", "list_item_counts", "list_operation_counts", "graph_vertex_counts"},
    "data.seeds": {"pilot_expressions", "pilot_lists", "pilot_graphs", "development", "sealed"},
}
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_TARGET_MODULES = ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True, slots=True)
class V1DataConfig:
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
class V1LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    seeds: tuple[int, int]
    target_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class V1OptimizerConfig:
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
class V1ResourceConfig:
    maximum_calibration_cuda_seconds: int
    maximum_study_cuda_seconds: int
    maximum_artifact_bytes: int


@dataclass(frozen=True, slots=True)
class V1Config:
    source_run: str
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
    data: V1DataConfig
    lora: V1LoRAConfig
    optimizer: V1OptimizerConfig
    resources: V1ResourceConfig


def load_v1_config(path: Path) -> V1Config:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    _exact_keys(raw, _SCHEMA)
    for section_name, expected in {
        "source": {"run"},
        "model": {"repo_id", "revision"},
        "inference": {"dtype", "attention_implementation", "layer_start", "layer_stop", "use_cache",
                      "enable_thinking", "greedy", "max_new_tokens", "max_sequence_tokens"},
        "lora": {"rank", "alpha", "dropout", "seeds", "target_modules"},
        "optimizer": {"name", "updates", "microbatch_size", "accumulation_steps", "warmup_updates", "learning_rate",
                      "beta1", "beta2", "epsilon", "weight_decay", "max_grad_norm"},
        "resources": {"maximum_calibration_cuda_seconds", "maximum_study_cuda_seconds", "maximum_artifact_bytes"},
    }.items():
        _exact_keys(_table(raw, section_name), expected, section_name + ".")
    _exact_keys(
        _table(raw, "data"),
        {"output_root", "pilot_per_candidate", "development_per_family", "sealed_per_family",
         "maximum_attempts_per_example", "maximum_absolute_expression_answer", "graph_edge_probability",
         "candidates", "seeds"},
        "data.",
    )
    for dotted, expected in _SUBSCHEMAS.items():
        section: Mapping[str, object] = raw
        for component in dotted.split("."):
            value = section.get(component)
            if not isinstance(value, Mapping):
                raise ValueError(f"configuration section {dotted} must be a table")
            section = value
        _exact_keys(section, expected, dotted + ".")

    source, model = _table(raw, "source"), _table(raw, "model")
    inference, data = _table(raw, "inference"), _table(raw, "data")
    candidate, seeds = _table(data, "candidates"), _table(data, "seeds")
    lora, optimizer, resources = _table(raw, "lora"), _table(raw, "optimizer"), _table(raw, "resources")
    source_run = _string(source, "run")
    output_root = _string(data, "output_root")
    _valid_relative_path(source_run, "source.run")
    _valid_relative_path(output_root, "data.output_root")
    if source_run == output_root:
        raise ValueError("source and output paths must be distinct")

    revision = _string(model, "revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("model.revision must be an immutable 40-character revision")
    if revision != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e":
        raise ValueError("model.revision differs from the registered local snapshot")
    if _string(model, "repo_id") != "Qwen/Qwen3-1.7B":
        raise ValueError("model.repo_id differs from the registered source model")
    dtype = _string(inference, "dtype")
    attention = _string(inference, "attention_implementation")
    start, stop = _integer(inference, "layer_start"), _integer(inference, "layer_stop")
    if (dtype, attention, start, stop) != ("bfloat16", "eager", 12, 16):
        raise ValueError("inference settings differ from the registered V1 protocol")
    cache = _boolean(inference, "use_cache")
    thinking = _boolean(inference, "enable_thinking")
    greedy = _boolean(inference, "greedy")
    output_tokens = _integer(inference, "max_new_tokens")
    sequence_tokens = _integer(inference, "max_sequence_tokens")
    if cache or thinking or not greedy or output_tokens <= 0 or sequence_tokens < output_tokens:
        raise ValueError("invalid V1 decoding flags or impossible token budget")
    if (output_tokens, sequence_tokens) != (256, 512):
        raise ValueError("token budgets differ from the registered V1 protocol")

    pilot_seeds = MappingProxyType({
        family: _pair(seeds, key)
        for family, key in (("expressions", "pilot_expressions"), ("lists", "pilot_lists"), ("graphs", "pilot_graphs"))
    })
    data_config = V1DataConfig(
        output_root, _integer(data, "pilot_per_candidate"), _integer(data, "development_per_family"),
        _integer(data, "sealed_per_family"), _integer(data, "maximum_attempts_per_example"),
        _integer(data, "maximum_absolute_expression_answer"), _number(data, "graph_edge_probability"),
        _pair(candidate, "expression_leaf_counts"), _pair(candidate, "list_item_counts"),
        _pair(candidate, "list_operation_counts"), _pair(candidate, "graph_vertex_counts"), pilot_seeds,
        MappingProxyType(dict(zip(("expressions", "lists", "graphs"), _triple(seeds, "development")))),
        MappingProxyType(dict(zip(("expressions", "lists", "graphs"), _triple(seeds, "sealed")))),
    )
    lora_config = V1LoRAConfig(
        _integer(lora, "rank"), _integer(lora, "alpha"), _number(lora, "dropout"),
        _pair(lora, "seeds"), _strings(lora, "target_modules")
    )
    optimizer_config = V1OptimizerConfig(
        _string(optimizer, "name"), _integer(optimizer, "updates"), _integer(optimizer, "microbatch_size"),
        _integer(optimizer, "accumulation_steps"), _integer(optimizer, "warmup_updates"),
        _number(optimizer, "learning_rate"), _number(optimizer, "beta1"), _number(optimizer, "beta2"),
        _number(optimizer, "epsilon"), _number(optimizer, "weight_decay"), _number(optimizer, "max_grad_norm"),
    )
    resource_config = V1ResourceConfig(
        _integer(resources, "maximum_calibration_cuda_seconds"), _integer(resources, "maximum_study_cuda_seconds"),
        _integer(resources, "maximum_artifact_bytes"),
    )
    if (data_config.pilot_per_candidate, data_config.development_per_family, data_config.sealed_per_family,
        data_config.maximum_attempts_per_example, data_config.maximum_absolute_expression_answer,
        data_config.graph_edge_probability, data_config.expression_leaf_counts, data_config.list_item_counts,
        data_config.list_operation_counts, data_config.graph_vertex_counts) != (30, 100, 200, 100, 9999, 0.2,
        (6, 7), (4, 5), (1, 2), (7, 8)):
        raise ValueError("data candidate grid differs from the registered V1 protocol")
    if pilot_seeds != {"expressions": (621001, 621002), "lists": (622001, 622002), "graphs": (623001, 623002)}:
        raise ValueError("pilot seeds differ from the registered V1 protocol")
    if data_config.development_seeds != {"expressions": 631001, "lists": 631002, "graphs": 631003}:
        raise ValueError("development seeds differ from the registered V1 protocol")
    if data_config.sealed_seeds != {"expressions": 641001, "lists": 641002, "graphs": 641003}:
        raise ValueError("sealed seeds differ from the registered V1 protocol")
    if (lora_config.rank, lora_config.alpha, lora_config.dropout, lora_config.seeds,
        lora_config.target_modules) != (8, 16, 0.0, (47, 48), _TARGET_MODULES):
        raise ValueError("LoRA settings differ from the registered V0 target configuration")
    if (optimizer_config.name, optimizer_config.updates, optimizer_config.warmup_updates, optimizer_config.learning_rate,
        optimizer_config.beta1, optimizer_config.beta2, optimizer_config.epsilon,
        optimizer_config.weight_decay, optimizer_config.max_grad_norm, optimizer_config.microbatch_size,
        optimizer_config.accumulation_steps) != ("AdamW", 1000, 20, 1e-4, 0.9, 0.95, 1e-8, 0.0, 1.0, 1, 8):
        raise ValueError("optimizer hyperparameters differ from the registered V0 protocol")
    if (resource_config.maximum_calibration_cuda_seconds, resource_config.maximum_study_cuda_seconds,
        resource_config.maximum_artifact_bytes) != (7200, 86400, 21474836480):
        raise ValueError("resource limits differ from the registered V1 protocol")
    if source_run != "artifacts/recurrence-v0/129d39646933" or output_root != "artifacts/recurrence-v1":
        raise ValueError("source or output path differs from the registered V1 protocol")
    return V1Config(source_run, "Qwen/Qwen3-1.7B", revision, dtype, attention, start, stop, cache,
                    thinking, greedy, output_tokens, sequence_tokens, data_config, lora_config,
                    optimizer_config, resource_config)


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


def _field(raw: Mapping[str, object], key: str) -> object:
    return raw[key]


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = _field(raw, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(raw: Mapping[str, object], key: str) -> float:
    value = _field(raw, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _boolean(raw: Mapping[str, object], key: str) -> bool:
    value = _field(raw, key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _string(raw: Mapping[str, object], key: str) -> str:
    value = _field(raw, key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _strings(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = _field(raw, key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a string array")
    return tuple(value)


def _integer_array(raw: Mapping[str, object], key: str, length: int) -> tuple[int, ...]:
    value = _field(raw, key)
    if not isinstance(value, list) or len(value) != length or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{key} must contain {length} integers")
    return tuple(value)


def _pair(raw: Mapping[str, object], key: str) -> tuple[int, int]:
    return _integer_array(raw, key, 2)  # type: ignore[return-value]


def _triple(raw: Mapping[str, object], key: str) -> tuple[int, int, int]:
    return _integer_array(raw, key, 3)  # type: ignore[return-value]


def _valid_relative_path(value: str, name: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not value:
        raise ValueError(f"{name} must be a safe relative POSIX path")
