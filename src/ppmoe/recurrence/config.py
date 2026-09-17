"""Typed, frozen configuration for the registered recurrence V0 protocol."""

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Mapping


REGISTERED_SCHEMA: dict[str, dict[str, None]] = {
    "model": {
        "repo_id": None,
        "model_class": None,
        "num_hidden_layers": None,
        "hidden_size": None,
        "intermediate_size": None,
        "num_attention_heads": None,
        "num_key_value_heads": None,
        "head_dim": None,
        "sliding_attention": None,
    },
    "loop": {
        "layer_start": None,
        "layer_stop": None,
        "frozen_mode": None,
        "repeat_mode": None,
        "damped_mode": None,
    },
    "precision": {
        "frozen_dtype": None,
        "activation_dtype": None,
        "lora_dtype": None,
        "optimizer_state_dtype": None,
        "attention_implementation": None,
        "torch_compile": None,
        "quantization": None,
    },
    "lora": {"rank": None, "alpha": None, "dropout": None, "target_modules": None},
    "optimizer": {
        "learning_rate": None,
        "beta1": None,
        "beta2": None,
        "epsilon": None,
        "weight_decay": None,
        "max_grad_norm": None,
        "warmup_updates": None,
        "updates": None,
        "microbatch_size": None,
        "accumulation_steps": None,
    },
    "data": {
        "train_examples": None,
        "dev_examples_per_family": None,
        "test_examples_per_family": None,
        "train_expression_examples": None,
        "train_list_examples": None,
        "train_graph_examples": None,
        "train_seeds": None,
        "dev_seeds": None,
        "test_seeds": None,
        "max_sequence_tokens": None,
    },
    "decoding": {
        "use_cache": None,
        "past_key_values": None,
        "enable_thinking": None,
        "max_new_tokens": None,
        "greedy": None,
        "system_message": None,
    },
    "retention": {
        "dataset_repo_id": None,
        "tokens": None,
        "chunk_tokens": None,
        "eos_between_documents": None,
    },
    "statistics": {
        "accuracy_bootstrap_seed": None,
        "retention_bootstrap_seed": None,
        "bootstrap_replicates": None,
        "family_accuracy_lower": None,
        "family_accuracy_upper": None,
        "maximum_family_invalid_rate": None,
        "minimum_informative_families": None,
        "discovery_macro_improvement": None,
        "maximum_family_regression": None,
        "maximum_retention_ppl_ratio": None,
    },
    "resources": {
        "artifact_root": None,
        "maximum_cuda_seconds": None,
        "maximum_download_bytes": None,
        "maximum_artifact_bytes": None,
        "reserved_vram_bytes": None,
    },
    "stages": {
        "discovery_seed": None,
        "confirmation_seed": None,
        "timing_pilot_updates": None,
        "resource_pilot_updates": None,
        "checkpoint_interval_updates": None,
        "progress_interval_updates": None,
    },
}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    repo_id: str
    model_class: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    sliding_attention: bool


@dataclass(frozen=True, slots=True)
class LoopConfig:
    layer_start: int
    layer_stop: int
    frozen_mode: str
    repeat_mode: str
    damped_mode: str


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    frozen_dtype: str
    activation_dtype: str
    lora_dtype: str
    optimizer_state_dtype: str
    attention_implementation: str
    torch_compile: bool
    quantization: bool


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    max_grad_norm: float
    warmup_updates: int
    updates: int
    microbatch_size: int
    accumulation_steps: int


@dataclass(frozen=True, slots=True)
class DataConfig:
    train_examples: int
    dev_examples_per_family: int
    test_examples_per_family: int
    train_expression_examples: int
    train_list_examples: int
    train_graph_examples: int
    train_seeds: tuple[int, ...]
    dev_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    max_sequence_tokens: int


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    use_cache: bool
    past_key_values: bool
    enable_thinking: bool
    max_new_tokens: int
    greedy: bool
    system_message: str


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    dataset_repo_id: str
    tokens: int
    chunk_tokens: int
    eos_between_documents: bool


@dataclass(frozen=True, slots=True)
class StatisticsConfig:
    accuracy_bootstrap_seed: int
    retention_bootstrap_seed: int
    bootstrap_replicates: int
    family_accuracy_lower: float
    family_accuracy_upper: float
    maximum_family_invalid_rate: float
    minimum_informative_families: int
    discovery_macro_improvement: float
    maximum_family_regression: float
    maximum_retention_ppl_ratio: float


@dataclass(frozen=True, slots=True)
class ResourcesConfig:
    artifact_root: str
    maximum_cuda_seconds: float
    maximum_download_bytes: int
    maximum_artifact_bytes: int
    reserved_vram_bytes: int


@dataclass(frozen=True, slots=True)
class StagesConfig:
    discovery_seed: int
    confirmation_seed: int
    timing_pilot_updates: int
    resource_pilot_updates: int
    checkpoint_interval_updates: int
    progress_interval_updates: int


@dataclass(frozen=True, slots=True)
class RecurrenceConfig:
    model: ModelConfig
    loop: LoopConfig
    precision: PrecisionConfig
    lora: LoRAConfig
    optimizer: OptimizerConfig
    data: DataConfig
    decoding: DecodingConfig
    retention: RetentionConfig
    statistics: StatisticsConfig
    resources: ResourcesConfig
    stages: StagesConfig

    @classmethod
    def load(cls, path: Path) -> "RecurrenceConfig":
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        validate_exact_keys(raw, REGISTERED_SCHEMA)
        config = cls.from_mapping(raw)
        config.validate_registered_values()
        return config

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "RecurrenceConfig":
        model = _table(raw, "model")
        loop = _table(raw, "loop")
        precision = _table(raw, "precision")
        lora = _table(raw, "lora")
        optimizer = _table(raw, "optimizer")
        data = _table(raw, "data")
        decoding = _table(raw, "decoding")
        retention = _table(raw, "retention")
        statistics = _table(raw, "statistics")
        resources = _table(raw, "resources")
        stages = _table(raw, "stages")
        return cls(
            model=ModelConfig(
                _string(model, "repo_id"), _string(model, "model_class"),
                _integer(model, "num_hidden_layers"), _integer(model, "hidden_size"),
                _integer(model, "intermediate_size"), _integer(model, "num_attention_heads"),
                _integer(model, "num_key_value_heads"), _integer(model, "head_dim"),
                _boolean(model, "sliding_attention"),
            ),
            loop=LoopConfig(
                _integer(loop, "layer_start"), _integer(loop, "layer_stop"),
                _string(loop, "frozen_mode"), _string(loop, "repeat_mode"),
                _string(loop, "damped_mode"),
            ),
            precision=PrecisionConfig(
                _string(precision, "frozen_dtype"), _string(precision, "activation_dtype"),
                _string(precision, "lora_dtype"), _string(precision, "optimizer_state_dtype"),
                _string(precision, "attention_implementation"), _boolean(precision, "torch_compile"),
                _boolean(precision, "quantization"),
            ),
            lora=LoRAConfig(
                _integer(lora, "rank"), _integer(lora, "alpha"), _number(lora, "dropout"),
                _strings(lora, "target_modules"),
            ),
            optimizer=OptimizerConfig(
                _number(optimizer, "learning_rate"), _number(optimizer, "beta1"),
                _number(optimizer, "beta2"), _number(optimizer, "epsilon"),
                _number(optimizer, "weight_decay"), _number(optimizer, "max_grad_norm"),
                _integer(optimizer, "warmup_updates"), _integer(optimizer, "updates"),
                _integer(optimizer, "microbatch_size"), _integer(optimizer, "accumulation_steps"),
            ),
            data=DataConfig(
                _integer(data, "train_examples"), _integer(data, "dev_examples_per_family"),
                _integer(data, "test_examples_per_family"), _integer(data, "train_expression_examples"),
                _integer(data, "train_list_examples"), _integer(data, "train_graph_examples"),
                _integers(data, "train_seeds"), _integers(data, "dev_seeds"),
                _integers(data, "test_seeds"), _integer(data, "max_sequence_tokens"),
            ),
            decoding=DecodingConfig(
                _boolean(decoding, "use_cache"), _boolean(decoding, "past_key_values"),
                _boolean(decoding, "enable_thinking"), _integer(decoding, "max_new_tokens"),
                _boolean(decoding, "greedy"), _string(decoding, "system_message"),
            ),
            retention=RetentionConfig(
                _string(retention, "dataset_repo_id"), _integer(retention, "tokens"),
                _integer(retention, "chunk_tokens"), _boolean(retention, "eos_between_documents"),
            ),
            statistics=StatisticsConfig(
                _integer(statistics, "accuracy_bootstrap_seed"),
                _integer(statistics, "retention_bootstrap_seed"),
                _integer(statistics, "bootstrap_replicates"),
                _number(statistics, "family_accuracy_lower"),
                _number(statistics, "family_accuracy_upper"),
                _number(statistics, "maximum_family_invalid_rate"),
                _integer(statistics, "minimum_informative_families"),
                _number(statistics, "discovery_macro_improvement"),
                _number(statistics, "maximum_family_regression"),
                _number(statistics, "maximum_retention_ppl_ratio"),
            ),
            resources=ResourcesConfig(
                _string(resources, "artifact_root"), _number(resources, "maximum_cuda_seconds"),
                _integer(resources, "maximum_download_bytes"),
                _integer(resources, "maximum_artifact_bytes"), _integer(resources, "reserved_vram_bytes"),
            ),
            stages=StagesConfig(
                _integer(stages, "discovery_seed"), _integer(stages, "confirmation_seed"),
                _integer(stages, "timing_pilot_updates"), _integer(stages, "resource_pilot_updates"),
                _integer(stages, "checkpoint_interval_updates"), _integer(stages, "progress_interval_updates"),
            ),
        )

    def validate_registered_values(self) -> None:
        actual = asdict(self)
        for section, expected_values in REGISTERED_VALUES.items():
            for key, expected in expected_values.items():
                if actual[section][key] != expected:
                    raise ValueError(f"locked recurrence-v0 protocol rejects {section}.{key}")

    def scientific_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["resources"].pop("artifact_root")
        return payload


REGISTERED_VALUES: dict[str, dict[str, object]] = {
    "model": {"repo_id": "Qwen/Qwen3-1.7B", "model_class": "Qwen3ForCausalLM", "num_hidden_layers": 28, "hidden_size": 2048, "intermediate_size": 6144, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "sliding_attention": False},
    "loop": {"layer_start": 12, "layer_stop": 16, "frozen_mode": "single", "repeat_mode": "repeat", "damped_mode": "damped"},
    "precision": {"frozen_dtype": "bfloat16", "activation_dtype": "bfloat16", "lora_dtype": "float32", "optimizer_state_dtype": "float32", "attention_implementation": "eager", "torch_compile": False, "quantization": False},
    "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")},
    "optimizer": {"learning_rate": 0.0001, "beta1": 0.9, "beta2": 0.95, "epsilon": 0.00000001, "weight_decay": 0.0, "max_grad_norm": 1.0, "warmup_updates": 20, "updates": 1000, "microbatch_size": 1, "accumulation_steps": 8},
    "data": {"train_examples": 8000, "dev_examples_per_family": 200, "test_examples_per_family": 800, "train_expression_examples": 2667, "train_list_examples": 2667, "train_graph_examples": 2666, "train_seeds": (101, 102, 103), "dev_seeds": (201, 202, 203), "test_seeds": (301, 302, 303), "max_sequence_tokens": 512},
    "decoding": {"use_cache": False, "past_key_values": False, "enable_thinking": False, "max_new_tokens": 32, "greedy": True, "system_message": ""},
    "retention": {"dataset_repo_id": "HuggingFaceFW/fineweb-edu", "tokens": 20000, "chunk_tokens": 512, "eos_between_documents": True},
    "statistics": {"accuracy_bootstrap_seed": 9047, "retention_bootstrap_seed": 9157, "bootstrap_replicates": 10000, "family_accuracy_lower": 0.05, "family_accuracy_upper": 0.95, "maximum_family_invalid_rate": 0.10, "minimum_informative_families": 2, "discovery_macro_improvement": 0.02, "maximum_family_regression": 0.02, "maximum_retention_ppl_ratio": 1.02},
    "resources": {"artifact_root": "artifacts/recurrence-v0", "maximum_cuda_seconds": 86400.0, "maximum_download_bytes": 8589934592, "maximum_artifact_bytes": 21474836480, "reserved_vram_bytes": 2147483648},
    "stages": {"discovery_seed": 47, "confirmation_seed": 48, "timing_pilot_updates": 20, "resource_pilot_updates": 25, "checkpoint_interval_updates": 50, "progress_interval_updates": 10},
}


def validate_exact_keys(
    raw: Mapping[str, object], schema: Mapping[str, Mapping[str, object]], path: str = ""
) -> None:
    unknown = set(raw) - set(schema)
    if unknown:
        raise ValueError(f"unknown configuration key {path}{sorted(unknown)[0]}")
    missing = set(schema) - set(raw)
    if missing:
        raise ValueError(f"missing configuration key {path}{sorted(missing)[0]}")
    for key, child_schema in schema.items():
        if child_schema is None:
            continue
        value = raw[key]
        if not isinstance(value, Mapping):
            raise ValueError(f"configuration section {path}{key} must be a table")
        validate_exact_keys(value, child_schema, f"{path}{key}.")


def _table(raw: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = raw[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {name} must be a table")
    return value


def _field(values: Mapping[str, object], key: str) -> object:
    return values[key]


def _integer(values: Mapping[str, object], key: str) -> int:
    value = _field(values, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{_section_field(values, key)} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = _field(values, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{_section_field(values, key)} must be a number")
    return float(value)


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = _field(values, key)
    if not isinstance(value, bool):
        raise ValueError(f"{_section_field(values, key)} must be a boolean")
    return value


def _string(values: Mapping[str, object], key: str) -> str:
    value = _field(values, key)
    if not isinstance(value, str):
        raise ValueError(f"{_section_field(values, key)} must be a string")
    return value


def _integers(values: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = _field(values, key)
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{_section_field(values, key)} must be an integer array")
    return tuple(value)


def _strings(values: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = _field(values, key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{_section_field(values, key)} must be a string array")
    return tuple(value)


def _section_field(values: Mapping[str, object], key: str) -> str:
    for section, section_values in REGISTERED_VALUES.items():
        if set(values) == set(section_values):
            return f"{section}.{key}"
    return key
