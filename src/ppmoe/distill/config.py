"""Strict configuration for the registered verified-distillation source stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelPin:
    repo_id: str
    revision: str
    license: str


@dataclass(frozen=True, slots=True)
class DatasetPin:
    repo_id: str
    revision: str
    license: str
    train_file: str | None = None
    test_file: str | None = None
    url: str | None = None


@dataclass(frozen=True, slots=True)
class PromptConfig:
    template: str
    system_message: str
    few_shot_examples: int
    thinking: bool
    greedy: bool
    candidates: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    quantization: str
    double_quantization: bool
    compute_dtype: str
    target_modules: str
    gradient_checkpointing: bool
    optimizer: str
    scheduler: str
    rank: int
    alpha: int
    dropout: float
    learning_rate: float
    updates: int
    warmup_updates: int
    microbatch_size: int
    accumulation_steps: int
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GsmTrainPartition:
    calibration: int
    pilot: int
    development: int
    pool: int


@dataclass(frozen=True, slots=True)
class PartitionConfig:
    gsm_train: GsmTrainPartition
    gsm_test_sealed: int
    svamp_transfer_development: int
    svamp_sealed_transfer: int


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    calibration_questions: int
    minimum_calibration_eos: int
    pilot_questions: int
    minimum_teacher_accuracy: float
    minimum_teacher_advantage: float
    generation_pool: int
    minimum_accepted: int
    rationale_audit_size: int
    rationale_audit_minimum_valid: int
    maximum_systematic_error_count: int
    maximum_parser_reviewer_disagreement: float
    maximum_invalid_rate: float


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    cuda_seconds: int
    wall_seconds: int
    download_bytes: int
    artifact_bytes: int
    vram_reserve_bytes: int
    phase_retries: int


@dataclass(frozen=True, slots=True)
class DistillConfig:
    student: ModelPin
    teacher: ModelPin
    gsm8k: DatasetPin
    svamp: DatasetPin
    prompt: PromptConfig
    training: TrainingConfig
    partition: PartitionConfig
    qualification: QualificationConfig
    resources: ResourceConfig
    data_cache: str
    retention_manifest: str
    retention_manifest_sha256: str
    retention_chunks: str
    retention_chunks_sha256: str


_KEYS: dict[str, set[str]] = {
    "student": {"repo_id", "revision", "license"},
    "teacher": {"repo_id", "revision", "license"},
    "gsm8k": {"repo_id", "revision", "license", "train_file", "test_file"},
    "svamp": {"repo_id", "revision", "license", "url"},
    "prompt": {"template", "system_message", "few_shot_examples", "thinking", "greedy", "candidates"},
    "training": {
        "quantization", "double_quantization", "compute_dtype", "target_modules",
        "gradient_checkpointing", "optimizer", "scheduler", "rank", "alpha", "dropout",
        "learning_rate", "updates", "warmup_updates", "microbatch_size", "accumulation_steps", "seeds",
    },
    "partition": {
        "gsm_train", "gsm_test_sealed", "svamp_transfer_development", "svamp_sealed_transfer",
    },
    "qualification": {
        "calibration_questions", "minimum_calibration_eos", "pilot_questions",
        "minimum_teacher_accuracy", "minimum_teacher_advantage", "generation_pool",
        "minimum_accepted", "rationale_audit_size", "rationale_audit_minimum_valid",
        "maximum_systematic_error_count",
        "maximum_parser_reviewer_disagreement", "maximum_invalid_rate",
    },
    "resources": {"cuda_seconds", "wall_seconds", "download_bytes", "artifact_bytes", "vram_reserve_bytes", "phase_retries"},
    "paths": {
        "data_cache", "retention_manifest", "retention_manifest_sha256", "retention_chunks",
        "retention_chunks_sha256",
    },
}


def _exact_keys(raw: dict[str, Any]) -> None:
    if set(raw) != set(_KEYS):
        raise ValueError(f"unknown or missing config sections: {sorted(set(raw) ^ set(_KEYS))}")
    for section, expected in _KEYS.items():
        value = raw[section]
        if not isinstance(value, dict):
            raise ValueError(f"config section {section} must be a table")
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise ValueError(
                f"unknown or missing keys in {section}: {sorted(unknown | missing)}"
            )


def _expect(raw: dict[str, Any], name: str, kind: type, section: str) -> Any:
    value = raw[section][name]
    expected_name = kind.__name__ if isinstance(kind, type) else "numeric"
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise ValueError(f"{section}.{name} has invalid type; expected {expected_name}")
    return value


def _pin(raw: dict[str, Any], section: str, *, dataset: bool = False) -> ModelPin | DatasetPin:
    table = raw[section]
    if dataset:
        return DatasetPin(
            repo_id=_expect(raw, "repo_id", str, section),
            revision=_expect(raw, "revision", str, section),
            license=_expect(raw, "license", str, section),
            train_file=table.get("train_file"),
            test_file=table.get("test_file"),
            url=table.get("url"),
        )
    return ModelPin(
        repo_id=_expect(raw, "repo_id", str, section),
        revision=_expect(raw, "revision", str, section),
        license=_expect(raw, "license", str, section),
    )


def load_config(path: Path) -> DistillConfig:
    """Load the only registered v1 config, rejecting schema and scientific drift."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    _exact_keys(raw)
    partition = raw["partition"]
    qualification = raw["qualification"]
    resources = raw["resources"]
    candidate_value = _expect(raw, "candidates", list, "prompt")
    if any(
        not isinstance(pair, list) or len(pair) != 2
        or any(not isinstance(n, int) or isinstance(n, bool) for n in pair)
        for pair in candidate_value
    ):
        raise ValueError("prompt.candidates has invalid type")
    train_counts = partition["gsm_train"]
    if not isinstance(train_counts, dict) or set(train_counts) != {
        "calibration", "pilot", "development", "pool"
    } or any(not isinstance(n, int) or isinstance(n, bool) for n in train_counts.values()):
        raise ValueError("partition.gsm_train has invalid type or keys")

    for section, keys in {
        "student": ("repo_id", "revision", "license"),
        "teacher": ("repo_id", "revision", "license"),
        "gsm8k": ("repo_id", "revision", "license", "train_file", "test_file"),
        "svamp": ("repo_id", "revision", "license", "url"),
    }.items():
        for key in keys:
            if not isinstance(raw[section][key], str):
                raise ValueError(f"{section}.{key} has invalid type")

    config = DistillConfig(
        student=_pin(raw, "student"),
        teacher=_pin(raw, "teacher"),
        gsm8k=_pin(raw, "gsm8k", dataset=True),
        svamp=_pin(raw, "svamp", dataset=True),
        prompt=PromptConfig(
            template=_expect(raw, "template", str, "prompt"),
            system_message=_expect(raw, "system_message", str, "prompt"),
            few_shot_examples=_expect(raw, "few_shot_examples", int, "prompt"),
            thinking=_expect(raw, "thinking", bool, "prompt"),
            greedy=_expect(raw, "greedy", bool, "prompt"),
            candidates=tuple(tuple(pair) for pair in candidate_value),
        ),
        training=TrainingConfig(
            quantization=_expect(raw, "quantization", str, "training"),
            double_quantization=_expect(raw, "double_quantization", bool, "training"),
            compute_dtype=_expect(raw, "compute_dtype", str, "training"),
            target_modules=_expect(raw, "target_modules", str, "training"),
            gradient_checkpointing=_expect(raw, "gradient_checkpointing", bool, "training"),
            optimizer=_expect(raw, "optimizer", str, "training"),
            scheduler=_expect(raw, "scheduler", str, "training"),
            rank=_expect(raw, "rank", int, "training"),
            alpha=_expect(raw, "alpha", int, "training"),
            dropout=_expect(raw, "dropout", float, "training"),
            learning_rate=_expect(raw, "learning_rate", float, "training"),
            updates=_expect(raw, "updates", int, "training"),
            warmup_updates=_expect(raw, "warmup_updates", int, "training"),
            microbatch_size=_expect(raw, "microbatch_size", int, "training"),
            accumulation_steps=_expect(raw, "accumulation_steps", int, "training"),
            seeds=tuple(_expect(raw, "seeds", list, "training")),
        ),
        partition=PartitionConfig(
            gsm_train=GsmTrainPartition(**train_counts),
            gsm_test_sealed=_expect(raw, "gsm_test_sealed", int, "partition"),
            svamp_transfer_development=_expect(raw, "svamp_transfer_development", int, "partition"),
            svamp_sealed_transfer=_expect(raw, "svamp_sealed_transfer", int, "partition"),
        ),
        qualification=QualificationConfig(
            **{key: _expect(raw, key, int if key in {
                "calibration_questions", "minimum_calibration_eos", "pilot_questions", "generation_pool",
                "minimum_accepted", "rationale_audit_size", "rationale_audit_minimum_valid",
                "maximum_systematic_error_count",
            } else float, "qualification") for key in qualification}
        ),
        resources=ResourceConfig(
            **{key: _expect(raw, key, int, "resources") for key in resources}
        ),
        data_cache=_expect(raw, "data_cache", str, "paths"),
        retention_manifest=_expect(raw, "retention_manifest", str, "paths"),
        retention_manifest_sha256=_expect(raw, "retention_manifest_sha256", str, "paths"),
        retention_chunks=_expect(raw, "retention_chunks", str, "paths"),
        retention_chunks_sha256=_expect(raw, "retention_chunks_sha256", str, "paths"),
    )
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in config.training.seeds):
        raise ValueError("training.seeds has invalid type")
    if config.prompt.candidates != ((256, 768), (512, 1024), (768, 1536)):
        raise ValueError("registered token budget candidates drifted")
    if any(
        getattr(config, section).revision != expected_revision
        for section, expected_revision in (
            ("student", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
            ("teacher", "1cfa9a7208912126459214e8b04321603b3df60c"),
            ("gsm8k", "740312add88f781978c0658806c59bc2815b9866"),
            ("svamp", "78e727689e1c1bebfc4be39c446898e8e10b0518"),
        )
    ):
        raise ValueError("registered source revision drift")
    expected = _registered()
    actual = asdict(config)
    if actual != expected:
        raise ValueError("config differs from registered values")
    return config


def _registered() -> dict[str, Any]:
    return {
        "student": {"repo_id": "Qwen/Qwen3-1.7B", "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "license": "Apache-2.0"},
        "teacher": {"repo_id": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c", "license": "Apache-2.0"},
        "gsm8k": {"repo_id": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866", "license": "MIT", "train_file": "main/train-00000-of-00001.parquet", "test_file": "main/test-00000-of-00001.parquet", "url": None},
        "svamp": {"repo_id": "arkilpatel/SVAMP", "revision": "78e727689e1c1bebfc4be39c446898e8e10b0518", "license": "MIT", "train_file": None, "test_file": None, "url": "https://raw.githubusercontent.com/arkilpatel/SVAMP/78e727689e1c1bebfc4be39c446898e8e10b0518/SVAMP.json"},
        "prompt": {"template": "Solve the following problem. Show concise working, then finish with Final answer: <number>.\n\n{question}", "system_message": "", "few_shot_examples": 0, "thinking": False, "greedy": True, "candidates": ((256, 768), (512, 1024), (768, 1536))},
        "training": {"quantization": "nf4", "double_quantization": True, "compute_dtype": "bfloat16", "target_modules": "all-linear", "gradient_checkpointing": True, "optimizer": "AdamW", "scheduler": "cosine", "rank": 16, "alpha": 32, "dropout": 0.0, "learning_rate": 0.0001, "updates": 300, "warmup_updates": 10, "microbatch_size": 1, "accumulation_steps": 8, "seeds": (47, 48)},
        "partition": {"gsm_train": {"calibration": 60, "pilot": 120, "development": 200, "pool": 1200}, "gsm_test_sealed": 400, "svamp_transfer_development": 200, "svamp_sealed_transfer": 500},
        "qualification": {"calibration_questions": 60, "minimum_calibration_eos": 57, "pilot_questions": 120, "minimum_teacher_accuracy": 0.60, "minimum_teacher_advantage": 0.05, "generation_pool": 1200, "minimum_accepted": 800, "rationale_audit_size": 60, "rationale_audit_minimum_valid": 54, "maximum_systematic_error_count": 2, "maximum_parser_reviewer_disagreement": 0.05, "maximum_invalid_rate": 0.10},
        "resources": {"cuda_seconds": 86400, "wall_seconds": 172800, "download_bytes": 21474836480, "artifact_bytes": 21474836480, "vram_reserve_bytes": 2147483648, "phase_retries": 2},
        "data_cache": ".cache/distillation-v1/data",
        "retention_manifest": "artifacts/recurrence-v0/129d39646933/retention/retention-manifest.json",
        "retention_manifest_sha256": "52a88366a7402e51848006a2bce13b02fd1db709ed0887a5f25fddd9fc98fbc3",
        "retention_chunks": "artifacts/recurrence-v0/129d39646933/retention/retention-chunks.jsonl",
        "retention_chunks_sha256": "84cb2c4ca17ec33884144d30d1a8f1f3c7a454440c108564b67a7fab8de2a991",
    }


def canonical_config(config: DistillConfig) -> dict[str, Any]:
    """Return stable JSON-compatible config values for signatures/manifests."""
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(value[key]) for key in sorted(value)}
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if hasattr(value, "__dataclass_fields__"):
            return normalize(asdict(value))
        return value

    return normalize(config)
