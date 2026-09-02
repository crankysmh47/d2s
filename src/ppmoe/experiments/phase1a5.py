"""Gated Phase 1A.5 calibration orchestration.

Only source verification, smoke, local calibration, and frozen selection are
implemented here.  Rollout and report retain their gates but deliberately do
not execute until their separately reviewed implementations land.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor, nn

from ppmoe.adapters.qwen35_runtime import iter_text_mlps, load_qwen35_pair
from ppmoe.data.token_stream import iter_evaluation_batches
from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.experiments.phase1a5_provenance import (
    phase1a5_signature,
    verify_phase1a_sources,
)
from ppmoe.experiments.phase1a5_selection import select_calibration_conditions
from ppmoe.moefication.qwen35 import PartitionedQwen35MLP, convert_qwen35_text_ffns
from ppmoe.oracle.exact import exact_gram_oracle
from ppmoe.oracle.reconstruction import (
    FamilyBenchmark,
    ReconstructionCondition,
    SolverSettings,
    score_reconstruction,
)
from ppmoe.oracle.reconstruction_local import (
    ConditionMetrics,
    ReconstructionLocalResult,
    run_reconstruction_local,
)
from ppmoe.provenance import atomic_write_json, sha256_file


class StageError(RuntimeError):
    """A registered experiment gate did not pass."""


class StagePrerequisiteError(StageError):
    """A required earlier stage is absent, stopped, or incompatible."""


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    return value


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _config_signature(config: Phase1A5Config) -> str:
    return _canonical_hash({"config": asdict(config)})


def _stage_path(artifact_root: Path, stage: str) -> Path:
    return artifact_root / "stages" / f"{stage}.json"


def _write_stage(
    artifact_root: Path,
    stage: str,
    config_signature: str,
    input_signature: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    state = {
        "stage": stage,
        "status": "complete",
        "config_signature": config_signature,
        "input_signature": input_signature,
        **_json_safe(payload),
    }
    atomic_write_json(_stage_path(artifact_root, stage), state)
    return state


def _read_stage(
    artifact_root: Path,
    stage: str,
    config_signature: str,
    *,
    allow_scientific_stop: bool = False,
) -> dict[str, Any]:
    path = _stage_path(artifact_root, stage)
    if not path.exists():
        raise StagePrerequisiteError(f"required stage {stage} is absent")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StagePrerequisiteError(f"required stage {stage} is invalid") from error
    if state.get("config_signature") != config_signature:
        raise StagePrerequisiteError(f"required stage {stage} signature does not match")
    allowed = (
        {"complete", "scientifically_stopped"}
        if allow_scientific_stop
        else {"complete"}
    )
    if state.get("status") not in allowed:
        raise StagePrerequisiteError(f"required stage {stage} is incomplete")
    return state


def _source_hashes(source_manifest: Mapping[str, object]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for group in ("stage_sha256", "dataset_manifest_sha256", "model_manifest_sha256"):
        values = source_manifest.get(group)
        if not isinstance(values, Mapping):
            raise StageError(f"source manifest is missing {group}")
        for name, value in values.items():
            if not isinstance(value, str) or not value:
                raise StageError(f"source manifest has invalid {group} value")
            prefix = {
                "stage_sha256": "stage",
                "dataset_manifest_sha256": "dataset",
                "model_manifest_sha256": "model",
            }[group]
            hashes[f"{prefix}:{name}"] = value
    return hashes


def _source_signature(
    config: Phase1A5Config, source_manifest: Mapping[str, object], microbatch_size: int
) -> str:
    revision = source_manifest.get("code_revision")
    if not isinstance(revision, Mapping) or not isinstance(revision.get("commit"), str):
        raise StageError("source manifest is missing the verified code revision")
    return phase1a5_signature(
        config,
        source_manifest,
        config.solver_version,
        str(revision["commit"]),
        microbatch_size,
    )


def _execution_signature(
    config: Phase1A5Config,
    source_manifest: Mapping[str, object],
    microbatch_size: int,
    *,
    stage: str,
    token_hash: str,
    extra: Mapping[str, object] | None = None,
) -> str:
    return _canonical_hash(
        {
            "phase1a5": _source_signature(config, source_manifest, microbatch_size),
            "stage": stage,
            "tokens": token_hash,
            "extra": dict(extra or {}),
        }
    )


def _load_calibration_tokens(
    source_root: Path, source_manifest: Mapping[str, object]
) -> tuple[Tensor, Tensor]:
    """Load the pinned calibration tensors only after rechecking their manifest."""

    manifest_path = source_root / "tokens" / "calibration" / "manifest.json"
    datasets = source_manifest.get("dataset_manifest_sha256")
    if not isinstance(datasets, Mapping) or sha256_file(manifest_path) != datasets.get(
        "calibration"
    ):
        raise StageError("calibration manifest does not match verified source hash")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_ids = torch.load(
            manifest_path.parent / "input_ids.pt", map_location="cpu", weights_only=True
        )
        attention_mask = torch.load(
            manifest_path.parent / "attention_mask.pt",
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        raise StageError("unable to load verified calibration tensors") from error
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(manifest.get("input_ids_sha256"), str)
        or not isinstance(manifest.get("attention_mask_sha256"), str)
    ):
        raise StageError("calibration manifest is malformed")
    if (
        sha256_file(manifest_path.parent / "input_ids.pt")
        != manifest["input_ids_sha256"]
        or sha256_file(manifest_path.parent / "attention_mask.pt")
        != manifest["attention_mask_sha256"]
    ):
        raise StageError("calibration tensor hash does not match manifest")
    if (
        input_ids.ndim != 2
        or attention_mask.shape != input_ids.shape
        or int(attention_mask.sum()) != int(manifest.get("total_tokens", -1))
    ):
        raise StageError("calibration tensors do not match manifest token count")
    return input_ids, attention_mask


def _first_valid_tokens(
    input_ids: Tensor, attention_mask: Tensor, count: int
) -> tuple[Tensor, Tensor]:
    valid = input_ids[attention_mask.to(dtype=torch.bool)]
    if valid.numel() < count:
        raise StageError("calibration stream is shorter than the required token count")
    width = input_ids.shape[1]
    contexts = (count + width - 1) // width
    ids = torch.zeros((contexts, width), dtype=input_ids.dtype)
    mask = torch.zeros_like(ids)
    ids.reshape(-1)[:count] = valid[:count]
    mask.reshape(-1)[:count] = 1
    return ids, mask


def _all_conditions(
    config: Phase1A5Config, *, include_bounded: bool
) -> dict[int, tuple[ReconstructionCondition, ...]]:
    conditions = [ReconstructionCondition("unit")]
    conditions.extend(
        ReconstructionCondition("ridge", rho) for rho in config.rho_values
    )
    if include_bounded:
        conditions.append(ReconstructionCondition("bounded"))
    conditions.append(ReconstructionCondition("free_bound"))
    return {k: tuple(conditions) for k in config.k_values}


def _family_conditions(
    config: Phase1A5Config, family: str
) -> dict[int, tuple[ReconstructionCondition, ...]]:
    if family == "unit":
        values = (ReconstructionCondition("unit"),)
    elif family == "ridge":
        values = tuple(
            ReconstructionCondition("ridge", rho) for rho in config.rho_values
        )
    elif family == "bounded":
        values = (ReconstructionCondition("bounded"),)
    elif family == "free_bound":
        values = (ReconstructionCondition("free_bound"),)
    else:
        raise ValueError(f"unsupported reconstruction family: {family}")
    return {k: values for k in config.k_values}


def _solver_settings(config: Phase1A5Config) -> SolverSettings:
    return SolverSettings(
        tau_g=config.tau_g,
        epsilon_zero=config.epsilon_zero,
        rho_values=config.rho_values,
    )


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _default_unit_regression(
    teacher: nn.Module,
    student: nn.Module,
    batches: Iterable[object],
    config: Phase1A5Config,
) -> Mapping[str, object]:
    """Compare the untied unit solver to the independent exact Gram oracle."""

    teacher_mlps, student_mlps = (
        dict(iter_text_mlps(teacher)),
        dict(iter_text_mlps(student)),
    )
    if teacher_mlps.keys() != student_mlps.keys() or not teacher_mlps:
        raise StageError("unit regression model layer layouts differ")
    if not all(
        isinstance(value, PartitionedQwen35MLP) for value in student_mlps.values()
    ):
        raise StageError("unit regression requires converted student MLPs")
    checked_tokens = 0
    checked_token_layers = 0
    teacher_device, student_device = _model_device(teacher), _model_device(student)
    for batch in batches:
        captured: dict[int, tuple[Tensor, Tensor]] = {}
        handles: list[object] = []
        try:
            for layer, mlp in teacher_mlps.items():

                def capture(
                    _module: nn.Module,
                    args: tuple[Tensor, ...],
                    output: Tensor,
                    *,
                    layer_index: int = layer,
                ) -> None:
                    if (
                        not args
                        or not isinstance(args[0], Tensor)
                        or not isinstance(output, Tensor)
                    ):
                        raise TypeError(
                            "unit regression hook requires tensor input/output"
                        )
                    captured[layer_index] = (args[0].detach(), output.detach())

                handles.append(mlp.register_forward_hook(capture))
            with torch.inference_mode():
                teacher.model.language_model(
                    input_ids=batch.input_ids.to(teacher_device),
                    attention_mask=batch.attention_mask.to(teacher_device),
                    use_cache=False,
                )
            if captured.keys() != teacher_mlps.keys():
                raise StageError("unit regression teacher forward missed a text MLP")
            for layer, (teacher_input, target) in captured.items():
                experts = (
                    student_mlps[layer]
                    .expert_outputs(teacher_input.to(student_device))
                    .to(target.device)
                )
                valid = batch.attention_mask.to(
                    target.device, dtype=torch.bool
                ).reshape(-1)
                target = target.reshape(-1, target.shape[-1])[valid]
                experts = experts.reshape(-1, 8, experts.shape[-1])[valid]
                for k in config.k_values:
                    actual = score_reconstruction(
                        target,
                        experts,
                        k,
                        ReconstructionCondition("unit"),
                        _solver_settings(config),
                    )
                    expected = exact_gram_oracle(target, experts, k)
                    if not torch.equal(actual.subsets, expected.subsets):
                        raise StageError(
                            f"untied unit-subset regression failed at layer {layer}, K={k}"
                        )
                checked_token_layers += int(target.shape[0])
            checked_tokens += int(batch.attention_mask.sum())
        finally:
            for handle in handles:
                handle.remove()
    return {
        "passed": checked_tokens == 2_048,
        "checked_tokens": checked_tokens,
        "checked_token_layers": checked_token_layers,
    }


@dataclass(slots=True)
class Phase1A5Dependencies:
    """Replaceable slow boundaries; defaults perform the frozen experiment."""

    source_verifier: Callable[..., Mapping[str, object]] | None = None
    calibration_loader: Callable[..., tuple[Tensor, Tensor]] | None = None
    converted_pair_loader: Callable[[Phase1A5Config], tuple[Any, Any]] | None = None
    local_runner: Callable[..., ReconstructionLocalResult] | None = None
    unit_regression_runner: Callable[..., Mapping[str, object]] | None = None
    selection_runner: Callable[..., object] | None = None
    cuda_peak_probe: Callable[[], int] | None = None
    cuda_peak_reset: Callable[[], None] | None = None
    clock: Callable[[], float] | None = None


def _dependencies(values: Phase1A5Dependencies | None) -> Phase1A5Dependencies:
    values = values or Phase1A5Dependencies()
    if values.source_verifier is None:
        values.source_verifier = verify_phase1a_sources
    if values.calibration_loader is None:
        values.calibration_loader = _load_calibration_tokens
    if values.converted_pair_loader is None:

        def converted_pair_loader(config: Phase1A5Config) -> tuple[Any, Any]:
            pair = load_qwen35_pair(config.phase1_config, device=torch.device("cuda"))
            convert_qwen35_text_ffns(
                pair.student, num_experts=8, seed_base=config.phase1_config.seed_base
            )
            return pair.teacher, pair.student

        values.converted_pair_loader = converted_pair_loader
    if values.local_runner is None:
        values.local_runner = run_reconstruction_local
    if values.unit_regression_runner is None:
        values.unit_regression_runner = _default_unit_regression
    if values.selection_runner is None:
        values.selection_runner = select_calibration_conditions
    if values.cuda_peak_probe is None:
        values.cuda_peak_probe = lambda: (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
    if values.cuda_peak_reset is None:
        values.cuda_peak_reset = lambda: (
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        )
    if values.clock is None:
        values.clock = perf_counter
    return values


def _source_stage(
    config: Phase1A5Config,
    artifact_root: Path,
    microbatch_size: int,
    dependencies: Phase1A5Dependencies,
) -> dict[str, object]:
    assert dependencies.source_verifier is not None
    source_root = config.source_artifact_root.resolve()
    manifest = dependencies.source_verifier(source_root, artifact_root, config)
    signature = _source_signature(config, manifest, microbatch_size)
    return _write_stage(
        artifact_root,
        "source",
        _config_signature(config),
        signature,
        {
            "execution_microbatch_size": microbatch_size,
            "source_manifest": manifest,
            "source_hashes": _source_hashes(manifest),
        },
    )


def _source_from_stage(
    artifact_root: Path, config: Phase1A5Config
) -> tuple[dict[str, Any], Mapping[str, object], dict[str, str]]:
    state = _read_stage(artifact_root, "source", _config_signature(config))
    manifest = state.get("source_manifest")
    if not isinstance(manifest, Mapping):
        raise StagePrerequisiteError("source stage lacks its verified manifest")
    return state, manifest, _source_hashes(manifest)


def _diagnostic_count(diagnostics: Mapping[object, object], name: str) -> int:
    total = 0
    for value in diagnostics.values():
        if isinstance(value, Mapping):
            item = value.get(name, 0)
            if isinstance(item, (int, float)) and isfinite(float(item)):
                total += int(item)
    return total


def _smoke_stage(
    config: Phase1A5Config,
    artifact_root: Path,
    microbatch_size: int,
    dependencies: Phase1A5Dependencies,
) -> dict[str, object]:
    _, source_manifest, source_hashes = _source_from_stage(artifact_root, config)
    assert (
        dependencies.calibration_loader is not None
        and dependencies.converted_pair_loader is not None
    )
    assert (
        dependencies.local_runner is not None
        and dependencies.unit_regression_runner is not None
    )
    assert (
        dependencies.cuda_peak_probe is not None
        and dependencies.cuda_peak_reset is not None
        and dependencies.clock is not None
    )
    input_ids, attention_mask = dependencies.calibration_loader(
        Path(str(source_manifest["source_root"])), source_manifest
    )
    input_ids, attention_mask = _first_valid_tokens(input_ids, attention_mask, 2_048)
    token_hash = _canonical_hash(
        {"tokens": input_ids.tolist(), "mask": attention_mask.tolist()}
    )
    teacher, student = dependencies.converted_pair_loader(config)
    batches = tuple(
        iter_evaluation_batches(
            input_ids, attention_mask, microbatch_size=microbatch_size
        )
    )
    regression = dependencies.unit_regression_runner(teacher, student, batches, config)
    if not regression.get("passed") or regression.get("checked_tokens") != 2_048:
        raise StageError("exact untied unit-subset smoke regression did not pass")
    families: dict[str, object] = {}
    for family in ("unit", "ridge", "bounded", "free_bound"):
        dependencies.cuda_peak_reset()
        started = dependencies.clock()
        signature = _execution_signature(
            config,
            source_manifest,
            microbatch_size,
            stage="smoke",
            token_hash=token_hash,
            extra={"family": family},
        )
        result = dependencies.local_runner(
            teacher,
            student,
            batches,
            _family_conditions(config, family),
            artifact_root / "checkpoints" / "smoke" / family / signature,
            signature,
            source_hashes=source_hashes,
        )
        elapsed = dependencies.clock() - started
        if elapsed < 0.0:
            raise StageError("smoke clock moved backwards")
        solver_failures = _diagnostic_count(
            result.solver_diagnostics, "bounded_failure_count"
        )
        r8_failures = _diagnostic_count(
            result.reference_diagnostics, "bounded_failure_count"
        )
        r8_fallback_faces = _diagnostic_count(
            result.reference_diagnostics, "bounded_face_enumeration_count"
        )
        certification_failures = _diagnostic_count(
            result.solver_diagnostics, "bounded_uncertified_count"
        ) + _diagnostic_count(result.reference_diagnostics, "bounded_uncertified_count")
        failures = int(result.failed_batches) + solver_failures + r8_failures
        all_certified = certification_failures == 0
        if family != "bounded" and (not result.complete or failures):
            raise StageError(f"{family} smoke run did not complete without failures")
        benchmark = FamilyBenchmark.from_measurement(
            elapsed_seconds=elapsed,
            peak_cuda_bytes=int(dependencies.cuda_peak_probe()),
            processed_tokens=2_048,
            failures=failures,
            all_certified=all_certified,
        )
        families[family] = {
            "elapsed_seconds": benchmark.elapsed_seconds,
            "peak_cuda_bytes": benchmark.peak_cuda_bytes,
            "processed_tokens": benchmark.processed_tokens,
            "projected_full_seconds": benchmark.projected_full_seconds,
            "feasible": benchmark.feasible
            if family == "bounded"
            else result.complete and failures == 0,
            "failures": failures,
            "all_certified": all_certified,
            "r8_fallback_face_enumeration_count": r8_fallback_faces,
            "completed_batches": result.completed_batches,
            "result": _json_safe(result),
        }
    signature = _execution_signature(
        config, source_manifest, microbatch_size, stage="smoke", token_hash=token_hash
    )
    return _write_stage(
        artifact_root,
        "smoke",
        _config_signature(config),
        signature,
        {
            "execution_microbatch_size": microbatch_size,
            "valid_calibration_tokens": 2_048,
            "unit_regression": regression,
            "families": families,
            "source_hashes": source_hashes,
        },
    )


def _local_stage(
    config: Phase1A5Config,
    artifact_root: Path,
    microbatch_size: int,
    dependencies: Phase1A5Dependencies,
) -> dict[str, object]:
    _, source_manifest, source_hashes = _source_from_stage(artifact_root, config)
    try:
        smoke = _read_stage(artifact_root, "smoke", _config_signature(config))
    except StagePrerequisiteError as error:
        raise StagePrerequisiteError("local requires feasible smoke") from error
    families = smoke.get("families")
    if not isinstance(families, Mapping) or not isinstance(
        families.get("bounded"), Mapping
    ):
        raise StagePrerequisiteError("required feasible smoke record is invalid")
    for family in ("unit", "ridge", "free_bound"):
        if not isinstance(families.get(family), Mapping) or not families[family].get(
            "feasible"
        ):
            raise StagePrerequisiteError(
                "local requires feasible smoke for unit, ridge, and free_bound"
            )
    include_bounded = bool(families["bounded"].get("feasible"))
    assert (
        dependencies.calibration_loader is not None
        and dependencies.converted_pair_loader is not None
        and dependencies.local_runner is not None
    )
    input_ids, attention_mask = dependencies.calibration_loader(
        Path(str(source_manifest["source_root"])), source_manifest
    )
    input_ids, attention_mask = _first_valid_tokens(input_ids, attention_mask, 100_000)
    token_hash = _canonical_hash(
        {"tokens": input_ids.tolist(), "mask": attention_mask.tolist()}
    )
    signature = _execution_signature(
        config,
        source_manifest,
        microbatch_size,
        stage="local",
        token_hash=token_hash,
        extra={"bounded": include_bounded},
    )
    teacher, student = dependencies.converted_pair_loader(config)
    result = dependencies.local_runner(
        teacher,
        student,
        iter_evaluation_batches(
            input_ids, attention_mask, microbatch_size=microbatch_size
        ),
        _all_conditions(config, include_bounded=include_bounded),
        artifact_root / "checkpoints" / "local" / signature,
        signature,
        source_hashes=source_hashes,
    )
    if (
        not result.complete
        or result.failed_batches
        or dict(result.source_hashes) != source_hashes
    ):
        raise StageError(
            "local reconstruction did not complete against the verified source"
        )
    selection_metrics = [
        {"k": k, "condition": condition, "metric": _json_safe(metric)}
        for (k, condition), metric in sorted(result.metrics.items())
    ]
    return _write_stage(
        artifact_root,
        "local",
        _config_signature(config),
        signature,
        {
            "execution_microbatch_size": microbatch_size,
            "valid_calibration_tokens": 100_000,
            "bounded_included": include_bounded,
            "source_hashes": source_hashes,
            "selection_metrics": selection_metrics,
            "result": _json_safe(result),
        },
    )


def _select_stage(
    config: Phase1A5Config,
    artifact_root: Path,
    microbatch_size: int,
    dependencies: Phase1A5Dependencies,
) -> dict[str, object]:
    local = _read_stage(artifact_root, "local", _config_signature(config))
    metrics_payload = local.get("selection_metrics")
    source_hashes = local.get("source_hashes")
    persisted_result = local.get("result")
    if (
        not isinstance(metrics_payload, list)
        or not isinstance(source_hashes, Mapping)
        or not isinstance(persisted_result, Mapping)
        or persisted_result.get("complete") is not True
        or int(persisted_result.get("failed_batches", -1)) != 0
    ):
        raise StagePrerequisiteError("complete local stage lacks selection metrics")
    metrics: dict[tuple[int, str], ConditionMetrics] = {}
    try:
        for entry in metrics_payload:
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("metric"), Mapping
            ):
                raise ValueError
            metric = entry["metric"]
            metrics[(int(entry["k"]), str(entry["condition"]))] = ConditionMetrics(
                token_count=int(metric["token_count"]),
                total_squared_error=float(metric["total_squared_error"]),
                total_d0_squared_norm=float(metric["total_d0_squared_norm"]),
                coefficient_count=int(metric["coefficient_count"]),
                negative_count=int(metric["negative_count"]),
                above_two_count=int(metric["above_two_count"]),
                absolute_deviation_sum=float(metric["absolute_deviation_sum"]),
                unresolved_failures=int(metric["unresolved_failures"]),
                bounded_lower_saturation_count=int(
                    metric["bounded_lower_saturation_count"]
                ),
                bounded_upper_saturation_count=int(
                    metric["bounded_upper_saturation_count"]
                ),
                finite=bool(metric["finite"]),
            )
    except (KeyError, TypeError, ValueError) as error:
        raise StagePrerequisiteError(
            "complete local stage has malformed selection metrics"
        ) from error
    result = ReconstructionLocalResult(
        metrics=metrics,
        layer_metrics={},
        selection_counts={},
        co_selection={},
        solver_diagnostics={},
        reference_distances={},
        reference_diagnostics={},
        coefficient_quantiles={},
        completed_batches=1,
        failed_batches=0,
        complete=True,
        run_signature=str(local["input_signature"]),
        source_hashes={str(key): str(value) for key, value in source_hashes.items()},
    )
    assert dependencies.selection_runner is not None
    output_path = artifact_root / "selection-manifest.json"
    manifest = dependencies.selection_runner(result, config, output_path)
    if not output_path.is_file():
        raise StageError(
            "selection runner did not create an immutable selection manifest"
        )
    signature = _canonical_hash(
        {
            "local": local["input_signature"],
            "local_sha256": sha256_file(_stage_path(artifact_root, "local")),
            "microbatch_size": microbatch_size,
        }
    )
    return _write_stage(
        artifact_root,
        "select",
        _config_signature(config),
        signature,
        {
            "execution_microbatch_size": microbatch_size,
            "selection_manifest_sha256": sha256_file(output_path),
            "manifest": _json_safe(manifest),
            "source_hashes": source_hashes,
        },
    )


def _immutable_selection(artifact_root: Path, config: Phase1A5Config) -> None:
    state = _read_stage(artifact_root, "select", _config_signature(config))
    path = artifact_root / "selection-manifest.json"
    expected = state.get("selection_manifest_sha256")
    if (
        not path.is_file()
        or not isinstance(expected, str)
        or sha256_file(path) != expected
    ):
        raise StagePrerequisiteError("rollout requires an immutable selection manifest")


def run_phase1a5(
    config: Phase1A5Config,
    stage: str,
    resume: bool,
    artifact_root: Path,
    microbatch_size: int,
    *,
    dependencies: Phase1A5Dependencies | None = None,
) -> Mapping[str, object]:
    """Execute one Phase 1A.5 gate without deleting or sharing checkpoints."""

    del (
        resume
    )  # Resume belongs to the signature-isolated local accumulator checkpoints.
    if stage not in {"source", "smoke", "local", "select", "rollout", "report"}:
        raise ValueError(f"unsupported Phase 1A.5 stage: {stage}")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    artifact_root = Path(artifact_root)
    values = _dependencies(dependencies)
    if stage == "source":
        return _source_stage(config, artifact_root, microbatch_size, values)
    if stage == "smoke":
        return _smoke_stage(config, artifact_root, microbatch_size, values)
    if stage == "local":
        return _local_stage(config, artifact_root, microbatch_size, values)
    if stage == "select":
        return _select_stage(config, artifact_root, microbatch_size, values)
    if stage == "rollout":
        _immutable_selection(artifact_root, config)
        raise StageError("rollout is not implemented until Task 6")
    _read_stage(artifact_root, "source", _config_signature(config))
    _read_stage(
        artifact_root, "smoke", _config_signature(config), allow_scientific_stop=True
    )
    raise StageError("report is not implemented until Task 7")
