"""Gated Phase 1A.5 calibration orchestration.

Only source verification, smoke, local calibration, and frozen selection are
implemented here.  Rollout and report retain their gates but deliberately do
not execute until their separately reviewed implementations land.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor, nn

from ppmoe.adapters.qwen35_runtime import iter_text_mlps, load_qwen35_pair
from ppmoe.data.token_stream import iter_evaluation_batches
from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.experiments.phase1a5_provenance import (
    collect_phase1a_sources,
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
from ppmoe.oracle.reconstruction_rollout import (
    ReconstructionRolloutResult,
    run_reconstruction_rollout,
)
from ppmoe.provenance import atomic_write_json, sha256_file
from ppmoe.reporting.phase1a5 import ReportValidationError, render_phase1a5_report


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
    *,
    status: str = "complete",
) -> dict[str, object]:
    if status not in {"complete", "scientifically_stopped"}:
        raise ValueError(f"unsupported stage status: {status}")
    state = {
        "stage": stage,
        "status": status,
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


def _load_evaluation_tokens(
    source_root: Path, source_manifest: Mapping[str, object]
) -> Mapping[str, tuple[Tensor, Tensor]]:
    """Load only the two source-verified external evaluation corpora."""

    datasets = source_manifest.get("dataset_manifest_sha256")
    if not isinstance(datasets, Mapping):
        raise StageError("source manifest lacks evaluation dataset hashes")
    loaded: dict[str, tuple[Tensor, Tensor]] = {}
    for corpus, directory, source_key in (
        ("wikitext", "wikitext", "wikitext"),
        ("fineweb_heldout", "fineweb-heldout", "fineweb-heldout"),
    ):
        manifest_path = source_root / "tokens" / directory / "manifest.json"
        if sha256_file(manifest_path) != datasets.get(source_key):
            raise StageError(f"{corpus} manifest does not match verified source hash")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            input_path = manifest_path.parent / "input_ids.pt"
            mask_path = manifest_path.parent / "attention_mask.pt"
            if (
                not isinstance(manifest, Mapping)
                or not isinstance(manifest.get("input_ids_sha256"), str)
                or not isinstance(manifest.get("attention_mask_sha256"), str)
            ):
                raise StageError(f"{corpus} manifest is malformed")
            if (
                sha256_file(input_path) != manifest["input_ids_sha256"]
                or sha256_file(mask_path) != manifest["attention_mask_sha256"]
            ):
                raise StageError(f"{corpus} tensor hash does not match manifest")
            input_ids = torch.load(
                input_path, map_location="cpu", weights_only=True
            )
            attention_mask = torch.load(
                mask_path, map_location="cpu", weights_only=True
            )
        except StageError:
            raise
        except (OSError, json.JSONDecodeError, RuntimeError) as error:
            raise StageError(f"unable to load verified {corpus} tensors") from error
        if (
            input_ids.ndim != 2
            or attention_mask.shape != input_ids.shape
            or int(attention_mask.sum()) <= 0
        ):
            raise StageError(f"{corpus} tensors do not match their manifest")
        loaded[corpus] = input_ids, attention_mask
    return loaded


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
            # The hook captures inference tensors.  Their expert evaluation and
            # both oracle calls must remain in this mode: a trainable Linear
            # otherwise attempts to save the inference tensor for backward.
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
    source_collector: Callable[..., Mapping[str, object]] | None = None
    calibration_loader: Callable[..., tuple[Tensor, Tensor]] | None = None
    converted_pair_loader: Callable[[Phase1A5Config], tuple[Any, Any]] | None = None
    local_runner: Callable[..., ReconstructionLocalResult] | None = None
    unit_regression_runner: Callable[..., Mapping[str, object]] | None = None
    selection_runner: Callable[..., object] | None = None
    evaluation_loader: Callable[..., Mapping[str, tuple[Tensor, Tensor]]] | None = None
    rollout_runner: Callable[..., ReconstructionRolloutResult] | None = None
    cuda_peak_probe: Callable[[], int] | None = None
    cuda_peak_reset: Callable[[], None] | None = None
    clock: Callable[[], float] | None = None


def _dependencies(values: Phase1A5Dependencies | None) -> Phase1A5Dependencies:
    values = values or Phase1A5Dependencies()
    if values.source_verifier is None:
        values.source_verifier = verify_phase1a_sources
    if values.source_collector is None:
        values.source_collector = collect_phase1a_sources
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
    if values.evaluation_loader is None:
        values.evaluation_loader = _load_evaluation_tokens
    if values.rollout_runner is None:
        values.rollout_runner = run_reconstruction_rollout
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


def _matches_current_source(
    state: Mapping[str, object],
    source_stage: Mapping[str, object],
    source_hashes: Mapping[str, str],
) -> bool:
    return (
        state.get("source_stage_input_signature") == source_stage.get("input_signature")
        and state.get("source_hashes") == source_hashes
    )


def _fresh_smoke_provenance(
    config: Phase1A5Config,
    artifact_root: Path,
    microbatch_size: int,
    dependencies: Phase1A5Dependencies,
) -> tuple[Mapping[str, object], dict[str, str], str, str]:
    """Reverify the frozen source and smoke token identity without writing a stage."""

    assert (
        dependencies.source_collector is not None
        and dependencies.calibration_loader is not None
    )
    manifest = dependencies.source_collector(config.source_artifact_root.resolve(), config)
    source_hashes = _source_hashes(manifest)
    input_ids, attention_mask = dependencies.calibration_loader(
        Path(str(manifest["source_root"])), manifest
    )
    input_ids, attention_mask = _first_valid_tokens(input_ids, attention_mask, 2_048)
    token_hash = _canonical_hash(
        {"tokens": input_ids.tolist(), "mask": attention_mask.tolist()}
    )
    return (
        manifest,
        source_hashes,
        token_hash,
        _execution_signature(
            config, manifest, microbatch_size, stage="smoke", token_hash=token_hash
        ),
    )


def _read_current_smoke(
    artifact_root: Path,
    config: Phase1A5Config,
    source_stage: Mapping[str, object],
    source_hashes: Mapping[str, str],
    microbatch_size: int,
) -> dict[str, Any]:
    smoke = _read_stage(artifact_root, "smoke", _config_signature(config))
    if not _matches_current_source(smoke, source_stage, source_hashes):
        raise StagePrerequisiteError(
            "local requires smoke from the current verified source provenance"
        )
    if smoke.get("execution_microbatch_size") != microbatch_size:
        raise StagePrerequisiteError(
            "local requires smoke with the requested execution microbatch size"
        )
    return smoke


def _smoke_family_permits_progress(
    evidence: object, family: str, config: Phase1A5Config
) -> bool:
    """Validate persisted smoke measurements before they unlock later stages."""
    if not isinstance(evidence, Mapping):
        raise StagePrerequisiteError("required feasible smoke record is invalid")
    elapsed, peak, processed, completed, failures = (
        evidence.get("elapsed_seconds"), evidence.get("peak_cuda_bytes"),
        evidence.get("processed_tokens"), evidence.get("completed_batches"),
        evidence.get("failures"),
    )
    if (
        not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
        or not isfinite(float(elapsed)) or float(elapsed) < 0.0
        or not isinstance(peak, int) or isinstance(peak, bool) or peak < 0
        or processed != 2_048
        or not isinstance(completed, int) or completed <= 0
        or not isinstance(failures, int) or failures < 0
    ):
        raise StagePrerequisiteError("required feasible smoke record is invalid")
    if family == "free_bound" and not _svd_verification_is_complete(
        evidence.get("svd_verification"), config
    ):
        return False
    if family != "bounded":
        return failures == 0 and evidence.get("feasible") is True
    if failures or evidence.get("all_certified") is not True or evidence.get("feasible") is not True:
        return False
    kkt = evidence.get("kkt_by_k")
    if not isinstance(kkt, Mapping) or set(kkt) != {str(k) for k in config.k_values}:
        return False
    for record in kkt.values():
        if not isinstance(record, Mapping):
            return False
        try:
            if any(int(record[name]) < 0 for name in ("bounded_failure_count", "bounded_uncertified_count", "bounded_certified_count", "bounded_solver_candidate_count", "bounded_coefficient_bound_violation_count")):
                return False
            if record["bounded_failure_count"] or record["bounded_uncertified_count"] or record["bounded_coefficient_bound_violation_count"] or record["bounded_solver_candidate_count"] <= 0 or record["bounded_certified_count"] != record["bounded_solver_candidate_count"]:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _expected_svd_identities(config: Phase1A5Config) -> tuple[str, ...]:
    return (
        *(f"sparse:K{k}:free_bound" for k in config.k_values),
        "R8[free_bound]",
    )


def _svd_verification_is_complete(
    evidence: object, config: Phase1A5Config
) -> bool:
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        _expected_svd_identities(config)
    ):
        return False
    expected_layers = {str(layer): 16 for layer in range(24)}
    for identity in _expected_svd_identities(config):
        record = evidence.get(identity)
        if not isinstance(record, Mapping) or record.get("identity") != identity:
            return False
        maximum = record.get("max_relative_objective_error")
        if (
            record.get("sample_count") != 384
            or record.get("rank_equal") is not True
            or record.get("sample_count_by_layer") != expected_layers
            or not isinstance(maximum, (int, float))
            or isinstance(maximum, bool)
            or not isfinite(float(maximum))
            or float(maximum) > 1e-9
            or float(maximum) < 0.0
        ):
            return False
    return True


def _validated_svd_verification(
    result: ReconstructionLocalResult, config: Phase1A5Config
) -> dict[str, object]:
    evidence = _json_safe(result.svd_verification)
    if not _svd_verification_is_complete(evidence, config):
        raise StageError(
            "free-bound SVD verification requires exactly 16 tokens per layer "
            "for sparse K4/K6 and R8"
        )
    assert isinstance(evidence, dict)
    return evidence


def _diagnostic_count(diagnostics: Mapping[object, object], name: str) -> int:
    total = 0
    for value in diagnostics.values():
        if isinstance(value, Mapping):
            item = value.get(name, 0)
            if isinstance(item, (int, float)) and isfinite(float(item)):
                total += int(item)
    return total


def _bounded_kkt_by_k(
    result: ReconstructionLocalResult, config: Phase1A5Config
) -> dict[str, dict[str, int]]:
    """Require complete, zero-failure bounded certificates for every locked K."""
    records: dict[str, dict[str, int]] = {}
    for k in config.k_values:
        values = result.solver_diagnostics.get((k, "bounded"))
        if not isinstance(values, Mapping):
            raise StageError(f"bounded smoke lacks K={k} solver diagnostics")
        required = (
            "bounded_failure_count",
            "bounded_uncertified_count",
            "bounded_certified_count",
            "bounded_solver_candidate_count",
            "bounded_coefficient_bound_violation_count",
        )
        try:
            record = {name: int(values[name]) for name in required}
        except (KeyError, TypeError, ValueError) as error:
            raise StageError(f"bounded smoke has malformed K={k} KKT diagnostics") from error
        if (
            any(value < 0 for value in record.values())
            or record["bounded_failure_count"]
            or record["bounded_uncertified_count"]
            or record["bounded_coefficient_bound_violation_count"]
            or record["bounded_solver_candidate_count"] <= 0
            or record["bounded_certified_count"] != record["bounded_solver_candidate_count"]
        ):
            raise StageError(f"bounded smoke has invalid K={k} KKT certificate")
        records[str(k)] = record
    return records


def _smoke_failure_path(artifact_root: Path, signature: str) -> Path:
    return artifact_root / "failures" / "smoke" / f"{signature}.json"


def _preserve_smoke_failure(path: Path, payload: Mapping[str, object]) -> dict[str, Any]:
    """Atomically create immutable source-native failure evidence once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return _read_json_failure(path)
    encoded = (json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _read_json_failure(path)
        return json.loads(encoded)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json_failure(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageError("immutable smoke failure evidence is invalid") from error
    if not isinstance(value, dict):
        raise StageError("immutable smoke failure evidence is invalid")
    return value


def _smoke_stage(
    config: Phase1A5Config,
    artifact_root: Path,
    microbatch_size: int,
    dependencies: Phase1A5Dependencies,
    *,
    resume: bool,
) -> dict[str, object]:
    source_stage, source_manifest, source_hashes = _source_from_stage(
        artifact_root, config
    )
    existing_path = _stage_path(artifact_root, "smoke")
    if existing_path.exists():
        existing = _read_stage(artifact_root, "smoke", _config_signature(config))
        if resume:
            fresh_manifest, fresh_hashes, _token_hash, fresh_signature = (
                _fresh_smoke_provenance(
                    config, artifact_root, microbatch_size, dependencies
                )
            )
            if (
                source_stage.get("input_signature")
                != _source_signature(config, fresh_manifest, microbatch_size)
                or source_hashes != fresh_hashes
                or existing.get("input_signature") != fresh_signature
            ):
                raise StageError(
                    "smoke resume provenance does not match freshly verified source, "
                    "code, configuration, token identity, and microbatch"
                )
        compatible = (
            _matches_current_source(existing, source_stage, source_hashes)
            and existing.get("execution_microbatch_size") == microbatch_size
        )
        if compatible:
            if resume:
                families = existing.get("families")
                if not isinstance(families, Mapping) or not _svd_verification_is_complete(
                    families.get("free_bound", {}).get("svd_verification")
                    if isinstance(families.get("free_bound"), Mapping)
                    else None,
                    config,
                ):
                    raise StageError(
                        "compatible smoke evidence lacks the required complete SVD evidence; cannot resume"
                    )
                return existing
            raise StageError(
                "compatible smoke evidence is immutable; rerun with --resume"
            )
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
    signature = _execution_signature(
        config, source_manifest, microbatch_size, stage="smoke", token_hash=token_hash
    )
    failure_path = _smoke_failure_path(artifact_root, signature)
    if failure_path.exists():
        raise StageError("immutable failure evidence exists for this smoke signature")
    started = dependencies.clock()
    try:
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
            family_started = dependencies.clock()
            family_signature = _execution_signature(
                config, source_manifest, microbatch_size, stage="smoke", token_hash=token_hash,
                extra={"family": family},
            )
            result = dependencies.local_runner(
            teacher,
            student,
            batches,
            _family_conditions(config, family),
            artifact_root / "checkpoints" / "smoke" / family / family_signature,
            family_signature,
            source_hashes=source_hashes,
            )
            elapsed = dependencies.clock() - family_started
            peak_cuda_bytes = int(dependencies.cuda_peak_probe())
            if not isfinite(elapsed) or elapsed < 0.0 or peak_cuda_bytes < 0:
                raise StageError("smoke measurement is invalid")
            if not result.complete or result.completed_batches <= 0 or result.failed_batches != 0:
                raise StageError(f"{family} smoke run lacks complete batches")
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
            if family == "bounded":
                try:
                    kkt_by_k = _bounded_kkt_by_k(result, config)
                except StageError:
                    # Preserve the bounded engineering failure in the smoke
                    # record; it makes bounded infeasible rather than allowing
                    # local selection to treat an incomplete certificate as pass.
                    kkt_by_k = {}
                    all_certified = False
            else:
                kkt_by_k = {}
            svd_verification = (
                _validated_svd_verification(result, config)
                if family == "free_bound"
                else None
            )
            if family != "bounded" and failures:
                raise StageError(f"{family} smoke run did not complete without failures")
            benchmark = FamilyBenchmark.from_measurement(
            elapsed_seconds=elapsed,
            peak_cuda_bytes=peak_cuda_bytes,
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
            "kkt_by_k": kkt_by_k,
            **(
                {"svd_verification": svd_verification}
                if svd_verification is not None
                else {}
            ),
            **(
                {"bounded_stop_reason": "engineering_infeasible_under_v0_5"}
                if family == "bounded" and not benchmark.feasible
                else {}
            ),
            "result": _json_safe(result),
            }
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
            "source_stage_input_signature": source_stage["input_signature"],
        },
        )
    except Exception as error:
        elapsed = dependencies.clock() - started
        try:
            peak_cuda_bytes: int | None = int(dependencies.cuda_peak_probe())
        except Exception:
            peak_cuda_bytes = None
        _preserve_smoke_failure(failure_path, {
            "stage": "smoke", "status": "failed", "input_signature": signature,
            "source_stage_input_signature": source_stage["input_signature"],
            "source_hashes": source_hashes, "execution_microbatch_size": microbatch_size,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "elapsed_seconds": elapsed if isfinite(elapsed) and elapsed >= 0.0 else None,
            "peak_cuda_bytes": peak_cuda_bytes,
            "completed_batches": None, "failed_batches": None,
        })
        raise


def _local_stage(
    config: Phase1A5Config,
    artifact_root: Path,
    microbatch_size: int,
    dependencies: Phase1A5Dependencies,
) -> dict[str, object]:
    source_stage, source_manifest, source_hashes = _source_from_stage(
        artifact_root, config
    )
    try:
        smoke = _read_current_smoke(
            artifact_root,
            config,
            source_stage,
            source_hashes,
            microbatch_size,
        )
    except StagePrerequisiteError as error:
        if "required stage smoke is absent" in str(error):
            raise StagePrerequisiteError("local requires feasible smoke") from error
        raise
    families = smoke.get("families")
    if not isinstance(families, Mapping):
        raise StagePrerequisiteError("required feasible smoke record is invalid")
    for family in ("unit", "ridge", "free_bound"):
        if not _smoke_family_permits_progress(families.get(family), family, config):
            raise StagePrerequisiteError(
                "local requires feasible smoke for unit, ridge, and free_bound"
            )
    include_bounded = _smoke_family_permits_progress(
        families.get("bounded"), "bounded", config
    )
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
    _validated_svd_verification(result, config)
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
            "source_stage_input_signature": source_stage["input_signature"],
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
    source_stage, _, source_hashes = _source_from_stage(artifact_root, config)
    if not _matches_current_source(local, source_stage, source_hashes):
        raise StagePrerequisiteError(
            "select requires local results from the current verified source provenance"
        )
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
                **{
                    field.name: metric[field.name]
                    for field in fields(ConditionMetrics)
                }
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
    if isinstance(manifest, Mapping):
        practical = manifest.get("practical_conditions", {})
        free_bound = manifest.get("free_bound_conditions", {})
    else:
        practical = getattr(manifest, "practical_conditions", {})
        free_bound = getattr(manifest, "free_bound_conditions", {})
    stopped = not practical and not free_bound
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
            "source_stage_input_signature": source_stage["input_signature"],
            **(
                {"scientific_stop_reason": "selection_no_eligible_conditions"}
                if stopped
                else {}
            ),
        },
        status="scientifically_stopped" if stopped else "complete",
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


def _condition_from_frozen_id(
    identifier: object, config: Phase1A5Config
) -> ReconstructionCondition:
    """Decode a manifest ID only when it is an exact registered condition ID."""

    if not isinstance(identifier, str):
        raise StagePrerequisiteError("selection manifest contains a non-string condition ID")
    if identifier == "unit":
        return ReconstructionCondition("unit")
    if identifier == "bounded":
        return ReconstructionCondition("bounded")
    if identifier == "free_bound":
        return ReconstructionCondition("free_bound")
    for rho in config.rho_values:
        candidate = ReconstructionCondition("ridge", rho)
        if identifier == f"ridge:{rho:g}":
            return candidate
    raise StagePrerequisiteError("selection manifest contains an unregistered condition")


def _frozen_condition_id(condition: ReconstructionCondition) -> str:
    return condition.family if condition.rho is None else f"ridge:{condition.rho:g}"


def _frozen_rollout_conditions(
    artifact_root: Path, config: Phase1A5Config
) -> tuple[str, Mapping[int, tuple[ReconstructionCondition, ...]]]:
    """Return exactly unit plus the immutable practical/free selections by K."""

    _immutable_selection(artifact_root, config)
    path = artifact_root / "selection-manifest.json"
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise StagePrerequisiteError("rollout selection manifest is invalid") from error
    if not isinstance(payload, Mapping):
        raise StagePrerequisiteError("rollout selection manifest is invalid")
    practical, free_bound = payload.get("practical_conditions", {}), payload.get("free_bound_conditions", {})
    if not isinstance(practical, Mapping) or not isinstance(free_bound, Mapping):
        raise StagePrerequisiteError("rollout selection manifest has invalid condition maps")
    conditions: dict[int, tuple[ReconstructionCondition, ...]] = {}
    allowed_k = {str(k) for k in config.k_values}
    if any(str(key) not in allowed_k for key in practical) or any(str(key) not in allowed_k for key in free_bound):
        raise StagePrerequisiteError("selection manifest names an unregistered K")
    for k in config.k_values:
        selected: list[ReconstructionCondition] = []
        practical_id, free_id = practical.get(str(k)), free_bound.get(str(k))
        if practical_id is not None:
            selected.append(_condition_from_frozen_id(practical_id, config))
        if free_id is not None:
            condition = _condition_from_frozen_id(free_id, config)
            if condition.family != "free_bound":
                raise StagePrerequisiteError("free-bound selection must name free_bound")
            selected.append(condition)
        if selected:
            values = (ReconstructionCondition("unit"), *selected)
            identifiers = [_frozen_condition_id(value) for value in values]
            if len(set(identifiers)) != len(identifiers):
                raise StagePrerequisiteError("selection manifest repeats a rollout condition")
            conditions[k] = values
    if not conditions:
        raise StagePrerequisiteError("rollout has no eligible frozen condition")
    return sha256_file(path), conditions


def _rollout_stage(
    config: Phase1A5Config,
    artifact_root: Path,
    microbatch_size: int,
    dependencies: Phase1A5Dependencies,
) -> dict[str, object]:
    source_stage, source_manifest, source_hashes = _source_from_stage(artifact_root, config)
    selection = _read_stage(artifact_root, "select", _config_signature(config))
    if not _matches_current_source(selection, source_stage, source_hashes):
        raise StagePrerequisiteError("rollout selection provenance does not match the current verified source")
    manifest_sha256, conditions = _frozen_rollout_conditions(artifact_root, config)
    assert dependencies.evaluation_loader is not None and dependencies.converted_pair_loader is not None and dependencies.rollout_runner is not None
    streams = dependencies.evaluation_loader(Path(str(source_manifest["source_root"])), source_manifest)
    if set(streams) != {"wikitext", "fineweb_heldout"}:
        raise StageError("rollout requires exactly WikiText and FineWeb-heldout streams")
    teacher, student = dependencies.converted_pair_loader(config)
    results: dict[str, object] = {}
    for corpus in ("wikitext", "fineweb_heldout"):
        input_ids, attention_mask = streams[corpus]
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise StageError(f"rollout {corpus} tensors are malformed")
        token_hash = _canonical_hash({"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()})
        for k, selected in conditions.items():
            for condition in selected:
                identifier = _frozen_condition_id(condition)
                signature = _execution_signature(
                    config, source_manifest, microbatch_size, stage="rollout", token_hash=token_hash,
                    extra={"corpus": corpus, "k": k, "condition": identifier, "selection_manifest_sha256": manifest_sha256},
                )
                result = dependencies.rollout_runner(
                    teacher, student,
                    iter_evaluation_batches(input_ids, attention_mask, microbatch_size=microbatch_size),
                    condition, k,
                    artifact_root / "checkpoints" / "rollout" / corpus / f"k{k}" / identifier.replace(":", "_") / signature,
                    signature, source_hashes=source_hashes,
                )
                if (
                    not result.complete
                    or result.failed_batches
                    or result.scored_tokens <= 0
                    or dict(result.source_hashes) != source_hashes
                ):
                    raise StageError(f"rollout is incomplete for {corpus} K={k} {identifier}")
                results[f"{corpus}:k{k}:{identifier}"] = _json_safe(result)
    signature = _canonical_hash({
        "source": source_stage["input_signature"], "selection_manifest_sha256": manifest_sha256,
        "microbatch_size": microbatch_size,
        "conditions": {
            str(k): [_frozen_condition_id(value) for value in values]
            for k, values in conditions.items()
        },
    })
    return _write_stage(artifact_root, "rollout", _config_signature(config), signature, {
        "execution_microbatch_size": microbatch_size,
        "source_hashes": source_hashes,
        "source_stage_input_signature": source_stage["input_signature"],
        "selection_manifest_sha256": manifest_sha256,
        "results": results,
    })


def _read_report_stage(
    artifact_root: Path,
    stage: str,
    config: Phase1A5Config,
    source_stage: Mapping[str, object],
    source_hashes: Mapping[str, str],
    *,
    allow_scientific_stop: bool = False,
) -> dict[str, Any]:
    state = _read_stage(
        artifact_root,
        stage,
        _config_signature(config),
        allow_scientific_stop=allow_scientific_stop,
    )
    if not _matches_current_source(state, source_stage, source_hashes):
        raise StagePrerequisiteError(
            f"required stage {stage} provenance does not match the current verified source"
        )
    return state


def _selection_is_scientifically_stopped(
    artifact_root: Path, selection: Mapping[str, object]
) -> bool:
    if selection.get("status") != "scientifically_stopped":
        return False
    if selection.get("scientific_stop_reason") != "selection_no_eligible_conditions":
        raise StagePrerequisiteError(
            "selection scientific stop has no registered reason"
        )
    path = artifact_root / "selection-manifest.json"
    expected = selection.get("selection_manifest_sha256")
    if (
        not path.is_file()
        or not isinstance(expected, str)
        or sha256_file(path) != expected
    ):
        raise StagePrerequisiteError(
            "selection scientific stop lacks an immutable manifest"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StagePrerequisiteError(
            "selection scientific stop has invalid manifest"
        ) from error
    if payload.get("practical_conditions") or payload.get("free_bound_conditions"):
        raise StagePrerequisiteError(
            "selection scientific stop has eligible conditions"
        )
    return True


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

    if stage not in {"source", "smoke", "local", "select", "rollout", "report"}:
        raise ValueError(f"unsupported Phase 1A.5 stage: {stage}")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    artifact_root = Path(artifact_root)
    values = _dependencies(dependencies)
    if stage == "source":
        return _source_stage(config, artifact_root, microbatch_size, values)
    if stage == "smoke":
        return _smoke_stage(
            config, artifact_root, microbatch_size, values, resume=resume
        )
    if stage == "local":
        return _local_stage(config, artifact_root, microbatch_size, values)
    if stage == "select":
        return _select_stage(config, artifact_root, microbatch_size, values)
    if stage == "rollout":
        return _rollout_stage(config, artifact_root, microbatch_size, values)
    source_stage, _, source_hashes = _source_from_stage(artifact_root, config)
    _read_report_stage(artifact_root, "smoke", config, source_stage, source_hashes)
    _read_report_stage(artifact_root, "local", config, source_stage, source_hashes)
    selection = _read_report_stage(
        artifact_root,
        "select",
        config,
        source_stage,
        source_hashes,
        allow_scientific_stop=True,
    )
    stopped = _selection_is_scientifically_stopped(artifact_root, selection)
    if not stopped:
        _read_report_stage(artifact_root, "rollout", config, source_stage, source_hashes)
    try:
        return _json_safe(asdict(render_phase1a5_report(artifact_root, config)))
    except ReportValidationError as error:
        raise StageError(f"report integrity validation failed: {error}") from error
