"""Validated, hash-addressed reporting for the Phase 1A.5 oracle."""

from __future__ import annotations

import ast
import csv
from dataclasses import asdict, dataclass
import json
from math import isclose, isfinite, sqrt
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.provenance import atomic_write_json, sha256_file


class ReportValidationError(RuntimeError):
    """Persisted evidence is incomplete, inconsistent, or unsafe to report."""


@dataclass(frozen=True, slots=True)
class Phase1A5ReportManifest:
    output_hashes: Mapping[str, str]
    source_stage_hashes: Mapping[str, str]
    decision: str
    local_conditions: Mapping[str, tuple[str, ...]]
    rollout_conditions: Mapping[str, tuple[str, ...]]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportValidationError(f"{label} is missing or invalid") from error
    if not isinstance(payload, dict):
        raise ReportValidationError(f"{label} is invalid")
    return payload


def _stage(root: Path, name: str, *, allow_scientific_stop: bool = False) -> dict[str, Any]:
    stage = _read_json(root / "stages" / f"{name}.json", f"{name} stage")
    allowed = {"complete", "scientifically_stopped"} if allow_scientific_stop else {"complete"}
    if stage.get("stage") != name or stage.get("status") not in allowed:
        raise ReportValidationError(f"{name} stage is incomplete")
    return stage


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
        raise ReportValidationError(f"{label} is non-finite")
    result = float(value)
    if positive and result <= 0:
        raise ReportValidationError(f"{label} must be positive")
    return result


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        raise ReportValidationError(f"{label} is invalid")
    return value


def _key(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, str):
        raise ReportValidationError(f"{label} has invalid key")
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as error:
        raise ReportValidationError(f"{label} has invalid key") from error
    if not isinstance(parsed, tuple):
        raise ReportValidationError(f"{label} has invalid key")
    return parsed


def _hashes(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ReportValidationError(f"{label} is missing")
    result = {str(key): str(item) for key, item in value.items()}
    if any(len(item) != 64 for item in result.values()):
        raise ReportValidationError(f"{label} is malformed")
    return result


def _validate_existing_manifest(root: Path) -> None:
    path = root / "report-manifest.json"
    if not path.exists():
        return
    manifest = _read_json(path, "existing report manifest")
    hashes = manifest.get("output_hashes")
    if not isinstance(hashes, Mapping):
        raise ReportValidationError("existing report manifest is malformed")
    for relative, expected in hashes.items():
        output = root / str(relative)
        if not output.is_file() or not isinstance(expected, str) or sha256_file(output) != expected:
            raise ReportValidationError("output hash mismatch")


def _optional_distribution(
    metric: Mapping[str, object],
    names: tuple[str, ...],
    label: str,
    *,
    defined_count: int,
) -> dict[str, float | None]:
    if defined_count == 0:
        if any(metric.get(name) is not None for name in names):
            raise ReportValidationError(f"{label} local metric schema is invalid")
        return {name: None for name in names}
    return {name: _finite(metric.get(name), f"{label} local metric schema") for name in names}


def _metric(metric: object, label: str) -> dict[str, object]:
    if not isinstance(metric, Mapping):
        raise ReportValidationError(f"{label} is missing")
    values = {
        "token_count": _integer(metric.get("token_count"), f"{label} token count", positive=True),
        "total_squared_error": _finite(metric.get("total_squared_error"), f"{label} error"),
        "total_d0_squared_norm": _finite(metric.get("total_d0_squared_norm"), f"{label} norm", positive=True),
        "coefficient_count": _integer(metric.get("coefficient_count"), f"{label} coefficient count", positive=True),
        "negative_count": _integer(metric.get("negative_count"), f"{label} negative count"),
        "above_two_count": _integer(metric.get("above_two_count"), f"{label} above-two count"),
        "absolute_deviation_sum": _finite(metric.get("absolute_deviation_sum"), f"{label} coefficient deviation"),
        "unresolved_failures": _integer(metric.get("unresolved_failures"), f"{label} unresolved failures"),
        "finite": metric.get("finite") is True,
        "bounded_lower_saturation_count": _integer(metric.get("bounded_lower_saturation_count", 0), f"{label} lower saturation"),
        "bounded_upper_saturation_count": _integer(metric.get("bounded_upper_saturation_count", 0), f"{label} upper saturation"),
        "coefficient_sum": _finite(metric.get("coefficient_sum"), f"{label} local metric schema"),
        "coefficient_squared_sum": _finite(metric.get("coefficient_squared_sum"), f"{label} local metric schema"),
        "above_four_count": _integer(metric.get("above_four_count"), f"{label} local metric schema"),
        "squared_deviation_sum": _finite(metric.get("squared_deviation_sum"), f"{label} local metric schema"),
        "relative_error_sum": _finite(metric.get("relative_error_sum"), f"{label} local metric schema"),
        "relative_error_defined_count": _integer(metric.get("relative_error_defined_count"), f"{label} local metric schema"),
        "relative_error_undefined_count": _integer(metric.get("relative_error_undefined_count"), f"{label} local metric schema"),
        "cosine_sum": _finite(metric.get("cosine_sum"), f"{label} local metric schema"),
        "cosine_defined_count": _integer(metric.get("cosine_defined_count"), f"{label} local metric schema"),
        "cosine_undefined_count": _integer(metric.get("cosine_undefined_count"), f"{label} local metric schema"),
        "gram_condition_defined_count": _integer(metric.get("gram_condition_defined_count"), f"{label} local metric schema"),
        "gram_condition_undefined_count": _integer(metric.get("gram_condition_undefined_count"), f"{label} local metric schema"),
        "coefficient_mean": _finite(metric.get("coefficient_mean"), f"{label} local metric schema"),
        "coefficient_std": _finite(metric.get("coefficient_std"), f"{label} local metric schema"),
        "coefficient_p01": _finite(metric.get("coefficient_p01"), f"{label} local metric schema"),
        "coefficient_p05": _finite(metric.get("coefficient_p05"), f"{label} local metric schema"),
        "coefficient_median": _finite(metric.get("coefficient_median"), f"{label} local metric schema"),
        "coefficient_p95": _finite(metric.get("coefficient_p95"), f"{label} local metric schema"),
        "coefficient_p99": _finite(metric.get("coefficient_p99"), f"{label} local metric schema"),
        "coefficient_p99_abs": _finite(metric.get("coefficient_p99_abs"), f"{label} local metric schema"),
        "coefficient_max_abs": _finite(metric.get("coefficient_max_abs"), f"{label} local metric schema"),
        "negative_rate": _finite(metric.get("negative_rate"), f"{label} local metric schema"),
        "above_two_rate": _finite(metric.get("above_two_rate"), f"{label} local metric schema"),
        "above_four_rate": _finite(metric.get("above_four_rate"), f"{label} local metric schema"),
        "bounded_lower_saturation_rate": _finite(metric.get("bounded_lower_saturation_rate"), f"{label} local metric schema"),
        "bounded_upper_saturation_rate": _finite(metric.get("bounded_upper_saturation_rate"), f"{label} local metric schema"),
        "mean_absolute_deviation_from_one": _finite(metric.get("mean_absolute_deviation_from_one"), f"{label} local metric schema"),
        "l2_deviation_from_one": _finite(metric.get("l2_deviation_from_one"), f"{label} local metric schema"),
        "l2_deviation_denominator": _integer(metric.get("l2_deviation_denominator"), f"{label} local metric schema", positive=True),
    }
    values.update(
        _optional_distribution(
            metric,
            ("relative_error_mean", "relative_error_median", "relative_error_p90", "relative_error_p95", "relative_error_p99"),
            label,
            defined_count=int(values["relative_error_defined_count"]),
        )
    )
    values.update(
        _optional_distribution(
            metric,
            ("cosine_mean", "cosine_median", "cosine_p90", "cosine_p95", "cosine_p99"),
            label,
            defined_count=int(values["cosine_defined_count"]),
        )
    )
    values.update(
        _optional_distribution(
            metric,
            ("gram_condition_median", "gram_condition_p90", "gram_condition_p99", "gram_condition_max"),
            label,
            defined_count=int(values["gram_condition_defined_count"]),
        )
    )
    rank_counts = metric.get("effective_rank_counts")
    if not isinstance(rank_counts, Mapping) or not rank_counts:
        raise ReportValidationError(f"{label} local metric schema is invalid")
    parsed_ranks: dict[str, int] = {}
    try:
        for rank, count in rank_counts.items():
            parsed_rank = int(rank)
            if str(parsed_rank) != str(rank) or not 0 <= parsed_rank <= 8:
                raise ValueError
            parsed_count = _integer(count, f"{label} local metric schema")
            if parsed_count < 0:
                raise ValueError
            parsed_ranks[str(parsed_rank)] = parsed_count
    except (TypeError, ValueError) as error:
        raise ReportValidationError(f"{label} local metric schema is invalid") from error
    values["effective_rank_counts"] = parsed_ranks
    if not values["finite"]:
        raise ReportValidationError(f"{label} local metric schema finite flag is invalid")
    physical = ("total_squared_error", "total_d0_squared_norm", "coefficient_squared_sum", "absolute_deviation_sum", "squared_deviation_sum", "relative_error_sum", "negative_count", "above_two_count", "above_four_count", "unresolved_failures", "bounded_lower_saturation_count", "bounded_upper_saturation_count")
    if any(float(values[name]) < 0 for name in physical):
        raise ReportValidationError(f"{label} has negative physical metric")
    if any(int(values[name]) > int(values["coefficient_count"]) for name in ("negative_count", "above_two_count", "above_four_count", "bounded_lower_saturation_count", "bounded_upper_saturation_count")):
        raise ReportValidationError(f"{label} coefficient counts are invalid")
    coefficient_count = int(values["coefficient_count"])
    token_count = int(values["token_count"])
    if (
        int(values["l2_deviation_denominator"]) != coefficient_count
        or int(values["relative_error_defined_count"])
        + int(values["relative_error_undefined_count"])
        != token_count
        or int(values["cosine_defined_count"])
        + int(values["cosine_undefined_count"])
        != token_count
        or int(values["gram_condition_defined_count"])
        + int(values["gram_condition_undefined_count"])
        != token_count
        or sum(parsed_ranks.values()) != token_count
    ):
        raise ReportValidationError(f"{label} local metric schema counts are invalid")
    expected = {
        "coefficient_mean": float(values["coefficient_sum"]) / coefficient_count,
        "coefficient_std": sqrt(
            max(
                float(values["coefficient_squared_sum"]) / coefficient_count
                - (float(values["coefficient_sum"]) / coefficient_count) ** 2,
                0.0,
            )
        ),
        "negative_rate": int(values["negative_count"]) / coefficient_count,
        "above_two_rate": int(values["above_two_count"]) / coefficient_count,
        "above_four_rate": int(values["above_four_count"]) / coefficient_count,
        "bounded_lower_saturation_rate": int(values["bounded_lower_saturation_count"]) / coefficient_count,
        "bounded_upper_saturation_rate": int(values["bounded_upper_saturation_count"]) / coefficient_count,
        "mean_absolute_deviation_from_one": float(values["absolute_deviation_sum"]) / coefficient_count,
        "l2_deviation_from_one": sqrt(float(values["squared_deviation_sum"]) / coefficient_count),
    }
    if int(values["relative_error_defined_count"]):
        expected["relative_error_mean"] = float(values["relative_error_sum"]) / int(values["relative_error_defined_count"])
    if int(values["cosine_defined_count"]):
        expected["cosine_mean"] = float(values["cosine_sum"]) / int(
            values["cosine_defined_count"]
        )
    if any(
        not isclose(float(values[name]), target, rel_tol=1e-12, abs_tol=1e-15)
        for name, target in expected.items()
    ):
        raise ReportValidationError(f"{label} local metric schema summary mismatch")
    signed = [float(values[name]) for name in ("coefficient_p01", "coefficient_p05", "coefficient_median", "coefficient_p95", "coefficient_p99")]
    relative = [float(values[name]) for name in ("relative_error_median", "relative_error_p90", "relative_error_p95", "relative_error_p99")] if int(values["relative_error_defined_count"]) else []
    if (
        signed != sorted(signed)
        or relative != sorted(relative)
        or (relative and min(relative) < 0.0)
        or float(values["coefficient_p99_abs"]) < 0.0
        or float(values["coefficient_max_abs"]) < float(values["coefficient_p99_abs"])
    ):
        raise ReportValidationError(f"{label} local metric schema quantiles are invalid")
    if int(values["cosine_defined_count"]):
        cosines = [float(values[name]) for name in ("cosine_median", "cosine_p90", "cosine_p95", "cosine_p99")]
        if cosines != sorted(cosines) or any(not -1.000000000001 <= value <= 1.000000000001 for value in cosines):
            raise ReportValidationError(f"{label} local metric schema cosine is invalid")
    if int(values["gram_condition_defined_count"]):
        conditions = [float(values[name]) for name in ("gram_condition_median", "gram_condition_p90", "gram_condition_p99", "gram_condition_max")]
        if conditions != sorted(conditions) or min(conditions) < 0.0:
            raise ReportValidationError(f"{label} local metric schema conditioning is invalid")
    return values


def _validate_svd_verification(value: object, config: Phase1A5Config) -> None:
    expected = {
        *(f"sparse:K{k}:free_bound" for k in config.k_values),
        "R8[free_bound]",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ReportValidationError("SVD verification evidence is incomplete")
    expected_layers = {str(layer): 16 for layer in range(24)}
    for identity in expected:
        record = value.get(identity)
        maximum = record.get("max_relative_objective_error") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or record.get("identity") != identity
            or record.get("sample_count") != 384
            or record.get("sample_count_by_layer") != expected_layers
            or record.get("rank_equal") is not True
            or not isinstance(maximum, (int, float))
            or isinstance(maximum, bool)
            or not isfinite(float(maximum))
            or not 0.0 <= float(maximum) <= 1e-9
        ):
            raise ReportValidationError("SVD verification evidence is invalid")


_SOLVER_ROW_FIELDS = (
    "scope",
    "layer",
    "k",
    "condition",
    "token_count",
    "coefficient_count",
    "numerical_failure_count",
    "layer_coverage_count",
    "solver_source_counts",
    "effective_rank_counts",
    "gram_condition_defined_count",
    "gram_condition_undefined_count",
    "gram_condition_median",
    "gram_condition_p90",
    "gram_condition_p99",
    "gram_condition_max",
    "cholesky_fallback_count",
    "degenerate_ridge_count",
    "max_relative_system_residual",
    "tau_g",
    "pseudoinverse_effective_rank_counts",
    "bounded_lower_saturation_count",
    "bounded_upper_saturation_count",
    "bounded_failure_count",
    "bounded_uncertified_count",
    "bounded_certified_count",
    "bounded_solver_candidate_count",
    "bounded_coefficient_bound_violation_count",
    "bounded_max_primal_residual",
    "bounded_max_stationarity_residual",
    "bounded_max_complementarity_residual",
    "bounded_max_iterations",
    "bounded_face_enumeration_count",
    "bounded_max_face_enumeration_count",
    "svd_identity",
    "svd_sample_count",
    "svd_sample_count_by_layer",
    "svd_layer_sample_count",
    "svd_rank_equal",
    "svd_max_relative_objective_error",
)


def _histogram(
    value: object, label: str, *, token_count: int
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{label} is invalid")
    parsed: dict[str, int] = {}
    try:
        for raw_rank, raw_count in value.items():
            rank = int(raw_rank)
            if str(rank) != str(raw_rank) or not 0 <= rank <= 8:
                raise ValueError
            count = _integer(raw_count, label)
            if count < 0:
                raise ValueError
            parsed[str(rank)] = count
    except (TypeError, ValueError) as error:
        raise ReportValidationError(f"{label} is invalid") from error
    if not parsed or sum(parsed.values()) != token_count:
        raise ReportValidationError(f"{label} does not reconcile to coverage")
    return parsed


def _flat_mapping(value: Mapping[str, int] | None) -> str | None:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        if value is not None
        else None
    )


def _mean_effective_rank(
    value: object, label: str, *, token_count: int
) -> tuple[dict[str, int], float]:
    counts = _histogram(value, label, token_count=token_count)
    return (
        counts,
        sum(int(rank) * count for rank, count in counts.items()) / token_count,
    )


def _validate_solver_record(
    value: object,
    label: str,
    *,
    identifier: str,
    token_count: int,
    expected_tau_g: float,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{label} are invalid")
    source_counts = value.get("solver_source_counts")
    if not isinstance(source_counts, Mapping) or not source_counts:
        raise ReportValidationError(f"{label} lack solver counts")
    parsed_sources: dict[str, int] = {}
    for source, count in source_counts.items():
        parsed = _integer(count, f"{label} solver count")
        if parsed < 0:
            raise ReportValidationError(f"{label} solver counts are invalid")
        parsed_sources[str(source)] = parsed
    if sum(parsed_sources.values()) != token_count:
        raise ReportValidationError(f"{label} solver counts are incomplete")
    bounded_fields = (
        "bounded_lower_saturation_count",
        "bounded_upper_saturation_count",
        "bounded_failure_count",
        "bounded_uncertified_count",
        "bounded_certified_count",
        "bounded_solver_candidate_count",
        "bounded_coefficient_bound_violation_count",
        "bounded_max_primal_residual",
        "bounded_max_stationarity_residual",
        "bounded_max_complementarity_residual",
        "bounded_max_iterations",
        "bounded_face_enumeration_count",
        "bounded_max_face_enumeration_count",
    )
    ridge_fields = (
        "cholesky_fallback_count",
        "degenerate_ridge_count",
        "max_relative_system_residual",
    )
    free_fields = ("tau_g", "pseudoinverse_effective_rank_counts")
    if identifier == "bounded":
        if any(name in value for name in (*ridge_fields, *free_fields)):
            raise ReportValidationError(f"{label} contain inapplicable bounded evidence")
        for name in bounded_fields:
            item = value.get(name)
            if (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not isfinite(float(item))
                or float(item) < 0.0
            ):
                raise ReportValidationError(f"{label} lack {name}")
        counts = {
            name: _integer(value[name], f"{label} {name}")
            for name in (
                "bounded_failure_count",
                "bounded_uncertified_count",
                "bounded_certified_count",
                "bounded_solver_candidate_count",
                "bounded_coefficient_bound_violation_count",
                "bounded_lower_saturation_count",
                "bounded_upper_saturation_count",
                "bounded_face_enumeration_count",
                "bounded_max_face_enumeration_count",
                "bounded_max_iterations",
            )
        }
        if (
            counts["bounded_failure_count"] != counts["bounded_uncertified_count"]
            or counts["bounded_certified_count"]
            + counts["bounded_uncertified_count"]
            != counts["bounded_solver_candidate_count"]
        ):
            raise ReportValidationError(
                f"{label} bounded KKT evidence has inconsistent bounded certification"
            )
    else:
        if any(name in value for name in bounded_fields):
            raise ReportValidationError(f"{label} contain inapplicable bounded evidence")
        for name in ridge_fields:
            item = value.get(name)
            if (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not isfinite(float(item))
                or float(item) < 0.0
            ):
                raise ReportValidationError(f"{label} lack {name}")
        if identifier == "free_bound":
            tau = value.get("tau_g")
            if (
                not isinstance(tau, (int, float))
                or isinstance(tau, bool)
                or not isfinite(float(tau))
                or float(tau) != expected_tau_g
            ):
                raise ReportValidationError(f"{label} lack registered tau_g")
            _histogram(
                value.get("pseudoinverse_effective_rank_counts"),
                f"{label} pseudoinverse effective rank",
                token_count=token_count,
            )
        elif any(name in value for name in free_fields):
            raise ReportValidationError(f"{label} contain inapplicable free-bound evidence")
    return {str(name): item for name, item in value.items()}


def _flat_solver_row(
    *,
    scope: str,
    layer: int | None,
    k: int,
    condition: str,
    identifier: str,
    record: Mapping[str, object],
    token_count: int,
    metric: Mapping[str, object] | None,
    svd: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the single flat evidence schema used by both CSV and JSON sources."""

    metric_ranks: dict[str, int] | None = None
    if metric is not None:
        metric_ranks = _histogram(
            metric["effective_rank_counts"],
            "solver row effective rank",
            token_count=token_count,
        )
    free_ranks: dict[str, int] | None = None
    if identifier == "free_bound":
        free_ranks = _histogram(
            record.get("pseudoinverse_effective_rank_counts"),
            "solver row pseudoinverse effective rank",
            token_count=token_count,
        )
        if metric_ranks is not None and metric_ranks != free_ranks:
            raise ReportValidationError("solver row effective-rank evidence disagrees")
        if not isinstance(svd, Mapping):
            raise ReportValidationError("free-bound solver row lacks SVD evidence")
        identity = svd.get("identity")
        sample_count = svd.get("sample_count")
        by_layer = svd.get("sample_count_by_layer")
        maximum = svd.get("max_relative_objective_error")
        if (
            not isinstance(identity, str)
            or not isinstance(sample_count, int)
            or not isinstance(by_layer, Mapping)
            or svd.get("rank_equal") is not True
            or not isinstance(maximum, (int, float))
            or isinstance(maximum, bool)
            or not isfinite(float(maximum))
        ):
            raise ReportValidationError("free-bound solver row has malformed SVD evidence")
        svd_layer_count = (
            by_layer.get(str(layer)) if layer is not None else None
        )
    elif svd is not None:
        raise ReportValidationError("non-free solver row has inapplicable SVD evidence")
    else:
        identity = sample_count = by_layer = maximum = svd_layer_count = None
    bounded = identifier == "bounded"
    row = {
        "scope": scope,
        "layer": layer,
        "k": k,
        "condition": condition,
        "token_count": token_count,
        "coefficient_count": metric.get("coefficient_count") if metric else k * token_count,
        "numerical_failure_count": (
            metric.get("unresolved_failures")
            if metric is not None
            else record.get("bounded_failure_count", 0)
        ),
        "layer_coverage_count": 24 if scope == "pooled" else 1,
        "solver_source_counts": _flat_mapping(
            {str(name): int(count) for name, count in record["solver_source_counts"].items()}  # type: ignore[union-attr]
        ),
        "effective_rank_counts": _flat_mapping(
            metric_ranks if metric_ranks is not None else free_ranks
        ),
        "gram_condition_defined_count": metric.get("gram_condition_defined_count") if metric else None,
        "gram_condition_undefined_count": metric.get("gram_condition_undefined_count") if metric else None,
        "gram_condition_median": metric.get("gram_condition_median") if metric else None,
        "gram_condition_p90": metric.get("gram_condition_p90") if metric else None,
        "gram_condition_p99": metric.get("gram_condition_p99") if metric else None,
        "gram_condition_max": metric.get("gram_condition_max") if metric else None,
        "cholesky_fallback_count": record.get("cholesky_fallback_count") if not bounded else None,
        "degenerate_ridge_count": record.get("degenerate_ridge_count") if not bounded else None,
        "max_relative_system_residual": record.get("max_relative_system_residual") if not bounded else None,
        "tau_g": record.get("tau_g") if identifier == "free_bound" else None,
        "pseudoinverse_effective_rank_counts": _flat_mapping(free_ranks),
        "bounded_lower_saturation_count": record.get("bounded_lower_saturation_count") if bounded else None,
        "bounded_upper_saturation_count": record.get("bounded_upper_saturation_count") if bounded else None,
        "bounded_failure_count": record.get("bounded_failure_count") if bounded else None,
        "bounded_uncertified_count": record.get("bounded_uncertified_count") if bounded else None,
        "bounded_certified_count": record.get("bounded_certified_count") if bounded else None,
        "bounded_solver_candidate_count": record.get("bounded_solver_candidate_count") if bounded else None,
        "bounded_coefficient_bound_violation_count": record.get("bounded_coefficient_bound_violation_count") if bounded else None,
        "bounded_max_primal_residual": record.get("bounded_max_primal_residual") if bounded else None,
        "bounded_max_stationarity_residual": record.get("bounded_max_stationarity_residual") if bounded else None,
        "bounded_max_complementarity_residual": record.get("bounded_max_complementarity_residual") if bounded else None,
        "bounded_max_iterations": record.get("bounded_max_iterations") if bounded else None,
        "bounded_face_enumeration_count": record.get("bounded_face_enumeration_count") if bounded else None,
        "bounded_max_face_enumeration_count": record.get("bounded_max_face_enumeration_count") if bounded else None,
        "svd_identity": identity,
        "svd_sample_count": sample_count,
        "svd_sample_count_by_layer": _flat_mapping(
            {str(name): int(count) for name, count in by_layer.items()}  # type: ignore[union-attr]
        )
        if by_layer is not None
        else None,
        "svd_layer_sample_count": svd_layer_count,
        "svd_rank_equal": True if identifier == "free_bound" else None,
        "svd_max_relative_objective_error": maximum,
    }
    if set(row) != set(_SOLVER_ROW_FIELDS):
        raise AssertionError("solver row schema implementation is incomplete")
    return row


def _selection_conditions(selection: Mapping[str, Any], config: Phase1A5Config) -> dict[int, set[str]]:
    selected: dict[int, set[str]] = {k: {"unit"} for k in config.k_values}
    for field in ("practical_conditions", "free_bound_conditions"):
        values = selection.get(field)
        if not isinstance(values, Mapping):
            raise ReportValidationError("selection manifest is malformed")
        for raw_k, identifier in values.items():
            try:
                k = int(raw_k)
            except (TypeError, ValueError) as error:
                raise ReportValidationError("selection manifest names invalid K") from error
            if k not in selected or not isinstance(identifier, str):
                raise ReportValidationError("selection manifest names an unregistered condition")
            if field == "free_bound_conditions" and identifier != "free_bound":
                raise ReportValidationError("selection manifest free-bound condition is invalid")
            if identifier not in {"bounded", "free_bound", *(f"ridge:{rho:g}" for rho in config.rho_values)}:
                raise ReportValidationError("selection manifest names an unregistered condition")
            selected[k].add(identifier)
    return selected


def _write_source(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _write_csv(
    path: Path,
    rows: list[Mapping[str, object]],
    *,
    schema: tuple[str, ...] | None = None,
) -> None:
    if not rows:
        raise ReportValidationError(f"authoritative source {path.name} is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    observed = {str(key) for row in rows for key in row}
    if schema is not None:
        if len(set(schema)) != len(schema) or observed != set(schema):
            raise ReportValidationError(
                f"authoritative source {path.name} does not match its schema"
            )
        fields = list(schema)
    else:
        fields = sorted(observed)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(path: Path, title: str, rows: list[Mapping[str, object]], x: str, y: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    values = [row for row in rows if isinstance(row.get(x), (int, float)) and isinstance(row.get(y), (int, float))]
    if values:
        axis.scatter([float(row[x]) for row in values], [float(row[y]) for row in values])
    axis.set(title=title, xlabel=x, ylabel=y)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _decision(selection: Mapping[str, Any]) -> str:
    practical = selection["practical_conditions"]
    free = selection["free_bound_conditions"]
    if not free:
        return "No registered numerical reconstruction qualifies; proceed to Phase 1B."
    if not practical:
        return "Registered free bound only; preserve the Registered Free-Reconstruction Candidate Bound and proceed to Phase 1B."
    practical_ids = set(practical.values())
    if "4" in practical:
        return f"K=4 practical frontier with {practical['4']} as the registered practical condition."
    if set(practical) == {"6"}:
        return "K=6-only frontier; retain the 75%-active FFN as the supported fallback frontier."
    if practical_ids == {"bounded"}:
        return "Practical bounded qualifies under the registered local protocol."
    return "Practical ridge qualifies under the registered local protocol."


def _dense_ppl_from_pinned_m0(source: Mapping[str, Any], source_hashes: Mapping[str, str]) -> dict[str, float]:
    root = source.get("source_root")
    expected = source_hashes.get("stage:m0")
    if not isinstance(root, str) or not isinstance(expected, str):
        raise ReportValidationError("pinned M0 dense baseline is missing")
    path = Path(root) / "stages" / "m0.json"
    if not path.is_file() or sha256_file(path) != expected:
        raise ReportValidationError("pinned M0 dense baseline hash drift")
    results = _read_json(path, "pinned M0 stage").get("results")
    if not isinstance(results, Mapping):
        raise ReportValidationError("pinned M0 dense baseline is invalid")
    dense: dict[str, float] = {}
    for corpus in ("wikitext", "fineweb_heldout"):
        entry = results.get(corpus)
        teacher = entry.get("teacher") if isinstance(entry, Mapping) else None
        dense[corpus] = _finite(teacher.get("ppl") if isinstance(teacher, Mapping) else None, f"pinned M0 {corpus} PPL", positive=True)
    return dense


def render_phase1a5_report(artifact_root: Path, config: Phase1A5Config) -> Phase1A5ReportManifest:
    """Render only from complete, immutable, provenance-compatible stage evidence."""
    root = Path(artifact_root)
    _validate_existing_manifest(root)
    source, smoke, local = (_stage(root, name) for name in ("source", "smoke", "local"))
    select = _stage(root, "select", allow_scientific_stop=True)
    selection_stopped = select.get("status") == "scientifically_stopped"
    rollout = None if selection_stopped else _stage(root, "rollout")
    source_hashes = _hashes(source.get("source_hashes"), "source hash manifest")
    source_manifest = source.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise ReportValidationError("source manifest is missing")
    dense_ppl = _dense_ppl_from_pinned_m0(source_manifest, source_hashes)
    stages_to_validate = (("smoke", smoke), ("local", local), ("select", select))
    if rollout is not None:
        stages_to_validate += (("rollout", rollout),)
    for name, stage in stages_to_validate:
        if _hashes(stage.get("source_hashes"), f"{name} source hashes") != source_hashes:
            raise ReportValidationError("source hash drift")
        if stage.get("source_stage_input_signature") != source.get("input_signature"):
            raise ReportValidationError("source stage signature drift")
    if local.get("valid_calibration_tokens") != 100_000 or smoke.get("valid_calibration_tokens") != 2_048:
        raise ReportValidationError("bad token coverage")
    unit = smoke.get("unit_regression")
    if not isinstance(unit, Mapping) or unit.get("passed") is not True or unit.get("checked_tokens") != 2_048:
        raise ReportValidationError("unit regression is incomplete")
    families = smoke.get("families")
    if not isinstance(families, Mapping):
        raise ReportValidationError("smoke family evidence is missing")
    for family in ("unit", "ridge", "free_bound", "bounded"):
        evidence = families.get(family)
        if not isinstance(evidence, Mapping):
            raise ReportValidationError("smoke failed batches")
        elapsed = _finite(evidence.get("elapsed_seconds"), "smoke elapsed")
        if elapsed < 0.0:
            raise ReportValidationError("smoke elapsed is negative")
        peak = evidence.get("peak_cuda_bytes")
        if not isinstance(peak, int) or isinstance(peak, bool) or peak < 0:
            raise ReportValidationError("smoke peak CUDA is invalid")
        if evidence.get("processed_tokens") != 2_048:
            raise ReportValidationError("smoke processed token coverage is invalid")
        if not isinstance(evidence.get("completed_batches"), int) or evidence["completed_batches"] <= 0:
            raise ReportValidationError("smoke completed batches are invalid")
        bounded_included = local.get("bounded_included") is True
        if family != "bounded" and evidence.get("failures") != 0:
            raise ReportValidationError("smoke failed batches")
        if family == "bounded" and bounded_included and evidence.get("all_certified") is not True:
            raise ReportValidationError("bounded KKT evidence is invalid")
        if family == "bounded" and bounded_included:
            if evidence.get("failures") != 0:
                raise ReportValidationError("bounded smoke failed batches")
            if evidence.get("feasible") is not True:
                raise ReportValidationError("bounded smoke is not feasible")
            kkt = evidence.get("kkt_by_k")
            if not isinstance(kkt, Mapping) or set(kkt) != {str(k) for k in config.k_values}:
                raise ReportValidationError("bounded KKT evidence is incomplete")
            for k, record in kkt.items():
                if not isinstance(record, Mapping):
                    raise ReportValidationError("bounded KKT evidence is malformed")
                required = ("bounded_failure_count", "bounded_uncertified_count", "bounded_certified_count", "bounded_solver_candidate_count", "bounded_coefficient_bound_violation_count")
                if any(not isinstance(record.get(name), int) or record[name] < 0 for name in required):
                    raise ReportValidationError("bounded KKT evidence is malformed")
                if record["bounded_failure_count"] or record["bounded_uncertified_count"] or record["bounded_coefficient_bound_violation_count"] or record["bounded_solver_candidate_count"] <= 0 or record["bounded_certified_count"] != record["bounded_solver_candidate_count"]:
                    raise ReportValidationError(f"bounded KKT evidence is invalid for K={k}")
        elif family == "bounded" and evidence.get("feasible") is not True:
            if not isinstance(evidence.get("bounded_stop_reason"), str) or not evidence["bounded_stop_reason"]:
                raise ReportValidationError("bounded engineering stop reason is missing")
        elif family == "bounded":
            raise ReportValidationError("bounded exclusion requires engineering stop")
        if family == "free_bound":
            _validate_svd_verification(evidence.get("svd_verification"), config)

    selection_path = root / "selection-manifest.json"
    expected_selection_hash = select.get("selection_manifest_sha256")
    if not isinstance(expected_selection_hash, str) or not selection_path.is_file() or sha256_file(selection_path) != expected_selection_hash:
        raise ReportValidationError("immutable selection manifest does not match its stage")
    selection = _read_json(selection_path, "selection manifest")
    if _hashes(selection.get("source_hashes"), "selection source hashes") != source_hashes:
        raise ReportValidationError("selection source hash drift")
    if selection.get("local_hashes", {}).get("run_signature") != local.get("result", {}).get("run_signature"):
        raise ReportValidationError("selection local signature drift")
    selected = _selection_conditions(selection, config)
    if selection_stopped:
        if select.get("scientific_stop_reason") != "selection_no_eligible_conditions":
            raise ReportValidationError("selection scientific stop is invalid")
        if selection.get("practical_conditions") or selection.get("free_bound_conditions"):
            raise ReportValidationError("selection scientific stop has eligible conditions")

    local_result = local.get("result")
    if not isinstance(local_result, Mapping) or local_result.get("complete") is not True or local_result.get("failed_batches") != 0:
        raise ReportValidationError("local failed batches")
    local_svd = local_result.get("svd_verification")
    _validate_svd_verification(local_svd, config)
    assert isinstance(local_svd, Mapping)
    metrics = local_result.get("metrics")
    staged_metrics = local.get("selection_metrics")
    if not isinstance(metrics, Mapping) or not isinstance(staged_metrics, list):
        raise ReportValidationError("local metrics are missing")
    expected_ids = {"unit", "free_bound", *(f"ridge:{rho:g}" for rho in config.rho_values)}
    if local.get("bounded_included") is True:
        expected_ids.add("bounded")
    local_rows: list[dict[str, object]] = []
    observed: set[tuple[int, str]] = set()
    for entry in staged_metrics:
        if not isinstance(entry, Mapping):
            raise ReportValidationError("missing local condition")
        k, identifier = entry.get("k"), entry.get("condition")
        if not isinstance(k, int) or k not in config.k_values or not isinstance(identifier, str):
            raise ReportValidationError("missing local condition")
        parsed = _metric(entry.get("metric"), f"local K={k} {identifier}")
        if (k, identifier) in observed:
            raise ReportValidationError("duplicate local metric key")
        observed.add((k, identifier))
        family, separator, rho_value = identifier.partition(":")
        local_rows.append({"k": k, "condition": identifier, "family": family, "rho": float(rho_value) if separator else None, "constraint": "[0,2]" if identifier == "bounded" else None, **parsed, "nmse": parsed["total_squared_error"] / parsed["total_d0_squared_norm"], "active_fraction": k / 8})
    if observed != {(k, identifier) for k in config.k_values for identifier in expected_ids}:
        raise ReportValidationError("missing local condition")
    for row in local_rows:
        k = int(row["k"])
        if (
            row["token_count"] != 24 * 100_000
            or row["coefficient_count"] != k * int(row["token_count"])
        ):
            raise ReportValidationError("pooled local metric coverage is invalid")
    by_condition = {(int(row["k"]), str(row["condition"])): row for row in local_rows}
    gain_recovery_rows: list[dict[str, object]] = []
    for row in local_rows:
        k, identifier = int(row["k"]), str(row["condition"])
        unit_row = by_condition[(k, "unit")]
        free_row = by_condition[(k, "free_bound")]
        denominator = float(unit_row["total_squared_error"]) - float(free_row["total_squared_error"])
        tolerance = 1e-9 * max(abs(float(unit_row["total_squared_error"])), abs(float(free_row["total_squared_error"])), 1.0)
        recovery = None if identifier == "free_bound" or denominator <= tolerance else (float(unit_row["total_squared_error"]) - float(row["total_squared_error"])) / denominator
        gain_recovery_rows.append({"k": k, "condition": identifier, "gain_recovery": recovery})
    persisted_metrics: dict[tuple[int, str], dict[str, float | int | bool]] = {}
    for key, value in metrics.items():
        parsed = _key(key, "local metric")
        if len(parsed) != 2 or not isinstance(parsed[0], int) or not isinstance(parsed[1], str):
            raise ReportValidationError("local persisted metric key is invalid")
        metric_key = (parsed[0], parsed[1])
        if metric_key in persisted_metrics:
            raise ReportValidationError("duplicate local persisted metric key")
        persisted_metrics[metric_key] = _metric(value, "local persisted metric")
    if set(persisted_metrics) != observed:
        raise ReportValidationError("local metric reconciliation is incomplete")
    for row in local_rows:
        key = (int(row["k"]), str(row["condition"]))
        if persisted_metrics[key] != {name: row[name] for name in persisted_metrics[key]}:
            raise ReportValidationError("local metric reconciliation mismatch")
    coefficient_quantiles = local_result.get("coefficient_quantiles")
    if not isinstance(coefficient_quantiles, Mapping):
        raise ReportValidationError("coefficient quantile evidence is missing")
    observed_quantiles: set[tuple[int, str]] = set()
    for key, value in coefficient_quantiles.items():
        parsed = _key(key, "coefficient quantile")
        if (
            len(parsed) != 2
            or not isinstance(parsed[0], int)
            or not isinstance(parsed[1], str)
            or not isinstance(value, Mapping)
        ):
            raise ReportValidationError("coefficient quantile evidence is invalid")
        identity = (parsed[0], parsed[1])
        if identity not in persisted_metrics:
            raise ReportValidationError("coefficient quantile evidence is invalid")
        quantiles = {
            name: _finite(value.get(name), "coefficient quantile evidence")
            for name in (
                "p01",
                "p05",
                "median",
                "p95",
                "p99",
                "p99_signed",
                "max_abs",
            )
        }
        metric = persisted_metrics[identity]
        expected_quantiles = {
            "p01": metric["coefficient_p01"],
            "p05": metric["coefficient_p05"],
            "median": metric["coefficient_median"],
            "p95": metric["coefficient_p95"],
            "p99": metric["coefficient_p99_abs"],
            "p99_signed": metric["coefficient_p99"],
            "max_abs": metric["coefficient_max_abs"],
        }
        if any(quantiles[name] != expected_quantiles[name] for name in quantiles):
            raise ReportValidationError("coefficient quantile evidence mismatch")
        observed_quantiles.add(identity)
    if observed_quantiles != observed:
        raise ReportValidationError("coefficient quantile evidence is incomplete")
    for k, values in selected.items():
        if any((k, identifier) not in observed for identifier in values):
            raise ReportValidationError("selected condition is absent from local metrics")
        if "bounded" in values and local.get("bounded_included") is not True:
            raise ReportValidationError("bounded selection lacks feasible smoke evidence")

    layer_metrics = local_result.get("layer_metrics")
    if not isinstance(layer_metrics, Mapping):
        raise ReportValidationError("layer coverage is missing")
    observed_layers: set[int] = set()
    observed_layer_keys: set[tuple[int, int, str]] = set()
    layer_rows: list[dict[str, object]] = []
    for key, value in layer_metrics.items():
        parsed = _key(key, "layer metric")
        if (
            len(parsed) != 3
            or not isinstance(parsed[0], int)
            or not isinstance(parsed[1], int)
            or not isinstance(parsed[2], str)
        ):
            raise ReportValidationError("layer coverage is invalid")
        metric = _metric(value, "layer metric")
        observed_layers.add(parsed[0])
        identity = (parsed[0], parsed[1], parsed[2])
        observed_layer_keys.add(identity)
        if (
            metric["token_count"] != 100_000
            or metric["coefficient_count"]
            != parsed[1] * int(metric["token_count"])
        ):
            raise ReportValidationError("layer metric coverage is invalid")
        layer_rows.append({"layer": parsed[0], "k": parsed[1], "condition": parsed[2], "nmse": metric["total_squared_error"] / metric["total_d0_squared_norm"], **metric})
    expected_layer_keys = {
        (layer, k, identifier)
        for layer in range(24)
        for k in config.k_values
        for identifier in expected_ids
    }
    if observed_layers != set(range(24)) or observed_layer_keys != expected_layer_keys:
        raise ReportValidationError("layer coverage is incomplete")
    solver = local_result.get("solver_diagnostics")
    if not isinstance(solver, Mapping):
        raise ReportValidationError("solver diagnostics are missing")
    solver_rows: list[dict[str, object]] = []
    observed_solver: set[tuple[int, str]] = set()
    for key, value in solver.items():
        parsed = _key(key, "solver diagnostic")
        if (
            len(parsed) != 2
            or not isinstance(parsed[0], int)
            or not isinstance(parsed[1], str)
            or (parsed[0], parsed[1]) not in observed
        ):
            raise ReportValidationError("solver diagnostics are invalid")
        identity = (parsed[0], parsed[1])
        if identity in observed_solver:
            raise ReportValidationError("duplicate solver diagnostics")
        metric = persisted_metrics[identity]
        record = _validate_solver_record(
            value,
            "solver diagnostics",
            identifier=parsed[1],
            token_count=int(metric["token_count"]),
            expected_tau_g=config.tau_g,
        )
        observed_solver.add(identity)
        if parsed[1] == "bounded" and any(
            record[name] != metric[name]
            for name in (
                "bounded_lower_saturation_count",
                "bounded_upper_saturation_count",
            )
        ):
            raise ReportValidationError("bounded pooled solver diagnostics disagree with metrics")
        solver_rows.append(
            _flat_solver_row(
                scope="pooled",
                layer=None,
                k=parsed[0],
                condition=parsed[1],
                identifier=parsed[1],
                record=record,
                token_count=int(metric["token_count"]),
                metric=metric,
                svd=(
                    local_svd[f"sparse:K{parsed[0]}:free_bound"]
                    if parsed[1] == "free_bound"
                    else None
                ),
            )
        )
        if parsed[1] == "bounded" and (record.get("bounded_failure_count", 0) != 0 or record.get("bounded_uncertified_count", 0) != 0):
            raise ReportValidationError("bounded KKT evidence is invalid")
    if observed_solver != observed:
        raise ReportValidationError("solver diagnostics are incomplete")
    layer_solver = local_result.get("layer_solver_diagnostics")
    if not isinstance(layer_solver, Mapping):
        raise ReportValidationError("layer solver diagnostics are missing")
    observed_layer_solver: set[tuple[int, int, str]] = set()
    for key, value in layer_solver.items():
        parsed = _key(key, "layer solver diagnostic")
        if (
            len(parsed) != 3
            or not isinstance(parsed[0], int)
            or not isinstance(parsed[1], int)
            or not isinstance(parsed[2], str)
        ):
            raise ReportValidationError("layer solver diagnostics are invalid")
        identity = (parsed[0], parsed[1], parsed[2])
        if identity not in observed_layer_keys:
            raise ReportValidationError("layer solver diagnostics are invalid")
        if identity in observed_layer_solver:
            raise ReportValidationError("duplicate layer solver diagnostics")
        layer_metric = next(
            row
            for row in layer_rows
            if (row["layer"], row["k"], row["condition"]) == identity
        )
        record = _validate_solver_record(
            value,
            "layer solver diagnostics",
            identifier=parsed[2],
            token_count=int(layer_metric["token_count"]),
            expected_tau_g=config.tau_g,
        )
        observed_layer_solver.add(identity)
        if parsed[2] == "bounded" and any(
            record[name] != layer_metric[name]
            for name in (
                "bounded_lower_saturation_count",
                "bounded_upper_saturation_count",
            )
        ):
            raise ReportValidationError("bounded layer solver diagnostics disagree with metrics")
        solver_rows.append(
            _flat_solver_row(
                scope="layer",
                layer=parsed[0],
                k=parsed[1],
                condition=parsed[2],
                identifier=parsed[2],
                record=record,
                token_count=int(layer_metric["token_count"]),
                metric=layer_metric,
                svd=(
                    local_svd[f"sparse:K{parsed[1]}:free_bound"]
                    if parsed[2] == "free_bound"
                    else None
                ),
            )
        )
    if observed_layer_solver != observed_layer_keys:
        raise ReportValidationError("layer solver diagnostics are incomplete")
    reference_solver = local_result.get("reference_diagnostics")
    if not isinstance(reference_solver, Mapping):
        raise ReportValidationError("R8 solver diagnostics are missing")
    expected_reference_solver = {
        (layer, f"R8[{identifier}]")
        for layer in range(24)
        for _, identifier in observed
    }
    observed_reference_solver: set[tuple[int, str]] = set()
    for key, value in reference_solver.items():
        parsed = _key(key, "R8 solver diagnostic")
        if (
            len(parsed) != 2
            or not isinstance(parsed[0], int)
            or not isinstance(parsed[1], str)
            or (parsed[0], parsed[1]) not in expected_reference_solver
        ):
            raise ReportValidationError("R8 solver diagnostics are invalid")
        identifier = parsed[1].removeprefix("R8[").removesuffix("]")
        if parsed[1] != f"R8[{identifier}]":
            raise ReportValidationError("R8 solver diagnostics are invalid")
        record = _validate_solver_record(
            value,
            "R8 solver diagnostics",
            identifier=identifier,
            token_count=100_000,
            expected_tau_g=config.tau_g,
        )
        if identifier == "bounded" and (
            record.get("bounded_failure_count", 0) != 0
            or record.get("bounded_uncertified_count", 0) != 0
        ):
            raise ReportValidationError("bounded KKT evidence is invalid")
        if (parsed[0], parsed[1]) in observed_reference_solver:
            raise ReportValidationError("duplicate R8 solver diagnostics")
        observed_reference_solver.add((parsed[0], parsed[1]))
        solver_rows.append(
            _flat_solver_row(
                scope="r8",
                layer=parsed[0],
                k=8,
                condition=parsed[1],
                identifier=identifier,
                record=record,
                token_count=100_000,
                metric=None,
                svd=local_svd["R8[free_bound]"] if identifier == "free_bound" else None,
            )
        )
    if observed_reference_solver != expected_reference_solver:
        raise ReportValidationError("R8 solver diagnostics are incomplete")
    rank_condition_rows: list[dict[str, object]] = []
    for row in local_rows:
        token_count = int(row["token_count"])
        rank_counts, mean_rank = _mean_effective_rank(
            row["effective_rank_counts"],
            "effective-rank plot source",
            token_count=token_count,
        )
        unit_squared_error = float(
            by_condition[(int(row["k"]), "unit")]["total_squared_error"]
        )
        if unit_squared_error <= 0.0:
            raise ReportValidationError("effective-rank gain has no positive unit loss")
        rank_condition_rows.append(
            {
                "scope": "pooled",
                "k": row["k"],
                "condition": row["condition"],
                "effective_rank_counts": rank_counts,
                "mean_effective_rank": mean_rank,
                "gram_condition_median": row["gram_condition_median"],
                "gram_condition_p90": row["gram_condition_p90"],
                "gram_condition_p99": row["gram_condition_p99"],
                "gram_condition_max": row["gram_condition_max"],
                "gram_condition_undefined_count": row[
                    "gram_condition_undefined_count"
                ],
                "reconstruction_gain": (
                    unit_squared_error - float(row["total_squared_error"])
                )
                / unit_squared_error,
            }
        )

    expected_rollout = set() if selection_stopped else {f"{corpus}:k{k}:{condition}" for corpus in ("wikitext", "fineweb_heldout") for k, conditions in selected.items() for condition in conditions}
    results = {} if rollout is None else rollout.get("results")
    if not isinstance(results, Mapping) or set(results) != expected_rollout:
        raise ReportValidationError("evaluation condition leakage or incomplete registered rollout")
    evaluation_rows: list[dict[str, object]] = []
    for name, result in results.items():
        if not isinstance(result, Mapping) or result.get("complete") is not True or result.get("failed_batches") != 0:
            raise ReportValidationError("evaluation failed batches")
        ppl = _finite(result.get("ppl"), f"evaluation PPL {name}", positive=True)
        scored = _integer(result.get("scored_tokens"), f"evaluation scored tokens {name}", positive=True)
        layers = result.get("layer_metrics")
        if not isinstance(layers, Mapping) or {int(layer) for layer in layers} != set(range(24)):
            raise ReportValidationError("evaluation layer coverage is incomplete")
        for metric in layers.values():
            if not isinstance(metric, Mapping):
                raise ReportValidationError("evaluation layer metric is invalid")
            _finite(metric.get("hidden_squared_error_sum"), "evaluation hidden error")
            _finite(metric.get("teacher_squared_norm_sum"), "evaluation teacher norm", positive=True)
        corpus, k_token, condition = str(name).split(":", 2)
        evaluation_rows.append({"corpus": corpus, "k": int(k_token.removeprefix("k")), "condition": condition, "ppl": ppl, "scored_tokens": scored})

    reference_rows = []
    references = local_result.get("reference_distances", {})
    if not isinstance(references, Mapping):
        raise ReportValidationError("reference distances are invalid")
    for key, value in references.items():
        if not isinstance(value, Mapping):
            raise ReportValidationError("reference distances are invalid")
        reference_rows.append({"key": key, **{str(k): v for k, v in value.items()}})
    reference_labels = {
        str(item)
        for row in reference_rows
        for item in _key(str(row["key"]), "reference distance")[1:]
    }
    if not {"D0", "E8"}.issubset(reference_labels) or not any(
        label.startswith("R8[") for label in reference_labels
    ):
        raise ReportValidationError("D0/E8/R8 distance evidence is incomplete")
    stability = local_result.get("subset_stability")
    if not isinstance(stability, Mapping) or not stability:
        raise ReportValidationError("subset stability evidence is absent")
    stability_rows: list[dict[str, object]] = []
    expected_stability = {
        (k, left, right)
        for k in config.k_values
        for index, left in enumerate(sorted(identifier for candidate_k, identifier in observed if candidate_k == k))
        for right in sorted(identifier for candidate_k, identifier in observed if candidate_k == k)[index + 1 :]
    }
    seen_stability: set[tuple[int, str, str]] = set()
    for key, value in stability.items():
        if not isinstance(value, Mapping):
            raise ReportValidationError("subset stability evidence is malformed")
        parsed = _key(key, "subset stability")
        if len(parsed) != 3 or not isinstance(parsed[0], int) or not isinstance(parsed[1], str) or not isinstance(parsed[2], str):
            raise ReportValidationError("subset stability evidence is malformed")
        identity = (parsed[0], parsed[1], parsed[2])
        if identity in seen_stability:
            raise ReportValidationError("duplicate subset stability evidence")
        seen_stability.add(identity)
        total = _finite(value.get("jaccard_sum"), "subset stability")
        tokens = _integer(value.get("token_count"), "subset stability token count", positive=True)
        score = total / tokens
        if score < 0.0 or score > 1.0:
            raise ReportValidationError("subset stability is outside [0,1]")
        stability_rows.append({"key": key, "jaccard": score, **{str(name): item for name, item in value.items()}})
    if seen_stability != expected_stability:
        raise ReportValidationError("subset stability pair coverage is incomplete")
    plot_source = root / "plot-source"
    sources = {
        "local-metrics.json": local_rows,
        "coefficient-distributions.json": [
            {
                key: row[key]
                for key in (
                    "k",
                    "condition",
                    "coefficient_count",
                    "coefficient_mean",
                    "coefficient_std",
                    "coefficient_p01",
                    "coefficient_p05",
                    "coefficient_median",
                    "coefficient_p95",
                    "coefficient_p99",
                    "coefficient_p99_abs",
                    "coefficient_max_abs",
                    "negative_count",
                    "negative_rate",
                    "above_two_count",
                    "above_two_rate",
                    "above_four_count",
                    "above_four_rate",
                )
            }
            for row in local_rows
        ],
        "bound-saturation.json": [
            {
                key: row[key]
                for key in (
                    "k",
                    "condition",
                    "bounded_lower_saturation_count",
                    "bounded_lower_saturation_rate",
                    "bounded_upper_saturation_count",
                    "bounded_upper_saturation_rate",
                )
            }
            for row in local_rows
            if row["condition"] == "bounded"
        ],
        "coefficient-distortion-gain.json": [
            {
                "k": row["k"],
                "condition": row["condition"],
                "mean_absolute_deviation_from_one": row[
                    "mean_absolute_deviation_from_one"
                ],
                "l2_deviation_from_one": row["l2_deviation_from_one"],
                "l2_deviation_denominator": row["l2_deviation_denominator"],
                "local_relative_improvement": (
                    float(by_condition[(int(row["k"]), "unit")]["total_squared_error"])
                    - float(row["total_squared_error"])
                ) / float(by_condition[(int(row["k"]), "unit")]["total_squared_error"]),
                "gain_recovery": next(
                    item["gain_recovery"]
                    for item in gain_recovery_rows
                    if item["k"] == row["k"]
                    and item["condition"] == row["condition"]
                ),
            }
            for row in local_rows
        ],
        "gain-recovery.json": gain_recovery_rows,
        "rank-conditioning.json": rank_condition_rows,
        "effective-rank.json": [
            {
                "scope": row["scope"],
                "k": row["k"],
                "condition": row["condition"],
                "effective_rank_counts": row["effective_rank_counts"],
                "mean_effective_rank": row["mean_effective_rank"],
                "reconstruction_gain": row["reconstruction_gain"],
            }
            for row in rank_condition_rows
        ],
        "solver-diagnostics.json": solver_rows,
        "layer-improvements.json": layer_rows,
        "subset-stability.json": stability_rows,
        "reference-distances.json": reference_rows,
        "evaluation-ppl.json": evaluation_rows,
    }
    for name, value in sources.items():
        _write_source(plot_source / name, value)
    feasibility_rows = [
        {"family": family, "feasible": evidence.get("feasible"), "bounded_stop_reason": evidence.get("bounded_stop_reason")}
        for family, evidence in families.items()
        if isinstance(evidence, Mapping)
    ]
    _write_source(plot_source / "smoke-feasibility.json", feasibility_rows)
    _write_csv(
        plot_source / "solver-diagnostics.csv",
        solver_rows,
        schema=_SOLVER_ROW_FIELDS,
    )
    primary_rows: list[dict[str, object]] = []
    for row in local_rows:
        matching = [item for item in evaluation_rows if item["k"] == row["k"] and item["condition"] == row["condition"]]
        if not matching:
            matching = [{"corpus": None, "ppl": None, "scored_tokens": None}]
        unit_sse = float(by_condition[(int(row["k"]), "unit")]["total_squared_error"])
        for evaluation in matching:
            corpus, ppl = evaluation["corpus"], evaluation["ppl"]
            evaluated = corpus is not None
            baseline = dense_ppl[str(corpus)] if evaluated else None
            primary_rows.append({**row, "local_relative_improvement": (unit_sse - float(row["total_squared_error"])) / unit_sse, "corpus": corpus, "ppl": ppl, "ppl_delta": float(ppl) - baseline if evaluated else None, "ppl_ratio": float(ppl) / baseline if evaluated else None, "scored_tokens": evaluation["scored_tokens"], "deployment_status": "evaluated" if evaluated else "not_evaluated"})
    _write_source(plot_source / "primary-table.json", primary_rows)
    _write_csv(plot_source / "primary-table.csv", primary_rows)
    plots = root / "plots"
    specs = (
        ("01-local-nmse-rho.png", "Local NMSE versus ridge rho", local_rows, "rho", "nmse"),
        ("02-local-active-fraction.png", "Local NMSE versus active fraction", local_rows, "active_fraction", "nmse"),
        ("03-evaluation-ppl.png", "Evaluation PPL", evaluation_rows, "k", "ppl"),
        ("04-coefficient-distributions.png", "Coefficient above-two rate", local_rows, "active_fraction", "above_two_rate"),
        ("05-coefficient-distortion-gain.png", "Coefficient distortion versus gain", sources["coefficient-distortion-gain.json"], "mean_absolute_deviation_from_one", "local_relative_improvement"),
        ("06-rank-conditioning-gain.png", "Winning-subset Gram conditioning", sources["rank-conditioning.json"], "gram_condition_p99", "reconstruction_gain"),
        ("07-layer-improvements.png", "Layerwise local NMSE", layer_rows, "layer", "nmse"),
        ("08-reference-distances.png", "D0/E8/R8 reference distances", reference_rows, "token_count", "squared_error_sum"),
        ("09-subset-stability.png", "Cross-condition subset Jaccard", stability_rows, "token_count", "jaccard"),
        ("10-effective-rank-gain.png", "Effective rank versus reconstruction gain", sources["effective-rank.json"], "mean_effective_rank", "reconstruction_gain"),
    )
    for name, title, rows, x, y in specs:
        _save_plot(plots / name, title, rows, x, y)
    outputs = [*sorted(plot_source.glob("*.json")), *sorted(plot_source.glob("*.csv")), *sorted(plots.glob("*.png"))]
    output_hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in outputs}
    stage_names = ("source", "smoke", "local", "select") if selection_stopped else ("source", "smoke", "local", "select", "rollout")
    source_stage_hashes = {f"stages/{name}.json": sha256_file(root / "stages" / f"{name}.json") for name in stage_names}
    decision = _decision(selection)
    bounded_stop_reason = families["bounded"].get("bounded_stop_reason") if isinstance(families.get("bounded"), Mapping) else None
    report_lines = [
        "# Phase 1A.5 sparse reconstruction oracle report", "",
        f"Decision: {decision}", "",
        "The Registered Free-Reconstruction Candidate Bound is a registered candidate bound, not an exact-span optimum.",
        "This report makes no claim of an exact-span optimum, a trained router, or an inference speedup.",
        "Gain recovery is analysis-only and never affects eligibility.",
        (
            f"Cosine quantiles exclude {sum(int(row['cosine_undefined_count']) for row in local_rows)} "
            "undefined cosine samples caused by zero-vector denominators."
        ),
    ]
    if isinstance(bounded_stop_reason, str):
        report_lines.append(f"Bounded engineering stop: {bounded_stop_reason}.")
    report = "\n".join(report_lines)
    (root / "report.md").write_text(report, encoding="utf-8")
    output_hashes["report.md"] = sha256_file(root / "report.md")
    manifest = Phase1A5ReportManifest(
        output_hashes=output_hashes,
        source_stage_hashes=source_stage_hashes,
        decision=decision,
        local_conditions={str(k): tuple(sorted(identifier for candidate_k, identifier in observed if candidate_k == k)) for k in config.k_values},
        rollout_conditions={str(k): tuple(sorted(values)) for k, values in selected.items()},
    )
    atomic_write_json(root / "report-manifest.json", asdict(manifest))
    return manifest
