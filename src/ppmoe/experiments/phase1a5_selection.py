"""Calibration-only immutable selection for the Phase 1A.5 local oracle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Mapping

from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.oracle.reconstruction_local import ReconstructionLocalResult
from ppmoe.provenance import atomic_write_json


@dataclass(frozen=True, slots=True)
class SelectionManifest:
    practical_conditions: Mapping[int, str]
    free_bound_conditions: Mapping[int, str]
    rejected: Mapping[tuple[int, str], str]
    diagnostics: Mapping[tuple[int, str], str]
    source_hashes: Mapping[str, str]
    local_hashes: Mapping[str, str]
    selection_frozen_at: str


def _payload(manifest: SelectionManifest) -> dict[str, object]:
    return {
        "practical_conditions": {
            str(k): v for k, v in manifest.practical_conditions.items()
        },
        "free_bound_conditions": {
            str(k): v for k, v in manifest.free_bound_conditions.items()
        },
        "rejected": {
            f"{k}|{condition}": value
            for (k, condition), value in manifest.rejected.items()
        },
        "diagnostics": {
            f"{k}|{condition}": value
            for (k, condition), value in manifest.diagnostics.items()
        },
        "source_hashes": dict(manifest.source_hashes),
        "local_hashes": dict(manifest.local_hashes),
        "selection_frozen_at": manifest.selection_frozen_at,
    }


def _from_payload(payload: Mapping[str, object]) -> SelectionManifest:
    def split(values: object) -> dict[tuple[int, str], str]:
        return {
            (int(key.split("|", 1)[0]), key.split("|", 1)[1]): str(value)
            for key, value in dict(values).items()
        }

    return SelectionManifest(
        {int(k): str(v) for k, v in dict(payload["practical_conditions"]).items()},
        {int(k): str(v) for k, v in dict(payload["free_bound_conditions"]).items()},
        split(payload["rejected"]),
        split(payload["diagnostics"]),
        {str(k): str(v) for k, v in dict(payload["source_hashes"]).items()},
        {str(k): str(v) for k, v in dict(payload["local_hashes"]).items()},
        str(payload["selection_frozen_at"]),
    )


def select_calibration_conditions(
    result: ReconstructionLocalResult, config: Phase1A5Config, output_path: Path
) -> SelectionManifest:
    """Choose only from local metrics and atomically freeze the outcome once."""
    if not result.complete or result.failed_batches:
        raise ValueError(
            "incomplete local reconstruction result cannot select conditions"
        )
    practical: dict[int, str] = {}
    free_bound: dict[int, str] = {}
    rejected: dict[tuple[int, str], str] = {}
    diagnostics: dict[tuple[int, str], str] = {}
    for k in config.k_values:
        unit = result.metrics.get((k, "unit"))
        if unit is None or not unit.finite or unit.total_squared_error <= 0.0:
            raise ValueError(f"K={k} has no finite unit baseline")
        candidates: list[tuple[object, str]] = []
        for (candidate_k, identifier), metric in result.metrics.items():
            if candidate_k != k or identifier == "unit":
                continue
            improvement = (
                unit.total_squared_error - metric.total_squared_error
            ) / unit.total_squared_error
            if (
                not isfinite(metric.aggregate_nmse)
                or not metric.finite
                or metric.unresolved_failures
            ):
                rejected[(k, identifier)] = "non-finite metrics or unresolved failures"
                continue
            if improvement < config.local_improvement_gate:
                rejected[(k, identifier)] = "does not meet 30% pooled-NMSE reduction"
                continue
            if identifier == "free_bound":
                free_bound[k] = identifier
                continue
            bounded = identifier == "bounded"
            if metric.above_two_rate > config.practical_above_two_rate_max:
                rejected[(k, identifier)] = "above-2 coefficient rate exceeds 1%"
                continue
            if (
                not bounded
                and metric.negative_rate > config.practical_negative_rate_max
            ):
                rejected[(k, identifier)] = "negative coefficient rate exceeds 5%"
                continue
            if bounded:
                diagnostics[(k, identifier)] = (
                    f"bounded saturation lower={metric.bounded_lower_saturation_count} upper={metric.bounded_upper_saturation_count}"
                )
            candidates.append((metric, identifier))
        if candidates:

            def ordering(item: tuple[object, str]) -> tuple[float, float, float]:
                metric, identifier = item
                rho = (
                    float(identifier.split(":", 1)[1])
                    if identifier.startswith("ridge:")
                    else 0.0
                )
                return (
                    metric.aggregate_nmse,
                    metric.mean_absolute_deviation_from_one,
                    -rho,
                )

            practical[k] = min(candidates, key=ordering)[1]
    metric_payload = {
        f"{k}|{identifier}": asdict(metric)
        for (k, identifier), metric in sorted(result.metrics.items())
    }
    metrics_sha256 = sha256(
        json.dumps(metric_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    manifest = SelectionManifest(
        practical,
        free_bound,
        rejected,
        diagnostics,
        dict(result.source_hashes),
        {"run_signature": result.run_signature, "metrics_sha256": metrics_sha256},
        datetime.now(timezone.utc).isoformat(),
    )
    proposed = _payload(manifest)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        # The timestamp is part of the frozen content but repeated identical selection
        # returns the prior immutable manifest rather than attempting an overwrite.
        comparison = dict(proposed)
        comparison.pop("selection_frozen_at")
        existing_comparison = dict(existing)
        existing_comparison.pop("selection_frozen_at", None)
        if existing_comparison != comparison:
            raise ValueError(
                "selection manifest is immutable and differs from existing content"
            )
        return _from_payload(existing)
    atomic_write_json(output_path, proposed)
    return manifest
