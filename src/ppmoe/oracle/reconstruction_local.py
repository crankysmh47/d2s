"""Checkpointed calibration-only aggregation for reconstruction conditions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
from math import ceil, isfinite, sqrt
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from ppmoe.adapters.qwen35_runtime import iter_text_mlps
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.moefication.qwen35 import PartitionedQwen35MLP
from ppmoe.oracle.reconstruction import (
    ReconstructionCondition,
    ReconstructionResult,
    SolverSettings,
    score_full_reference,
    score_reconstruction,
    verify_pseudoinverse_svd,
)
from ppmoe.provenance import atomic_write_json


def condition_id(condition: ReconstructionCondition) -> str:
    """Return the stable, human-readable ID persisted in local artifacts."""

    return condition.family if condition.rho is None else f"ridge:{condition.rho:g}"


@dataclass(frozen=True, slots=True)
class ConditionMetrics:
    token_count: int
    total_squared_error: float
    total_d0_squared_norm: float
    coefficient_count: int
    negative_count: int
    above_two_count: int
    absolute_deviation_sum: float
    unresolved_failures: int
    bounded_lower_saturation_count: int
    bounded_upper_saturation_count: int
    finite: bool
    coefficient_sum: float = 0.0
    coefficient_squared_sum: float = 0.0
    above_four_count: int = 0
    squared_deviation_sum: float = 0.0
    relative_error_sum: float = 0.0
    cosine_sum: float = 0.0
    cosine_defined_count: int = 0
    cosine_undefined_count: int = 0
    gram_condition_defined_count: int = 0
    gram_condition_undefined_count: int = 0
    coefficient_mean: float | None = None
    coefficient_std: float | None = None
    coefficient_p01: float | None = None
    coefficient_p05: float | None = None
    coefficient_median: float | None = None
    coefficient_p95: float | None = None
    coefficient_p99: float | None = None
    coefficient_p99_abs: float | None = None
    coefficient_max_abs: float | None = None
    negative_rate: float | None = None
    above_two_rate: float | None = None
    above_four_rate: float | None = None
    bounded_lower_saturation_rate: float | None = None
    bounded_upper_saturation_rate: float | None = None
    mean_absolute_deviation_from_one: float | None = None
    l2_deviation_from_one: float | None = None
    l2_deviation_denominator: int = 0
    relative_error_mean: float | None = None
    relative_error_median: float | None = None
    relative_error_p90: float | None = None
    relative_error_p95: float | None = None
    relative_error_p99: float | None = None
    cosine_mean: float | None = None
    cosine_median: float | None = None
    cosine_p90: float | None = None
    cosine_p95: float | None = None
    cosine_p99: float | None = None
    gram_condition_median: float | None = None
    gram_condition_p90: float | None = None
    gram_condition_p99: float | None = None
    gram_condition_max: float | None = None
    effective_rank_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coefficient_count = self.coefficient_count
        defaults = {
            "coefficient_mean": self.coefficient_sum / coefficient_count
            if coefficient_count
            else 0.0,
            "coefficient_std": sqrt(
                max(
                    self.coefficient_squared_sum / coefficient_count
                    - (self.coefficient_sum / coefficient_count) ** 2,
                    0.0,
                )
            )
            if coefficient_count
            else 0.0,
            "negative_rate": self.negative_count / coefficient_count
            if coefficient_count
            else 0.0,
            "above_two_rate": self.above_two_count / coefficient_count
            if coefficient_count
            else 0.0,
            "above_four_rate": self.above_four_count / coefficient_count
            if coefficient_count
            else 0.0,
            "bounded_lower_saturation_rate": self.bounded_lower_saturation_count
            / coefficient_count
            if coefficient_count
            else 0.0,
            "bounded_upper_saturation_rate": self.bounded_upper_saturation_count
            / coefficient_count
            if coefficient_count
            else 0.0,
            "mean_absolute_deviation_from_one": self.absolute_deviation_sum
            / coefficient_count
            if coefficient_count
            else 0.0,
            "l2_deviation_from_one": sqrt(
                self.squared_deviation_sum / coefficient_count
            )
            if coefficient_count
            else 0.0,
            "relative_error_mean": self.relative_error_sum / self.token_count
            if self.token_count
            else 0.0,
            "cosine_mean": self.cosine_sum / self.cosine_defined_count
            if self.cosine_defined_count
            else None,
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
        if self.l2_deviation_denominator == 0 and coefficient_count:
            object.__setattr__(self, "l2_deviation_denominator", coefficient_count)

    @property
    def aggregate_nmse(self) -> float:
        if self.total_d0_squared_norm == 0.0:
            return 0.0 if self.total_squared_error == 0.0 else float("inf")
        return self.total_squared_error / self.total_d0_squared_norm



@dataclass(frozen=True, slots=True)
class SVDVerificationEvidence:
    identity: str
    sample_count: int
    sample_count_by_layer: Mapping[int, int]
    rank_equal: bool
    max_relative_objective_error: float


@dataclass(frozen=True, slots=True)
class ReferenceDistance:
    squared_error_sum: float
    left_squared_norm_sum: float
    right_squared_norm_sum: float
    dot_sum: float
    token_count: int

    @property
    def relative_error(self) -> float:
        if self.left_squared_norm_sum == 0.0:
            return 0.0 if self.squared_error_sum == 0.0 else float("inf")
        return self.squared_error_sum / self.left_squared_norm_sum

    @property
    def cosine(self) -> float:
        denominator = (self.left_squared_norm_sum * self.right_squared_norm_sum) ** 0.5
        return self.dot_sum / denominator if denominator else 1.0


@dataclass(frozen=True, slots=True)
class SubsetJaccard:
    jaccard_sum: float
    token_count: int

    @property
    def mean_jaccard(self) -> float:
        return self.jaccard_sum / self.token_count if self.token_count else float("nan")


@dataclass(frozen=True, slots=True)
class ReconstructionLocalResult:
    metrics: Mapping[tuple[int, str], ConditionMetrics]
    layer_metrics: Mapping[tuple[int, int, str], ConditionMetrics]
    selection_counts: Mapping[tuple[int, int, str], tuple[int, ...]]
    co_selection: Mapping[tuple[int, int, str], tuple[tuple[int, ...], ...]]
    solver_diagnostics: Mapping[tuple[int, str], Mapping[str, object]]
    reference_distances: Mapping[tuple[int, str, str], ReferenceDistance]
    reference_diagnostics: Mapping[tuple[int, str], Mapping[str, object]]
    coefficient_quantiles: Mapping[tuple[int, str], Mapping[str, float]]
    completed_batches: int
    failed_batches: int
    complete: bool
    run_signature: str
    source_hashes: Mapping[str, str]
    subset_stability: Mapping[tuple[int, str, str], SubsetJaccard] = field(
        default_factory=dict
    )
    layer_solver_diagnostics: Mapping[
        tuple[int, int, str], Mapping[str, object]
    ] = field(default_factory=dict)
    svd_verification: Mapping[str, SVDVerificationEvidence] = field(
        default_factory=dict
    )

    def relative_improvement(self, k: int, identifier: str) -> float | None:
        """Return pooled SSE improvement over unit; this is never an eligibility rule."""
        unit = self.metrics.get((k, "unit"))
        candidate = self.metrics.get((k, identifier))
        if unit is None or candidate is None or unit.total_squared_error <= 0.0:
            return None
        return (
            unit.total_squared_error - candidate.total_squared_error
        ) / unit.total_squared_error

    def gain_recovery(
        self, k: int, identifier: str, *, objective_tolerance: float = 1e-9
    ) -> float | None:
        """Return analysis-only recovery against the registered free-bound SSE."""
        unit = self.metrics.get((k, "unit"))
        candidate = self.metrics.get((k, identifier))
        free_bound = self.metrics.get((k, "free_bound"))
        if unit is None or candidate is None or free_bound is None:
            return None
        denominator = unit.total_squared_error - free_bound.total_squared_error
        tolerance = objective_tolerance * max(
            abs(unit.total_squared_error), abs(free_bound.total_squared_error), 1.0
        )
        if denominator <= tolerance:
            return None
        return (unit.total_squared_error - candidate.total_squared_error) / denominator


class ReconstructionLocalAccumulator:
    """Bounded-memory transaction accumulator; only completed batches are durable."""

    _STATE_NAME = "reconstruction_local_state.json"

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        conditions: Mapping[int, Sequence[ReconstructionCondition]],
        checkpoint_dir: Path | None,
        run_signature: str,
        source_hashes: Mapping[str, str] | None = None,
        solver_settings: SolverSettings = SolverSettings(),
    ) -> None:
        if num_layers <= 0 or num_experts != 8:
            raise ValueError(
                "reconstruction local requires positive layers and exactly eight experts"
            )
        if not conditions or any(
            not 1 <= k <= 8 or not values for k, values in conditions.items()
        ):
            raise ValueError(
                "conditions must provide non-empty condition lists for K in [1, 8]"
            )
        self.num_layers, self.num_experts = num_layers, num_experts
        self.conditions = {int(k): tuple(values) for k, values in conditions.items()}
        self._condition_keys = tuple(
            (k, condition_id(condition))
            for k in sorted(self.conditions)
            for condition in self.conditions[k]
        )
        if len(set(self._condition_keys)) != len(self._condition_keys):
            raise ValueError("condition IDs must be unique within each K")
        self.checkpoint_dir, self.run_signature = checkpoint_dir, run_signature
        self.source_hashes = dict(source_hashes or {})
        self.solver_settings = solver_settings
        self.completed_batches: set[int] = set()
        self.failures: dict[int, str] = {}
        self._metrics = {
            key: self._empty_metric_state()
            for key in self._condition_keys
        }
        self._layer_metrics = {
            (layer, *key): self._empty_metric_state()
            for layer in range(num_layers)
            for key in self._condition_keys
        }
        self._selection = {
            (layer, *key): [0] * num_experts
            for layer in range(num_layers)
            for key in self._condition_keys
        }
        self._co_selection = {
            (layer, *key): [[0] * num_experts for _ in range(num_experts)]
            for layer in range(num_layers)
            for key in self._condition_keys
        }
        self._solver: dict[tuple[int, str], dict[str, object]] = {
            key: {} for key in self._condition_keys
        }
        self._layer_solver: dict[tuple[int, int, str], dict[str, object]] = {
            (layer, *key): {}
            for layer in range(num_layers)
            for key in self._condition_keys
        }
        self._references: dict[tuple[int, str, str], list[float | int]] = {}
        self._reference_solver: dict[tuple[int, str], dict[str, object]] = {}
        self._subset_stability = {
            (k, left, right): [0.0, 0]
            for k, values in self.conditions.items()
            for index, left in enumerate(sorted(condition_id(value) for value in values))
            for right in sorted(condition_id(value) for value in values)[index + 1 :]
        }
        self._pending_distributions: dict[
            tuple[int, str], dict[int, dict[str, list[float]]]
        ] = {
            key: {} for key in self._condition_keys
        }
        self._svd_verification: dict[str, dict[str, object]] = {
            **{
                f"sparse:K{k}:free_bound": {
                    "identity": f"sparse:K{k}:free_bound",
                    "sample_count": 0,
                    "sample_count_by_layer": {layer: 0 for layer in range(num_layers)},
                    "rank_equal": True,
                    "max_relative_objective_error": 0.0,
                }
                for k in sorted(self.conditions)
                if any(value.family == "free_bound" for value in self.conditions[k])
            },
            **(
                {
                    "R8[free_bound]": {
                        "identity": "R8[free_bound]",
                        "sample_count": 0,
                        "sample_count_by_layer": {
                            layer: 0 for layer in range(num_layers)
                        },
                        "rank_equal": True,
                        "max_relative_objective_error": 0.0,
                    }
                }
                if any(
                    value.family == "free_bound"
                    for values in self.conditions.values()
                    for value in values
                )
                else {}
            ),
        }
        self._transaction_snapshot: tuple[object, ...] | None = None
        self._load()

    @staticmethod
    def _empty_metric_state() -> list[object]:
        return [
            0,
            0.0,
            0.0,
            0,
            0,
            0,
            0.0,
            0,
            0,
            0,
            True,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            {},
        ]

    @property
    def _state_path(self) -> Path | None:
        return self.checkpoint_dir / self._STATE_NAME if self.checkpoint_dir else None

    @staticmethod
    def _key(key: tuple[int, str]) -> str:
        return f"{key[0]}|{key[1]}"

    @staticmethod
    def _layer_key(key: tuple[int, int, str]) -> str:
        return f"{key[0]}|{key[1]}|{key[2]}"

    def _load(self) -> None:
        path = self._state_path
        if path is None or not path.exists():
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("run_signature") != self.run_signature:
            raise ValueError(
                "reconstruction local checkpoint signature does not match inputs"
            )
        if state.get("shape") != [self.num_layers, self.num_experts] or state.get(
            "conditions"
        ) != [list(key) for key in self._condition_keys]:
            raise ValueError(
                "reconstruction local checkpoint shape does not match configuration"
            )
        self.completed_batches = {int(value) for value in state["completed_batches"]}
        self.failures = {
            int(key): str(value) for key, value in state["failures"].items()
        }
        self.source_hashes = {
            str(key): str(value)
            for key, value in state.get("source_hashes", {}).items()
        }
        for key in self._condition_keys:
            self._metrics[key] = list(state["metrics"][self._key(key)])
            self._solver[key] = dict(state["solver"][self._key(key)])
        for key in self._layer_metrics:
            self._layer_metrics[key] = list(
                state["layer_metrics"][self._layer_key(key)]
            )
            self._layer_solver[key] = dict(
                state.get("layer_solver", {}).get(self._layer_key(key), {})
            )
            self._selection[key] = list(state["selection"][self._layer_key(key)])
            self._co_selection[key] = [
                list(row) for row in state["co_selection"][self._layer_key(key)]
            ]
        self._references = {
            tuple(key.split("|", 2)): list(value)
            for key, value in state.get("references", {}).items()
        }
        self._references = {
            (int(layer), left, right): value
            for (layer, left, right), value in self._references.items()
        }
        self._reference_solver = {
            (int(layer), name): dict(value)
            for (layer, name), value in (
                (key.split("|", 1), value)
                for key, value in state.get("reference_solver", {}).items()
            )
        }
        for key in self._subset_stability:
            self._subset_stability[key] = list(
                state.get("subset_stability", {})["|".join(map(str, key))]
            )
        persisted_svd = state.get("svd_verification", {})
        if not isinstance(persisted_svd, Mapping):
            raise ValueError("reconstruction local SVD checkpoint is malformed")
        for identity in self._svd_verification:
            value = persisted_svd.get(identity)
            if not isinstance(value, Mapping):
                raise ValueError("reconstruction local SVD checkpoint is incomplete")
            by_layer = value.get("sample_count_by_layer")
            if not isinstance(by_layer, Mapping):
                raise ValueError("reconstruction local SVD layer coverage is malformed")
            parsed_by_layer = {
                int(layer): int(count) for layer, count in by_layer.items()
            }
            sample_count = int(value.get("sample_count", -1))
            maximum = float(
                value.get("max_relative_objective_error", float("nan"))
            )
            if (
                value.get("identity") != identity
                or set(parsed_by_layer) != set(range(self.num_layers))
                or any(not 0 <= count <= 16 for count in parsed_by_layer.values())
                or sample_count != sum(parsed_by_layer.values())
                or value.get("rank_equal") is not True
                or not isfinite(maximum)
                or not 0.0 <= maximum <= self.solver_settings.objective_tolerance
            ):
                raise ValueError("reconstruction local SVD checkpoint is invalid")
            self._svd_verification[identity] = {
                "identity": str(value.get("identity")),
                "sample_count": sample_count,
                "sample_count_by_layer": parsed_by_layer,
                "rank_equal": bool(value.get("rank_equal")),
                "max_relative_objective_error": maximum,
            }

    def _persist(self) -> None:
        path = self._state_path
        if path is None:
            return
        atomic_write_json(
            path,
            {
                "run_signature": self.run_signature,
                "shape": [self.num_layers, self.num_experts],
                "conditions": [list(key) for key in self._condition_keys],
                "source_hashes": self.source_hashes,
                "completed_batches": sorted(self.completed_batches),
                "failures": {
                    str(key): self.failures[key] for key in sorted(self.failures)
                },
                "metrics": {
                    self._key(key): value for key, value in self._metrics.items()
                },
                "layer_metrics": {
                    self._layer_key(key): value
                    for key, value in self._layer_metrics.items()
                },
                "selection": {
                    self._layer_key(key): value
                    for key, value in self._selection.items()
                },
                "co_selection": {
                    self._layer_key(key): value
                    for key, value in self._co_selection.items()
                },
                "solver": {
                    self._key(key): value for key, value in self._solver.items()
                },
                "layer_solver": {
                    self._layer_key(key): value
                    for key, value in self._layer_solver.items()
                },
                "references": {
                    "|".join(map(str, key)): value
                    for key, value in self._references.items()
                },
                "reference_solver": {
                    "|".join(map(str, key)): value
                    for key, value in self._reference_solver.items()
                },
                "subset_stability": {
                    "|".join(map(str, key)): value
                    for key, value in self._subset_stability.items()
                },
                "svd_verification": self._svd_verification,
            },
        )

    def snapshot(self) -> tuple[object, ...]:
        return deepcopy(
            (
                self._metrics,
                self._layer_metrics,
                self._selection,
                self._co_selection,
                self._solver,
                self._layer_solver,
                self._references,
                self._reference_solver,
                self._subset_stability,
                self._pending_distributions,
                self._svd_verification,
                self.completed_batches,
                self.failures,
            )
        )

    def restore(self, snapshot: tuple[object, ...]) -> None:
        (
            self._metrics,
            self._layer_metrics,
            self._selection,
            self._co_selection,
            self._solver,
            self._layer_solver,
            self._references,
            self._reference_solver,
            self._subset_stability,
            self._pending_distributions,
            self._svd_verification,
            self.completed_batches,
            self.failures,
        ) = snapshot
        self._transaction_snapshot = None

    @staticmethod
    def _slice_result(result: ReconstructionResult, count: int) -> ReconstructionResult:
        token_count = result.subsets.shape[0]
        diagnostics = {
            name: value[:count]
            if isinstance(value, Tensor)
            and value.ndim > 0
            and value.shape[0] == token_count
            else value
            for name, value in result.diagnostics.items()
        }
        return replace(
            result,
            subsets=result.subsets[:count],
            coefficients=result.coefficients[:count],
            squared_error=result.squared_error[:count],
            effective_rank=result.effective_rank[:count]
            if result.effective_rank is not None
            else None,
            solver_source=result.solver_source[:count],
            diagnostics=diagnostics,
        )

    def _verify_pseudoinverse_sample(
        self,
        identity: str,
        layer: int,
        target: Tensor,
        experts: Tensor,
        result: ReconstructionResult,
    ) -> None:
        record = self._svd_verification[identity]
        by_layer = record["sample_count_by_layer"]
        assert isinstance(by_layer, dict)
        checked = int(by_layer[layer])
        count = min(16 - checked, target.shape[0])
        if count <= 0:
            return
        sampled_result = self._slice_result(result, count)
        verification = verify_pseudoinverse_svd(
            target[:count], experts[:count], sampled_result, self.solver_settings
        )
        rank_equal = verification.get("rank_equal") is True
        maximum = verification.get("max_relative_objective_error")
        if (
            not rank_equal
            or not isinstance(maximum, (int, float))
            or isinstance(maximum, bool)
            or not isfinite(float(maximum))
            or float(maximum) > self.solver_settings.objective_tolerance
        ):
            raise FloatingPointError(
                f"{identity} pseudoinverse and direct SVD verification disagree"
            )
        record["sample_count"] = int(record["sample_count"]) + count
        by_layer[layer] = checked + count
        record["rank_equal"] = bool(record["rank_equal"]) and rank_equal
        record["max_relative_objective_error"] = max(
            float(record["max_relative_objective_error"]), float(maximum)
        )

    def _winning_subset_diagnostics(
        self, expert_outputs: Tensor, result: ReconstructionResult
    ) -> tuple[Tensor, Tensor, Tensor]:
        rows = torch.arange(expert_outputs.shape[0], device=expert_outputs.device)
        selected = expert_outputs.double()[rows[:, None], result.subsets]
        gram = torch.einsum("tkh,tlh->tkl", selected, selected)
        eigenvalues = torch.linalg.eigvalsh(gram)
        maximum = eigenvalues[:, -1].clamp_min(0.0)
        keep = (maximum > 0.0)[:, None] & (
            eigenvalues > self.solver_settings.tau_g * maximum[:, None]
        )
        ranks = keep.sum(dim=-1)
        minimum = torch.where(
            keep, eigenvalues, torch.full_like(eigenvalues, float("inf"))
        ).min(dim=-1).values
        defined = ranks > 0
        condition = maximum[defined] / minimum[defined]
        if not bool(torch.isfinite(condition).all()):
            raise FloatingPointError("winning-subset Gram conditioning is non-finite")
        if result.effective_rank is not None and not torch.equal(
            ranks, result.effective_rank.to(device=ranks.device, dtype=ranks.dtype)
        ):
            raise FloatingPointError("winning-subset effective rank is inconsistent")
        return ranks, condition, defined

    @staticmethod
    def _distribution_bucket(
        pending: dict[int, dict[str, list[float]]], layer: int
    ) -> dict[str, list[float]]:
        return pending.setdefault(
            layer,
            {
                "coefficients": [],
                "coefficient_abs": [],
                "relative_error": [],
                "cosine": [],
                "gram_condition": [],
            },
        )

    def _add_metric(
        self,
        values: list[object],
        layer: int,
        d0: Tensor,
        expert_outputs: Tensor,
        result: ReconstructionResult,
        pending: dict[int, dict[str, list[float]]],
        *,
        record_distribution: bool = True,
    ) -> None:
        coefficients = result.coefficients.detach().double().cpu()
        squared_error = float(result.squared_error.detach().double().sum().item())
        values[0] += int(result.squared_error.numel())
        values[1] += squared_error
        target64 = d0.detach().double()
        target_norm_by_token = target64.square().sum(dim=-1)
        values[2] += float(target_norm_by_token.sum().item())
        values[3] += int(coefficients.numel())
        values[4] += int((coefficients < 0).sum().item())
        values[5] += int((coefficients.abs() > 2).sum().item())
        values[6] += float((coefficients - 1).abs().sum().item())
        values[11] += float(coefficients.sum().item())
        values[12] += float(coefficients.square().sum().item())
        values[13] += int((coefficients.abs() > 4).sum().item())
        values[14] += float((coefficients - 1).square().sum().item())
        diagnostics = result.diagnostics
        values[7] += int(diagnostics.get("bounded_failure_count", 0))
        bounded = "bounded_lower_saturation_count" in diagnostics
        lower = int((coefficients < 1e-6).sum().item()) if bounded else 0
        upper = int((coefficients > 2.0 - 1e-6).sum().item()) if bounded else 0
        if bounded and int(diagnostics["bounded_lower_saturation_count"]) != lower:
            raise FloatingPointError("bounded lower-saturation diagnostic mismatch")
        if bounded and int(diagnostics["bounded_upper_saturation_count"]) != upper:
            raise FloatingPointError("bounded upper-saturation diagnostic mismatch")
        values[8] += lower
        values[9] += upper
        values[10] = (
            bool(values[10])
            and bool(torch.isfinite(result.squared_error).all())
            and bool(torch.isfinite(result.coefficients).all())
        )
        rows = torch.arange(d0.shape[0], device=d0.device)
        selected = expert_outputs.double()[rows[:, None], result.subsets]
        reconstruction = torch.einsum(
            "tk,tkh->th", result.coefficients.double(), selected
        )
        materialized_error = (reconstruction - target64).square().sum(dim=-1)
        tolerance = self.solver_settings.objective_tolerance * target_norm_by_token.clamp_min(1.0)
        if bool((materialized_error - result.squared_error.double()).abs().gt(tolerance).any()):
            raise FloatingPointError("stored reconstruction loss is not materialized loss")
        relative_error = torch.sqrt(
            materialized_error / target_norm_by_token.clamp_min(torch.finfo(torch.float64).eps)
        )
        reconstruction_norm = reconstruction.square().sum(dim=-1)
        cosine_defined = (target_norm_by_token > 0.0) & (reconstruction_norm > 0.0)
        cosine = (
            (target64[cosine_defined] * reconstruction[cosine_defined]).sum(dim=-1)
            / torch.sqrt(
                target_norm_by_token[cosine_defined]
                * reconstruction_norm[cosine_defined]
            )
        )
        ranks, gram_condition, gram_defined = self._winning_subset_diagnostics(
            expert_outputs, result
        )
        if not bool(torch.isfinite(relative_error).all()) or not bool(
            torch.isfinite(cosine).all()
        ):
            raise FloatingPointError("per-token reconstruction metrics are non-finite")
        values[15] += float(relative_error.sum().item())
        values[16] += float(cosine.sum().item())
        values[17] += int(cosine.numel())
        values[18] += int((~cosine_defined).sum().item())
        values[19] += int(gram_condition.numel())
        values[20] += int((~gram_defined).sum().item())
        rank_counts = values[21]
        assert isinstance(rank_counts, dict)
        for rank in ranks.detach().cpu().tolist():
            label = str(int(rank))
            rank_counts[label] = int(rank_counts.get(label, 0)) + 1
        if record_distribution:
            bucket = self._distribution_bucket(pending, layer)
            bucket["coefficients"].extend(coefficients.reshape(-1).tolist())
            bucket["coefficient_abs"].extend(coefficients.abs().reshape(-1).tolist())
            bucket["relative_error"].extend(relative_error.detach().cpu().tolist())
            bucket["cosine"].extend(cosine.detach().cpu().tolist())
            bucket["gram_condition"].extend(gram_condition.detach().cpu().tolist())

    def _add_reference(
        self, layer: int, left_name: str, left: Tensor, right_name: str, right: Tensor
    ) -> None:
        key = (layer, left_name, right_name)
        values = self._references.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
        left64, right64 = left.detach().double(), right.detach().double()
        values[0] += float((left64 - right64).square().sum().item())
        values[1] += float(left64.square().sum().item())
        values[2] += float(right64.square().sum().item())
        values[3] += float((left64 * right64).sum().item())
        values[4] += int(left.shape[0])

    @staticmethod
    def _accumulate_tensor_diagnostic(
        diagnostics: dict[str, object], name: str, value: Tensor
    ) -> None:
        if name == "pseudoinverse_effective_rank":
            counts = diagnostics.setdefault("pseudoinverse_effective_rank_counts", {})
            assert isinstance(counts, dict)
            ranks = value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
            for rank in ranks.tolist():
                label = str(int(rank))
                counts[label] = int(counts.get(label, 0)) + 1
            return
        summaries = diagnostics.setdefault("tensor_diagnostics", {})
        assert isinstance(summaries, dict)
        summary = summaries.setdefault(
            name,
            {
                "element_count": 0,
                "finite_count": 0,
                "sum": 0.0,
                "min": None,
                "max": None,
            },
        )
        assert isinstance(summary, dict)
        values = value.detach().double().reshape(-1)
        finite = values[torch.isfinite(values)]
        if finite.numel() != values.numel():
            raise FloatingPointError(f"solver diagnostic {name} is non-finite")
        summary["element_count"] = int(summary["element_count"]) + int(values.numel())
        summary["finite_count"] = int(summary["finite_count"]) + int(finite.numel())
        if finite.numel():
            minimum = float(finite.min().item())
            maximum = float(finite.max().item())
            summary["sum"] = float(summary["sum"]) + float(finite.sum().item())
            summary["min"] = (
                minimum
                if summary["min"] is None
                else min(float(summary["min"]), minimum)
            )
            summary["max"] = (
                maximum
                if summary["max"] is None
                else max(float(summary["max"]), maximum)
            )

    def _accumulate_result_diagnostics(
        self, diagnostics: dict[str, object], result: object
    ) -> None:
        if result.effective_rank is not None:
            self._accumulate_tensor_diagnostic(
                diagnostics, "effective_rank", result.effective_rank
            )
        source_counts = diagnostics.setdefault("solver_source_counts", {})
        assert isinstance(source_counts, dict)
        for source in result.solver_source:
            source_counts[source] = int(source_counts.get(source, 0)) + 1
        for name, value in result.diagnostics.items():
            if isinstance(value, Tensor):
                self._accumulate_tensor_diagnostic(diagnostics, name, value)
            elif isinstance(value, (tuple, list)):
                sequences = diagnostics.setdefault("sequence_diagnostics", {})
                assert isinstance(sequences, dict)
                serialized = list(value)
                previous = sequences.setdefault(name, serialized)
                if previous != serialized:
                    raise RuntimeError(
                        f"solver diagnostic sequence {name} changed within a run"
                    )
            elif isinstance(value, (int, float)):
                if not isfinite(float(value)):
                    raise FloatingPointError(
                        f"solver diagnostic {name} is non-finite"
                    )
                if name.startswith("max_") or "_max_" in name:
                    diagnostics[name] = max(
                        float(diagnostics.get(name, value)), float(value)
                    )
                elif name == "tau_g":
                    previous = diagnostics.setdefault(name, value)
                    if previous != value:
                        raise RuntimeError(
                            f"solver diagnostic {name} changed within a run"
                        )
                elif name.endswith("_count") and isinstance(value, int):
                    diagnostics[name] = int(diagnostics.get(name, 0)) + value
                else:
                    diagnostics[name] = float(diagnostics.get(name, 0)) + float(value)

    def _accumulate_solver_diagnostics(
        self, key: tuple[int, str], result: object
    ) -> None:
        self._accumulate_result_diagnostics(self._solver[key], result)

    def update_layer(self, layer: int, d0: Tensor, expert_outputs: Tensor) -> None:
        if not 0 <= layer < self.num_layers or d0.ndim != 2 or expert_outputs.ndim != 3:
            raise ValueError("invalid reconstruction local layer tensors")
        if expert_outputs.shape != (d0.shape[0], self.num_experts, d0.shape[1]):
            raise ValueError("D0 and expert outputs have incompatible shapes")
        if not bool(torch.isfinite(d0).all()) or not bool(
            torch.isfinite(expert_outputs).all()
        ):
            raise ValueError("reconstruction local inputs must be finite")
        if d0.shape[0] == 0:
            return
        if self._transaction_snapshot is None:
            self._transaction_snapshot = self.snapshot()
        e8 = expert_outputs.sum(dim=1)
        self._add_reference(layer, "D0", d0, "E8", e8)
        unique_conditions = {
            condition_id(condition): condition
            for candidates in self.conditions.values()
            for condition in candidates
        }
        for identifier, condition in unique_conditions.items():
            r8_result = score_full_reference(
                d0.double(), expert_outputs.double(), condition, self.solver_settings
            )
            if condition.family == "free_bound":
                self._verify_pseudoinverse_sample(
                    "R8[free_bound]",
                    layer,
                    d0.double(),
                    expert_outputs.double(),
                    r8_result,
                )
            selected_experts = expert_outputs.double()[
                torch.arange(d0.shape[0], device=d0.device)[:, None],
                r8_result.subsets,
            ].to(r8_result.coefficients.dtype)
            reconstruction = torch.einsum(
                "tk,tkh->th", r8_result.coefficients, selected_experts
            )
            r8_name = f"R8[{identifier}]"
            self._add_reference(layer, "D0", d0, r8_name, reconstruction)
            self._add_reference(layer, "E8", e8, r8_name, reconstruction)
            self._accumulate_result_diagnostics(
                self._reference_solver.setdefault((layer, r8_name), {}), r8_result
            )
        for k, candidates in self.conditions.items():
            subsets_by_condition: dict[str, Tensor] = {}
            for condition in candidates:
                key = (k, condition_id(condition))
                result = score_reconstruction(
                    d0, expert_outputs, k, condition, self.solver_settings
                )
                if condition.family == "free_bound":
                    self._verify_pseudoinverse_sample(
                        f"sparse:K{k}:free_bound",
                        layer,
                        d0,
                        expert_outputs,
                        result,
                    )
                metric_key = (layer, *key)
                self._add_metric(
                    self._metrics[key],
                    layer,
                    d0,
                    expert_outputs,
                    result,
                    self._pending_distributions[key],
                )
                self._add_metric(
                    self._layer_metrics[metric_key],
                    layer,
                    d0,
                    expert_outputs,
                    result,
                    self._pending_distributions[key],
                    record_distribution=False,
                )
                selection = (
                    torch.bincount(
                        result.subsets.reshape(-1), minlength=self.num_experts
                    )
                    .cpu()
                    .tolist()
                )
                for expert, count in enumerate(selection):
                    self._selection[metric_key][expert] += int(count)
                mask = torch.zeros(
                    (d0.shape[0], self.num_experts),
                    dtype=torch.bool,
                    device=result.subsets.device,
                )
                mask.scatter_(1, result.subsets, True)
                co_selection = (
                    (mask[:, :, None] & mask[:, None, :]).sum(dim=0).cpu().tolist()
                )
                for first in range(self.num_experts):
                    for second in range(self.num_experts):
                        self._co_selection[metric_key][first][second] += int(
                            co_selection[first][second]
                        )
                self._accumulate_solver_diagnostics(key, result)
                self._accumulate_result_diagnostics(
                    self._layer_solver[metric_key], result
                )
                subsets_by_condition[condition_id(condition)] = result.subsets.detach()
            identifiers = sorted(subsets_by_condition)
            for index, left_name in enumerate(identifiers):
                for right_name in identifiers[index + 1 :]:
                    left, right = subsets_by_condition[left_name], subsets_by_condition[right_name]
                    intersection = (left[:, :, None] == right[:, None, :]).any(dim=2).sum(dim=1)
                    union = 2 * k - intersection
                    values = self._subset_stability[(k, left_name, right_name)]
                    values[0] += float((intersection.to(torch.float64) / union).sum().item())
                    values[1] += int(left.shape[0])

    def _chunk_path(self, key: tuple[int, str], batch_index: int) -> Path:
        assert self.checkpoint_dir is not None
        safe = (
            condition_id(ReconstructionCondition("unit"))
            if key[1] == "unit"
            else key[1].replace(":", "_")
        )
        return (
            self.checkpoint_dir
            / "coefficients"
            / f"k{key[0]}-{safe}-batch-{batch_index:09d}.json"
        )

    def mark_batch_completed(self, batch_index: int) -> None:
        if batch_index in self.completed_batches or batch_index in self.failures:
            raise ValueError("batch is already terminal")
        transaction = self._transaction_snapshot or self.snapshot()
        chunk_paths = (
            [self._chunk_path(key, batch_index) for key in self._condition_keys]
            if self.checkpoint_dir
            else []
        )
        # Chunks are atomically written first.  An interrupted checkpoint leaves only
        # ignored orphan chunks because the completed-batch state remains unchanged.
        try:
            if self.checkpoint_dir:
                for key, path in zip(self._condition_keys, chunk_paths, strict=True):
                    atomic_write_json(
                        path,
                        {
                            "layers": {
                                str(layer): {
                                    name: sorted(values)
                                    for name, values in distributions.items()
                                }
                                for layer, distributions in sorted(
                                    self._pending_distributions[key].items()
                                )
                            }
                        },
                    )
            self._pending_distributions = {key: {} for key in self._condition_keys}
            self.completed_batches.add(batch_index)
            self._persist()
        except Exception:
            self.restore(transaction)
            for path in chunk_paths:
                path.unlink(missing_ok=True)
            raise
        self._transaction_snapshot = None

    def record_failure(self, batch_index: int, message: str) -> None:
        if batch_index in self.completed_batches:
            raise ValueError("completed batch cannot become failed")
        snapshot = self.snapshot()
        try:
            self.failures[batch_index] = message
            self._persist()
        except Exception:
            self.restore(snapshot)
            raise

    def _distribution_values(
        self, key: tuple[int, str], name: str, layer: int | None = None
    ) -> list[float]:
        if not self.checkpoint_dir or not self.completed_batches:
            return []
        values: list[float] = []
        for batch in sorted(self.completed_batches):
            path = self._chunk_path(key, batch)
            if not path.exists():
                raise RuntimeError("completed batch is missing its coefficient chunk")
            payload = json.loads(path.read_text(encoding="utf-8"))
            layers = payload.get("layers")
            if not isinstance(layers, Mapping):
                raise RuntimeError("completed batch has malformed distribution chunk")
            selected_layers = (
                (str(layer), layers.get(str(layer)))
                if layer is not None
                else None
            )
            items = (selected_layers,) if selected_layers is not None else layers.items()
            for _, distributions in items:
                if distributions is None:
                    continue
                if not isinstance(distributions, Mapping) or not isinstance(
                    distributions.get(name), list
                ):
                    raise RuntimeError("completed batch has malformed distribution chunk")
                values.extend(float(value) for value in distributions[name])
        values.sort()
        return values

    @staticmethod
    def _nearest(sorted_values: list[float], percentile: float) -> float | None:
        if not sorted_values:
            return None
        return sorted_values[max(0, ceil(percentile * len(sorted_values)) - 1)]

    def _quantiles(self, key: tuple[int, str]) -> Mapping[str, float]:
        values = self._distribution_values(key, "coefficients")
        absolute = self._distribution_values(key, "coefficient_abs")
        if not values or not absolute:
            return {}
        return {
            "p01": float(self._nearest(values, 0.01)),
            "p05": float(self._nearest(values, 0.05)),
            "median": float(self._nearest(values, 0.5)),
            "p95": float(self._nearest(values, 0.95)),
            "p99": float(self._nearest(absolute, 0.99)),
            "p99_signed": float(self._nearest(values, 0.99)),
            "max_abs": absolute[-1],
        }

    def _public_metric(
        self,
        values: list[object],
        key: tuple[int, str],
        *,
        layer: int | None = None,
    ) -> ConditionMetrics:
        coefficients = self._distribution_values(key, "coefficients", layer)
        coefficient_abs = self._distribution_values(key, "coefficient_abs", layer)
        relative_error = self._distribution_values(key, "relative_error", layer)
        cosine = self._distribution_values(key, "cosine", layer)
        gram_condition = self._distribution_values(key, "gram_condition", layer)
        return ConditionMetrics(
            int(values[0]),
            float(values[1]),
            float(values[2]),
            int(values[3]),
            int(values[4]),
            int(values[5]),
            float(values[6]),
            int(values[7]),
            int(values[8]),
            int(values[9]),
            bool(values[10]),
            coefficient_sum=float(values[11]),
            coefficient_squared_sum=float(values[12]),
            above_four_count=int(values[13]),
            squared_deviation_sum=float(values[14]),
            relative_error_sum=float(values[15]),
            cosine_sum=float(values[16]),
            cosine_defined_count=int(values[17]),
            cosine_undefined_count=int(values[18]),
            gram_condition_defined_count=int(values[19]),
            gram_condition_undefined_count=int(values[20]),
            coefficient_p01=self._nearest(coefficients, 0.01),
            coefficient_p05=self._nearest(coefficients, 0.05),
            coefficient_median=self._nearest(coefficients, 0.5),
            coefficient_p95=self._nearest(coefficients, 0.95),
            coefficient_p99=self._nearest(coefficients, 0.99),
            coefficient_p99_abs=self._nearest(coefficient_abs, 0.99),
            coefficient_max_abs=coefficient_abs[-1] if coefficient_abs else None,
            relative_error_median=self._nearest(relative_error, 0.5),
            relative_error_p90=self._nearest(relative_error, 0.9),
            relative_error_p95=self._nearest(relative_error, 0.95),
            relative_error_p99=self._nearest(relative_error, 0.99),
            cosine_median=self._nearest(cosine, 0.5),
            cosine_p90=self._nearest(cosine, 0.9),
            cosine_p95=self._nearest(cosine, 0.95),
            cosine_p99=self._nearest(cosine, 0.99),
            gram_condition_median=self._nearest(gram_condition, 0.5),
            gram_condition_p90=self._nearest(gram_condition, 0.9),
            gram_condition_p99=self._nearest(gram_condition, 0.99),
            gram_condition_max=gram_condition[-1] if gram_condition else None,
            effective_rank_counts={
                str(rank): int(count)
                for rank, count in dict(values[21]).items()
            },
        )

    def finalize(self) -> ReconstructionLocalResult:
        return ReconstructionLocalResult(
            metrics={
                key: self._public_metric(value, key)
                for key, value in self._metrics.items()
            },
            layer_metrics={
                key: self._public_metric(value, (key[1], key[2]), layer=key[0])
                for key, value in self._layer_metrics.items()
            },
            selection_counts={
                key: tuple(value) for key, value in self._selection.items()
            },
            co_selection={
                key: tuple(tuple(row) for row in value)
                for key, value in self._co_selection.items()
            },
            solver_diagnostics={
                key: dict(value) for key, value in self._solver.items()
            },
            reference_distances={
                key: ReferenceDistance(
                    float(value[0]),
                    float(value[1]),
                    float(value[2]),
                    float(value[3]),
                    int(value[4]),
                )
                for key, value in self._references.items()
            },
            reference_diagnostics={
                key: dict(value) for key, value in self._reference_solver.items()
            },
            coefficient_quantiles={
                key: self._quantiles(key) for key in self._condition_keys
            },
            completed_batches=len(self.completed_batches),
            failed_batches=len(self.failures),
            complete=bool(self.completed_batches) and not self.failures,
            run_signature=self.run_signature,
            source_hashes=dict(self.source_hashes),
            subset_stability={
                key: SubsetJaccard(float(value[0]), int(value[1]))
                for key, value in self._subset_stability.items()
            },
            layer_solver_diagnostics={
                key: dict(value) for key, value in self._layer_solver.items()
            },
            svd_verification={
                identity: SVDVerificationEvidence(
                    identity=str(value["identity"]),
                    sample_count=int(value["sample_count"]),
                    sample_count_by_layer={
                        int(layer): int(count)
                        for layer, count in dict(
                            value["sample_count_by_layer"]
                        ).items()
                    },
                    rank_equal=bool(value["rank_equal"]),
                    max_relative_objective_error=float(
                        value["max_relative_objective_error"]
                    ),
                )
                for identity, value in self._svd_verification.items()
            },
        )


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def run_reconstruction_local(
    teacher: nn.Module,
    student: nn.Module,
    batches: Iterable[EvaluationBatch],
    conditions: Mapping[int, Sequence[ReconstructionCondition]],
    checkpoint_dir: Path,
    run_signature: str,
    *,
    source_hashes: Mapping[str, str] | None = None,
) -> ReconstructionLocalResult:
    """Score reconstruction conditions against teacher D0 with batch transactions."""
    teacher_mlps, student_mlps = (
        dict(iter_text_mlps(teacher)),
        dict(iter_text_mlps(student)),
    )
    if teacher_mlps.keys() != student_mlps.keys() or not teacher_mlps:
        raise ValueError("teacher and student text-layer layouts differ")
    if not all(isinstance(mlp, PartitionedQwen35MLP) for mlp in student_mlps.values()):
        raise TypeError("student text MLPs must be PartitionedQwen35MLP modules")
    accumulator = ReconstructionLocalAccumulator(
        num_layers=len(teacher_mlps),
        num_experts=next(iter(student_mlps.values())).num_experts,
        conditions=conditions,
        checkpoint_dir=checkpoint_dir,
        run_signature=run_signature,
        source_hashes=source_hashes,
    )
    teacher_device, student_device = _model_device(teacher), _model_device(student)
    for batch in batches:
        if batch.index in accumulator.completed_batches:
            continue
        captured: dict[int, tuple[Tensor, Tensor]] = {}
        handles: list[object] = []
        snapshot = accumulator.snapshot()
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
                        raise TypeError("text MLP hook requires tensors")
                    captured[layer_index] = (args[0].detach(), output.detach())

                handles.append(mlp.register_forward_hook(capture))
            with torch.inference_mode():
                teacher.model.language_model(
                    input_ids=batch.input_ids.to(teacher_device),
                    attention_mask=batch.attention_mask.to(teacher_device),
                    use_cache=False,
                )
                if captured.keys() != teacher_mlps.keys():
                    raise RuntimeError("teacher forward did not visit every text MLP")
                for layer, (teacher_input, d0) in captured.items():
                    experts = (
                        student_mlps[layer]
                        .expert_outputs(teacher_input.to(student_device))
                        .to(d0.device)
                    )
                    valid = batch.attention_mask.to(
                        d0.device, dtype=torch.bool
                    ).reshape(-1)
                    accumulator.update_layer(
                        layer,
                        d0.reshape(-1, d0.shape[-1])[valid],
                        experts.reshape(-1, 8, experts.shape[-1])[valid],
                    )
            accumulator.mark_batch_completed(batch.index)
        except OSError:
            accumulator.restore(snapshot)
            raise
        except Exception as error:
            accumulator.restore(snapshot)
            accumulator.record_failure(batch.index, f"{type(error).__name__}: {error}")
            break
        finally:
            for handle in handles:
                handle.remove()
    return accumulator.finalize()
