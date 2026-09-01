"""Checkpointed calibration-only aggregation for reconstruction conditions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from math import ceil, isfinite
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from ppmoe.adapters.qwen35_runtime import iter_text_mlps
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.moefication.qwen35 import PartitionedQwen35MLP
from ppmoe.oracle.reconstruction import (
    ReconstructionCondition,
    SolverSettings,
    score_reconstruction,
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

    @property
    def aggregate_nmse(self) -> float:
        if self.total_d0_squared_norm == 0.0:
            return 0.0 if self.total_squared_error == 0.0 else float("inf")
        return self.total_squared_error / self.total_d0_squared_norm

    @property
    def negative_rate(self) -> float:
        return (
            self.negative_count / self.coefficient_count
            if self.coefficient_count
            else 0.0
        )

    @property
    def above_two_rate(self) -> float:
        return (
            self.above_two_count / self.coefficient_count
            if self.coefficient_count
            else 0.0
        )

    @property
    def mean_absolute_deviation_from_one(self) -> float:
        return (
            self.absolute_deviation_sum / self.coefficient_count
            if self.coefficient_count
            else 0.0
        )


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
class ReconstructionLocalResult:
    metrics: Mapping[tuple[int, str], ConditionMetrics]
    layer_metrics: Mapping[tuple[int, int, str], ConditionMetrics]
    selection_counts: Mapping[tuple[int, int, str], tuple[int, ...]]
    co_selection: Mapping[tuple[int, int, str], tuple[tuple[int, ...], ...]]
    solver_diagnostics: Mapping[tuple[int, str], Mapping[str, float | int]]
    reference_distances: Mapping[tuple[int, str, str], ReferenceDistance]
    coefficient_quantiles: Mapping[tuple[int, str], Mapping[str, float]]
    completed_batches: int
    failed_batches: int
    complete: bool
    run_signature: str
    source_hashes: Mapping[str, str]

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
            key: [0, 0.0, 0.0, 0, 0, 0, 0.0, 0, 0, 0, True]
            for key in self._condition_keys
        }
        self._layer_metrics = {
            (layer, *key): [0, 0.0, 0.0, 0, 0, 0, 0.0, 0, 0, 0, True]
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
        self._solver = {key: {} for key in self._condition_keys}
        self._references: dict[tuple[int, str, str], list[float | int]] = {}
        self._pending_coefficients: dict[tuple[int, str], list[float]] = {
            key: [] for key in self._condition_keys
        }
        self._load()

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
                "references": {
                    "|".join(map(str, key)): value
                    for key, value in self._references.items()
                },
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
                self._references,
                self._pending_coefficients,
            )
        )

    def restore(self, snapshot: tuple[object, ...]) -> None:
        (
            self._metrics,
            self._layer_metrics,
            self._selection,
            self._co_selection,
            self._solver,
            self._references,
            self._pending_coefficients,
        ) = snapshot

    @staticmethod
    def _add_metric(
        values: list[float | int | bool], target_norm: float, result: object
    ) -> None:
        coefficients = result.coefficients.detach().double().cpu()
        squared_error = float(result.squared_error.detach().double().sum().item())
        values[0] += int(result.squared_error.numel())
        values[1] += squared_error
        values[2] += target_norm
        values[3] += int(coefficients.numel())
        values[4] += int((coefficients < 0).sum().item())
        values[5] += int((coefficients.abs() > 2).sum().item())
        values[6] += float((coefficients - 1).abs().sum().item())
        diagnostics = result.diagnostics
        values[7] += int(diagnostics.get("bounded_failure_count", 0))
        values[8] += int(diagnostics.get("bounded_lower_saturation_count", 0))
        values[9] += int(diagnostics.get("bounded_upper_saturation_count", 0))
        values[10] = (
            bool(values[10])
            and bool(torch.isfinite(result.squared_error).all())
            and bool(torch.isfinite(result.coefficients).all())
        )

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
        target_norm = float(d0.detach().double().square().sum().item())
        e8 = expert_outputs.sum(dim=1)
        self._add_reference(layer, "D0", d0, "E8", e8)
        for k, candidates in self.conditions.items():
            for condition in candidates:
                key = (k, condition_id(condition))
                result = score_reconstruction(
                    d0, expert_outputs, k, condition, self.solver_settings
                )
                metric_key = (layer, *key)
                self._add_metric(self._metrics[key], target_norm, result)
                self._add_metric(self._layer_metrics[metric_key], target_norm, result)
                self._pending_coefficients[key].extend(
                    sorted(
                        float(value)
                        for value in result.coefficients.detach()
                        .double()
                        .cpu()
                        .reshape(-1)
                        .tolist()
                    )
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
                for name, value in result.diagnostics.items():
                    if isinstance(value, (int, float)) and isfinite(float(value)):
                        self._solver[key][name] = self._solver[key].get(name, 0) + value
                if k == self.num_experts:
                    selected_experts = expert_outputs[
                        torch.arange(d0.shape[0], device=d0.device)[:, None],
                        result.subsets,
                    ].to(result.coefficients.dtype)
                    reconstruction = torch.einsum(
                        "tk,tkh->th", result.coefficients, selected_experts
                    )
                    r8_name = f"R8[{condition_id(condition)}]"
                    self._add_reference(layer, "D0", d0, r8_name, reconstruction)
                    self._add_reference(layer, "E8", e8, r8_name, reconstruction)

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
        # Chunks are atomically written first.  An interrupted checkpoint leaves only
        # ignored orphan chunks because the completed-batch state remains unchanged.
        if self.checkpoint_dir:
            for key in self._condition_keys:
                atomic_write_json(
                    self._chunk_path(key, batch_index),
                    {"coefficients": sorted(self._pending_coefficients[key])},
                )
        self._pending_coefficients = {key: [] for key in self._condition_keys}
        self.completed_batches.add(batch_index)
        self._persist()

    def record_failure(self, batch_index: int, message: str) -> None:
        if batch_index in self.completed_batches:
            raise ValueError("completed batch cannot become failed")
        self.failures[batch_index] = message
        self._persist()

    def _quantiles(self, key: tuple[int, str]) -> Mapping[str, float]:
        if not self.checkpoint_dir or not self.completed_batches:
            return {}
        values: list[float] = []
        for batch in sorted(self.completed_batches):
            path = self._chunk_path(key, batch)
            if not path.exists():
                raise RuntimeError("completed batch is missing its coefficient chunk")
            values.extend(
                float(value)
                for value in json.loads(path.read_text(encoding="utf-8"))[
                    "coefficients"
                ]
            )
        if not values:
            return {}
        values.sort()
        absolute = sorted(abs(value) for value in values)

        def nearest(sorted_values: list[float], percentile: float) -> float:
            return sorted_values[max(0, ceil(percentile * len(sorted_values)) - 1)]

        return {
            "p01": nearest(values, 0.01),
            "p05": nearest(values, 0.05),
            "median": nearest(values, 0.5),
            "p95": nearest(values, 0.95),
            "p99": nearest(absolute, 0.99),
            "max_abs": absolute[-1],
        }

    @staticmethod
    def _public_metric(values: list[float | int | bool]) -> ConditionMetrics:
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
        )

    def finalize(self) -> ReconstructionLocalResult:
        return ReconstructionLocalResult(
            metrics={
                key: self._public_metric(value) for key, value in self._metrics.items()
            },
            layer_metrics={
                key: self._public_metric(value)
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
            coefficient_quantiles={
                key: self._quantiles(key) for key in self._condition_keys
            },
            completed_batches=len(self.completed_batches),
            failed_batches=len(self.failures),
            complete=bool(self.completed_batches) and not self.failures,
            run_signature=self.run_signature,
            source_hashes=dict(self.source_hashes),
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
        except Exception as error:
            accumulator.restore(snapshot)
            accumulator.record_failure(batch.index, f"{type(error).__name__}: {error}")
            break
        finally:
            for handle in handles:
                handle.remove()
    return accumulator.finalize()
