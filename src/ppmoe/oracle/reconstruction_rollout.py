"""Checkpointed Reconstruction-H rollout under frozen coefficient conditions."""

from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack
from dataclasses import dataclass
import json
from math import exp
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from ppmoe.oracle.reconstruction import (
    ReconstructionCondition,
    ReconstructionResult,
    SolverSettings,
    score_full_reference,
    score_reconstruction,
)
from ppmoe.oracle.reconstruction_local import ReferenceDistance
from ppmoe.adapters.qwen35_runtime import iter_text_mlps
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.moefication.qwen35 import PartitionedQwen35MLP
from ppmoe.provenance import atomic_write_json
from torch import Tensor, nn


def _condition_id(condition: ReconstructionCondition) -> str:
    return condition.family if condition.rho is None else f"ridge:{condition.rho:g}"


def _cpu_fp64_sum(values: Tensor) -> float:
    return float(values.detach().to(device="cpu", dtype=torch.float64).sum().item())


@dataclass(frozen=True, slots=True)
class ReconstructionRolloutLayerMetrics:
    token_count: int
    hidden_squared_error_sum: float
    teacher_squared_norm_sum: float

    @property
    def normalized_mse(self) -> float:
        if self.teacher_squared_norm_sum == 0.0:
            return 0.0 if self.hidden_squared_error_sum == 0.0 else float("inf")
        return self.hidden_squared_error_sum / self.teacher_squared_norm_sum


@dataclass(frozen=True, slots=True)
class ReconstructionRolloutResult:
    condition_id: str
    layer_metrics: Mapping[int, ReconstructionRolloutLayerMetrics]
    selection_counts: Mapping[int, tuple[int, ...]]
    solver_diagnostics: Mapping[int, Mapping[str, object]]
    reference_distances: Mapping[int, Mapping[str, ReferenceDistance]]
    coefficient_diagnostics: Mapping[str, float | int]
    nll_sum: float
    scored_tokens: int
    ppl: float | None
    completed_batches: int
    failed_batches: int
    peak_cuda_bytes: int
    complete: bool
    run_signature: str
    source_hashes: Mapping[str, str]


@dataclass(slots=True)
class _BatchDelta:
    layer_stats: list[list[float | int]]
    selection_counts: list[list[int]]
    solver_diagnostics: list[dict[str, object]]
    reference_distances: list[dict[str, list[float | int]]]
    coefficient_diagnostics: dict[str, float | int]
    nll_sum: float = 0.0
    scored_tokens: int = 0


def _new_coefficient_diagnostics() -> dict[str, float | int]:
    return {
        "coefficient_count": 0,
        "negative_count": 0,
        "above_two_count": 0,
        "absolute_deviation_sum": 0.0,
        "bounded_lower_saturation_count": 0,
        "bounded_upper_saturation_count": 0,
    }


class _Accumulator:
    """Persist only complete-batch deltas for one immutable condition/K run."""

    _STATE_NAME = "reconstruction_rollout_state.json"

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        condition: ReconstructionCondition,
        k: int,
        checkpoint_dir: Path,
        run_signature: str,
        source_hashes: Mapping[str, str] | None,
    ) -> None:
        self.num_layers, self.num_experts, self.k = num_layers, num_experts, k
        self.condition_id = _condition_id(condition)
        self.checkpoint_dir, self.run_signature = checkpoint_dir, run_signature
        self.source_hashes = dict(source_hashes or {})
        self.completed_batches: set[int] = set()
        self.failures: dict[int, str] = {}
        self.layer_stats = [[0.0, 0.0, 0] for _ in range(num_layers)]
        self.selection_counts = [[0] * num_experts for _ in range(num_layers)]
        self.solver_diagnostics = [{} for _ in range(num_layers)]
        self.reference_distances = [{} for _ in range(num_layers)]
        self.coefficient_diagnostics = _new_coefficient_diagnostics()
        self.nll_sum, self.scored_tokens, self.peak_cuda_bytes = 0.0, 0, 0
        self._load()

    @property
    def _state_path(self) -> Path:
        return self.checkpoint_dir / self._STATE_NAME

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        state = json.loads(self._state_path.read_text(encoding="utf-8"))
        if state.get("run_signature") != self.run_signature:
            raise ValueError("reconstruction rollout checkpoint signature does not match inputs")
        if state.get("shape") != [self.num_layers, self.num_experts] or state.get("k") != self.k or state.get("condition_id") != self.condition_id:
            raise ValueError("reconstruction rollout checkpoint layout does not match invocation")
        self.completed_batches = {int(value) for value in state["completed_batches"]}
        self.failures = {int(key): str(value) for key, value in state["failures"].items()}
        self.layer_stats = [[float(row[0]), float(row[1]), int(row[2])] for row in state["layer_stats"]]
        self.selection_counts = [[int(value) for value in row] for row in state["selection_counts"]]
        self.solver_diagnostics = [dict(value) for value in state["solver_diagnostics"]]
        self.reference_distances = [
            {str(name): list(values) for name, values in layer.items()}
            for layer in state["reference_distances"]
        ]
        self.coefficient_diagnostics = {
            str(name): value for name, value in state["coefficient_diagnostics"].items()
        }
        self.nll_sum = float(state["nll_sum"])
        self.scored_tokens = int(state["scored_tokens"])
        self.peak_cuda_bytes = int(state["peak_cuda_bytes"])
        self.source_hashes = {str(key): str(value) for key, value in state.get("source_hashes", {}).items()}

    def _persist(self) -> None:
        atomic_write_json(self._state_path, {
            "run_signature": self.run_signature,
            "shape": [self.num_layers, self.num_experts],
            "k": self.k,
            "condition_id": self.condition_id,
            "source_hashes": self.source_hashes,
            "completed_batches": sorted(self.completed_batches),
            "failures": {str(key): self.failures[key] for key in sorted(self.failures)},
            "layer_stats": self.layer_stats,
            "selection_counts": self.selection_counts,
            "solver_diagnostics": self.solver_diagnostics,
            "reference_distances": self.reference_distances,
            "coefficient_diagnostics": self.coefficient_diagnostics,
            "nll_sum": self.nll_sum,
            "scored_tokens": self.scored_tokens,
            "peak_cuda_bytes": self.peak_cuda_bytes,
        })

    def _snapshot(self) -> tuple[object, ...]:
        return deepcopy((self.completed_batches, self.layer_stats, self.selection_counts, self.solver_diagnostics, self.reference_distances, self.coefficient_diagnostics, self.nll_sum, self.scored_tokens, self.peak_cuda_bytes))

    def _restore(self, snapshot: tuple[object, ...]) -> None:
        (
            self.completed_batches,
            self.layer_stats,
            self.selection_counts,
            self.solver_diagnostics,
            self.reference_distances,
            self.coefficient_diagnostics,
            self.nll_sum,
            self.scored_tokens,
            self.peak_cuda_bytes,
        ) = snapshot

    @staticmethod
    def _merge_diagnostics(destination: dict[str, object], source: Mapping[str, object]) -> None:
        for name, value in source.items():
            if isinstance(value, bool):
                destination[name] = int(destination.get(name, 0)) + int(value)
            elif isinstance(value, (int, float)):
                if name.startswith("max_"):
                    destination[name] = max(float(destination.get(name, value)), float(value))
                else:
                    destination[name] = float(destination.get(name, 0.0)) + float(value)

    def merge_completed(self, batch_index: int, delta: _BatchDelta, peak_cuda_bytes: int) -> None:
        if batch_index in self.completed_batches:
            raise ValueError("batch is already completed")
        snapshot = self._snapshot()
        try:
            self.completed_batches.add(batch_index)
            self.nll_sum += delta.nll_sum
            self.scored_tokens += delta.scored_tokens
            self.peak_cuda_bytes = max(self.peak_cuda_bytes, peak_cuda_bytes)
            for layer in range(self.num_layers):
                for index, value in enumerate(delta.layer_stats[layer]):
                    self.layer_stats[layer][index] += value
                for expert, value in enumerate(delta.selection_counts[layer]):
                    self.selection_counts[layer][expert] += value
                self._merge_diagnostics(self.solver_diagnostics[layer], delta.solver_diagnostics[layer])
                for name, values in delta.reference_distances[layer].items():
                    existing = self.reference_distances[layer].setdefault(name, [0.0, 0.0, 0.0, 0.0, 0])
                    for index, value in enumerate(values):
                        existing[index] += value
            for name, value in delta.coefficient_diagnostics.items():
                self.coefficient_diagnostics[name] = self.coefficient_diagnostics.get(name, 0) + value
            self._persist()
        except Exception:
            self._restore(snapshot)
            raise

    def record_failure(self, batch_index: int, message: str, peak_cuda_bytes: int) -> None:
        if batch_index in self.completed_batches:
            raise ValueError("completed batch cannot become failed")
        failures, peak = dict(self.failures), self.peak_cuda_bytes
        try:
            self.failures[batch_index] = message
            self.peak_cuda_bytes = max(self.peak_cuda_bytes, peak_cuda_bytes)
            self._persist()
        except Exception:
            self.failures, self.peak_cuda_bytes = failures, peak
            raise

    def finalize(self) -> ReconstructionRolloutResult:
        layers = {
            layer: ReconstructionRolloutLayerMetrics(int(values[2]), float(values[0]), float(values[1]))
            for layer, values in enumerate(self.layer_stats)
        }
        complete = bool(self.completed_batches) and not self.failures and self.scored_tokens > 0 and all(metric.token_count > 0 for metric in layers.values())
        return ReconstructionRolloutResult(
            condition_id=self.condition_id,
            layer_metrics=layers,
            selection_counts={layer: tuple(values) for layer, values in enumerate(self.selection_counts)},
            solver_diagnostics={layer: dict(values) for layer, values in enumerate(self.solver_diagnostics)},
            reference_distances={
                layer: {name: ReferenceDistance(float(values[0]), float(values[1]), float(values[2]), float(values[3]), int(values[4])) for name, values in distances.items()}
                for layer, distances in enumerate(self.reference_distances)
            },
            coefficient_diagnostics=dict(self.coefficient_diagnostics),
            nll_sum=self.nll_sum,
            scored_tokens=self.scored_tokens,
            ppl=exp(self.nll_sum / self.scored_tokens) if complete else None,
            completed_batches=len(self.completed_batches),
            failed_batches=len(self.failures),
            peak_cuda_bytes=self.peak_cuda_bytes,
            complete=complete,
            run_signature=self.run_signature,
            source_hashes=dict(self.source_hashes),
        )


def select_reconstruction_h(
    target_next: Tensor,
    residual_baseline: Tensor,
    expert_outputs: Tensor,
    condition: ReconstructionCondition,
    k: int,
    settings: SolverSettings,
) -> ReconstructionResult:
    """Solve the frozen condition for ``teacher_next - student_baseline``."""

    if target_next.shape != residual_baseline.shape:
        raise ValueError("target_next and residual_baseline must have the same shape")
    if expert_outputs.shape[:-2] != target_next.shape[:-1] or (
        expert_outputs.shape[-1] != target_next.shape[-1]
    ):
        raise ValueError("expert_outputs must align with target_next token dimensions")
    if (
        target_next.device != residual_baseline.device
        or target_next.device != expert_outputs.device
    ):
        raise ValueError("Reconstruction-H tensors must share a device")
    if target_next.ndim < 2:
        raise ValueError("Reconstruction-H requires token and hidden dimensions")
    correction = (target_next - residual_baseline).detach().reshape(
        -1, target_next.shape[-1]
    )
    experts = expert_outputs.detach().reshape(
        -1, expert_outputs.shape[-2], expert_outputs.shape[-1]
    )
    return score_reconstruction(correction, experts, k, condition, settings)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _peak_cuda_bytes(device: torch.device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def _reference_delta(
    distances: dict[str, list[float | int]],
    left_name: str,
    left: Tensor,
    right_name: str,
    right: Tensor,
) -> None:
    name = f"{left_name}|{right_name}"
    left64, right64 = left.detach().double(), right.detach().double()
    distances[name] = [
        _cpu_fp64_sum((left64 - right64).square().sum(dim=-1)),
        _cpu_fp64_sum(left64.square().sum(dim=-1)),
        _cpu_fp64_sum(right64.square().sum(dim=-1)),
        _cpu_fp64_sum((left64 * right64).sum(dim=-1)),
        int(left.shape[0]),
    ]


def _weighted_output(expert_outputs: Tensor, result: ReconstructionResult) -> Tensor:
    flat = expert_outputs.reshape(-1, expert_outputs.shape[-2], expert_outputs.shape[-1])
    selected = flat[
        torch.arange(flat.shape[0], device=flat.device)[:, None], result.subsets
    ].to(dtype=result.coefficients.dtype)
    return torch.einsum("tk,tkh->th", result.coefficients, selected).reshape(
        *expert_outputs.shape[:-2], expert_outputs.shape[-1]
    )


def run_reconstruction_rollout(
    teacher: nn.Module,
    student: nn.Module,
    batches: Iterable[EvaluationBatch],
    condition: ReconstructionCondition,
    k: int,
    checkpoint_dir: Path,
    run_signature: str,
    *,
    source_hashes: Mapping[str, str] | None = None,
    solver_settings: SolverSettings = SolverSettings(),
) -> ReconstructionRolloutResult:
    """Roll out one immutable condition using teacher-next minus student baseline.

    Teacher FFN output is never used as the selection target.  It is only
    possible for the teacher decoder hook to supply its next hidden state.
    """

    teacher_mlps, student_mlps = dict(iter_text_mlps(teacher)), dict(iter_text_mlps(student))
    if teacher_mlps.keys() != student_mlps.keys() or not teacher_mlps:
        raise ValueError("teacher and student text-layer layouts differ")
    if not all(isinstance(mlp, PartitionedQwen35MLP) for mlp in student_mlps.values()):
        raise TypeError("student text MLPs must be PartitionedQwen35MLP modules")
    num_experts = next(iter(student_mlps.values())).num_experts
    if num_experts != 8 or not 1 <= k <= num_experts:
        raise ValueError("Reconstruction-H requires K in [1, 8] and exactly eight experts")
    if any(mlp.num_experts != num_experts for mlp in student_mlps.values()):
        raise ValueError("student text MLPs use inconsistent expert counts")
    accumulator = _Accumulator(
        num_layers=len(teacher_mlps), num_experts=num_experts, condition=condition,
        k=k, checkpoint_dir=checkpoint_dir, run_signature=run_signature,
        source_hashes=source_hashes,
    )
    teacher_device, student_device = _model_device(teacher), _model_device(student)
    if student_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(student_device)

    for batch in batches:
        if batch.index in accumulator.completed_batches:
            continue
        teacher_targets: dict[int, Tensor] = {}
        residuals: dict[int, Tensor] = {}
        teacher_handles: list[object] = []
        student_handles: list[object] = []
        delta = _BatchDelta(
            layer_stats=[[0.0, 0.0, 0] for _ in teacher_mlps],
            selection_counts=[[0] * num_experts for _ in teacher_mlps],
            solver_diagnostics=[{} for _ in teacher_mlps],
            reference_distances=[{} for _ in teacher_mlps],
            coefficient_diagnostics=_new_coefficient_diagnostics(),
        )
        try:
            try:
                teacher_text, student_text = teacher.model.language_model, student.model.language_model
            except AttributeError as error:
                raise ValueError("rollout models must expose model.language_model") from error
            for layer in teacher_mlps:
                decoder_layer = teacher_text.layers[layer]

                def capture_target(
                    _module: nn.Module, _args: tuple[Tensor, ...], output: Tensor,
                    *, layer_index: int = layer,
                ) -> None:
                    if not isinstance(output, Tensor):
                        raise TypeError("teacher decoder layer output must be a tensor")
                    teacher_targets[layer_index] = output.detach()

                teacher_handles.append(decoder_layer.register_forward_hook(capture_target))
            with torch.inference_mode():
                teacher_text(input_ids=batch.input_ids.to(teacher_device), attention_mask=batch.attention_mask.to(teacher_device), use_cache=False)
            for handle in teacher_handles:
                handle.remove()
            teacher_handles.clear()
            if teacher_targets.keys() != teacher_mlps.keys():
                raise RuntimeError("teacher text backbone did not visit every decoder layer")

            for layer in student_mlps:
                decoder_layer = student_text.layers[layer]

                def capture_baseline(
                    _module: nn.Module, args: tuple[Tensor, ...], *, layer_index: int = layer,
                ) -> None:
                    if not args or not isinstance(args[0], Tensor):
                        raise TypeError("post-attention pre-hook requires a hidden-state tensor")
                    residuals[layer_index] = args[0].detach()

                student_handles.append(decoder_layer.post_attention_layernorm.register_forward_pre_hook(capture_baseline))

            with ExitStack() as transforms:
                for layer, mlp in student_mlps.items():

                    def choose_output(
                        _normalized_student_state: Tensor, expert_outputs: Tensor,
                        *, layer_index: int = layer,
                    ) -> Tensor:
                        if layer_index not in residuals:
                            raise RuntimeError("missing post-attention residual baseline for rollout layer")
                        baseline = residuals.pop(layer_index).to(expert_outputs.device)
                        target_next = teacher_targets[layer_index].to(expert_outputs.device)
                        if baseline.shape != target_next.shape:
                            raise ValueError("teacher and student hidden-state shapes differ")
                        result = select_reconstruction_h(target_next, baseline, expert_outputs, condition, k, solver_settings)
                        weighted = _weighted_output(expert_outputs, result)
                        valid = batch.attention_mask.to(device=expert_outputs.device, dtype=torch.bool)
                        correction = (target_next - baseline).reshape(-1, target_next.shape[-1])
                        flat_experts = expert_outputs.reshape(-1, num_experts, expert_outputs.shape[-1])
                        valid_flat = valid.reshape(-1)
                        correction_valid = correction[valid_flat]
                        e8_valid = flat_experts.sum(dim=1)[valid_flat]
                        r8_result = score_full_reference(correction, flat_experts, condition, solver_settings)
                        r8_valid = _weighted_output(flat_experts, r8_result)[valid_flat]
                        r8_name = f"R8[{_condition_id(condition)}]"
                        _reference_delta(delta.reference_distances[layer_index], "D0", correction_valid, "E8", e8_valid)
                        _reference_delta(delta.reference_distances[layer_index], "D0", correction_valid, r8_name, r8_valid)
                        _reference_delta(delta.reference_distances[layer_index], "E8", e8_valid, r8_name, r8_valid)
                        _Accumulator._merge_diagnostics(delta.solver_diagnostics[layer_index], result.diagnostics)
                        _Accumulator._merge_diagnostics(delta.solver_diagnostics[layer_index], {f"r8_{name}": value for name, value in r8_result.diagnostics.items()})
                        chosen_next = baseline.double() + weighted.double()
                        target_valid, chosen_valid = target_next[valid].double(), chosen_next[valid]
                        if not bool(torch.isfinite(target_valid).all()) or not bool(torch.isfinite(chosen_valid).all()):
                            raise ValueError("Reconstruction-H hidden metrics must be finite")
                        delta.layer_stats[layer_index][0] += _cpu_fp64_sum((chosen_valid - target_valid).square().sum(dim=-1))
                        delta.layer_stats[layer_index][1] += _cpu_fp64_sum(target_valid.square().sum(dim=-1))
                        delta.layer_stats[layer_index][2] += int(valid.sum().item())
                        selected = result.subsets.reshape(*expert_outputs.shape[:-2], k)
                        counts = torch.bincount(selected[valid].reshape(-1), minlength=num_experts).cpu().tolist()
                        for expert, count in enumerate(counts):
                            delta.selection_counts[layer_index][expert] += int(count)
                        coefficients = result.coefficients.detach().double()
                        delta.coefficient_diagnostics["coefficient_count"] += int(coefficients.numel())
                        delta.coefficient_diagnostics["negative_count"] += int((coefficients < 0.0).sum().item())
                        delta.coefficient_diagnostics["above_two_count"] += int((coefficients.abs() > 2.0).sum().item())
                        delta.coefficient_diagnostics["absolute_deviation_sum"] += _cpu_fp64_sum((coefficients - 1.0).abs())
                        delta.coefficient_diagnostics["bounded_lower_saturation_count"] += int((coefficients < 1e-6).sum().item()) if condition.family == "bounded" else 0
                        delta.coefficient_diagnostics["bounded_upper_saturation_count"] += int((coefficients > 2.0 - 1e-6).sum().item()) if condition.family == "bounded" else 0
                        return weighted.to(dtype=expert_outputs.dtype)

                    transforms.enter_context(mlp.output_transform(choose_output))
                with torch.inference_mode():
                    output = student_text(input_ids=batch.input_ids.to(student_device), attention_mask=batch.attention_mask.to(student_device), use_cache=False)
                    hidden = getattr(output, "last_hidden_state", None)
                    if not isinstance(hidden, Tensor):
                        raise TypeError("student text backbone did not return last_hidden_state")
                    logits = student.lm_head(hidden)
                    ids = batch.input_ids.to(logits.device)
                    mask = batch.attention_mask.to(logits.device, dtype=torch.bool)
                    valid_labels = mask[..., 1:] & mask[..., :-1]
                    if bool(valid_labels.any()):
                        nll = F.cross_entropy(logits[..., :-1, :][valid_labels].float(), ids[..., 1:][valid_labels], reduction="none")
                        if not bool(torch.isfinite(nll).all()):
                            raise ValueError("Reconstruction-H causal NLL is not finite")
                        delta.nll_sum, delta.scored_tokens = _cpu_fp64_sum(nll), int(nll.numel())
            accumulator.merge_completed(batch.index, delta, _peak_cuda_bytes(student_device))
        except Exception as error:
            accumulator.record_failure(batch.index, f"{type(error).__name__}: {error}", _peak_cuda_bytes(student_device))
            break
        finally:
            for handle in teacher_handles:
                handle.remove()
            for handle in student_handles:
                handle.remove()
    return accumulator.finalize()
