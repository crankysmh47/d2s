"""Greedy, token-wise rollout oracle for converted Qwen3.5 text FFNs."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import json
from math import exp
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ppmoe.adapters.qwen35_runtime import iter_text_mlps
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.moefication.qwen35 import PartitionedQwen35MLP
from ppmoe.oracle.exact import OracleResult, exact_gram_oracle
from ppmoe.provenance import atomic_write_json


@dataclass(frozen=True, slots=True)
class RolloutLayerMetrics:
    token_count: int
    hidden_squared_error_sum: float
    teacher_squared_norm_sum: float

    @property
    def normalized_mse(self) -> float:
        if self.teacher_squared_norm_sum == 0.0:
            return 0.0 if self.hidden_squared_error_sum == 0.0 else float("inf")
        return self.hidden_squared_error_sum / self.teacher_squared_norm_sum


@dataclass(frozen=True, slots=True)
class RolloutResult:
    layer_metrics: Mapping[int, RolloutLayerMetrics]
    selection_counts: Mapping[int, tuple[int, ...]]
    nll_sum: float
    scored_tokens: int
    ppl: float | None
    completed_batches: int
    failed_batches: int
    peak_cuda_bytes: int
    complete: bool


@dataclass(slots=True)
class _BatchDelta:
    layer_stats: list[list[float | int]]
    selection_counts: list[list[int]]
    nll_sum: float = 0.0
    scored_tokens: int = 0


def _scoring_dtype(*values: Tensor) -> torch.dtype:
    return torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32


def _cpu_fp64_sum(per_token: Tensor) -> float:
    return float(per_token.to(device="cpu", dtype=torch.float64).sum().item())


def select_rollout_subsets(
    target_next: Tensor,
    residual: Tensor,
    expert_outputs: Tensor,
    k: int,
) -> OracleResult:
    """Select fixed-K experts that best reconstruct ``target_next - residual``."""

    if target_next.shape != residual.shape:
        raise ValueError("target_next and residual must have the same shape")
    if expert_outputs.shape[:-2] != target_next.shape[:-1] or expert_outputs.shape[-1] != target_next.shape[-1]:
        raise ValueError("expert_outputs must align with target_next token dimensions")
    if target_next.device != residual.device or target_next.device != expert_outputs.device:
        raise ValueError("rollout oracle tensors must share a device")
    if target_next.ndim < 2:
        raise ValueError("rollout oracle requires token and hidden dimensions")
    scoring_dtype = _scoring_dtype(target_next, residual, expert_outputs)
    target = (target_next - residual).detach().to(dtype=scoring_dtype).reshape(-1, target_next.shape[-1])
    experts = expert_outputs.detach().to(dtype=scoring_dtype).reshape(
        -1, expert_outputs.shape[-2], expert_outputs.shape[-1]
    )
    return exact_gram_oracle(target, experts, k)


class _RolloutAccumulator:
    _STATE_NAME = "rollout_oracle_state.json"

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        k: int,
        checkpoint_dir: Path,
        run_signature: str,
    ) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.k = k
        self.checkpoint_dir = checkpoint_dir
        self.run_signature = run_signature
        self.completed_batches: set[int] = set()
        self.failures: dict[int, str] = {}
        self.nll_sum = 0.0
        self.scored_tokens = 0
        self.peak_cuda_bytes = 0
        self.layer_stats = [[0.0, 0.0, 0] for _ in range(num_layers)]
        self.selection_counts = [[0 for _ in range(num_experts)] for _ in range(num_layers)]
        self._load()

    @property
    def _state_path(self) -> Path:
        return self.checkpoint_dir / self._STATE_NAME

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        state = json.loads(self._state_path.read_text(encoding="utf-8"))
        if state.get("run_signature") != self.run_signature:
            raise ValueError("rollout checkpoint signature does not match inputs")
        if (
            state.get("k") != self.k
            or state.get("num_layers") != self.num_layers
            or state.get("num_experts") != self.num_experts
        ):
            raise ValueError("rollout checkpoint layout does not match invocation")
        self.completed_batches = {int(index) for index in state["completed_batches"]}
        self.failures = {int(index): str(message) for index, message in state["failures"].items()}
        self.nll_sum = float(state["nll_sum"])
        self.scored_tokens = int(state["scored_tokens"])
        self.peak_cuda_bytes = int(state["peak_cuda_bytes"])
        self.layer_stats = [
            [float(values[0]), float(values[1]), int(values[2])]
            for values in state["layer_stats"]
        ]
        self.selection_counts = [
            [int(value) for value in row] for row in state["selection_counts"]
        ]

    def _persist(self) -> None:
        atomic_write_json(
            self._state_path,
            {
                "completed_batches": sorted(self.completed_batches),
                "failures": {str(index): self.failures[index] for index in sorted(self.failures)},
                "k": self.k,
                "layer_stats": self.layer_stats,
                "nll_sum": self.nll_sum,
                "num_experts": self.num_experts,
                "num_layers": self.num_layers,
                "peak_cuda_bytes": self.peak_cuda_bytes,
                "run_signature": self.run_signature,
                "scored_tokens": self.scored_tokens,
                "selection_counts": self.selection_counts,
            },
        )

    def merge_completed(self, batch_index: int, delta: _BatchDelta, peak_cuda_bytes: int) -> None:
        if batch_index in self.completed_batches or batch_index in self.failures:
            raise ValueError(f"batch {batch_index} already has durable state")
        self.nll_sum += delta.nll_sum
        self.scored_tokens += delta.scored_tokens
        self.peak_cuda_bytes = max(self.peak_cuda_bytes, peak_cuda_bytes)
        for layer, values in enumerate(delta.layer_stats):
            for index, value in enumerate(values):
                self.layer_stats[layer][index] += value
            for expert, count in enumerate(delta.selection_counts[layer]):
                self.selection_counts[layer][expert] += count
        self.completed_batches.add(batch_index)
        self._persist()

    def record_failure(self, batch_index: int, message: str, peak_cuda_bytes: int) -> None:
        if batch_index in self.completed_batches:
            raise ValueError(f"batch {batch_index} is already completed")
        self.peak_cuda_bytes = max(self.peak_cuda_bytes, peak_cuda_bytes)
        self.failures[batch_index] = message
        self._persist()

    def finalize(self) -> RolloutResult:
        complete = not self.failures and self.scored_tokens > 0
        return RolloutResult(
            layer_metrics={
                layer: RolloutLayerMetrics(
                    token_count=int(values[2]),
                    hidden_squared_error_sum=float(values[0]),
                    teacher_squared_norm_sum=float(values[1]),
                )
                for layer, values in enumerate(self.layer_stats)
            },
            selection_counts={
                layer: tuple(values) for layer, values in enumerate(self.selection_counts)
            },
            nll_sum=self.nll_sum,
            scored_tokens=self.scored_tokens,
            ppl=exp(self.nll_sum / self.scored_tokens) if complete else None,
            completed_batches=len(self.completed_batches),
            failed_batches=len(self.failures),
            peak_cuda_bytes=self.peak_cuda_bytes,
            complete=complete,
        )


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _peak_cuda_bytes(device: torch.device) -> int:
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0


def run_rollout_oracle(
    teacher: nn.Module,
    student: nn.Module,
    batches: Iterable[EvaluationBatch],
    *,
    k: int,
    checkpoint_dir: Path,
    run_signature: str,
) -> RolloutResult:
    """Run one fixed-K greedy rollout, persisting only whole successful batches."""

    teacher_mlps = dict(iter_text_mlps(teacher))
    student_mlps = dict(iter_text_mlps(student))
    if teacher_mlps.keys() != student_mlps.keys() or not teacher_mlps:
        raise ValueError("teacher and student text-layer layouts differ")
    if not all(isinstance(mlp, PartitionedQwen35MLP) for mlp in student_mlps.values()):
        raise TypeError("student text MLPs must be PartitionedQwen35MLP modules")
    num_experts = next(iter(student_mlps.values())).num_experts
    if not 1 <= k <= num_experts:
        raise ValueError("k is outside the student expert range")
    if any(mlp.num_experts != num_experts for mlp in student_mlps.values()):
        raise ValueError("student text MLPs use inconsistent expert counts")
    accumulator = _RolloutAccumulator(
        num_layers=len(teacher_mlps),
        num_experts=num_experts,
        k=k,
        checkpoint_dir=checkpoint_dir,
        run_signature=run_signature,
    )
    teacher_device = _model_device(teacher)
    student_device = _model_device(student)
    if student_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(student_device)

    for batch in batches:
        if batch.index in accumulator.completed_batches:
            continue
        teacher_targets: dict[int, Tensor] = {}
        residuals: dict[int, Tensor] = {}
        teacher_handles = []
        student_handles = []
        delta = _BatchDelta(
            layer_stats=[[0.0, 0.0, 0] for _ in range(len(teacher_mlps))],
            selection_counts=[[0 for _ in range(num_experts)] for _ in range(len(teacher_mlps))],
        )
        try:
            try:
                teacher_text = teacher.model.language_model
                student_text = student.model.language_model
            except AttributeError as error:
                raise ValueError("rollout models must expose model.language_model") from error
            for layer in teacher_mlps:
                decoder_layer = teacher_text.layers[layer]

                def capture_target(
                    _module: nn.Module,
                    _args: tuple[Tensor, ...],
                    output: Tensor,
                    *,
                    layer_index: int = layer,
                ) -> None:
                    if not isinstance(output, Tensor):
                        raise TypeError("teacher decoder layer output must be a tensor")
                    teacher_targets[layer_index] = output.detach()

                teacher_handles.append(decoder_layer.register_forward_hook(capture_target))
            with torch.inference_mode():
                teacher_text(
                    input_ids=batch.input_ids.to(teacher_device),
                    attention_mask=batch.attention_mask.to(teacher_device),
                    use_cache=False,
                )
            for handle in teacher_handles:
                handle.remove()
            teacher_handles.clear()
            if teacher_targets.keys() != teacher_mlps.keys():
                raise RuntimeError("teacher text backbone did not visit every decoder layer")

            for layer in student_mlps:
                decoder_layer = student_text.layers[layer]

                def capture_residual(
                    _module: nn.Module,
                    args: tuple[Tensor, ...],
                    *,
                    layer_index: int = layer,
                ) -> None:
                    if not args or not isinstance(args[0], Tensor):
                        raise TypeError("post-attention pre-hook requires a hidden-state tensor")
                    residuals[layer_index] = args[0]

                student_handles.append(
                    decoder_layer.post_attention_layernorm.register_forward_pre_hook(capture_residual)
                )

            with ExitStack() as transforms:
                for layer, mlp in student_mlps.items():
                    def choose_output(
                        _x: Tensor,
                        expert_outputs: Tensor,
                        *,
                        layer_index: int = layer,
                    ) -> Tensor:
                        if layer_index not in residuals:
                            raise RuntimeError("missing post-attention residual for rollout layer")
                        residual = residuals.pop(layer_index)
                        target_next = teacher_targets[layer_index].to(expert_outputs.device)
                        if target_next.dtype != expert_outputs.dtype:
                            target_next = target_next.to(dtype=expert_outputs.dtype)
                        if residual.shape != target_next.shape:
                            raise ValueError("teacher and student hidden-state shapes differ")
                        selection = select_rollout_subsets(
                            target_next,
                            residual,
                            expert_outputs,
                            k,
                        ).subsets.reshape(*expert_outputs.shape[:-2], k)
                        selected_output = torch.gather(
                            expert_outputs,
                            -2,
                            selection[..., None].expand(*selection.shape, expert_outputs.shape[-1]),
                        ).sum(dim=-2)
                        valid = batch.attention_mask.to(device=target_next.device, dtype=torch.bool)
                        selected_next = residual + selected_output
                        target_valid = target_next[valid]
                        next_valid = selected_next[valid]
                        scoring_dtype = _scoring_dtype(target_valid, next_valid)
                        target_scoring = target_valid.detach().to(dtype=scoring_dtype)
                        next_scoring = next_valid.detach().to(dtype=scoring_dtype)
                        if not torch.isfinite(target_scoring).all() or not torch.isfinite(next_scoring).all():
                            raise ValueError("rollout hidden metrics must be finite")
                        delta.layer_stats[layer_index][0] += _cpu_fp64_sum(
                            (next_scoring - target_scoring).square().sum(dim=-1)
                        )
                        delta.layer_stats[layer_index][1] += _cpu_fp64_sum(
                            target_scoring.square().sum(dim=-1)
                        )
                        delta.layer_stats[layer_index][2] += int(valid.sum().item())
                        selection_counts = torch.bincount(
                            selection[valid].reshape(-1), minlength=num_experts
                        ).cpu().tolist()
                        for expert, count in enumerate(selection_counts):
                            delta.selection_counts[layer_index][expert] += int(count)
                        return selected_output

                    transforms.enter_context(mlp.output_transform(choose_output))
                with torch.inference_mode():
                    student_output = student_text(
                        input_ids=batch.input_ids.to(student_device),
                        attention_mask=batch.attention_mask.to(student_device),
                        use_cache=False,
                    )
                    hidden_states = getattr(student_output, "last_hidden_state", None)
                    if not isinstance(hidden_states, Tensor):
                        raise TypeError("student text backbone did not return last_hidden_state")
                    logits = student.lm_head(hidden_states)
                    input_ids = batch.input_ids.to(logits.device)
                    attention_mask = batch.attention_mask.to(logits.device, dtype=torch.bool)
                    valid_labels = attention_mask[..., 1:] & attention_mask[..., :-1]
                    if valid_labels.any():
                        nll = F.cross_entropy(
                            logits[..., :-1, :][valid_labels].float(),
                            input_ids[..., 1:][valid_labels],
                            reduction="none",
                        )
                        if not torch.isfinite(nll).all():
                            raise ValueError("rollout causal NLL is not finite")
                        delta.nll_sum = _cpu_fp64_sum(nll)
                        delta.scored_tokens = int(nll.numel())
            accumulator.merge_completed(batch.index, delta, _peak_cuda_bytes(student_device))
        except Exception as error:
            accumulator.record_failure(
                batch.index,
                f"{type(error).__name__}: {error}",
                _peak_cuda_bytes(student_device),
            )
            break
        finally:
            for handle in teacher_handles:
                handle.remove()
            for handle in student_handles:
                handle.remove()
            teacher_targets.clear()
            residuals.clear()
    return accumulator.finalize()
