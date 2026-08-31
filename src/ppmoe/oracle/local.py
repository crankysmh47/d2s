"""Exact local-oracle aggregation for converted Qwen3.5 text FFNs.

The local oracle deliberately scores every converted expert on the dense
teacher's FFN input.  It therefore measures the best possible *local*
reconstruction without introducing rollout error from preceding sparse layers.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn

from ppmoe.adapters.qwen35_runtime import iter_text_mlps
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.moefication.qwen35 import PartitionedQwen35MLP
from ppmoe.oracle.exact import exact_direct_oracle, exact_gram_oracle
from ppmoe.provenance import atomic_write_json


@dataclass(frozen=True, slots=True)
class LocalOracleLayerK:
    """Aggregate fixed-K reconstruction statistics for one text layer."""

    token_count: int
    squared_error_sum: float
    target_squared_norm_sum: float

    @property
    def normalized_mse(self) -> float:
        if self.target_squared_norm_sum == 0.0:
            return 0.0 if self.squared_error_sum == 0.0 else float("inf")
        return self.squared_error_sum / self.target_squared_norm_sum


@dataclass(frozen=True, slots=True)
class LocalOracleResult:
    layer_k: Mapping[tuple[int, int], LocalOracleLayerK]
    selection_counts: Mapping[tuple[int, int], tuple[int, ...]]
    co_selection: Mapping[tuple[int, int], tuple[tuple[int, ...], ...]]
    interaction_sum: tuple[tuple[tuple[float, ...], ...], ...]
    cancellation_count: tuple[tuple[tuple[int, ...], ...], ...]
    completed_batches: int
    failed_batches: int
    direct_check_count: int
    all_expert_check_count: int
    complete: bool


class LocalOracleAccumulator:
    """FP64 local-oracle metrics with atomic completed-batch checkpoints."""

    _STATE_NAME = "local_oracle_state.json"
    _DIRECT_CHECK_PREFIX_TOKENS = 16

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        k_values: tuple[int, ...],
        checkpoint_dir: Path | None = None,
        run_signature: str | None = None,
    ) -> None:
        if num_layers <= 0 or num_experts <= 0:
            raise ValueError("num_layers and num_experts must be positive")
        if not k_values or tuple(sorted(k_values, reverse=True)) != k_values:
            raise ValueError("k_values must be non-empty and descending")
        if len(set(k_values)) != len(k_values) or any(
            k < 1 or k > num_experts for k in k_values
        ):
            raise ValueError("k_values must be unique values within the expert range")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.k_values = k_values
        self.checkpoint_dir = checkpoint_dir
        self.run_signature = run_signature
        self.completed_batches: set[int] = set()
        self.failures: dict[int, str] = {}
        self.direct_check_count = 0
        self.all_expert_check_count = 0
        self._direct_checked_pairs: set[tuple[int, int]] = set()
        self._stats = {
            (layer, k): [0.0, 0.0, 0]
            for layer in range(num_layers)
            for k in k_values
        }
        self._selection_counts = {
            (layer, k): [0 for _ in range(num_experts)]
            for layer in range(num_layers)
            for k in k_values
        }
        self._co_selection = {
            (layer, k): [[0 for _ in range(num_experts)] for _ in range(num_experts)]
            for layer in range(num_layers)
            for k in k_values
        }
        self._interaction_sum = [
            [[0.0 for _ in range(num_experts)] for _ in range(num_experts)]
            for _ in range(num_layers)
        ]
        self._cancellation_count = [
            [[0 for _ in range(num_experts)] for _ in range(num_experts)]
            for _ in range(num_layers)
        ]
        self._load()

    @property
    def _state_path(self) -> Path | None:
        return None if self.checkpoint_dir is None else self.checkpoint_dir / self._STATE_NAME

    @staticmethod
    def _key(layer: int, k: int) -> str:
        return f"{layer}:{k}"

    def _load(self) -> None:
        state_path = self._state_path
        if state_path is None or not state_path.exists():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_signature") != self.run_signature:
            raise ValueError("local oracle checkpoint signature does not match inputs")
        if (
            state.get("num_layers") != self.num_layers
            or state.get("num_experts") != self.num_experts
            or tuple(state.get("k_values", ())) != self.k_values
        ):
            raise ValueError("local oracle checkpoint shape does not match configuration")
        self.completed_batches = {int(index) for index in state["completed_batches"]}
        self.failures = {int(index): str(message) for index, message in state["failures"].items()}
        self.direct_check_count = int(state.get("direct_check_count", 0))
        self.all_expert_check_count = int(state.get("all_expert_check_count", 0))
        self._direct_checked_pairs = {
            (int(layer), int(k)) for layer, k in state.get("direct_checked_pairs", ())
        }
        if any(
            layer < 0
            or layer >= self.num_layers
            or k not in self.k_values
            for layer, k in self._direct_checked_pairs
        ):
            raise ValueError("local oracle checkpoint has invalid direct-check pairs")
        for (layer, k), values in self._stats.items():
            saved = state["stats"][self._key(layer, k)]
            values[:] = [float(saved[0]), float(saved[1]), int(saved[2])]
            self._selection_counts[(layer, k)] = [
                int(value) for value in state["selection_counts"][self._key(layer, k)]
            ]
            self._co_selection[(layer, k)] = [
                [int(value) for value in row]
                for row in state["co_selection"][self._key(layer, k)]
            ]
        self._interaction_sum = [
            [[float(value) for value in row] for row in layer]
            for layer in state["interaction_sum"]
        ]
        self._cancellation_count = [
            [[int(value) for value in row] for row in layer]
            for layer in state["cancellation_count"]
        ]

    def _persist(self) -> None:
        state_path = self._state_path
        if state_path is None:
            return
        atomic_write_json(
            state_path,
            {
                "cancellation_count": self._cancellation_count,
                "co_selection": {
                    self._key(layer, k): self._co_selection[(layer, k)]
                    for layer, k in self._stats
                },
                "completed_batches": sorted(self.completed_batches),
                "all_expert_check_count": self.all_expert_check_count,
                "direct_check_count": self.direct_check_count,
                "direct_checked_pairs": [list(pair) for pair in sorted(self._direct_checked_pairs)],
                "failures": {str(index): self.failures[index] for index in sorted(self.failures)},
                "interaction_sum": self._interaction_sum,
                "k_values": list(self.k_values),
                "num_experts": self.num_experts,
                "num_layers": self.num_layers,
                "run_signature": self.run_signature,
                "selection_counts": {
                    self._key(layer, k): self._selection_counts[(layer, k)]
                    for layer, k in self._stats
                },
                "stats": {
                    self._key(layer, k): self._stats[(layer, k)]
                    for layer, k in self._stats
                },
            },
        )

    def snapshot(self) -> tuple[object, ...]:
        """Capture unpersisted state so one failed batch can be rolled back."""

        return (
            deepcopy(self._stats),
            deepcopy(self._selection_counts),
            deepcopy(self._co_selection),
            deepcopy(self._interaction_sum),
            deepcopy(self._cancellation_count),
            self.direct_check_count,
            self.all_expert_check_count,
            deepcopy(self._direct_checked_pairs),
        )

    def restore(self, snapshot: tuple[object, ...]) -> None:
        """Discard all metric deltas made after :meth:`snapshot`."""

        (
            self._stats,
            self._selection_counts,
            self._co_selection,
            self._interaction_sum,
            self._cancellation_count,
            self.direct_check_count,
            self.all_expert_check_count,
            self._direct_checked_pairs,
        ) = snapshot

    def update_layer(self, layer: int, target: Tensor, expert_outputs: Tensor) -> None:
        """Add one layer's unpadded token results without retaining activations."""

        if not 0 <= layer < self.num_layers:
            raise ValueError("layer index is out of range")
        if target.ndim != 2 or expert_outputs.ndim != 3:
            raise ValueError("target must be [tokens, hidden] and experts [tokens, experts, hidden]")
        if expert_outputs.shape[0] != target.shape[0] or expert_outputs.shape[2] != target.shape[1]:
            raise ValueError("target and expert_outputs shapes do not match")
        if expert_outputs.shape[1] != self.num_experts:
            raise ValueError("expert output count does not match accumulator")
        if target.device != expert_outputs.device:
            raise ValueError("target and expert_outputs must share a device")
        if not torch.isfinite(target).all() or not torch.isfinite(expert_outputs).all():
            raise ValueError("local oracle inputs must be finite")
        if target.shape[0] == 0:
            return

        scoring_dtype = (
            torch.float64
            if target.dtype == torch.float64 or expert_outputs.dtype == torch.float64
            else torch.float32
        )
        scoring_target = target.detach().to(dtype=scoring_dtype)
        scoring_experts = expert_outputs.detach().to(dtype=scoring_dtype)
        interactions = torch.einsum("teh,tfh->tef", scoring_experts, scoring_experts)
        interaction_sum = interactions.to(device="cpu", dtype=torch.float64).sum(dim=0).tolist()
        cancellation = (interactions < 0).sum(dim=0).cpu().tolist()
        for expert_a in range(self.num_experts):
            for expert_b in range(self.num_experts):
                self._interaction_sum[layer][expert_a][expert_b] += float(
                    interaction_sum[expert_a][expert_b]
                )
                self._cancellation_count[layer][expert_a][expert_b] += int(
                    cancellation[expert_a][expert_b]
                )

        target_norm = float(
            scoring_target.square().sum(dim=-1).to(device="cpu", dtype=torch.float64).sum().item()
        )
        for k in self.k_values:
            chosen = exact_gram_oracle(scoring_target, scoring_experts, k).subsets
            pair = (layer, k)
            if pair not in self._direct_checked_pairs:
                checked_tokens = min(target.shape[0], self._DIRECT_CHECK_PREFIX_TOKENS)
                direct = exact_direct_oracle(
                    scoring_target[:checked_tokens],
                    scoring_experts[:checked_tokens],
                    k,
                )
                if not torch.equal(chosen[:checked_tokens], direct.subsets):
                    raise RuntimeError("direct and Gram local-oracle subsets disagree")
                self._direct_checked_pairs.add(pair)
                self.direct_check_count += checked_tokens
            selected = torch.gather(
                scoring_experts,
                1,
                chosen[..., None].expand(-1, -1, scoring_experts.shape[-1]),
            ).sum(dim=1)
            if k == self.num_experts:
                expected = torch.arange(self.num_experts, device=chosen.device).expand_as(chosen)
                if not torch.equal(chosen, expected):
                    raise RuntimeError("all-expert oracle did not select every expert in order")
                if not torch.equal(selected, scoring_experts.sum(dim=1)):
                    raise RuntimeError("all-expert oracle output disagrees with the literal expert sum")
                self.all_expert_check_count += target.shape[0]
            squared_error = float(
                (selected - scoring_target)
                .square()
                .sum(dim=-1)
                .to(device="cpu", dtype=torch.float64)
                .sum()
                .item()
            )
            stats = self._stats[(layer, k)]
            stats[0] += squared_error
            stats[1] += target_norm
            stats[2] += target.shape[0]
            selection_counts = torch.bincount(
                chosen.reshape(-1), minlength=self.num_experts
            ).cpu().tolist()
            for expert, count in enumerate(selection_counts):
                self._selection_counts[(layer, k)][expert] += int(count)
            selected_mask = torch.zeros(
                (target.shape[0], self.num_experts),
                dtype=torch.bool,
                device=chosen.device,
            )
            selected_mask.scatter_(1, chosen, True)
            co_selection = (
                selected_mask[:, :, None]
                .logical_and(selected_mask[:, None, :])
                .sum(dim=0)
                .cpu()
                .tolist()
            )
            for expert_a in range(self.num_experts):
                for expert_b in range(self.num_experts):
                    self._co_selection[(layer, k)][expert_a][expert_b] += int(
                        co_selection[expert_a][expert_b]
                    )

    def mark_batch_completed(self, batch_index: int) -> None:
        if batch_index in self.completed_batches:
            raise ValueError(f"batch {batch_index} is already completed")
        if batch_index in self.failures:
            raise ValueError(f"batch {batch_index} has already failed")
        self.completed_batches.add(batch_index)
        self._persist()

    def record_failure(self, batch_index: int, message: str) -> None:
        if batch_index in self.completed_batches:
            raise ValueError(f"batch {batch_index} is already completed")
        self.failures[batch_index] = message
        self._persist()

    def finalize(self) -> LocalOracleResult:
        layer_k = {
            key: LocalOracleLayerK(
                token_count=int(values[2]),
                squared_error_sum=float(values[0]),
                target_squared_norm_sum=float(values[1]),
            )
            for key, values in self._stats.items()
        }
        return LocalOracleResult(
            layer_k=layer_k,
            selection_counts={
                key: tuple(values) for key, values in self._selection_counts.items()
            },
            co_selection={
                key: tuple(tuple(row) for row in matrix)
                for key, matrix in self._co_selection.items()
            },
            interaction_sum=tuple(
                tuple(tuple(row) for row in layer) for layer in self._interaction_sum
            ),
            cancellation_count=tuple(
                tuple(tuple(row) for row in layer) for layer in self._cancellation_count
            ),
            completed_batches=len(self.completed_batches),
            failed_batches=len(self.failures),
            direct_check_count=self.direct_check_count,
            all_expert_check_count=self.all_expert_check_count,
            complete=not self.failures and bool(self.completed_batches),
        )


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def run_local_oracle(
    teacher: nn.Module,
    student: nn.Module,
    batches: Iterable[EvaluationBatch],
    *,
    k_values: tuple[int, ...],
    checkpoint_dir: Path,
    run_signature: str,
) -> LocalOracleResult:
    """Score all fixed-K subsets on dense teacher FFN inputs batch by batch."""

    teacher_mlps = dict(iter_text_mlps(teacher))
    student_mlps = dict(iter_text_mlps(student))
    if teacher_mlps.keys() != student_mlps.keys() or not teacher_mlps:
        raise ValueError("teacher and student text-layer layouts differ")
    if not all(isinstance(mlp, PartitionedQwen35MLP) for mlp in student_mlps.values()):
        raise TypeError("student text MLPs must be PartitionedQwen35MLP modules")
    num_experts = next(iter(student_mlps.values())).num_experts
    if any(mlp.num_experts != num_experts for mlp in student_mlps.values()):
        raise ValueError("student text MLPs use inconsistent expert counts")
    accumulator = LocalOracleAccumulator(
        num_layers=len(teacher_mlps),
        num_experts=num_experts,
        k_values=k_values,
        checkpoint_dir=checkpoint_dir,
        run_signature=run_signature,
    )
    teacher_device = _model_device(teacher)
    student_device = _model_device(student)

    for batch in batches:
        if batch.index in accumulator.completed_batches:
            continue
        captured: dict[int, tuple[Tensor, Tensor]] = {}
        handles = []
        batch_snapshot = accumulator.snapshot()
        try:
            for layer, mlp in teacher_mlps.items():
                def capture(module: nn.Module, args: tuple[Tensor, ...], output: Tensor, *, layer_index: int = layer) -> None:
                    if not args or not isinstance(args[0], Tensor) or not isinstance(output, Tensor):
                        raise TypeError("Qwen text MLP hooks require tensor input and output")
                    captured[layer_index] = (args[0].detach(), output.detach())

                handles.append(mlp.register_forward_hook(capture))
            with torch.inference_mode():
                try:
                    language_model = teacher.model.language_model
                except AttributeError as error:
                    raise ValueError("teacher is missing model.language_model") from error
                language_model(
                    input_ids=batch.input_ids.to(teacher_device),
                    attention_mask=batch.attention_mask.to(teacher_device),
                    use_cache=False,
                )
                if captured.keys() != teacher_mlps.keys():
                    raise RuntimeError("teacher forward did not visit every text MLP")
                for layer, (teacher_input, target) in captured.items():
                    experts = student_mlps[layer].expert_outputs(
                        teacher_input.to(student_device)
                    )
                    if experts.device != target.device:
                        experts = experts.to(target.device)
                    valid = batch.attention_mask.to(target.device, dtype=torch.bool).reshape(-1)
                    accumulator.update_layer(
                        layer,
                        target.reshape(-1, target.shape[-1])[valid],
                        experts.reshape(-1, num_experts, experts.shape[-1])[valid],
                    )
            accumulator.mark_batch_completed(batch.index)
        except Exception as error:
            accumulator.restore(batch_snapshot)
            accumulator.record_failure(batch.index, f"{type(error).__name__}: {error}")
            break
        finally:
            for handle in handles:
                handle.remove()
            captured.clear()
    return accumulator.finalize()
