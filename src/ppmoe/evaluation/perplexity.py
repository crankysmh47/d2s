"""Causal perplexity accumulation with durable batch-level resume."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import exp
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ppmoe.provenance import atomic_write_json


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    index: int
    input_ids: Tensor
    attention_mask: Tensor


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    nll_sum: float
    scored_tokens: int
    failed_batches: int
    completed_batches: int
    ppl: float | None
    complete: bool
    peak_cuda_bytes: int


@dataclass(frozen=True, slots=True)
class M0Result:
    teacher: PerplexityResult
    student: PerplexityResult
    ppl_ratio: float
    logits_cosine: float
    fixed_batch_index: int
    input_ids_sha256: str
    attention_mask_sha256: str
    logits_finite: bool
    max_abs_logit_difference: float
    passed: bool


class M0GateError(RuntimeError):
    """Raised when dense/all-expert equivalence is not established."""


class NLLAccumulator:
    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.nll_sum = 0.0
        self.scored_tokens = 0
        self.completed_batches: set[int] = set()
        self.failures: dict[int, str] = {}
        if checkpoint_dir is not None:
            state_path = checkpoint_dir / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.nll_sum = float(state["nll_sum"])
                self.scored_tokens = int(state["scored_tokens"])
                self.completed_batches = {
                    int(value) for value in state["completed_batches"]
                }
                self.failures = {
                    int(index): str(message)
                    for index, message in state["failures"].items()
                }

    def _persist(self) -> None:
        if self.checkpoint_dir is None:
            return
        atomic_write_json(
            self.checkpoint_dir / "state.json",
            {
                "completed_batches": sorted(self.completed_batches),
                "failures": {
                    str(index): self.failures[index] for index in sorted(self.failures)
                },
                "nll_sum": self.nll_sum,
                "scored_tokens": self.scored_tokens,
            },
        )

    def update(
        self,
        logits: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        batch_index: int,
    ) -> None:
        if batch_index in self.completed_batches:
            raise ValueError(f"batch {batch_index} is already completed")
        if logits.ndim != 3 or input_ids.shape != attention_mask.shape:
            raise ValueError("logits and token tensors have incompatible shapes")
        if logits.shape[:2] != input_ids.shape:
            raise ValueError("logits leading dimensions must match input_ids")
        if input_ids.shape[-1] < 2:
            raise ValueError("causal perplexity batches require at least two tokens")
        valid = attention_mask[..., 1:].bool() & attention_mask[..., :-1].bool()
        shifted_logits = logits[..., :-1, :]
        shifted_labels = input_ids[..., 1:]
        selected_logits = shifted_logits[valid]
        selected_labels = shifted_labels[valid]
        if selected_labels.numel():
            per_token_nll = F.cross_entropy(
                selected_logits.float(),
                selected_labels,
                reduction="none",
            )
            if not torch.isfinite(per_token_nll).all():
                raise ValueError("batch negative log likelihood is not finite")
            self.nll_sum += float(
                per_token_nll.to(device="cpu", dtype=torch.float64).sum().item()
            )
            self.scored_tokens += int(selected_labels.numel())
        self.completed_batches.add(batch_index)
        self._persist()

    def record_failure(self, batch_index: int, message: str) -> None:
        if batch_index in self.completed_batches:
            raise ValueError(f"batch {batch_index} is already completed")
        self.failures[batch_index] = message
        self._persist()

    def finalize(self, *, peak_cuda_bytes: int = 0) -> PerplexityResult:
        complete = not self.failures and self.scored_tokens > 0
        ppl = exp(self.nll_sum / self.scored_tokens) if complete else None
        return PerplexityResult(
            nll_sum=self.nll_sum,
            scored_tokens=self.scored_tokens,
            failed_batches=len(self.failures),
            completed_batches=len(self.completed_batches),
            ppl=ppl,
            complete=complete,
            peak_cuda_bytes=peak_cuda_bytes,
        )


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _forward_logits(model: nn.Module, batch: EvaluationBatch) -> Tensor:
    device = _model_device(model)
    output = model(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        use_cache=False,
        logits_to_keep=0,
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("model output must expose a logits tensor")
    return logits


def _tensor_sha256(tensor: Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    header = f"{cpu.dtype}:{tuple(cpu.shape)}:".encode("ascii")
    return sha256(header + cpu.numpy().tobytes()).hexdigest()


def evaluate_perplexity(
    model: nn.Module,
    batches: Iterable[EvaluationBatch],
    checkpoint_dir: Path,
) -> PerplexityResult:
    accumulator = NLLAccumulator(checkpoint_dir)
    device = _model_device(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch in batches:
            if batch.index in accumulator.completed_batches:
                continue
            try:
                logits = _forward_logits(model, batch)
                accumulator.update(
                    logits,
                    batch.input_ids.to(logits.device),
                    batch.attention_mask.to(logits.device),
                    batch_index=batch.index,
                )
            except Exception as error:
                accumulator.record_failure(batch.index, f"{type(error).__name__}: {error}")
                break
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return accumulator.finalize(peak_cuda_bytes=peak)


def evaluate_m0(
    teacher: nn.Module,
    student: nn.Module,
    batches: Iterable[EvaluationBatch],
    checkpoint_dir: Path,
) -> M0Result:
    materialized_batches = tuple(batches)
    if not materialized_batches:
        raise ValueError("M0 requires at least one evaluation batch")
    teacher_result = evaluate_perplexity(
        teacher,
        materialized_batches,
        checkpoint_dir / "teacher",
    )
    student_result = evaluate_perplexity(
        student,
        materialized_batches,
        checkpoint_dir / "student",
    )
    if not teacher_result.complete or not student_result.complete:
        raise M0GateError("perplexity evaluation is incomplete")
    assert teacher_result.ppl is not None and student_result.ppl is not None
    ratio = student_result.ppl / teacher_result.ppl
    if not 0.999 <= ratio <= 1.001:
        raise M0GateError(f"perplexity ratio {ratio:.9f} is outside [0.999, 1.001]")
    first = materialized_batches[0]
    with torch.inference_mode():
        teacher_logits = _forward_logits(teacher, first)
        student_logits = _forward_logits(student, first)
    if not bool(torch.isfinite(teacher_logits).all()) or not bool(torch.isfinite(student_logits).all()):
        raise M0GateError("M0 discrepancy logits are not finite")
    teacher_logits = teacher_logits.double().reshape(-1)
    student_logits = student_logits.double().reshape(-1)
    if teacher_logits.shape != student_logits.shape:
        raise M0GateError("teacher and student logit shapes differ")
    denominator = teacher_logits.norm() * student_logits.norm()
    if float(denominator) == 0.0:
        cosine = 1.0 if torch.equal(teacher_logits, student_logits) else 0.0
    else:
        cosine = float(torch.dot(teacher_logits, student_logits) / denominator)
    max_abs_difference = float((teacher_logits - student_logits).abs().max().item())
    if not torch.isfinite(torch.tensor(cosine)) or cosine < 0.99999:
        raise M0GateError(f"logits cosine {cosine:.9f} is below 0.99999")
    return M0Result(
        teacher=teacher_result,
        student=student_result,
        ppl_ratio=ratio,
        logits_cosine=cosine,
        fixed_batch_index=first.index,
        input_ids_sha256=_tensor_sha256(first.input_ids),
        attention_mask_sha256=_tensor_sha256(first.attention_mask),
        logits_finite=True,
        max_abs_logit_difference=max_abs_difference,
        passed=True,
    )
