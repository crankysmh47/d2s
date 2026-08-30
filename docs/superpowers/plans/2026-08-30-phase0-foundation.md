# Phase 0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove exact parameter-preserving SwiGLU permutation and balanced expert partitioning on a tiny synthetic model, validate a fixed-K exact oracle, measure BF16 drift, and produce a configuration-only Qwen3.5 compatibility report.

**Architecture:** A small `ppmoe` Python package separates neuron mapping, reference expert execution, numerical verification, oracle scoring, and Qwen adaptation. Dense and partitioned modules use real PyTorch operations; optimized kernels and full model conversion remain outside this slice. Every mathematical implementation is tested against a simpler reference before it becomes an experiment dependency.

**Tech Stack:** Python 3.11, `uv`, PyTorch 2.6 or newer, Transformers 5.12 or newer but below 6, pytest 8 or newer.

**Spec:** `docs/superpowers/specs/2026-08-30-phase0-foundation-design.md`

## Global Constraints

- Run all production behavior through red-green-refactor: observe the intended test failure before adding implementation.
- Keep all changes on `phase0-foundation`; do not modify `main`.
- Use `trust_remote_code=False` for Hugging Face configuration access.
- Do not download Qwen model weights, datasets, checkpoints, or the 100K-token calibration corpus in this slice.
- Do not modify the Qwen vision stack.
- Add the down-projection output bias exactly once after expert summation.
- Treat FP32 all-expert relative error `<= 1e-5` as a hard gate.
- Use `rtol=1e-5` and `atol=1e-6` for FP32 input/parameter-gradient comparisons unless an observed and logged numerical reason justifies a change.
- Log BF16 maximum absolute error, RMS error, normalized MSE, and cosine similarity; do not label BF16 accumulation as algebraically exact.
- Keep caches, weights, datasets, checkpoints, credentials, and large generated arrays out of Git.
- Append commands, outcomes, failures, mitigations, commit IDs, and the next gate to `PROJECT_LOG.md`; never include secrets.

---

## File map

```text
pyproject.toml                         # package metadata and dependency bounds
uv.lock                                # exact resolved environment
.gitignore                             # caches, environments, weights, datasets, outputs
src/ppmoe/__init__.py                  # public package metadata
src/ppmoe/moefication/neuron.py        # parameter bundle and permutation validation
src/ppmoe/moefication/partition.py     # balanced expert index construction
src/ppmoe/moefication/reference.py     # dense and reference partitioned SwiGLU
src/ppmoe/moefication/verify.py        # numerical metrics
src/ppmoe/oracle/exact.py              # direct and Gram fixed-K oracle
src/ppmoe/adapters/qwen35.py           # config and MLP layout inspection
src/ppmoe/phase0.py                    # reproducible Phase 0 numerical probe
scripts/verify_phase0.py               # JSON-producing Phase 0 entry point
scripts/probe_qwen35.py                 # config-only Qwen compatibility entry point
tests/helpers.py                        # tiny two-layer SwiGLU Transformer fixture
tests/test_neuron_triplet.py            # validation and bundle tests
tests/test_permutation.py               # neuron-axis permutation equivalence
tests/test_partition.py                 # balanced partition tests
tests/test_ffn_equivalence.py           # all-expert functional equivalence
tests/test_gradient_equivalence.py      # dense/partitioned gradient mapping
tests/test_model_equivalence.py         # two-layer synthetic integration
tests/test_oracle_bruteforce.py         # direct fixed-K subset scoring
tests/test_gram_oracle.py               # Gram/direct agreement
tests/test_bf16_drift.py                # CUDA BF16 metrics
tests/test_phase0_probe.py               # deterministic summary schema
tests/test_qwen35_probe.py               # offline config/layout inspection
PROJECT_LOG.md                           # append-only execution record
```

---

### Task 1: Bootstrap the package and validate neuron permutations

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/ppmoe/__init__.py`
- Create: `src/ppmoe/moefication/__init__.py`
- Create: `src/ppmoe/moefication/neuron.py`
- Create: `tests/test_neuron_triplet.py`
- Modify: `PROJECT_LOG.md`
- Generate: `uv.lock`

**Interfaces:**
- Consumes: one-dimensional integer `torch.Tensor` permutations.
- Produces: `validate_permutation(permutation: Tensor, intermediate_size: int) -> Tensor` and immutable `SwiGLUParameters`.

- [x] **Step 1: Add package metadata, ignore rules, and the failing validation tests**

`pyproject.toml` must declare a `src` package, Python `>=3.11,<3.12`, runtime dependencies `torch>=2.6,<3` and `transformers>=5.12,<6`, plus `pytest>=8,<9` in the development group. Route only `torch` through the explicit official `https://download.pytorch.org/whl/cu130` uv index so Windows does not resolve a CPU-only PyPI wheel. `.gitignore` must cover `.venv/`, Python caches, pytest caches, Hugging Face caches, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `checkpoints/`, `datasets/`, `outputs/`, and `results/` except explicitly checked-in small fixtures.

Create these tests:

```python
import pytest
import torch

from ppmoe.moefication.neuron import SwiGLUParameters, validate_permutation


def test_validate_permutation_returns_long_tensor():
    result = validate_permutation(torch.tensor([2, 0, 1]), 3)
    assert result.dtype == torch.long
    assert result.tolist() == [2, 0, 1]


@pytest.mark.parametrize(
    "values,size,message",
    [
        ([0, 1], 3, "length"),
        ([0, 0, 2], 3, "exactly once"),
        ([-1, 0, 1], 3, "range"),
        ([0, 1, 3], 3, "range"),
    ],
)
def test_validate_permutation_rejects_invalid_indices(values, size, message):
    with pytest.raises(ValueError, match=message):
        validate_permutation(torch.tensor(values), size)


def test_swiglu_parameters_checks_neuron_axes():
    with pytest.raises(ValueError, match="intermediate"):
        SwiGLUParameters(
            gate_weight=torch.zeros(4, 3),
            up_weight=torch.zeros(5, 3),
            down_weight=torch.zeros(3, 4),
        )
```

- [x] **Step 2: Create and lock the environment**

Run: `uv sync --group dev`

Expected: `.venv` and `uv.lock` are created; dependency resolution succeeds without fetching model weights.

- [x] **Step 3: Run the validation tests and observe RED**

Run: `uv run pytest tests/test_neuron_triplet.py -v`

Expected: collection fails with `ModuleNotFoundError` or import failure for `ppmoe.moefication.neuron`.

- [x] **Step 4: Implement the minimal validated parameter bundle**

Implement this complete validation behavior:

```python
_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


@dataclass(frozen=True)
class SwiGLUParameters:
    gate_weight: Tensor
    up_weight: Tensor
    down_weight: Tensor
    gate_bias: Tensor | None = None
    up_bias: Tensor | None = None
    down_bias: Tensor | None = None

    def __post_init__(self) -> None:
        weights = (self.gate_weight, self.up_weight, self.down_weight)
        if any(weight.ndim != 2 for weight in weights):
            raise ValueError("gate, up, and down weights must all be rank 2")
        intermediate_size, hidden_size = self.gate_weight.shape
        if self.up_weight.shape != (intermediate_size, hidden_size):
            raise ValueError("up weight has an inconsistent intermediate or hidden axis")
        if self.down_weight.shape != (hidden_size, intermediate_size):
            raise ValueError("down weight has an inconsistent hidden or intermediate axis")
        expected_bias_sizes = (
            (self.gate_bias, intermediate_size, "gate"),
            (self.up_bias, intermediate_size, "up"),
            (self.down_bias, hidden_size, "down"),
        )
        for bias, expected_size, name in expected_bias_sizes:
            if bias is not None and bias.shape != (expected_size,):
                raise ValueError(f"{name} bias must have shape ({expected_size},)")
        tensors = [tensor for tensor in (*weights, self.gate_bias, self.up_bias, self.down_bias) if tensor is not None]
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("all SwiGLU parameters must have the same dtype")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all SwiGLU parameters must be on the same device")

    @property
    def intermediate_size(self) -> int:
        return self.gate_weight.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.gate_weight.shape[1]


def validate_permutation(permutation: Tensor, intermediate_size: int) -> Tensor:
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if permutation.ndim != 1:
        raise ValueError("permutation must be rank 1")
    if permutation.dtype not in _INTEGER_DTYPES:
        raise ValueError("permutation must use an integer dtype")
    if permutation.numel() != intermediate_size:
        raise ValueError(f"permutation length must equal {intermediate_size}")
    if torch.any(permutation < 0) or torch.any(permutation >= intermediate_size):
        raise ValueError(f"permutation indices must be in range [0, {intermediate_size})")
    if torch.unique(permutation).numel() != intermediate_size:
        raise ValueError("permutation must contain each index exactly once")
    return permutation.detach().to(dtype=torch.long).clone()
```

- [x] **Step 5: Run tests and observe GREEN**

Run: `uv run pytest tests/test_neuron_triplet.py -v`

Expected: all tests pass with no warnings.

- [x] **Step 6: Record and commit the bootstrap milestone**

Append the resolved Torch/Transformers versions, RED and GREEN commands, and result to `PROJECT_LOG.md`.

```powershell
git add .gitignore pyproject.toml uv.lock src/ppmoe tests/test_neuron_triplet.py PROJECT_LOG.md
git commit -m "feat: validate SwiGLU neuron parameters"
```

---

### Task 2: Permute complete SwiGLU neuron triplets

**Files:**
- Modify: `src/ppmoe/moefication/neuron.py`
- Create: `tests/test_permutation.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: `SwiGLUParameters`, plus a validated permutation.
- Produces: `permute_swiglu_parameters(parameters: SwiGLUParameters, permutation: Tensor) -> SwiGLUParameters`.

- [x] **Step 1: Write the failing equivalence and non-mutation tests**

```python
import torch
import torch.nn.functional as F

from ppmoe.moefication.neuron import SwiGLUParameters, permute_swiglu_parameters


def evaluate(parameters, x):
    gate = F.linear(x, parameters.gate_weight, parameters.gate_bias)
    up = F.linear(x, parameters.up_weight, parameters.up_bias)
    return F.linear(F.silu(gate) * up, parameters.down_weight, parameters.down_bias)


def test_permuting_all_three_neuron_axes_preserves_output_and_inputs():
    generator = torch.Generator().manual_seed(7)
    parameters = SwiGLUParameters(
        gate_weight=torch.randn(8, 4, generator=generator),
        up_weight=torch.randn(8, 4, generator=generator),
        down_weight=torch.randn(4, 8, generator=generator),
        gate_bias=torch.randn(8, generator=generator),
        up_bias=torch.randn(8, generator=generator),
        down_bias=torch.randn(4, generator=generator),
    )
    x = torch.randn(2, 3, 4, generator=generator)
    permutation = torch.tensor([5, 1, 7, 0, 3, 2, 6, 4])
    original_gate = parameters.gate_weight.clone()

    permuted = permute_swiglu_parameters(parameters, permutation)

    torch.testing.assert_close(evaluate(parameters, x), evaluate(permuted, x))
    torch.testing.assert_close(parameters.gate_weight, original_gate)
    torch.testing.assert_close(permuted.down_bias, parameters.down_bias)
```

- [x] **Step 2: Observe RED**

Run: `uv run pytest tests/test_permutation.py -v`

Expected: import fails because `permute_swiglu_parameters` does not exist.

- [x] **Step 3: Implement consistent, non-mutating indexing**

Use `index_select(0, permutation)` for gate/up weights and biases, `index_select(1, permutation)` for the down weight, and clone the unpermuted down bias. Return a new `SwiGLUParameters` object.

- [x] **Step 4: Observe GREEN and run the bootstrap regression**

Run: `uv run pytest tests/test_neuron_triplet.py tests/test_permutation.py -v`

Expected: all tests pass.

- [x] **Step 5: Log and commit**

```powershell
git add src/ppmoe/moefication/neuron.py tests/test_permutation.py PROJECT_LOG.md
git commit -m "feat: preserve SwiGLU under neuron permutation"
```

---

### Task 3: Build balanced partitions and all-expert reference execution

**Files:**
- Create: `src/ppmoe/moefication/partition.py`
- Create: `src/ppmoe/moefication/reference.py`
- Create: `tests/test_partition.py`
- Create: `tests/test_ffn_equivalence.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: validated permutation, number of experts, and `SwiGLUParameters`.
- Produces: `balanced_partition(permutation: Tensor, num_experts: int) -> Sequence[Tensor]`, `DenseSwiGLU`, `ExpertSwiGLU`, and `PartitionedSwiGLU.from_dense(dense, partitions)`.

- [x] **Step 1: Write failing balanced-partition tests**

```python
import pytest
import torch

from ppmoe.moefication.partition import balanced_partition


def test_balanced_partition_preserves_order_and_covers_indices():
    permutation = torch.tensor([4, 0, 6, 2, 7, 1, 5, 3])
    groups = balanced_partition(permutation, 4)
    assert [group.tolist() for group in groups] == [[4, 0], [6, 2], [7, 1], [5, 3]]


def test_balanced_partition_rejects_non_divisible_width():
    with pytest.raises(ValueError, match="divisible"):
        balanced_partition(torch.arange(7), 4)
```

- [x] **Step 2: Observe partition RED, implement, and observe GREEN**

Run RED: `uv run pytest tests/test_partition.py -v`

Implement exact positive expert-count validation, divisibility validation, and `torch.split` with equal widths.

Run GREEN: `uv run pytest tests/test_partition.py -v`

- [x] **Step 3: Write the failing all-expert output and bias tests**

```python
import torch

from ppmoe.moefication.reference import DenseSwiGLU, PartitionedSwiGLU


def test_all_experts_match_dense_fp32_with_down_bias_added_once():
    torch.manual_seed(11)
    dense = DenseSwiGLU(16, 32, gate_bias=True, up_bias=True, down_bias=True)
    partitioned = PartitionedSwiGLU.from_dense(
        dense,
        partitions=tuple(torch.arange(32).reshape(4, 8)),
    )
    x = torch.randn(2, 5, 16)

    expected = dense(x)
    actual = partitioned(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert partitioned.down_bias is not None
    assert all(expert.down_proj.bias is None for expert in partitioned.experts)


def test_selected_experts_returns_sum_for_requested_indices():
    torch.manual_seed(13)
    dense = DenseSwiGLU(8, 16, down_bias=False)
    partitioned = PartitionedSwiGLU.from_dense(
        dense,
        partitions=tuple(torch.arange(16).reshape(4, 4)),
    )
    x = torch.randn(3, 8)
    expected = partitioned.expert_outputs(x)[:, [0, 2], :].sum(dim=1)
    torch.testing.assert_close(partitioned(x, selected_experts=(0, 2)), expected)
```

- [x] **Step 4: Observe reference execution RED**

Run: `uv run pytest tests/test_ffn_equivalence.py -v`

Expected: import failure for `ppmoe.moefication.reference`.

- [x] **Step 5: Implement minimal dense and partitioned modules**

Implement the following concrete public surface; every method body follows the formulas immediately below the code block:

```python
class DenseSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, gate_bias: bool = False, up_bias: bool = False, down_bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=gate_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=up_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=down_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertSwiGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)


class PartitionedSwiGLU(nn.Module):
    @classmethod
    def from_dense(cls, dense: DenseSwiGLU, partitions: Sequence[Tensor]) -> "PartitionedSwiGLU":
        return cls(dense=dense, partitions=partitions)

    def expert_outputs(self, x: Tensor) -> Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=-2)

    def forward(
        self,
        x: Tensor,
        selected_experts: Sequence[int] | None = None,
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> Tensor:
        indices = range(len(self.experts)) if selected_experts is None else selected_experts
        outputs = [self.experts[index](x) for index in indices]
        if not outputs:
            raise ValueError("selected_experts must not be empty")
        target_dtype = accumulation_dtype or outputs[0].dtype
        result = torch.stack([output.to(target_dtype) for output in outputs]).sum(dim=0)
        if self.down_bias is not None:
            result = result + self.down_bias.to(target_dtype)
        return result.to(x.dtype)
```

`ExpertSwiGLU.__init__` creates bias-compatible gate/up linear layers and a bias-free down layer, then copies the dense rows/columns named by its partition inside `torch.no_grad()`. `PartitionedSwiGLU.__init__` validates that flattened partition indices are a complete permutation of `range(dense.intermediate_size)`, that all partitions have equal nonzero width, and that index tensors share the source device. It stores one cloned down bias as `nn.Parameter` when the dense module has one; it never installs that bias on an expert.

- [x] **Step 6: Observe GREEN and run all current tests**

Run: `uv run pytest tests/test_neuron_triplet.py tests/test_permutation.py tests/test_partition.py tests/test_ffn_equivalence.py -v`

Expected: all pass, and FP32 relative error stays below `1e-5`.

- [x] **Step 7: Log numerical maxima and commit**

```powershell
git add src/ppmoe/moefication/partition.py src/ppmoe/moefication/reference.py tests/test_partition.py tests/test_ffn_equivalence.py PROJECT_LOG.md
git commit -m "feat: add balanced reference experts"
```

---

### Task 4: Verify gradient and two-layer model equivalence

**Files:**
- Create: `tests/helpers.py`
- Create: `tests/test_gradient_equivalence.py`
- Create: `tests/test_model_equivalence.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: `DenseSwiGLU`, `PartitionedSwiGLU`, and each module's public projections.
- Produces: test-only `TinySwiGLUBlock` and `TinySwiGLUTransformer`; no new production API.

- [x] **Step 1: Add a mapped-gradient characterization test**

The test must clone one dense FFN into four experts, run the same scalar loss, and compare input gradients. For parameter gradients, concatenate each expert's gate/up gradients along dimension 0 and down gradients along dimension 1, restore original neuron order from the partitions, and compare with the dense gradients using `rtol=1e-5`, `atol=1e-6`.

Core assertion:

```python
torch.testing.assert_close(partitioned_input.grad, dense_input.grad, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(restored_gate_grad, dense.gate_proj.weight.grad, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(restored_up_grad, dense.up_proj.weight.grad, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(restored_down_grad, dense.down_proj.weight.grad, rtol=1e-5, atol=1e-6)
```

- [x] **Step 2: Run the gradient characterization**

Run: `uv run pytest tests/test_gradient_equivalence.py -v`

Expected: PASS when the existing expert construction preserved autograd; any failure is a Task 3 defect and must be debugged before proceeding.

- [x] **Step 3: Confirm no production change is needed**

Verify expert weights are leaf `nn.Parameter` objects copied from dense slices and that no forward path uses `.detach()` on activations or expert parameters. The characterization passed, so production code remained unchanged.

Run: `uv run pytest tests/test_gradient_equivalence.py -v`

Expected: PASS.

- [x] **Step 4: Add the two-layer integration characterization**

`tests/helpers.py` defines a seeded block with `RMSNorm`, `nn.MultiheadAttention(batch_first=True)`, a residual connection, a second `RMSNorm`, and `DenseSwiGLU`. Build a two-layer dense model, deep-copy it, replace both copied MLPs with `PartitionedSwiGLU`, and assert final output equivalence:

```python
torch.testing.assert_close(converted(tokens), dense(tokens), rtol=1e-5, atol=1e-6)
```

- [x] **Step 5: Run the integration characterization**

Add the test-only fixture and replacement helper, then run:

`uv run pytest tests/test_gradient_equivalence.py tests/test_model_equivalence.py -v`

Expected GREEN: both pass.

- [x] **Step 6: Run the complete CPU suite, log, and commit**

Run: `uv run pytest -v -m "not cuda"`

```powershell
git add src/ppmoe/moefication/reference.py tests/helpers.py tests/test_gradient_equivalence.py tests/test_model_equivalence.py PROJECT_LOG.md
git commit -m "test: prove synthetic model gradient equivalence"
```

---

### Task 5: Add stable numerical verification metrics

**Files:**
- Create: `src/ppmoe/moefication/verify.py`
- Create: `tests/test_verify_metrics.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: equal-shaped reference and candidate tensors.
- Produces: immutable `NumericalMetrics` and `compare_tensors(reference: Tensor, candidate: Tensor) -> NumericalMetrics`.

- [x] **Step 1: Write failing metric tests with hand-computable tensors**

```python
import math
import pytest
import torch

from ppmoe.moefication.verify import compare_tensors


def test_compare_tensors_reports_expected_error_metrics():
    reference = torch.tensor([1.0, 2.0])
    candidate = torch.tensor([2.0, 2.0])
    metrics = compare_tensors(reference, candidate)
    assert metrics.max_absolute_error == 1.0
    assert math.isclose(metrics.rms_error, math.sqrt(0.5))
    assert math.isclose(metrics.relative_error, 1.0 / math.sqrt(5.0))
    assert math.isclose(metrics.normalized_mse, 1.0 / 5.0)
    assert math.isclose(metrics.cosine_similarity, 6.0 / math.sqrt(40.0))


def test_compare_tensors_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        compare_tensors(torch.zeros(2), torch.zeros(3))
```

- [x] **Step 2: Observe RED, implement in FP64, and observe GREEN**

Run RED: `uv run pytest tests/test_verify_metrics.py -v`

Implementation converts detached inputs to FP64 for metric calculation, defines relative error as `norm(candidate-reference) / max(norm(reference), eps)`, defines normalized MSE as `mean((candidate-reference)^2) / max(mean(reference^2), eps)`, and handles two all-zero tensors with cosine similarity `1.0`.

Run GREEN: `uv run pytest tests/test_verify_metrics.py -v`

- [x] **Step 3: Log and commit**

```powershell
git add src/ppmoe/moefication/verify.py tests/test_verify_metrics.py PROJECT_LOG.md
git commit -m "feat: add numerical fidelity metrics"
```

---

### Task 6: Implement the direct fixed-K exact oracle

**Files:**
- Create: `src/ppmoe/oracle/__init__.py`
- Create: `src/ppmoe/oracle/exact.py`
- Create: `tests/test_oracle_bruteforce.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: `target: Tensor[tokens, hidden]`, `expert_outputs: Tensor[tokens, experts, hidden]`, and integer `k`.
- Produces: `OracleResult(subsets: Tensor[tokens, k], squared_errors: Tensor[tokens])` and `exact_direct_oracle(target, expert_outputs, k) -> OracleResult`.

- [x] **Step 1: Write failing direct-oracle tests**

```python
import pytest
import torch

from ppmoe.oracle.exact import exact_direct_oracle


def test_direct_oracle_selects_best_subset_per_token():
    experts = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 2.0]],
    ])
    target = torch.tensor([[1.0, 1.0], [1.0, 2.0]])
    result = exact_direct_oracle(target, experts, k=2)
    assert result.subsets.tolist() == [[0, 1], [1, 2]]
    torch.testing.assert_close(result.squared_errors, torch.zeros(2))


def test_direct_oracle_uses_lexicographic_tie_breaking():
    experts = torch.zeros(1, 3, 2)
    result = exact_direct_oracle(torch.zeros(1, 2), experts, k=1)
    assert result.subsets.tolist() == [[0]]


@pytest.mark.parametrize("k", [0, 4])
def test_direct_oracle_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="1 <= k <= 3"):
        exact_direct_oracle(torch.zeros(1, 2), torch.zeros(1, 3, 2), k=k)
```

- [x] **Step 2: Observe RED**

Run: `uv run pytest tests/test_oracle_bruteforce.py -v`

Expected: import failure for `ppmoe.oracle.exact`.

- [x] **Step 3: Implement deterministic enumeration**

Use `itertools.combinations(range(num_experts), k)` in lexicographic order, materialize the small Phase 0 combination index tensor, reconstruct candidates with indexed expert sums, calculate squared error in FP32 or FP64, and use the first `argmin` on ties.

- [x] **Step 4: Observe GREEN, log, and commit**

Run: `uv run pytest tests/test_oracle_bruteforce.py -v`

```powershell
git add src/ppmoe/oracle tests/test_oracle_bruteforce.py PROJECT_LOG.md
git commit -m "feat: add fixed-K direct oracle"
```

---

### Task 7: Implement and cross-check the Gram exact oracle

**Files:**
- Modify: `src/ppmoe/oracle/exact.py`
- Create: `tests/test_gram_oracle.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: the same tensors and `k` as `exact_direct_oracle`.
- Produces: `exact_gram_oracle(target: Tensor, expert_outputs: Tensor, k: int) -> OracleResult`.

- [x] **Step 1: Write failing randomized agreement and cancellation tests**

```python
import torch

from ppmoe.oracle.exact import exact_direct_oracle, exact_gram_oracle


def test_gram_oracle_matches_direct_oracle_across_k():
    generator = torch.Generator().manual_seed(23)
    experts = torch.randn(5, 8, 6, generator=generator)
    target = experts.sum(dim=1) + 0.01 * torch.randn(5, 6, generator=generator)
    for k in (1, 2, 4, 8):
        direct = exact_direct_oracle(target, experts, k)
        gram = exact_gram_oracle(target, experts, k)
        assert gram.subsets.tolist() == direct.subsets.tolist()
        torch.testing.assert_close(gram.squared_errors, direct.squared_errors, rtol=1e-5, atol=1e-6)


def test_gram_oracle_accounts_for_expert_cancellation():
    experts = torch.tensor([[[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[0.0, 0.0]])
    result = exact_gram_oracle(target, experts, k=2)
    assert result.subsets.tolist() == [[0, 1]]
```

- [x] **Step 2: Observe RED**

Run: `uv run pytest tests/test_gram_oracle.py -v`

Expected: import failure because `exact_gram_oracle` is missing.

- [x] **Step 3: Implement the Gram objective**

For every token calculate `target_norm = t^T t`, `target_expert = t^T e_i`, and `expert_gram = e_i^T e_j`. For subset indicator `m`, score `target_norm - 2*m^T*target_expert + m^T*expert_gram*m`. Clamp only tiny negative roundoff in the returned error to zero; never clamp values before subset selection.

- [x] **Step 4: Observe GREEN and run oracle regressions**

Run: `uv run pytest tests/test_oracle_bruteforce.py tests/test_gram_oracle.py -v`

Expected: direct and Gram subsets agree for every seeded case.

- [x] **Step 5: Log interaction evidence and commit**

```powershell
git add src/ppmoe/oracle/exact.py tests/test_gram_oracle.py PROJECT_LOG.md
git commit -m "feat: add Gram exact oracle"
```

---

### Task 8: Produce reproducible FP32 and BF16 Phase 0 reports

**Files:**
- Create: `src/ppmoe/phase0.py`
- Create: `scripts/verify_phase0.py`
- Create: `tests/test_phase0_probe.py`
- Create: `tests/test_bf16_drift.py`
- Modify: `pyproject.toml`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: `device: torch.device`, `dtype: torch.dtype`, integer seed, hidden/intermediate sizes, and expert count.
- Produces: `run_phase0_probe(device, dtype, seed, hidden_size, intermediate_size, num_experts) -> dict[str, object]` with versioned JSON-safe fields.

- [ ] **Step 1: Write the failing deterministic report-schema test**

```python
from ppmoe.phase0 import run_phase0_probe


def test_phase0_probe_is_deterministic_and_json_safe():
    first = run_phase0_probe(device="cpu", dtype="float32", seed=31, hidden_size=16, intermediate_size=32, num_experts=4)
    second = run_phase0_probe(device="cpu", dtype="float32", seed=31, hidden_size=16, intermediate_size=32, num_experts=4)
    assert first == second
    assert first["schema_version"] == 1
    assert first["all_expert"]["relative_error"] <= 1e-5
    assert first["oracle"]["direct_gram_subsets_equal"] is True
```

- [ ] **Step 2: Observe RED, implement the probe, and observe GREEN**

Run RED: `uv run pytest tests/test_phase0_probe.py -v`

The implementation seeds a local generator, constructs one dense FFN and a deterministic balanced random permutation, captures all expert outputs, compares dense/all-expert output, runs direct and Gram oracles for valid `k` values, and returns only JSON-safe scalars/lists plus PyTorch, CUDA, device, dtype, dimensions, and tolerance metadata.

Run GREEN: `uv run pytest tests/test_phase0_probe.py -v`

- [ ] **Step 3: Write the CUDA BF16 drift test and observe its first result**

Mark the test `@pytest.mark.cuda`. Skip only if CUDA is unavailable or `torch.cuda.is_bf16_supported()` is false. Compare native BF16 expert summation with expert outputs accumulated in FP32 and cast back to BF16. Assert metrics are finite and the FP32-accumulated normalized MSE is no worse than native BF16 beyond `1e-7` tolerance.

Run: `uv run pytest tests/test_bf16_drift.py -v`

Expected: PASS on the RTX 5060 Ti or a precise skip reason. If FP32 accumulation is empirically worse, stop and investigate rather than weakening the assertion.

- [ ] **Step 4: Add the JSON CLI and run both reports**

`scripts/verify_phase0.py` accepts `--device`, `--dtype`, `--seed`, `--hidden-size`, `--intermediate-size`, `--num-experts`, and required `--output`. It creates only the output parent directory and serializes the report with sorted keys and indentation.

Run:

```powershell
uv run python scripts/verify_phase0.py --device cpu --dtype float32 --seed 31 --hidden-size 128 --intermediate-size 512 --num-experts 8 --output results/phase0-fp32.json
uv run python scripts/verify_phase0.py --device cuda --dtype bfloat16 --seed 31 --hidden-size 128 --intermediate-size 512 --num-experts 8 --output results/phase0-bf16.json
```

Expected: both commands exit zero; `results/` remains ignored by Git.

- [ ] **Step 5: Run the full suite, log exact metrics, and commit**

Run: `uv run pytest -v`

```powershell
git add pyproject.toml src/ppmoe/phase0.py scripts/verify_phase0.py tests/test_phase0_probe.py tests/test_bf16_drift.py PROJECT_LOG.md
git commit -m "feat: report Phase 0 numerical fidelity"
```

---

### Task 9: Add the configuration-only Qwen3.5 readiness probe

**Files:**
- Create: `src/ppmoe/adapters/__init__.py`
- Create: `src/ppmoe/adapters/qwen35.py`
- Create: `scripts/probe_qwen35.py`
- Create: `tests/test_qwen35_probe.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: a Transformers Qwen3.5 composite or text config and an `nn.Module` expected to be a text MLP.
- Produces: `Qwen35ConfigReport`, `Qwen35MLPReport`, `inspect_qwen35_config(config)`, and `inspect_qwen35_mlp(module)`.

- [ ] **Step 1: Write failing offline config-inspection tests**

Use `types.SimpleNamespace` to create a composite config with `text_config.hidden_size=1024`, `intermediate_size=3584`, `num_hidden_layers=24`, `hidden_act="silu"`, and eight experts. Assert the report selects the nested text config, reports expert width 448, and rejects widths not divisible by eight.

```python
report = inspect_qwen35_config(composite_config, num_experts=8)
assert report.hidden_size == 1024
assert report.intermediate_size == 3584
assert report.expert_width == 448
assert report.num_hidden_layers == 24
```

- [ ] **Step 2: Write failing real-module layout tests**

Construct small `nn.Linear` projections with names `gate_proj`, `up_proj`, and `down_proj` in a local test module. Assert `inspect_qwen35_mlp` reports row/row/column-compatible shapes and all three bias flags. Add negative cases for missing projections and incompatible shapes, with expected/actual shapes present in the error.

- [ ] **Step 3: Observe RED, implement pure inspection, and observe GREEN**

Run RED: `uv run pytest tests/test_qwen35_probe.py -v`

Implement only attribute/shape inspection; do not import or instantiate the full Qwen model in the adapter.

Run GREEN: `uv run pytest tests/test_qwen35_probe.py -v`

- [ ] **Step 4: Add and run the opt-in network integration CLI**

The script must call:

```python
config = AutoConfig.from_pretrained(
    "Qwen/Qwen3.5-0.8B-Base",
    trust_remote_code=False,
)
text_config = config.text_config
mlp = Qwen3_5MLP(text_config, text_config.intermediate_size)
```

It must inspect only this newly initialized text MLP, print/write a JSON report, and verify that no `*.safetensors` or PyTorch weight file was fetched into the repo. It must not call any `from_pretrained` model class.

Run:

`uv run python scripts/probe_qwen35.py --output results/qwen35-config-probe.json`

Expected: config and tokenizer-sized metadata may enter the external Hugging Face cache; no model weights are downloaded. Report must show hidden size 1024, intermediate size 3584, 24 layers, expert width 448, and bias-free text MLP projections.

- [ ] **Step 5: Record the dtype-loader hazard and commit**

Log that later weight loading must avoid relying on `AutoModelForCausalLM(dtype=requested_dtype)` for this composite checkpoint. Require the future loader to use `Qwen3_5ForCausalLM`, verify representative parameter dtypes after load, and measure actual VRAM before conversion.

```powershell
git add src/ppmoe/adapters scripts/probe_qwen35.py tests/test_qwen35_probe.py PROJECT_LOG.md
git commit -m "feat: probe Qwen3.5 text FFN compatibility"
```

---

### Task 10: Final Phase 0 verification and gate decision

**Files:**
- Modify: `PROJECT_LOG.md`
- Modify: this plan's checkbox states as tasks complete.

**Interfaces:**
- Consumes: all prior test suites and generated ignored JSON reports.
- Produces: a durable Phase 0 gate result and explicit Phase 1 next action.

- [ ] **Step 1: Verify repository integrity**

Run:

```powershell
git status --short
git diff --check
uv lock --check
```

Expected: no untracked caches/weights/results, no whitespace errors, and the lock is current.

- [ ] **Step 2: Run the complete test suite with explicit skip reporting**

Run: `uv run pytest -v -ra`

Expected: all CPU and CUDA tests pass; any skip names the unavailable capability precisely.

- [ ] **Step 3: Re-run both Phase 0 reports and the Qwen configuration probe**

Use the exact commands from Tasks 8 and 9. Capture their exit codes, result paths, SHA256 hashes, numerical maxima, package versions, GPU name, driver, and peak allocated CUDA memory in `PROJECT_LOG.md`.

- [ ] **Step 4: Make the gate decision**

Record `M0 PASS` only if permutation, all-expert FFN, two-layer model, gradients, direct/Gram oracle agreement, BF16 measurement, and Qwen configuration inspection all meet their stated criteria. Otherwise record `M0 FAIL`, the first failing invariant, evidence, and the smallest next diagnostic; do not begin Phase 1 weight loading.

- [ ] **Step 5: Commit the verified gate record**

```powershell
git add PROJECT_LOG.md docs/superpowers/plans/2026-08-30-phase0-foundation.md
git commit -m "docs: record Phase 0 gate result"
```

Run: `git status --short --branch`

Expected: clean `phase0-foundation` worktree, with `main` still pointing to `3712046`.
