# Phase 0 Foundation and Phase 1 Readiness Design

## Status

The scope was approved in chat on 2026-08-30. This written specification is awaiting the required review before implementation planning begins.

## Objective

Build the smallest reliable implementation that proves parameter-preserving SwiGLU neuron permutation and expert partitioning, verifies all-expert functional and gradient equivalence, implements an exact fixed-K subset oracle, measures FP32/BF16 numerical behavior, and then inspects Qwen3.5-0.8B-Base compatibility without starting the expensive Phase 1 experiment.

## Scope

### Included

- A Python package managed with `uv` and tested with `pytest`.
- A tiny deterministic SwiGLU FFN and two-layer synthetic Transformer fixture.
- A neuron-triplet abstraction that keeps `gate_proj` row, `up_proj` row, and `down_proj` column indices inseparable.
- Validated neuron permutations and balanced fixed-width partitions.
- A reference partitioned SwiGLU implementation with an all-expert path.
- FP32 output and input/parameter-gradient equivalence tests.
- BF16 drift measurements on CUDA, including BF16 and FP32 expert-output accumulation.
- Exact fixed-K subset enumeration and a Gram-form scorer checked against direct reconstruction.
- A configuration-first Qwen3.5 probe that reports the installed Transformers support, published configuration fields, model class, and FFN projection layout before any conversion.
- An append-only `PROJECT_LOG.md` and machine-readable result artifacts for measurements.

### Excluded

- The 100K-token Phase 1 local oracle run.
- Model-wide Qwen conversion.
- Dataset acquisition, learned routing, distillation, clustering, shared experts, kernel optimization, and performance claims.
- Vision-stack modification.

## Repository layout for this slice

```text
pyproject.toml
PROJECT_LOG.md
src/ppmoe/
  __init__.py
  moefication/
    __init__.py
    neuron.py       # neuron-triplet indices and permutation validation
    partition.py    # deterministic balanced partitions
    reference.py    # dense and partitioned reference SwiGLU modules
    verify.py       # numerical comparison metrics
  oracle/
    __init__.py
    exact.py        # direct and Gram-form fixed-K subset scoring
  adapters/
    __init__.py
    qwen35.py       # configuration/module-layout inspection only
scripts/
  probe_qwen35.py
tests/
  test_neuron_triplet.py
  test_permutation.py
  test_ffn_equivalence.py
  test_gradient_equivalence.py
  test_oracle_bruteforce.py
  test_gram_oracle.py
  test_bf16_drift.py
  test_qwen35_probe.py
```

Each module has one responsibility. The reference implementation is deliberately not optimized; later kernels must be checked against it.

## Core interfaces and invariants

### Neuron permutation

`validate_permutation(permutation, intermediate_size)` accepts a one-dimensional integer permutation containing every index exactly once. Invalid length, duplicates, negative indices, and out-of-range indices raise `ValueError`.

`permute_swiglu_parameters(gate_weight, up_weight, down_weight, permutation, gate_bias=None, up_bias=None)` returns consistently permuted tensors without mutating inputs. The down-projection output bias is not part of a neuron triplet and is therefore never duplicated or permuted.

### Balanced partition

`balanced_partition(permutation, num_experts)` splits the permutation into equal contiguous expert index sets. Phase 0 rejects non-divisible intermediate widths instead of silently creating unequal experts.

### Reference all-expert execution

The dense reference computes:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

The partitioned reference computes each expert from its assigned neuron triplets and sums the expert outputs. A down-projection bias, if present, is added exactly once after the sum. Selecting all experts must preserve the dense function within the dtype-specific tolerance.

### Exact oracle

For expert outputs shaped `[tokens, experts, hidden]`, the direct scorer enumerates every fixed-K subset and minimizes squared reconstruction error to the dense FFN target. The Gram scorer computes the same objective from target/expert dot products and expert-pair dot products. Deterministic lexicographic ordering resolves exact ties.

## Data flow

```text
seeded dense SwiGLU weights
  -> validated permutation
  -> balanced neuron index sets
  -> reference experts
  -> all-expert equivalence and gradient checks
  -> per-expert output capture
  -> direct and Gram exact-oracle comparison
  -> JSON numerical summary
```

Only after those checks pass:

```text
official Qwen model identifier
  -> configuration metadata
  -> Transformers class/module inspection
  -> compatibility report
  -> explicit later decision on weight download
```

## Numerical gates

- FP32 single-FFN all-expert relative error target: at most `1e-5`.
- FP32 permutation equivalence target: at most `1e-5` relative error.
- FP32 dense/partitioned input and parameter gradients: `torch.testing.assert_close` with explicit `rtol=1e-5`, `atol=1e-6` unless empirical evidence requires a documented tighter or looser bound.
- Direct and Gram oracle: identical chosen subset and objective values within FP32 comparison tolerance.
- BF16: log maximum absolute error, RMS error, normalized MSE, and cosine similarity for BF16 summation and FP32 summation followed by BF16 cast. BF16 results are measurements, not a hard exactness claim.
- The Qwen probe must fail clearly if the installed Transformers release lacks Qwen3.5 support or if projection names/shapes differ from assumptions.

## Testing strategy

All production behavior follows red-green-refactor. Each test must first fail for the intended missing behavior. Unit tests use small seeded tensors and real PyTorch operations, not mocked arithmetic. CUDA/BF16 tests skip with a recorded reason only when CUDA or BF16 support is unavailable.

The Qwen test may mock network access only at the boundary; parsing and compatibility validation use a real checked-in minimal configuration fixture. A separate opt-in integration probe may access the official model repository.

## Error handling and harmful-operation controls

- Validate shapes, dtypes, device consistency, permutation integrity, expert count, and fixed-K bounds at public boundaries.
- Never execute arbitrary remote model code (`trust_remote_code=False`).
- Do not download full weights during configuration probing.
- Do not commit Hugging Face caches, model weights, datasets, checkpoints, credentials, or large result arrays.
- Emit actionable errors containing expected and actual shapes or configuration fields.
- Stop and report before expanding scope when a failed equivalence gate indicates the mathematical mapping or implementation is wrong.

## Logging and reproducibility

`PROJECT_LOG.md` records timestamps, branch and commit identifiers, commands, dependency versions, test outcomes, numerical results, failures, mitigations, and the next gate. Generated JSON results include seeds, dtype, device, dimensions, tolerances, and package versions. Log entries never claim a gate passed without a corresponding command result.

## Completion criteria

This slice is complete when:

1. All Phase 0 unit and integration tests pass on CPU FP32.
2. CUDA BF16 measurements complete or a precise environment limitation is recorded.
3. The direct and Gram exact oracles agree across seeded cases and edge cases.
4. The configuration-first Qwen probe produces a compatibility report without downloading model weights.
5. `PROJECT_LOG.md` contains the exact verification evidence and identifies whether Phase 1 weight loading is safe to attempt next.
