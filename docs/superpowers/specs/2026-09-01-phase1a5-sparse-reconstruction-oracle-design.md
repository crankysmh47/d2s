# Phase 1A.5 Sparse Reconstruction Oracle Design

**Date:** 2026-09-01  
**Status:** APPROVED FOR IMPLEMENTATION
**Branch:** `exp`

## 1. Purpose

Phase 1A showed that the frozen random 8x448 expert decomposition is not sufficiently sparse when selected experts retain unit coefficients. At K=4, WikiText perplexity increased from 20.78 to 54.12 and FineWeb-Edu held-out perplexity increased from 16.10 to 39.69 under the rollout oracle.

Phase 1A.5 isolates one alternative explanation before changing expert formation: selected expert outputs may span the dense FFN output but require amplitude adjustment. It measures the best reconstruction attainable by selecting K frozen expert outputs and assigning per-token scalar coefficients. It trains no weights or router and changes no partition.

The experiment answers whether Phase 1A failed because random experts lack necessary output directions or because unit-weight deletion is too restrictive.

## 2. Frozen protocol

Phase 1A remains immutable. Reuse its exact:

- Qwen3.5-0.8B-Base checkpoint and revision;
- deterministic random balanced partition and neuron permutation;
- eight experts of 448 SwiGLU neurons with no shared expert;
- 100,000-token FineWeb-Edu calibration stream and document IDs;
- WikiText-103-raw-v1 test and 100,000-token FineWeb-Edu held-out streams;
- tokenizer, context length, seeds, precision, and evaluation semantics;
- dense teacher, converted model, M0 evidence, and source-stage hashes.

Phase 1A.5 must not recollect data, alter weights, change the partition, increase calibration tokens, introduce a shared expert, or train a router.

Artifacts live under `artifacts/phase1-v0_5/`. They must reference and verify the frozen Phase 1A manifests rather than silently recreating them.

## 3. Three full-compute references

Keep these references separate at every layer:

- **D0:** monolithic dense teacher FFN output.
- **E8:** literal sum of eight independently evaluated expert outputs.
- **R8[condition]:** all-eight-expert coefficient reconstruction under one explicitly named family, rho/constraint, cutoff, and solver version.

E8 is not declared equal to D0. Phase 1A measured a BF16 independent-GEMM numerical floor. Report pairwise D0/E8/R8[condition] error, cosine, and relative error per layer. End-to-end reference evaluation must additionally report logits and PPL where applicable.

Sparse reconstruction error is always measured against D0 locally. E8 and R8[condition] diagnose how much error comes from independent expert execution or solver behavior rather than sparsity.

## 4. Oracle families

The initial local experiment evaluates K in `{4, 6}` only.

### 4.1 Unit

Enumerate every K-subset and fix each selected coefficient to one. This is the exact Phase 1A regression condition. The source manifest must pin Phase 1A local-stage SHA-256 `90382EA91C74B1850FB7AEF81C6F4705DD625A8559EF0FA725BD4511FB2D8341`, rollout-stage SHA-256 `0F415AF16EBFFB6231041605C0A72C4A3505C6DB1F5FF832D6126B8B339E334E`, and M0-stage SHA-256 `B9BE2ECFE2B73A906682B652D65CE1C9DB094E0D23ECC825AB00E677BD3A40D0`. On the 2,048-token smoke, winning unit subsets must match exactly wherever objective ties are absent. If unit metrics are derived only from identical stored Phase 1A sufficient statistics, total squared error, total target norm, and aggregate NMSE must match within relative tolerance `1e-10`. If model outputs are recomputed, the aggregate relative tolerance is `1e-8`; otherwise execution stops as a regression failure. The manifest records which comparison path was used.

### 4.2 Scale-normalized ridge

For candidate subset S, expert-output matrix E_S, Gram matrix G_S, and b_S = E_S^T y, define:

```text
lambda_eff = rho * trace(G_S) / K
```

and solve:

```text
(G_S + lambda_eff I) alpha = b_S + lambda_eff 1
```

Sweep `rho in {10, 1, 0.1, 0.01, 0.001}`. The dimensionless rho is comparable across tokens and layers despite changing activation scale.

Use FP64 Cholesky when `lambda_eff > 0`. Register `epsilon_zero = 1e-15`. If `trace(G_S) <= epsilon_zero * max(||y||^2, 1)` in FP64, define the deterministic ridge result as `alpha = 1`, because every coefficient vector is numerically indistinguishable at the registered scale and the prior is minimized at one. If Cholesky fails despite positive `lambda_eff`, use an FP64 symmetric eigensolve of the same regularized system and require relative linear-system residual `||A alpha - rhs|| / max(||rhs||, 1) <= 1e-9`. A failure of both paths is unresolved. Solver failures, non-finite values, residuals, and fallback counts are durable metrics, never silently discarded.

### 4.3 Bounded

Solve the unregularized objective `min ||y - E_S alpha||^2` subject to `0 <= alpha_i <= 2`. This is a practical coefficient upper bound because it preserves contribution sign and limits amplification. Use a deterministic batched convex box-QP solver with lexicographic subset enumeration. Its KKT certificate uses three separate dimensionless tests: maximum coefficient-bound violation at most `1e-7`; projected-gradient/stationarity residual divided by `max(||b_S||_inf, trace(G_S), 1)` at most `1e-7`; and complementarity residual divided by the same gradient scale at most `1e-7`. The projected-gradient test requires near-zero gradient for interior coefficients, nonnegative gradient at the lower bound, and nonpositive gradient at the upper bound. Equal candidate objectives within `1e-10 * max(||y||^2, 1)` are broken by lexicographically smaller subset. Within one subset, tied minimizers use minimum coefficient L2 norm, followed by lexicographic coordinate minimization if the norm also ties within `1e-12`. Synthetic and sampled cases must agree with exhaustive active-face enumeration to the same objective tolerance.

### 4.4 Registered Free-Reconstruction Candidate Bound

Compute a numerical unconstrained reconstruction bound from the registered candidate pool below. Its loss is an upper bound on the unknown true minimum reconstruction error, equivalently a lower bound on the reconstruction quality attainable by the exact unconstrained optimum. This condition is explicitly non-deployable unless later evidence establishes sane coefficients. Because finite-precision cutoff semantics can discard real low-energy modes, failure of this condition is not proof that no exact span solution exists.

The primary pseudoinverse candidate uses an FP64 symmetric eigendecomposition of G with relative eigenvalue cutoff `tau_G = 1e-12`: retain eigenvalue `lambda_j` iff `lambda_j > tau_G * lambda_max`; a zero matrix has effective rank zero. The direct FP64 SVD cross-check uses the equivalent singular-value rule `sigma_j > sqrt(tau_G) * sigma_max`. Sampled pseudoinverse candidates must agree in effective rank and reconstructed objective within `1e-9 * max(||y||^2, 1)`, independently of which candidate later wins the pool. Because a finite cutoff can discard a real low-energy mode, the authoritative result for each subset is the coefficient vector with the lowest materialized FP64 reconstruction loss among the cutoff pseudoinverse candidate, unit coefficients, and every computed ridge candidate. This remains a valid unconstrained coefficient solution and guarantees the Registered Free-Reconstruction Candidate Bound is no worse than those registered feasible solutions. Objective ties within `1e-9 * max(||y||^2, 1)` select the smaller coefficient L2 norm, then the lexicographically smaller coefficient vector. Persist which candidate source won. The cutoff is immutable after synthetic tests; the result is never described as the exact span optimum.

## 5. Exact subset search

Enumerate every candidate subset: 70 at K=4 and 28 at K=6. Greedy search is prohibited.

Compute full-token G, b, and target norm once. Rank candidates with the quadratic objective, clamp only tiny negative losses attributable to floating-point cancellation, and record larger negative losses as numerical failures. Materialize the winning reconstruction once per token/condition for authoritative metrics and sampled formula-versus-vector validation.

All oracle scoring uses at least FP32. Registered Free-Reconstruction Candidate Bound solves, correctness samples, and authoritative solver cross-checks use FP64. A quadratic-form loss in `[-1e-8 * max(||y||^2, 1), 0)` is clamped to zero; a more negative value is a numerical failure.

## 6. Local experiment and metrics

The execution funnel is:

```text
synthetic tests
  -> 2,048-token smoke
  -> 100,000-token calibration local oracle
  -> pre-registered eligibility gate
  -> frozen rollout conditions
```

For every layer, K, and oracle condition, record:

- squared error and target squared norm;
- aggregate NMSE plus per-token relative error and cosine summaries;
- mean, median, p90, p95, and p99 where per-token distributions are required;
- coefficient mean, standard deviation, p01, p05, median, p95, p99, and maximum absolute value;
- negative coefficient rate, rates above absolute 2 and 4, mean absolute deviation from one, and L2 deviation from one;
- for the bounded family, exact pooled rates `P(alpha < 1e-6)` and `P(alpha > 2 - 1e-6)` to reveal lower/upper-bound saturation;
- effective rank, condition diagnostics, solver/fallback/failure counts;
- expert selection frequencies and co-selection matrices;
- subset Jaccard similarity across coefficient families;
- D0/E8/R8[condition] distances.

Aggregate NMSE means total squared reconstruction error divided by total D0 squared norm, pooled over every calibration token and all 24 converted layers. It is not a mean of layer NMSE values. Coefficient sanity metrics pool every selected coefficient over all calibration tokens and layers for one `(K, condition)`; they are also reported per layer. Negative rate is the exact count of coefficients below zero divided by the pooled count. The practical p99 rule uses the equivalent exact gate `count(|alpha| > 2) / count(alpha) <= 0.01`; the displayed empirical nearest-rank p99 is generated by deterministic sorted disk chunks and must agree with that gate at the boundary.

Streaming sufficient statistics and bounded-memory sorted chunks are required. Production execution must not retain token x layer x subset tensors after each processed microbatch.

## 7. Pre-registered local eligibility

Eligibility is determined only from the frozen FineWeb-Edu calibration stream. WikiText and FineWeb-heldout results cannot select K, rho, solver settings, or coefficient family.

For each of K=4 and K=6:

1. A candidate must reduce aggregate local NMSE by at least 30% relative to the corresponding unit oracle.
2. A practical candidate must additionally have:
   - empirical p99 absolute coefficient at most 2, enforced by an exact above-2 rate of at most 1%;
   - negative coefficient rate at most 5%; for bounded `[0,2]`, this is satisfied by construction and bound-saturation rates replace it as the primary sign/limit diagnostic;
   - finite metrics;
   - zero unresolved numerical failures. A rank-deficient pseudoinverse candidate is diagnostic metadata, not automatically a failure. An unresolved failure is any non-finite input/output, violated bounded KKT certificate, pseudoinverse/direct-reference disagreement, objective-order violation, or solver failure remaining after its pre-registered fallback.
3. Among eligible practical candidates, select the lowest aggregate NMSE. Resolve exact ties by smaller mean absolute deviation from one, then by stronger regularization.
4. The Registered Free-Reconstruction Candidate Bound advances as a diagnostic whenever it reaches the 30% NMSE-reduction threshold, regardless of coefficient sanity.

If no practical candidate qualifies, record that outcome; do not relax thresholds. A qualifying Registered Free-Reconstruction Candidate Bound advances to rollout even when no practical condition qualifies. If neither a practical condition nor this bound qualifies for either K, stop Phase 1A.5 after local reporting and proceed to Phase 1B.

The 2,048-token smoke must benchmark each family separately. Projected full-local runtime is `smoke elapsed seconds * 100000 / 2048`. The 100K bounded run is permitted only when projected runtime is at most two GPU-hours, measured peak CUDA allocation is at most 12 GiB, there are zero solver failures, and every sampled bounded result has a valid KKT certificate. Otherwise bounded is reported as engineering-infeasible under V0.5 and is not replaced by a weaker solver without a new reviewed spec.

For analysis only, report free-reconstruction gain recovery for every practical condition `c`:

```text
R_c = (L_unit - L_c) / (L_unit - L_free_bound)
```

where losses are the pooled aggregate local squared errors for the same K. If the denominator is non-positive within the registered objective tolerance, report the metric as undefined. This metric never affects eligibility.

## 8. Rollout reconstruction target

Rollout uses Reconstruction-H as its primary objective. At layer l, let the current student follow its normal non-FFN path to a residual baseline state h_base,l+1^S before the candidate FFN correction is added. The target correction is:

```text
delta_l = h_l+1^T - h_base,l+1^S
```

Evaluate the current student expert outputs E_i(x_l^S) and choose the subset and coefficients minimizing:

```text
||delta_l - sum(alpha_i E_i(x_l^S))||^2
```

Then propagate that selected reconstruction through subsequent layers. Reconstructing the teacher FFN output on the drifted student state may be retained as a diagnostic but cannot select the primary result.

## 9. Frozen rollout conditions

After local calibration selection, freeze and persist the exact condition IDs before opening external evaluation results. For each K with an eligible practical or numerical-upper-bound condition, rollout evaluates only:

- the unit oracle baseline;
- the single selected practical condition, if one exists;
- the Registered Free-Reconstruction Candidate Bound, if it passed its local gate.

If the practical winner is the bounded oracle, persist its solver configuration. If it is ridge, persist rho and every numerical setting. Evaluation code must refuse an unregistered condition.

Report WikiText and FineWeb-heldout PPL, PPL delta, PPL ratio relative to the same dense teacher, scored/failed batch counts, and layerwise hidden-state divergence. Preserve the Phase 1A unit results as regression evidence rather than recomputing them unless the new implementation's regression gate requires a controlled rerun.

## 10. Correctness and failure handling

Required tests before real execution:

1. Unit mode reproduces the Phase 1A exact subset oracle.
2. D0, E8, and condition-indexed R8 remain separately labeled and measured.
3. Gram-form candidate loss agrees with materialized-vector loss.
4. The authoritative Registered Free-Reconstruction Candidate Bound objective is no worse than any registered ridge or unit coefficient vector for the same subset by more than `1e-9 * max(||y||^2, 1)`.
5. Bounded optimum satisfies all bounds and agrees with brute-force active-set references on synthetic cases.
6. Scale-normalized ridge is invariant to a common scaling of E and y, within tolerance.
7. Pseudoinverse-candidate effective rank and cutoff are deterministic and serialized.
8. Sampled FP64 pseudoinverse candidates agree with direct QR/SVD reference loss independently of the Registered Free-Reconstruction Candidate Bound winner.
9. Reconstruction-H uses the teacher-next-state correction after student drift.
10. Failed microbatches roll back all statistics and resume without double counting.
11. Calibration selection is persisted before evaluation and evaluation cannot alter it.
12. Stage signatures include configuration, frozen source hashes, solver version, and code revision so superseded checkpoints cannot be reused silently.

Every stage uses atomic checkpoints, exact completed/failed counts, finite-value guards, and signature isolation. OOM or numerical failure is preserved as evidence; microbatch reduction may resume only through a new signature.

## 11. Artifacts and reporting

Required structure:

```text
artifacts/phase1-v0_5/
  config.json
  source-manifest.json
  selection-manifest.json
  stages/
  checkpoints/
  plot-source/
  plots/
  report-manifest.json
  report.md
```

Plot sources are authoritative CSV/JSON files with SHA-256 hashes. Required views include local error versus rho, local error versus active fraction, evaluation PPL, coefficient distributions, coefficient distortion versus gain, conditioning/effective rank versus gain, layerwise improvement, subset stability, and D0/E8/R8[condition] distances.

The primary table contains K, family, rho or constraint, active fraction, local NMSE, relative improvement, WikiText/FineWeb PPL and ratios, coefficient sanity metrics, effective-rank/conditioning failures, and deployment-status label.

## 12. Decision outcomes

- **Registered Free-Reconstruction Candidate Bound fails:** no registered numerical reconstruction condition qualified. This does not prove that no exact span solution exists, but it provides no actionable evidence for coefficient-aware routing under the pre-registered solver. Proceed directly to Phase 1B.
- **Registered Free-Reconstruction Candidate Bound succeeds but practical conditions fail:** registered numerical reconstruction redundancy exists but is not credibly exploitable. Proceed to Phase 1B and preserve the bound without claiming an exact span optimum.
- **A practical condition succeeds at K=4:** investigate coefficient-aware routing that predicts both subset and amplitudes before changing the partition.
- **Only K=6 succeeds:** retain 75%-active FFN as the supported fallback frontier while Phase 1B attempts to move the knee toward 50%.

No outcome may claim a practical router, inference speedup, coefficient predictability, optimal partition, or transfer to Qwen3.8-27B.

## 13. Implementation boundary

Phase 1A.5 ends with the frozen local selection, gated rollout, hashed report, and a documented decision. It does not implement router training, expert reformation, shared experts, larger calibration corpora, or variable K.
