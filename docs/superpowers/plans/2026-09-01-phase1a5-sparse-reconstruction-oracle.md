# Phase 1A.5 Sparse Reconstruction Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether per-token coefficients let K=4 or K=6 experts from the frozen Phase 1A random partition reconstruct the dense Qwen FFN well enough to change the oracle frontier.

**Architecture:** Add a separate `phase1a5` experiment pipeline that verifies and consumes the immutable Phase 1A artifacts. Put coefficient solvers in a pure tensor module, local and rollout accumulation in separate checkpointed modules, and selection/reporting behind calibration-only manifests so evaluation cannot influence condition choice.

**Tech Stack:** Python 3.12, PyTorch/CUDA, Transformers, NumPy, Matplotlib, pytest, TOML, JSON/CSV, uv.

**Spec:** `docs/superpowers/specs/2026-09-01-phase1a5-sparse-reconstruction-oracle-design.md`

## Global Constraints

- Work only on `exp`; do not modify `main` until a reviewed major checkpoint is ready.
- Freeze Qwen3.5-0.8B-Base revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, the 8x448 partition, datasets, token manifests, seeds, BF16 execution, and Phase 1A evaluation semantics.
- Pin Phase 1A hashes: M0 `B9BE2ECFE2B73A906682B652D65CE1C9DB094E0D23ECC825AB00E677BD3A40D0`, local `90382EA91C74B1850FB7AEF81C6F4705DD625A8559EF0FA725BD4511FB2D8341`, rollout `0F415AF16EBFFB6231041605C0A72C4A3505C6DB1F5FF832D6126B8B339E334E`.
- Initial K values are exactly `(4, 6)` and ridge rho values exactly `(10, 1, 0.1, 0.01, 0.001)`.
- Registered constants: `tau_G=1e-12`, `epsilon_zero=1e-15`, bounded coefficient interval `[0,2]`, KKT tolerances `1e-7`, local improvement gate 30%, practical above-2 rate at most 1%, negative rate at most 5%.
- Select conditions only from the frozen FineWeb-Edu calibration stream and persist selection before opening WikiText or FineWeb-heldout evaluation.
- Use atomic checkpoints and signatures containing config, source hashes, solver version, code revision, token manifest, model identity, and microbatch size.
- Preserve every negative result, solver failure, OOM, infeasible timing result, and superseded checkpoint in `PROJECT_LOG.md`.
- Respect the user's reduced commit cadence: each task below is a major independently reviewable checkpoint, not a commit per tiny step.

---

### Task 1: Locked Phase 1A.5 configuration and source provenance

**Files:**
- Create: `configs/phase1-v0_5.toml`
- Create: `src/ppmoe/experiments/phase1a5_config.py`
- Create: `src/ppmoe/experiments/phase1a5_provenance.py`
- Create: `tests/test_phase1a5_config.py`
- Create: `tests/test_phase1a5_provenance.py`

**Interfaces:**
- Produces: `Phase1A5Config.load(path: Path) -> Phase1A5Config`
- Produces: `verify_phase1a_sources(source_root: Path, output_root: Path, config: Phase1A5Config) -> Mapping[str, object]`
- Produces: `phase1a5_signature(config, source_manifest, solver_version, code_revision, microbatch_size) -> str`

- [ ] **Step 1: Write locked-config failure tests**

```python
def test_phase1a5_config_locks_oracle_grid_and_numerics(tmp_path):
    config = Phase1A5Config.load(CONFIG_PATH)
    assert config.k_values == (4, 6)
    assert config.rho_values == (10.0, 1.0, 0.1, 0.01, 0.001)
    assert config.tau_g == 1e-12
    assert config.epsilon_zero == 1e-15
    assert config.coefficient_bounds == (0.0, 2.0)
    assert config.local_improvement_gate == 0.30

@pytest.mark.parametrize("field,value", [("tau_g", 1e-8), ("k_values", [4, 5, 6])])
def test_phase1a5_config_rejects_protocol_drift(tmp_path, field, value):
    path = mutate_toml(CONFIG_PATH, tmp_path, field, value)
    with pytest.raises(ValueError, match="locked Phase 1A.5"):
        Phase1A5Config.load(path)
```

- [ ] **Step 2: Run config tests RED**

Run: `uv run pytest -q tests/test_phase1a5_config.py`

Expected: import failure because `phase1a5_config.py` does not exist.

- [ ] **Step 3: Implement the frozen config dataclass and TOML**

```python
@dataclass(frozen=True, slots=True)
class Phase1A5Config:
    name: str
    artifact_root: Path
    source_artifact_root: Path
    k_values: tuple[int, ...]
    rho_values: tuple[float, ...]
    tau_g: float
    epsilon_zero: float
    coefficient_bounds: tuple[float, float]
    local_improvement_gate: float
    practical_negative_rate_max: float
    practical_above_two_rate_max: float
    bounded_runtime_seconds_max: float
    bounded_peak_cuda_bytes_max: int
    solver_version: str
```

Validate every value against the spec rather than merely parsing it. Reuse `Phase1Config.load(configs/phase1-v0.toml)` for the frozen model/data protocol instead of duplicating those fields.

- [ ] **Step 4: Write source-hash and mutation tests**

```python
def test_source_verifier_pins_all_phase1a_stage_hashes(tmp_path):
    copy_required_phase1_files(PHASE1_ROOT, tmp_path)
    manifest = verify_phase1a_sources(tmp_path, tmp_path / "v0_5", CONFIG)
    assert manifest["stage_sha256"] == EXPECTED_STAGE_HASHES

def test_source_verifier_rejects_mutated_phase1a_stage(tmp_path):
    copy_required_phase1_files(PHASE1_ROOT, tmp_path)
    with (tmp_path / "stages/local.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(SourceVerificationError, match="local SHA-256"):
        verify_phase1a_sources(tmp_path, tmp_path / "v0_5", CONFIG)
```

- [ ] **Step 5: Implement provenance verification and atomic source manifest**

The manifest must include absolute source root, all three pinned hashes, dataset manifest hashes, conversion/model manifest hashes, current Git commit, dirty-state boolean, solver version, and creation timestamp. Refuse a dirty source tree for production stages; tests may inject a clean code-revision provider.

- [ ] **Step 6: Verify Task 1**

Run: `uv run pytest -q tests/test_phase1a5_config.py tests/test_phase1a5_provenance.py tests/test_phase1_config.py`

Expected: all pass.

- [ ] **Step 7: Major checkpoint commit**

Stage only Task 1 files and commit `feat: lock phase 1a5 protocol provenance`.

---

### Task 2: Exact coefficient reconstruction solver core

**Files:**
- Create: `src/ppmoe/oracle/reconstruction.py`
- Create: `tests/test_reconstruction_oracle.py`

**Interfaces:**
- Produces: `ReconstructionCondition(family: Literal["unit", "ridge", "bounded", "free_bound"], rho: float | None)`
- Produces: `ReconstructionResult(subsets, coefficients, squared_error, effective_rank, solver_source, diagnostics)`
- Produces: `enumerate_subsets(num_experts: int, k: int, device: torch.device) -> Tensor`
- Produces: `score_reconstruction(target: Tensor, experts: Tensor, k: int, condition: ReconstructionCondition, settings: SolverSettings) -> ReconstructionResult`

- [ ] **Step 1: Write RED tests for unit and quadratic scoring**

```python
def test_unit_reconstruction_matches_existing_exact_oracle():
    target, experts = deterministic_case(tokens=7, experts=8, hidden=11)
    old = exact_gram_oracle(target, experts, 4)
    new = score_reconstruction(target, experts, 4, UNIT, SETTINGS)
    assert torch.equal(new.subsets, old.subsets)
    assert torch.allclose(new.squared_error, materialized_loss(target, experts, new), rtol=1e-8)

def test_scale_normalized_ridge_is_invariant_to_common_scale():
    first = score_reconstruction(target, experts, 4, ridge(0.1), SETTINGS)
    second = score_reconstruction(target * 17, experts * 17, 4, ridge(0.1), SETTINGS)
    assert torch.allclose(first.coefficients, second.coefficients, rtol=1e-9, atol=1e-10)
```

- [ ] **Step 2: Run solver tests RED**

Run: `uv run pytest -q tests/test_reconstruction_oracle.py`

Expected: missing reconstruction module.

- [ ] **Step 3: Implement validation, subset cache, G/b calculation, and unit mode**

Use lexicographically ordered `itertools.combinations`. Validate `[tokens, hidden]` target and `[tokens, 8, hidden]` finite expert tensors on one device. Calculate FP32 or FP64 G/b once and rank every subset without repeatedly materializing hidden vectors.

- [ ] **Step 4: Implement scale-normalized ridge with registered degeneracy behavior**

```python
trace = torch.diagonal(gs, dim1=-2, dim2=-1).sum(-1)
zero = trace <= settings.epsilon_zero * target_norm.clamp_min(1.0)
lambda_eff = condition.rho * trace / k
alpha[zero] = 1.0
alpha[~zero] = torch.cholesky_solve(rhs[~zero, :, None], chol).squeeze(-1)
```

Persist Cholesky fallback count and relative system residual; do not silently convert failed tokens.

- [ ] **Step 5: Write and run RED tests for pseudoinverse cutoff and candidate-bound semantics**

Test effective rank zero, a deliberately rank-deficient matrix, SVD-equivalent cutoff, candidate source persistence, and `free_bound.loss <= unit/ridge loss + tolerance` for every subset.

- [ ] **Step 6: Implement the Registered Free-Reconstruction Candidate Bound**

Use FP64 `torch.linalg.eigh`, `tau_G=1e-12`, and the registered candidate pool. Materialize FP64 losses for pseudoinverse, unit, and all ridge candidates; choose minimum loss, then minimum coefficient norm, then lexicographic coefficient vector. Expose the primary pseudoinverse result separately for sampled SVD validation.

- [ ] **Step 7: Implement direct SVD sample verification**

```python
u, s, vh = torch.linalg.svd(es.double(), full_matrices=False)
keep = s > settings.tau_g**0.5 * s.max()
alpha = vh[keep].T @ ((u[:, keep].T @ y.double()) / s[keep])
```

Require rank equality and relative objective agreement `<=1e-9`.

- [ ] **Step 8: Verify Task 2**

Run: `uv run pytest -q tests/test_reconstruction_oracle.py tests/test_gram_oracle.py tests/test_oracle_bruteforce.py`

Expected: all pass on CPU and available CUDA parametrizations.

- [ ] **Step 9: Major checkpoint commit**

Commit `feat: add coefficient reconstruction solvers`.

---

### Task 3: Certified bounded box-QP and smoke feasibility benchmark

**Files:**
- Create: `src/ppmoe/oracle/box_qp.py`
- Modify: `src/ppmoe/oracle/reconstruction.py`
- Create: `tests/test_box_qp.py`
- Modify: `tests/test_reconstruction_oracle.py`

**Interfaces:**
- Produces: `solve_box_qp(gs: Tensor, bs: Tensor, lower: float, upper: float, settings: BoxQPSettings) -> BoxQPSolution`
- Produces: `BoxQPSolution(coefficients, objective, primal_residual, stationarity_residual, complementarity_residual, iterations, certified)`

- [ ] **Step 1: Write exhaustive-reference tests**

Generate deterministic K=2..6 positive-semidefinite systems, including rank deficiency and optima on lower/upper/interior faces. Compare the batched solver against exhaustive active-face enumeration.

- [ ] **Step 2: Run RED box-QP tests**

Run: `uv run pytest -q tests/test_box_qp.py`

Expected: missing module.

- [ ] **Step 3: Implement deterministic batched active-set box-QP**

Initialize from clamped unit coefficients. At each iteration, solve the free-variable system, clamp the first lexicographically violating coordinate, release the first bound coordinate violating the projected-gradient sign, and stop only when all registered KKT tests pass. Cap iterations at 64 and return `certified=False` instead of accepting an approximate solution.

- [ ] **Step 4: Implement canonical tied minimizer behavior**

For rank-deficient free-variable systems, use minimum-L2 pseudoinverse solution. Apply objective tolerance `1e-10 * max(||y||^2,1)`, then norm tolerance `1e-12`, then lexicographic coordinates.

- [ ] **Step 5: Integrate bounded mode and saturation diagnostics**

Record exact counts for `alpha < 1e-6` and `alpha > 2 - 1e-6`, coefficient bound violations, KKT residual maxima, iterations, and failures.

- [ ] **Step 6: Add a benchmark result type and feasibility calculation**

```python
@dataclass(frozen=True, slots=True)
class FamilyBenchmark:
    elapsed_seconds: float
    peak_cuda_bytes: int
    processed_tokens: int
    projected_full_seconds: float
    feasible: bool
```

`feasible` is true only for projected runtime <=7200 seconds, peak CUDA <=12 GiB, zero failures, and all sampled solutions certified.

- [ ] **Step 7: Verify Task 3**

Run: `uv run pytest -q tests/test_box_qp.py tests/test_reconstruction_oracle.py`

Expected: all pass; bounded objective never exceeds unit objective for the same subset beyond tolerance.

- [ ] **Step 8: Major checkpoint commit**

Commit `feat: add certified bounded reconstruction`.

---

### Task 4: Checkpointed local reconstruction experiment and calibration selection

**Files:**
- Create: `src/ppmoe/oracle/reconstruction_local.py`
- Create: `src/ppmoe/experiments/phase1a5_selection.py`
- Create: `tests/test_reconstruction_local.py`
- Create: `tests/test_phase1a5_selection.py`

**Interfaces:**
- Produces: `run_reconstruction_local(teacher, student, batches, conditions, checkpoint_dir, run_signature) -> ReconstructionLocalResult`
- Produces: `select_calibration_conditions(result: ReconstructionLocalResult, config: Phase1A5Config, output_path: Path) -> SelectionManifest`

- [ ] **Step 1: Write transactional accumulator tests**

Require atomic completed batches, rollback of coefficient/reference/quantile deltas after an injected layer failure, exact resume counts, signature rejection, and deterministic sorted coefficient chunks.

- [ ] **Step 2: Write D0/E8/R8 reference tests**

Use a synthetic BF16 case where independent expert summation differs from a monolithic projection. Assert D0/E8/R8[condition] are separately named and none is overwritten with another reference.

- [ ] **Step 3: Implement streaming local hooks and accumulator**

Reuse `iter_text_mlps`, teacher pre-MLP inputs, and `PartitionedQwen35MLP.expert_outputs`. Store pooled total SSE/total D0 norm, per-layer statistics, selection/co-selection, solver diagnostics, coefficient counters, and bounded saturation counters. Write deterministic sorted coefficient chunks per completed batch and merge them only during finalization.

- [ ] **Step 4: Implement exact pooled metrics and gain recovery**

```python
aggregate_nmse = total_squared_error / total_d0_squared_norm
relative_improvement = (unit_sse - candidate_sse) / unit_sse
gain_recovery = (unit_sse - candidate_sse) / (unit_sse - free_bound_sse)
```

Return `gain_recovery=None` when the denominator is non-positive within objective tolerance. Never use gain recovery for eligibility.

- [ ] **Step 5: Write calibration selection RED tests**

Cover no eligible condition, ridge winner, bounded winner with constructed zero negative rate and saturation diagnostics, free-bound-only advancement, tie-breaking by coefficient deviation then stronger rho, and refusal to read evaluation artifacts.

- [ ] **Step 6: Implement and atomically freeze `selection-manifest.json`**

Require >=30% pooled-NMSE reduction. Practical candidates require above-2 rate <=1%, negative rate <=5% except bounded by construction, finite metrics, and zero unresolved failures. Persist selected condition IDs, all rejected reasons, source/local hashes, and a `selection_frozen_at` timestamp. Refuse overwrite with different content.

- [ ] **Step 7: Verify Task 4**

Run: `uv run pytest -q tests/test_reconstruction_local.py tests/test_phase1a5_selection.py tests/test_local_oracle.py`

Expected: all pass.

- [ ] **Step 8: Major checkpoint commit**

Commit `feat: add phase 1a5 local selection gate`.

---

### Task 5: Gated Phase 1A.5 orchestration and CLI

**Files:**
- Create: `src/ppmoe/experiments/phase1a5.py`
- Create: `scripts/run_phase1a5.py`
- Create: `tests/test_phase1a5_orchestration.py`
- Create: `tests/test_phase1a5_cli.py`

**Interfaces:**
- Produces: `run_phase1a5(config, stage, resume, artifact_root, microbatch_size, dependencies=None) -> Mapping[str, object]`
- Stages: `source`, `smoke`, `local`, `select`, `rollout`, `report`

- [ ] **Step 1: Write stage-order and signature RED tests**

Assert `smoke` requires verified source, `local` requires feasible smoke, `select` requires complete local, rollout requires immutable selection, and report requires complete or scientifically stopped stages. Assert code revision and microbatch changes isolate checkpoint paths.

- [ ] **Step 2: Implement source/smoke/local/select stages**

Use Phase 1A token tensors and converted model loader. The smoke runs exactly 2,048 valid calibration tokens for all families and K values, records per-family wall time/peak CUDA, performs exact untied unit-subset regression, and enforces bounded feasibility. The local stage runs exactly 100,000 tokens and omits bounded only if smoke declared it infeasible.

- [ ] **Step 3: Implement CLI with explicit artifact roots**

```text
uv run python scripts/run_phase1a5.py --stage source --artifact-root artifacts/phase1-v0_5
uv run python scripts/run_phase1a5.py --stage smoke --artifact-root artifacts/phase1-v0_5 --microbatch-size 1
```

Catch only registered stage/config/OS errors and return nonzero without deleting checkpoints.

- [ ] **Step 4: Verify Task 5**

Run: `uv run pytest -q tests/test_phase1a5_orchestration.py tests/test_phase1a5_cli.py tests/test_phase1_orchestration.py`

Expected: all pass.

- [ ] **Step 5: Major checkpoint commit**

Commit `feat: orchestrate phase 1a5 local experiment`.

---

### Task 6: Reconstruction-H rollout with frozen coefficient conditions

**Files:**
- Create: `src/ppmoe/oracle/reconstruction_rollout.py`
- Create: `tests/test_reconstruction_rollout.py`
- Modify: `src/ppmoe/experiments/phase1a5.py`
- Modify: `tests/test_phase1a5_orchestration.py`

**Interfaces:**
- Produces: `select_reconstruction_h(target_next, residual_baseline, expert_outputs, condition, k, settings) -> ReconstructionResult`
- Produces: `run_reconstruction_rollout(teacher, student, batches, condition, k, checkpoint_dir, run_signature) -> ReconstructionRolloutResult`

- [ ] **Step 1: Write Reconstruction-H semantic RED test**

Construct teacher-next, student residual baseline, and experts so matching teacher FFN output chooses a different subset from matching `teacher_next - residual_baseline`. Assert the implementation chooses the latter.

- [ ] **Step 2: Write checkpoint and frozen-selection tests**

Assert full batch rollback, resume, scored-token/NLL integrity, layer coverage, peak CUDA persistence, and rejection of any condition not byte-for-byte present in `selection-manifest.json`.

- [ ] **Step 3: Implement rollout hooks by adapting the proven Phase 1A flow**

Capture teacher decoder next states. Capture the student's post-non-FFN residual baseline at each decoder layer before adding the MLP output. Evaluate student experts on its current normalized state, solve the frozen condition against `teacher_next - residual_baseline`, inject only the selected weighted reconstruction, and continue the student trajectory.

- [ ] **Step 4: Persist rollout metrics**

Record PPL/NLL/scored tokens, failed/completed batches, layer hidden SSE/norm/count, coefficient diagnostics, selection counts, D0/E8/R8[condition] distances where defined, condition ID, and peak CUDA. Require every expected layer and nonzero scored tokens.

- [ ] **Step 5: Wire gated corpora and conditions**

For each eligible K, evaluate exactly unit, the one frozen practical winner if present, and the Registered Free-Reconstruction Candidate Bound if eligible, on WikiText and FineWeb-heldout. Do not expose a CLI option that can add an unselected rho after selection.

- [ ] **Step 6: Verify Task 6**

Run: `uv run pytest -q tests/test_reconstruction_rollout.py tests/test_phase1a5_orchestration.py tests/test_rollout_oracle.py`

Expected: all pass.

- [ ] **Step 7: Major checkpoint commit**

Commit `feat: add reconstruction-h rollout oracle`.

---

### Task 7: Hashed report, plots, decision record, and production execution

**Files:**
- Create: `src/ppmoe/reporting/phase1a5.py`
- Create: `tests/test_phase1a5_reporting.py`
- Modify: `src/ppmoe/experiments/phase1a5.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Produces: `render_phase1a5_report(artifact_root: Path, config: Phase1A5Config) -> Phase1A5ReportManifest`

- [ ] **Step 1: Write report validation RED tests**

Reject source-hash drift, missing local conditions, mutable selection, evaluation condition leakage, non-finite metrics, bad token/layer coverage, failed batches, invalid bounded KKT state, and output-hash mismatch.

- [ ] **Step 2: Implement authoritative plot sources and report manifest**

Write CSV/JSON sources before plots. Include local NMSE/rho, active fraction, PPL, coefficient distributions, bound saturation, coefficient distortion versus gain, gain recovery, rank/conditioning, layer improvements, subset stability, and D0/E8/R8[condition] distances. Hash every source, PNG, and source stage.

- [ ] **Step 3: Implement the decision matrix without overclaiming**

Generate one of: no registered numerical reconstruction qualifies; registered free bound only; practical ridge qualifies; practical bounded qualifies; K=6-only frontier; K=4 practical frontier. Explicitly prohibit exact-span, trained-router, or speedup claims.

- [ ] **Step 4: Run complete test and integrity verification**

Run:

```text
uv run pytest -q
uv lock --check
git diff --check
```

Expected: all tests pass, 85-package lock remains valid, no whitespace errors.

- [ ] **Step 5: Execute source and 2,048-token smoke**

Run source then smoke once. Record exact command, signature, elapsed time per family, peak CUDA, completed/failed counts, unit regression, solver/KKT/SVD evidence, and bounded feasibility in `PROJECT_LOG.md`. Stop rather than weaken the method if a gate fails.

- [ ] **Step 6: Execute 100K calibration and freeze selection**

Only after smoke passes, run local then select. Record pooled NMSE, improvement, coefficient sanity, saturation, gain recovery, failures, and the immutable selection-manifest hash. Do not inspect evaluation PPL beforehand.

- [ ] **Step 7: Execute gated rollout and report**

Only if selection permits, run rollout for the registered conditions, then report. Record exact PPL ratios, hidden errors, D0/E8/R8 distances, hashes, failures, and scientific decision in `PROJECT_LOG.md`.

- [ ] **Step 8: Fresh review and final major checkpoint**

Request a read-only correctness/methodology review, resolve blockers, rerun the full verification suite, commit `feat: complete phase 1a5 reconstruction oracle`, push `exp`, and fast-forward `main` only if all required stages and the report pass.

---

### Task 8: Evidence scalability and report-correctness remediation

**Files:**
- Modify: `src/ppmoe/oracle/reconstruction_local.py`
- Modify: `src/ppmoe/experiments/phase1a5.py`
- Modify: `src/ppmoe/reporting/phase1a5.py`
- Modify: `tests/test_reconstruction_local.py`
- Modify: `tests/test_phase1a5_orchestration.py`
- Modify: `tests/test_phase1a5_reporting.py`

**Interfaces:**
- Preserves all Task 1-7 public result/stage interfaces.
- Produces exact empirical quantiles by k-way streaming merge of sorted checkpoint chunks with memory bounded by chunk count and execution microbatch size.
- Produces hashed `solver-diagnostics.csv/json` rows for every pooled/per-layer condition and R8 reference.

- [ ] **Step 1: Add failing scalability and zero-target tests**

Monkeypatch chunk iterators with sentinels that fail if more than one distribution is materialized or if all chunk values are concatenated. Require exact nearest-rank quantiles from a multi-chunk fixture. Add a zero-D0 token fixture asserting `relative_error_undefined_count` increments, the value is excluded from relative-error quantiles, and checkpoint rollback/resume preserves counts.

- [ ] **Step 2: Implement externally streamed exact quantiles**

Read each already-sorted chunk lazily and use a heap-based k-way merge to capture only requested nearest-rank positions. Finalize coefficient, absolute-coefficient, relative-error, cosine, and conditioning distributions sequentially so no five full lists coexist. Memory must be `O(number_of_chunks * execution_microbatch_chunk_size)`, never `O(total_tokens)`.

- [ ] **Step 3: Reject stale smoke resume evidence**

Before any early `smoke --resume` return, verify current source/config/code/microbatch signature and the full SVD evidence contract: exact K4/K6/R8 identities, 384 samples each, 16 samples per 24 layers, literal boolean rank agreement, and finite maximum relative objective error at most `1e-9`. Add a fixture representing the existing pre-SVD smoke and require rejection rather than reuse.

- [ ] **Step 4: Emit complete solver evidence**

Populate one unique solver row for every pooled and per-layer `(K, condition)` plus every R8 condition. Include KKT, certification, solver-source, rank, conditioning, fallback, face-enumeration, numerical-failure, and SVD fields where applicable. Validate rows against local results and write/hash both CSV and JSON sources.

- [ ] **Step 5: Correct report mappings**

Use `above_two_count / coefficient_count` for above-two rate. Plot coefficient mean absolute deviation from one against registered local relative improvement or gain recovery. Plot conditioning p99 against reconstruction gain; emit effective rank in a separate evidence-backed panel/source. Emit bounded saturation rows only for the bounded family. Reject missing inputs instead of substituting zero or synthetic values.

- [ ] **Step 6: Verify Task 8**

Run focused local/orchestration/reporting tests, then `uv run pytest -q`, `uv lock --check`, and `git diff --check`. Expected: all pass with no production command executed.

- [ ] **Step 7: Review and production gate**

Commit `fix: complete phase1a5 evidence contract`, obtain a fresh scoped review, and only then run one new clean source + 2,048-token smoke attempt. Preserve every earlier source/smoke/failure artifact as superseded evidence. Do not run 100K local until the new smoke passes all runtime, unit, bounded, and SVD gates.

---

### Task 9: Streamable evidence storage and strict provenance/schema closure

**Files:**
- Modify: `src/ppmoe/oracle/reconstruction_local.py`
- Modify: `src/ppmoe/experiments/phase1a5.py`
- Modify: `src/ppmoe/reporting/phase1a5.py`
- Modify: `tests/test_reconstruction_local.py`
- Modify: `tests/test_phase1a5_orchestration.py`
- Modify: `tests/test_phase1a5_reporting.py`

**Interfaces:**
- Replaces array-valued distribution chunk payloads with an atomically committed manifest plus concatenated little-endian FP64 binary data whose offset/count records permit fixed-buffer sequential readers.
- Keeps exact nearest-rank quantiles and all frozen scientific semantics unchanged.
- Makes every resumed smoke depend on freshly verified current source/code/config/token/microbatch provenance.
- Defines one explicit, condition-aware flat solver-evidence schema shared by validation, JSON, and CSV.

- [ ] **Step 1: Add adversarial RED tests against real persisted chunks**

Create multi-batch checkpoint files through the production writer, instrument reads so loading an array-valued distribution or reading an entire binary payload fails, and prove exact pooled/per-layer nearest-rank quantiles. Add rollback-after-zero-D0 coverage. Add smoke-resume fixtures for code/source/config/token/microbatch drift and malformed SVD evidence. Add solver-row field-completeness tests and a required effective-rank plot assertion.

- [ ] **Step 2: Introduce a streamable atomic chunk format**

For each condition and completed batch, write a small JSON manifest and one concatenated binary FP64 data file. Each `(layer, distribution)` manifest entry records byte offset, count, ordering, and finite min/max; data are sorted and encoded little-endian. Commit both before checkpoint state advances, delete both on rollback, reject missing/malformed/truncated/trailing data, and preserve orphan-chunk safety. Readers retain at most one fixed-size buffer/file handle and heap head per batch chunk; they must never call `read_text`, `read_bytes`, `json.loads` on distribution arrays, or materialize a complete distribution.

- [ ] **Step 3: Preserve exact quantile and checkpoint semantics**

Use the existing k-way heap over fixed-buffer binary iterators, validate manifest counts against accumulated counts, and keep exact nearest-rank `ceil(p*n)` results. Preserve pending in-memory data only for the current execution microbatch. Ensure save/load/resume and transaction rollback cover manifest/data pairs and zero-target defined/undefined counts.

- [ ] **Step 4: Bind smoke resume to live provenance**

Before any immutable-artifact early return, freshly run the same clean-repository/source verification used by the `source` stage and recompute the complete smoke input signature from current source hashes, config, token identity, code revision, and microbatch size. Resume only if it equals the stored signature and the full SVD predicate passes; otherwise reject without overwriting prior evidence.

- [ ] **Step 5: Close the solver-evidence schema**

Define explicit common and condition-specific row fields. Common rows include counts, numerical failures, solver-source counts, effective-rank/conditioning diagnostics, and coverage. Bounded rows additionally require certification/KKT residuals, active-bound/saturation, fallback, and face-enumeration fields. Ridge/free rows require their registered numerical diagnostics; free-bound sparse and R8 rows embed the applicable SVD identity/count/per-layer coverage/rank-agreement/max-error fields. Reject missing, non-finite, inconsistent, duplicate, or inapplicable synthetic evidence. Emit the exact validated flat rows to both hashed JSON and CSV.

- [ ] **Step 6: Add the effective-rank panel**

Derive a numeric mean effective rank from each persisted rank histogram with exact count reconciliation, retain the histogram in source evidence, and add a dedicated effective-rank-versus-reconstruction-gain plot sourced from `effective-rank.json`.

- [ ] **Step 7: Verify, commit, and review**

Run focused suites, `uv run pytest -q`, `uv lock --check`, and `git diff --check`; run no production stage. Commit once as `fix: stream phase1a5 evidence safely`, then obtain a fresh read-only review. Only a clean review may unlock a new source + 2,048-token smoke.

---

### Task 10: Recoverable transient accelerator failures

**Files:**
- Modify: `src/ppmoe/oracle/reconstruction_local.py`
- Modify: `src/ppmoe/experiments/phase1a5.py`
- Modify: `src/ppmoe/reporting/phase1a5.py`
- Modify: the corresponding local/orchestration/reporting tests
- Modify: `PROJECT_LOG.md` only after new production evidence is finalized

- [ ] Preserve every failed attempt in append-only checkpoint `failure_history`, including batch, exception type/message, and attempt ordinal; keep `failures` as unresolved terminal state.
- [ ] Permit an unresolved batch to be retried only in a fresh invocation loading the checkpoint. A successful atomic chunk/state commit removes that batch from unresolved `failures` while retaining history; an unsuccessful retry appends history and remains unresolved. Never retry inside a CUDA-poisoned process.
- [ ] Surface recovered-attempt counts/history in local stage and report evidence without weakening the registered requirement that final `failed_batches == 0`.
- [ ] Add RED/GREEN tests for failure -> reload -> successful completion, repeated failure, rollback during recovery commit, source/signature mismatch, and report reconciliation. Add a resource-lifecycle test proving all fixed-buffer file handles close after successful and exceptional quantile traversal.
- [ ] Run focused and full verification, commit once, obtain a fresh review, then rerun clean source + smoke. Preserve the failed 37,888-token run and do not reuse its checkpoint across the changed code signature.
