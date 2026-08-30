# Phase 1 Oracle V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Work inline because no delegated agents were requested. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce conversion-equivalence, local-oracle, rollout-oracle, and relative-perplexity results for the pinned Qwen3.5-0.8B Phase 1 V0 experiment.

**Architecture:** A pinned, manifest-driven pipeline materializes identical token streams for every comparison, converts the Qwen text FFNs into deterministic balanced experts, and gates expensive oracle stages on loader and equivalence checks. Every long stage writes atomic resumable accumulators and compact final summaries.

**Tech Stack:** Python 3.11, PyTorch CUDA/BF16, Transformers 5.x, Hugging Face Datasets 4.x, pytest, Matplotlib.

**Spec:** `docs/superpowers/specs/2026-08-30-phase1-oracle-v0-design.md`

## Global Constraints

- Work only on branch `exp`; do not modify `main`.
- Use model revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` and concrete `Qwen3_5ForCausalLM` with `trust_remote_code=False`.
- Use FineWeb-Edu `sample-10BT/train` revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` for exactly 100,000 calibration tokens.
- Use WikiText `wikitext-103-raw-v1/test` revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Use BF16 model weights, context length 512, eight experts of width 448, and K values `[8, 6, 4, 3, 2]`.
- Add no router, training, distillation, shared expert, quantization, or vision conversion.
- Generated weights, datasets, raw text, and bulky artifacts remain ignored by Git.
- Append evidence to `PROJECT_LOG.md`; commit only at the three approved milestones.

## File structure

```text
configs/phase1-v0.toml                 immutable experiment configuration
src/ppmoe/config.py                    typed config loading and validation
src/ppmoe/provenance.py                canonical JSON, hashes, atomic writes
src/ppmoe/data/token_stream.py         deterministic document token packing
src/ppmoe/data/huggingface.py          pinned FineWeb/WikiText acquisition
src/ppmoe/adapters/qwen35_runtime.py    concrete model load and layer discovery
src/ppmoe/moefication/qwen35.py        Qwen MLP conversion and selected execution
src/ppmoe/evaluation/perplexity.py     shared causal-NLL/PPL accumulation
src/ppmoe/oracle/local.py              expert capture and local aggregation
src/ppmoe/oracle/rollout.py            greedy layer-wise rollout selection
src/ppmoe/experiments/phase1_v0.py     gated orchestration and resume
src/ppmoe/reporting/phase1.py          CSV/JSON summaries and five plots
scripts/run_phase1_v0.py               command-line entry point
tests/...                              focused unit/integration tests
```

---

### Task 1: Immutable configuration and provenance

**Files:**
- Create: `configs/phase1-v0.toml`
- Create: `src/ppmoe/config.py`
- Create: `src/ppmoe/provenance.py`
- Test: `tests/test_phase1_config.py`
- Modify: `.gitignore`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `Phase1Config.load(path: Path) -> Phase1Config`
- Produces: `atomic_write_json(path: Path, value: Mapping[str, object]) -> str`
- Produces: `sha256_file(path: Path) -> str`

- [ ] Write tests asserting the exact model/dataset SHAs, K ordering, 100,000-token target, 512 context, expert width, invalid configuration failures, canonical JSON hashing, and atomic replacement.
- [ ] Run `uv run pytest tests/test_phase1_config.py -v` and require failure because the modules do not exist.
- [ ] Add `datasets>=4,<5` and `matplotlib>=3.10,<4`, configure `artifacts/` and Hugging Face cache paths in `.gitignore`, and encode every immutable value from the spec in TOML.
- [ ] Implement frozen dataclasses that reject altered revisions, duplicate/unsorted K values, non-divisible widths, non-positive token targets, and paths outside the repository artifact root.
- [ ] Implement canonical UTF-8 JSON (`sort_keys=True`, compact separators), SHA256, and same-directory temporary-file replacement.
- [ ] Run `uv sync --group dev`, the focused tests, `uv lock --check`, and `git diff --check`.

### Task 2: Deterministic corpus materialization

**Files:**
- Create: `src/ppmoe/data/__init__.py`
- Create: `src/ppmoe/data/token_stream.py`
- Create: `src/ppmoe/data/huggingface.py`
- Create: `tests/test_token_stream.py`
- Create: `tests/test_huggingface_data.py`

**Interfaces:**
- Consumes: `Phase1Config`, `atomic_write_json`
- Produces: `DocumentRecord(id: str, row_index: int, text_sha256: str, token_count: int, truncated: bool)`
- Produces: `materialize_token_stream(rows, tokenizer, target_tokens, context_length, output_dir) -> CorpusManifest`
- Produces: `stream_fineweb(config)`, `load_wikitext_test(config)`

- [ ] Write fake-tokenizer tests proving EOS insertion, exact token stopping, final partial contexts, whole-document exclusion after a truncated calibration document, stable row ordering, and zero overlap between calibration and held-out IDs.
- [ ] Write mocked dataset-loader tests requiring repository, config, split, revision, and `streaming=True` for FineWeb; require the raw WikiText configuration and pinned revision.
- [ ] Run both test files and confirm missing-module failures.
- [ ] Implement packing into persisted `input_ids.pt` and `attention_mask.pt` plus a manifest containing row indices, FineWeb IDs, text hashes, per-document counts, truncation, total tokens, context counts, and final length.
- [ ] Reject absent/duplicate FineWeb IDs, empty token streams, missing EOS, revision mismatches, and calibration/held-out overlap.
- [ ] Add a round-trip test that reloads persisted tensors and proves their hash and token count match the manifest.
- [ ] Run the focused tests and the full non-CUDA suite.

### Task 3: Guarded Qwen loader and full FFN conversion

**Files:**
- Create: `src/ppmoe/adapters/qwen35_runtime.py`
- Create: `src/ppmoe/moefication/qwen35.py`
- Test: `tests/test_qwen35_runtime.py`
- Test: `tests/test_qwen35_conversion.py`

**Interfaces:**
- Produces: `load_qwen35_pair(config) -> LoadedModelPair`
- Produces: `iter_text_mlps(model) -> Iterator[tuple[int, nn.Module]]`
- Produces: `convert_qwen35_text_ffns(model, num_experts, seed_base) -> ConversionManifest`
- Produces: `PartitionedQwen35MLP.forward(x, selected_experts=None) -> Tensor`
- Produces: `PartitionedQwen35MLP.expert_outputs(x) -> Tensor`

- [ ] Build miniature real `Qwen3_5ForCausalLM` fixtures and tests for concrete class enforcement, BF16 parameter validation, 24-layer discovery semantics, and vision-module identity preservation.
- [ ] Add conversion tests proving physical complete-triplet permutation, eight contiguous 448-neuron groups at production dimensions, equal scalar parameter count, deterministic per-layer seeds, selected execution, and FP32 maximum error `<=1e-5`.
- [ ] Run the tests and confirm missing-interface failures.
- [ ] Implement `from_pretrained` with exact revision, concrete class, BF16 dtype, `trust_remote_code=False`, `eval()`, and explicit device placement; inspect actual parameters after loading and return CUDA memory evidence.
- [ ] Implement an in-place Qwen MLP replacement using physically permuted gate/up rows and down columns. Store permutations as non-persistent buffers so they do not change the serialized parameter budget.
- [ ] Implement all-expert and per-token selected-expert execution without adding trainable parameters.
- [ ] Run focused tests, the existing Phase 0 tests, and a real config-only check.

### Task 4: Shared perplexity evaluator and M0 gate

**Files:**
- Create: `src/ppmoe/evaluation/__init__.py`
- Create: `src/ppmoe/evaluation/perplexity.py`
- Test: `tests/test_perplexity.py`
- Create: `tests/test_qwen35_live_equivalence.py`

**Interfaces:**
- Produces: `NLLAccumulator.update(logits, input_ids, attention_mask) -> None`
- Produces: `NLLAccumulator.finalize() -> PerplexityResult`
- Produces: `evaluate_perplexity(model, batches, checkpoint_dir) -> PerplexityResult`
- Produces: `evaluate_m0(teacher, student, token_stream) -> M0Result`

- [ ] Write hand-computed tests for one-token shifting, mask exclusion, stream-boundary exclusion, FP64 aggregation, scored-token counts, failed-batch invalidation, and resume without double-counting.
- [ ] Run the focused tests and confirm the module is absent.
- [ ] Implement batch JSONL records keyed by deterministic batch index and an atomic aggregate containing NLL sum, scored tokens, failures, PPL, elapsed time, and peak VRAM.
- [ ] Add a miniature Qwen integration test comparing dense/all-expert logits and PPL on identical tokens.
- [ ] Implement M0 acceptance as finite cosine `>=0.99999` and `0.999 <= PPL_student/PPL_teacher <= 1.001`; failure raises a typed gate error that prevents oracle stages.
- [ ] Run focused tests and the full suite including CUDA.

### Task 5: Local exact-oracle pipeline

**Files:**
- Create: `src/ppmoe/oracle/local.py`
- Test: `tests/test_local_oracle.py`

**Interfaces:**
- Consumes: `PartitionedQwen35MLP.expert_outputs`, `exact_gram_oracle`
- Produces: `LocalOracleAccumulator.update(layer, k, target, expert_outputs) -> None`
- Produces: `run_local_oracle(model_pair, batches, k_values, checkpoint_dir) -> LocalOracleResult`

- [ ] Write synthetic multi-layer tests for per-token/per-layer aggregation, K=8 reconstruction, normalized squared error, subset counts, co-selection matrices, interaction/cancellation matrices, checkpoint resume, and direct/Gram verification samples.
- [ ] Run the focused test and confirm failure because `local.py` is absent.
- [ ] Capture teacher FFN inputs/outputs and student expert contributions with removable hooks; immediately detach aggregates so activations do not accumulate across batches.
- [ ] Compute exact subsets for K `[8,6,4,3,2]`, FP64 metric sums, selection counts, pairwise co-selection, signed cross-expert inner products, and cancellation rates.
- [ ] Persist one atomic accumulator per completed batch and refuse resume if config, model, conversion, or token hashes differ.
- [ ] Run the 2,048-token smoke fixture and require all smoke gates before enabling a 100,000-token target.
- [ ] Run focused tests and the full suite.

### Task 6: Greedy rollout oracle

**Files:**
- Create: `src/ppmoe/oracle/rollout.py`
- Test: `tests/test_rollout_oracle.py`

**Interfaces:**
- Produces: `select_rollout_subsets(target_next, candidate_next, k) -> OracleResult`
- Produces: `run_rollout_oracle(teacher, student, batches, k, checkpoint_dir) -> RolloutResult`

- [ ] Write a two-layer toy decoder where local and rollout choices differ, then assert rollout advances with the selected student state and lexicographically resolves ties.
- [ ] Test attention-mask propagation, K=8 dense behavior, per-layer hidden divergence, causal NLL accumulation, and resume without repeated batches.
- [ ] Run the focused tests and confirm missing-module failure.
- [ ] Implement teacher next-layer target capture for the current batch, candidate subset evaluation after the student's non-FFN layer portion, token-wise selection, and sequential state continuation.
- [ ] Reuse the perplexity accumulator and store hidden-state relative L2, normalized MSE, subset counts, PPL, and peak VRAM for each K.
- [ ] Run focused tests, local-oracle regression, and the full CUDA suite.

### Task 7: Gated CLI and required reporting

**Files:**
- Create: `src/ppmoe/experiments/__init__.py`
- Create: `src/ppmoe/experiments/phase1_v0.py`
- Create: `src/ppmoe/reporting/__init__.py`
- Create: `src/ppmoe/reporting/phase1.py`
- Create: `scripts/run_phase1_v0.py`
- Test: `tests/test_phase1_orchestration.py`
- Test: `tests/test_phase1_reporting.py`

**Interfaces:**
- Produces: `run_phase1(config, stage, resume) -> StageResult`
- Produces CLI stages: `preflight`, `datasets`, `m0`, `local-smoke`, `local`, `rollout`, `report`
- Produces: `render_phase1_report(artifact_root) -> ReportManifest`

- [ ] Write orchestration tests proving stage ordering, M0/local-smoke kill switches, resumable idempotence, disk-space gate, exact token targets, and nonzero exit status for partial/failed aggregates.
- [ ] Write reporting tests from fixed synthetic metrics requiring all five PNGs, plot-source CSVs, correct active fractions, both PPL ratios, and hashes in the report manifest.
- [ ] Run tests and confirm missing-interface failures.
- [ ] Implement a CLI with explicit `--stage`, `--config`, `--resume`, `--artifact-root`, and `--microbatch-size`; do not provide flags that mutate pinned scientific inputs.
- [ ] Implement Matplotlib plots with labeled axes, dense reference lines, token counts, K values, and no smoothing that obscures measurements.
- [ ] Run the complete test suite, `git diff --check`, and a CLI preflight using the official pinned repositories without downloading weights.
- [ ] Append verified implementation evidence and remaining hazards to `PROJECT_LOG.md`.
- [ ] Create the second milestone commit only after all implementation tests pass and push `exp`.

### Task 8: Execute Phase 1 V0 and decide M0/M1

**Files:**
- Generated: `artifacts/phase1-v0/**` (ignored)
- Modify: `PROJECT_LOG.md`
- Create: `results/phase1-v0-summary.json`
- Create: `results/phase1-v0-report.md`

**Interfaces:**
- Consumes: every tested CLI stage
- Produces: compact reproducible evidence and the M0/M1 decision

- [ ] Run `preflight`; record disk, dependency, CUDA, BF16, model/dataset revision, expected download bytes, and cache-policy evidence.
- [ ] Run `datasets`; verify calibration is exactly 100,000 tokens, held-out is exactly 100,000 tokens, document ID intersection is empty, and WikiText uses only the raw test split.
- [ ] Run `m0`; stop if conversion or `0.999..1.001` PPL-ratio gates fail. Preserve negative evidence.
- [ ] Run `local-smoke`; stop if any finite/K=8/direct-versus-Gram/resume assertion fails.
- [ ] Run the 100,000-token local oracle with resume enabled; record exact processed, failed, and skipped batch counts.
- [ ] Run rollout for K `[8,6,4,3,2]` on the identical WikiText and held-out FineWeb token tensors; never replace incomplete PPL with a partial aggregate.
- [ ] Run `report`; verify all five PNGs and their CSV sources can be regenerated from final JSON metrics.
- [ ] Write `results/phase1-v0-summary.json` with input hashes, scored counts, M0/M1 status, local/rollout metrics, both PPL ratios, and artifact hashes; write the Markdown report without raw corpus text.
- [ ] Append exact commands, elapsed times, failures/recoveries, hashes, and the next decision gate to `PROJECT_LOG.md`.
- [ ] Re-run tests and summary validation, create the third milestone commit, push `exp`, and leave `main` unchanged pending user review.

## Self-review

- Spec coverage: all pinned inputs, token rules, conversion invariants, M0, local oracle, rollout oracle, both PPL datasets, five plots, resume, logging, and safety gates map to Tasks 1–8.
- Completeness scan: every implementation and validation step is concrete; no deferred markers remain.
- Type consistency: configuration/provenance feed data and loading; conversion feeds both oracles; the shared NLL accumulator feeds M0 and rollout; reporting consumes only finalized metrics.
- Commit cadence intentionally overrides the skill default: one combined design/plan commit, one verified implementation commit, and one first-results commit, as requested by the user.
