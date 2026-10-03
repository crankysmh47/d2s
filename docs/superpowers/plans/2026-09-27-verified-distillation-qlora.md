# Option 2: verified teacher-rationale QLoRA study — execution plan

**Status (2026-10-03): revision 2, execution authorized.** The user directed revision of the four identified defects and continuation of this plan. See [execution state](../../distillation-v1-execution-state.md). The V0–V3 recurrence decisions remain terminal and unchanged. Work on `exp`; never merge to `main` without a separate request.

> For an implementer: read this entire plan before editing. Work in ordered tasks; run the named tests and gates before advancing. A failed scientific gate is terminal for this candidate, not permission to tune and rerun. Preserve unrelated `task-14*.md` files and every recurrence artifact. Keep a signed, hash-linked run ledger and write a report even for a stop.

**Question:** Can verified worked solutions from a larger dense teacher, distilled into the existing dense `Qwen3-1.7B` through low-cost QLoRA, improve held-out math-word-problem accuracy over both the untouched student and ordinary gold-solution QLoRA on the *same training questions*? This is a post-training capability experiment. It does **not** create an MoE-like parameter footprint, demonstrate general reasoning improvement, or guarantee a gain.

**Rationale:** QLoRA trains adapters while the base is 4-bit frozen [QLoRA](https://arxiv.org/abs/2305.14314); rationale supervision can transmit useful intermediate behavior, but the benefit is task/model dependent [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301). Full teacher-logit distillation adds storage, alignment, and compute complexity beyond this first 16-GiB-GPU experiment [MiniLLM](https://arxiv.org/abs/2306.08543). Choose one teacher and one fixed recipe, not a model or prompt search.

**Existing evidence:** Recurrence V0/V1/V2 failed their registered calibration gates and V3 stopped before GPU on review-evidence validation ([V3 qualification](../../recurrence-v3-scorer-qualification.md)). No recurrence LoRA trained. This plan is a new, isolated study and must not reinterpret old controls as positive evidence.

## 1. Frozen scientific contract

### Models, hardware, and scope

| Role | Snapshot | Use |
| --- | --- | --- |
| Student F0 and all adapters | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B), revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Same tokenizer/chat template/base for every arm. |
| Teacher T | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), revision `1cfa9a7208912126459214e8b04321603b3df60c` | One fixed, offline rationale source. Never train teacher. |

The local target is one RTX 5060 Ti with about 16 GiB VRAM. Run teacher and student sequentially; never require both resident simultaneously. First verify actual driver, BF16, PyTorch/Transformers/PEFT/bitsandbytes compatibility and peak memory. The [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) describes the intended NF4/double-quant/BF16/all-linear recipe. Treat support on this precise Windows/CUDA/GPU stack as **unverified until a real smoke**, not as established by documentation. Pin working dependency versions and wheel hashes in `uv.lock` before any model output; no paid API or remote inference.

VRAM enforcement: physical device capacity must be <=16 GiB for this local study; reserve 2 GiB for desktop/other allocations. Each phase must keep both PyTorch peak allocated and peak reserved bytes <=`min(16 GiB, reported physical capacity) - 2 GiB`; fail before advancing if either exceeds this process budget. Record device free/used snapshots before/after each phase and allocator peaks; do not label allocator measurements as whole-device peaks. On OOM preserve state and stop that phase. F0/A1/A2 inference must all apply the same frozen-base preparation/casts used by the QLoRA training loader (including promoted floating-point embeddings/head), with gradient checkpointing disabled for evaluation, so quantization and base precision do not confound the comparison.

The fixed user prompt is `Solve the following problem. Show concise working, then finish with Final answer: <number>.\n\n{question}`; Qwen chat formatting, thinking disabled, no system message/few-shot examples, greedy decoding. On 60 separate calibration questions, try budget pairs (new/total) **256/768, 512/1024, 768/1536**, in that order. Run both models for each attempted pair; choose the first pair with >=57/60 EOS completions for each model and all prompts fitting total-minus-new tokens. Selection uses completion lengths and EOS only, never gold/accuracy. Freeze prompt, parser and chosen budget before the 120-question viability pilot. If no pair qualifies, stop before corpus generation. Those 60 questions are excluded from all later sets. Training uses the chosen total-token limit. Task 0 tests synthetic training feasibility at the maximum 1536 length.

### Data, separation, and contamination caveat

Use [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) revision `740312add88f781978c0658806c59bc2815b9866`, `main` train/test only (7,473/1,319 rows). Use official [SVAMP](https://github.com/arkilpatel/SVAMP) commit `78e727689e1c1bebfc4be39c446898e8e10b0518`, `SVAMP.json` only, as transfer evaluation; it is never training data. Authenticate SHA-256 of materialized source bytes, row counts, unique IDs, and licenses before partitioning. If source revisions, counts, or format differ, stop and amend *before* output. Do not quietly substitute a mirror.

Canonical ID = SHA-256 of `dataset-name + NUL + source-split + NUL + exact question UTF-8`; duplicate exact questions across all sets are excluded and logged before selecting. For ordering, sort by SHA-256 of `"d2s-distill-v1|47|" + canonical ID`, then canonical ID. Freeze ordered IDs and source bytes in a manifest. From GSM8K train: first 60 = budget/scorer calibration, next 120 = teacher/student pilot, next 200 = development, next 1,200 = teacher-generation pool; the remainder are unused. From GSM8K test: first 400 = sealed primary; remainder unused. From SVAMP: first 200 = transfer development, next 500 = sealed transfer; remainder unused. Cross-source normalized-question exact and near-duplicate audit (lowercase/whitespace/punctuation normalization plus character 5-gram Jaccard >=0.85) must remove overlaps from later pools **before** fixed selection; if required counts cannot be met, stop before output. Do not use gold solutions from held-out splits in prompts, training, or model choice.

These are public, older datasets and might overlap pretraining. A positive result supports this narrowly scoped *post-training benchmark comparison*, not uncontaminated reasoning generalization. Release licenses and any restrictions must be recorded in the manifest.

Deduplication priority is GSM8K test, then SVAMP, then GSM8K train. Keep the first normalized exact question in each source under canonical-ID ordering; discard duplicates in lower-priority sources. Near-duplicate removal applies across sources only, preserving the higher-priority source. Record exclusions and counts before partitioning. Source materialization may serialize sealed answers internally to preserve the dataset snapshot; model-facing loaders must reject sealed splits before the authorized sealed stage, and preparation never prints their text or gold.

### Arms and one hypothesis

- **F0:** unchanged student loaded in the identical NF4/double-quant/BF16-compute configuration used by A1/A2, adapters absent. Optional BF16 F0 on calibration questions measures quantization effects descriptively and cannot replace the primary F0 after outputs are seen.
- **A1:** ordinary supervised QLoRA on the accepted GSM8K IDs using the human-written GSM8K solution, normalized to the same final-answer format.
- **A2:** QLoRA on exactly those IDs using the teacher's verifier-accepted worked solution. No teacher target is shown at evaluation.
- **A3 (optional diagnostic):** if A2 training CUDA time exceeds A1 by >5%, extend the A1 checkpoint with the same deterministic order and final learning rate to the closest whole-update CUDA time (maximum 30 extra updates). Report achieved mismatch. This is descriptive sensitivity analysis; A3 never selects the primary comparator or gates confirmation. Report teacher-generation/audit cost separately.

A1 and A2 share data IDs/order, tokenizer, prompt mask, adapter initialization seed, optimizer, schedule, sequence length, batch/accumulation, and **300 optimizer updates**. Pad every input to the frozen total-token length; no sequence packing, dynamic padding, length bucketing, or padding-free attention. Both arms process 2,400 examples and identical dense tensor shapes, so forward/backward work is comparable despite different target lengths. Loss is the mean over each example's unmasked target tokens, followed by the mean over examples/accumulation steps. Record processed/padded/supervised tokens, exact elapsed CUDA time, peak memory and total pipeline cost. Never claim supervised-token equality. Reject an ID if either target does not fit; apply the same accepted ID set to both arms. If fewer than 800 accepted matched rows remain, stop. Use seeds 47 and (conditionally) 48; no hyperparameter sweep, teacher ensemble, prompt search, or additional candidates. The first seed is discovery, the second is a single confirmation.

Initial fixed training recipe to smoke-test: frozen 4-bit NF4 + double quantization, BF16 compute, `prepare_model_for_kbit_training`, PEFT `target_modules="all-linear"`, rank 16, alpha 32, dropout 0, sequence length equal to the calibrated total-token limit, microbatch 1, accumulation 8, AdamW 1e-4, 10 warmup updates then cosine decay, gradient checkpointing, cycling through deterministic seed-specific shuffled epochs until 300 updates (2,400 example presentations; not necessarily one epoch). If the exact combination cannot fit the local GPU during the *pre-output* smoke, stop and revise the protocol before any pilot result; do not selectively alter one arm. Existing `src/ppmoe/recurrence/lora.py` is FP32 selective-layer LoRA and is **not** a QLoRA trainer.

### Teacher and evaluator qualification

1. Before training or corpus generation, score T and F0 on the same 120 pilot questions. T must have at least 60% correct final answers and beat F0 by at least 5 percentage points. This is a heuristic viability gate, not a significance claim. Teacher inference plus pilot are charged to the run budget. A failed pilot ends this candidate.
2. Generate once on the 1,200 fixed train candidates. Accept only an EOS-terminated response with one unambiguous `Final answer:` numeric value mathematically equal to GSM8K gold, no conflicting later final assertion, no cap, no incomplete solution, and both teacher/human targets fitting the frozen calibrated training template. Compare exact rational/decimal values without floating-point tolerance; units/currency symbols may be stripped only by a frozen grammar. Store rejection reason for every candidate. At least 800 matched examples must survive.
3. Final-answer equality is insufficient for a trustworthy rationale. Before training, independently audit the first 60 accepted rows ordered by SHA-256 of `"d2s-distill-audit|47|" + canonical ID`, blind to model identity but with the question and gold solution available to check each substantive step. Require at least 54/60 valid worked solutions and no systematic error pattern affecting more than two examples. Otherwise stop. A reviewer may mark `valid`, `invalid`, or `uncertain`; uncertain counts against the gate. A second independent reviewer checks all invalid/uncertain and 20 deterministically selected valid cases; disagreements get a third reviewer. Record reviewer/session IDs and immutable review files. Never repair teacher text after audit.
4. Build a new numeric final-answer extractor from **full response text**, not V3's citation-span grammar. First qualifying final assertion wins only if no conflicting subsequent final assertion; missing/ambiguous/capped is invalid and wrong for accuracy. Validate on synthetic tricky cases and a blind answer audit on the separate 60-question calibration set before freezing. Reviewers record the extracted value and supporting substring, but a short citation is evidence of what they saw, not an independently parseable input. Reconcile reviewer values against the frozen full-response parser; any parser/reviewer disagreement above 5% or invalid rate above 10% for either model on the 60-row calibration preflight stops before training. Do not patch scorer after seeing study-arm outputs; make a new version instead. Explanation validity is a separate field and cannot silently change final-answer correctness.

### Evaluation and stop rules

Use paired question IDs for all arms. Development primary = 200 GSM8K train questions not in training; transfer development = 200 SVAMP. Sealed primary = 400 GSM8K test; sealed transfer = 500 SVAMP. All evaluation uses the same prompt/decoding/budget, one output per question. The scorer only sees completion text and stop metadata; gold enters after extraction. Reject a parser that cannot pass predeclared unit cases or blind calibration. Keep raw completions, scores, IDs, hashes, and token/latency/VRAM counters.

FineWeb general-language retention: reuse the existing pinned recurrence FineWeb token sample and paired perplexity computation **only after authenticating its source/split/token IDs and adapting it for the new dense adapters**. Same exact tokens, context, dtype, and batch for F0/A1/A2/A3; a new sample needs preregistration. Require each candidate PPL point ratio vs F0 <=1.02 and paired bootstrap 95% upper bound <=1.02. If the old sample cannot be authenticated or its tokenization differs, stop before training. Include three fixed, non-math, non-graded prompts for qualitative sanity only; they cannot select a winner.

Discovery after seed 47 has two separately reported paths. **Practical path:** A1 or A2 beats F0 by >=2 percentage points on GSM8K development, passes retention, has invalid <=10%, and is no worse than F0 by >3 points on SVAMP development. Carry every qualifying arm into seed 48; if neither qualifies, stop before confirmation/sealed evaluation. **Teacher-specific path:** in addition, A2 beats A1 by >=2 points and is no worse than A1 by >3 points on SVAMP; this determines whether teacher-specific discovery is promising, but its failure does not cancel a passing practical path. A3 is descriptive only. These are screening rules.

If discovery passes, train seed 48 for both A1/A2 with identical settings; carry an arm to a practical claim only if it passes practical discovery in both seeds. Open sealed sets once after checkpoint freeze. The **primary registered scientific comparison is A2 minus A1**, regardless of test results: teacher-specific success requires >=4 absolute points on sealed GSM8K and a paired bootstrap 95% lower bound >0 in each seed, plus A2 retention and <=3-point SVAMP regression vs A1/F0 in each seed. Separately, **practical success** for A1 or A2 requires >=4 points vs F0 and retention/<=3-point SVAMP regression vs F0 in both seeds, with paired bootstrap **97.5% intervals** excluding zero (Bonferroni correction for the two practical arm comparisons). An arm must also pass both development gates to earn that claim. Never choose a comparator from sealed accuracy. Report all three comparisons even if none qualify. Bootstrap uses 10,000 paired resamples of ordered IDs, NumPy PCG64 seed 927047; scientific percentile 2.5/97.5 and practical percentile 1.25/98.75. Report all raw counts, confidence intervals, timing, VRAM, tokens, and adapter/on-disk size, including nonsuccess outcomes. No extrapolation to MoE-style storage or general reasoning.

Hard resource caps for one study: 24 CUDA-active hours across all phases, 48 wall hours after first GPU stage, 16 GiB physical VRAM (record peak and stop on OOM), 20 GiB newly downloaded model/data cache, 20 GiB generated run artifacts excluding shared pre-existing cache. Reserve at least 4 CUDA hours for evaluation before starting training; if estimated remaining budget is insufficient, stop. Scientific gates remain terminal. Infrastructure recovery may resume the same frozen protocol from validated complete checkpoints or completed output rows, at most two recovery attempts per phase. Recheck code/config/model/data/checkpoint hashes and retain all failure records and consumed budget; do not alter hyperparameters, prompt, scorer or selection. Lost/corrupt state, exhausted budget, or a third failure ends as an environment stop. Allow at most two transient download retries per file; no cached-source substitution. Include download size/space preflight before network access.

## 2. Repository layout and interfaces

Authorized implementation paths:

```
configs/distillation-v1.toml
src/ppmoe/distill/{config,data,answers,model,teacher,train,evaluate,gates,study}.py
scripts/run_distillation_v1.py
tests/test_distill_{config,data,answers,model,teacher,train,evaluate,gates,study}.py
artifacts/distillation-v1/<run-id>/  # generated, immutable run output
docs/distillation-v1-report.md        # generated final report, including stops
```

Core callable contracts (exact names/typed signatures can be refined *before* first output, then hash-frozen): `load_config(path: Path) -> DistillConfig`; `prepare_splits(config) -> SplitManifest`; `extract_final_numeric(text: str, stop_reason: str) -> Extraction`; `score_answer(extraction: Extraction, gold: Fraction) -> Score`; `generate_answers(model_role, split, run_root) -> RawManifest`; `verify_teacher_pool(...) -> AcceptedManifest`; `train_adapter(arm, seed, accepted_manifest, run_root) -> CheckpointManifest`; `evaluate_arm(arm, seed, split, run_root) -> ScoreManifest`; `run_stage(stage, config_path, run_root=None) -> StageRecord`. Use dataclasses or typed dictionaries with explicit JSON schemas, no pickle as authoritative state. The CLI accepts only `preflight|prepare|calibrate|qualify|generate|audit|train47|evaluate47|train48|evaluate48|seal|report`; it checks prerequisites and terminal status before work. A failed stage yields a durable `stopped_scientific` or `stopped_environment` report, never an implicit pass.

Run ID = SHA-256 of canonical config bytes, pinned source revisions, exact package lock, Python/CUDA/device IDs, closed named list of code/test file hashes, and *precomputed* source/dataset manifests produced before run-root creation. Write manifest and phase reservation exclusively; record phase start/end, actual GPU seconds, wall time, peak memory, token counts, input/output SHA-256, and prerequisite hashes. Close an interrupted phase with its failure record before recovery. Inference resumes after the last hash-valid complete row; training resumes only from an atomic checkpoint containing adapter, optimizer, scheduler, RNG and sampler state at an optimizer-update boundary. Discard only uncommitted output/gradient work, retaining its cost. Reject hash drift and concurrent live workers. Disallow symlink/reparse traversal from run root, overwrite, and source/split substitution. Keep evaluation sealed values inaccessible to earlier stages; test with a poison path. The run's report must independently recompute all gates from raw authenticated rows rather than trusting summaries.

The run ID binds preparation's active source list. Each later stage additionally binds its own code/tests and every transitive source it executes; adding a future trainer before training does not invalidate an earlier calibration record. Freeze stage signatures before that stage's first output and recheck at recovery. Changing a scorer/prompt/model/data used by a completed stage invalidates that candidate; newly implemented, previously unused stage code is permitted. Record the initial feasibility smoke as a separate artifact and charge its GPU time to the experiment ledger.

## 3. Sequential implementation and execution tasks

### Task 0 — dependency and hardware feasibility

- [x] User authorized revision and execution on 2026-10-03. Freeze this revision before pilot output; no additional approval checkpoint is needed for routine in-scope work.
- [x] Read-only verify `exp`, clean tracked worktree, source revisions/licenses/counts, pre-existing FineWeb retention provenance, GPU/driver/BF16/disk; preserve unrelated files. Run `uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"` and record output. No model output yet.
- [x] Pin PEFT/bitsandbytes compatible versions and wheel hashes in `pyproject.toml`/`uv.lock`; create an isolated short smoke that loads the student 4-bit, attaches all-linear adapters, performs one backward/update with a padded synthetic 1536-token sequence, unloads, then loads the teacher BF16 and generates **only a synthetic prompt**. Record peak VRAM, model/cache size, versions and exact failure. This smoke is compatibility evidence, not a benchmark. If it cannot run within cap, stop and propose a revised protocol.

### Task 1 — config, source, partitions, and provenance

- [ ] Write RED tests for malformed config, revision drift, duplicate/near-duplicate questions, deterministic ordered IDs, exact 60/120/200/1,200/400/200/500 counts, train/test separation, forbidden held-out gold access, tampered bytes, unsafe paths, and insufficient pools. Run `uv run pytest tests/test_distill_config.py tests/test_distill_data.py -q` and confirm failure before implementation.
- [ ] Implement `config.py`, `data.py`, `configs/distillation-v1.toml` and `preflight|prepare` stage. Pin exact prompts, token budgets, source IDs, seeds, caps and qualification thresholds from section 1; reject unknown config keys. Materialize manifests without opening sealed answers. RED/GREEN focused tests, full CPU suite, Ruff, `compileall`, `git diff --check`, independent code review, then commit/push before model output.

### Task 2 — scorer and blind development calibration

- [ ] Write RED synthetic tests: negative/decimal/fraction/comma/currency numbers; boxed or narrative intermediate numbers preceding a final cue; conflicting final cues; malformed or multiple values; cap vs EOS; incomplete text; answer aliases; change only gold and prove extraction unchanged. Add gold-free review packet and tamper/session tests.
- [ ] Implement `answers.py`, `evaluate.py`, minimal `model.py` loading/inference, and `calibrate` stage plus reviewer export/reconciliation. After synthetic tests and code freeze, run the three prespecified budget candidates sequentially on the 60 calibration questions. Audit all 60 outputs per model under the selected budget with independent Luna extraction review and adjudication. Reviewer files must show full response, not gold or arm. Check <=5% parser/reviewer disagreement and <=10% invalid response rate; stop if not. Freeze scorer/test/prompt hashes before all study-arm outputs. This calibration cannot use sealed questions or revise prior V3 evidence.
- [ ] Focused/full CPU tests, Ruff, compileall, diff check, independent review, signed freeze commit and push. Do not let a reviewer citation substring become a second parser input.

### Task 3 — teacher viability and verified matched corpus

- [ ] Implement `model.py`, `teacher.py`, `qualify|generate|audit` stages with fake-model RED/GREEN tests for fixed order, EOS, token cap, one completion, exact rational comparison, target fit, partial phase, output hash, and forbidden split access. Review/test/freeze before first pilot inference.
- [ ] Run 120 paired T/F0 pilot and stop unless T >=60% and T-F0 >=5 points. Generate one teacher response for each of 1,200 candidate train questions only if passed. Produce immutable accepted/rejected manifests; require >=800 matched IDs. Audit 60 deterministic accepted rationales under the rule above; preserve raw reviewer decisions and stop on quality gate failure. No LoRA before all three checks pass.

### Task 4 — QLoRA trainer and retention evaluator

- [ ] Write RED CPU/fake-quant tests for identical A1/A2 IDs/order/base/adapter seed, training target masking, target-fit exclusion, fixed 300 updates, validated optimizer-boundary recovery and bounded attempts, optimizer/scheduler state hashes, checkpoint exactness, and no trainable base weights. Add tiny CUDA smoke **within** resource ledger after qualifying corpus, then GREEN implementation in `train.py` and `gates.py`.
- [ ] Authenticate FineWeb retention source/token IDs, implement paired PPL/bootstrap, completion timing and memory capture, and check unchanged base F0 and adapter-loading equivalence. Run focused/full CPU tests, Ruff, compileall, diff check, independent review, commit/push. Freeze all training/evaluation code before arm training; no post-result edit under this candidate.

### Task 5 — seed-47 discovery, conditional seed-48, sealed result

- [ ] Run A1/A2 seed 47 from the same accepted manifest; evaluate F0/A1/A2 on development and FineWeb. Record padded/supervised tokens, time and optional A3 sensitivity. Recompute both discovery paths; continue if any practical arm qualifies.
- [ ] Only if discovery passes and remaining cap suffices, train/evaluate A1/A2(/A3) seed 48 with identical settings; require practical discovery and retention in at least one arm in both seeds. Record teacher-specific discovery separately. Stop if neither practical arm qualifies.
- [ ] Freeze/check all code, config, model, checkpoint, source and split hashes; then open sealed GSM8K/SVAMP once and score F0 and all completed arms/seeds. Apply exact paired bootstrap and final gate. Independently audit raw counts, IDs, arithmetic, cost, retention, and report hashes. Write `docs/distillation-v1-report.md`; record independent `practical_success`, `teacher_specific_success`, and `environment_complete` flags with precise scope. Update `PROJECT_LOG.md` and `todo.md`, commit/push `exp`; no `main` merge.

### Exact command surface for later execution

Implement each command before running it; `--config` and `--run-root` are required, and CLI never silently selects a prior run:

```
uv run pytest tests -q
uv run python scripts/run_distillation_v1.py preflight --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py prepare --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py calibrate --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py qualify --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py generate --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py audit --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py train47 --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py evaluate47 --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py train48 --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py evaluate48 --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py seal --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
uv run python scripts/run_distillation_v1.py report --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/<run-id>
```

For focused tests, invoke explicit test filenames so PowerShell glob behavior is irrelevant. Use `uv run ruff check` only after adding/pinning Ruff or use the repository's existing lint command. At every boundary record exact command, exit, tests, code/manifest/checkpoint hashes, gate, and next permitted stage. A human-only decision is a genuine unresolved environment/license blocker; ordinary negative results are an end-state, not a request to retune.
