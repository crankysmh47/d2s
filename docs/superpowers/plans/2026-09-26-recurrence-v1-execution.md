# Recurrence V1 Benchmark and Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether pretrained shared-block recurrence adapted by LoRA improves verified answer accuracy beyond ordinary LoRA under bounded compute.

**Architecture:** Add a V1 answer scorer, separate reproducible benchmark splits, and a staged runner with durable gate records. Reuse the pinned Qwen3, recurrence, LoRA, training, retention, and statistics code where its public contracts match; keep V0 untouched.

**Tech Stack:** Python 3.11, uv, PyTorch/CUDA, Transformers 5, pytest, existing ppmoe recurrence modules.

**Spec:** [2026-09-26-recurrence-v1-design.md](../specs/2026-09-26-recurrence-v1-design.md).

## Global Constraints

- Work on `exp`; preserve unrelated untracked `task-14-*` files and all V0 artifacts.
- Model `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` from the verified local snapshot; no download.
- Fixed block `[12,16)`, BF16/eager, no cache, `enable_thinking=False`, greedy, 256 output tokens, 512 total tokens for all V1 arms.
- Source training pool: 8,000 V0 training examples, read-only and hash-verified. New pilot/dev/test IDs must not overlap V0 train/dev or one another.
- Exactly two pilot difficulties per family, 30 rows each, six pilots total; select first eligible in table order. Seeds, task rules, gate thresholds, split counts, and stop conditions are exactly those in the spec.
- An absent/ambiguous/capped final answer is incorrect and `reason_code="invalid_format"` in normalized score rows; keep more specific `failure_detail` separately so existing gate aggregation counts it.
- Freeze the scorer code hash before V1 GPU generation. Never open sealed confirmation contents before a persisted passing discovery and seed-48 prerequisite.
- New artifact root `artifacts/recurrence-v1/<run-id>`; derive the initial ID from config, verified source, scorer/generator code, and read-only training-data hashes. The selected fresh split hashes are unknown until calibration and bind the immutable calibration stage, not the initial ID. No overwrites of existing output or stage records. Partial runs retain raw rows and failure evidence.
- Calibration cap 7,200 CUDA-active seconds; whole V1 cap 86,400 CUDA-active seconds; new artifact cap 20 GiB. Stop with a report when a scientific/resource gate fails. Do not search prompts, model blocks, ranks, output caps, or new difficulty candidates.
- After each implemented task: focused tests, source diff review, coherent commit. Use GPT-6 Luna implementers per user's request; parent makes scientific decisions and checks integration. Full suite once before first V1 GPU generation and after any later cross-cutting code change.

## Review Focus

The five most consequential inputs and their expected behavior are pinned in tasks below: a correct intermediate but wrong final value (Task 1), a conflicting yes/no conclusion (Task 1), truncated list completion (Tasks 1 and 3), paired edits to split plus manifest (Task 2), and sealed confirmation read before promotion (Task 7).

## File map

| File | Responsibility |
|---|---|
| `src/ppmoe/recurrence/answer_v1.py` | Frozen deterministic final-answer and graph explanation scoring |
| `src/ppmoe/recurrence/data_v1.py` | Two difficulty candidates and independent pilot/dev/sealed splits |
| `src/ppmoe/recurrence/config_v1.py` | Strict separate V1 TOML config and source/run path validation |
| `src/ppmoe/recurrence/study_v1.py` | Stage prerequisites, immutable raw evidence, budget, gate orchestration |
| `src/ppmoe/recurrence/v1_runtime.py` | Local model loading, frozen and adapted scoring, checkpoint restoration |
| `scripts/run_recurrence_v1.py` | One-stage CLI and terminal exit codes |
| `configs/recurrence-v1.toml` | New frozen values, candidate grid, seeds, model/training/resource bounds |
| `tests/test_recurrence_answer_v1.py` | Parser/adversarial corpus regression |
| `tests/test_recurrence_data_v1.py` | Generator determinism and leakage/manifest checks |
| `tests/test_recurrence_study_v1.py` | CPU fake stage/gate/recovery end-to-end |
| `tests/test_recurrence_v1_runtime.py` | Tiny-Qwen integration when needed |

## Task 1: deterministic final-answer scorer

**Files:** create `answer_v1.py`, `tests/test_recurrence_answer_v1.py`.

**Interfaces:**

```python
def score_completion(example: Example, completion_text: str | None,
                     stop_reason: str) -> dict[str, object]:
    # keys: parsed_value, correct (bool), reason_code,
    # failure_detail, evidence_span, explanation_status, explanation_detail
    ...
```

- [ ] Write failing tests for expression `Final answer: \\boxed{-7}` after intermediate `2`, a wrong final value despite an intermediate gold value, two different final boxes, and an uncued box. Tests must assert `correct`, `reason_code`, and `parsed_value` or absence.
- [ ] Write failing list tests for `Final result: [1, 11, -2, 4]`, multiple interim lists, a wrong final vector, boxed final vector, incorrect length, and an incomplete final bracket at `stop_reason="length"`. Out-of-range integers are parsed and score wrong rather than discarded.
- [ ] Write failing graph tests for leading `Yes.`, terminal `Answer: no`, queried-endpoint conclusive path sentence, opening/terminal conflict, incidental `no` inside edge prose, and a correct yes answer with a false stated edge/path explanation.
- [ ] Implement the grammar verbatim from the spec using anchored, bounded scans. Compare only a unique extracted value with `interpret`; never use reference while extracting. Normalize all ungradable reasons to `invalid_format`, set `correct=False`, retain details. Explanation status is independent of correctness.
- [ ] Verify the frozen old-30 offline audit exactly: 23 parsed, 18 correct, five wrong, seven ungradable. Compare against Luna labels only in this development regression, not inside the scorer. Record scorer SHA256 in the spec addendum/log; do not change the grammar after V1 GPU output.
- [ ] Run `uv run pytest tests/test_recurrence_answer_v1.py -q`, Ruff, compileall, diff check; commit.

## Task 2: strict V1 config and fresh deterministic data

**Files:** create `config_v1.py`, `data_v1.py`, `configs/recurrence-v1.toml`, `tests/test_recurrence_data_v1.py`.

**Interfaces:**

```python
def load_v1_config(path: Path) -> V1Config: ...
def make_pilot(config: V1Config, family: str, candidate: int,
               forbidden_ids: set[str]) -> tuple[Example, ...]: ...
def make_selected_splits(config: V1Config, choices: Mapping[str, int],
                         forbidden_ids: set[str]) -> dict[str, tuple[Example, ...]]: ...
def freeze_v1_split(path: Path, examples: Sequence[Example],
                    tokenizer: object, *, revision: str,
                    tokenizer_sha256: str) -> dict[str, object]: ...
```

- [ ] Encode exact candidate grid, seed constants, 30 pilot/100 dev/200 sealed per family, source paths, token bounds, time/disk caps, and adaptation settings in a strict new TOML; reject unknown/missing keys, bool-as-int, mutable revision, invalid source/output paths, and impossible token budgets.
- [ ] Generate using independent local `random.Random(seed)`, canonical ID `sha256_payload({"family": family, "latent_spec": latent})`, `interpret`, V0 prompt templates, 9,999 expression answer cap, identity/repeated-reverse rejection, alternating expression shape and graph yes/no labels. Never mutate global RNG or V0 generator.
- [ ] Write tests showing identical rows on repeat, distinct seed streams, exact counts/answer verification, graph label balance, no overlap with forbidden train/dev IDs, and explicit failure after at most 100 attempts per accepted example. Do not open V0 sealed test rows in tests or production.
- [ ] Freeze split JSONL/manifest with exact SHA256 and tokenizer revision/hash, prompt/output length evidence, and exclusive create. Test tampering with both split and its manifest: the independent stage hash must reject it. During selected split generation, write sealed values once; expose only its hash/count to calibration and discovery readers.
- [ ] Run focused tests, Ruff, compileall, diff check; commit.

## Task 3: pilot scoring, selection, and calibration gate

**Files:** create `study_v1.py`, `v1_runtime.py`, `scripts/run_recurrence_v1.py`, `tests/test_recurrence_study_v1.py`, `tests/test_recurrence_v1_runtime.py`.

**Interfaces:**

```python
def run_v1_stage(stage: str, config_path: Path, run_root: Path) -> dict[str, object]:
    # stages: prepare, calibrate, frozen, discover, confirm, report
    ...
def score_v1_examples(model: object, tokenizer: object,
                      examples: Sequence[Example], mode: str,
                      output_path: Path, signatures: Mapping[str, str]) -> dict[str, object]: ...
```

- [ ] `prepare`: verify the completed V0 source snapshot/files, original train/dev hashes, scorer/config/code hashes; create a fresh V1 root and read-only train reference. Never read V0 test. Persist config/source/scorer signatures, new run ID, budget ledger and stage record with exclusive-create semantics.
- [ ] `calibrate`: score the six 30-example pilots as F0 using one verified local model load, same chat rendering and generation settings as the completed 256-token probe, `score_completion`, per-token deadline, per-row fsync. Validate exact row IDs/hashes before summary; incomplete execution is a failed stage, not a quality no-go.
- [ ] Summarize each candidate through `summarize_exact_scores`; pick the first meeting strict 5%-95% accuracy and <=10% invalid. Persist full candidate table and choice. Fewer than two chosen informative families => terminal `stopped_scientific` report. Never try a third candidate.
- [ ] If exactly two families qualify, assign candidate 1 to the ineligible third family as a frozen fallback; mark it ineligible and include it in development/confirmation macro metrics. This is not a pilot-based selection.
- [ ] Generate independent 100/family development and 200/family sealed splits using chosen difficulty. Score only development F0. Gate macro 5%-95%, global invalid <=10%, and >=2 individually informative families. Persist status, all counts, raw hashes and exact next allowed stage. Stop scientifically on failure.
- [ ] Export a blinded 10/family development answer sample through the existing editable review tool; assign a Luna model reviewer to grade `good/bad` and compare disagreements with V1 objective scores. Record reviewer identity and hashes. The audit diagnoses parser/task failures but never changes the frozen gate or candidate choice.
- [ ] Implement stage prerequisites so `frozen`/`discover`/`confirm` cannot run early; `confirm` must reject before opening any sealed bytes. Test this with a poisoned/sealed path whose read raises. Test paired manifest+split tamper, output overwrite, partial score crash and recovery, CUDA budget exhaustion, and wrong model/hash.
- [ ] Verify real tiny-Qwen local inference/scorer data flow on CPU, then run focused tests, Ruff, compileall, diff check and full suite. Get independent Luna code review; fix actionable issues; commit before real pilot.
- [ ] Run `uv run python scripts/run_recurrence_v1.py --stage prepare --config configs/recurrence-v1.toml`, then `--stage calibrate --run-id <printed-id>`. Record all measured outputs in `PROJECT_LOG.md`; if scientific or resource stop, run `--stage report` and end without training.

## Task 4: frozen controls on the selected development set

**Files:** extend `study_v1.py`, `v1_runtime.py`, `tests/test_recurrence_study_v1.py`, `tests/test_recurrence_v1_runtime.py`.

- [ ] Implement one shared F0/F1/F2 evaluation path over exactly the persisted 300 development IDs; reuse calibration F0 only if its model/scorer/config/data hashes and token budget match exactly. Load a fresh verified source model per other mode if needed for memory correctness. Score with V1 semantics, 256-token budget, EOS checks, and latency/VRAM/execute-layer evidence. Require paired IDs and immutable summary hashes.
- [ ] Verify old frozen FineWeb scores may be reused only when V0 source revision, unchanged wrapper implementation, chunk IDs, and raw hashes match; otherwise score the original 40 chunks and persist new hashes. Do not use old V0 exact-answer scores as V1 evidence.
- [ ] Test all-mode ID equality, double-pass sharing, unchanged prompt tokens, cap handling, retention evidence rejection on mismatch, and refusal when calibration did not pass.
- [ ] Run focused tests and code review, commit, then `--stage frozen --run-id <id>`. Persist evidence and resource projection. A budget stop goes directly to report.

## Task 5: paired adaptation and seed-47 discovery

**Files:** extend `study_v1.py`, `v1_runtime.py`, `tests/test_recurrence_study_v1.py`, `tests/test_recurrence_v1_runtime.py`.

- [ ] Verify direct reuse of `install_adapters(wrapper.stock_model, ...)`, `capture_initialization`, `load_initialization`, `train_arm`, `measure_training_arms`, `derive_a3_steps`, `save_checkpoint/load_checkpoint` under the new V1 run signatures. Write tiny-Qwen tests proving identical initialized adapter tensors and correct wrapper reload for A1/A2/A3. Do not copy or silently modify the V0 trainer.
- [ ] Pilot 5 warmups and 20 paired A1/A2 timing updates on fixed batches, then freeze A3 steps per `derive_a3_steps`. Preflight 24-hour cumulative CUDA projection, disk and VRAM before each arm. Train 1,000 updates for A1/A2 on the same exact 8,000-row stream at seed 47; A3 repeats the stream to match A2 time. Persist checkpoints, events, elapsed time and hashes; resume only validated complete-update checkpoints in a fresh process after a recoverable fault.
- [ ] Score final checkpoints only on the identical 300 development examples with V1 scorer and on 40 FineWeb chunks. Compute paired bootstrap and `discovery_decision`; persist all quality/family, PPL point/CI, realized A3/A2 time, and cost predicates. Never select the best checkpoint or retune failed thresholds.
- [ ] Test mismatch of adapter naming/initialization, missing checkpoint, incomplete scores, wrong paired ID set, PPL CI uncertainty, time ratio outside [0.90,1.10], and budget stop before further work.
- [ ] Run focused tests, full integration review, commit, then `--stage discover --run-id <id>` only after a passing frozen stage. If discovery stops, report the terminal outcome and stop.

## Task 6: seed-48 confirmation, sealed results, and report

**Files:** extend `study_v1.py`, `v1_runtime.py`, create `report_v1.py`, extend their tests.

- [ ] Enforce persisted passing discovery and authenticated complete seed-47 data before launching seed 48. Repeat A1/A2/A3 from a fresh seed-48 initialization and independent optimizer state, with the same frozen training pool and A3 timing rule. Persist full checkpoints, budget and retention evidence.
- [ ] Only after both seeds have complete valid final checkpoints, retention, and timing, open the new sealed 600-example split. Assert its hash, expected IDs and selected difficulty. Score F0 and every required adapted arm under identical V1 evaluator/token budget; keep partial raw records on environment failure but never report them as complete metrics.
- [ ] Use existing `paired_accuracy_bootstrap` and `confirmation_decision` with seed 9047 and 10,000 replicates; require the same two-seed quality, no-family-regression, retention and timing predicates as the spec. Test early sealed access and duplicate/missing rows, seed confusion, selection bypass, raw/hash mismatch, terminal no-go, and passing report fixtures.
- [ ] Produce a report with source/config/scorer/data/checkpoint hashes, all gates, family accuracy, invalid and transition counts, retention, training/evaluation CUDA time, peak VRAM, layer evaluations, latency, provenance/deviations and calibrated conclusion. Independent hash audit before any push.
- [ ] Run focused/full tests and final Luna review, commit, execute `--stage confirm` only if discovery passed, then `--stage report`. Update `PROJECT_LOG.md`/`todo.md` with a supported go/no-go statement; push `exp` milestones. Never merge `main` automatically.

## Execution/stop ledger

- V0 registered G3 stop stays closed. V1 has its own run ID and artifacts.
- User authorized the sequence and GPT-6 Luna implementers. The parent rules on scientific choices and independent review findings.
- Stage order: prepare -> calibrate -> frozen -> discover -> confirm -> report, with report also callable after a terminal stop.
- Gate stop, resource stop, or repeated environment failure ends the run with a report; it does not trigger a new difficulty/cap/LoRA search.
