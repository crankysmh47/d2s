# Recurrence V2 Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and run one fresh, integrity-checked three-family V2 benchmark calibration, then report its registered go/no-go decision without altering V0/V1 evidence.

**Architecture:** Separate V2 scorer, prompt/data generator, config, inference runtime, and stage controller preserve the sealed V1 run. The controller authenticates pinned source and V1 pilot IDs, binds exact code/data hashes to a new run ID, runs six pilots, and generates development/sealed splits only when the pilot gate passes. A terminal report validates raw evidence and never opens sealed values after a stop.

**Tech Stack:** Python 3.11, PyTorch/Transformers from `uv.lock`, local pinned Qwen3-1.7B, pytest, Ruff, PowerShell CLI.

**Spec:** `docs/superpowers/specs/2026-09-26-recurrence-v2-design.md`. The spec governs any conflict with this plan. The user directed autonomous continuation through process-only reviews; scientific and resource gates remain binding.

## Global Constraints

- Work only on `exp`; preserve untracked `task-14-review.md` and `task-14-rereview-*.md`; never edit V0/V1 stage records, scores, scorer, or raw splits. No merge to `main`.
- Source is the locally pinned `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; BF16/eager/cacheless/greedy, `enable_thinking=False`, block `[12,16)`, 256 new tokens, 512 total tokens.
- Three families remain expressions/lists/graphs; pilot 30 per candidate, development 100 per family, sealed 200 per family. Six pilot seeds `(721001,721002)`, `(722001,722002)`, `(723001,723002)` in family order; development `(731001,731002,731003)`; sealed `(741001,741002,741003)`.
- Candidate grid is expression leaves `(6,7)`, list items `(4,5)` and operations `(1,2)`, graph vertices `(7,8)` at edge probability `0.20`; at most 100 attempts per accepted row; expression absolute answer <=9,999.
- Pilot family eligibility is `0.05 < accuracy < 0.95` and invalid rate `<=0.10`. At least two eligible families precede development generation. Development requires three-family macro `0.05 < accuracy < 0.95`, global invalid `<=0.10`, and two individually informative families. Candidate 1 is a fixed fallback when a family's two candidates fail.
- Calibration limits: 7,200 CUDA-active seconds and 7,200 wall seconds; whole-study 86,400 CUDA-active seconds and 20 GiB new artifacts. An open interrupted GPU phase is terminal, never same-run retry.
- Freeze scorer/generator/config/runtime hashes before first V2 GPU output. No V2 output-informed parser, prompt, difficulty, threshold, or token-cap edits. Do not open V0 sealed test or V2 sealed values before confirmation.
- All LoRA work is conditional on passing calibration; this plan stops at the calibration/report decision. If calibration passes, immediately write the conditional controls/training plan from the approved V2 spec and continue. If it fails, report and stop without training.

## Review Focus

1. An opening `Yes` and later Markdown `Answer: No` must be invalid, even when the gold is `no` (Task 1 test).
2. Repeated list operations must remain repeated, but each *listed* definition appears once and no unlisted definition appears (Task 2 test).
3. A V1 pilot split with a changed byte or a duplicate ID must fail V2 `prepare` before any V2 artifact is created (Task 4 test).
4. A process loss after a phase-start marker must report terminal unknown CUDA duration and must never silently retry (Task 5 test).
5. Report after a pilot no-go must not open any sealed path, including a poisoned path that raises on read (Task 5 test).

## File map and interfaces

| File | Responsibility |
| --- | --- |
| `src/ppmoe/recurrence/answer_v2.py` | V2 graph cue extraction, V1-compatible score record, separate explanation status |
| `src/ppmoe/recurrence/config_v2.py`, `configs/recurrence-v2.toml` | Strict immutable V2 constants and typed configuration |
| `src/ppmoe/recurrence/data_v2.py` | Exact list prompt, deterministic pilots/splits, V2 train view, exclusive split writer |
| `src/ppmoe/recurrence/v2_runtime.py` | Local model load and append/fsync V2 scorer loop |
| `src/ppmoe/recurrence/study_v2.py` | Prepare/calibrate/report stage signatures, budget ledger, selection, integrity checks |
| `scripts/run_recurrence_v2.py` | One-stage CLI, full 64-character run ID |
| `tests/test_recurrence_answer_v2.py`, `tests/test_recurrence_data_v2.py`, `tests/test_recurrence_study_v2.py`, `tests/test_recurrence_v2_runtime.py` | Behavioral and tamper/failure tests |

Interfaces to preserve across tasks:

```python
def score_completion_v2(example: Example, completion_text: str | None, stop_reason: str) -> dict[str, object]: ...
def load_v2_config(path: Path) -> V2Config: ...
def render_v2_list_prompt(latent_spec: Mapping[str, object]) -> str: ...
def make_pilot_v2(config: V2Config, family: str, candidate: int, forbidden_ids: set[str]) -> tuple[Example, ...]: ...
def make_selected_splits_v2(config: V2Config, choices: Mapping[str, int], forbidden_ids: set[str]) -> dict[str, tuple[Example, ...]]: ...
def make_training_view_v2(train_rows: Sequence[Example]) -> tuple[Example, ...]: ...
def freeze_v2_split(path: Path, examples: Sequence[Example], tokenizer: object, *, revision: str, tokenizer_sha256: str) -> dict[str, object]: ...
def score_v2_examples(model: object, tokenizer: object, examples: Sequence[Example], mode: str, output_path: Path, signatures: Mapping[str, str], *, max_output_tokens: int = 256, max_total_tokens: int = 512, deadline: float | None = None) -> dict[str, object]: ...
def run_v2_stage(stage: str, config_path: Path, run_root: Path | None = None) -> dict[str, object]: ...
```

### Task 1: frozen V2 answer scorer

**Files:** Create `answer_v2.py` and `tests/test_recurrence_answer_v2.py`; do not edit `answer_v1.py`.

- [ ] Write RED tests using `Example("id", "graphs", "dev", "n7", "Directed edges: A->B. Is there a directed path from A to B? Return only yes or no.", "yes", {"n": 7, "edges": ["A->B"], "start": "A", "end": "B"})`. Assert `score_completion_v2(row, "Yes\n\n**Answer: No**", "eos")["reason_code"] == "invalid_format"` and `correct is False`; assert bare `Yes`, `**Answer: yes**`, `So, the answer is: **Yes**.`, and `there is **a directed path** from **A** to **B**` extract `yes`; assert wrong queried endpoints, incidental `no` in prose, length cap, and conflicting path conclusion fail. Assert explanation contradiction is preserved separately for a wrong claimed edge. Run `uv run pytest tests/test_recurrence_answer_v2.py -q` and observe the missing API failure.
- [ ] Implement only graph cue recognition with first-nonblank-line, complete answer-cue line, and queried-endpoint conclusion patterns; collect *all* explicit values and reject disagreement. Keep raw evidence spans. For expression/list delegate to unchanged V1 `score_completion`; for graph use the same V1 explanation checker or equivalent deterministic edge check. Never use `row.answer` or `interpret` before candidate extraction. Return the seven V1 score fields exactly. Run focused tests to green.

```python
def test_conflicting_graph_cues_are_invalid(graph_example: Example) -> None:
    scored = score_completion_v2(graph_example, "Yes\n\n**Answer: No**", "eos")
    assert scored["correct"] is False
    assert scored["reason_code"] == "invalid_format"
    assert "ambiguous" in str(scored["failure_detail"])

# Implementation shape: collect (value, original_span) pairs before consulting interpret().
values = {value for value, _span in candidates}
if len(values) != 1:
    return {
        "parsed_value": None, "correct": False, "reason_code": "invalid_format",
        "failure_detail": "ambiguous disagreeing explicit graph conclusions",
        "evidence_span": " | ".join(span for _value, span in candidates),
        "explanation_status": explanation_status, "explanation_detail": explanation_detail,
    }
```

- [ ] Add adversarial RED tests for `Yes, because ...` (not standalone), Markdown bold `no` and endpoints, `Answer: yes` followed by `Final answer: no`, arbitrary edge-line `no`, missing EOS, and a wrong but well-formed value (`wrong_value`, not invalid). Run red, implement minimally, run green. Run old V1 scorer tests, Ruff, compileall, and the full suite; record the `answer_v2.py` SHA256 in the design/log before first GPU output. Commit only V2 scorer/tests and documentation of its hash.

### Task 2: strict config, exact prompts, training view

**Files:** Create `config_v2.py`, `configs/recurrence-v2.toml`, `data_v2.py`, `tests/test_recurrence_data_v2.py`.

- [ ] Write RED config tests for the exact constants in Global Constraints, unknown/missing TOML keys, bool-as-int, changed source/output root, mutable revision, reused V1 seeds, `max_new_tokens != 256`, `max_sequence_tokens != 512`, and unsafe relative paths. Run `uv run pytest tests/test_recurrence_data_v2.py -q` to observe missing API. Implement `load_v2_config` with typed data/resources/LoRA/optimizer fields; use `config_v1.py` schema shape as reference, but never accept V1 output root or V1 seeds. Fix source run to `artifacts/recurrence-v0/129d39646933`, V1 pilot root to `artifacts/recurrence-v1/4f1ddbedf7b4cbd11d73699d579a3e299a049aa8fb2b036e9124842d695eb2ca`, and V2 output root to `artifacts/recurrence-v2`.
- [ ] Write RED exact-prompt tests. For `{input:[3,8,7,2], ops:["ADD1"]}`, expect `Initial list: [3,8,7,2]. Operations to apply in order: ADD1. Apply exactly these operations once each, from left to right; do not apply any operation not listed. Definitions for listed operations: ADD1 adds 1 modulo 10 to each element. Return only the final list.` For `ops:["REVERSE","ADD1","REVERSE"]`, assert both `REVERSE` applications remain in the sequence, definition order is `REVERSE ...; ADD1 ...`, and no `ROTL1` definition appears. Implement `render_v2_list_prompt` and green the tests.

```python
def test_prompt_defines_only_requested_operations() -> None:
    text = render_v2_list_prompt({"input": [3, 8, 7, 2], "ops": ["ADD1"]})
    assert "Operations to apply in order: ADD1." in text
    assert "ADD1 adds 1 modulo 10 to each element" in text
    assert "ROTL1 moves" not in text and "REVERSE reverses" not in text

# Implementation shape: preserve sequence, de-duplicate definitions only.
operation_names = ", ".join(ops)
definitions = "; ".join(OPERATION_DEFINITIONS[op] for op in dict.fromkeys(ops))
```

- [ ] Write RED training-view tests: 8,000 input rows yield 8,000 rows in identical ID/order, latent, answer, split, and structure; only list prompts differ; non-list prompt bytes match exactly; `interpret` matches each gold. Implement `make_training_view_v2` using new `Example` objects and fail on malformed/mismatched rows. The 512-token bound is checked later against the pinned tokenizer before the view is frozen, not guessed from characters.
- [ ] Run focused/full tests, Ruff, compileall, `git diff --check`; commit config/prompt/training-view code. Do not write V2 artifacts yet.

### Task 3: fresh deterministic split generation and exclusive freeze

**Files:** Extend `data_v2.py` and `tests/test_recurrence_data_v2.py`.

- [ ] Write RED tests for six 30-row pilots and selected 300/600-row splits, exact family/difficulty/seed schedules, graph label balance, alternating expression shape, list identity and adjacent-reverse rejection, 100-attempt failure, local PRNG isolation, and no overlap against supplied forbidden IDs or between produced splits. Use hand-computed fixed latents for answer tests—`[3,8]` with `ADD1` yields `[4,9]`, and graph `A->B` queried `A` to `B` yields `yes`. A whole-pilot cross-check against `interpret` may supplement but must not be the only answer assertion. Run tests red.
- [ ] Implement V2 generator using V1 math helpers and `random.Random(seed)`, but V2 list prompts only. IDs remain `sha256_payload({"family":family,"latent_spec":latent})`. `make_selected_splits_v2` must accept a complete three-family choice map and consume/update a local `seen` set; it must not mutate the caller's set. Run focused tests green.

```python
def test_pilot_is_deterministic_and_disjoint(v2_config: V2Config) -> None:
    first = make_pilot_v2(v2_config, "lists", 1, set())
    second = make_pilot_v2(v2_config, "lists", 1, set())
    assert first == second and len(first) == 30
    assert len({row.id for row in first}) == 30
    assert all(row.family == "lists" and row.split == "pilot" for row in first)
    assert all(row.prompt.startswith("Initial list:") for row in first)

# Implementation shape: each call owns its random state and exclusion copy.
rng = random.Random(config.data.pilot_seeds[family][candidate - 1])
seen = set(forbidden_ids)
```

- [ ] Write RED split-writer tests for exact tokenizer revision/hash, prompt+256<=512, gold verification, exclusive creation of JSONL and `.manifest.json`, reordered or altered rows, existing symlink/reparse-point ancestors, and cleanup only of files created on a handled write failure. Implement `freeze_v2_split` with the V1 atomic writer contract and V2-specific error text; run green. Verify sidecar ordered IDs/count/raw SHA. Run focused/full tests, Ruff, compileall, diff check; commit.

### Task 4: authenticated source, V1 exclusion, and `prepare`

**Files:** Create `study_v2.py`, `tests/test_recurrence_study_v2.py`; later CLI consumes this interface.

- [ ] Write RED tests for V0 source-manifest and snapshot hash authentication, 8,000 train/600 dev rows and 8,600 distinct IDs, six V1 pilot files with 30 IDs each authenticated against V1 `calibrate.json` split/sidecar hashes and ordered IDs, duplicate IDs or tampered bytes, no V1 sealed access, and rejection before V2 root creation. Use temp fixtures with intentionally poisoned sealed paths. Run focused tests red.
- [ ] Implement a read-only `verify_v2_sources(config: V2Config, project_root: Path) -> dict[str, object]`: verify pinned V0 source/data stage and inventory as V1 `_verify_source` did; verify V1 prepare/calibrate stage hashes and six bound pilot split/manifest hashes; read only V1 pilot IDs, require 180 unique IDs, and combine them with V0 train/dev IDs. Bind all source and code hashes (including reused V1 scorer/runtime helpers) to the returned signatures. Do not open any V0/V1 test or sealed split.

```python
def test_changed_v1_pilot_prevents_prepare(v2_source_fixture: tuple[V2Config, Path, Path]) -> None:
    config, project_root, split = v2_source_fixture
    split.write_bytes(split.read_bytes() + b" ")
    with pytest.raises(ValueError, match="V1 pilot.*hash"):
        verify_v2_sources(config, project_root)
    assert not (project_root / "artifacts/recurrence-v2").exists()

# Implementation shape: source and exclusion signatures are computed before root creation.
assert sha256_file(v1_split_path) == bound_split_sha256
assert sha256_file(v1_sidecar_path) == bound_sidecar_sha256
forbidden_ids = v0_train_dev_ids | v1_pilot_ids
```

- [ ] Write RED tests for deterministic 64-hex run ID from config bytes/signatures, exclusive root and `prepare.json` creation, hash-bound stage payload, no overwrite of an existing root, and mismatch rejection after changed code/config. Implement `run_v2_stage("prepare",...)` and reusable stage-record reader/writer without modifying V1 files. A prepare failure before root creation must leave no V2 run; a failure after exclusive root creation must durably record `failed_environment` without impersonating a passing prepare. Run focused/full tests, Ruff, compileall, diff check; commit.
- [ ] During `prepare`, convert authenticated V0 train JSONL records to `Example`, create the 8,000-row V2 training view in memory, verify every answer and rendered prompt-plus-answer length with the pinned tokenizer, and compute its canonical JSONL SHA256 before deriving the run ID. After exclusive root creation, write `data/train-view.jsonl` and its manifest once; bind them to `prepare.json`. No training is allowed to reload an unverified view.

### Task 5: one-pass F0 runtime, calibration, and terminal report

**Files:** Create `v2_runtime.py`, `scripts/run_recurrence_v2.py`, `tests/test_recurrence_v2_runtime.py`; extend `study_v2.py` and `tests/test_recurrence_study_v2.py`.

- [ ] Write RED runtime tests with a tiny local Qwen model and tokenizer fixture: same rendered prompt/256/512 limits, greedy `mode="single"`, V2 scorer call, one fsynced JSONL row per ID, complete ordered-ID reconciliation, preserved partial rows on generation exception, no overwrite, and measured CUDA-active duration. Run focused tests red. Implement `score_v2_examples` using V1 `score_v1_examples` record schema and `generate_answer` semantics but importing V2 scorer. Include a CPU tiny-model test path; real production load still requires CUDA. Run green.

```python
def test_v2_runtime_scores_graph_with_markdown_answer(tmp_path: Path, controlled_model, tokenizer, graph_example):
    signatures = {"source": "a" * 64, "scorer": "b" * 64}
    result = score_v2_examples(controlled_model, tokenizer, [graph_example], "single", tmp_path / "scores.jsonl", signatures)
    rows = [json.loads(line) for line in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert result["validated_count"] == 1
    assert rows[0]["completion_text"] == "**Answer: yes**"
    assert rows[0]["reason_code"] == "correct" and rows[0]["parsed_value"] == "yes"

# Implementation shape: durable raw row precedes aggregate publication.
stream.write(canonical_json(record))
stream.flush()
os.fsync(stream.fileno())
```

- [ ] Write RED controller tests for all six pilots scored once, first-eligible selection, candidate-1 fallback, fewer-than-two-family terminal stop before dev/sealed, development macro/global/two-family gate, review-sheet export only if dev exists, and identical source/scorer/signature checks after prepare. Implement `calibrate` with exclusive pilot split/score paths, authenticated summaries, `CudaCostLedger`, durable phase-start/completion markers, and 7,200-second CUDA/wall plus 20 GiB caps. Bind every raw hash/count to the stage record. Normal scientific stop returns `stopped_scientific`, `next_stage=report` and does not masquerade as a failure.
- [ ] Write RED failure tests for interrupted open phases (unknown exact CUDA duration, no retry), corrupted score row, changed split/sidecar after scoring, exhausted resource cap, live-process recovery refusal, and report with poisoned sealed bytes. Implement terminal `report` to rehash/recount all non-sealed stage-bound raw files, authenticate stage prerequisites and ledger, and seal `next_stage=complete` on a stop. A passing calibration reports `next_stage=frozen` but cannot execute `frozen` until a separately verified implementation exists. Do not add a fake passing stage.
- [ ] Implement CLI `--stage prepare|calibrate|report --config configs/recurrence-v2.toml [--run-id <64 lowercase hex>]`; emit one machine-readable JSON result, nonzero exit on resource/environment failure, and refuse missing/malformed run IDs. Run focused/full tests, Ruff, compileall, diff check and an independent Luna code review; fix reviewed defects with regression tests, then commit and push `exp` *before* real V2 output. Recompute and record frozen scorer/config/generator/runtime SHA256; no further code change under that run identity.

### Task 6: production calibration, audit, and branch decision

**Files:** Update `PROJECT_LOG.md`, `todo.md`; create `docs/recurrence-v2-calibration-report.md` only after observed results. Generated artifacts remain ignored.

- [ ] Read-only preflight: verify active branch `exp`, clean tracked worktree, exact source and V1 pilot hashes/counts, local pinned snapshot, CUDA capacity, disk headroom, and no existing deterministic V2 root. Do not open V0 sealed test rows. Run `uv run python scripts/run_recurrence_v2.py --stage prepare --config configs/recurrence-v2.toml` once and retain its printed full run ID.
- [ ] Run `uv run python scripts/run_recurrence_v2.py --stage calibrate --config configs/recurrence-v2.toml --run-id <printed-id>` once. If a wrapper times out, inspect child process, phase markers, raw outputs, and ledger before any recovery; never launch a competing or same-run retry against an open phase.
- [ ] Run `--stage report` for a terminal scientific/resource/environment stop. Independently recompute every non-sealed split/score SHA/count, summary, candidate choice, gate, cost ledger, and stage prerequisite hash; verify absence/presence of dev/sealed/training artifacts as appropriate. Report exact results and limitations, not an efficacy claim. Commit and push evidence documentation on `exp`.
- [ ] If the registered V2 calibration gate fails, end the study and do not run frozen controls or LoRA. If it passes, continue without a user pause by writing a separate detailed frozen-controls/paired-training execution plan under the approved V2 spec, then implement/test each stage before executing it. A later discovery/retention/confirmation gate failure is likewise terminal and reported.

## Stop and review ledger

Before each code task: record HEAD as its diff base; Luna implementer works on only that task; independently review both spec compliance and code quality before advancing. After any review fix, rerun covering tests and re-review the fix diff. The parent rules on tradeoffs and scientific gates. V2 artifact roots and all V0/V1 evidence are never rewritten. A registered gate stop is a final outcome, not permission for a third candidate or another prompt revision.
