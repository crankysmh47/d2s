# Recurrence Answer Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose V2 answer extraction against blind model review without changing its signed scientific stop.

**Architecture:** Authenticate the existing V2 development split and raw scores, export gold-free packets with a private reconciliation manifest, collect separate Luna extraction files, and compare those claims with the deterministic interpreter. Source data and terminal stage files are read-only; output lives in a new diagnostic artifact root.

**Tech Stack:** Python 3.11, stdlib JSON/hashlib/random/pathlib, pytest, Ruff, existing `ppmoe.recurrence` data/stage helpers.

**Spec:** `docs/superpowers/specs/2026-09-27-recurrence-answer-audit-design.md`.

## Global Constraints

- Work on `exp`; do not edit V0/V1/V2 scorer or signed artifact files. Do not open V0 test or V2 sealed split or sidecar. Preserve user-owned untracked `task-14*.md` files.
- V2 run ID is `18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d`; only its `data/dev.jsonl`, `scores/dev-f0.jsonl`, and signed stages are diagnostic inputs.
- Export all 93 `invalid_format` rows and ten accepted rows per family, selected from sorted IDs using one `random.Random(927001)` in family order `expressions`, `lists`, `graphs`. Assert the source count 300 and selection count 123; reject changed source rather than adapting the count.
- Packet rows contain only `audit_id`, `family`, `question`, `response`, `stop_reason`. No gold, latent specification, V2 score, correctness, or selection reason. The manifest alone contains private original IDs, selection reasons, source hashes, and packet hashes.
- Reviewer files are model review, never human review. Never infer a missing grade, repair a review row silently, or turn audit estimates into official V2 scores.
- No GPU inference or V3 benchmark generation in this plan. Later scorer qualification and fresh V3 validation require a separate conditional implementation plan after the audit report.

## Review Focus

1. A changed V2 development split or score byte must reject packet export before creating an audit root (Task 1 test).
2. Packet bytes must not reveal reference answers, latent specs, scorer fields, or selection reason (Task 1 test).
3. Two rows with the same audit ID, a missing row, or a wrong family/type must reject reconciliation (Task 2 test).
4. A reviewer claim with a value absent from its response must be flagged for adjudication rather than accepted as model evidence (Task 2 test).
5. A non-EOS response marked conclusive must be rejected as inconsistent (Task 2 test).

## File map and interfaces

| File | Responsibility |
| --- | --- |
| `src/ppmoe/recurrence/audit_v2.py` | Authenticate V2 evidence, select rows, export blind packets, validate/reconcile review files |
| `scripts/run_recurrence_answer_audit.py` | `export` and `reconcile` CLI with one JSON status/result |
| `tests/test_recurrence_audit_v2.py` | Source-tamper, blind-packet, selection, review-schema, and reconciliation tests |
| `artifacts/recurrence-answer-audit-v2/<audit-id>/` | Ignored immutable packets, manifest, model reviews, and diagnostic result |
| `docs/recurrence-v2-answer-audit-report.md` | Human-readable diagnostic finding after reviews complete |

Interfaces:

```python
def verify_v2_audit_source(run_root: Path) -> dict[str, object]: ...
def select_v2_audit_rows(source: Mapping[str, object], seed: int = 927001) -> dict[str, tuple[dict[str, object], ...]]: ...
def export_v2_audit(run_root: Path, output_root: Path) -> dict[str, object]: ...
def reconcile_v2_audit(run_root: Path, audit_root: Path) -> dict[str, object]: ...
```

### Task 1: authenticated blind packet export

**Files:** Create `src/ppmoe/recurrence/audit_v2.py`, `scripts/run_recurrence_answer_audit.py`, `tests/test_recurrence_audit_v2.py`.

- [ ] Write RED tests using small fixture stage records and score/split bytes. Require canonical stage hashes/prerequisite linkage, `dev-f0` split/sidecar/score hashes and 300 ordered IDs, read-only source behavior, and a changed score byte rejected before output-root creation. Do not read `sealed.jsonl` even when that path is poisoned. Run `uv run pytest tests/test_recurrence_audit_v2.py -q` and observe missing API.
- [ ] Implement `verify_v2_audit_source` by reusing authenticated V2 stage readers and independently comparing nonsealed file SHA/count/ordered IDs. Require `calibrate.status == report.status == 'stopped_scientific'`, `next_stage == 'complete'`, `dev-f0` evidence, and the exact registered run ID. Return only needed in-memory dev split/score rows, hashes, and stage hashes. Never write during verification.
- [ ] Write RED selection tests for the exact 93 rejected and 30 accepted rows, ten accepted per family using sorted-ID input and one seeded PRNG; require 123 distinct IDs. A reordered score file must fail ordered-ID authentication, not be treated as equivalent. Test that the packet row schema has exactly the five allowed keys and contains no gold or scorer metadata. Implement `select_v2_audit_rows`.
- [ ] Write RED exclusive-export tests. `export_v2_audit` computes audit ID from source/stage hashes plus seed, verifies source *before* creating root, writes one `packets/<family>.jsonl` per family plus `manifest.json` exclusively, fsyncs each file, records ordered audit IDs and raw packet hashes, and refuses an existing root. Use audit IDs as the first 16 hex characters of `sha256('v2-audit:' + example_id)`, check collision. The manifest maps IDs to original IDs and hidden selection reasons. The CLI `--stage export --run-root <absolute-or-project-relative> --output-root artifacts/recurrence-answer-audit-v2` prints compact JSON. Run focused/full pytest, Ruff, compileall, `git diff --check`; commit scoped code/tests.

### Task 2: model review validation and diagnostic reconciliation

**Files:** Extend `audit_v2.py`, CLI, tests.

- [ ] Write RED tests for one JSONL review per family at `reviews/<family>-luna.jsonl`, each line exactly `audit_id`, `family`, `status`, `claimed_value`, `evidence_span`, `reason`, `reviewer_id`. Status is one of `conclusive|ambiguous|missing|truncated`; only conclusive has a typed non-null value (`int`, `list[int]`, or `yes|no` by family). Require all packet IDs once, no extra IDs, correct family, conclusive evidence substring of response, and non-EOS rows marked truncated. Reject all malformed rows before writing results.
- [ ] Implement `reconcile_v2_audit` to reauthenticate V2 source and packet/manifest hashes, validate all three review files, compare conclusive claims with `interpret(latent_spec, family)`, flag reviewer/V2 disagreements, and produce exact per-family counts for correct/wrong/ambiguous/missing/truncated. Write `diagnostic.json` exclusively with source hashes, review-file hashes, and every row's original V2 reason plus independent model claim. Do not modify review files or V2 scores. Add tests showing a wrong reviewer value remains wrong, a correct list under `Final list` is classified as a parser miss, and conflicting cues remain ambiguous.
- [ ] Extend CLI with `--stage reconcile --run-root ... --audit-root ...`; print compact JSON. Run focused/full pytest, Ruff, compileall, `git diff --check`; commit and push verified code before reviewing production answers.

### Task 3: execute blind review and report the decision boundary

**Files:** Generated ignored audit root; create `docs/recurrence-v2-answer-audit-report.md`; update `PROJECT_LOG.md` and `todo.md`.

- [ ] Read-only preflight: verify `exp`, tracked cleanliness, V2 terminal stages, exact source hashes, and no existing deterministic audit root. Execute export once and record full audit ID and packet hashes. Do not open sealed data.
- [ ] Dispatch three GPT-6 Luna reviewers, one per family packet, with no gold or V2 scores. Each reads its packet and writes its own JSONL review with all IDs exactly once. The parent checks reviewer files exist and are limited to their assigned family. If a reviewer cannot finish, preserve partial review and report `incomplete` rather than invent labels.
- [ ] Run reconcile once. Inspect every flagged disagreement independently and keep original model labels immutable. If gate interpretation remains sensitive to the unaudited accepted rows, expand the blind review to all 300 under a new diagnostic packet identity before claiming a family is informative; do not modify the first packet/review.
- [ ] Write a report with source/review hashes, per-family correct/wrong/ambiguous/missing/truncated counts, parser-miss examples, accepted-sample false-accept findings, uncertainty, and the recommendation for either a narrowly specified V3 scorer or a new task benchmark. State the V2 signed no-go separately. Commit and push diagnostic documentation. A V3 implementation plan follows only if the audit gives a defensible, bounded grammar repair.
