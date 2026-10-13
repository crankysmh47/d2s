# Task 14 re-review (round 2) — FAIL

Review date: 2026-09-18

Scope: adversarial re-review of every residual in
`task-14-rereview-1.md`, including the report builder, evaluator-produced
summary contracts, orchestration replay, CUDA ledger, and report tests. No
implementation files were changed or committed; only this review was added.

## Fresh verification

- `uv run pytest -q tests/test_recurrence_report.py tests/test_recurrence_experiment.py tests/test_recurrence_gates.py`: **118 passed, 1 skipped**.
- `uv run pytest -q`: **608 passed, 2 skipped**.
- `uv lock --check`: passed.
- `git diff --check`: passed (existing LF/CRLF warnings only).

## Round-two checks that now pass

- A realistic source→data signature progression (`data: ""` → finalized
  SHA-256) now reports successfully; `_validate_stage_dag` permits only that
  registered transition (`report.py:175-195`) and `_signatures` carries the
  finalized value (`report.py:240-254`).
- A valid `CudaCostLedger` entry and anchor are reportable; ledger chain,
  arithmetic, integrity, anchor, and pending-transaction checks run in
  `_validate_cuda_ledger` (`report.py:685-716`).
- Nested and run-root Windows junction probes are rejected by lstat/reparse
  checks (`report.py:197-223`).
- Replay now rejects added, changed, and removed non-output files by comparing
  the current inventory with `report-sources.json` (`experiment.py:1222-1230`).
- Summary signatures are bound to the authorized producer stage
  (`report.py:381-404`).
- Negative raw retention NLL is rejected (`report.py:428-443`).
- A bare `a3_valid: true` is no longer timing evidence; timing now derives a
  ratio from persisted A2/A3 samples for both seeds (`report.py:625-658`).
- The round-one matrix/family/ID checks, incomplete/hashless handling, count
  reconciliation, Markdown reconciliation, and output-tamper replay checks
  remain passing.

## Residual findings

### [BLOCKER] Valid evaluator retention summaries are rejected by the report schema

`evaluate._finalize_retention` emits `schema_version`, kind/counts, token/NLL
metrics, `mean_nll`, `perplexity`, mode, arm, seed, signatures, and raw hashes
(`src/ppmoe/recurrence/evaluate.py:445-465`). It does not emit the required
`families` or `failures` fields. Conversely, `_validate_summary_schema`
requires `families` and `failures` for every complete summary and does not allow
`mean_nll` (`report.py:381-399`). An independent fixture matching the actual
evaluator output was rejected with:
`ValueError: summary retention/f0.summary.json has an invalid schema`.

This means a real retention result cannot reach G6 reporting, regardless of the
scientific outcome. The producer and consumer schemas must be made identical
and tested from the actual evaluator output.

### [HIGH] Bootstrap paired matrices are not validated against the registered universe

`_has_registered_bootstrap` verifies a 10,000-element vector, SHA-256 files,
nonnegative integer rows, and comparison-name presence, but it does not verify
the paired matrix's family mapping, column count, ID universe, or shared sampled
indices across the required comparisons (`report.py:563-602`). I wrapped the
existing full-success fixture, replaced each persisted paired matrix with
10,000 one-column rows, updated its declared hash, and the success report still
passed: `MALFORMED_PAIRED_MATRIX_ACCEPTED`.

The registered gate emits family-keyed matrices whose columns correspond to
the common IDs. Arbitrary one-column matrices cannot support the required
paired accuracy/retention claim.

### [HIGH] A passed success decision still does not require decisive CI/gate values

The success fixture's vectors are all zeros and its stage payload contains only
hashes, paths, and comparison labels; no recomputed CI bounds or gate values are
required. `_has_registered_bootstrap` recomputes percentiles only when optional
CI fields happen to be present (`report.py:594-601`), while
`_require_success_evidence` accepts the presence test and the stage's arbitrary
`decision: {status: passed}` (`report.py:520-560`). Thus a vector can be
persisted/read/hashed yet still be unrelated to the reported gate outcome.
Success must bind the persisted vectors and paired matrices to the registered
comparisons, recompute all required bounds, and require the decisive values and
passes for both seeds.

### [MEDIUM] Timing accepts one-sample “production” controls

The new timing logic correctly rejects a bare boolean and requires both seeds,
but it accepts lists of any positive length. A direct probe with one A2 and one
A3 sample for each seed returned `ONE_SAMPLE_TIMING True`. The registered pilot
and realized-control contract should validate the expected sample count and
persisted per-seed timing metadata before allowing a passed conclusion.

### [HIGH] Evidence reads remain path-based rather than descriptor-bound

`_read_verified` performs lstat/reparse checks, then calls `path.stat()`,
`path.read_bytes()`, and `path.stat()` (`report.py:225-237`). A concurrent
replacement after the no-follow check can point the path at a same-size,
same-hash target and pass the size/hash comparison. This is not a descriptor-
bound read and does not satisfy the requested TOCTOU defense. The raw evidence,
JSONL, source inventory, and report replay reads need an opened no-follow
descriptor/handle whose identity is checked before and after consumption.

### [MEDIUM] Hidden evidence is excluded from the authoritative inventory

`_read_inputs` skips any path containing a dot-prefixed component
(`report.py:102-119`). Replay therefore accepts a newly added hidden evidence
file: a probe added `.hidden-evidence` after the first report and replay
returned successfully (`HIDDEN_STALE_ACCEPTED`). This violates exact current
non-output inventory reconciliation and can undercount disk usage. Only the
explicit report outputs (and safely scoped temporary files, if any) should be
excluded.

### [MEDIUM] Event schema/failure completeness remains permissive

`_load_events` validates stage/attempt types and monotonic indexes only when an
index is present (`report.py:821-842`). It accepts events without index,
timestamp, registered kind/outcome, or required payload fields. A
`{"stage":"pilot","attempt":1,"outcome":"error"}` event was accepted and
omitted from `failure_history` (`[]`). This leaves unreported-failure and
non-monotonic-ledger gaps; every persisted event should use the append-event
schema and every failure outcome should be represented in the report.

## Overall decision

**FAIL — Task 14 is not ready for acceptance.** Round two fixes now protect the
basic stage transition, valid CUDA ledger, reparse paths, stale ordinary
inputs, summary producer signatures, and negative retention values. However,
the evaluator/report retention-schema mismatch prevents valid real reports,
and unsupported success claims remain possible through malformed paired
bootstrap matrices and unbound gate/CI values. Descriptor-bound TOCTOU,
hidden-inventory, timing sample-count, and event-schema gaps also remain.

