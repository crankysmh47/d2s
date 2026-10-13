# Task 14 review — FAIL

Review date: 2026-09-18

Scope: read-only review of the Task 14 plan (steps 14.1–14.9 and the Task 22
report contract), `src/ppmoe/recurrence/report.py`, report/orchestration tests,
and `src/ppmoe/recurrence/experiment.py`. No implementation files were
changed by this review.

## Verification evidence

- `uv run pytest -q tests/test_recurrence_report.py`: **8 passed**.
- Task 14's requested focused suite (config through report): **288 passed, 2
  skipped**.
- `uv run pytest -q`: **595 passed, 2 skipped**.
- `uv lock --check`: passed.
- `git diff --check`: passed (only existing LF/CRLF warnings).
- The passing tests do not establish Task 14: the report test file has only
  eight cases and does not provide the complete negative/positive fixtures
  required by Step 14.1 (notably no complete G4 accuracy no-go, G5
  confirmation-failure, or G5 success fixture).

## Findings

### [BLOCKER] Incomplete and hashless aggregates are accepted

`_collect_summaries` only requires `records_path` when `complete is True`
(`report.py:216-232`). A valid JSON summary with
`{"kind":"exact_scores","complete":false}` and no raw records is appended
to `aggregates`, making the section status `evaluated` (`report.py:257-262`)
instead of `not_evaluated` or rejected. A complete summary with a raw path but
no declared raw hash is also accepted: the builder computes and inserts
`raw_sha256` (`report.py:235-246`) rather than requiring the registered hash
field. This violates the anti-fabrication requirement to reject aggregates
without raw score/chunk hashes and permits an incomplete evaluation to appear
as evidence.

Reproduction: starting from the existing `_root` fixture, adding an incomplete
summary produced `status == "evaluated"` with the summary in the report. Adding
a complete one-row summary without `records_sha256`/`raw_sha256` also succeeded.

### [BLOCKER] Aggregate counts are not reconciled

`_validate_raw` compares the raw row length only to `expected_count` (or, if
absent, `record_count`) (`report.py:265-268`). It never requires
`record_count == expected_count == len(raw_rows)`, and report totals trust the
unvalidated summary `record_count` (`report.py:257-262`). A one-row raw file
with `record_count: 999, expected_count: 1` was accepted and rendered
`examples.scored == 999`. This directly violates the required count checks and
can corrupt all reported denominators.

### [BLOCKER] Success does not prove the registered arm/seed/control matrix

`_require_success_evidence` checks only global arm and seed presence, one A2
entry, and one complete retention entry (`report.py:341-352`). It does not
require every `{F0,A1,A2,A3} × {47,48}` arm/seed pair, paired ID sets, all
three registered families, matching family assignments, bootstrap gates,
retention F0/candidate pairs, timing-control evidence, or complete failures.
Consequently, one aggregate per arm and seed can satisfy a `passed` payload
without being the registered two-seed study. This fails Step 14.2's “success
missing any arm/seed/control” requirement and Task 22's source reconciliation.

### [BLOCKER] Decision and evaluation authorization are forgeable

`_decision` accepts a `decision`, `gate_decision`, or `outcome` mapping from
any stage, including `source`, and accepts `status: "passed"` without requiring
the `confirm` gate or a validated persisted gate record (`report.py:318-337`).
`_validate_evaluation_authorization` merely checks whether `discover` or
`confirm` exists in either passed *or terminal* status
(`report.py:299-315`); it does not enforce the DAG, status, stage signatures,
or the correct producer for each arm/seed. A failed downstream stage therefore
authorizes score files that should not be consumable.

### [HIGH] Source-manifest inventory and hashes are not reconciled

`_load_source_manifest` validates only revision/hash syntax and that `files` is
a list (`report.py:185-213`). It does not resolve each `relative_path` inside
the run/source root, require that each path exists, or compare its byte count
and SHA-256 to the current file. The existing passing report fixture declares
`config.json` in the source manifest even though that file is not present in
the fixture tree. This fails the Task 22 requirement that every source path
exists and every recorded SHA matches.

### [HIGH] TOCTOU and symlink/junction handling are incomplete

`_read_inputs` hashes files once (`report.py:96-107`), but later loaders reread
the paths without checking the recorded size/hash. A file changed between the
inventory pass and aggregation can therefore be reported under the old source
hash. `_resolve_evidence_path` resolves the path before checking
`is_symlink()` (`report.py:502-508`), so an in-root junction/symlink component is
silently converted to its target and accepted. A Windows junction probe
(`scoreslink -> target` inside the run root) was accepted and rendered
`target/raw.jsonl`. The requested path traversal/symlink/TOCTOU protections are
therefore incomplete.

### [HIGH] Report Markdown is materially incomplete

The JSON object has placeholders for required content, but `_render_markdown`
only emits decision/conclusion/next action, signatures/stages/examples/families,
retention/resources, and limitations (`report.py:468-473`). It omits required
transitions, accuracy CIs, NLL/token counts, training steps/examples/times,
A3 validity, parameter counts, layer evaluations, failure history, protocol
deviations, source revision/files/hashes, and report-source hashes. It also does
not compare the Markdown content against the JSON beyond reusing a few fields,
so the required full report and Markdown reconciliation are not met.

### [HIGH] Retention and CI evidence is not independently validated

Retention raw rows are checked for basic field types only
(`report.py:280-289`); summary `nll_sum` and `scored_tokens` are not reconciled
to raw chunk sums/counts, and candidate/F0 chunk ID/token universes are not
checked. `_validate_ci_metadata` only checks that a few keys have the right
container types (`report.py:292-296`); it does not require 10,000 replicates,
registered seeds, finite bounds, paired sample-index metadata, or a matching
bootstrap vector. Thus decisive PPL/CI values can be supplied without
independent recomputation.

### [HIGH] Report replay is not stable after a report stage is written

`run_stage` performs report-root discovery before checking whether the report
stage already exists (`experiment.py:215-242`). After the first report,
`stages/report.json` sorts before `stages/source.json`; `_find_report_staging_root`
then calls `_any_signatures`, which selects the report record and rejects its
empty data signature. A direct probe showed: first report succeeds; second
report fails with `StagePrerequisiteError: report found an incompatible staging
candidate`. This violates the required replay behavior. In addition, when an
existing report record is found, the replay path returns it without checking
that `report.json`, `report.md`, and `report-sources.json` still exist and match
their persisted hashes (`experiment.py:231-242`).

### [MEDIUM] Report-stage “read-only” path can mutate resource evidence

`run_stage` unconditionally calls `_reconcile_all_chargeable_cuda` before the
report branch (`experiment.py:231`), even though the report branch is declared
read-only (`experiment.py:255-259`). If prior failure events are present,
reconciliation can append CUDA-cost ledger entries and anchor files while
building/replaying the report. This violates the post-stop safe/read-only
contract and can change the report's resource inputs during reporting.

### [MEDIUM] Strict schema and failure-history validation is incomplete

Stage records accept missing `payload`/`attempts` by silently defaulting them
(`report.py:138-147`), accept unknown stage filenames (`report.py:129-155`),
and do not validate attempt types/ranges or stage dependency/order. Malformed
JSON evidence is swallowed by `_collect_summaries` (`report.py:223-227`), so a
corrupt summary can disappear and become `not_evaluated`. Failure history only
recognizes a narrow set of event kinds/outcome prefixes
(`report.py:355-364`), with no event schema or completeness check. These gaps
conflict with strict schema, exact terminal-status, and “unreported failures”
requirements.

## Overall decision

**FAIL — Task 14 is not review-ready.** The implementation has useful basic
finite-value, duplicate-ID, path-containment, and immutable-output mechanics,
and the available tests pass, but the blockers above permit fabricated or
incomplete success reports, fail the required full-content/independent-source
reconciliation, and break report replay. Do not treat Task 14 as complete or
advance to production execution until the blockers are fixed and the missing
registered terminal fixtures and mutation tests are added.
