# Task 14 final review (round 7) — PASS

Review date: 2026-09-19

Scope: targeted closure review of the round-six producer-point and failure-
identity findings, plus regression of the registered bootstrap, retention,
replay, schema, path-safety, and read-only gates. No implementation files
were changed or committed; this file is the only round-seven artifact.

## Verification

- `uv run pytest -q tests/test_recurrence_report.py tests/test_recurrence_experiment.py tests/test_recurrence_evaluate.py tests/test_recurrence_gates.py`: **158 passed, 1 skipped**.
- `uv run pytest -q`: **625 passed, 2 skipped**.
- Targeted adversarial report tests: **5 passed, 33 deselected**.
- `uv lock --check`: passed.
- `git diff --check`: passed (only existing LF/CRLF warnings).

## Round-seven closure checks

- Nonconstant authorized accuracy evidence was independently generated with
  `paired_accuracy_bootstrap(..., seed=9047)`. The producer point matched the
  raw observed three-family macro (`0.9166666666666666`) while the bootstrap
  vector mean differed (`0.9162666666666668`). The report now derives the raw
  point independently from persisted paired records and checks it separately
  from the vector percentiles (`report.py:784-800`, `report.py:880-944`).
- Nonconstant authorized retention evidence was independently generated with
  `paired_retention_bootstrap(..., seed=9157)`. The producer point matched the
  full-data token-weighted ratio (`2.45960311115695`) while the bootstrap mean
  differed (`2.1591873093300586`). The report now checks this full-data point
  separately and checks only CI endpoints against the vector percentiles.
- A fabricated bootstrap mean substituted for the raw accuracy point was
  rejected by the registered report gate. The focused regression
  `test_registered_bootstrap_reconciles_accuracy_point_to_raw_producer_semantics`
  passed, as did the nonconstant producer-point and distinct retention-vector
  tests.
- The canonical Task 9 identity is now shared through
  `event_identity.task9_failure_event_id()` (`event_identity.py:5-20`), used by
  both training and report validation (`train.py:950-951`, `report.py:1365-1366`).
  An independently constructed complete failure plus matching
  `failed_environment` link was accepted; changing its ID to arbitrary
  64-hex (`"a" * 64`) was rejected with:
  `ValueError training failure event identity does not match canonical Task-9 fields`.
- Prior round-six gates remain covered by the passing full suite: exact
  registered RNG matrices and biased-matrix rejection, complete per-seed
  retention vectors/points/CIs, full-point and upper-bound gates, strict event
  schemas and failure linkage, no-follow descriptor/reparse protections,
  immutable/replay report artifacts, source inventories, and read-only report
  behavior.

## Decision

**PASS — the round-six residuals are closed.** Authorized nonconstant producer
points now reconcile to raw/full-data statistics rather than bootstrap means;
bootstrap CIs reconcile to the persisted vectors; and every training failure
uses a recomputed canonical Task 9 identity that linked failed-environment
records must reference. No remaining Task 14 finding was reproduced.
