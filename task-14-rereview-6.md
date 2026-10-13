# Task 14 final review (round 6) — FAIL

Review date: 2026-09-19

Scope: final independent review of the registered bootstrap RNG contracts,
per-seed retention evidence, producer/report comparison semantics, failure
event identity, Windows handle cleanup, and all earlier Task 14
anti-fabrication, schema, replay, and read-only requirements. No
implementation files were changed or committed; this file is the only
round-six artifact.

## Verification

- `uv run pytest -q tests/test_recurrence_report.py tests/test_recurrence_experiment.py tests/test_recurrence_evaluate.py tests/test_recurrence_gates.py`: **155 passed, 1 skipped**.
- `uv run pytest -q`: **622 passed, 2 skipped**.
- `uv lock --check`: passed.
- `git diff --check`: passed.
- Direct adversarial probes below were run against the current checkout.

## Checks that pass

- `_read_bootstrap_matrix()` now requires the exact registered family order,
  ID universe, 10,000 rows, valid index ranges, and exact equality with
  `np.random.default_rng(9047)` for accuracy or `np.random.default_rng(9157)`
  for retention (`report.py:826-858`; producer calls in `gates.py:827-831` and
  `gates.py:889`). An independently generated exact matrix was accepted;
  replacing it with a shape- and hash-valid all-zero matrix was rejected:
  `EXACT_MATRIX_ACCEPTED True`, `BIASED_MATRIX_REJECTED True`.
- Accuracy remains derived from the registered three-family, per-seed
  unweighted macro, and retention derives distinct seed-47 and seed-48
  vectors/paths/hashes and checks both full-data points and bootstrap upper
  bounds (`report.py:873-953`). Negative per-seed macro and bad full-data
  retention probes were rejected: `NEGATIVE_SEED_MACRO True`,
  `BAD_FULL_RETENTION_POINT True`.
- Windows evidence reads use handle-relative `NtOpenFile` chains with
  `FILE_OPEN_REPARSE_POINT`, and POSIX reads use `openat`/`O_NOFOLLOW`
  (`report.py:260-317`). One hundred successful nested reads and twenty
  missing-file/error reads produced no Windows handle-count increase:
  `HANDLE_COUNT 299 299 DELTA 0` and `ERROR_HANDLE_COUNT 299 299 DELTA 0`.
  Existing root/nested reparse, descriptor, source inventory, replay, and
  component-swap probes pass.
- Per-kind event schemas reject unknown/missing fields, parse UTC timestamps,
  enforce monotonic indexes and registered stage/arm/seed/attempt/update
  ranges, and require complete training-failure and failed-environment
  linkage (`report.py:1158-1269`). Cross-arm/seed linkage was rejected:
  `REJECTED ValueError failed-environment event crosses failure stage/arm/seed identity`.
- Earlier checks remain passing: strict read/hash and hidden-file inventory,
  source/summary/event schema and producer signatures, complete failures,
  CUDA ledger/chain checks, raw-derived bootstrap values, report Markdown
  reconciliation, immutable report/source manifests, repeated-build hashes,
  stale/tampered/missing/extra evidence rejection, stopped/failed decision
  anti-smuggling, and read-only report execution.

## Residual findings

### [HIGH] Actual paired-bootstrap producer points are incompatible with report validation

`_has_registered_bootstrap()` requires each persisted comparison `point` to
equal the mean of the bootstrap vector (`report.py:788-796`). The authorized
accuracy producer instead stores the observed raw three-family macro as
`point` (`gates.py:844`), while the authorized retention producer stores the
full-data token-weighted ratio as `point` (`gates.py:891`). Those values are
not generally equal to the bootstrap-vector mean.

Independent probes using nonconstant valid records demonstrated the mismatch:

```text
ACCURACY_POINT 0.3333333333333333 BOOTSTRAP_MEAN 0.3300777777777778 DIFF 0.0032555555555555116
RETENTION_POINT 1.005012520859401 BOOTSTRAP_MEAN 1.005033935239709 DIFF 2.141438030811571e-05
```

Thus a correct artifact emitted by the registered `paired_accuracy_bootstrap`
or `paired_retention_bootstrap` contract can be rejected by the report, while
an artifact that fabricates the bootstrap mean as its `point` can satisfy this
check. The report must compare `point` to the producer-defined raw/full-data
point and compare only `ci_low`/`ci_high` to the persisted bootstrap vector.
This is a decisive integration and anti-fabrication failure.

### [MEDIUM] Training-failure event IDs are only shape-checked, not recomputed

The event validator requires a 64-hex `event_id` but does not recompute it
from the canonical Task 9 identity tuple (`report.py:1222-1252`). The
canonical producer formula is `_task9_failure_event_id()` in
`train.py:949-956`, hashing `kind`, `stage`, `arm`, `seed`, `attempt`,
`update`, `checkpoint_path`, `checkpoint_sha256`, and
`checkpoint_manifest_digest`.

An otherwise complete registered training-failure event with
`event_id = "a" * 64` was accepted by `_load_events`; the same arbitrary ID
can then be repeated in a linked failed-environment event. Cross-identity
linkage is now rejected, but a hash-looking fabricated identity remains
accepted. The validator must recompute and compare the canonical event ID for
each training failure before accepting or linking it.

## Overall decision

**FAIL — Task 14 is not ready for acceptance.** Round six closes the
registered-RNG, biased-matrix, separate-retention-vector, per-seed gate,
Windows handle-cleanup, and failure-linkage residuals. However, the current
report does not faithfully accept the actual paired-bootstrap producer point
contract, and failure event IDs remain forgeable despite their 64-hex shape.
Both are provenance/anti-fabrication defects requiring implementation fixes
and a fresh replay/adversarial review.
