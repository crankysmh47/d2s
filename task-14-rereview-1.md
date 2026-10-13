# Task 14 re-review (round 1) — FAIL

Review date: 2026-09-18

Scope: re-review of every finding in `task-14-review.md` after the round-one
changes. Only this review file was added; no implementation files were edited
or committed.

## Fresh verification

- `uv run pytest -q tests/test_recurrence_report.py tests/test_recurrence_experiment.py tests/test_recurrence_gates.py`: **113 passed, 1 skipped**.
- `uv run pytest -q`: **603 passed, 2 skipped**.
- `uv lock --check`: passed.
- `git diff --check`: passed (existing LF/CRLF warnings only).

The new report tests cover incomplete/hashless summaries, count mismatch,
source-file tampering, a complete matrix fixture, G4/G5 terminal fixtures,
source decision isolation, and Markdown generation. They do not cover the
real monotonic signature sequence, resource-ledger reports, stale-input replay,
reparse roots/junctions, summary signature binding, or bootstrap-vector
reconciliation.

## Resolved checks

- Incomplete summaries are skipped as `not_evaluated`, and complete summaries
  require a declared raw hash (`report.py:282-303`).
- `record_count`, `expected_count`, and raw length must agree; raw failure
  counts are checked (`report.py:350-386`).
- Source inventory paths are now checked for existence, inventory membership,
  byte count, and SHA-256 (`report.py:260-279`).
- A complete success fixture requires all eight arm/seed entries, three
  families, common ID/family universes, paired retention universes, and zero
  failures (`report.py:463-503`).
- Source/data payloads cannot directly authorize a scientific `passed` decision;
  decision selection is limited to frozen/discover/confirm (`report.py:428-460`).
- Markdown now serializes the required report sections and replay compares
  regenerated Markdown to `report.json` (`report.py:660-669`,
  `experiment.py:1193-1218`).
- Basic first/second report replay succeeds, and tampering `report.md` is
  rejected with `StageError: persisted report Markdown does not reconcile with
  JSON`.
- Report entry no longer calls CUDA reconciliation (`experiment.py:231-232`
  now guards it with `stage != "report"`).

## Residual findings

### [BLOCKER] Real stage signature progression is rejected

`_validate_stage_dag` requires each stage's complete signature mapping to be
byte-identical to its predecessor (`report.py:166-176`), and `_signatures`
rejects any differing non-empty field (`report.py:214-229`). The orchestrator
intentionally finalizes `data` by adding a data hash after the source stage;
therefore a real tree has source signatures `{source:S,data:"",config:C,code:K}`
and data signatures `{source:S,data:D,config:C,code:K}`. An independent probe
with exactly this source/data pair failed:
`ValueError: stage data signature mismatch with source`.

This prevents reporting any finalized run beyond the source stage and violates
the required signed stage chain. Validation must bind each stage to the
predecessor's unchanged fields while allowing the registered stage to add its
own finalized digest.

### [BLOCKER] Reports containing `cuda-cost.jsonl` crash

`_resource_evidence` calls `_read_jsonl(ledger, root)` without the required
`inventory` argument (`report.py:559-566`), while `_read_jsonl` requires
`(path, root, inventory)` (`report.py:683-701`). A fixture containing one valid
CUDA ledger line failed with:
`TypeError: _read_jsonl() missing 1 required positional argument: 'inventory'`.

Thus resource/cost evidence cannot be rendered, and the report cannot satisfy
the required memory/disk/cost and read-only post-stop path.

### [HIGH] Junction/reparse protections remain bypassable

`_reject_reparse_path` calls `current.stat()` and checks the target's file
attributes (`report.py:180-197`). On Windows, a directory junction's followed
target reports ordinary attributes; a probe with `scoreslink -> target` inside
the run root was accepted and rendered `target/raw.jsonl`. A run-root junction
was also accepted because `build_report` resolves the root before checking it
(`report.py:33-38`).

The requirement explicitly includes symlink/junction/reparse rejection. The
path checks need no-follow handle/descriptor identity checks for every
component, including the root and the raw evidence file.

### [HIGH] Replay accepts stale/new evidence after the first report

`_verify_report_artifacts` validates only the persisted report JSON digest,
source-manifest digest, and Markdown (`experiment.py:1193-1218`). It does not
reconcile current evidence files against `report-sources.json`. A probe built a
report, then added `extra.txt`, and a second `run_stage("report")` returned
`report` successfully. This violates the stale-report-input requirement: replay
must fail if any non-output input is added, removed, or changed.

### [HIGH] Summary signatures are still ignored

`_collect_summaries` validates raw hashes/counts but never validates a summary's
`signatures` field or binds it to the authorized stage (`report.py:282-341`).
An A2/47 complete summary carrying four all-zero signatures was accepted after
a passed discover stage. This permits evidence from another protocol/code/data
revision to be included in a report.

### [HIGH] Registered bootstrap evidence can be fabricated with arbitrary hashes

`_has_registered_bootstrap` accepts any 64-hex `bootstrap_sha256` and paired
metadata hash without reading or hashing a vector (`report.py:506-521`). The
passing success fixture supplies arbitrary repeated-character hashes and no
10,000-value bootstrap vector; the report accepts it. It also does not verify
that the vector/hash, paired sample indices, comparison labels, and registered
seed correspond to the decisive aggregate. This fails the requested CI
10,000/seed/shared-paired-metadata/vector-hash contract.

### [HIGH] Timing gate validity is forgeable

`_has_timing_gate` accepts a bare `a3_valid: true` without a realized A3/A2
training-time ratio or training-time samples (`report.py:524-532`). A payload
containing only that boolean satisfies the success timing requirement. The
registered `[0.90, 1.10]` control must be derived from persisted measured times,
not an unchecked flag.

### [MEDIUM] Summary and event schemas are still not strict enough

Stage records and basic summaries are improved, but summary `schema_version`,
arm/seed types, signatures, registered family set, and unknown-field contract
are not validated for every evaluated summary (`report.py:293-317`). Exact raw
rows only require `family` to be a string and can contain an empty/unregistered
family outside a successful matrix. Source manifests allow duplicate inventory
entries and accept a non-integer `bytes` value via `int(...)`
(`report.py:266-278`). Events only require one of `kind/stage/outcome` and an
optional nonnegative index (`report.py:672-680`); they do not validate event
schema, monotonic indexes, stage/attempt identity, or that all failure events
are represented. These remain gaps against the strict stage/summary/event and
complete-failure-history requirement.

### [MEDIUM] Raw retention physical-domain validation is incomplete

Retention totals and contiguous chunk IDs now reconcile, but raw NLL values
are checked only for finiteness (`report.py:372-386`). A raw chunk with
`nll_sum: -1.0` and matching summary was accepted. The registered retention
metric is a nonnegative NLL sum; negative values must be rejected before
aggregation.

## Overall decision

**FAIL — Task 14 is not ready for acceptance.** The round-one work fixed the
original basic report-shape and anti-fabrication tests, but the signature-chain
blocker and CUDA-ledger crash prevent reporting real completed runs. Reparse
and stale-input replay gaps undermine evidence integrity, while bootstrap,
timing, summary-signature, and strict-schema gaps still allow unsupported
success claims. Do not advance to production execution until these findings
are fixed and independently covered by adversarial tests.
