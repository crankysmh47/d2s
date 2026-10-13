# Task 14 re-review (round 4) — FAIL

Review date: 2026-09-18

Scope: independent re-review of round-three fixes and residuals, including
raw-derived per-comparison bootstrap values/CIs/gates, actual paired
bootstrap output compatibility, decision/timing authorization, stable
no-follow descriptor chains, and exact event schemas. No implementation files
were changed or committed; only this review was added.

## Verification

- `uv run pytest -q tests/test_recurrence_report.py tests/test_recurrence_experiment.py tests/test_recurrence_evaluate.py tests/test_recurrence_gates.py`: **147 passed, 1 skipped**.
- `uv run pytest -q`: **614 passed, 2 skipped**.
- `uv lock --check`: passed.
- `git diff --check`: passed (existing LF/CRLF warnings only).
- Direct `paired_accuracy_bootstrap()` probe: 10,000 values for each of
  `A2-F0`, `A2-A1`, and `A2-A3`, with distinct comparison vectors and one
  shared family sample-index matrix.

## Round-four checks that pass

- Fabricated hash-consistent accuracy vectors/points now fail report
  validation: replacing one persisted vector with all `2.0` and point `999`
  produced `ValueError: success report is missing registered bootstrap/gate
  evidence`.
- Stopped/failed-stage decision smuggling is rejected. `_decision()` returned
  `stopped_scientific` and `failed_environment` for stopped/failed records
  carrying payload `passed` values.
- Timing is now limited to passed `discover`/`confirm` records, requires the
  expected seeds 47/48, 20 positive samples, five warmups, A2/A3 metadata,
  synchronization, and the derived ratio. Unauthorized source/data timing
  returned `False`; correctly staged timing returned `True`.
- UTC timestamps are parsed, unknown event fields and non-monotonic indexes
  are rejected, and per-kind required-field checks reject underspecified
  events. Hidden inventory, exact replay reconciliation, descriptor reads on
  POSIX, source/data signatures, matrices/IDs/hashes, actual retention output,
  CUDA ledger, Markdown, and prior round fixes remain passing.

## Residual findings

### [HIGH] Accuracy recomputation does not enforce the registered per-seed macro gate

`_derive_bootstrap_comparisons()` checks per-seed pooled correctness
improvement across all IDs (`report.py:789-842`) rather than the required
unweighted macro improvement across the three families for each seed. It only
checks two-seed family means and the resulting bootstrap vector threshold.

Independent direct probe used 100 expression IDs and one ID in each of lists
and graphs. For seed 47, A2 improved every expression but regressed both small
families, so its registered unweighted macro gain was `-1/3`; the pooled
per-ID gain remained positive because expressions dominated. Seed 48 improved
all families. The helper nevertheless returned a valid 10,000-vector:

```text
DERIVE_ACCEPTS_NEGATIVE_SEED_MACRO True 0.3333333333333333
```

The report can therefore accept a success evidence set that fails the G5
`each_seed_a2_*_positive` macro predicates. Recompute and require each
seed's unweighted family macro gain, then reconcile two-seed means and family
regression gates before accepting vectors/CI values.

### [HIGH] Retention recomputation does not enforce each seed's full-data PPL point

The retention branch of `_derive_bootstrap_comparisons()` only checks each
resampled seed ratio and returns the bootstrap vector (`report.py:833-856`).
It never computes/rejects the registered full-data A2/F0 PPL ratio per seed.

Independent direct probe used two chunks with F0 NLLs `[0, 0]` and A2 NLLs
`[0, 100]`, one token each. The full-data ratio is `exp(50)`, far above the
registered `1.02` limit, but the persisted matrix sampled only chunk 0 for all
10,000 replicates. The helper accepted it and returned all-one ratios:

```text
DERIVE_ACCEPTS_BAD_FULL_RETENTION_POINT True 1.0
```

The report must independently compute the full-data token-weighted PPL ratio
for each seed and require it, as well as the paired bootstrap upper bound, to
meet the registered threshold.

### [HIGH] Windows parent handles are not used to open the evidence file

The POSIX path uses an anchored `dir_fd` chain (`report.py:260-268`), but the
Windows branch only opens parent handles and then resolves the evidence again
through the original path:

```text
report.py:284-292  CreateFileW(str(parent), ...)
report.py:292       CreateFileW(str(path), ...)
```

The parent handles are closed after the read but never passed as a relative
open root. A parent component can be replaced with a junction/reparse point
between the parent checks and `CreateFileW(str(path))`, escaping the trusted
root despite the stable-handle bookkeeping. The Windows implementation needs
a true handle-relative/open-by-handle chain (or equivalent race-resistant
identity proof), with component replacement tests; the POSIX `openat` chain is
the required model.

### [MEDIUM] Per-kind event fields are not fully type- or identity-validated

Although `_load_events()` now parses UTC and enforces a required-field set
(`report.py:1058-1120`), several required values are accepted with arbitrary
types and some producer events lack the globally required stage/payload/
outcome identity. For example, this event was accepted:

```json
{
  "index": 0,
  "timestamp": "2026-09-18T10:00:00+00:00",
  "kind": "training_started",
  "arm": 999,
  "seed": "47",
  "attempt": 1,
  "update": []
}
```

Probe output was `EVENT_TYPES_ACCEPTED [...]`. Registered arms, integer seed/
update/cursor values, checkpoint hash/path types, failure message/error types,
stage binding, and exact outcome/payload requirements need per-kind schema
validation. Otherwise malformed events can still enter event history and
weaken complete failure reporting even though index/timestamp/unknown-field
checks pass.

## Overall decision

**FAIL — Task 14 is not ready for acceptance.** Round four closes fabricated
vector/point acceptance, stopped/failed decision smuggling, unauthorized
timing, and weak timestamp/field-presence checks. The actual gate output shape
is compatible with distinct vectors plus a shared matrix. However, the report
still fails to enforce per-seed accuracy macro and full-data retention-point
gates, Windows descriptor reads do not use the opened parent handle chain, and
event value types/identity are not exact. These gaps still permit unsupported
success claims or unsafe evidence reads.
