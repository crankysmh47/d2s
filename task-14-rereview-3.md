# Task 14 re-review (round 3) — FAIL

Review date: 2026-09-18

Scope: independent re-review of the round-two residuals and the requested
round-three contracts: actual evaluator retention integration, exact paired
bootstrap universes/hashes, independent CI/gate recomputation, registered
timing evidence, descriptor-bound reads, hidden-file inventory, strict event
schemas, and failure reporting. No implementation files were changed or
committed; only this review was added.

## Verification

- `uv run pytest -q tests/test_recurrence_report.py tests/test_recurrence_experiment.py tests/test_recurrence_evaluate.py tests/test_recurrence_gates.py`: **145 passed, 1 skipped**.
- `uv run pytest -q`: **612 passed, 2 skipped**.
- Focused replay/report/adversarial selection: **10 passed** (2 experiment, 8 report).
- `uv lock --check`: passed.
- `git diff --check`: passed (existing LF/CRLF warnings only).

## Round-three checks that pass

- The actual `evaluate_retention()` producer output, including `mean_nll`,
  `families`, `failures`, records path/hash, and ordered chunk hash, is now
  accepted by `build_report()` (`evaluate.py:423-465`, report test
  `test_actual_evaluator_retention_summary_is_reportable`).
- `_read_bootstrap_matrix()` now requires the declared SHA-256, exactly the
  expected family keys and ordered IDs, exactly 10,000 rows per family, and
  each row to have the complete in-range column universe
  (`report.py:681-708`). The malformed-column adversarial test rejects.
- Registered timing shape is enforced at 20 positive finite A2/A3 samples,
  `sample_count == 20`, seconds, synchronized metadata, both seeds, and a
  derived median A3/A2 ratio in `[0.90, 1.10]` (`report.py:710-748`). One
  sample and bare-boolean probes reject.
- Hidden files are now in `_read_inputs()` rather than silently skipped, and
  report replay rejects added/changed/removed non-output evidence through the
  exact source-manifest inventory comparison (`report.py:105-120`,
  `experiment.py:1222-1231`).
- Actual source-to-data signature progression, valid CUDA ledger/anchors,
  nested/root reparse rejection, negative NLL rejection, summary producer
  signature binding, first/second replay artifact verification, and Markdown
  reconciliation remain covered and passing.

## Residual findings

### [HIGH] Success gates still accept bootstrap vectors and decisive values that are unrelated to raw evidence

`_has_registered_bootstrap()` verifies that a vector and family-keyed matrix
are present, hashed, finite, and 10,000 elements/rows, then recomputes only
the vector percentiles. It does not derive the vector or each comparison's
point estimate from the authorized raw score/chunk records and paired matrix
(`report.py:611-660`). The comparison records may all reuse one vector path,
one matrix hash, and arbitrary finite `point` values; the checks only require
the persisted CI endpoints to equal the same vector's percentiles and the
threshold predicates to pass.

Independent adversarial probe: starting from the complete success fixture,
all exact raw arms remained identical, but the discovery 10,000-vector was
replaced by all `2.0`; its declared hash and comparison CIs were updated, and
the three comparison points were set to `999.0`. `build_report()` still
returned `decision.status == "passed"` (`UNBOUND_ACCURACY_GATE_ACCEPTED
passed`). This is a fabricated improvement unrelated to raw IDs, families,
or correctness differences. It also means the consumer cannot faithfully
consume the actual `paired_accuracy_bootstrap()` shape, which produces a
distinct bootstrap vector/point/CI for each comparison
(`gates.py:798-868`), while the report contract requires every comparison to
share one `vector_path` and hash (`report.py:640`).

The report must independently recompute each registered comparison from the
raw records/chunks, the exact family/ID universe, and the shared sampled-index
matrix, then reconcile its point, CI, seed pairing, and gate predicates.

### [HIGH] Decision producer authorization can be bypassed by payloads on stopped/failed stages

`_decision()` searches payload `decision`, `gate_decision`, and `outcome`
objects before checking whether the containing stage itself passed
(`report.py:515-547`). Direct probes returned a passed report decision for
both of these invalid records:

```text
_decision({"confirm": {"status": "stopped_scientific",
                       "payload": {"decision": {"status": "passed"}}}}, [])
=> {"status": "passed"}

_decision({"discover": {"status": "failed_environment",
                         "payload": {"gate_decision": {"status": "passed"}}}}, [])
=> {"status": "passed"}
```

The terminal decision must be produced only by the authorized, passed G4/G5
producer with a decision status consistent with the persisted stage/DAG and
its signatures. A child payload cannot override a stopped or failed stage.

### [HIGH] Path checks still do not anchor the descriptor open to the trusted-root identity

`_descriptor_read()` reads through a path after `_reject_reparse_path()` has
checked the path components (`report.py:196-267`). The opened descriptor is
fstat-checked before and after consumption, which detects replacement of the
opened file, but the root and parent directories are not opened and anchored;
an attacker can race a checked parent component into a junction/symlink before
`os.open()`/`CreateFileW()` resolves the path. `O_NOFOLLOW`/`OPEN_REPARSE_POINT`
only protects the final component and does not establish a trusted directory
handle chain. This remains a descriptor-identity/TOCTOU gap for every report,
source-manifest, JSONL, replay, and bootstrap read.

### [MEDIUM] Timing evidence is not restricted to the registered producer stages or full pilot metadata

`_has_timing_gate()` walks every stage payload and accepts an explicit seed
from any stage (`report.py:710-748`). A direct probe with valid 20-sample
payloads returned:

```text
TIMING_DISCOVER_CONFIRM True
TIMING_WRONG_STAGES True
```

The second result used the same valid seed-47/seed-48 payloads under `source`
and `data`, not `discover` and `confirm`. The check also only requires
`metadata.units == "seconds"` and `metadata.synchronized == true`; it does
not bind the samples to the registered five-warmup/20-sample pilot mechanism,
arm pairing, or an authorized stage. Timing evidence must be producer-bound
and carry the complete registered metadata.

### [MEDIUM] Event schema is still optional where the contract requires exact append-event fields

`_load_events()` requires index, a suffix-looking timestamp, and a registered
kind, but `stage`, `attempt`, `payload`, and `outcome` remain optional and the
timestamp is not parsed as an actual UTC datetime (`report.py:908-945`). The
following malformed event was accepted directly:

```json
{"index": 0, "timestamp": "garbage+00:00", "kind": "training_started"}
```

Probe output was `EVENT_OPTIONAL_FIELDS [{'index': 0, 'timestamp':
'garbage+00:00', 'kind': 'training_started'}]`. `append_event()` itself also
accepts arbitrary underspecified mappings and adds only index/timestamp
(`provenance.py:391-409`). This permits events with no stage/attempt/payload/
outcome and weakens complete failure accounting; every persisted event should
match the exact append-event schema, with all failure outcomes represented in
`failure_history`.

## Overall decision

**FAIL — Task 14 is not ready for acceptance.** Round three closes the real
retention producer/schema mismatch and structural paired-matrix checks, and
the focused/full suites pass. However, a hash-consistent fabricated bootstrap
vector can still authorize a passed scientific report, stopped/failed stages
can smuggle a passed decision, descriptor reads are not rooted by stable
directory handles, timing can come from unauthorized stages, and event fields
and timestamps are not exact. These defects leave the report vulnerable to
unsupported success claims and incomplete provenance/failure history.
