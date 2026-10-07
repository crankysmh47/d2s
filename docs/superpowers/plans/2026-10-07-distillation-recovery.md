# Distillation recovery implementation plan

> For agentic workers: use superpowers:subagent-driven-development, Luna implementers and independent reviewers. Root owns decisions and production launch.

**Goal:** repair the real calibration contract and authorize one evidence-linked recovery without budget refunds.

**Architecture:** keep the supervisor's fail-closed consumer; correct the producer and add bounded failure messages. A narrow recovery module authenticates the exact known stopped predecessor, creates a schema2 manifest and carries its costs/clock into existing shared accounting.

**Tech stack:** existing pinned Python3.11/pytest/Windows Job Object stack, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-07-distillation-recovery-design.md`.

## Global constraints

- Work on `exp`; preserve unrelated `task-14*.md`, stopped executions and read-only prepared source.
- No dependency/model/prompt/scorer/data/training changes, no network/GPU from implementers or reviewers.
- Overall caps172800 active seconds/86400 CUDA seconds/21474836480 bytes downloads and artifacts; capacity-minus2GiB VRAM reserve.
- Recovery inherits31 active seconds,1199 cumulative conservative CUDA seconds and8066717862 download bytes. No cost or cap override CLI.
- One recovery from exact predecessor `fc901bb0766a9c79fb205302d220fe727a99ebbd0b315e6ee4ad4e66e6733634`; no chained recovery or terminal reopening.
- New artifacts authenticate new originating code; historical records are never rewritten/re-signed.
- Root verifies each task; independent task reviews plus final integration review before registration/launch. Record RED/GREEN evidence, freeze code before review.

## Review focus

1. Complete production calibration payload vs supervisor schema, not just a manually fabricated signature.
2. Failure messages that contain secrets/newlines/huge content, without losing ordinary actionable cause text.
3. First recovery job/status/ledger agree on inherited31 active seconds; no refund or double charge.
4. Mutated predecessor file, uncertain interval/live owner, scientific stop or alternate predecessor must fail before writes/spawn.
5. Failed and reused-cache costs/artifacts carry once; schema1 historical status stays valid after recovery code changes.

### Task 1: Repair actual producer contract and failure diagnostics

Files: modify `src/ppmoe/distill/study.py`, `src/ppmoe/distill/evaluate.py`; tests `tests/test_distill_study.py`, `tests/test_distill_evaluate.py`, `tests/test_distill_background.py`; modify `model.py`/its test only if message preservation demonstrably requires it. No execution/recovery/native code changes in this task.

Interfaces: actual `run_calibration` and `_persist_calibration_terminal` write the terminal producer dictionary consumed by `background._authenticated_stage_artifact`. `calibrate_candidates` records exceptions through `RawRowJournal.record_failure`. Preserve existing call signatures/row digests and charge logic.

- [ ] Step1 RED: extend an existing fake-worker `run_calibration` test to produce both awaiting_review and exhausted-worker failure through the real terminal serialization. Feed actual saved bytes into the supervisor authentication with a bound fake execution context; assert acceptance and `saved['stage']=='calibrate'`. Do not insert stage/signature into a fabricated result within the test. Test mismatched stage still rejects.
- [ ] Step2 RED: fail worker construction with `RuntimeError('isolated model worker failed: ValueError: actionable load cause')`, read the persisted journal and assert that actionable text survives, type remainsRuntimeError and digests validate. Test generation/close failures where recorded, a10000-character message, `Bearer abcdef`, `hf_`+30letters, and `token=secretvalue`/`password: secretvalue` redaction. Existing costs/retries/teardown assertions unchanged.
- [ ] Step3 minimal implementation: set stage at the shared real terminal persistence boundary, checking that an already supplied conflicting stage cannot be normalized into success. Add one bounded sanitizer reused by the existing failure dictionary builders, e.g. `failed['error_message']=sanitized_error_message(exc)`, retaining type and existing costs. No traceback/environment dumps. Keep terminal serialization immutable.
- [ ] Step4 GREEN: `.venv/Scripts/python.exe -m pytest -q tests/test_distill_evaluate.py tests/test_distill_study.py tests/test_distill_background.py tests/test_distill_model.py`; pinned offline Ruff0.15/compile/diff checks. Use injected CPU workers only.
- [ ] Step5 self-review and write task report with exact RED/GREEN commands/output, changed files and concrete concerns. Root commits named files, packages diff, independent task review; resolve important findings before Task2.

### Task 2: Evidence-linked registration and carried accounting

Files: create `src/ppmoe/distill/recovery.py`, `tests/test_distill_recovery.py`; modify `execution.py`, `background.py`, `study.py`, `scripts/run_distillation_background.py`, their covering tests and private CPU fixture as required. Root owns public docs. Do not change windows_job.py/native containment or scientific config.

Interfaces: `register_recovery(previous_execution_root, executions_root, *, repo_root, config_path)->Path` in recovery.py; `validate_recovery_manifest(manifest, *, repo_root, config_path)->None` authenticates the supported schema2 immutable bindings. Use local imports in execution.py to avoid cycles. Existing schema1 `register_execution`/`validate_execution` remain supported and unchanged semantically. Recovery registration may extend register_execution with a private verified context, but must not create then mutate an already-ID-bound manifest.

- [ ] Step1 RED: CPU fixture builds exact-shaped stopped predecessor with canonical files and ordered lifecycle; monkeypatch the recovery module's reviewed-pin table for disposable test bytes only. Assert recovery registration does not touch predecessor bytes, schema2 inherited_active_seconds31, costs1199/8066717862, source/limits unchanged, initial statusused31/remaining172769; first job start prior31 and settled prior31+duration. Schema1 still starts0; rejectbool/negative/mutated baselines, hash mismatches, open intervals, live owner, scientific stop, wrong source/path/reparse, second recovery and any completed/raw-answer rows.
- [ ] Step2 reconcile pinned production evidence read-only. Pin predecessor ID and SHA256: execution.json `400eb64e940c8ba9e9ed4de7e3135573c1614fbff87e5e349b02e3641b432931`; events.jsonl `14c440a1dcd2eb31e4f6d0f6d99533c4eb5422e61faae9e4f8a80cc89445d5be`; terminal.json `e81f942480cd605e3e5bcc2612404456c587d4840a0dad6321240865828269aa`; stages/calibrate.json `e8ee6708573b9ba10df1677befa5044c5b8d84e7c9f42d86db446a93f11411f1`; worker attempts `e7ecddc7dc198250a81d837be5efc78a51f6bfc537054521ced2344f14b341bc`; reservations `31a4342e00a7ea8b011d2d7b0c51756e15c25f10c4af1c4f80e6ad6af6831079`; teacher inventory `5bde59ee552551a8c81266a79019e62081accb99c614254ea508b0bd17dae74a`. Include other immutable retained files in the new inventory; any change must reject. Pins authenticate the exceptional missing-stage historical producer as observed bytes, not as a successful result under the repaired consumer. Read old producer signature using its stored digest and old origin, never recompute current code into it.
- [ ] Step3 implement recovery.py narrowly: verify exact predecessor using schema1 validation, event-chain/lifecycle settlement31, terminal environment/attempt matches, old sealed producer failure status/no answers, exactly3 failed student loads,18427 teacher metadata reservation+completion and inventory reconciliation. Confirm no live lease/process owner or current-job; require disjoint new output/source/predecessor roots. Create manifest schema2 with `recovery` containing predecessor absolute root/ID, pinned and inventoried evidence, carried cost/clock rationale, fixed one-hop authorization revision. Register exclusively only after all validation; append registered event and save status as existing CLI does.

```python
# Shared cumulative accounting starts at the authenticated manifest baseline.
initial = manifest.get('inherited_active_seconds', 0)
# validate_job_intervals seeds latest_settled_seconds=initial.
# _base_status and supervise_stage use default=initial for no settled jobs.
# study's existing active-job ledger consumes that job prior without recharging.
```

Keep caps global, not reduced/caller-set. Existing active_elapsed_seconds and settlement math are unchanged. Read schema2 inherited costs from the validated manifest in study as existing code already does. Artifact projection/replay includes predecessor inventory bytes plus original baseline exactly once; immutable metadata validates that baseline and retained provenance, mutable predecessor-status byte reserve is fixed at registration. Avoid per-poll heavy duplicate scans beyond existing immutable validation. Include recovery.py/test in execution stage code registry.

- [ ] Step4 CLI adds register-recovery with previous-execution-root/executions-root/config/repo-root only, no arbitrary command/fake-stage/budget knobs. It prints root/status path. Refuse recovery registration from a recovery manifest. Test CLI parse/dispatch and unsupported override flags; normal register still emits schema1.

One-registration requirement: executions_root must equal previous_execution_root.parent; after all read-only checks acquire `recovery-authorization-<predecessor-id>.json` there via O_CREAT|O_EXCL, fsync a bounded canonical reservation binding predecessor identity/approved protocol before creating any new child. It is separate from the immutable predecessor. Existing reservation, even partial, refuses another registration; retain failed evidence. Tests cover concurrent/two sequential registration attempts yielding one reservation/new manifest, plus registration failure after reservation never silently reopening. Pin the entire immutable predecessor inventory; exclude only its two convenience status files and reserve their observed bytes once. Reject extra immutable files when revalidating, not just mutations of known files.

The complete pinned inventory also includes: student inventory `b771e48e0eca0812c4a2271fe565cdecd2970bc5b2f0875bbfc27df7bd015566`; launch record `4d84ad1ef2a115074389dbbd1f2991ad2db8cf6ef5a553f87826f77a8e5a96fe`; stderr `7a94e8679ed0fbabfc9d2df058470b2c83879875bcfd30ebfa7209e8d2ea2912`; stdout `4272c8c0f2386c86914ab884a93c997687239dc7ef36b6ed10649187d3ff613e`; empty supervisor log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; stage-attempt journal `49ee83ee44d35f3b680faaf08a0edb3748332c148e0312f8e118edbe90713b8e`. Attempt-specific launch/log filenames use `dd45bbdd2ecf45cb8d8fcc653132ee0a`. All13 immutable files total35747 bytes; old status pair observed4482bytes; originalbaseline2007336 =>2047565 inherited artifact bytes before adding immutable authorization-reservation bytes. Authenticate all13 before registration, not only the primaryseven; tests can monkeypatch the pin table for disposable copies.
- [ ] Step5 GREEN: focused suite `tests/test_distill_recovery.py tests/test_distill_execution.py tests/test_distill_background.py tests/test_distill_study.py tests/test_distill_evaluate.py tests/test_distill_windows_job.py`; pinned offline Ruff/compile/diff. Add native CPU-only inherited31 job proof (start afterlauncherexit, settled cumulative, all descendantsdead). Existing fixture uses actual producer serialization from Task1 or exact shared terminal persistence, not a manually repaired payload.
- [ ] Step6 report/freeze/rootcommit/taskreview. Root runs coherent `.venv/Scripts/python.exe -m pytest -q -m 'not cuda'` and independent integration review of Task1+2. No production registration before gates clear.

### Task 3: Root release and one production recovery

- [ ] Recheck config/lock/runtime/GPU identity, prepared source/oldstop and all predecessor pins; verify no tracked unintended edits. Push approved named commits toorigin/exp; preserve unrelated files.
- [ ] Register once using `.venv/Scripts/python.exe scripts/run_distillation_background.py register-recovery --config configs/distillation-v1.toml --previous-execution-root artifacts/distillation-v1-executions/fc901bb0766a9c79fb205302d220fe727a99ebbd0b315e6ee4ad4e66e6733634 --executions-root artifacts/distillation-v1-executions`. Record exact printed new root; assert initialused31/cost1199/8066717862/inventoried artifacts. No placeholder root guessing.
- [ ] Invoke existing `launch calibrate` with printedroot once. Save separate-tool live PID+creation/log/heartbeat proof. Inspect model failure messages if any; do not bypass load/cap/scientific/retry gates or launch a third execution.
- [ ] At awaiting_review dispatch original scientific Task2 blind Luna answer extraction audits; if environment/scientific stop persist exact reason/cost/absence of downstream work. Update public state/todo/PROJECT_LOG and pushdocs. No quality/training claim without actual evidence.

## Self-review

Spec coverage: Task1 owns true producer/messages; Task2 owns exact predecessor recovery/cumulative budgets/provenance; Task3 owns authorized launch/review boundaries. Interface overlaps are explicit; Task2 follows Task1 serially. This is one new engineering revision, not a second scientific candidate or permission for repeated recovery registrations.
