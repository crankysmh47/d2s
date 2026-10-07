# Production-path repair implementation plan

> For agentic workers: use superpowers:subagent-driven-development, GPT6 Luna implementers/reviewers. Root owns decisions, commits, push and production launch.

**Goal:** fix actual placement and cost propagation, then run the single newly approved linked calibration.

**Architecture:** strict tensor placement validation and manifest-seeded real producer accounting; a separate schema3 repair module authenticates the exact schema2 stop without changing old authorizations.

**Tech stack:** existing pinned Python3.11/Transformers5.16.1/torch2.13/Windows native runner; no dependencies added.

**Spec:** docs/superpowers/specs/2026-10-07-production-path-repair-design.md.

## Global constraints

- Work on exp; preserve unrelated task-14*.md, prepared source and both stopped executions.
- No GPU/model/network/dependency changes by workers; CPU fake dependencies only. Root runs one approved calibration after reviews.
- Caps172800 active/86400 CUDA seconds/21474836480 download and artifact bytes/capacity-minus2GiB VRAM.
- New baseline60 active/1228 conservative CUDA seconds/8066717862 downloads, artifact2094882 before new authorization bytes. No override/reset or repeated registration.
- Only exact stopped predecessor b0c082cba66a37f6f49bacc4bee182102b926ac5ac3af4c0a657823213e1802f; retain old schema2 and schema1 validation, no arbitrary chain.
- Real producer serialization/projection tests required. Actual placement summary contains counts/devices only, no tensor contents or secret dumps.
- TDD RED/GREEN, independent task reviews plus root verification and final release review before registration.

## Review focus

1. Missing/stale device map with actual CPU/meta/buffer/quant-state tensors must reject, not falsely accept missing metadata.
2. Real pre-model and exhausted-worker producer ledgers must use manifest carry; manually shaped CPU fixture is insufficient.
3. Replay existing failures/download reservations once, without refund or double charge, including interruptions.
4. New authorization must not invalidate old schema2 validation via its existing namespace-wide glob guard.
5. New initial clock/status/job and artifact accounting must agree with validated60 baseline and both ancestral evidence costs.

### Task 1: Strict placement and manifest-seeded actual producer

Files: modify src/ppmoe/distill/model.py, study.py; tests/test_distill_model.py, test_distill_study.py, test_distill_evaluate.py only as needed. Do not modify execution/recovery/native modules. Root owns docs.

Interfaces: retain validate_cuda_placement(model); it may return a compact summary usable by HuggingFaceWorker. Use named_parameters()/named_buffers() and bitsandbytes quant_state fields absmax/code/offset/state2 recursively, with visited-object guards. Tensor device must be cuda:0 (unindexed cuda means configureddevice0); malformed/missing device rejects. Present map must be nonempty Mapping with only device0 CUDA entries (integer0, cuda/cuda:0, torch.device); reject bools/unknown strings/CPU/disk/meta/cuda:1 even if tensors appear CUDA. Do not trust map instead of actual tensors. Nonempty parameter count mandatory. Persist counts/device-set summary in worker lifecycle metadata without changing response text/digests or scientific prompt. Existing generation metadata/journal should carry summary where available, bounded/no values. If lifecycle has no safe slot, propose a small load-success journal entry, charge its bytes normally, and report exact interface to root.

- [ ] RED: CPU-only model doubles with CUDA device descriptors (or torch FakeTensorMode) but no hf_device_map must pass if parameters/buffers/quant state allcuda0; current guard rejects. Cases CPU/meta/malformed buffer/quant-state nested tensor, empty parameters, misleading good map, mapcuda1/unknown/bool reject. Assert returned summary counts and actual downstream metadata, not source strings. Real CPU tensors reject, never initialize CUDA in tests.
- [ ] Implement minimal strict check; no package/load/precision changes. Preserve validate_cuda_memory and verify_student_is_frozen calls.
- [ ] RED: extend actual run_calibration_stage producer tests with bound authenticated execution context and inherited_costs1199/8066717862, then1228/8066717862. Persist normal awaiting_review and exhausted-load terminal; supervisor must expose complete costs >= inherited, and initial reservation values equal inherited. Never hand-insert stage/ledger/signature to make a fixture succeed.
- [ ] Also test pre-model durable stop before any worker, existing failed-row replay, cached teacher metadata/no download, invalid types including bool/negative costs, legacy1168/8066699435 behavior. Hand-check one old5second failure +new7second load =>baseline+12, replay again unchanged. Keep inherited active context separate from CUDA costs.
- [ ] Implementation blueprint: `_initial_resource_ledger(*, inherited_costs=None)` uses ResourceLedger.initial() legacy; otherwise validate strict two nonnegative ints and dataclasses.replace(cuda_seconds=..., download_bytes=...). `_replay_calibration_resources(..., inherited_costs=None)` seeds via that helper, retains existing permissive replay caps then restores fixedcaps. In run_calibration_stage obtain `inherited_costs=execution_manifest['inherited_costs']` only from validated manifest and pass to BOTH pre-model and recovered-ledger calls, plus EVERY error/pre-model replay branch. Reservation uses the same inherited baseline, not constants or already-charged result. Do not change ResourceLedger class legacy defaults globally.
- [ ] GREEN `.venv/Scripts/python.exe -m pytest -q tests/test_distill_model.py tests/test_distill_study.py tests/test_distill_evaluate.py tests/test_distill_background.py`; pinned offline Ruff0.15/compile/diff. Write task-1-report.md exact RED/GREEN, changed interfaces/metadata and concerns. Root commits/reviews; do not run fullCPU until Task2 integration freeze.

### Task 2: Exact second authorization and schema3 cumulative accounting

Files: create src/ppmoe/distill/repair_recovery.py, tests/test_distill_repair_recovery.py; modify execution.py, background.py, study.py signature registry, scripts/run_distillation_background.py, private CPU fixture/native tests as needed. No old recovery.py protocol changes; no native containment changes. Task1 study.py edit ownership is sequential.

Interfaces: `register_repair(previous_execution_root, executions_root, *, repo_root, config_path)->Path`, `validate_repair_manifest(manifest, *, repo_root, config_path)->None`. Add CLI `register-repair` with previous-execution-root/executions-root/config/repo-root only. Schema3 uses a `repair` context alongside saved original `recovery` ancestry; validates costs1228/8066717862 and inherited_active_seconds60. validation of schema3 calls repair module, schema2 still old recovery module. `validate_job_intervals`, status, supervision use schema3 initial60; artifact_baseline_bytes includes inherited schema2 fields PLUS repair predecessor immutable/status/newauthorization fields once.

Exact predecessor ID b0c082cba66a37f6f49bacc4bee182102b926ac5ac3af4c0a657823213e1802f. Attempt216e170f91e1458aa74d679e457a159e. Immutable pins (all13 required before writes):

```text
execution.json 4f8f0071edea9ec9e60c7ee2b1f92265a7485d2c9973179f67964abe65468715 13411
events.jsonl 31da70086f7b84deebc73d81264d8fd774cc7a4d4f01eccce35d260d78a16868 3619
terminal.json 6ae485ccb972bcd7648a6ab67b2d0da9b43901ff6fc4727678f377f6c086fd3d 612
stages/calibrate.json 4c3fb2996c9e91b26b1187efa38ad9ca39846ed8cc8f8965fa30509760d993bd 4136
stages/calibrate-reservations.jsonl 7c4b29a8329b4e695df00f0fa560be3b88efea41c023c8af32bfc6393e51089f 3474
stages/calibrate-attempts.jsonl 4adda447dee77b56c35757e50a2a215bbf677c5d9b642eb419cf963359bd8b0d 136
calibration/raw/256-768-student.jsonl.attempts.jsonl e6f7b50678c363ef6da873c6827ab7d7eed24a0535adae3da30c808138b87378 2980
calibration/model-inventory/student.json b771e48e0eca0812c4a2271fe565cdecd2970bc5b2f0875bbfc27df7bd015566 2790
calibration/model-inventory/teacher.json 5bde59ee552551a8c81266a79019e62081accb99c614254ea508b0bd17dae74a 3072
launches/216e170f91e1458aa74d679e457a159e.json bfde45b207f6835d9df0e618f56b3bcc408e2de27c3397c0b817c09870500dcd 239
logs/216e170f91e1458aa74d679e457a159e.stderr.log e9c55510876a5d8de68e610d43696cb6bc7e3383f3806dcb8338d8370a148a91 3160
logs/216e170f91e1458aa74d679e457a159e.stdout.log 4adcde87f24effd6ee8798c324a4ef9cee2aac5138fc02398a57c2bd618d0c56 4641
logs/216e170f91e1458aa74d679e457a159e.supervisor.log e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 0
```

Mutable convenience status reserve4744bytes, immutable42270, predecessor authenticatedbaseline2047868 =>2094882 before newslot. Newslot filename `repair-authorization-<predecessor-id>.json`, protocol `distillation-production-path-repair-v1`, revision1, no oldnamespaceglob changes.

- [ ] RED disposable copies/monkeypatch fixed pins only in tests: baseline60/1228/8066717862, artifact2094882+newslotbytes; no oldfilebytes change. All13pins/exactset, ancestor validation, stage authenticated canonical signature using stored origin, exactly3 stated placement failures, noanswers/no downloads, settledprior31/new29/cumulative60, no recorded live owner or locks. Preserve observed bad producerledger1189/8066699435 as evidence, explicitly reconcile without accepting/refunding it.
- [ ] Reject scientific/unknown stop, live/unsettled jobs, links/reparse, changed/extra/missingfiles, altered source/config/policy, wrongnamespace and generic chained/schema3 predecessor. Registration writes only after readonlyvalidation and exclusivefsyncedrepairslot; concurrent/sequential/partialfailure slot staysconsumed. Original recovery slot remains untouched and schema2validation still passes after repairslot exists. Tests cover this namespace interaction directly.
- [ ] Register schema3 from fully built manifest then hashID once, never mutate signedmanifest. Store full predecessor pin/inventory +frozenstatusreserve +fixedcarriedclock/cost/rationale +newslotpath/hash/size. Original artifactbaseline fields copiedexactly/authenticated againstpinnedpredecessor; newfields `repair_predecessor_immutable_files/bytes`, `repair_predecessor_status_reserve_bytes`, `repair_authorization_reservation` append once.
- [ ] Runtime helpers support schema3 explicitly, never silently take unvalidated cost/refund. Add repair module/test to execution signature registry. Preserve schema1/2 behavior and existing exclusivitytests.
- [ ] Private CPU `prepare-repair` copies schema2 predecessor into disposable namespace, validates authentic absolute ancestor/oldreservation refs read-only, creates only disposable repairslot. Fake stage MUSTuse shared_persist_calibration_terminal. Add detached60clock/status/artifactproof and actual producer regression with schema3context/manifestseed fromTask1. CLI has no fake/command/cap flags.
- [ ] GREEN focused test_distill_repair_recovery/recovery/execution/background/study/model/evaluate/windows_job; offlineRuff/compile/diff. FullCPU oncebeforefreeze `.venv/Scripts/python.exe -m pytest -q -m 'not cuda'`. Report exactcommands/nativeproof; root verifies/commits/taskreview, then rootfullCPU and final integrationLuna review. No production registration byworker.
- [ ] Observed Windows lifecycle races: detached tests must confirm recorded owners exit with a bounded condition wait after terminal persistence, not assume the still-finalizing supervisor disappears instantaneously. Launch acknowledgement reads may retry only temporary permission/not-found errors within the existing10-second deadline; preserve strict ack identity, malformed/refused failures and exactly one supervisor spawn. Add focused RED/GREEN boundary tests and keep failed run evidence; no native containment or budget change.
- [ ] Immediate pre-model stop tests must allow equal start/end UTC timestamps at clock resolution. Cover that case with a fixed test clock and terminal/journal timestamp agreement, without fabricating elapsed time or changing monotonic job/resource accounting.

### Task 3: Root launch and registered scientific continuation

- [ ] Final source/config/lock/runtime/GPU/predecessor/ancestor recheck; root focused/native/fullCPU +final review clear; push named commits exp. Do not add uncharged model smoke or alter precision/prompt.
- [ ] Register once `scripts/run_distillation_background.py register-repair --config configs/distillation-v1.toml --previous-execution-root artifacts/distillation-v1-executions/b0c082cba66a37f6f49bacc4bee182102b926ac5ac3af4c0a657823213e1802f --executions-root artifacts/distillation-v1-executions`; save returnedroot, assert initial60/cost1228/8066717862/artifact2094882+actualslotbytes.
- [ ] Launch calibrate existingCLI once with returnedroot. Verify separate-terminal ownedPID+creation/liveheartbeat/logs. Keep codefrozen. Inspect placementsummary/realreservation/budgetpropagation before treating output asvalid. Monitor without changingrunningidentity; reviewdowntime notcharged.
- [ ] At awaiting_review continue original blind Luna calibration extraction audits and scientificplan gates (docs/superpowers/plans/2026-09-27-verified-distillation-qlora.md). At terminalstop preserve exactcause/costs, no furtherautomatic registration. No scientificclaims fromtests/no prematureteacherortraining. Update state/todo/log/durableledger andpush.

## Self-review

Task1 repairs actualproducer and strictplacement; Task2 adds exactnewauthorization without mutatingoldnamespace/contracts; Task3 verifiesrealruntime andfollows existing scientificbranches. Overlapstudy.py sequential, baselinefields jointlyspecified, allfive reviewfocus casesassignedtests. No scientificretuning or arbitraryrepeatmechanism.
