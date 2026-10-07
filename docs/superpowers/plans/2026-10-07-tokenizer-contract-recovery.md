# Tokenizer-contract recovery implementation plan

> For agentic workers: use superpowers:subagent-driven-development with GPT6 Luna implementers/reviewers. Root owns decisions, commits, push and production launch.

**Goal:** repair the actual tokenizer boundary, then run one approved linked calibration with all costs conserved.

**Architecture:** explicit token-list contract plus offline real-tokenizer/HuggingFaceWorker boundary tests; an exact schema4 authorization authenticates the reviewed schema3 stop without rewriting earlier records.

**Tech stack:** pinned Python3.11/torch2.13.0+cu130/Transformers5.16.1/PEFT0.21.2/Windows native runner; no dependencies added.

**Spec:** docs/superpowers/specs/2026-10-07-tokenizer-contract-recovery-design.md.

## Global Constraints

- Keep exp, unrelated task-14*.md, prepared source and all three stopped executions/slots unchanged. Root preserves scratch evidence; no PR/main merge.
- Workers use CPU/local-tokenizer only: no GPU, pretrained model generation, network, installs or production registrations. Root launches once after release gates.
- Frozen scientific prompt/models/data/scorer/precision/training unchanged; explicit return contract must preserve input IDs exactly.
- Caps172800active/86400CUDAseconds/21474836480download+artifactbytes/capacity-minus2GiBVRAM. Initial91active/1250CUDA/8066717862downloads; artifact2153595 plus newslot bytes. No double charging the prior31active interval.
- TDD RED/GREEN, actual dependency boundary evidence, independent reviews and root coherent verification required before registration.

## Review Focus

1. Real BatchEncoding/default and list return, malformed IDs/batches: explicit list request and fail-closed validation, not coercion.
2. Real generation-result/tensor/decode boundaries and effective two EOS IDs: no mock-only API contract.
3. New slot coexists with old recovery/repair namespace guards and preserves schema3 validation.
4. Actual schema4 producer initial reservation/pre-model/normal/exhausted ledger/status costs equal inherited baseline plus only new work.
5. Rehashed nested contexts/floats, inventory/slot tampering, concurrent registration and stale/live owners fail before writes.

### Task 1: Actual tokenizer contract and consolidated CPU inference boundary

Files: modify src/ppmoe/distill/model.py and tests/test_distill_model.py; create tests/test_distill_tokenizer_integration.py. Root owns all public docs. No authorization/native/study/evaluator changes in this task.

Interfaces: retain `build_prompt_ids(tokenizer,prompt_config,question)->list[int]` and HuggingFaceWorker.generate(row,candidate) schema. Exact call adds only `return_dict=False` alongside tokenizeTrue/add_generation_promptTrue/enable_thinkingFalse. Before indexing, normalize .tolist if supported; reject non-Sequence/mapping/strings; unwrap exactly one nested sequence; reject empty or any non-integer/nonnegative ID (exclude bool, no float/string coercion), preserving actual integer-like numpy scalar IDs if needed without accepting lossy casts.

- [x] RED regression with dependency-contract double returning actual Transformers BatchEncoding by default and list only if return_dictFalse; existing helper must fail. Update FakeTokenizer accordingly rather than a double that always returns a list. Add literal malformed cases: empty list, dict, string, bool, negative, float, numeric string, two batches, nested-invalid; tensor/singleton valid cases. Assert caller-visible IDs and unchanged flags.
- [x] Minimal fix: explicitly set `return_dict=False`; validate before indexed/batched access, then return verified integer list. No prompt/template/model/generation/scorer change.
- [x] Real offline integration: locate exactly retained pinned student source above, authenticate tokenizer-relevant files (tokenizer.json/tokenizer_config.json/chat_template.jinja if present/special_tokens_map.json if present) against calibration/model-inventory/student.json in reviewed predecessor. Use AutoTokenizer.from_pretrained(path,local_files_only=True). Synthetic question `What is 1 plus 1?`; compare helper to independent explicit-list tokenizer output and defaultEncoding.input_ids, literal token count39; forbid CUDA/network/model loads. Test missing/tampered auth before tokenizer use. Portable missing cache/dependencies may skip; record root launch gate must not skip.
- [x] CPU boundary test with real torch tensors and real cached tokenizer, controlled model.generate returning Transformers GenerateDecoderOnlyOutput. Force cuda.is_availableFalse only for this CPU test; ensure inputtensorCPU, promptshape/dtype long, exactgreedy/cache/return_dict/eos flags [151643,151645], slice prompt beforedecode, literal expected completion `Final answer: 2`, EOS and cap plus overflow skips model.generate. No pretrained model inference. If code-inspection reveals an additional load-bearing API mismatch, report it to root before changing behavior.
- [x] Consolidated boundary report: inspect installed tokenizer/apply_chat_template, model.generate's Transformers return shape, tokenizer.decode tensor acceptance, PEFT prepare casts/freeze and EOS metadata calls used by HFW; identify concrete remaining risks with filelines and tests, not claim live teacher/generation passes from CPU.
- [x] GREEN `.venv/Scripts/python.exe -m pytest -q tests/test_distill_model.py tests/test_distill_tokenizer_integration.py tests/test_distill_evaluate.py`; pinned offlineRuff0.15/compile/diff. Root owns fullCPU after Task2 integration; report all focused failures/fixes, exact RED/GREEN and real integration skip/pass in task-1-report.md; handoff without commits/subagents/GPU/network. Root verifies/freezes/reviews before Task2.

### Task 2: Exact tokenization recovery authorization and shared consumers

Files: create src/ppmoe/distill/tokenizer_recovery.py and tests/test_distill_tokenizer_recovery.py; modify execution.py/background.py/study.py signature registry/scripts/run_distillation_background.py/tests/distill_background_cpu_fixture.py and focused corresponding tests. Do not change recovery.py/repair_recovery.py protocols/native containment or scientific producer behavior. Root owns docs.

Interfaces: `register_tokenizer_recovery(previous_execution_root,executions_root,*,repo_root,config_path)->Path`; `validate_tokenizer_recovery_manifest(manifest,*,repo_root,config_path)->None`. Public CLI `register-tokenizer-recovery` takes only previous-execution-root/executions-root/config/repo-root. Schema4 context `tokenizer_recovery`, protocol `distillation-tokenizer-contract-recovery-v1`, revision1, prefix `tokenizer-authorization-<predecessor-id>.json`. Retain original `recovery`/`repair` ancestry verbatim. Original manifest fields retained except ID/schema/created/inheritedclock+cost/artifact additions/newcontext.

Exact predecessor3116768bd90444328e1ec0e4f8b518a18c238ed74e643d24ae0a69affd0683eb, attempt41c657fa3727456e812988a0cf54d4c9, produceroriginda8a15819f92f6b8c28673dc89473019f0f7fe31. Exact immutable pins:

```text
calibration/model-inventory/student.json b771e48e0eca0812c4a2271fe565cdecd2970bc5b2f0875bbfc27df7bd015566 2790
calibration/model-inventory/teacher.json 5bde59ee552551a8c81266a79019e62081accb99c614254ea508b0bd17dae74a 3072
calibration/raw/256-768-student.jsonl.attempts.jsonl 2f9aed0ab7e108b10d2faae2391b7db7ea926b04bfd562597eb934126bb8c6fe 4562
events.jsonl c1f3c1b92573c16179de21ac9f460c0d07f463bc02f5bd99c5244d4223d3c64e 3616
execution.json 475d289ad8e4d8bc6618e46d6fe55df138f4ec394f643de7d45291cf310eae06 22960
launches/41c657fa3727456e812988a0cf54d4c9.json d0e2bb2023058441104de4d3834d785004ca928d522afcf293bb524bcc6a4ebe 238
logs/41c657fa3727456e812988a0cf54d4c9.stderr.log 7816a030fdc56592b773294910ab32b262a5e7590a6b579049004de5b7268b5f 3024
logs/41c657fa3727456e812988a0cf54d4c9.stdout.log 6e475d0be935d9336a375e4abe49d133f7eba0d2086d85e38e59b9efc7ef46d3 5054
logs/41c657fa3727456e812988a0cf54d4c9.supervisor.log e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 0
stages/calibrate-attempts.jsonl ccfddf0501d90524b0bef7bb0135cff337a20739ae4d2d4e59353a2ce0a30209 136
stages/calibrate-reservations.jsonl eb30c4b0012170df1b0968d1d834c6bdb6fdba4ecffd31ccb50fe7ba273ac510 3473
stages/calibrate.json 2d5bcfdd5a2b3d2ec2dc15ec5aedf6a7ba438ed0f236c824941b765d0a00d6e4 4138
terminal.json 33cd54d85fb2d115257db4b54dba9826829e857a2c5ba34555dae6bfb06c9c2a 611
```

- [x] RED disposable copied predecessor registration/context tests. Authenticate schema3 ancestry before exact inventory, fixed status files size2776+1960=4736; immutable53674; old authenticated baseline2095185 =>2153595 before newslot. Validate exact terminal stopped_environment/retry-exhaustion/attempt/91active/172709remaining; single settled job prior60 cumulative91, no locks/live owner identities; no links/reparse/extrafiles/answers/review/pilot/training. Exactly3 failed tokenizer events after3worker_started and placementstdout; no download reservation/completion in predecessor. Stage signature canonicalhash, originalorigin/predecessors/policy/executionID preserved; authentic stage ledger1250CUDA/8066717862downloads and initialreservation1228/8066717862, no discrepancy/refund.
- [x] Implement narrow `_verify_predecessor` using validate_execution/read_events/validate_job_intervals/_inventory/_checked_path/sharedcost authentication where possible. Exclusive canonical O_EXCL/fsync newslot before childfolder. Reject schema1/2/4/unrelatedsource/differentpred, source/config/lock/runtime drift, changed/extra/missingpins and scientific/unknown/unsettled stop. Failed/partialslot stays consumed; sequential/concurrentduplicates cannotcreatechild. Test namespace coexistence: old schema3 remains valid after newslot, original authorization files byteidentical.
- [x] Build entire schema4manifest once then hashID; copy oldfields/artifactbaseline exactly, append `tokenizer_predecessor_immutable_files/bytes`, `tokenizer_predecessor_status_reserve_bytes`, `tokenizer_authorization_reservation`. Context pins/root/originalsignature/inventory/initial91/cost1250+8066717862/rationale/newslotpathhashbytes bind all; strict exactints in top/context costs+clocks (no float/bool/extra keys), rehash tampering mustreject. `_load_authorization` boundedcanonicalnewline and fixedpath/prefixprotocol/date bind; exactoneslot glob. No generic extensible schema framework.
- [x] Shared execution helpers explicitly schema4 initial91 andcost1250/8066717862, appendedartifactfieldssummedonce; other schemas unchanged. Add source/test/signature registry entries. Real validated schema4 run_calibration_stage pre-model, normalawaiting_review, exhausted-worker tests assert initialreservation and producedledger/status complete1250baseline plus newcost exactlyonce, including replay/resume. Do not mock manifestvalidation/manualledgers to pass. Reuse realfixture structure existing tests/test_distill_repair_recovery.py but preserve ownership/path originals read-only.
- [x] Private CPU fixture `prepare-tokenizer-recovery` copies latestpredecessor into disposableparent, keeps originalancestorrefs read-only, registers onlydisposable newslot. Detached91 baseline test uses shared_persist_calibration_terminal, confirms launcherexit/liveidentity/naturalawaiting_review/ownerexit with existing boundedwait; no hand-shaped terminal substitute. Root cross-terminal proof after freeze. PublicCLI no fake/command/budget flags.
- [x] GREEN focused tokenizer_recovery/repair_recovery/recovery/execution/background/study/model/evaluate/windows_job tests, offlineRuff0.15/compile/diff. Worker onefullCPU `.venv/Scripts/python.exe -m pytest -q -m 'not cuda'` afterfinalintegrationselfreview; allfailures retained, no exclusion-as-release. Root fresh verification and independenttask/finalreviews. Write task-2-report.md exactRED/GREEN/interface/nativeevidence; handoffwithoutcommit/subagents/productionwork.

### Task 3: Root release and one registered scientific continuation

- [x] Root focused real-tokenizer gate (no skip), native/finalCPU and independent finalreview clear; source/config/lock/runtime/GPU/allancestorpins recheck; push exp without PR/main merge.
- [x] Register once with public register-tokenizer-recovery CLI and exact predecessor/root/config; assert printedroot's initial91/cost1250/8066717862/artifact2153595+newslotbytes before modelwork.
- [x] Launch existing calibrate once, separate-terminal verify actualownedprocess creation identities/heartbeat/logs, initialreservation and placementcounts/costs. Sourcecodefrozen; noextra pretrained-modelsmoke.
- [x] At awaiting_review continue original blind Luna extractionaudit and registered teacher/corpus/training gates from docs/superpowers/plans/2026-09-27-verified-distillation-qlora.md. At terminalstop preserve precisecause/costs/absenceoutputs; nofurtherautomaticregistration. Update executionstate/todo/log/ledger andpush.

## Self-review

Task1 changes only token-list boundary/tests; Task2 signatures consume its frozenhash and preserve ancestral identities; Task3 reads authenticated91/1250/8066717862 baseline from Task2. Reviewfocus1–2 assigned Task1,3–5 Task2. No prompt/output/scientificretuning, modelsmoke or authorization reuse; source and native ownership retained. Routine review/approval checkpoints waived by userautonomy, not verification gates.
