# Distillation background execution design

Status: user authorized a background worker and selected **48 hours of worker execution**, excluding review/agent downtime, on 2026-10-07. Implement on `exp` with Luna implementers and independent reviewers. The user's standing autonomous-execution instruction covers routine design/plan reviews; this document does not authorize changing scientific gates.

## Outcome and scope

An already implemented compute stage must continue after its launcher exits and after the chat reaches its usage limit. Save enough evidence to inspect results and resume a validated checkpoint without relying on conversation memory. The initial supported production stage is `calibrate`; the runner must reject registered-but-unimplemented stages. It is a reusable stage runner, not an unattended coding agent or an automatic review substitute.

The execution allowance is **up to** 172,800 seconds of cumulative active stage-job runtime. A completed phase stops early; it must not consume the remaining allowance by idling. Independent extraction/rationale/code reviews remain gates. At `awaiting_review`, settle the job's runtime, save the review-packet paths, and exit. No further compute until the required reviewed prerequisite exists.

## Chosen approach

Use a local, detached Python supervisor with a Windows Job Object containing its stage process and all model descendants. No cloud service, paid agent loop, task scheduler, new dependency, credential, or global power-setting change. A foreground invocation is useful for diagnostics but does not satisfy detached acceptance.

Rejected alternatives: keeping an assistant terminal session open leaves compute tied to session lifecycle; a scheduled task/service adds persistent machine configuration without helping the present stage/review boundary. The chosen approach needs native process-ownership testing but keeps its effects within this project.

## Preserve history; separate source and execution

The prepared source remains read-only at:

`artifacts/distillation-v1/acd8dfc400c8f079d0403da5d4b981b8aac8e94226db8760f632c8307ff785c2`

Its manifest SHA-256 is `bf0233bc65cd7f6554c7ab5269f17e051c31ff3903751ef7a24297ee203370ed`; prepare-stage SHA-256 is `970f1f669326ac26bb08af7c929180a61ba72915e5a1fb081b69914835ccfe9e`. Verify these and the stored preparation signature, not a recomputed preparation ID incorporating future code. Its calendar deadline expired at `2026-10-05T08:01:07Z`. Finish its reviewed `stopped_environment` closeout; never replace its manifest, clock, or terminal record.

New outputs live separately under `artifacts/distillation-v1-executions/<execution-id>/`. An immutable `execution.json` binds schema version, creation UTC, source root/ID/hashes, original stop-record hash, unchanged scientific-config hash, runtime/lock identity, policy and historical costs. Execution ID is SHA-256 of canonical manifest content excluding its own ID. Creation UTC makes registration distinct but is not a reason to rerun scientific candidates. Refuse overwrite, links/reparse traversal, overlapping source/output roots, changed config/source bytes, or mismatched execution IDs.

Carry **1,168 CUDA seconds** and **8,066,699,435 downloaded bytes** exactly once from prior feasibility/preparation. The new active-runtime window starts at zero, explicitly a new approved execution policy, not a rewritten historical wall total. Artifact accounting includes the inherited generated-source/feasibility bytes plus new execution bytes; do not charge pre-existing shared model cache or count the same file twice. Preserve historical costs in reports.

## Runtime and resource accounting

The only policy change is wall-budget semantics for the new execution namespace:

- Active limit: 172,800 cumulative seconds across all stage jobs, including loading, CPU preparation/I/O within the stage job, successful work, failures and teardown. Exclude code implementation, reviews, idle chat and gaps between jobs. Count from stage-job start until confirmed process-tree exit, rounding cumulative seconds conservatively upward.
- CUDA limit: 86,400 cumulative seconds including the inherited 1,168; existing conservative row/startup/failure charging remains once per attempt. Where synchronized CUDA timing is unavailable, retain the existing conservative worker-wall upper bound. Do not add whole-stage CUDA time on top of row/event CUDA charges.
- Downloads: 21,474,836,480 cumulative bytes including inherited download costs. Artifacts: 21,474,836,480 bytes; reserve 1 MiB for bounded terminal metadata. Include logs, events and status files in artifact projection.
- Memory: existing capacity-minus-2-GiB reserve, allocated **and** reserved checks; no CPU/disk model offload. Reserve 4 CUDA hours before training as already specified.

A replay reconstructs historical CUDA/download charges from authenticated journals exactly once. Wall runtime is reconstructed from settled job events plus the current job's monotonic elapsed duration, **not** summed from replayed row durations. A registered running job carries monotonic start, settled prior runtime, stage, attempt identity, supervisor/child ownership and frozen code/policy signatures. A stage's resource ledger consumes this runtime context; old calls with no execution context retain calendar-wall behavior for the expired-run closeout and legacy tests.

The supervisor checks active remaining runtime at most every second and kills its owned process tree at exhaustion. Stages continue existing immediate CUDA/memory/download/artifact checks; quota interruption cannot disable them. Supervisor/child state must agree on the same execution and attempt. Refuse standalone production `--execution-root` work without a valid supervisor-owned active-job context.

## Ownership, persistence and recovery

Acquire an exclusive per-execution supervisor lease before stage launch. Validate process identity with PID **and creation time**, using read-only Windows APIs; never use `os.kill(pid, 0)` on Windows. A live owner prevents duplicate launch. Hold process handles for owned children; never terminate a process merely because a stale file contains its PID.

Create stage children suspended, assign to a Job Object with kill-on-close before resuming, and prevent breakaway so no model worker can race outside containment. Do not inherit the supervisor's job handle. Fail closed if containment is unavailable. On timeout/error, terminate that job, wait for all owned descendants to exit, then settle costs and release ownership. If exit cannot be confirmed, retain ownership evidence and stop; no retry.

Append/fsync `events.jsonl` with canonical payload digests and contiguous sequence IDs. Record job reservation before spawning; record process identities and stage signature before resuming the stage. Use exclusive immutable stage/terminal records, append-only raw-answer journals, and atomic replace for nonauthoritative `status.json` and `status.md`. Flush per-attempt stdout/stderr logs directly to files. Write a heartbeat at most every five seconds, with runtime used/remaining and paths, not answer text. A final supervisor record must be written even when the child dies before producing a stage artifact.

Normal recovery requires matching source/policy/stage signatures, validated journal prefix, proven prior process-tree exit, sufficient remaining budget, and the existing maximum two recoveries per phase. Terminal stages do not restart. On supervisor crash/power loss, Windows containment kills descendants, but an unclosed active interval has no trusted end time: retain the heartbeat/attempt evidence, mark `stopped_environment` with uncertain accounting, and conservatively exhaust the remaining active allowance rather than silently refunding unknown work. Do not launch another job on this execution automatically. A new policy/registration after such a terminal stop needs explicit user direction.

## User-facing artifacts

Each execution contains:

- `execution.json`: immutable identity, policy, inherited costs.
- `events.jsonl`: authenticated append-only supervisor attempts, heartbeat/checkpoint references and settlements.
- `status.json` and `status.md`: convenient current stage/state, process identity, runtime used/remaining, CUDA/download/artifact totals, stop reason, and next action.
- `logs/<attempt-id>.stdout.log` and `.stderr.log`: retained output/error evidence.
- Existing `stages/`, `calibration/raw/`, `calibration/review/`, inventories and later checkpoints, under the new execution root; exact inner paths follow the existing stage implementation.

States: `registered`, `running`, `awaiting_review`, `completed`, `stopped_environment`, `stopped_scientific`. `completed` means the requested compute stage completed, not that the entire study succeeded. A zero child exit without an authenticated expected stage record is an environment failure, never success. Status views must reconstruct from authoritative records and clearly label a stale/missing heartbeat; a convenience file is not evidence of a pass.

Provide `register`, `launch`, `status` and supervisor-internal `run` CLI actions. A detached `launch` returns after the supervisor persists ownership/start evidence, or reports startup failure without claiming launch success. Report the execution root and status/log paths. A repeated `launch` of a live attempt must return a useful refusal, not start another process.

## Verification and release

CPU/fake tests cover legacy expired rejection before network/worker, separate source/output namespaces, inherited costs once, wall replay without double count, waiting days without budget charge, code/config/policy drift, path attacks, corrupt events, live/stale leases, missing stage artifacts, artifact/log growth, timed-out startup and immutable terminals.

On Windows, run short real CPU integration children in paths containing spaces: launcher exits while supervisor/child continue and append logs; deadline terminates a child and its sleeping grandchild; deliberate supervisor death closes the Job Object and kills descendants; two simultaneous launches yield one owner; `awaiting_review` stops runtime. These tests use a separate test policy and never alter production caps or source files. No network/GPU for implementation tests.

Release requires independent Luna spec/code review, root fresh focused tests plus one coherent full CPU suite, pinned lint/compile/diff checks, successful native detachment/containment evidence, then commit/push on `exp`. Before production: verify old terminal closeout, runtime/source hashes, register one fresh execution, freeze relevant code, launch calibration once, inspect durable ownership/status evidence. The computer must remain powered and awake; no reboot guarantee or review bypass is promised.

The original scientific plan remains binding for all prompts, data, model pins, token candidates, optimizer details, matched controls and quantitative gates. No claim of post-training gain exists until those gates have actually been evaluated.
