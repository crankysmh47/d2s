# Distillation Background Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use GPT-6 Luna implementers/reviewers as the user requested; root owns decisions, commits and push.

**Goal:** Run registered distillation compute independently of chat lifetime, with a cumulative 48-active-hour allowance and durable, inspectable evidence.

**Architecture:** Keep the expired source run immutable and register a separate execution namespace. A detached Windows supervisor owns a contained stage-process tree and settles active runtime between jobs; existing stage journals retain once-per-attempt CUDA/download costs. Reviews stop compute instead of consuming the wall allowance or being bypassed.

**Tech Stack:** Existing Python 3.11 virtual environment, standard library/ctypes Windows APIs, pytest, existing immutable JSON and raw-row journal helpers. No added dependencies.

**Spec:** [Background execution design](../specs/2026-10-07-distillation-background-runtime-design.md). The original [scientific plan](2026-09-27-verified-distillation-qlora.md) governs unchanged scientific decisions.

## Global Constraints

- Work on `exp`; preserve unrelated `task-14*.md`; do not merge to `main`.
- New execution: 172,800 cumulative active stage-job seconds; exclude review/agent/idle gaps. Legacy source run retains its expired calendar deadline.
- Carry 1,168 CUDA seconds and 8,066,699,435 download bytes once. CUDA cap 86,400; download/artifact caps 21,474,836,480 bytes each; reserve 1 MiB terminal metadata and existing 2-GiB desktop memory allowance.
- Keep config/model/data/prompt/scorer/scientific gates unchanged. No dependency/lock edits, GPU/network work, or real-source materialization during implementation tests.
- No production run before independent reviews, root verification, old terminal closeout, and explicit registration of the approved new policy.
- Supported production launch is `calibrate` only until later implemented stages gain their own prerequisite adapters. Never silently run an unimplemented stage.
- Save code/tests/runtime/source signatures before first output. No production budget overrides, review bypasses, live-worker duplicates or arbitrary-PID termination.
- Parent source root and new execution root must be disjoint; source preparation stays read-only.
- Root alone commits/pushes. Implementers report and freeze files; reviewers are read-only except their report.

## Review Focus

- A stopped chat must not kill a detached compute job: Task 2 real Windows launcher-exit test.
- PID reuse or a supervisor crash must not kill another process or leave model descendants: Task 2 handle/creation-time and kill-on-close tests.
- Review gaps and replayed rows must neither exhaust active wall time nor refund consumed CUDA: Task 1 clock/replay tests and Task 2 settlement test.
- Output namespace changes must not weaken source/gold/sealed controls: Task 1 separate-root poison-path tests.
- Logs, heartbeat files and a missing child result must not hide a cap crossing or fake success: Task 2 artifact-growth and missing-record tests.

## Release sequence

First finish the existing original Task 2 early-resource fix and scoped review; root invokes the old expired-run rejection without network/GPU and preserves its immutable terminal record. Only then modify the interfaces below. Task 1 and Task 2 here are sequential independent review gates; do not run multiple implementers on shared files. At release, perform one whole-change independent review and one coherent full CPU suite. Routine approval pauses are waived by the user's standing autonomous-execution instruction; scientific gates are not.

### Task 1: Separate execution identity and active-runtime accounting

**Files:**

- Create `src/ppmoe/distill/execution.py` and `tests/test_distill_execution.py`.
- Modify `src/ppmoe/distill/evaluate.py` (`ResourceLedger`), `src/ppmoe/distill/study.py` (calibration signatures/replay/source-output separation and dispatch), `scripts/run_distillation_v1.py` (new execution/context arguments).
- Modify `tests/test_distill_evaluate.py` and `tests/test_distill_study.py` only for the new paths and legacy regressions.

**Interfaces:**

```python
# execution.py: authoritative identity/event/runtime primitives, no top-level
# import of study/evaluate; import existing path/atomic helpers locally.
@dataclass(frozen=True)
class ActiveJobContext:
    execution_id: str
    attempt_id: str
    stage: str
    prior_active_seconds: int
    started_monotonic_ns: int
    supervisor_pid: int
    supervisor_created_utc: str
    stage_signature: str

def register_execution(source_root: Path, executions_root: Path, *,
                       repo_root: Path, config_path: Path) -> Path: ...
def validate_execution(execution_root: Path, *, repo_root: Path,
                       config_path: Path) -> dict[str, Any]: ...
def read_events(execution_root: Path) -> list[dict[str, Any]]: ...
def append_event(execution_root: Path, event: Mapping[str, Any]) -> None: ...
def validate_job_intervals(execution_root: Path) -> dict[str, dict[str, Any]]: ...
def active_elapsed_seconds(context: ActiveJobContext, *,
                           now_ns: int | None = None) -> int: ...
def process_creation_utc(pid: int) -> str | None: ...
def execution_stage_signature(execution_root: Path, stage: str, *,
                              repo_root: Path) -> str: ...
def load_active_job(execution_root: Path, attempt_id: str, *,
                    stage: str) -> ActiveJobContext: ...
```

`active_elapsed_seconds` returns `prior_active_seconds + ceil(max(0, now_ns-started_ns)/1e9)`. Inject clocks in tests, never cap overrides in production. `process_creation_utc` uses read-only Windows process queries and canonical microsecond UTC-Z precision; inaccessible/dead/unsupported cannot establish ownership. `execution_stage_signature` hashes the implemented stage's closed prelaunch code/config/source/policy set, without requiring not-yet-authenticated model inventories. `ActiveJobContext.stage_signature` is this prelaunch signature; final calibration additionally binds model inventories. `load_active_job` validates the signed event/current-job record, frozen execution/stage identity and live supervisor PID+creation time; no standalone unowned stage execution.

Reviewed interface clarification: prove actual liveness with `WaitForSingleObject(handle, 0) == WAIT_TIMEOUT` before and after reading creation time, using `SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION`; creation time alone accepts exited Windows processes with retained handles. `load_active_job` also matches `stage_pid`/`stage_created_utc` to the invoking process. `validate_job_intervals` is the single ordered lifecycle validator consumed by it, study replay and the supervisor: first prior zero, each subsequent prior equals previous cumulative settlement, no overlapping/duplicate/out-of-order intervals, owner/signature matches, exact integer clocks and ceil-duration settlement. Settlement `prior_active_seconds` must repeat its start, not merely yield a correct derived total.

Its return maps attempt ID to `start`, `settlement` (payload or `None`), `started_monotonic_ns`, `ended_monotonic_ns` (integer or `None`), `prior_active_seconds` and `settled_active_seconds` (integer or `None`). `current-job.json` is canonical `{payload: <exact unmatched job_started payload>, payload_sha256: SHA256(canonical(payload))}` plus LF. `job_settled` copies its start payload, changes `type`, adds end monotonic timestamp and **cumulative** `settled_active_seconds`. Worker reservations add `job_attempt_id` and `reserved_monotonic_ns`; bound unknown worker CUDA cost by its authenticated settled/live job interval, never current UTC across a review gap. Missing interval fails closed. Charge startup exactly once whether zero or one raw rows survived.

```python
# Existing signatures extended keyword-only, old callers unchanged.
def run_calibration_stage(run_root: Path, config: Any, *, repo_root: Path,
                          config_path: Path, worker_loader: Any | None = None,
                          execution_root: Path | None = None,
                          active_job: ActiveJobContext | None = None
                          ) -> dict[str, Any]: ...
def run_stage(stage: str, config_path: Path = DEFAULT_CONFIG,
              run_root: Path | None = None, *,
              execution_root: Path | None = None,
              attempt_id: str | None = None) -> StageRecord: ...
```

`--run-root` remains the prepared source; `--execution-root` and `--attempt-id` are paired options for `calibrate` only. Return the execution ID/root in new-mode `StageRecord`, with explicit `source_run_id` and execution-policy hash in details. Old-mode records retain original semantics.

- [ ] **Step 1: Add failing identity/runtime tests.** Create a synthetic authenticated prepared fixture, old environment-stop record and tiny files using pytest `tmp_path`. Test registration is exclusive/content-addressed; source hash/config/stop hash tampering fails; overlapping/reparse roots fail; inherited costs appear once. Add this exact clock test:

```python
def test_active_runtime_excludes_between_job_gap():
    first = ActiveJobContext("e", "a", "calibrate", 0, 100_000_000_000,
                             1, "2026-10-07T00:00:00Z", "s")
    assert active_elapsed_seconds(first, now_ns=105_100_000_000) == 6
    second = ActiveJobContext("e", "b", "calibrate", 6, 900_000_000_000,
                              1, "2026-10-09T00:00:00Z", "s")
    assert active_elapsed_seconds(second, now_ns=902_000_000_000) == 8
```

Run `.venv/Scripts/python.exe -m pytest -q tests/test_distill_execution.py`; record import/assertion RED failures.

- [ ] **Step 2: Implement immutable registration and authenticated events.** Build schema 1 manifest per spec. Compute inherited artifact byte inventory after old terminal closeout; bind paths/hashes, no interpretation of sealed labels. Use canonical JSON SHA-256, existing exclusive atomic JSON writer and path guard. Each event has `sequence`, `previous_digest`, `payload`, `payload_sha256`; digest includes sequence/prior digest, not just payload. Reject truncated/malformed/tampered events; fsync append before exposing status. Refuse registration if source does not have authenticated `stopped_environment` or if calibration raw model outputs already exist in the old run. Validate immutable source/stop/config/lock hashes every launch. Runtime identity from passed feasibility is checked, not invented.

- [ ] **Step 3: Add failing new-mode production-path tests.** Extend existing patched-Hub/worker integration fixture so source stays byte-for-byte unchanged while selected raw rows/review packets/stage JSON appear only under a distinct execution root. Test a multi-day gap then resume charges only new active time and preserves prior raw/startup CUDA once. Poison sealed interpretation and inference gold access. Old no-context expired path must still stop before Hub/worker; test absence of paired CLI args and invalid job identity. Run focused study/evaluate tests and record RED.

- [ ] **Step 4: Implement narrow source/output separation and wall policy.** Use distinct local `source_root` and `output_root`; preparation/manifest/split/private-gold reads target source root, journals/inventories/locks/reviews/stages target output root. Resolve private gold only after length selection as before. Add the execution policy/job identity to calibration signature and transitive signature list, including `execution.py`/its tests and later runner files before production freeze.

Extend `ResourceLedger` with JSON-serializable scalar fields for wall mode, settled prior seconds and current monotonic job start. Legacy `initial()` and legacy mode still use original UTC origin. New mode uses the authenticated active job's clock at every charge; replay never adds historical row wall durations. Preserve these fields with `dataclasses.replace` rather than hand-copying a subset. All replay call sites receive the same active context; inherited costs initialize once, not once per replayed event. Record truthful `wall_mode` in stage results. Keep current worker failure costs, retry limits and early durable stop paths intact. Include inherited artifact baseline plus all new output bytes; never include pre-existing model cache. Avoid broad `study.py` refactoring.

- [ ] **Step 5: Verify, self-review and hand off.** Run:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_distill_execution.py tests/test_distill_study.py tests/test_distill_evaluate.py tests/test_distill_model.py tests/test_distill_answers.py
uvx ruff@0.15.0 check src/ppmoe/distill/execution.py src/ppmoe/distill/evaluate.py src/ppmoe/distill/study.py scripts/run_distillation_v1.py tests/test_distill_execution.py tests/test_distill_evaluate.py tests/test_distill_study.py
.venv/Scripts/python.exe -m compileall -q src/ppmoe/distill scripts/run_distillation_v1.py
git diff --check
```

Write the task report with RED/GREEN commands/output and exact usable interfaces, freeze owned files. Root commits only the named paths, creates a review package, dispatches fresh Luna for spec/code review and resolves important findings before Task 2. No production registration or model work in Task 1.

### Task 2: Detached supervisor, owned watchdog, logs and status

**Files:**

- Create `src/ppmoe/distill/background.py`, `src/ppmoe/distill/windows_job.py`, `scripts/run_distillation_background.py`, `tests/test_distill_background.py`, `tests/test_distill_windows_job.py`.
- Modify `src/ppmoe/distill/execution.py` only where the reviewed Task 1 event/context interface needs supervision integration; update its tests for that change.
- Modify `src/ppmoe/distill/study.py` only to add runner/script/tests to new-mode transitive code signatures before first outputs.
- Update `docs/distillation-v1-execution-state.md`, `PROJECT_LOG.md`, `todo.md` with actual implemented/verified status and operating commands. Do not edit scientific config/lock/scorer.

**Interfaces:**

Consume Task 1 `register_execution`, `validate_execution`, `read_events`, `append_event`, `ActiveJobContext`, `active_elapsed_seconds` and `load_active_job`. Produce:

```python
# windows_job.py: use ctypes, no added package; hold owned handles.
class WindowsJob:
    def start(self, argv: Sequence[str], *, cwd: Path,
              stdout_path: Path, stderr_path: Path) -> dict[str, Any]: ...
    def resume(self) -> None: ...
    def poll(self) -> int | None: ...
    def terminate_and_wait(self, timeout_seconds: float = 10.0) -> bool: ...
    def close(self) -> None: ...

# background.py
def launch_stage(execution_root: Path, stage: str, *, repo_root: Path,
                 config_path: Path) -> dict[str, Any]: ...
def supervise_stage(execution_root: Path, stage: str, *, repo_root: Path,
                    config_path: Path) -> int: ...
def execution_status(execution_root: Path) -> dict[str, Any]: ...
def execution_stage_files(stage: str) -> tuple[str, ...]: ...  # in study.py
```

Before first answers, generalize only common lifecycle naming to registered compute names (`KNOWN_STAGES` excluding preflight/prepare); signing/launch still reject unimplemented stages. `execution_stage_signature` consumes `study.execution_stage_files`, whose only initial branch is calibration with the closed supervisor/launcher/native-job/test files included. Future scientific-plan tasks may add previously unused stage-specific registry/dispatch branches without changing frozen common execution/ledger/scoring helpers. Do not implement those future stages in this task.

Completed records retain their original signature and originating revision; status/recovery must not recalculate a settled stage signature against a later shared dispatcher. Whole-file hashes of `study.py`/CLI can differ after independently reviewed unused-stage additions. Later prerequisite authentication must verify the recorded revision/hashes, immutable artifacts and unchanged frozen leaf bytes; root independently checks that shared-file differences affect only previously unused code. Newly requested work binds its current reviewed transitive source. Never rewrite or re-sign historical results. The immutable policy's `supported_stage=calibrate` denotes initial registration scope; future reviewed unused adapters do not mutate that manifest.

Use mutable <=5-second status heartbeats and sparse authoritative lifecycle events, not full-journal parsing every watchdog poll. Copy saved start identities at settlement instead of querying a dead stage. A confirmed failure produces a trusted settlement plus separate terminal outcome; `job_failed` denotes uncertain/unsettled accounting and must not allow recovery/refund. Launch acknowledgment must match its own spawned supervisor, not a concurrent winner; authenticated matching completion can acknowledge a very short stage that already exited.

`start` creates suspended, contained child and returns PID/creation time; `resume` happens after its reservation/context is durable. `poll` is owned stage exit, not proof all descendants exited; confirm Job Object active process count is zero before settlement. `close` is kill-on-close for abrupt supervisor loss. Set correct ctypes arg/restypes, Unicode quoting via `subprocess.list2cmdline`, noninherited job handle, inheritable log handles limited to child startup, correct handle cleanup. Production unsupported platforms fail closed before launch; CPU unit tests inject a job implementation but native Windows acceptance uses the real one.

- [ ] **Step 1: Write failing supervisor-state tests.** Use injected monotonic clock and contained CPU child/job fixture. Assert one owner, immutable stage refusal, invalid policy/code hash refusal, zero exit without expected signed artifact => environment stop, scientific stop remains terminal, review completion stops active budget, no automatic next-stage launch, and log/artifact growth triggers termination. Add status reconstruction with stale convenience file and corrupt event stream. Run `.venv/Scripts/python.exe -m pytest -q tests/test_distill_background.py` and record RED.

- [ ] **Step 2: Implement containment and supervisor state machine.** Acquire exclusive lease with PID+creation time. Validate execution and registered implemented stage before launch; append reservation, open file logs, create Job Object/contained suspended stage, persist current-job context and process identities, then resume. The stage argv is an argument list, never shell interpolation:

```python
argv = [str(repo_root / ".venv/Scripts/python.exe"),
        str(repo_root / "scripts/run_distillation_v1.py"), stage,
        "--config", str(config_path), "--run-root", str(source_root),
        "--execution-root", str(execution_root), "--attempt-id", attempt_id]
```

Every <=1 second check owned exit, active remaining time and artifact size including logs/status/journals. Heartbeat every <=5 seconds. On exhaustion/exception: terminate job, confirm zero owned descendants, append truthful settlement and immutable bounded terminal record. Preserve ownership marker on uncertain cleanup. On normal exit authenticate expected stage signature/digest/status; settle total monotonic duration once. Set `awaiting_review` with packet paths, `completed` for a successful requested stage, or actual terminal reason. Never infer scientific success from exit code. Before recovery prove previous processes dead and use stored settlement; missing settlement is uncertain-cost terminal, not a free restart.

- [ ] **Step 3: Write native failing detachment/ownership tests.** Tests create short temporary CPU scripts using pytest fixtures, not model code. Native tests run in a directory with spaces and assert observable process/file evidence:

```python
def test_detached_supervisor_survives_launcher_exit(tmp_path):
    # Fixture launches a supervisor with a short CPU child which flushes
    # "started", waits 1 second, flushes "finished", and writes a signed
    # fake-stage record. Launcher must exit before "finished" appears.
    root = launch_detached_cpu_fixture(tmp_path / "path with spaces")
    assert root.launcher.wait(timeout=5) == 0
    assert root.wait_for_final_status(timeout=10)["state"] == "completed"
    assert "finished" in root.stdout_path.read_text()
```

Define `launch_detached_cpu_fixture` inside tests; production CLI has no arbitrary-command, fake-stage, short-cap or review-bypass flags. Also test owned parent+grandchild timeout, intentional supervisor termination kills descendants, simultaneous launch exclusion, and failed Job assignment cleans suspended child. No long sleeps; bounded test waits <=15 seconds. Record failures before implementation.

- [ ] **Step 4: Implement detached CLI and convenient status files.** `run_distillation_background.py` has actions `register`, `launch`, `status` and internal `run`, with explicit paths. `register` prints new execution root; `launch` spawns hidden detached supervisor (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`, no shell), redirects supervisor bootstrap errors to a retained attempt log and waits at most 10 seconds for durable ownership/start evidence. Return failure if no evidence; verify supervisor/contained job life before claiming launched. `status` works without torch/model loading and prints readable JSON plus the `status.md` path. Mutable status snapshots use same-directory temporary file/fsync/atomic replace; authoritative events and terminal records never overwrite. Limit status size and repeated heartbeat/log projection under the artifact cap. Fix malformed/truncated event handling with fail-closed diagnostics preserving bytes.

- [ ] **Step 5: Verify independently and release.** Focused commands:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_distill_execution.py tests/test_distill_background.py tests/test_distill_windows_job.py tests/test_distill_study.py tests/test_distill_evaluate.py
uvx ruff@0.15.0 check src/ppmoe/distill/execution.py src/ppmoe/distill/background.py src/ppmoe/distill/windows_job.py src/ppmoe/distill/study.py scripts/run_distillation_background.py tests/test_distill_execution.py tests/test_distill_background.py tests/test_distill_windows_job.py
.venv/Scripts/python.exe -m compileall -q src/ppmoe/distill scripts/run_distillation_background.py
git diff --check
```

Write report/freeze, root commits named files and dispatches independent Luna review. Resolve important findings; run root full CPU suite with the project's existing GPU deselections, not a changed test policy. Whole-change review covers Task 1+2 interfaces, native containment evidence and historical closeout. Push `exp` only, preserve unrelated files.

- [ ] **Step 6: Register and launch the one approved production calibration.** Root verifies unchanged source/preparation/runtime hashes, old stop and absence of previous answers, then invokes:

```powershell
.venv/Scripts/python.exe scripts/run_distillation_background.py register --config configs/distillation-v1.toml --run-root artifacts/distillation-v1/acd8dfc400c8f079d0403da5d4b981b8aac8e94226db8760f632c8307ff785c2 --executions-root artifacts/distillation-v1-executions
```

Use the exact root printed by registration for `launch calibrate` and `status`; record it in execution state, not a guessed placeholder. `launch` argument format: `launch calibrate --config configs/distillation-v1.toml --execution-root <printed absolute root>`. Inspect current-job record/actual owned processes, persisted events/logs and a later heartbeat after launcher exit. The GPU stage may still be running at handoff; report that honestly. On `awaiting_review`, dispatch actual Luna extraction reviews using original Task 2 contract, preserving editable human sheets. Remaining scientific-plan tasks proceed only on those gates; do not claim training ran or the entire study completed.

## Plan self-review

Coverage: identity/cost policy/source separation owned by Task 1; detachment/watchdog/process containment/logs/status/recovery owned by Task 2. Legacy expiration covered in Task 1 regression; independent gate handling and no unimplemented-stage dispatch covered in Task 2. Shared context and signature interfaces are explicitly named. Native CPU integration is required, not replaceable with mocks. The original teacher/training/evaluation implementations remain the original plan's later tasks; this plan does not pretend those are already implemented.
