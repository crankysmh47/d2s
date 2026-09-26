# Current work queue

Updated 2026-09-26. Historical V0 requirements and stop rules: [plan.md](plan.md). This file is a summary, not permission to skip gates.

Exact sequential execution contract: [docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md](docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md). Continue automatically through passing gates; pause only for a genuinely human-only blocker.

## Completed research

- [x] Phase 0: partition/conversion/oracle foundation.
- [x] Real-model all-expert conversion checks.
- [x] Phase 1A: random partition local and rollout evaluation; negative quality result.
- [x] Phase 1A.5: reconstruction calibration; no condition met the registered gate.
- [x] Select and document the new bounded recurrence study; preserve historical roadmap.
- [x] Harden the protocol with per-family informativeness and paired FineWeb retention-uncertainty gates.
- [x] Write the exhaustive sequential implementation-to-report execution plan.

## Pretrained recurrence V0: registered stop reached

- [x] Implement and test pinned source/data, shared-block Qwen3, LoRA/training, evaluation, gates, staged CLI, and reporting.
- [x] Pass G0 source/data, G1 real-model smoke, and G2 resource pilot in signed run `129d39646933`.
- [x] Complete G3 F0/F1/F2 development and retention diagnostics. G3 stopped scientifically: F0 had no informative family (4/600 correct overall).
- [x] Produce G6 terminal report and independently check its input hashes and decisive raw metrics.
- [x] Preserve implementation-failure runs and document the G6 wording erratum in [PROJECT_LOG.md](PROJECT_LOG.md) and [docs/recurrence-v0-report-erratum.md](docs/recurrence-v0-report-erratum.md).
- [x] Commit and push the observed-evidence closeout on `exp`.

G4 seed-47 discovery, G5 seed-48/sealed confirmation, and any claim about adapted recurrence quality are **not evaluated**. The registered stop forbids retuning this run; a new benchmark or protocol needs a separately approved study.

## Authorized follow-up: output-length and answer-review diagnostic

Detailed bounded contract: [recurrence-review-plan.md](docs/recurrence-review-plan.md).

- [x] Export editable Markdown `good/bad` review tables with validation and reviewer identity.
- [x] Implement a separate 256-output-token baseline probe within 512 total tokens; keep V0 unchanged.
- [x] Test/review code and run one balanced 30-example development probe.
- [x] Have Luna grade a separate copy, explicitly as model review; retain a blank human copy.
- [x] Record evidence: 24 EOS / 6 capped; Luna 18 good / 12 bad; strict parser 0/30. A fresh benchmark and revised gates are justified before training.

The blank human sheet is [artifacts/recurrence-review-v1/baseline-256/human-review/review.md](artifacts/recurrence-review-v1/baseline-256/human-review/review.md). Edit only the last column. The model-graded comparison is in `luna-review/review.md`. No training has started; the registered V0 stop still applies.

## Authorized V1 study: current execution queue

Design: [2026-09-26-recurrence-v1-design.md](docs/superpowers/specs/2026-09-26-recurrence-v1-design.md). Detailed sequential contract: [2026-09-26-recurrence-v1-execution.md](docs/superpowers/plans/2026-09-26-recurrence-v1-execution.md). This is a new run; V0 remains terminal.

- [ ] Freeze deterministic final-answer scoring and graph explanation audit; prove it on old answers and adversarial cases.
- [ ] Generate two bounded difficulty pilots per family, select deterministically, and pass a fresh development informativeness gate.
- [ ] Score F0/F1/F2 on the selected fresh development set and check resource evidence.
- [ ] If gates pass, run paired A1/A2/A3 LoRA discovery, retention, and timing controls; stop on any no-go.
- [ ] If discovery passes, run seed-48 confirmation and sealed evaluation, report and push the supported result.

## Deferred, not an execution queue

Phase 1B clustering; sparse expert LoRA recovery; native-MoE diagnostics; MoE-to-dense merging; specialist assembly; adaptive loop selectors; quantization; cached loop deployment; custom pretraining. Offloading belongs to ExpertFlow. These do not start automatically if recurrence stops.
