# Current work queue

Updated 2026-10-07. Historical V0 requirements and stop rules: [plan.md](plan.md). This file is a summary, not permission to skip gates.

Historical recurrence execution contract: [docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md](docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md). It is not an active queue after V0–V3 stops.

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

## Authorized V1 study: registered calibration stop

Design: [2026-09-26-recurrence-v1-design.md](docs/superpowers/specs/2026-09-26-recurrence-v1-design.md). Detailed sequential contract: [2026-09-26-recurrence-v1-execution.md](docs/superpowers/plans/2026-09-26-recurrence-v1-execution.md). This is a new run; V0 remains terminal.

- [x] Freeze deterministic final-answer scoring and graph explanation audit; prove it on old answers and adversarial cases (23 parsed, 18 correct, five wrong, seven ungradable; SHA256 in V1 spec).
- [x] Run all six fresh 30-example difficulty pilots and apply the frozen candidate gate. Only expressions qualified; fewer than two families means `stopped_scientific` before development generation.
- [x] Seal the V1 terminal report and independently check pilot counts, hashes, and the absence of development/sealed/training artifacts. See [V1 calibration report](docs/recurrence-v1-calibration-report.md).
- [ ] Frozen F0/F1/F2 controls, LoRA discovery, retention/timing, seed-48 confirmation, and sealed evaluation: **not run, and not authorized in this stopped V1 run**.

V1 provides no recurrence-efficacy result. Its no-go motivated the separately registered V2 design below; V1 remains sealed.

## Recurrence V2: design review

- [x] Approve the three-family measurement-repair direction in conversation.
- [x] Approve the written [V2 design](docs/superpowers/specs/2026-09-26-recurrence-v2-design.md); user directed autonomous continuation through process-only reviews.
- [x] Implement and review the [V2 calibration plan](docs/superpowers/plans/2026-09-26-recurrence-v2-calibration.md); stop at the registered scientific gate.
- [x] Implement and independently review V2 calibration Tasks 1–5; freeze hashes before output (details in [PROJECT_LOG.md](PROJECT_LOG.md)).
- [x] Run one V2 production `prepare` and `calibrate`, independently audit the terminal `report`, and follow only the registered gate branch. The development gate failed; see the [V2 calibration report](docs/recurrence-v2-calibration-report.md).

## V2 answer audit: separate diagnostic, not a V2 rescore

Design: [answer-audit design](docs/superpowers/specs/2026-09-27-recurrence-answer-audit-design.md). Execution: [answer-audit plan](docs/superpowers/plans/2026-09-27-recurrence-answer-audit.md).

- [x] Authenticate the signed V2 development evidence and export gold-free answer packets.
- [x] Obtain separate Luna extraction reviews and reconcile them against deterministic answers.
- [x] Report parser misses versus wrong or incomplete answers; preserve V2's terminal stop. See [audit report](docs/recurrence-v2-answer-audit-report.md).
- [x] The bounded scorer repair supports a separately reviewed V3 protocol, frozen before fresh model outputs. Training remains gated.

## V3 fresh final-answer qualifier (terminal pre-GPU stop)

The audit supports one fresh protocol: [V3 design](docs/superpowers/specs/2026-09-27-recurrence-v3-design.md) and [calibration-first plan](docs/superpowers/plans/2026-09-27-recurrence-v3-calibration.md). V2 stays stopped.

- [x] Implement, independently review, test, hash-freeze and push the V3-only scorer and blind-control harness (`01c9744`).
- [x] Run the one-shot replacement-control qualification. It stopped on review-evidence validation before any V3 scorer pass; see [qualification report](docs/recurrence-v3-scorer-qualification.md). V3 is terminal, not qualified.
- [ ] Fresh config/splits, GPU calibration, recurrence controls and LoRA: **not run and not authorized in this stopped V3 candidate**.
- [x] Decide separately to draft a post-training distillation pivot. Do not silently retune/retry V3.

## Option 2: verified teacher-rationale QLoRA (authorized 2026-10-03)

Detailed [Option 2 plan](docs/superpowers/plans/2026-09-27-verified-distillation-qlora.md). This tests a narrow dense-model post-training gain, not MoE-like footprint. It does not reopen recurrence.

- [x] Write a gated plan comparing Qwen3-4B teacher-rationale QLoRA with matched gold-solution QLoRA on Qwen3-1.7B.
- [x] User authorized the four revisions and execution; revision 2 separates practical/teacher gains, calibrates token budget, matches tensor work, and allows validated recovery.
- [x] Verify the pinned GPU/QLoRA recipe with an independently reviewed synthetic smoke; preserve the failed first attempt and recovery costs.
- [x] Finish/review deterministic sources, sealed partitions, and authenticated retention provenance; signed prepare passed and root verified its hashes.
- [x] Check the October 6 resume against the original wall deadline; the October 5 deadline expired during the usage-limit interruption, before any calibration answers.
- [x] Finish/review the early-resource terminal-reporting fix and persist the old expired run as `stopped_environment`; source hashes/costs unchanged, no answers generated.
- [x] Obtain approval for a fresh execution policy: user selected 48 cumulative worker hours excluding review/agent downtime on October 7; old window remains expired, cumulative CUDA/download costs carry forward.
- [x] Implement/review runtime Task 1: separate execution identity/source outputs, cumulative active clock and interrupted-cost replay; root full CPU 1,195 passed/5 skipped/3 deselected.
- [x] Implement and independently review runtime Task 2, including producer-schema/status fixes; root native cross-terminal containment/detachment proved and final CPU suite passed 1,217 tests/5 skips/3 CUDA deselections.
- [ ] Finish whole-change engineering release review for the [runner](docs/superpowers/plans/2026-10-07-distillation-background-runtime.md), verify unchanged context, register one distinct execution and launch calibration. No production registration or new model worker yet.
- [ ] Freeze/review the final-answer scorer, calibrate output budget on 60 exclusive questions, and obtain blind Luna extraction reviews before the teacher viability pilot.
- [ ] Require teacher viability, >=800 matched verified targets, and the independent rationale-quality audit before any study training.
- [ ] Only if all gates pass: run paired discovery, one confirmation, sealed evaluation, and independent report. Stop on any registered no-go.

## Deferred, not an execution queue

Phase 1B clustering; sparse expert LoRA recovery; native-MoE diagnostics; MoE-to-dense merging; specialist assembly; adaptive loop selectors; quantization; cached loop deployment; custom pretraining. Offloading belongs to ExpertFlow. These do not start automatically if recurrence stops.
