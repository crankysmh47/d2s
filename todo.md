# Current work queue

Updated 2026-09-11. Detailed requirements and stop rules: [plan.md](plan.md). This file is a summary, not permission to skip gates.

Exact sequential execution contract: [docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md](docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md). Continue automatically through passing gates; pause only for a genuinely human-only blocker.

## Completed research

- [x] Phase 0: partition/conversion/oracle foundation.
- [x] Real-model all-expert conversion checks.
- [x] Phase 1A: random partition local and rollout evaluation; negative quality result.
- [x] Phase 1A.5: reconstruction calibration; no condition met the registered gate.
- [x] Select and document the new bounded recurrence study; preserve historical roadmap.
- [x] Harden the protocol with per-family informativeness and paired FineWeb retention-uncertainty gates.
- [x] Write the exhaustive sequential implementation-to-report execution plan.

## Active direction: pretrained recurrence

- [ ] Implement pinned source/configuration and deterministic task data.
- [ ] Implement and verify the shared-block Qwen3 wrapper, with caches disabled.
- [ ] Implement block LoRA, loss masking and resumable training.
- [ ] Implement exact-answer evaluation, timing controls and decision gates.
- [ ] Implement staged CLI and honest success/stop reports.
- [ ] Run implementation checks and real-model correctness smoke.
- [ ] Pass the resource pilot within the 24-hour total study cap.
- [ ] Evaluate the three frozen controls on development data.
- [ ] Run seed47 discovery: ordinary LoRA, repeated-block LoRA, extra-training-time control.
- [ ] Only after discovery passes: run seed48 and sealed final evaluation.
- [ ] Report the measured result and stop; propose further work only from the evidence.

No recurrence code or experiment has run yet. Current request was planning/documentation only.

## Deferred, not an execution queue

Phase 1B clustering; sparse expert LoRA recovery; native-MoE diagnostics; MoE-to-dense merging; specialist assembly; adaptive loop selectors; quantization; cached loop deployment; custom pretraining. Offloading belongs to ExpertFlow. These do not start automatically if recurrence stops.
