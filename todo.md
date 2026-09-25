# Current work queue

Updated 2026-09-25. Detailed requirements and stop rules: [plan.md](plan.md). This file is a summary, not permission to skip gates.

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

## Deferred, not an execution queue

Phase 1B clustering; sparse expert LoRA recovery; native-MoE diagnostics; MoE-to-dense merging; specialist assembly; adaptive loop selectors; quantization; cached loop deployment; custom pretraining. Offloading belongs to ExpertFlow. These do not start automatically if recurrence stops.
