# Verified-distillation QLoRA execution state

Plan: [revision 2](superpowers/plans/2026-09-27-verified-distillation-qlora.md). Branch `exp`; starting commit `8309b2a`. Authorized 2026-10-03 by the user's instruction to revise and continue. No separate design file: the scientific contract in sections 1–2 of the plan is binding.

## Decisions before output

- Practical improvement (A1/A2 vs F0) and teacher-specific improvement (A2 vs A1) are separate outcomes. A practical discovery pass can advance without a teacher-specific pass.
- Choose the output budget using only length/EOS on 60 separate questions under three registered candidates, then freeze before viability/training.
- F0 and adapters use identical NF4 precision. Fixed padding/tensor shapes and 300 updates match training work; report target-token and wall/CUDA differences honestly. The primary comparator is always A1, never chosen after test results.
- Resume validated complete inference rows or optimizer-boundary checkpoints after at most two recoveries; retain failed work/cost. Scientific no-go remains terminal.
- Work in the user's existing `exp` checkout; preserve unrelated untracked review files. Luna implements/reviews; root makes research decisions.

## Task status

| Task | State | Evidence / next action |
| --- | --- | --- |
| 0 dependency/GPU feasibility | Passed | Reviewed/pushed fix `50164c1`; 28 covering CPU tests. Recovery1 exited 0 and result validator passed: student loss 1.262591, gradient norm 4.516547, 392 changed adapter tensors; peak allocated/reserved 13,761,504,256 / 14,222,884,864 bytes, below 14,926,938,112-byte cap. Teacher BF16 generated 16 synthetic tokens; peak allocated/reserved 8,087,715,840 / 8,090,812,416 bytes. No benchmark result is implied. |
| 1 source/config/splits | In progress (CPU only) | Luna implements pinned sources, strict config, deterministic disjoint splits, retention authentication and minimal stage records. No benchmark outputs authorized before Task 0 passes. |
| 2 budget/scorer calibration | Pending | 60 exclusive questions; no pilot/held-out outputs yet. |
| 3 teacher qualification/corpus | Pending | No model benchmark or teacher data yet. |
| 4 trainer/retention | Pending | No training yet. |
| 5 discovery/confirmation/report | Pending | Only through passing gates. |

Record exact commands, commits, tests and gate artifacts as each task advances. GPU-active budget is 24 hours; failures/recoveries consume it. No training result is implied by completed code.

First feasibility attempt ran `.venv/Scripts/python.exe scripts/distillation_feasibility.py` in agent-local session 3063/PID 30540; it exited 1. The raw student loss/gradient/peak metrics were not persisted, so only control-flow evidence shows its update guards passed. Attribute no numerical model performance to this attempt. Recovery will use sequential fresh GPU worker processes to guarantee phase teardown without assuming the owner of the residual 64 MiB. The completed teacher cache is reused; original failed artifacts remain immutable.

Recovery command: `.venv/Scripts/python.exe scripts/distillation_feasibility.py --artifact-root artifacts/distillation-v1/feasibility-20261003-recovery1 --prior-download-bytes 8066699435` (root session 14930/PID 6800). It ran 08:41:41–08:42:18 UTC on 2026-10-03. Independent scoped review closed all findings before launch. Result SHA-256: `118a6c6733f5a7fd8fedcc47d2a4952bea6cfe7885059f87816703b560d06f57`. Both GPU children exited sequentially; post-run device usage returned to 1,075 MiB. Device snapshots are not whole-device peaks.

Charge the failed attempt's full 1,130-second wall duration plus recovery's 38-second rounded-up wall duration conservatively against CUDA-active budget (1,168 seconds); actual first-attempt GPU duration was not persisted. Begin the 48-hour wall clock conservatively at the first attempt's 08:01:07 UTC start. The supplied prior download charge includes teacher, original GSM8K, the extra GSM8K cache copy, and SVAMP; keep it conservative until Task 1 reconciles its source ledger. No further model download occurred in recovery.
