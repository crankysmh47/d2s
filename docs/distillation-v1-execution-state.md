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
| 0 dependency/GPU feasibility | In progress | CUDA/BF16 available; torch 2.13.0+cu130, Transformers 5.16.1, Python 3.11.9. PEFT/bitsandbytes absent before changes. |
| 1 source/config/splits | Pending | Starts after feasibility or alongside non-GPU source preflight. |
| 2 budget/scorer calibration | Pending | 60 exclusive questions; no pilot/held-out outputs yet. |
| 3 teacher qualification/corpus | Pending | No model benchmark or teacher data yet. |
| 4 trainer/retention | Pending | No training yet. |
| 5 discovery/confirmation/report | Pending | Only through passing gates. |

Record exact commands, commits, tests and gate artifacts as each task advances. GPU-active budget is 24 hours; failures/recoveries consume it. No training result is implied by completed code.
