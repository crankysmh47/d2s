# Recurrence V2 calibration: registered scientific stop

Date: 2026-09-27. Branch: `exp`. Run ID: `18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d`. Frozen implementation: `5b9a680`; pre-output protocol/hash log: `PROJECT_LOG.md` (2026-09-27 entry). This was one separately registered V2 measurement-repair run. V0/V1 evidence was not rewritten, and V1-versus-V2 scores are not a model-improvement comparison because the benchmark and list wording changed.

## Decision

`prepare` passed; calibration reached all six pilots and 300 development F0 answers, then returned `stopped_scientific`. The terminal `report` records `next_stage=complete`. The pilot gate passed with two eligible families, but the development gate failed: macro accuracy was 0.66 (inside the required open 0.05–0.95 interval), global invalid rate was 0.31 (required at most 0.10), and only one family was individually informative (required at least two). No F1/F2 frozen controls, LoRA adaptation, retention comparison, seed-48 confirmation, or sealed evaluation ran. Thus V2 supplies no recurrence-efficacy or post-training improvement result.

## Pilot and development evidence

Each pilot candidate contains 30 fresh examples. Eligibility required accuracy strictly between 0.05 and 0.95 and invalid rate at most 0.10. The fixed fallback is candidate 1 when neither candidate qualifies.

| Family | Pilot c1 correct / invalid | Pilot c2 correct / invalid | Registered choice | Development correct / invalid |
| --- | ---: | ---: | --- | ---: |
| Expressions | 28 / 2 | 23 / 6 | c1, eligible | 94 / 6 of 100 |
| Lists | 9 / 21 | 12 / 18 | c1, fixed fallback | 32 / 68 of 100 |
| Graphs | 26 / 3 | 19 / 8 | c1, eligible | 72 / 19 of 100 |

Development totals were 198/300 final answers correct and 93/300 invalid. Its three-family macro accuracy was 0.66 and global invalid rate 0.31. Expressions alone satisfied the family informativeness rule. Graph explanation status, recorded separately from final-answer correctness, was 31 supported, 12 contradicted, and 57 none; two contradicted explanations accompanied correct final yes/no answers.

The failure is primarily an answer-measurement/format problem, not a demonstrated 68/100 list reasoning failure. All 100 list development completions ended with EOS; the 68 rejected ones had `invalid_format` detail “no complete integer vector follows a final-answer cue.” Some visibly present a final list under wording such as “Final list,” outside the frozen V1 list parser's accepted cue grammar. Graph invalids comprised 9 without an explicit recognized conclusion, 7 length-capped, and 3 conflicting explicit conclusions. These observations diagnose the stopped run; they do **not** authorize post-output parser edits, retrospective rescoring, a third difficulty candidate, or LoRA training under V2.

## Integrity and resource audit

An independent read-only Luna audit recomputed all three canonical stage hashes and prerequisite links, the deterministic run ID, config/code/train-view signatures, and all six pilot plus development split/sidecar/score raw hashes, counts, ordered IDs, and summaries. It independently reconstructed the two eligible pilot families, fixed list fallback, and failed development gate. The eight-entry CUDA ledger and its anchors validated. Calibration used 1,280.428839 CUDA-active seconds and 1,283.547 wall seconds, within the registered 7,200-second calibration caps; the whole-study CUDA cap was 86,400 seconds. The stage-bound artifact footprint was 7,466,800 bytes, well below 20 GiB. The current tree is 32,467 bytes larger because `calibrate.json` and `report.json` were written after that measurement; this is expected stage timing, not unbound experimental output.

The sealed 600-example split and sidecar were generated after the pilot gate and remain unopened/unscored. The audit checked only their presence and sizes. No training, checkpoint, or LoRA artifacts exist. The blank 30-answer [diagnostic review sheet](../artifacts/recurrence-v2/18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d/human-review/review.md) is not a completed human or model review and cannot alter the frozen gate.

## Next boundary

The V2 study is terminal. A new hypothesis and separately registered protocol would be required before any further answer-scoring or adaptation experiment. In particular, the observed “Final list” wording suggests a future benchmark should validate answer-cue coverage *before* model outputs, but it cannot repair this run's primary metric after exposure.
