# Recurrence V1 calibration report — registered stop

Date: 2026-09-26. Run ID: `4f1ddbedf7b4cbd11d73699d579a3e299a049aa8fb2b036e9124842d695eb2ca`. The authoritative local evidence is under `artifacts/recurrence-v1/<run-id>/`; the artifacts are ignored, immutable, and not included in Git. This document is an interpretation of those artifacts, not a replacement stage record.

## Decision

`prepare` passed. `calibrate` completed all six preregistered 30-row F0 pilots and sealed `stopped_scientific`; `report` sealed the same outcome with `next_stage=complete`. The gate required at least two families with a candidate whose baseline accuracy was strictly above 5% and below 95%, and whose invalid-format rate was at most 10%. Only expressions qualified. No selected development or sealed split was created, no frozen F1/F2 controls were run, and no LoRA training or retention/confirmation evaluation occurred. This is a **benchmark calibration no-go**, not evidence for or against adapted recurrence.

| Family | Candidate 1 correct / invalid of 30 | Candidate 2 correct / invalid of 30 | Gate result |
| --- | ---: | ---: | --- |
| Expressions | 28 / 2 | 28 / 2 | Candidate 1 eligible and selected by the frozen first-eligible rule |
| Lists | 0 / 16 | 0 / 18 | Neither eligible |
| Graphs | 20 / 7 | 21 / 6 | Neither eligible; invalid rates 23.3% and 20% |

The frozen scorer SHA256 is `2f138e6fe106c81dd03712d0a49a3a916cec2596a4acfa8f4e111f6463bab719`. The calibration stage SHA256 is `605e63abb58a9f901f2141357f20e75d369676afba4092b6a2448f47a06fae10`; the terminal report stage SHA256 is `7bf0cb016e0daae3f9f7dead5fe42d7a898d1ff753ad72c015ecb301aaa12daa`. CUDA-active calibration time was 666.724 seconds; peak reserved VRAM was 8,547,991,552 of 17,074,421,760 bytes.

## Independent evidence check

The six pilot split/manifest and score hashes match the calibration stage bindings. Each raw score file has exactly 30 distinct rows; independent tallies reproduce the table. The calibration stage records `dev=null`, `sealed=null`, and no selected difficulty; no development, sealed, or training artifact was found. The report binds the passed prepare and stopped calibration stage hashes. The full project suite before the GPU run was `791 passed, 3 skipped`; Ruff and compileall passed. These software checks do not turn the scientific no-go into a pass.

## Post-hoc diagnosis, outside the V1 gate

The graph scorer rejected five candidate-1 and three candidate-2 completions that ended normally, plus two and three length-capped completions respectively. Several normal-EOS responses have a natural final `**Answer: No**`, bare `no`, or Markdown-formatted conclusion that the frozen grammar did not accept. At least one also opens with `Yes` and ends with `No`, so simply taking the last answer would silently accept a contradiction. The graph explanation audit separately found contradicted explanations even for some final-answer-correct rows. A new grammar needs pre-output tests for Markdown, bare answers, endpoint conclusions, and contradictory cues; V1 scores must not be retrospectively changed.

For lists, 16 responses in each candidate hit the 256-output-token cap. All 30 answers per candidate failed objective scoring, including the non-capped answers. Inspection of examples suggests the template's definitions of all available operations can lead the model to apply operations beyond the requested sequence. That is a task-wording validity hypothesis, not a measured prevalence claim or a recurrence result. A future protocol should make the requested operation sequence unambiguous and verify task comprehension on fresh pilot data before training. This would repair measurement; it is not a prompt-based performance improvement claim.

## Safe next step

If a further study is approved, write and preregister a **V2** evaluator/benchmark revision, with fresh pilot seeds and the same explicit informativeness, invalidity, resource, and stop gates. Treat V1 as terminal. Only after a new baseline has at least two informative families should the frozen controls and paired ordinary/recurrent LoRA comparison be considered. Do not run the remaining V1 plan stages or tune thresholds against these observed outputs.
