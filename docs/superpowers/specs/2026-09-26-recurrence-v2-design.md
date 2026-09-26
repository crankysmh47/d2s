# Pretrained recurrence V2: measurement-repair study

Date: 2026-09-26. Status: written design awaiting user review. The user approved a three-family V2 design in conversation. This document does not authorize GPU generation or training until its implementation plan is reviewed.

## Purpose and scientific boundary

The question remains whether a shared extra pass through pretrained Qwen3-1.7B block `[12,16)`, adapted with rank-8 LoRA, can improve verified answer accuracy over ordinary LoRA and a training-time-matched ordinary-LoRA control while preserving general-language quality. Added inference computation is allowed but must be measured. This does not establish an MoE-like footprint.

V1 run `4f1ddbedf7b4cbd11d73699d579a3e299a049aa8fb2b036e9124842d695eb2ca` stopped at benchmark calibration: expressions qualified, graphs failed the invalid-format limit, and lists scored zero with frequent length caps. No recurrent LoRA training or frozen-control comparison occurred. V2 repairs those measurement issues and tests *fresh* examples. It is a response to V1's benchmark failure, not a fallback that guarantees an improvement. V0 and V1 stage records, scores, source files, and registered decisions remain unchanged. V1-vs-V2 accuracy is not a model-improvement comparison because the benchmark changes.

The selected design retains expressions, lists, and graphs. It rejects the quicker two-family option because dropping the failed list family after seeing V1 would make the follow-up easier to interpret as favorable family selection. It rejects a new task family because that would require new supervision and a larger research redesign. There will be one prespecified V2 calibration, not iterative prompt or threshold tuning.

## Isolated implementation and provenance

Create V2-only scorer, data/prompt, config, stage-runner, CLI, tests, and `artifacts/recurrence-v2/<run-id>/`. Reuse tested V1 source verification, atomic split writing, hash-bound stage records, CUDA cost ledger, score serialization, and terminal report mechanics where their contracts remain valid. Do not modify V1 scorer or run artifacts. A V2 run ID binds exact config bytes; model and tokenizer revision/file hashes; V0 train/dev source hashes; V1 pilot ID-set hash; V2 scorer, generator, training-view, and runtime-code hashes. Any change to a bound byte after `prepare` invalidates continuation rather than silently resuming under the same run ID.

Use the pinned local `Qwen/Qwen3-1.7B` snapshot at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, BF16 eager/cacheless greedy generation, `enable_thinking=False`, block `[12,16)`, 256 generated tokens and 512 total sequence tokens. No new model download, few-shot examples, teacher answers, model-selected checkpoints, or prompt search. The output allowance remains fixed; a length cap is ungradable and contributes to invalid rate.

## Graph answer repair, frozen before V2 outputs

Implement a deterministic graph-only V2 extractor. Expression and list final-answer extraction retain V1 semantics, including no oracle-assisted extraction, length/EOS rules, vector length from latent input, and separate explanation assessment. V2 may call unchanged V1 code for these families, but the V2 scorer signature must bind every called scorer file. A parsed answer is correct only if it equals the graph interpreter on the latent specification.

The V2 graph extractor collects only explicit answer candidates:

1. The first nonblank line is solely `yes` or `no`, case-insensitive, optionally followed by `.` or `!`; Markdown heading/bold around that token may be ignored. A leading `Yes` without a period is now valid, but a prose clause such as `Yes, because ...` is not a standalone cue.
2. A complete line beginning with `Answer:`, `Final answer:`, or `So, the answer is:` and ending in a standalone `yes` or `no`, with optional Markdown heading/bold and final punctuation. The cue may occur anywhere in the completion, so earlier and later explicit cues can conflict. Do not infer from arbitrary yes/no in edge prose.
3. A conclusive `there is [no] [direct or indirect] path from X to Y` sentence with endpoints exactly equal to the queried start/end, allowing Markdown emphasis around `no`, `path`, `X`, and `Y`. Do not treat a path statement about other endpoints as the answer.

Normalize only the permitted light Markdown decoration around these cues; do not strip arbitrary text until it happens to look like an answer. Preserve original-text evidence spans. If all explicit candidates agree, extract that value. If any explicit candidates disagree—including a leading `Yes` followed by `**Answer: No**`—return `invalid_format` with an ambiguity detail, not the last answer. If no explicit candidate exists, return `invalid_format`. Never inspect the gold answer while identifying or resolving candidates. Missing EOS, empty, capped, decoding-failed, or partially generated completions remain invalid.

Graph explanations are scored separately as `supported`, `contradicted`, `unverified`, or `none`, using the V1 deterministic path-edge check on complete responses. A correct yes/no with a false path explanation remains final-answer-correct but the contradiction is reported by family and arm; it cannot be hidden in the final report. Explanation status does not alter the frozen primary accuracy gate. Synthetic adversarial fixtures and read-only V1 answers can test grammar *before* V2 generation; V1 scores are never overwritten or retroactively promoted.

## List wording repair, applied symmetrically

The V2 list prompt is generated solely from the latent input and operation sequence. Its exact template is:

```text
Initial list: <compact JSON list>. Operations to apply in order: <comma-separated operation names>. Apply exactly these operations once each, from left to right; do not apply any operation not listed. Definitions for listed operations: <definitions in first-appearance order>. Return only the final list.
```

Use these fixed definition strings: `ROTL1 moves the first element to the end`; `REVERSE reverses the list`; `ADD1 adds 1 modulo 10 to each element`. If an operation occurs twice, its definition appears once in first-appearance order, but both occurrences remain in the operations-to-apply sequence. This removes the V1 ambiguity where definitions of unrequested operations looked like additional instructions. It does not include a worked example, target answer, or feedback. The same template is used for V2 pilot, development, sealed, and the prospective V2 training view; expression and graph prompts are unchanged from V1.
Render `<compact JSON list>` without spaces, join operation names with `, `, and join distinct first-appearance definitions with `; `. The placeholders describe deterministic substitutions, not text shown to the model.

The training view is a deterministic, separately hashed transformation of the existing 8,000 V0 train rows: preserve every row's latent specification, canonical ID, split, structure ID, gold answer, and order; change only list prompt text by the template above; require non-list prompts to be byte-identical. Verify every gold answer with `interpret`, every prompt-plus-answer training length under the existing 512-token bound, and every manifest/hash before any adaptation. Both ordinary and recurrent LoRA arms consume the *identical* V2 view, labels, and order. This common measurement correction is not evidence that prompt manipulation improved model capability; only within-V2 paired arms can support an adaptation claim. V0 train/dev bytes remain read-only.

## Fresh benchmark and calibration gate

Use the same predeclared difficulty grid and rejection rules as V1, but new independent PRNG seeds and V2 list wording:

| Family | Candidate 1 | Candidate 2 | Pilot seeds | Development seed | Sealed seed |
| --- | --- | --- | --- | ---: | ---: |
| Expressions | six leaves, alternating chain/balanced | seven leaves, alternating chain/balanced | `721001`, `721002` | `731001` | `741001` |
| Lists | four items, one operation | five items, two operations | `722001`, `722002` | `731002` | `741002` |
| Graphs | seven vertices, edge probability 0.20 | eight vertices, edge probability 0.20 | `723001`, `723002` | `731003` | `741003` |

The accepted expression answer has absolute value at most 9,999. Reject identity list programs and adjacent `REVERSE,REVERSE`; alternate expression shape and graph yes/no labels by accepted index. Draw each split from its own local PRNG, cap attempts at 100 per accepted example, compute canonical IDs from family and latent specification, and verify gold answers with `interpret`. Exclude all V0 train/dev IDs, all 180 V1 pilot IDs, and every already accepted V2 pilot/development/sealed ID. Read V1 pilot splits only for their IDs after authenticating their stage-bound hashes; never use their answers or outputs to choose V2 difficulty. Never open V0 sealed test rows. Freeze exactly 30 examples per pilot candidate, 100 per family in development, and 200 per family in sealed confirmation. Only the generator writes sealed values; downstream code may see only sealed hash/count until the confirmation gate.

Score each of the six pilots with the same F0 source model and V2 scorer, in table order. Candidate 1 is selected if F0 accuracy is strictly between 5% and 95% and invalid rate is at most 10%; otherwise try candidate 2. If neither qualifies, candidate 1 is the fixed fallback for that family, never a best-of-failed-pilots choice. Fewer than two eligible families is terminal `stopped_scientific` before development or sealed generation. Preserve all six raw pilots and their failures.

If at least two families qualify, generate the selected development and sealed splits, then score only development F0. The development gate requires three-family macro accuracy strictly between 5% and 95%, global invalid rate at most 10%, and at least two individually informative families under the same 5%-95% and <=10% rules. The unqualified fallback remains in the three-family macro/global metrics. If the gate fails, stop scientifically before frozen controls or training. Do not add a third difficulty candidate, relax the invalid threshold, extend output length, rewrite prompts, or rescore after observing V2 outputs.

If development exists, export a blinded ten-answer-per-family diagnostic sample with an editable `good/bad` sheet. GPT-6 Luna may grade a separate copy as a model reviewer; retain the blank human copy for spot-checks. Bind both copies to raw answer hashes and label the reviewer identity. Disagreements diagnose scorer or task failures but cannot change the frozen V2 gate or candidate choice.

Calibration has at most 7,200 CUDA-active seconds and a separate 7,200-second wall watchdog; the whole V2 run has at most 86,400 CUDA-active seconds and 20 GiB new artifacts. Persist phase-start markers, failed attempts, exact completed CUDA duration, peak memory, partial raw scores, stage hashes, and terminal status. An interrupted open GPU phase has unknown exact CUDA duration and cannot be retried under the same run ID. Resource or environment failure is never relabeled as a scientific no-go. `report` must authenticate all non-sealed available raw evidence without opening sealed content after a stop.

## Conditional recurrence comparison

Only a passed V2 development gate permits F0/F1/F2 frozen controls on the identical 300 development IDs and token budget. F0 uses ordinary depth, F1 repeats the shared block, and F2 uses the registered damped substeps; score all with the V2 extractor and report paired answer transitions, invalidity, graph explanation contradictions, latency, VRAM, and layer evaluations. FineWeb retention uses the registered 40 chunks and verified source/hashes. Frozen controls do not tune block choice or model parameters.

Only after that, run the paired V2 training comparison using the V2 train view: A1 ordinary LoRA, A2 shared-block recurrent LoRA, and A3 ordinary LoRA matched to A2 training time. Preserve V1's target modules (`q_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), rank 8, alpha 16, dropout zero, AdamW hyperparameters, answer-only labels, seed 47, 1,000 A1/A2 updates, identical initialization/example stream, and pilot-derived A3 steps. V2 discovery requires A2 to beat F0, A1, and A3 by at least two macro percentage points, no more than a two-point family regression versus F0 or A1, A3/A2 realized training time in `[0.90,1.10]`, and both FineWeb PPL-ratio point estimate and paired 10,000-replicate bootstrap 97.5th-percentile bound at most 1.02. A stop ends the study; no best-checkpoint or prompt selection.

Only passing discovery permits independent seed-48 confirmation and opening the V2 sealed 600-example split. The V1 two-seed paired-bootstrap, family-regression, retention, and timing decision rules apply to the V2 paired data. The final report must distinguish final-answer correctness from graph explanation validity, compare inference cost as well as quality, disclose all stops/deviations, and state that post-training gain is unsupported whenever a prior gate failed. No V2 calibration result alone supports a recurrence-efficacy claim.

## Verification and stop discipline

Before the first V2 GPU output: freeze and hash the scorer, exact list template, generator, config, and tests; audit V1 examples only as development fixtures. Tests must cover Markdown/bare graph answers, endpoint mismatch, conflicting leading/terminal cues, incidental yes/no, missing EOS, false stated graph paths, repeated list operations/definition order, unused-operation exclusion, train-view ID/answer/order preservation, cross-split exclusion, 512-token bounds, signature drift, sealed-access prohibition, hash tampering, incomplete score files, and CUDA budget/interruption behavior. Run focused and full suites, Ruff, compileall, and independent code review before production inference.

Stage progression is `prepare -> calibrate -> frozen -> discover -> confirm -> report`; `report` is also allowed after a terminal stop. Later stages may be implemented after a passing calibration to avoid coding/training expense on another invalid benchmark. A terminal scientific, resource, or environment decision never auto-activates another research direction. Push verified milestones on `exp`; do not merge to `main` automatically. Preserve unrelated untracked review notes and all V0/V1 evidence.
