# Recurrence answer audit and qualification design

Date: 2026-09-27. Status: authorized by the user's request to carry out the previously recommended answer audit and evaluator repair. This document governs a diagnostic phase; a fresh benchmark gets its own frozen protocol before model output.

## Purpose and boundary

Determine whether V2's development stop was dominated by answer extraction failures, genuine wrong answers, or both. The signed V2 decision and all V0/V1/V2 raw artifacts remain immutable. This audit is retrospective diagnosis, never a replacement V2 score or permission to train under V2. The ultimate test remains a paired recurrence versus ordinary LoRA comparison on a fresh, valid benchmark.

The existing list scorer accepts `Final answer`, `Final result`, and `Answer` cues, whereas 59 of 68 rejected list development completions contain a `Final list` heading. That observed mismatch motivates the audit, but it does not by itself show those vectors are correct or unambiguous. Expression and graph rejects and accepted-answer errors must also be checked.

## Authenticated input and selection

Use only V2 run `18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d` development split and F0 score files, after checking their raw SHA-256, counts, ordered IDs, and binding to the signed `calibrate.json` and `report.json`. Do not open any sealed split or V0 test rows. Preserve the source files byte-for-byte.

The initial diagnostic set is all 93 V2 `invalid_format` development answers plus exactly ten accepted answers (`correct` or `wrong_value`) from each family. Select accepted IDs by sorting within family and sampling with `random.Random(927001)`; record the selected ordered IDs and source hashes in a manifest. If the initial audit leaves the family classification near a gate boundary, expand to all 300 development answers before drawing a conclusion. The reviewer packet contains only a random audit ID, family, question, complete response, and stop reason. It omits gold, latent answer, V2 parsed value, correctness, reason code, and whether the row was selected as invalid or accepted.

## Blind answer audit

Assign each family packet to a distinct GPT-6 Luna reviewer. The reviewer records one row for every audit ID: `status` in `conclusive`, `ambiguous`, `missing`, `truncated`; `claimed_value` as a JSON integer, list of integers, `yes`/`no`, or null; a short evidence span; and a one-sentence reason when nonconclusive. Each graph review also records `explanation_status` as `valid`, `invalid`, or `indeterminate` with a short note; non-graph rows use `not_applicable`. Explanation validity judges whether any offered rationale supports the stated result, independently of whether that result matches the hidden reference. Use the asserted final result, not an intermediate calculation. Explicitly conflicting final results are ambiguous. A length-capped or otherwise non-EOS response is truncated regardless of a plausible partial answer. The reviewer receives no reference answer or scorer fields. The review files identify the reviewer as a model, never a human.

After locking the blind reviews, reconcile each audit ID to its authenticated V2 row. Verify claimed values against the existing deterministic `interpret` result. Preserve disagreements and inspect them separately; a second Luna reviewer or the parent adjudicates only those records without overwriting the first review. Report counts by family for correct, wrong, ambiguous, missing, and truncated; distinguish final-answer accuracy from graph explanation validity. Compute exploratory bounds for unaudited accepted rows, and state when a conclusion requires expanding to all 300. The blank V2 human review sheet stays available for spot checks and remains unfilled by model reviewers.

## Evaluator qualification, conditional on audit

If the audit identifies a bounded, clear grammar gap, implement a V3-only deterministic scorer. It may recognize explicit final list headings and light Markdown or math wrapping, but must reject conflicting final values, malformed vectors, wrong vector length, missing EOS, and answers inferred from intermediate calculations. Candidate extraction must not consult gold or the interpreter. Build tests first from synthetic adversarial responses and a fixed audit corpus. Preserve V1/V2 scorer files and every signed V2 stage. A Luna reviewer checks code and tests independently.

Any exploratory rescoring of V2 is labeled diagnostic. Before new F0 output, register fresh disjoint seeds, fixed task difficulty, the exact scorer hash, prompt and token budgets, stop rules, resource caps, and a fresh run identity in a separate V3 plan. Validate scorer agreement with blinded audit labels on held-out examples and require at least two informative task families on fresh F0 data. Only a passed fresh development gate can activate frozen recurrence controls and the paired LoRA study. If audit or fresh qualification fails, report the stop and make no efficacy claim.

## Safety and cost

The audit and scorer development use saved responses and CPU checks. No GPU inference runs during this phase. Keep the original V2 artifact root read-only and write new diagnostic outputs under `artifacts/recurrence-answer-audit-v2/`. Use exclusive creation and content hashes, retaining all raw disagreements. Work on `exp`, preserve unrelated untracked Task 14 notes, commit reviewed source/documentation milestones, and push the branch.
