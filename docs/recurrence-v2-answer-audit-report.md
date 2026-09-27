# V2 development-answer audit (diagnostic, not a V2 rescore)

Date: 2026-09-27. Branch: `exp`. Registered V2 run: `18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d`. Audit ID: `46ecaeb5c1135cb72211f8379ee47e6aa1b8afd07aa04fd1e07a3d0c752e0540`.

## Decision

V2 remains signed `stopped_scientific`; no frozen recurrence control or LoRA training has run. The audit found a real, bounded answer-extraction problem, especially the unrecognized `Final list` heading, **and** substantive wrong list answers. The evidence supports one separately frozen V3 qualification attempt, not retrospective promotion of V2 or an efficacy claim. Keep the same output allowance and objective tasks; validate the repaired scorer before new model output, then apply fresh pilot/development gates. If V3 fails, stop rather than retune this result.

## Provenance and method

The exporter authenticated the registered V2 prepare/calibrate/report stage records against their independently audited hashes and checked the development split, sidecar, score bytes, counts, and ordered IDs. It never read V0 test or V2 sealed rows. The blind packets contained only audit ID, family, question, response, and stop reason. Distinct GPT-6 Luna reviewers extracted final claims without gold, V2 scores, or selection reasons. Reconciliation compared the locked claims with the deterministic `interpret` result, retaining every disagreement. A second blind Luna reviewer independently inspected four graph disagreements; model review was never labeled human review. The original blank human-review file remains unfilled.

The initial set was all 93 V2 `invalid_format` development rows plus ten seeded accepted rows per family: 16 expressions, 78 lists, and 29 graphs, 123 total. This is **outcome-stratified**, not a representative random 123/300 sample. The 30 accepted controls had no reviewer/V2 disagreement (28 correct and two wrong, matching V2), but the other 177 accepted rows were not independently reviewed. Therefore no unbiased revised 300-row accuracy or retrospective pass/fail gate is asserted.

Authenticated development input SHA-256: split `36262998b3db1e28a46e3559105496d1b80c3a1e8f4914fce9c626fe134853f0`; sidecar `589f3e7723f67aaba3b2261941f4b77b92a40dff5f68c509b02e46f3204e4830`; score `1ba7f29946ebdc4bc03267f10641d5837ca00324e2cfe97e0f9c9123942189ab`. Blind manifest SHA-256: `1a405370345275d78c6031a6110239b2c6891ef7e85e67fd50cc2bfd5c2b948b`. Final diagnostic SHA-256: `9f1c32a1d664742d511d12a04650a374a467baca163f9424368aaa7ef44ae30b` at `artifacts/recurrence-answer-audit-v2/<audit-id>/diagnostic.json`. Final first-pass review SHA-256: expressions `8912c72072948bdd67d5e8e138e4b09ca0fb1d38717a800c7abb6f2274601446`, lists `0aef9d242e4d59e3a8a05d22462284a0f9e55e74d20719871b59f1691ee94c93`, graphs `5f6cb73faed448b1ce71dd227215dd3a44ba6d291c8aee8a83f11df80375bd21`. The independent four-row graph second pass is `e47ace42fc48bbbdffb379cc3ab2a4558d6a9c604b521c59c427d6d197512ca7`.

## Findings

| Family | Selected | Correct | Wrong | Ambiguous | Truncated | Correct V2-invalid final answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expressions | 16 | 16 | 0 | 0 | 0 | 6 |
| Lists | 78 | 58 | 20 | 0 | 0 | 48 |
| Graphs, raw validator | 29 | 14 | 4 | 4 | 7 | 6 |
| Graphs, after separate second-pass adjudication | 29 | 14 | 5 | 3 | 7 | 6 |

Across the 93 V2-invalid answers, 60 have a conclusive, mathematically correct final value despite rejection by V2 (six expression, 48 list, six graph). Twenty-three are wrong after the independent graph adjudication; three graphs contain contradictory final Yes/No claims and seven graph completions are length-capped. All list-invalid completions in this set reached EOS, and the common `Final list` heading explains many parser misses; however, 20 of 68 list-invalid answers state the wrong vector. This is not a mere formatting-only failure.

One graph reviewer cited a final `Yes` for F-to-A using a claimed F→C step, although the prompt supplies C→F and no outgoing edge from F. The deterministic reconciler conservatively flagged that citation because its final-cue grammar did not extract the wrapped Yes. The independent blind reviewer agreed the final Yes was unambiguous and that the path explanation was invalid. The report's adjudicated row counts move that one row from ambiguous to wrong; the raw `diagnostic.json` and the first reviewer labels are not edited. Three other graph responses explicitly open with Yes and end with No, so they remain ambiguous even where the final explanation suggests No. The first graph reviewer marked eight explanations invalid, seven valid, and 14 indeterminate; these labels concern explanation support and are separate from final-answer correctness.

For intuition only, **if** the 177 unreviewed accepted V2 labels all hold, adding the 60 confirmed parser misses to V2's 198/300 correct would yield 258/300 (86%) final-answer accuracy: expressions 100/100, lists 80/100, graphs 78/100. After the one graph adjudication, three ambiguous plus seven truncated graph answers would remain invalid (10/100), and global invalid would be 10/300. That counterfactual sits exactly on the graph invalid-rate gate and is **not** a V2 gate result: the accepted-label premise is unverified, the audit used development outputs to diagnose the scorer, and V2's evaluator and decision were frozen before those outputs.

Two earlier exclusive reconciliations are retained byte-for-byte as `diagnostic-superseded-1.json` (SHA-256 `64f67f254328958fffe35589eddf472c6a268d52da914d0db0f078501e115194`) and `diagnostic-superseded-2.json` (`0adc78526bb68baa1c7a9c7c3e329ff4c2fdb5f952b141b869d493c83862a5b4`). They exposed audit-validator bugs: demanding a final cue inside a value-only evidence span, then failing on math/Markdown wrappers. Both were repaired with adversarial RED/GREEN tests and independently reviewed; neither superseded count is used above. The final 56-test focused audit suite passes. Task 2's full CPU suite before the subsequent focused repair rounds passed 920, with four skipped and three CUDA-marked tests deselected; a new full-suite run is required before freezing V3 code.

## Recommendation and stop rules

Define a V3-only deterministic scorer from this bounded grammar: explicit final list/result/answer cues, including observed `Final list` and task-specific `So the final list ... is`; scalar and vector math/Markdown wrappers; graph natural conclusions and unique terminal Yes/No. It must reject conflicting final values, incomplete/malformed vectors, missing EOS, and wrong values. It must not consult gold during extraction. Pin scorer tests and hash **before** fresh pilot output; validate against a held-out blind audit slice and adversarial cases. Preserve V1/V2 scorer files and all prior stage artifacts.

Register fresh disjoint pilot/development/sealed seeds, output budget, task difficulties, source hashes, resource caps, and stop rules under a new V3 run identity. The 256-output/512-total-token allowance stays frozen. Require at least two informative families and the existing macro/invalid gates on fresh V3 development F0. Only then implement/run frozen recurrence controls and the paired ordinary/recurrent/compute-matched LoRA study with retention and inference-cost gates. Do not use the sealed split during qualification. If the fresh gate fails, report a scientific stop; do not select a new difficulty, widen grammar, or train on that stopped run.
