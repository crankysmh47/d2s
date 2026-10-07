# Verified-distillation V1: calibration qualification stop

## Outcome

The candidate is **stopped scientifically before pilot, corpus generation or QLoRA training**. Live loading and generation succeeded, and the registered output-length criterion passed. The frozen final-answer extractor failed the independent blind Luna extraction gate. This is a measurement-contract failure, not evidence of poor mathematical accuracy or a failed distillation hypothesis.

Practical and teacher-specific improvement are **not evaluated**. No adapter was trained. Do not infer model-quality, recurrence or MoE-footprint benefits from this run.

## Verified result

Execution `163c6c24c2fd409ade943ba2c9a0bef771d03c38338c9281851e701871943c11` registered and launched once on `9786f61`, 2026-10-07 at22:05 UTC. Both models passed actual CUDA-only placement and generated responses. The256-output/768-total-token budget failed length selection with51/60 student EOS and50/60 teacher EOS. The selected512/1024 budget reached59/60 student EOS and60/60 teacher EOS, with no prompt overflow. The third candidate was not run.

All120 selected-budget responses received gold-free, arm-blind Luna extraction decisions in three disjoint40-row batches. Every row/session/evidence quote and the original packet digest were validated. One evidence quote was corrected in a new file; the original remains preserved and all values/statuses remained unchanged. Luna marked119 responses extractable and one capped response invalid. These are **model reviews of extraction**, not human reviews or correctness scores.

| Role | Parser-invalid responses | Parser/reviewer disagreements | Required maximum |
| --- | ---: | ---: | --- |
| Student | 48/60 | 47/60 | 6 invalid; 3 disagreements |
| Teacher | 45/60 | 45/60 | 6 invalid; 3 disagreements |

An independent Luna audit recomputed the same failure from the authenticated journals and reviews. Extractor reasons across120 responses were89 `invalid_final_value`,3 `missing_final_cue`,1 `not_eos` and27 `ok`. Common boundaries include display-math delimiters around boxed scalars, punctuation outside Markdown emphasis, and literal prompt-template cues preceding the final value. This identifies a strong formatting contribution; no mathematical accuracy was computed.

## Evidence and costs

- [Terminal gate receipt](../artifacts/distillation-v1-executions/163c6c24c2fd409ade943ba2c9a0bef771d03c38338c9281851e701871943c11/calibration/review/extraction-gate.json): SHA256 `33b684487172de983f097adca02e626f2126a7d187a5d68462527a021cdb10fd`.
- [Original public packet](../artifacts/distillation-v1-executions/163c6c24c2fd409ade943ba2c9a0bef771d03c38338c9281851e701871943c11/calibration/review/packets/calibration.jsonl): SHA256 `b15edd1876d5320731e81841ae97eb6dbd44d73c41720ef292433e3086e4835a`.
- Frozen calibration artifact SHA256 `12b960c7d8d5230196f22e9abfa660a9039db93b238065b5cbbfae380fed3368`; parser SHA256 `3a7087d2496dd79eed2d9841df5a18d842551bffda2c5352c1535d6755146378`. Raw/reviewer hashes and sessions are bound in the receipt.
- Independent audit retained at `.superpowers/sdd/2026-10-07-tokenizer-contract-recovery/calibration-gate-audit.md`, SHA256 `aba35667504afd13ef0ba08c810fd0701f9e0ccd7b86aab3b211f00826b7127b`.
- Cumulative charged CUDA2,572 seconds; downloads8,066,718,062 bytes; settled active time1,305 seconds, including91 inherited, leaving171,495 of172,800 seconds. Costs were carried, not reset. At22:36:23UTC, measured artifacts were2,813,832 bytes including2,153,903 inherited bytes and reviewer artifacts, below the20GiB cap.
- Both recorded supervisor/stage identities are absent. The preserved runner snapshot is `awaiting_review`; the separate receipt is authoritative for this **`stopped_scientific`** gate. No further compute is authorized for the candidate.

Engineering evidence before launch: root CPU1,335 passed/7 skipped/3 CUDA deselected, pinned offline lint/compile/diff, real cached-tokenizer integration without skips, independent task/release reviews, and detached native proof. These did not establish scientific qualification. The pre-existing local snapshot authentication/loading check-use window remains documented.

## Next permitted action

Preserve this candidate and all prior stops. Do not edit its parser, relabel reviews to pass, reopen it, or train. A separately authorized evaluator/output-format repair can use these public responses as regression evidence, retain ambiguity/truncation/conflicting-answer rejection, and then qualify a **new** frozen protocol on fresh calibration questions. Any new execution must carry all prior costs. This is a recommendation requiring new authorization, not an automatic retry or promised performance gain.
