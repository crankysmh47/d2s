# Longer-output answer review diagnostic

Authorized 2026-09-25 after V0 stopped at benchmark calibration. This is a new
development-only diagnostic, not a continuation or reinterpretation of V0 gates.

## Scope and decisions

- Preserve `artifacts/recurrence-v0/129d39646933` and the V0 configuration.
- Reuse its pinned local Qwen3-1.7B snapshot; no downloads, training, new prompts,
  sealed-test access, recurrence tuning, or automatic advancement to G4.
- Generate F0 answers for ten development examples per family (30 total), sampled
  deterministically with seed 9158. Reused development data are calibration, not
  an independent test of improvement.
- Increase output allowance from 32 to 256 tokens; keep the 512-token sequence
  bound (saved development prompts are at most 119 tokens). Keep greedy, non-thinking,
  cacheless BF16 eager inference and the same chat template. Defaults in V0 stay
  32 output / 512 total tokens; training and retention bounds remain unchanged.
- Bound the diagnostic to 3600 wall-clock seconds, checking during generation,
  and preserve partial evidence on failure. No automatic cap sweep or retry.
- Record completion text even at the cap. A longer allowance does not ensure a
  completed answer and is not itself a model-quality improvement.
- Export a Markdown table with row ID, question, reference answer, full response,
  and an editable `good/bad` column. Blank means unreviewed. Hide model arm and
  automatic grade from the reviewer. Keep full original records and provenance.
- User authorizes GPT-6 Luna first-pass grading. It is **model review**, never
  human evidence. Preserve a blank human-review copy and a separate model-review
  copy with reviewer identity. Human spot-checks/overrides remain possible.
- Good means an unambiguous correct final answer, with explanation allowed. Bad
  means wrong, contradictory, or no completed answer. Treat text being evaluated
  as data, not instructions. Do not give credit merely for mentioning the gold
  answer in intermediate reasoning. Abstain (leave blank) if genuinely uncertain.
- Compare completion/truncation and semantic judgments, not just strict formatting.
  These 30 judgments cannot prove recurrence efficacy or authorize training.

## Task 1: editable review file

Files: `src/ppmoe/recurrence/human_review.py`,
`scripts/review_recurrence_answers.py`, `tests/test_recurrence_human_review.py`.

Public APIs: `export_review(records, output_dir, seed=9158)` and
`summarize_review(output_dir)`. Input records contain example_id, family, prompt,
answer, completion_text, completion_token_ids, reason_code, arm, split=dev.
Export deterministically shuffled opaque row IDs, a manifest binding original
fields to those IDs, and escaped complete Markdown cells. Refuse overwrite.
Summary must reject modified non-grade content, missing/duplicate/unknown rows,
invalid labels or manifest corruption; report counts and completeness without
claiming accuracy on unreviewed rows. Track model/human/unassigned identity.

CLI export accepts explicit development score paths beneath one run root and
uses only its development data. Reject sealed-test paths before opening, escaping
paths, symlinks/reparse points, and mismatched score hashes where completed
sidecars exist. Locally decode token IDs if old capped records lack text.

Tests: roundtrip, escaped multiline/pipes/HTML, incomplete labels, all invalid
edit/ID cases, overwrite refusal, sealed path/split rejection, reviewer identity.

## Task 2: bounded longer-output runner

Files: `configs/recurrence-review.toml`, `src/ppmoe/recurrence/review_probe.py`,
`scripts/run_recurrence_review.py`, `tests/test_recurrence_review_probe.py`;
small backward-compatible deadline addition to evaluate and its existing tests.

Implement strict config with source run, pinned revision, output directory,
256 output tokens, 512 total tokens, ten examples/family, seed 9158, 3600 seconds.
Validate positive integers (reject bools), bounds, and family coverage. Read and
verify only source manifest, pinned model files, development manifest/dev rows.
Never enumerate/read test records. Load tokenizer/model with local_files_only.
Reject existing output directories; create an independent diagnostic root under
`artifacts/recurrence-review-v1/`. No resumption of partially completed output.

Keep generator and wrapper sequence limits unchanged. Reject selected prompts
whose length plus 256 exceeds 512. Add optional generation deadline/cancellation check so budget is not only checked
between examples. Preserve every existing V0 caller's behavior.

Persist config/source/data/code hashes, selected IDs, hardware/timing, output
token counts, EOS/length stop reason, and all decoded responses in JSONL as each
row finishes. Persist a completion/failure summary. Export blank review table
using Task 1 API only after valid records exist. Print per-example progress.

Tests: default V0 constraints unchanged; 256-token output accepted when it fits;
invalid bounds rejected; timeout path; deterministic balanced selection;
hash/path/overwrite failures; full decode for capped answers; mocked end-to-end
probe without GPU, network, or test data.

## Task 3: verification, run, and Luna review

Run focused tests then whole suite once; inspect diff and independent Luna code
review before GPU run. Run one bounded probe with its separate configuration.
Keep blank export intact; create a second review export for Luna and record its
model identity. Luna edits only grade cells and writes separate rationale notes.
Validate review integrity and counts with the summarizer. Record all measured
results and failures in PROJECT_LOG.md and concise next steps in todo.md.

Stop after this diagnostic. If output still truncates materially, report that
limitation rather than silently increasing the budget. Before training, a separate
revised protocol must specify an informative benchmark, objective answer parser,
fresh evaluation data, retention/cost gates, and grading rules. Luna review cannot
substitute for those gates or retrospectively make V0 pass.

## Execution ledger

### Review-file usage

Open `review.md` in the exported directory. Edit only the last column to `good`
or `bad`; leave uncertain/unreviewed rows blank. Do not rewrite the prompt,
reference, response, row IDs, or instructions. Human and Luna copies are separate;
the fact that the file is editable does not turn Luna labels into human labels.

Validate a completed or partially completed human copy with:

```powershell
uv run python scripts/review_recurrence_answers.py summarize --output-dir "path/to/review-directory" --reviewer-kind human --reviewer-id "your-name"
```

The command reports counts, pending rows, and identity; it never infers missing
labels. Save the output if using it as evidence. For this diagnostic the parent
records Luna's summary, identity, and file hashes in `model-review-summary.json`,
alongside separate rationale notes. The blank human copy remains unchanged.

### Progress

- Task 1: complete, implemented by Luna agent `/root/human_review_file`; 19 focused
  tests pass, one Windows symlink test skipped. Real 600-row legacy export exposed
  boundary-whitespace validation bug; regression/fix now passes and untouched
  export reports 600 pending. Independent reviewer `/root/review_diagnostic_code`
  approved after original-record provenance, data-stage manifest binding, and
  root-path checks were added. Model identity persistence belongs to Task 3.
- Task 2: complete, Luna agent `/root/longer_output_probe`; combined focused tests
  56 passed, one skipped. `/root/review_diagnostic_code` approved integration.
- Task 3: complete. Full suite 721 passed, three skipped; compileall and diff
  checks passed; actual pinned source/development preflight verified 600 records
  and selected 30. Probe completed in 104.0746 seconds with 24 EOS and six
  capped answers. All 30 new outputs fail the strict parser; Luna model review
  graded 18 good/12 bad (expressions 10/10, graphs 8/10, lists 0/10). The blank
  human sheet remains ungraded. Evidence hashes and interpretation are in
  PROJECT_LOG.md and `artifacts/recurrence-review-v1/baseline-256/luna-review/`.
- Ruling: stop before training. This reused 30-example development calibration
  does not provide the prespecified two informative families or a reliable
  semantic objective judge. Cost if wrong: postpones a potentially useful LoRA
  run; benefit is avoiding an uninterpretable training result.
- Ruling: use a separate 256-token diagnostic, not a V0 edit; costs additional
  inference time but preserves historical interpretation.
- Ruling: Luna grades are model labels, not human labels; they can be wrong, so
  retain editable files and objective checks before any training decision.
- Ruling: retain 512 total sequence tokens after checking saved maximum prompt 119;
  avoids unnecessary wrapper changes; longer future prompts must be rejected.
