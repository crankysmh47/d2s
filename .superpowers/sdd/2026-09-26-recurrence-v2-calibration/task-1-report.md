# Task 1: V2 answer scorer report

## Scope and files

Implemented the V2 answer scorer and its tests only:

- `src/ppmoe/recurrence/answer_v2.py`
- `tests/test_recurrence_answer_v2.py`

`answer_v1.py` and existing V0/V1 artifacts were not changed. The pre-existing untracked task-14 review notes were left untouched.

## TDD record

- Initial RED: `uv run pytest tests/test_recurrence_answer_v2.py -q` produced 18 failures because `ppmoe.recurrence.answer_v2` did not exist. This confirmed the expected missing API failure.
- GREEN iterations tightened candidate collection and Markdown handling. The first implementation run exposed that answer token and path regexes rejected supported decorations; those cases were corrected while keeping candidate recognition bounded to the specified grammar.
- Adversarial RED: `Answer: n**o` was incorrectly normalized as `no`. The test failed on parsed value `no`; Markdown normalization was narrowed to balanced emphasis around words or phrases. The test then passed.
- Focused final run: `uv run pytest tests/test_recurrence_answer_v2.py tests/test_recurrence_answer_v1.py -q` — 40 passed.

## Behavior delivered

- Graph V2 recognizes a standalone first nonblank answer, complete `Answer:`, `Final answer:`, and `So, the answer is:` lines, and path conclusions for exactly the queried endpoints.
- It collects every explicit candidate before consulting `interpret`, rejects conflicting conclusions as `invalid_format`, preserves original cue evidence, and distinguishes a well-formed wrong answer as `wrong_value`.
- It rejects prose-only yes/no mentions, endpoint mismatches, malformed Markdown, non-EOS completions, and invalid negative claims containing the article `a`.
- Graph explanation assessment uses the unchanged V1 deterministic path-edge checker and remains separate from answer correctness. Expression and list families delegate to unchanged V1 scoring.
- The score result retains the seven V1 fields.

## Verification

- `uv run pytest tests/test_recurrence_answer_v2.py tests/test_recurrence_answer_v1.py -q` — 40 passed.
- `uv run pytest -q` — 811 passed, 3 skipped.
- `uv run ruff check src/ppmoe/recurrence/answer_v2.py tests/test_recurrence_answer_v2.py` — passed.
- `uv run python -m compileall -q src/ppmoe/recurrence/answer_v2.py` — passed.
- `git diff --check` — passed before report creation; rerun after staging as part of commit verification.

An earlier full-suite attempt was stopped after Ruff found a local-variable reference introduced during cleanup. The reference was fixed; the final full-suite run above completed successfully on the corrected source.

## Provenance and concerns

SHA256 of `src/ppmoe/recurrence/answer_v2.py`:

`3fe4db3694a2e37d167eeeefe5a524c5d229d0a920b7595e016e44ec8e08b9ee`

No unresolved Task 1 concerns. No GPU generation was run.
