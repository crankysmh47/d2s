# Pretrained recurrence V1: answer scoring and benchmark repair

Date: 2026-09-26. User authorized continuing the evaluator, benchmark, frozen
controls, and conditional LoRA comparison. This is a new study. V0 run
`129d39646933` remains stopped and immutable.

## Question and evidence

Question: can one shared extra pass through pretrained Qwen3 block `[12,16)`,
adapted by rank-8 LoRA, improve verified answer accuracy over ordinary LoRA and
an ordinary-LoRA training-time control while preserving general-language quality?
This evaluates capability at added inference cost, not a sparse or MoE footprint.

V0 F0 scored 4/600 under a 32-token exact-format evaluator. A separate
256-token, 30-development-example check matched all original 32-token prefixes;
24 responses reached EOS, six list responses capped. GPT-6 Luna judged 18/30
final answers correct (10 expression, eight graph, zero list); the V0 strict
parser accepted 0/30. Luna is an audit label, not the V1 primary scorer. The
30 examples are reused development data and cannot serve as a new test.

## Choice among scoring approaches

The primary scorer is a deterministic, conservative final-answer extractor.
This permits explanations while preserving exact mathematical verification and
reproducibility. A Luna or human judge may audit disagreement but cannot decide
the primary gate. A formatting LoRA before the capability study would introduce
another training intervention and obscure the recurrence comparison, so it is
not part of V1. The grammar below is frozen before V1 model generation.

### Final-answer grammar

Implement `score_completion(example, completion_text, stop_reason)` in a new
`answer_v1.py`, returning parsed value, correctness, reason code, evidence span,
and separate explanation assessment. Never change V0 `parse_answer` or its raw
scores. First require nonempty completion and normal EOS. Length cap, partial
generation, decode failure, and execution failure are ungradable/incorrect.

- Expression: use only a line beginning with optional Markdown heading and
  `Final answer`, or the exact line-anchored phrase `So the final answer is`,
  then within the next 128 characters a `\boxed{signed integer}`.
  Allow surrounding display/math markup. Any two distinct final-cue values make
  the response ambiguous. Integers in calculations or uncued boxes do not count.
- List: use only a line beginning with optional Markdown heading and `Final
  answer`, `Final result`, or `Answer`, optionally followed by a colon. Read the
  first complete bracketed integer vector in that cue payload or its next two
  lines, including an optional `\boxed{...}` wrapper. Preserve all signed and
  multi-digit values before verification: an `11` is a wrong value, not a parse
  failure. Require the expected list length and syntactically complete vector.
  Multiple distinct final-cue vectors are ambiguous. Earlier initial or
  intermediate lists cannot be used. A malformed or incomplete final-cue
  vector invalidates the completion even if an earlier final cue was complete;
  within each cue's payload, only its first complete vector is a candidate.
- Graph: accept a standalone leading `Yes.`/`No.` at the beginning of the
  completion, a terminal `Answer: yes/no`, or a conclusive sentence of the form
  `there is [no] [direct or indirect] path from X to Y` with X/Y matching the
  queried endpoints. If distinct explicit conclusions appear, mark ambiguous.
  Ignore incidental yes/no inside edge discussions or negated mentions.

The output `correct` is true only when the unique extracted answer equals
`interpret(example.latent_spec, example.family)`. A well-formed wrong answer is
`wrong_value`; no unique extracted answer is `invalid_format` or `ambiguous`.
All ungradable cases count toward the gate's invalid rate, including length cap.
Do not use the reference answer to search, pick, or disambiguate spans. Derive
the required list length from the latent input length, which list operations
preserve, rather than consulting the computed target during extraction.

Explanation assessment is an independent, non-gating field: `supported`,
`contradicted`, `unverified`, or `none`. For graph answers, if a path is asserted
as evidence, verify every stated edge and endpoints against the prompt's latent
graph. Accept ASCII `->` or Unicode `→` arrows in a bounded, possibly
line-wrapped path claim. A false claimed path is `contradicted` even if the
final yes/no is right. For non-EOS or absent completions, explanation is `none`
because the text is incomplete; for EOS completions it is assessed even when
the final answer is invalid.
Other explanation styles are `unverified` unless a small deterministic verifier
can prove them without guessing. Preserve a short reason and cited text span.

The existing 30 responses are an offline audit fixture: the proposed grammar
parses 23 and rejects seven, yielding 18 correct and five wrong among parsed
responses against Luna's labels. That agreement is diagnostic only. Test
adversarial intermediate values, conflicting conclusions, malformed vectors,
out-of-range list values, missing EOS, and wrong graph path explanations. Freeze
the implementation hash before generating V1 calibration outputs; later code
changes require a new diagnostic root and invalidate the old gate.

Scorer freeze before V1 GPU generation: `answer_v1.py` SHA256
`2f138e6fe106c81dd03712d0a49a3a916cec2596a4acfa8f4e111f6463bab719`.
The 30-row offline audit yielded 23 parsed, 18 correct, five wrong, and seven
ungradable. The immutable V1 run identity must bind this hash.

## Fresh benchmark and calibration

Keep the model, tokenizer, chat template (`enable_thinking=False`), greedy
cacheless BF16/eager inference, fixed block, 512 total sequence tokens, and
256 generated tokens from the completed review diagnostic. Reuse V0's 8,000
training examples read-only with recorded hashes; no new prompts, few-shot
examples, teacher answers, or training data selection. New pilot, development,
and sealed confirmation examples use independent seeds and the same prompt
templates/latent interpreter, with only task difficulty varied.

Generate exactly two predeclared difficulty candidates per family:

| Family | Candidate 1 | Candidate 2 |
|---|---|---|
| Expressions | six leaves, alternating chain/balanced | seven leaves, alternating chain/balanced |
| Lists | four items, one operation | five items, two operations |
| Graphs | seven vertices, edge probability 0.20 | eight vertices, edge probability 0.20 |

Pilot 30 new examples per candidate, 180 maximum, with independent PRNG streams,
balanced yes/no graph labels, V0 rejection rules, canonical IDs, and no overlap
within pilot/development/confirmation or against reused V0 train/development IDs.
The six pilot seeds are expressions `621001,621002`, lists `622001,622002`,
graphs `623001,623002`. The three development seeds are `631001,631002,631003`
in family order; sealed confirmation uses `641001,641002,641003`. Generation
rejects identity list programs and adjacent `REVERSE,REVERSE`, alternates
expression chain/balanced and graph yes/no labels by accepted index, and caps
absolute expression answers at 9,999.
Candidates are evaluated in table order. Choose the first in each family with
F0 accuracy strictly between 5% and 95% and ungradable rate <=10%; if the
first fails, try its second. Preserve every pilot score and rejection. There
is no third condition, cap change, prompt edit, or threshold retuning. If fewer
than two families have an eligible candidate, record a scientific stop.
If exactly two families have eligible candidates, assign the third family
candidate 1 as a prespecified fallback for the development/confirmation splits;
mark it ineligible in the pilot table and retain it in all three-family macro
metrics. Do not select a fallback based on which failed pilot scored better.

Once selected, generate fresh independent development (100/family) and sealed
confirmation (200/family) splits at the selected difficulty. Persist source,
generator, tokenizer, task-choice, split IDs, code/config SHA256, and split
file hashes. Only the generator writes the sealed split; evaluation/report code
must not open its contents before all discovery/confirmation preconditions pass.
Do not report pilot accuracy as development accuracy. The development F0 gate
again requires macro accuracy strictly 5%-95%, overall ungradable rate <=10%,
and at least two families individually with accuracy strictly 5%-95% and
ungradable rate <=10%. A failed gate is terminal for V1 training.

Calibration gets at most two CUDA-active hours, including failed attempts. The
whole V1 study gets at most 24 CUDA-active hours and 20 GiB new artifacts.
Persist failed attempts and elapsed CUDA time. No download is planned; use the
pinned local Qwen3-1.7B snapshot at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` after verifying source hashes.

## Frozen controls and conditional LoRA

After passing development calibration, score all 300 development examples
under F0, F1, and F2 using the identical extractor and token budget. F0 is
ordinary depth, F1 repeats the shared block, and F2 uses the previously
registered two damped substeps. Persist paired records, scores, inference time,
memory, and layer-execution counts. V0 frozen FineWeb retention scores may be
referenced only after verifying the source model, chunk IDs, raw hashes, and
unchanged frozen implementation. Otherwise rerun the 40 retention chunks.
Frozen controls do not select a block or update parameters.

For discovery, retain the registered LoRA placement, rank 8, alpha 16,
dropout 0, AdamW settings, 8,000-example stream, 1,000 updates for A1/A2, seed
47, and pilot-based A3 step count. A1 executes the block once, A2 twice with
shared adapters, and A3 executes once with training time matched to A2.
Trainable tensors, examples, answer-only labels, and initialization are paired.
Pilot and realized time ratios, checkpoint provenance, CUDA and artifact budgets
are checked before any result is promoted. Evaluate all adapted arms on the
frozen development set with the V1 scorer and retained FineWeb chunks.

Discovery promotion uses V0's prespecified thresholds: A2 must exceed F0, A1,
and A3 by at least two macro percentage points, avoid any >2-point family
regression versus F0 or A1, and pass both FineWeb PPL-ratio point and paired
97.5th-percentile bounds <=1.02. A3/A2 realized training time must be within
[0.90,1.10]. Failure is a reported no-go, uncertain retention, or timing stop.
Only a passing discovery permits seed-48 confirmation and sealed evaluation.
Confirmation uses the registered paired two-seed bootstrap, family-regression,
retention, and timing gates, applied to V1's 600-example sealed split. No
best-checkpoint or prompt selection is permitted. Report final quality, added
inference compute, VRAM, latency, and limitations, even on a terminal stop.

## Implementation boundaries

Create V1 config/data/scorer/stage runner modules and V1-only artifacts. Reuse
existing `Example`, `interpret`, `render_example`, `RecurrentQwen3`, LoRA,
retention, bootstrap and checkpoint components after checking their interfaces.
Do not loosen registered V0 constants to make V1 run. Its report hashes and
stage records remain immutable. The V1 runner must have explicit stage commands
and durable gate manifests; passing persisted gates advance automatically.

Critical adverse cases: natural answer with intermediate gold value but wrong
final value; contradictory yes/no; incomplete list at cap; changed data or
parser after calibration; sealed split access before confirmation; model or
tokenizer hash drift; GPU budget exhaustion; training checkpoint reload with
different adapter placement. Each has a regression or validation check.
