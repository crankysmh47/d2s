# Pretrained recurrence: bounded capability study

Date: 2026-09-08; amended 2026-09-11. Status: selected planning direction with an approved sequential execution contract; implementation and experiments have not started.

## Objective and scope

Determine whether reusing a pretrained transformer block, with inexpensive adaptation, improves held-out answer accuracy beyond ordinary adaptation on the same data. Additional inference computation is allowed and must be measured. This is a new recurrence track in the existing research workspace, not evidence of sparse conversion or reduced inference cost.

The user authorized selecting directions and revising documentation. This revision does not launch downloads, training, or evaluation. When subsequently asked to execute the plan, proceed automatically through passing gates and terminate on the explicit scientific/cost gates; do not request routine design decisions again.

Exactly two linked interventions are selected: fixed frozen block recurrence as a diagnostic baseline, then LoRA adaptation of the same block under one extra pass. They share one model, dataset, evaluator, and experiment funnel. Do not implement adaptive token halting, native-MoE studies, expert merging, sparse conversion, or specialist assembly in this track.

## Rationale

Phase 1A established that random 8x448 partitions lose substantial quality even under greedy hidden-state oracle routing. Phase 1A.5 completed 100K calibration tokens and stopped at its registered selection gate: K4 best registered reconstruction improved pooled local NMSE by 16.66%, K6 by 9.90%, below 30%. No reconstruction rollout ran. The recovery run mixed revisions a8ab03b and 20cd44b without changing scoring mathematics. These facts remain unchanged.

The old "proceed to Phase 1B" result is a historical experiment decision. The new project-level decision is to defer Phase 1B and sparse LoRA recovery while evaluating recurrence. No mathematical impossibility of sparsification has been established.

## Frozen defaults for the new study

- Branch: exp. Preserve main and old artifacts. Package remains ppmoe; new implementation lives in src/ppmoe/recurrence/.
- Model: Qwen/Qwen3-1.7B, the post-trained conventional attention checkpoint, not Qwen3.5 or the Qwen3 base variant. Resolve a full immutable model commit before weight download; record tokenizer and weight file hashes.
- Expected layout: Qwen3ForCausalLM, 28 layers, hidden size 2048, intermediate size 6144, 16 query heads, 8 KV heads, head dimension 128. Assert config and tensors at runtime.
- Precision: BF16 frozen weights and activations; FP32 trainable adapters and optimizer states. No quantization in this protocol.
- Loop block: layers 12,13,14,15 (zero-based; half-open [12,16)). No layer-range search.
- Backend: eager attention, no KV cache, one sequence per forward, maximum 512 total tokens. Recompute the entire prefix for each generated token. This is a correctness-first reference, not an optimized serving implementation.
- Chat template: checkpoint template, enable_thinking=False, no system message, no examples. Same template, prompts, greedy decoding and output cap for all arms. This is a controlled direct-answer setting, not a claim to beat unrestricted thinking-mode Qwen.
- Maximum generated tokens: 32; stop at EOS or limit. Output limit exhaustion is scored as an incorrect answer, never discarded.
- Synthetic exact-answer tasks: integer expressions, list transformations, directed-graph reachability. Exact generators, splits, parsing, and seeds are in plan.md.
- General-language regression set: the existing frozen FineWeb held-out text document IDs, re-tokenized for Qwen3 into a NEW artifact tree, 20,000 valid source tokens. Never reuse Qwen3.5 token IDs. Persist 512-token chunk NLL sums/counts and use 10,000 paired chunk-bootstrap replicates with seed 9157 for every adapted-arm/F0 PPL-ratio interval.
- Rank 8, LoRA alpha 16, dropout 0, target q_proj/v_proj/o_proj/gate_proj/up_proj/down_proj within the four block layers. No k_proj adaptation, no bias/norm/embedding/head updates. Same trainable tensors in all adapted arms.
- AdamW: learning rate 1e-4, betas (0.9,0.95), epsilon 1e-8, weight decay 0, gradient norm clip 1, linear warmup 20 steps then constant rate. Seed 47 discovery, seed 48 conditional confirmation.
- 1,000 optimizer steps for A1/A2, with the explicitly defined A3 time-control exception below; batch of 8 examples via accumulation, one pass through 8,000 training examples for A1/A2. Answer-plus-EOS cross-entropy only; no teacher, RL, hidden matching, answer rationales, or external API labels.
- Total study cap: 24 CUDA-active wall hours, including evaluation and failed attempts, excluding idle/user time and downloads. Maximum six training arms: three arms at two seeds. No automatic budget increase.

## Mathematical intervention

Let G be the original four-layer block, P the prefix, and Q the suffix. The baseline is Q(G(P(x))). G returns the complete residual-stream state, not just an FFN output. All repeated evaluations use the same causal mask and token position indices with no cached state.

Frozen candidates: F0 baseline; F1 naive G(G(h)); F2 two damped substeps z=h+0.5*(G(h)-h), out=z+0.5*(G(z)-z). F2 refines a residual update; it is inspired by numerical integration, not a proof of task-loss descent or a full reproduction of a published system. Run exactly these three conditions.

Adapted candidates: A1 executes G_phi once; A2 executes G_phi twice with shared adapters. A1 and A2 see identical examples and optimizer steps. A3 uses the A1 architecture with extra optimization steps, repeating the same fixed training stream; it controls for the extra training budget spent by A2.

Authoritative A3 rule: measure synchronized training-step wall times t1 and t2 on the same 20 pilot batches after 5 warmup steps for A1 and A2. Set A3_steps=max(1000,ceil(1000*median(t2)/median(t1))). Freeze that integer before production training. This is the sole exception to the 1,000-step/one-epoch default. It matches a pilot-based training-time estimate, not exact FLOPs or realized time. Report actual times and their ratio. If the realized A3/A2 training time is outside [0.90,1.10], label the time-control comparison inconclusive and do not claim budget-controlled improvement. No extra corrective training run is permitted. Count pilot time within the 24-hour cap.

Adapters change the shared block in both passes. They are not independent copies and not depth-conditioned in V0. This is a deliberately small recurrence-adaptation ablation, not a reproduction of Think-at-Hard. At rank-zero/disabled adapters, A1 is the dense baseline; A2 is frozen F1. Do not claim initial A2 identity to the dense model.

## Decisions and limits

Primary scientific comparison: A2 versus both A1 and A3 on a sealed structural-generalization test. Ordinary LoRA may improve the base model; that alone is not evidence for recurrence. Frozen candidates are diagnostic only; their failure does not block the adaptation test if implementation and budgets pass.

Before discovery, F0 must have macro exact accuracy strictly between 5% and 95% and global invalid-format rate <=10%. A task family is informative only when its F0 exact accuracy is strictly between 5% and 95% and its invalid-format rate is <=10%; at least two families must be informative. All families remain in macro scoring and reporting, but a non-informative family cannot support a cross-family capability claim.

Discovery promotion requires at least +2 percentage points macro exact accuracy versus F0 and both adapted controls on development, no task-family decrease greater than 2 points versus either F0 or A1, FineWeb PPL ratio point estimate <=1.02 versus F0, paired chunk-bootstrap 97.5 percentile PPL ratio <=1.02, and a valid A3 timing control. These are project decision thresholds, not theoretical guarantees. If it fails, write the negative, retention-uncertain, or inconclusive result and stop before opening final tests or running seed 48.

Confirmation requires two seeds with positive test improvements over F0 and both adapted controls, mean macro improvement >=2 points against each, a positive lower endpoint of the paired task-stratified 95% bootstrap CI for the seed-averaged improvement against each, no family regression >2 points averaged across seeds versus either F0 or A1, PPL ratio point estimate and paired chunk-bootstrap 97.5 percentile ratio <=1.02 per seed, and valid A3 timing controls for both seeds. Failure is no-go, retention-uncertain, or inconclusive under this protocol. No prompt, rank, layer, dataset, or seed search follows.

Passing these gates supports narrow verifiable-task improvement, not general intelligence. Recurrence increases executed layers from 28 to 32 per token; FLOPs, activation memory and measured uncached latency must be reported. Efficient cached serving is a separate future design only after quality passes.

## Evidence and reproducibility

Use new artifacts/recurrence-v0/<run-id>/ roots, immutable source and data manifests, append-only event JSONL, atomic training checkpoints, model/adapter/optimizer/RNG state, unique examples and failed-attempt histories. A failed score is never an omitted example. Final evaluation is an explicit stage enabled by a passing development manifest; commands cannot silently run all stages.

The protocol and task contracts in ../../../plan.md complete this specification. The exhaustive execution order is in ../plans/2026-09-11-pretrained-recurrence-execution.md. None of these documents is an experiment result.

## Primary references

- https://arxiv.org/html/2605.23872v1 — frozen recurrence with damping; motivation only, no benchmark reproduction claim.
- https://arxiv.org/html/2511.08577v4 — selective recurrence and depth-aware LoRA; motivates adaptation, but its selector and attention mechanism are out of scope.
- https://arxiv.org/abs/2502.05171 — recurrence trained at scale; not evidence of cheap post-hoc conversion.
- https://huggingface.co/Qwen/Qwen3-1.7B — model and tokenizer source; pin a revision before execution.

The repository paths and installed Transformers Qwen3 implementation were inspected on 2026-09-08. The model has not been downloaded or tested for this study.
