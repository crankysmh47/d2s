# Phase 1 Oracle V0 Design

**Status:** Approved on 2026-08-30

**Experiment:** `qwen35_0p8b_ppmoe_oracle_v0`

## Objective

Produce the first real dense-to-sparse result on `Qwen/Qwen3.5-0.8B-Base` without training a router. Convert every text FFN into eight balanced parameter-preserving experts, prove the all-expert student is equivalent to the dense teacher, then measure the best achievable local and greedy-rollout quality for `K = [8, 6, 4, 3, 2]`.

This phase answers whether the decomposition is viable before any learned routing, distillation, shared expert, or intelligent partition work begins.

## Immutable inputs

| Input | Pinned identity | Use |
|---|---|---|
| Model | `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | Dense teacher and converted student |
| Calibration | `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split `train`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | Exactly 100,000 tokenizer tokens |
| External evaluation | `Salesforce/wikitext`, config `wikitext-103-raw-v1`, split `test`, revision `b08601e04326c79dfdd32d625aee71d232d685c3` | Dense/all-expert/oracle perplexity |
| In-domain evaluation | Same pinned FineWeb-Edu stream | Exactly 100,000 held-out tokens initially; expandable to 500,000 without changing calibration |

The model uses text only, BF16 weights, context length 512, and no remote code. The repository is a composite checkpoint whose declared concrete class is `Qwen3_5ForConditionalGeneration`; only its `model.language_model` text stack is evaluated and converted.

## Execution architecture

The run is a gated funnel:

1. Resolve and record pinned artifacts without loading weights.
2. Load one dense model and verify class, parameter dtype, device placement, weight count, and CUDA memory.
3. Build a converted student and validate one FFN, one decoder layer, and the full text stack.
4. Establish dense and all-expert perplexity with one shared tokenization/evaluation path.
5. Run a small local-oracle smoke test before the 100,000-token calibration.
6. Run the full local oracle and generate layer/K statistics.
7. Run greedy rollout and oracle perplexity only after conversion and local-oracle gates pass.
8. Emit plots, summaries, hashes, and an M0/M1 decision.

Teacher and student may coexist on the GPU because the measured checkpoint size is small relative to the 16 GB device. Every loader records peak VRAM. If two copies do not fit, the permitted fallback is CPU-offloading the inactive teacher between batch passes; changing dtype, quantizing, or silently shortening the evaluation is not permitted.

## Dataset protocol

### Deterministic token stream

- Do not shuffle FineWeb-Edu. Stream rows in repository order from `sample-10BT/train`.
- Tokenize `text` with the pinned Qwen tokenizer and `add_special_tokens=False`.
- Insert one Qwen EOS token between non-empty documents. Separators count toward the token target.
- Calibration ends at exactly 100,000 tokens, including separators. A final document may be truncated at the token boundary.
- The entire final calibration document is considered used. Held-out selection begins at the next document, so no document ID can cross the calibration/evaluation boundary.
- Pack the resulting stream into consecutive contexts of at most 512 tokens. Preserve the final partial context and its attention mask.
- WikiText uses the raw test text, the same tokenizer and EOS separator rule, and no FineWeb-derived preprocessing.

### Provenance manifest

Each corpus manifest records:

- repository, configuration, split, and exact revision;
- tokenizer repository and revision;
- ordering and shuffle policy;
- every source row index, FineWeb `id`, URL when present, text SHA256, consumed token count, and truncation flag;
- total model-token count, context count, final-context length, and manifest SHA256;
- explicit overlap check result between calibration and held-out document IDs.

Raw document text and access credentials must not appear in Git or `PROJECT_LOG.md`.

## Conversion design

Each Qwen text MLP has gate and up weights shaped `[3584, 1024]` and a down weight shaped `[1024, 3584]`. A neuron is the triplet consisting of one gate row, one up row, and the corresponding down column.

For every layer:

- create a deterministic CPU permutation using seed `20260830 + layer_index`;
- physically permute complete neuron triplets;
- divide the permuted width into eight contiguous experts of width 448;
- preserve the exact number of scalar parameters;
- store both the physical permutation and original neuron indices in the conversion manifest;
- add no router, shared expert, or new trainable parameter.

The student exposes:

- all-expert execution for correctness;
- per-expert output capture for oracle scoring;
- token-wise selected-expert execution for rollout.

## Numerical gates

### Loader gate

Stop before conversion if any condition fails:

- resolved model revision differs from the pinned SHA;
- loaded class is not `Qwen3_5ForConditionalGeneration`;
- any floating-point parameter is not BF16;
- any required text MLP is missing or has an unexpected shape;
- CUDA is unavailable or peak memory leaves insufficient space for the next stage.

### Conversion gate

The following checks are mandatory:

- FP32 unit conversion maximum absolute FFN error `<= 1e-5`;
- converted student parameter count equals dense text-model parameter count;
- all 24 text FFNs are converted and the vision stack is untouched;
- fixed-sequence BF16 all-expert logits are finite, cosine similarity is at least `0.99999`, and discrepancies are reported rather than hidden.

### M0 gate

M0 passes only if all-expert WikiText evaluation uses an identical token stream and:

```text
0.999 <= PPL_all_expert / PPL_dense <= 1.001
```

Failure means a conversion or evaluation bug until disproved. Oracle experiments stop.

### Oracle smoke gate

Before the full calibration, run 2,048 FineWeb-Edu tokens through all 24 layers and every requested K. Require:

- finite expert outputs, Gram matrices, errors, and normalized metrics;
- K=8 reconstruction consistent with the all-expert output;
- exact Gram and direct reconstruction subsets agree on a seeded sample;
- resumable output survives an interrupted/restarted test.

### M1 research gate

At K=4 (50% active FFN), report rather than assume:

- normalized local reconstruction error;
- rollout hidden-state divergence;
- `PPL_sparse / PPL_dense` on WikiText raw test;
- the same ratio on FineWeb-Edu held-out.

The working target is perplexity degradation no greater than 2%. Missing the target is a valid negative result and blocks router training; it does not authorize retuning the protocol after seeing the result.

## Oracle definitions

### Local oracle

For teacher hidden inputs, capture the dense FFN target and all eight expert contributions. For each token, layer, and K, choose the exact fixed-K subset minimizing squared reconstruction error using the Gram objective. Verify selected cases by direct reconstruction.

### Greedy rollout oracle

Run the student sequentially. At each layer, use the current student state, evaluate every fixed-K expert subset, and select the subset minimizing distance to the dense teacher's next-layer hidden state. Continue from the selected student state. The selection is token-wise and deterministic; ties are lexicographic.

A global sequence-loss oracle is not part of V0.

## Perplexity contract

- Evaluate causal cross-entropy with a one-token shift and attention-mask-aware token counts.
- Never score the first token of an independent stream or any padded token.
- Accumulate summed negative log likelihood and token count in FP64; derive PPL once at the end.
- Dense, all-expert, and sparse evaluations consume the identical persisted token IDs and masks.
- Persist NLL sum, scored-token count, failed-batch count, PPL, PPL ratio, and peak VRAM.
- Any failed batch invalidates the aggregate; do not report partial PPL as complete.

## Durable outputs

Generated data lives under ignored `artifacts/phase1-v0/`:

```text
manifests/       pinned model, conversion, and corpus provenance
tokens/          deterministic token-ID tensors and masks
checkpoints/     resumable oracle accumulators
metrics/         JSONL batch records and final JSON summaries
plots/           five required PNG plots plus plot-source CSV files
logs/            command output and environment snapshots
```

The committed repository contains code, tests, configuration, compact result summaries, manifest hashes, and the append-only project log. It does not contain weights, datasets, raw text, bulky tensors, or secrets.

Required plots are:

1. K versus local FFN reconstruction error.
2. Active FFN fraction versus rollout hidden-state error.
3. Active FFN fraction versus perplexity ratio.
4. Layer-by-K sparsifiability heatmap.
5. Expert interaction/cancellation and co-selection heatmaps.

## Failure and safety policy

- Check free disk space before weights, corpus materialization, and each long run.
- Do not retry CUDA OOM unchanged. Record the failure, reduce evaluation microbatch size, and resume from a verified checkpoint.
- Do not downgrade model precision, alter K values, change the corpus, or reduce token targets to manufacture a pass.
- Keep `main` unchanged. All work remains on `exp` until first real results are reviewed.
- Append material decisions, commands, failures, token counts, hashes, and gate outcomes to `PROJECT_LOG.md`.
- Commit only at meaningful milestones: design/plan, verified implementation, and first results.
