# D2S project history and current status

Prepared 2026-10-13 from the conversation, repository plans, PROJECT_LOG.md, experiment reports and retained artifacts. This is a historical synthesis, not a new experiment or a replacement for signed results. Work remains on `exp`; `main` is unchanged.

## Bottom line

We began with parameter-preserving dense-to-sparse conversion, then moved to affordable post-training capability improvement. The conversion mathematics and all-expert real-model equivalence worked. The tested random partition lost substantial language-model quality when experts were omitted, and the registered reconstruction follow-up did not qualify. Recurrence was implemented and exercised, but its benchmark/evaluation gates prevented adapted-model training. The later teacher-rationale QLoRA study passed synthetic GPU feasibility and real generation, then stopped at a failed answer-extraction qualification.

There is **no demonstrated sparse-efficiency gain, trained recurrence-adapter result, or verified distillation performance gain**. There is reusable implementation, preserved negative evidence, and a concrete diagnosis of the present evaluator blocker.

## 1. Initial objective and ideas

The original objective was to take a pretrained dense model and give it MoE-like conditional computation while retaining approximately the same total parameters and acceptable quality. Expert offloading/residency belongs to ExpertFlow; D2S therefore needed to show useful active-compute reduction, not merely a different storage layout.

The distinction matters: partitioning a dense FFN stores all its weights. It does not automatically reduce model file size or resident memory. With all partitions active, it is a re-expression of dense computation; with some omitted, it becomes an approximation that must be measured.

The initial plan used Qwen3.5-0.8B-Base on one roughly16-GiB GPU, rather than first pretraining a new small model. It proposed exact conversion, all-expert equivalence, then an oracle upper-bound diagnostic before spending on a learned router. Both local teacher-forced and propagated rollout behavior were required. Pairwise expert cancellation/interactions and streaming statistics were included.

Other ideas discussed but not executed were:

- Distilling or adapting a sparsified model with LoRA/QLoRA to recover omitted dense behavior.
- Studying native-MoE expert outputs versus combined-expert signals to look for transferable patterns.
- Merging a native MoE into a dense model and testing whether quality changes.
- Training task specialists separately, then jointly adapting them with a central reasoning model.
- Better neuron grouping/clustering, shared correction modules, adaptive loop depth, and custom pretraining.

These remain proposals, not experimental results. No native-MoE merging study or specialist assembly was built. The project narrowed its scope instead of attempting all of them at once.

## 2. Phase 0: prove conversion and oracle mechanics

For SwiGLU, each intermediate neuron consists of a gate row, an up-projection row, and its down-projection column. The implementation permutes these triplets together and partitions them into complete, disjoint groups. Algebraically,

`FFN(x) = sum_j down_column_j * (SiLU(gate_row_j · x) * (up_row_j · x))`.

Grouping the terms gives expert contributions whose complete sum equals the dense FFN. This identity does not prove that a small subset reconstructs the sum.

Implemented and tested:

- Shape/dtype/device and permutation validation; non-mutating neuron-triplet permutation.
- Balanced partitions and dense/expert/partitioned reference FFNs, with down bias added once.
- Forward, input-gradient and parameter-gradient equivalence; synthetic multi-layer checks.
- Maximum error, relative L2, NMSE and cosine metrics with independently calculated expectations.
- Direct fixed-K subset enumeration and equivalent Gram-form scoring, including cancellation and deterministic ties.
- Reproducible FP32/BF16 reports and actual Qwen configuration inspection without loading weights.

The foundation passed. A representative FP32 synthetic probe had relative L2 error about2.74e-7 with unchanged parameter count. BF16 drift was measured separately, not called exact equivalence. The initial final suite recorded49 passing tests, including CUDA coverage.

## 3. Phase 1A: real-model conversion and sparsity test

The real Qwen text stack had24 layers, intermediate width3584, and8 experts of448 neurons per FFN. Conversion retained852,985,920 total checkpoint parameters and left the vision module untouched. The concrete composite-model loader was corrected before weight loading.

### All-expert equivalence

The first BF16 all-expert implementation summed separate expert GEMMs. Small rounding/order differences accumulated through the model and failed the fixed-batch cosine gate, despite almost identical perplexity. Diagnostics distinguished this from corrupt weights or a bad neuron mapping.

The exact correctness path restores original neuron order and uses a monolithic down projection when all experts are active. It matched logits and perplexity exactly. Actual sparse/oracle paths retained independent-expert arithmetic; the dense correctness path was not presented as a sparse speedup.

- WikiText-103 dense and all-expert PPL:20.7837, on299,357 scored tokens.
- FineWeb-Edu held-out dense and all-expert PPL:16.0961, on99,804 scored tokens.
- Parameter count unchanged; all24 FFNs converted.

### Oracle sparsification

The local oracle covered100,000 calibration tokens per layer/K condition. Aggregate local NMSE was about0.0973 atK6 and0.2486 atK4. The subsequent rollout completed both corpora and all registered settings without failed batches.

- Keeping6/8 experts: WikiText PPL29.0723; FineWeb PPL22.2418.
- Keeping4/8 experts: WikiText PPL54.1177; FineWeb PPL39.6923.
- More aggressive omission degraded further.

This is a strong negative result for the **tested frozen random partition and local-reconstruction-oracle policy**, not a proof that all sparse conversions are impossible. It did not justify proceeding directly to a learned router. No end-to-end deployable speedup or MoE-like storage reduction was established.

## 4. Phase 1A.5: can selected experts reconstruct omitted contributions?

The follow-up tested per-token expert-output recombination atK4/K6: unit coefficients, trace-scaled ridge, coefficients bounded to[0,2], and registered free-reconstruction diagnostics. Separate dense/all-expert/reconstructed references avoided attributing BF16 decomposition error to sparsification. Stable solvers and sampled direct FP64/SVD checks verified the numerical objective.

The100,000-token run completed196 batches after one recovered CUDA misaligned-address failure. The failure and mixed recovery revision remain documented; it was not a clean single-revision run.

- K4 best registered free reconstruction reduced NMSE from0.2485723 to0.2071603:16.66% relative improvement.
- K6 improved about9.90%.
- No condition met the predeclared30% pooled-NMSE improvement gate.

The result was `selection_no_eligible_conditions`. No downstream reconstruction rollout or held-out PPL evaluation ran. Better clustering and sparse recovery were deferred, not disproved or implemented automatically.

## 5. Recurrence: change the capability hypothesis

Your next objective was meaningful, low-cost post-training improvement rather than prompt tricks. We selected a bounded looped-transformer experiment: reuse pretrained layers with shared weights, adding computation without duplicating the stored block.

The model changed to conventional Qwen3-1.7B to avoid hybrid recurrent-state ambiguity. The fixed middle block comprised zero-based layers12–15. Planned controls were original execution, one extra block pass, and damped repetition; adapted comparisons were ordinary LoRA, recurrent LoRA and an extra-training-time ordinary-LoRA control. LoRA targeted24 projections in that block, with shared adapters on both recurrent passes. The protocol also required general-language retention, inference/training costs, discovery and conditional confirmation.

Implemented source/data provenance, deterministic expressions/list-program/reachability tasks, a causal recurrent Qwen wrapper, local LoRA, answer-only training loss, optimizer-boundary checkpoint recovery, inference/scoring, retention/bootstrap gates, staged orchestration and reporting. Report validation received substantial adversarial review: count/hash reconciliation, exact arm/seed matrices, producer/consumer agreement, paired bootstrap recomputation, path/reparse protection, failure linkage and stable replay. The historical Task14 review chain ultimately closed at round7.

### Recurrence V0

The final signed run passed source/data, real-model smoke and resource pilot. It completed frozen F0/F1/F2 development diagnostics and retention checks, but stopped because no baseline task family was informative.

- F0:4/600 correct; expressions0/200, graphs4/200, lists0/200.
- Frozen repetition F1:0/600; damped repetition F2:9/600.
- F1/F0 FineWeb PPL ratio about1.1385; F2/F0 about1.0114.

The32-token output cap and strict answer grammar were major measurement concerns. These results did **not** test adapted recurrence: no production LoRA arms trained. A misleading immutable report conclusion was corrected through a separate erratum, not by rewriting the run.

### Longer-output diagnostic

A separate30-example baseline probe used256 output tokens:24 EOS and6 capped, all caps on lists. Luna final-answer review marked18 good/12 bad, while the old parser rejected all30. Expressions were10/10 good, graphs8/10, lists0/10. One correct graph final answer had an invalid explanation, motivating separate answer/explanation scoring.

Editable blank human sheets and separate model-review files were created. No completed human review is claimed. This was a small reused-development diagnostic, not a performance-improvement result.

### V1 and V2 benchmark repairs

V1 froze a new evaluator and ran six30-row pilots. Only expressions qualified; lists had0/30 correct and many caps, while graph invalid rates exceeded the gate. It stopped before controls or training.

V2 clarified list wording, revised graph extraction and used fresh splits. Its pilot passed, but300-row development failed the invalidity/two-informative-family requirements:

- Expressions94/100 correct,6 invalid.
- Lists32/100 correct,68 invalid.
- Graphs72/100 correct,19 invalid.
- Overall198/300 correct and93/300 invalid under the frozen evaluator.

A separate blind audit of all93 rejected answers plus30 accepted controls found60 rejected responses with correct final answers, but also23 wrong answers,3 contradictory graph answers and7 capped graphs. The audit was outcome-stratified; it did not establish a new unbiased300-row accuracy or retrospectively pass V2.

### V3 pre-GPU qualification

V3 attempted qualification before new model inference. After a documented control-blinding incident, a fresh nonoverlapping control set was selected under a frozen selector. Three reviewers agreed on30 extracted values, but citation-validation failed:38 review citations across18 rows did not satisfy the registered evidence grammar. Full-response V3 scoring and GPU execution never began.

V0–V3 remain terminal. Recurrence has not been shown effective or ineffective after adaptation; that comparison never ran.

## 6. Option 2: verified teacher-rationale QLoRA

We moved to a more conventional capability study: can verified Qwen3-4B worked solutions improve Qwen3-1.7B through low-cost QLoRA beyond ordinary supervised training on the same questions?

The fixed planned comparison is:

- F0: unchanged student with the same NF4/double-quant/BF16 preparation as the adapters.
- A1: QLoRA on human-written GSM8K solutions.
- A2: QLoRA on verifier-accepted teacher solutions for exactly the same IDs.

The teacher is an offline target generator, not part of deployed student inference. This is rationale supervision, not attaching teacher layers or doing full teacher-logit distillation. It does not make the student a sparse MoE.

The plan separates practical gain over F0 from teacher-specific gain over A1. It fixes rank16/alpha32, all-linear adapters,300 updates, matched padded shapes/data order/initialization, one discovery seed and conditional second seed. An optional extra arm was explicitly omitted before output. Qualification requires a120-question teacher pilot, at least800 accepted matched training examples from a1,200-question pool, and independent worked-solution auditing before training. Development, transfer, sealed accuracy and FineWeb retention remain gated.

### Completed feasibility and preparation

- Pinned PEFT/bitsandbytes/Accelerate dependencies and model/data snapshots.
- Verified a synthetic student QLoRA update at the maximum1536-token length and BF16 teacher generation in separate GPU processes.
- Student peak reserved memory about14.22GB, below the registered14.93GB process cap; teacher about8.09GB.
- Prepared authenticated GSM8K/SVAMP partitions:60 calibration,120 pilot,200 development,1,200 pool,400 sealed primary,200 transfer development and500 sealed transfer.
- Implemented gold-independent numeric extraction, durable raw journals, budget calibration and gold-free review export/reconciliation.

The synthetic optimizer update is compatibility evidence, **not** benchmark training or a trained study arm. The QLoRA corpus, production trainer and downstream study stages were not completed/executed.

## 7. Background execution and production repairs

The initial48-hour calendar window expired during an interruption before calibration. You authorized48 cumulative hours of actual worker execution instead, excluding chat/review downtime, while retaining the24-CUDA-hour cap and prior costs.

Built a detached Windows supervisor with process-tree containment, PID/creation-time ownership checks, monotonic active-job accounting, chained events, logs/heartbeats/status, durable output rows, one-use recovery authorizations and cumulative CUDA/download/artifact accounting. Separate-terminal tests proved lifetime beyond launcher exit and natural process cleanup. It does not mean every later unimplemented study stage can already run unattended.

Three linked executions stopped before answers and exposed real integration issues:

1. Initial environment failure, with inadequate diagnostics and a producer/result-schema mismatch.
2. Missing `hf_device_map` under single-device loading, plus producer costs incorrectly initialized from legacy constants.
3. After strict tensor-placement and cost propagation repairs, Transformers returned `BatchEncoding` where the prompt helper expected token IDs.

Fixes retained the safeguards: inspect actual parameters/buffers/quantization-state devices rather than weakening offload validation; seed real producer paths from authenticated inherited costs; explicitly request `return_dict=False` and strictly validate token IDs. Real cached-tokenizer and structured-generation CPU tests were added because list-only fakes had missed the dependency contract.

Single-use schema2/3/4 recovery records preserve prior stops, costs and artifact inventories. Final pre-launch verification recorded1,335 CPU tests passed,7 skipped and3 CUDA deselected, with independent task/release reviews. This historical software result is not a new October13 test run or scientific success.

## 8. Latest completed calibration and present blocker

The approved tokenizer recovery ran once. Both student and teacher loaded on CUDA and produced responses.

- 256 output/768 total: student 51/60 EOS, teacher 50/60; rejected by length gate.
- 512 output/1024 total: student 59/60 EOS, teacher 60/60, no prompt overflow; selected.
- 768/1536 was unnecessary and not run.

All120 selected-budget outputs received blind Luna extraction reviews.119 were marked extractable and one capped response invalid. That is **extractability, not119 correct answers**. One non-exact evidence quote was corrected in a new review version without changing any status/value; the original remains retained.

The frozen parser nevertheless reported48/60 student-invalid and45/60 teacher-invalid responses, with47/60 and45/60 reviewer disagreements. Limits were6 invalid and3 disagreements per role. An independent audit reproduced the failure. Many misses involved display-math boxes, Markdown punctuation and literal template cues. No mathematical accuracy was measured in this phase.

The candidate is stopped scientifically. The preserved worker snapshot says `awaiting_review`; the separate hash-bound extraction receipt records the failed gate. No pilot, teacher corpus, study QLoRA training, retention comparison or sealed model evaluation followed.

For this distillation execution lineage, saved cumulative charges are2,572 CUDA seconds,1,305 active-job seconds and8,066,718,062 downloaded bytes. These include its inherited feasibility/failed-work costs, **not** total compute/download expenditure across the entire D2S project. Earlier sparsity and recurrence ledgers are separate.

## 9. Assessment and next decision

The strongest achievements are exact conversion evidence, a meaningful negative sparsity curve, numerically checked reconstruction limits, real recurrence/QLoRA compatibility, and reusable provenance/runtime/audit tooling. Negative and failed runs were not silently rewritten.

The main weakness is that evaluator/provenance plumbing repeatedly prevented the intended adaptation comparison. Passing many unit tests did not reliably establish real dependency, producer/consumer or answer-format compatibility. A large implementation exists, but no affordable post-training capability gain has yet been demonstrated.

The next recommendation is a **separately authorized evaluator/output-format repair**, not another model idea or immediate training: use observed public responses as regression fixtures, retain ambiguous/conflicting/truncated-answer rejection, freeze the grammar and audit contract, then qualify on fresh calibration questions. Preserve all stopped protocols and costs. If qualification passes, implement teacher viability, matched corpus/rationale audit and only then the A1/A2 training comparison. No guaranteed improvement is promised, and no new experiment is authorized by this historical summary.

## Source documents and backup scope

- [Project log](../PROJECT_LOG.md): chronological evidence and deviations.
- [Original dense-to-sparse roadmap](archive/2026-08-30-dense-to-sparse-plan.md).
- [Historical recurrence protocol](../plan.md) and [V0 report erratum](recurrence-v0-report-erratum.md).
- [V1 calibration](recurrence-v1-calibration-report.md), [V2 calibration](recurrence-v2-calibration-report.md), [V2 answer audit](recurrence-v2-answer-audit-report.md), [V3 qualification](recurrence-v3-scorer-qualification.md).
- [Verified-distillation plan](superpowers/plans/2026-09-27-verified-distillation-qlora.md), [execution state](distillation-v1-execution-state.md), [terminal report](distillation-v1-report.md), [short queue](../todo.md).

Git backs up source, tests, plans, logs, this summary and the eight historical Task14 review reports added at your request. `.gitignore` still excludes datasets, weights, caches, raw results/checkpoints and artifact directories. Those remain local; **a repository push is not a backup of the raw experimental evidence**. No ignored directory was force-added, no experiment was resumed, and no merge to `main` was performed.
