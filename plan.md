# Pretrained recurrence capability study — implementation plan

> For agentic workers: use the executing-plans skill to implement this plan sequentially. Use subagents only when the user explicitly requests delegation. Checkboxes are work items, not evidence that an experiment has run.

**Goal:** Determine whether one extra pass through a pretrained transformer block, adapted with LoRA, improves verifiable-task accuracy beyond ordinary LoRA at a modest training budget.

**Architecture:** A separate recurrence module wraps a conventional Qwen3 decoder, reusing four middle layers. Frozen controls, ordinary LoRA, recurrent LoRA, and an extra-training-time control share data and evaluation. Every stage produces durable evidence and obeys explicit gates.

**Tech stack:** Existing Python 3.11, uv, PyTorch/CUDA, Transformers, datasets, pytest; no package rename, new serving stack, or PEFT dependency required. Implement the small LoRA primitive locally and test it mathematically.

**Spec:** [2026-09-08-pretrained-recurrence-design.md](docs/superpowers/specs/2026-09-08-pretrained-recurrence-design.md).

**Execution plan:** [2026-09-11-pretrained-recurrence-execution.md](docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md). It expands this protocol into the only authorized sequential implementation and experiment order.

**Status (2026-10-03):** This is the historical recurrence protocol, **not an active execution queue**. V0/V1/V2 stopped at their registered calibration gates; V3 stopped at its pre-GPU review-evidence gate. No recurrence LoRA trained. The user authorized revised [Option 2 verified-distillation QLoRA](docs/superpowers/plans/2026-09-27-verified-distillation-qlora.md); [execution state](docs/distillation-v1-execution-state.md) records its progress. See [todo.md](todo.md), [PROJECT_LOG.md](PROJECT_LOG.md), and the [V3 qualification report](docs/recurrence-v3-scorer-qualification.md). The historical protocol below remains unchanged.

## 1. Direction and document precedence

The historical hypothesis was that limited adaptation can make existing layers useful when revisited. This changes the objective from sparse efficiency to capability improvement at approximately fixed stored parameters and increased inference computation. Keep this track in the current workspace because the model evaluation/provenance infrastructure is reusable; do not describe it as successful dense-to-sparse conversion.

Select only:

1. Frozen recurrence controls: original, naive repetition, and damped repetition. This is a small diagnostic, not a search campaign.
2. LoRA adaptation of a fixed repeated block, with ordinary adaptation and extra-training-time controls. This is the primary experiment.

Native-MoE signal analysis, expert merging, specialist assembly, shared correction modules, sparse LoRA recovery, variable-depth selectors, quantization, and custom pretraining are outside the active implementation queue. Do not combine recurrence and sparsification. Their discussion is not authorization to implement them.

Document precedence: this plan and its matching spec govern recurrence; PROJECT_LOG.md and immutable artifacts govern historical outcomes; [todo.md](todo.md) is a checklist only. The old roadmap is retained at [docs/archive/2026-08-30-dense-to-sparse-plan.md](docs/archive/2026-08-30-dense-to-sparse-plan.md). Its Phase 1B–4 instructions are deferred, not deleted or silently executed. Old implementation plans remain historical records for their own phases.

## 2. Established evidence to preserve

Phase 0 and real all-expert conversion checks passed. Phase 1A random 8x448 routing gave WikiText PPL 29.0723 at K6 and 54.1177 at K4 versus dense 20.7837; FineWeb PPL 22.2418 and 39.6923 versus 16.0961. Local calibration covered 100,000 tokens; rollout covered 299,357 and 99,804 causal tokens per setting with no failed batches.

Phase 1A.5 completed 100,000 calibration tokens, 196 batches, one recovered CUDA failure, and zero unresolved failures. K4 registered free reconstruction reduced NMSE 0.2485723 to 0.2071603 (16.66%); K6 improved 9.90%. No condition met the registered 30% gate; no reconstruction rollout ran. The continuation used a recovery-only revision; do not call it a single-revision rerun or an exact-span impossibility proof.

Before and after implementation, verify these files against these SHA256 values, comparing hexadecimal case-insensitively:

| File | SHA256 |
|---|---|
| artifacts/phase1-v0/stages/m0.json | b9be2ecfe2b73a906682b652d65ce1c9db094e0d23ecc825ab00e677bd3a40d0 |
| artifacts/phase1-v0/stages/local.json | 90382ea91c74b1850fb7aef81c6f4705dd625a8559ef0fa725bd4511fb2d8341 |
| artifacts/phase1-v0/stages/rollout.json | 0f415af16ebffb6231041605c0a72c4a3505c6db1f5ff832d6126b8b339e334e |
| artifacts/phase1-v0_5/runs/a8ab03b/stages/local.json | 5c48266c4cb5429e9b8aa3544b7afe112ef3b198c6084cc02921ac0c7da472b1 |

Historical source data are read-only. No existing checkpoint or experiment output may be overwritten.

## 3. Global constraints and fixed decisions

- Work on `exp`; never edit main as part of execution. Preserve unrelated edits. One commit per completed implementation gate if commits are requested; no commit per tiny step.
- Model: `Qwen/Qwen3-1.7B` (post-trained), not the Base variant. Its conventional attention removes hybrid recurrent-state ambiguity. Do not substitute the existing Qwen3.5 checkpoint.
- At source stage resolve the model's immutable full commit SHA using Hugging Face metadata, then download config/tokenizer/weights only at that revision. Save the SHA, license, file paths, sizes and hashes. Mutable `main` cannot appear in an executable run manifest.
- Expected model: Qwen3ForCausalLM, 28 decoder layers, hidden 2048, intermediate 6144, Q heads 16, KV heads 8, head dimension 128. Assert from config and loaded tensors. A mismatch is a source gate failure, not permission to choose another model.
- Loop range is half-open `[12,16)`, zero-based. It contains four complete decoder layers. No range, rank, or damping search.
- Use installed locked dependencies initially; do not upgrade to copy a web example. Inspect actual Qwen3 signatures. Any necessary dependency change requires lock update, fresh compatibility tests, and a new run signature before production results.
- BF16 frozen weights, FP32 adapter matrices and AdamW state. Eager attention, no torch.compile, no fused custom kernels, no quantization. Training microbatch 1, accumulation 8, max total sequence 512.
- `use_cache=False`, `past_key_values=None` everywhere. Generation recomputes full prefixes. This deliberately sacrifices speed to establish causal correctness; do not claim serving efficiency.
- Same checkpoint chat template for every arm, `enable_thinking=False`, no system message, no demonstrations. Greedy decoding, max_new_tokens 32, EOS stop. These protocol choices are constant controls, not prompt interventions.
- No online teacher/API cost, RL, model-generated rationales, benchmark prompt tuning, or synthetic labels produced by an LLM. Labels come from deterministic interpreters.
- Maximum 24 hours of active CUDA-job wall time across smoke, pilot, training, evaluation, and failures. Count synchronized job intervals, not idle/user time. Maximum download budget 8 GiB and new study artifact budget 20 GiB; stop before exceeding either. No automatic expansion.
- A model/data access problem is an environment stop. Insufficient accuracy is a scientific stop. Report them separately.

## 4. Exact model semantics

Write the original forward pass as `suffix(G(prefix(tokens)))`. G includes attention, both norms, FFN, and residual additions in layers 12–15. No output head or final norm occurs inside G.

The core tensor transform is:

```python
def apply_block(block, h, mode):
    if mode == "single":
        return block(h)
    if mode == "repeat":
        return block(block(h))
    if mode == "damped":
        z = h + 0.5 * (block(h) - h)
        return z + 0.5 * (block(z) - z)
    raise ValueError(f"unsupported mode: {mode}")
```

The actual `block` closure captures a causal attention mask, original position IDs and rotary position embeddings. They are identical across repetitions. Keep all original layer indices. No module copying, resetting the residual, adding fake tokens, depth-as-token position changes, or cache mutation.

Reuse the same underlying decoder layer objects for both calls. In the installed Transformers implementation, each Qwen3DecoderLayer returns a Tensor. Inspect and assert this contract instead of guessing tuple offsets. Build mask with the installed `create_causal_mask`; reject sliding attention, external cache, and multi-sequence padded batches in V0. Prefix length must be >=1 and <=512.

Keep the stock model registered exactly once as a child of `RecurrentQwen3`. Expose `forward(input_ids, mode)` returning full logits for tests/evaluation, and `answer_loss(input_ids, labels, mode)` using hidden states and chunked head projection for training. Do not edit site-packages or monkeypatch process-global forward methods. Access stock `model.embed_tokens`, `model.layers`, `model.rotary_emb`, `model.norm`, and `lm_head` only through this adapter module.

Frozen F0 uses `single`; F1 uses `repeat`; F2 uses `damped`. Adapted A1 and A3 use `single`; A2 uses `repeat`. F2 is a deliberately minimal damping baseline inspired by published recurrence, not a complete reproduction. A2 is not Think-at-Hard: it has no selective halting, depth-specific adapters, or duo-causal attention.

## 5. LoRA and training semantics

Wrap `q_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in each of layers 12–15: 24 projections in total. Freeze every original parameter. Each wrapper owns A of shape `[8,in_features]` and B of shape `[out_features,8]`, both FP32, scale 16/8=2. Initialize A with Kaiming uniform using the arm seed and B to zero; dropout is zero. Biases remain frozen.

```python
def lora_forward(base, a, b, x):
    delta = torch.nn.functional.linear(
        torch.nn.functional.linear(x.float(), a), b
    )
    return base(x) + (2.0 * delta).to(x.dtype)
```

Disable autocast around the FP32 adapter path if needed to enforce this contract. Optimizer sees A/B only. A1/A2/A3 start from identical saved adapter initialization for each seed. Both A2 passes share the same A/B objects; gradients accumulate from both. Do not maintain a frozen first-pass copy. Adapter-off A1 must match F0; adapter-off A2 must match F1. A2 is not initially function-equivalent to F0.

Use AdamW lr=1e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=0; clip global adapter gradient norm to 1.0. Linear warmup: lr at update s is `1e-4*min(s/20,1)` for s starting at 1. No scheduler search or early-best-checkpoint selection. Train 1,000 optimizer updates for A1/A2, eight examples per update, exactly 8,000 unique examples per seed. Sum per-example mean answer/EOS cross-entropy divided by 8, not an inconsistent mix of token-weighted and example-weighted loss. Include EOS once; prompt labels are -100.

Use non-reentrant activation checkpointing for each trainable-block call and suffix layer during training (`use_reentrant=False`), explicitly passing or safely binding all per-layer arguments so recomputation cannot capture a later loop index. Prefix may execute without gradient because it is frozen; suffix must retain gradient with respect to its input even though suffix weights are frozen. Do not wrap suffix or the second block call in `no_grad`.

The vocabulary is large: compute training CE on supervised positions only, in chunks of 16 positions, using FP32 log-softmax/CE and BF16 head projection. Apply the causal shift explicitly: for each label at token index j>=1 with labels[j]!=-100, project hidden[j-1] and predict labels[j]. Never use hidden[j] to predict its own token. Accumulate sums and divide by supervised-position count for that example. Full-prefix attention still runs; do not supply gold answers at inference.

Seeds: 47 discovery, 48 confirmation. Seed Python, NumPy, torch and CUDA; save all RNG states. Data order is a seeded permutation of the fixed 8,000-example pool. Within a seed it is identical for all arms; A3 cycles that permutation without reshuffling. Seed 48 does not generate a new dataset.

### Extra-training-time control A3

Before production, measure A1 and A2 on identical 20 optimizer-step batches after five warmup updates each. Synchronize CUDA around each complete step including forward, backward, clipping and optimizer; exclude data loading. Alternate measurement order A1/A2 to reduce thermal bias. Use disposable initialized adapters, then reload original initialization before production. Pilot consumes study budget but not production steps.

Let `n3=max(1000,ceil(1000*median_t2/median_t1))`. Freeze n3 and the raw timing samples in pilot.json. A3 uses A1 architecture for n3 updates, cycling the same examples. This is a pilot-estimated training-time control, not an exact FLOP or inference-time match. Report realized A3/A2 production training time; outside [0.90,1.10] means the budget-controlled conclusion is inconclusive. Never run extra corrective training to make the result pass.

No adapters are merged during V0 evaluation. Stored trainable parameters, executed layers, and runtime are reported separately. Loop-shared adapters may be mergeable later; depth-specific adapters would not have the same simple merge semantics.

## 6. Data protocol: deterministic, disjoint, verifiable

Create a new JSONL pool with fields `id,family,split,structure_id,prompt,answer,latent_spec`. `id=sha256(canonical_json({family,latent_spec}))`. Interpreters use integers/lists/graphs directly, never Python eval or arbitrary generated code. No LLM generation is involved.

Use `random.Random(seed)` independently per family/split: train seeds 101/102/103, dev 201/202/203, test 301/302/303. Deduplicate IDs globally, reject duplicates, and cap attempts at 100 times required count. Failure to fill is a generator error. Train counts 2667 expressions, 2667 lists, 2666 graphs. Dev 200 per family. Sealed final test 800 per family. Generate once; save hashes before any model scores. Task structures change with depth across train/dev/test; model checkpoint pretraining contamination remains unknown, but generated instances are new and split-disjoint.

### Expressions

Sample fully parenthesized binary trees with n leaves. Train n uniformly in {3,4}; dev n=5; test n=6. Each split uses both shapes: left-associated chain and balanced tree (recursive floor(n/2) split). Train alternates shape by accepted row index, as do dev/test. Leaves are independent integers 0..9. Internal operators sampled uniformly from `+,-,*`. Reject abs(answer)>9999. Canonical serialization uses spaces around operators.

Prompt exactly: `Compute {expression}. Return only the integer answer.` Answer is base-10 integer, no plus sign or leading zeros except zero itself. structure_id records shape and leaf count. This tests greater compositional depth, not an unseen operator.

### List programs

Input length train 3..5 uniformly, dev 6, test 7. Values 0..9 independently. Operation vocabulary ROTL1 (move first element to end), REVERSE, ADD1 (each value +1 modulo 10). Program length train {1,2}, dev 3, test 4. Sample operations uniformly; reject adjacent REVERSE/REVERSE pairs and any complete program returning the unchanged input. Store the original input and sequence; interpret step by step. Rejection is based on data only, never model performance.

Prompt exactly: `Start with {list}. Apply {ops} from left to right. ROTL1 moves the first element to the end; REVERSE reverses the list; ADD1 adds 1 modulo 10 to each element. Return only the final list.` Lists use `[1,2,3]` syntax without spaces; ops are comma-separated names. Answer is canonical JSON integer list. structure_id is operation sequence plus input length.

### Directed reachability

Vertices A onward, n train {5,6}, dev 7, test 8. Sample each ordered distinct vertex pair as an edge with probability 0.20; sort edges lexicographically. Query two distinct uniformly sampled vertices. Compute BFS using directed edges. Accept examples alternating target label yes/no, so each split is balanced to within one example. Empty edge set is allowed. Do not tell the model the desired label. structure_id records n and shortest path length (or unreachable).

Prompt exactly: `Directed edges: {edges}. Is there a directed path from {start} to {end}? Return only yes or no.` Edge syntax `A->B, B->C`; use `none` for empty. Answer `yes` or `no`. This is size generalization, not a claim all graph topology classes were held out.

### Rendering, parsing and masks

Render exactly one user turn with the model template and `enable_thinking=False`. Training input is rendered generation prefix plus tokenized canonical answer plus EOS. Assert it is <=512 tokens and answer+EOS <=32 tokens. Reject tokenizer-length violations before freezing data; record counts. Do not truncate problems or answers. At evaluation, enforce prompt<=480 tokens; no generation beyond total512.

Remove special tokens from completion and strip surrounding whitespace. Parse the entire remaining string: expression `-?(0|[1-9][0-9]*)`; list JSON array containing only integers 0..9 of the required length; graph case-insensitive `yes` or `no`. No substring extraction, rationale stripping, repair, majority voting or retries. Invalid format, NaN, missing EOS at generation cap and wrong values count incorrect; record distinct reason codes. Correctness is exact interpreted answer equality.

### General-language retention

Resolve original FineWeb held-out document IDs and text provenance from existing manifests/data loader; read source text from the pinned original dataset revision, or use its cached raw text. Never decode old token IDs as a substitute. If text cannot be recovered exactly, stop the retention stage. Re-tokenize under Qwen3 into a new stream of exactly20,000 valid tokens, EOS between documents, 512-token chunks, last partial retained. Score next-token NLL excluding the first position of each chunk; report the resulting scored-token count. All arms use identical token IDs. Persist per-chunk NLL sums and scored-token counts. For every adapted-arm/F0 comparison, run 10,000 paired chunk-bootstrap replicates with seed9157, resampling the same chunk indices for both arms and recomputing the token-weighted PPL ratio. The point estimate and 2.5/97.5 percentile interval are both reported. This set is a retention monitor, not a new independent reasoning test.

## 7. Runtime, errors, and artifact contract

Artifact root: `artifacts/recurrence-v0/<run-id>/`, where run-id is the first12 hex of SHA256 of source/data/config/code signatures. Refuse an existing incompatible root. Source stage may create a staging manifest before the final data signature is known; finalize the signed run root before smoke. Never silently replace manifests.

Minimum files:

```text
source.json                   # checkpoint SHA, dependencies, hardware, source-file hashes
data/manifest.json             # generator config, split hashes/counts, template/tokenizer hashes
data/train.jsonl
data/dev.jsonl
data/test.jsonl                # sealed from model scoring until selection passes
stages/<stage>.json            # passed, stopped_scientific, stopped_budget, failed_environment
events.jsonl                  # monotonically indexed, append-only stage/progress events
pilot.json                    # timings, memory, n3, pessimistic projection
checkpoints/<arm>/<seed>/      # adapter + optimizer + RNG + next update
scores/<arm>/<seed>/<split>.jsonl
failures/<attempt-id>.json
selection.json                # dev-only decision, fixed final-test conditions
report.json
report.md
```

Reuse canonical hashing and atomic JSON helpers from ppmoe.provenance; they allow replacement, so wrap them in an explicit immutable-manifest writer that rejects an existing different payload. Mutable checkpoint pointers are separate from immutable completed stage records. Torch checkpoint writes use sibling temporary files and atomic replace; files are local trusted artifacts only.

Checkpoint every50 optimizer updates and on orderly stop, with latest and previous recovery checkpoints retained. Save adapter state, optimizer state, scheduler/update index, data cursor, CPU/CUDA RNG, exact initialization hash, source/config hashes and accumulated times. Flush one progress record every10 updates. Resume only from a complete optimizer step, replaying uncommitted work. Do not retry within a CUDA-faulted process. At most one fresh-process retry per stage; preserve all attempt costs/failure history, and stop after a repeated fault.

Infrastructure exceptions during evaluation invalidate the stage. They are not counted as wrong answers, and no partial aggregate is eligible for promotion. Invalid model output is a scored wrong answer, not infrastructure failure. Exact expected example IDs and counts must reconcile before a stage is complete.

## 8. Gates and fixed experiment sequence

G0 source/data: current branch exp; old hashes preserved; immutable model revision; correct architecture; all split checks pass; signed inputs and all fixed values present. No benchmark outputs inspected while choosing defaults.

G1 implementation: tiny-model identity, causality, gradient, loss, serialization and resume checks pass. Real-model baseline wrapper versus stock logits on 8 fixed dev prompts: finite, BF16 max absolute difference<=1e-3 and cosine>=0.999999; report exact differences. They should normally be identical with eager execution. Fail without relaxing thresholds. Full model eval mode, caches off, identical tensors.

G2 resource: run 25-step pilot per A1/A2 and 12 generated dev examples per frozen condition (4 per family), plus 4 retention chunks. Peak reserved CUDA <= total VRAM minus2 GiB; disk bounds respected. Pessimistic time estimate = 1.5 times projected cost of all planned arms, evaluations, two seeds, and already spent time; must fit24 hours. Use longest observed prompt and full32-token completion to bound evaluation cost. If it fails, write budget stop. Do not shrink sample sizes, switch models, quantize, or change output cap automatically.

G3 frozen diagnostic: F0/F1/F2 on the 600 dev examples plus retention. Reject the benchmark as uninformative if F0 macro accuracy<=5% or>=95%, or invalid-format rate>10%. Also classify each family as informative only when its F0 exact accuracy is strictly between5% and95% and its F0 invalid-format rate is<=10%; require at least two informative families or stop without redesigning prompts. Non-informative families remain scored and reported but cannot support a cross-family capability claim. Record wrong-to-right and right-to-wrong versus F0. No frozen quality threshold blocks A2: fixed-weight failure does not prove adaptation failure. No final-test evaluation of frozen alternatives at this stage.

G4 discovery: train seed47 A1,A2,A3 in that order; evaluate final checkpoint of each on dev/retention. Promote only if A2 macro exact accuracy is at least 2 percentage points higher than F0, A1 and A3, no family drops more than 2 points versus either F0 or A1, A2 retention PPL/F0 point estimate<=1.02, the paired chunk-bootstrap 97.5 percentile ratio<=1.02, and A3/A2 realized training time lies in [0.90,1.10]. Otherwise stop: scientific no-go, retention uncertainty, or inconclusive budget control as appropriate. No hyperparameter sweep. Select A2, not the best of an expanding candidate set.

G5 confirmation: after selection.json is frozen, train seed48 A1/A2/A3 from fresh seeded initialization using the same fixed n3. Do not use final tests until all six final checkpoints exist. Score final test for F0 and each trained arm/seed, then retention for seed48. Each test has2,400 examples; reconcile all. No best-seed or checkpoint choice.

Success requires all: positive macro gain for each seed versus F0 and each adapted control; mean gain>=2 percentage points versus each; mean family regression no worse than -2 points versus either F0 or A1; A2 PPL/F0 point estimate<=1.02 and paired chunk-bootstrap 97.5 percentile ratio<=1.02 for both seeds; both A3 timing controls valid. Compute 10,000 paired accuracy-bootstrap replicates, seed9047, resampling example IDs WITHIN each task family and using the same sampled IDs for all arms/seeds. Average the two seed differences per example before resampling (reuse the single deterministic F0 correctness vector for both seed comparisons); take 2.5/97.5 percentiles of macro differences. All three lower endpoints must exceed0. These CIs describe item uncertainty for two fixed seeds, not population-wide seed uncertainty. If any requirement fails, report no-go/inconclusive; do not silently weaken the claim.

G6 report: quality, family scores, paired transitions, confidence intervals, invalid-output rates, PPL/NLL counts, storage/trainable parameters, active decoder-layer evaluations, peak allocated/reserved memory, pilot and realized CUDA-job time, complete failure history. Record all frozen dev outcomes; do not hide F1 failures. Show A2's additional inference work (32 versus28 layer evaluations per token) and uncached measured latency. No inference-efficiency or general-reasoning claim from this reference study.

If success: recommend a separately scoped cached-runtime/general-benchmark replication. If failure: close the study and summarize which hypothesis failed. Neither outcome automatically resumes Phase1B or starts another idea.

## 9. Implementation map and task contracts

New files only unless explicitly listed. Existing modules are imported, not refactored. New `configs/recurrence-v0.toml` serializes the constants above. Do not edit historical TOMLs.

### Task 1 — source, configuration and data (G0)

Files: `src/ppmoe/recurrence/__init__.py`, `config.py`, `data.py`, `provenance.py`; `configs/recurrence-v0.toml`; `tests/test_recurrence_data.py`, `tests/test_recurrence_config.py`.

Interfaces:

```python
@dataclass(frozen=True)
class Example:
    id: str
    family: str
    split: str
    structure_id: str
    prompt: str
    answer: str
    latent_spec: dict

@dataclass(frozen=True)
class RenderedExample:
    example: Example
    prompt_input_ids: tuple[int, ...]
    train_input_ids: tuple[int, ...]
    labels: tuple[int, ...]

def make_examples() -> dict[str, list[Example]]: ...
def parse_answer(text: str, example: Example) -> object: ...
def render_example(tokenizer, example: Example) -> RenderedExample: ...
def write_immutable_json(path: Path, payload: dict) -> str: ...
def validate_config(payload: dict) -> dict: ...
```

These signatures specify interfaces, not placeholder implementation permission. Implement algorithms exactly as sections3/6/7 describe. Config rejects any altered registered constant, unknown key, invalid range and mutable revision in a finalized source manifest. Source resolver produces immutable revision before loader runs. Save source Python files/config/license snapshots with hashes.

- [ ] Write deterministic generation/count/disjointness/answer-interpreter tests first. Check train/dev/test structures meet the prescribed depths/sizes and no ID crosses splits.
- [ ] Test strict whole-string parsing, not just happy paths:

```python
def test_no_answer_substring_repair():
    example = Example("id", "expressions", "dev", "chain5", "q", "12", {})
    with pytest.raises(ValueError):
        parse_answer("The answer is 12", example)
    assert parse_answer(" 12\n", example) == 12
```

- [ ] Test JSON booleans rejected as list integers, duplicate generation rejection, generation attempt cap, answer/EOS mask alignment, and overlength rejection.
- [ ] Implement data/source/config functions. Run `uv run pytest -q tests/test_recurrence_data.py tests/test_recurrence_config.py`.
- [ ] Run source/data stage only after CLI exists; no automatic model evaluation during data creation.

### Task 2 — exact recurrent Qwen wrapper (G1 part1)

Files: `src/ppmoe/recurrence/model.py`; `tests/test_recurrence_model.py`.

Interfaces: `apply_block(block,h,mode)->Tensor`; `RecurrentQwen3(stock_model)`; `.forward(input_ids:Tensor, mode:str)->Tensor` returns `[1,T,V]` logits; `.hidden(input_ids,mode)->Tensor` returns final-normalized `[1,T,d]`; `.answer_loss(input_ids,labels,mode)->Tensor` returns scalar.

- [ ] Prove the tensor recurrence before integrating Transformers:

```python
def test_recurrence_algebra():
    h = torch.tensor([2.0])
    g = lambda x: 2*x
    assert torch.equal(apply_block(g,h,"single"), torch.tensor([4.0]))
    assert torch.equal(apply_block(g,h,"repeat"), torch.tensor([8.0]))
    assert torch.equal(apply_block(g,h,"damped"), torch.tensor([4.5]))
```

- [ ] Build tiny Qwen3ForCausalLM from config locally (no weights download), with 6 layers and test override window[2,4), hidden32, intermediate64, Qheads4, KVheads2, head_dim8, vocab97. Test-only construction bypasses production layout checks explicitly.
- [ ] Compare single wrapper with stock at FP32 rtol1e-5/atol1e-6. Check shared object IDs and unchanged parameter count for repeat.
- [ ] Causality test: sequences with identical first5 tokens and different suffixes must have equal first5 logits in all modes at FP32 tolerance. Cache input and unsupported mode must raise ValueError.
- [ ] Implement explicit prefix/block/suffix execution with the existing API and disabled caches. Test final norm/head called exactly once and block layers called twice in repeat/damped.
- [ ] Run `uv run pytest -q tests/test_recurrence_model.py`.

### Task 3 — adapters, loss and recoverable trainer (G1 part2)

Files: `src/ppmoe/recurrence/lora.py`, `train.py`; `tests/test_recurrence_lora.py`, `tests/test_recurrence_train.py`.

Interfaces: `LoRALinear(base:nn.Linear, rank:int=8, alpha:float=16)`; `install_adapters(model,seed)->list[nn.Parameter]`; `train_arm(model,examples,arm,seed,steps,checkpoint_dir,event_path)->dict`. Returned manifest includes completed updates/examples, init/final adapter hashes, trainable count, elapsed time, peak memory, and completion status.

- [ ] Test zero-B identity and analytical nonzero adapter result on a small linear matrix. Assert all base parameters frozen and only A/B included in optimizer.
- [ ] Test repeated block gradients against explicitly calling the same block twice, and against a finite-difference gradient on a tiny FP64 adapter. Test suffix gradients reach adapters.
- [ ] Test chunked answer loss equals full-logits masked CE, prompt tokens do not contribute, EOS does contribute, and accumulation matches eight sequential example losses divided by8.
- [ ] Implement the exact optimizer/order/checkpoint rules. On a tiny deterministic CPU model, compare uninterrupted4 updates with2+resume2; assert adapter/optimizer/RNG/cursor agreement. Resuming a different signature must fail.
- [ ] Test interrupted temporary checkpoint cannot replace a complete checkpoint; previous recovery state remains usable. Test failure history is not erased by retry.
- [ ] Run `uv run pytest -q tests/test_recurrence_lora.py tests/test_recurrence_train.py`.

### Task 4 — evaluator, timing and gates

Files: `src/ppmoe/recurrence/evaluate.py`, `gates.py`; `tests/test_recurrence_evaluate.py`, `tests/test_recurrence_gates.py`.

Interfaces: `generate_answer(model,input_ids,mode,max_new_tokens=32)->dict`; `evaluate_examples(model,examples,mode,output_path)->dict`; `evaluate_retention(model,batches,mode)->dict`; `discovery_decision(scores,retention,timing)->dict`; `confirmation_decision(scores,retention,timing)->dict`.

- [ ] Use tiny scripted logits to verify greedy decoding, EOS, cap handling and full-prefix recomputation. No gold answer passed into generation. Test invalid output remains a scored item; an execution exception prevents complete=true.
- [ ] Validate expected ID sets, duplicate IDs, exact family counts and signed split authorization. Reject sealed test stage without passing selection manifest.
- [ ] Use synthetic paired correct/incorrect arrays to test macro gain, wrong/right transitions, seeded stratified bootstrap, global/per-family informativeness and every exact gate boundary. Changing one task's sample count must not change equal-family weighting. Require at least two informative F0 families.
- [ ] Test a retention stream's causal count independently against a hand-computed two-chunk example. Compute NLL sums in FP64; reject zero scored tokens/nonfinite values. Test the 10,000-replicate seed9157 paired chunk bootstrap, exact chunk pairing and the PPL-ratio point/upper-interval boundaries.
- [ ] Implement synchronized pilot timing, ceil n3 rule, actual-time qualification, memory thresholds and cumulative cost ledger. Test attempts count toward cap and a stage cannot start with insufficient projected budget.
- [ ] Run `uv run pytest -q tests/test_recurrence_evaluate.py tests/test_recurrence_gates.py`.

### Task 5 — CLI and bounded orchestration

Files: `scripts/run_recurrence_v0.py`, `src/ppmoe/recurrence/experiment.py`; `tests/test_recurrence_cli.py`, `tests/test_recurrence_experiment.py`.

Interface: CLI flags `--stage {source,data,smoke,pilot,frozen,discover,confirm,report}`, `--config configs/recurrence-v0.toml`, `--artifact-root artifacts/recurrence-v0`, optional `--run-id` for stages after source/data. No `all` stage and no flags overriding fixed scientific constants. CLI prints the resolved run-id/path before work.

Source creates staging metadata; data finalizes signed run-id, then all later stages require `--run-id`. Source/data do not open any scores. `smoke` performs identity/short gradients with disposable adapters; `pilot` fixes n3 and budget; `frozen` runs G3; `discover` runs seed47/G4; `confirm` runs seed48 and sealed tests/G5. `report` works after success or any terminal stop and never launches missing work.

- [ ] Test dependency DAG with fake model/data backends: missing predecessor blocks; scientific stop cannot be bypassed; unchanged completed stage is read-only; incompatible signature fails.
- [ ] Test source/data finalization yields stable run-id independent of timestamps. Runtime code changes invalidate production stages; a recovery-only change must be explicit and record mixed revision rather than silently claim identical provenance.
- [ ] Implement process exit codes:0 complete/pass,2 scientific stop,3 budget stop,1 environment/implementation failure. Always persist a stage/failure record before exiting when filesystem is writable.
- [ ] CLI help tests must not load a model, contact network or create run artifacts.
- [ ] Run `uv run pytest -q tests/test_recurrence_cli.py tests/test_recurrence_experiment.py`.

### Task 6 — report, production gates and handoff

Files: `src/ppmoe/recurrence/report.py`; `tests/test_recurrence_report.py`; update PROJECT_LOG.md and todo.md only after observed evidence.

Interface: `build_report(run_root:Path)->dict` writes report.json/Markdown from persisted stages and score records. Missing evaluation is `not_evaluated`, never zero accuracy. Include source links, model revision, protocol deviation list, per-family informativeness, retention point/bootstrap interval and narrow supported conclusion. A stopped study must explain what was completed and what did not run.

- [ ] Test complete negative, budget-stop, partial failure and positive fixtures. Hash output sources and reject unverifiable aggregates, fabricated confidence intervals or missing arm/seed comparisons.
- [ ] Run all recurrence tests, then `uv run pytest -q`, `uv lock --check`, and `git diff --check`. Do not rewrite old tests to hide regressions.
- [ ] Verify historical hashes; run real smoke and pilot. Advance only through sections8/10. Record production outputs separately from implementation verification.
- [ ] Append concise evidence to PROJECT_LOG.md at each gate: exact IDs/counts, steps, scored/failure counts, time/memory, decision and next allowed command. Mark todo items only when their deliverables exist.

## 10. Execution commands and automatic stopping

These commands are planned interfaces, not currently available executables. After implementation and an execution request, run sequentially:

```powershell
uv run python scripts/run_recurrence_v0.py --stage source
uv run python scripts/run_recurrence_v0.py --stage data
```

Read the emitted signed run-id. Assign it to `$recurrenceRunId` as its literal12-character value; never derive it from an unrelated latest directory. Then:

```powershell
uv run python scripts/run_recurrence_v0.py --stage smoke --run-id $recurrenceRunId
uv run python scripts/run_recurrence_v0.py --stage pilot --run-id $recurrenceRunId
uv run python scripts/run_recurrence_v0.py --stage frozen --run-id $recurrenceRunId
uv run python scripts/run_recurrence_v0.py --stage discover --run-id $recurrenceRunId
uv run python scripts/run_recurrence_v0.py --stage confirm --run-id $recurrenceRunId
uv run python scripts/run_recurrence_v0.py --stage report --run-id $recurrenceRunId
```

Inspect exit status and persisted gate after EACH command; this block is not permission to execute downstream stages after a stop. On any stop, run report, document the blocker/negative finding, and end execution. Do not automatically download a different model, collect new tasks, add a specialist, or resume sparse work.

## 11. Why these decisions and what they cannot establish

One checkpoint and one block prevent a large validation search from masquerading as a method. Conventional attention and no cache simplify the first correctness question. LoRA limits trainable state; it does not guarantee small activation memory or inexpensive repeated inference. Verified synthetic labels eliminate teacher-generation cost, but restrict the conclusion to the selected task families. Structural depth/size shifts test transfer beyond memorized examples; they are not proof of general reasoning.

A1 isolates adaptation-data benefit, A3 tests extra training budget, F1 isolates naive repetition and F2 checks damping. No control matches inference cost exactly, so the study claims only an accuracy-versus-added-inference-cost point, not a superior compute frontier. The shared adapter setup deliberately tests the smallest learnable recurrence hypothesis before selective depth or specialist composition.

References and evidence boundaries are in the matching spec. They motivate the study; all rank/window/budget/gate values here are explicit project choices, not settings claimed to reproduce published headline results.
