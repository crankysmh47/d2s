# D2S Project Log

This file is an append-only record of material project actions, decisions, verification evidence, failures, and research gates. Do not record credentials, access tokens, private dataset contents, or large generated outputs here.

## 2026-08-30

### 10:37–10:47 PKT — Initial project assessment and relocation

- User authorized the initial phases of `plan.md`, requested proactive reporting of problems or harmful conditions, and required a durable project log.
- Agreed first slice: Phase 0 synthetic validation plus a guarded Phase 1 readiness probe. The 100K-token Phase 1 experiment remains blocked until Phase 0 equivalence gates pass.
- Verified local environment:
  - Python 3.11.9
  - `uv` 0.12.5
  - NVIDIA GeForce RTX 5060 Ti, 16,311 MiB VRAM
  - NVIDIA driver 581.57
  - approximately 201 GB free on `C:`
- Verified the official `Qwen/Qwen3.5-0.8B-Base` model repository exists and its published text dimensions match the research plan: hidden size 1024, intermediate size 3584, and 24 layers.
- Found `C:\Users\Hank47\Documents\d2s` was an empty, unrelated Git repository with no files, commits, or remote. The actual project remained intact in the private GitHub repository and the earlier checkout.
- Attempted to move the empty folder to `C:\sem4\d2s`. Windows rejected the move because the running Codex task holds the source folder open. No files changed during the failed move.
- Recovery: cloned `https://github.com/crankysmh47/d2s.git` into `C:\sem4\d2s` and created branch `phase0-foundation` from commit `3712046`.
- Residual cleanup: the old `C:\Users\Hank47\Documents\d2s` folder contains only its empty `.git` directory. It must be removed after this Codex task releases the filesystem lock.

### Initial safety and cost gates

- Do not run model-weight downloads or large dataset downloads implicitly.
- Do not begin the 100K-token oracle experiment until Phase 0 tests pass.
- Prefer configuration-only Qwen inspection first; record download size before fetching weights.
- Keep generated checkpoints, caches, datasets, and experiment outputs out of Git.
- Treat FP32 all-expert equivalence as a hard correctness gate, not a best-effort metric.
- Log BF16 drift separately; do not describe ordinary BF16 accumulation differences as exact equivalence.

### 10:47 PKT — Design approval and dependency hazard

- User approved `docs/superpowers/specs/2026-08-30-phase0-foundation-design.md` and authorized implementation planning.
- Confirmed from the upstream Transformers Qwen3.5 implementation that the text MLP uses bias-free `gate_proj`, `up_proj`, and `down_proj` linear layers with the expected row/row/column neuron axes.
- Identified an upstream Transformers compatibility hazard: composite Qwen3.5 checkpoints loaded through `AutoModelForCausalLM` have had a reported path where the requested `dtype` is silently dropped. This slice performs configuration-only inspection and does not load weights. Any later weight loader must use the concrete `Qwen3_5ForCausalLM` class and verify actual parameter dtypes after loading.
- Implementation mode: inline execution on `phase0-foundation`; no delegated agents were requested.

### 11:00 PKT — Task 1 environment resolution issue

- `uv sync --group dev` resolved 58 packages and installed `torch==2.13.0+cpu`, `transformers==5.16.1`, and `pytest==8.4.2`.
- Verification showed `torch.version.cuda is None` and `torch.cuda.is_available() is False`; therefore this lock cannot satisfy the required CUDA BF16 gate despite the host RTX 5060 Ti.
- Root cause: the default PyPI Windows Torch artifact is CPU-only in this resolution path.
- Official PyTorch wheel listings provide a CPython 3.11 Windows CUDA 13.0 build for Torch 2.13.0. NVIDIA driver 581.57 is from the CUDA 13-capable driver family.
- Mitigation: configure an explicit uv source for `torch` at `https://download.pytorch.org/whl/cu130`, re-lock, and verify `torch.version.cuda` plus an actual CUDA tensor operation before accepting the environment.

### 11:15 PKT — Task 1 RED/GREEN and environment recovery

- RED: `uv run pytest tests/test_neuron_triplet.py -v` failed during collection with `ModuleNotFoundError: No module named 'ppmoe.moefication.neuron'`. This was the intended missing-production-module failure.
- Implemented `SwiGLUParameters` shape/dtype/device validation and `validate_permutation` without adding permutation or partition behavior.
- GREEN: the same command passed 9 tests in 1.05 seconds.
- The first CUDA sync attempt timed out after updating `uv.lock`; a resumed sync installed `torch==2.13.0+cu130` but left an incomplete CPU `torch-2.13.0.dist-info` metadata directory.
- Permanent deletion of that generated orphan was blocked by environment policy. It was recoverably moved to `.venv/orphaned-metadata/torch-2.13.0.dist-info`; a clean `uv sync --group dev` then reported 39 packages checked with no warning.
- CUDA proof used the project venv and a real BF16 matrix multiplication:
  - Torch `2.13.0+cu130`
  - CUDA runtime 13.0
  - GPU `NVIDIA GeForce RTX 5060 Ti`
  - `torch.cuda.is_bf16_supported() == True`
  - matrix result contained only finite values
- Transformers resolved to `5.16.1`; pytest resolved to `8.4.2`.
- Next gate: complete-neuron permutation must preserve FP32 output and leave source tensors unchanged.

### 11:20 PKT — Task 2 neuron-triplet permutation

- RED: `uv run pytest tests/test_permutation.py -v` failed during collection because `permute_swiglu_parameters` was absent.
- Implemented a non-mutating permutation that applies one validated order to gate rows, up rows, and down columns; gate/up biases move with neuron indices and the down bias is cloned without permutation.
- GREEN regression: `uv run pytest tests/test_neuron_triplet.py tests/test_permutation.py -v` passed 12 tests in 2.44 seconds.
- Seed 7 FP32 measurement: maximum absolute output difference `1.90734863e-06`; relative L2 error `7.84195393e-08`.
- Source tensor non-mutation, requested index order, absent-bias behavior, and output equivalence are covered by tests.
- Next gate: balanced partitions must be complete/disjoint and all-expert execution must match the dense FFN with the down bias added exactly once.

### 11:25 PKT — Task 3 balanced reference experts

- Partition RED: `uv run pytest tests/test_partition.py -v` failed because `ppmoe.moefication.partition` did not exist.
- Partition GREEN: 4 tests passed, covering ordered equal groups, non-divisible widths, non-positive expert counts, and invalid permutations.
- FFN RED: `uv run pytest tests/test_ffn_equivalence.py -v` failed because `ppmoe.moefication.reference` did not exist.
- Implemented unoptimized `DenseSwiGLU`, `ExpertSwiGLU`, and `PartitionedSwiGLU`. Each expert owns complete gate/up rows and down columns; only the partitioned parent owns the down bias.
- FFN GREEN: partition and equivalence subset passed 11 tests in 1.46 seconds.
- Full regression: `uv run pytest -v` passed 23 tests in 1.45 seconds.
- Seed 11 FP32 all-expert measurement:
  - maximum absolute error `1.1920929e-07`
  - relative L2 error `2.00065458e-07`
  - dense parameters `1,616`
  - partitioned parameters `1,616`
- This passes the `1e-5` single-FFN FP32 gate and preserves the parameter count in the measured configuration.
- Next gate: input and mapped parameter gradients plus two-layer synthetic model output must match.

### 11:30 PKT — Task 4 gradient and model characterization

- Plan deviation recorded before execution: gradient propagation was already part of the Task 3 reference implementation, so fabricating a RED failure would have been misleading. Task 4 added characterization tests only and made no production-code change.
- `uv run pytest tests/test_gradient_equivalence.py tests/test_model_equivalence.py -v` passed 2 tests in 1.49 seconds on first execution.
- Seed 17 maximum absolute gradient differences:
  - input `1.39698386e-09`
  - gate weight `5.58793545e-09`
  - up weight `3.25962901e-09`
  - down weight `4.65661287e-09`
- Seed 19 two-layer synthetic Transformer:
  - maximum absolute final-state difference `2.38418579e-07`
  - relative L2 error `7.29517566e-08`
- Full non-CUDA regression: 25 tests passed in 1.50 seconds.
- All measurements pass the FP32 tolerances. Next gate: centralize numerical metrics with independently hand-computed expectations.

### 11:35 PKT — Task 5 numerical metrics

- Corrected a planning arithmetic error before implementation: cosine similarity for reference `[1, 2]` and candidate `[2, 2]` is `6 / sqrt(40)`, not `4 / sqrt(20)`.
- RED: `uv run pytest tests/test_verify_metrics.py -v` failed because `ppmoe.moefication.verify` did not exist.
- Implemented FP64 calculation of maximum absolute error, RMS error, relative L2 error, normalized MSE, and cosine similarity in immutable `NumericalMetrics`.
- GREEN: 4 hand-computed and boundary tests passed in 1.01 seconds.
- Full non-CUDA regression: 29 tests passed in 1.48 seconds.
- Next gate: direct fixed-K subset enumeration must choose the hand-derived best subset independently for each token and resolve exact ties lexicographically.

### 11:40 PKT — Task 6 direct exact oracle

- Corrected a test-fixture arithmetic error before implementation: the all-expert literal reconstruction `[0.5, 2.5]` has squared error `2.5` from `[1, 1]`, not `1.25`.
- RED: `uv run pytest tests/test_oracle_bruteforce.py -v` failed because `ppmoe.oracle.exact` did not exist.
- Implemented deterministic lexicographic fixed-K subset enumeration with per-token direct reconstruction scoring in FP32, or FP64 when the source is FP64.
- GREEN: 6 direct-oracle tests passed in 1.45 seconds.
- Full non-CUDA regression: 35 tests passed in 1.40 seconds.
- Covered distinct token-wise selections, exact zero error, lexicographic ties, all-expert error, invalid K, and incompatible shapes.
- Next gate: Gram-form scoring must return the identical subsets and errors, including an explicit cancellation case.

### 11:45 PKT — Task 7 Gram exact oracle

- RED: `uv run pytest tests/test_gram_oracle.py -v` failed because `exact_gram_oracle` was absent.
- Implemented the expanded objective `||t||^2 - 2 m^T E^T t + m^T E^T E m`; subset selection uses unclamped scores and only the returned selected error is clamped for tiny negative roundoff.
- GREEN oracle regression: 9 tests passed in 1.43 seconds, including explicit expert cancellation and lexicographic ties.
- Seed 23 direct-versus-Gram evidence:
  - K=1: subsets identical, maximum error delta `7.62939453e-06`
  - K=2: subsets identical, maximum error delta `7.62939453e-06`
  - K=4: subsets identical, maximum error delta `1.23977661e-05`
  - K=8: subsets identical, maximum error delta `9.60961916e-06`
- Full non-CUDA regression: 38 tests passed in 1.44 seconds.
- The small objective deltas are FP32 algebra/accumulation-order effects; subset choices remained identical for all seeded cases.
- Next gate: generate deterministic JSON Phase 0 reports and compare native BF16 versus FP32 expert accumulation on CUDA.

### 11:55 PKT — Task 8 reproducible FP32/BF16 reports

- RED core: Phase 0 and BF16 tests failed during collection because `ppmoe.phase0` did not exist.
- Implemented `run_phase0_probe` with deterministic seeds, versioned JSON-safe output, parameter counts, numerical metrics, and direct/Gram agreement across every K.
- CPU probe tests passed 2/2; real CUDA BF16 drift test passed 1/1.
- CLI RED: subprocess test failed with exit code 2 because `scripts/verify_phase0.py` did not exist.
- Implemented the real JSON CLI; subprocess GREEN passed and verified exit code, output creation, schema, seed, and device.
- Full suite including CUDA: 42 tests passed in 3.05 seconds with no skips.
- `results/phase0-fp32.json` SHA256: `36157F5053644885C293DBF90B4450296B5EF5F0B5E417F3B0EFB191624D9964`.
- FP32 report, seed 31, hidden 128, intermediate 512, 8 experts:
  - maximum absolute error `1.1920928955078125e-07`
  - relative L2 error `2.7393640199064574e-07`
  - normalized MSE `7.50411523355806e-14`
  - cosine similarity `0.9999999999999634`
  - direct/Gram subsets identical for K=1 through K=8
  - maximum direct/Gram squared-error delta `4.023313522338867e-07`
- `results/phase0-bf16.json` SHA256: `C43B6CFC0040463D01AFAF7355E6AB04CA7DFAD367D9DA590ECC09C9B09B5380`.
- CUDA BF16 report on RTX 5060 Ti:
  - maximum absolute error `0.001953125`
  - relative L2 error `0.0025266201633892898`
  - normalized MSE `6.383809450045322e-06`
  - cosine similarity `0.9999968138585839`
  - peak allocated CUDA memory `35,479,040` bytes
  - native BF16 and FP32 expert accumulation followed by BF16 cast produced identical reported metrics for this seed
  - direct/Gram subsets identical for K=1 through K=8
- Both dense and partitioned forms contain `196,608` parameters. Generated `results/` files remain ignored by Git.
- Next gate: inspect the official Qwen3.5 config and text MLP layout without downloading weights.

### 12:05 PKT — Task 9 Qwen3.5 configuration readiness

- Adapter RED: `uv run pytest tests/test_qwen35_probe.py -v` failed because `ppmoe.adapters.qwen35` did not exist.
- Implemented pure config and `nn.Linear` MLP layout inspection with explicit missing-field and shape errors.
- CLI RED: the real subprocess test failed with exit code 2 because `scripts/probe_qwen35.py` did not exist.
- Implemented a CLI that calls `AutoConfig.from_pretrained(..., trust_remote_code=False)`, initializes only one untrained `Qwen3_5MLP`, and never calls a model `from_pretrained` method.
- Offline GREEN: 7 adapter/CLI tests passed in 5.68 seconds using a locally saved real miniature `Qwen3_5Config`.
- Official config probe: `uv run python scripts/probe_qwen35.py --output results/qwen35-config-probe.json` exited zero.
- Official `Qwen/Qwen3.5-0.8B-Base` evidence:
  - model type `qwen3_5`; text model type `qwen3_5_text`
  - hidden size `1024`; intermediate size `3584`; 24 text layers; activation `silu`
  - 8 balanced experts imply width `448`
  - gate weight `[3584, 1024]`, up weight `[3584, 1024]`, down weight `[1024, 3584]`
  - gate/up/down projections are bias-free
  - Transformers `5.16.1`; `trust_remote_code=false`; `weights_loaded=false`
- Repository weight-file scan before and after remained zero.
- Probe SHA256: `C6819750E63090AFE56F89090B2B1887E4605C1496AC3953824627FFFC151EC5`.
- Operational warnings for future Phase 1:
  - Hugging Face requests are unauthenticated, giving lower rate limits.
  - Windows Developer Mode/symlink support is disabled; Hugging Face cache falls back to a degraded mode that can duplicate data and use more disk.
- Later weight loading must use concrete `Qwen3_5ForCausalLM`, verify actual parameter dtypes after load, and measure VRAM. Do not rely on `AutoModelForCausalLM(dtype=requested_dtype)` for this composite checkpoint.
- Next gate: run final integrity/tests/reports and decide M0 PASS or FAIL. No weight download is authorized in this slice.

### 12:15 PKT — Task 10 final verification and gate decision

- Terminology correction: the implementation checklist initially called this synthetic completion decision `M0`. The original `plan.md` defines global M0 as a fully converted Qwen model matching the dense teacher with no meaningful perplexity difference. This slice did not load/convert Qwen weights or measure perplexity, so global M0 cannot honestly be marked PASS.
- Fresh integrity verification:
  - branch `phase0-foundation`
  - `git diff --check` exited zero
  - `uv lock --check` resolved 58 packages and exited zero
  - all three generated result files are ignored by `.gitignore`
  - repository model-weight file count is zero
- Fresh test command: `uv run pytest -v -ra` passed 49 tests in 7.48 seconds with zero failures and zero skips, including the RTX 5060 Ti BF16 test.
- Fresh report regeneration produced the same SHA256 hashes as Task 8/9:
  - FP32 `36157F5053644885C293DBF90B4450296B5EF5F0B5E417F3B0EFB191624D9964`
  - BF16 `C43B6CFC0040463D01AFAF7355E6AB04CA7DFAD367D9DA590ECC09C9B09B5380`
  - Qwen config `C6819750E63090AFE56F89090B2B1887E4605C1496AC3953824627FFFC151EC5`
- Fresh FP32 values: relative error `2.73936401990646e-07`, maximum absolute error `1.19209289550781e-07`, cosine `0.999999999999963`, equal parameter counts, identical direct/Gram subsets.
- Fresh BF16 values: relative error `0.00252662016338929`, maximum absolute error `0.001953125`, cosine `0.999996813858584`, equal parameter counts, identical direct/Gram subsets, peak allocated CUDA memory `35,479,040` bytes.
- Fresh official Qwen config values: hidden 1024, intermediate 3584, 24 layers, expert width 448, `weights_loaded=false`, `trust_remote_code=false`.
- Gate decision:
  - `PHASE 0 FOUNDATION: PASS`
  - `GLOBAL M0 (FULL QWEN CONVERSION + PERPLEXITY): PENDING`
  - `PHASE 1 WEIGHT DOWNLOAD: NOT STARTED`
- Recommended next slice: resolve Hugging Face authentication/cache policy, record expected checkpoint bytes, load with concrete `Qwen3_5ForCausalLM`, verify parameter dtypes and VRAM, convert one text FFN, then one text layer, then the full text stack before perplexity evaluation.
- Residual filesystem cleanup: `C:\Users\Hank47\Documents\d2s` remains an empty locked Git shell from task startup and can be removed after the Codex task releases it.

### Branch publication — `exp`

- User chose to keep `main` unchanged until the project has its first real Phase 1 results.
- Renamed local branch `phase0-foundation` to `exp`.
- Verified neither `exp` nor `dev` existed on the remote before publication.
- Pushed `exp` to the private GitHub repository and configured it to track `origin/exp`.
- First remote verification matched local commit `813fe9df0c9659489f979d8433481d076cf404aa` before this logging commit.
- Next research dependency: the original plan's Phase 2 learned router is blocked on Phase 1 local/rollout oracle results and oracle-derived labels. Do not train a router before the Phase 1 go/no-go gate.

### Phase 1 V0 protocol approval and design lock

- User clarified that current work continues the original plan's Phase 1; it must not be relabeled as Phase 2.
- User approved deterministic FineWeb-Edu calibration plus raw WikiText-103 evaluation:
  - model `Qwen/Qwen3.5-0.8B-Base` at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`;
  - FineWeb-Edu `sample-10BT/train` at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, exactly 100,000 Qwen tokenizer tokens;
  - WikiText `wikitext-103-raw-v1/test` at revision `b08601e04326c79dfdd32d625aee71d232d685c3`;
  - an initially 100,000-token disjoint FineWeb-Edu held-out evaluation.
- Live dataset-server inspection verified FineWeb-Edu exposes `id`, `text`, `url`, and provenance-related fields; document IDs can therefore be recorded directly. Calibration and held-out sets will be disjoint at the whole-document level.
- Approved execution is a gated funnel: loader verification, one-FFN/one-layer/full-stack equivalence, dense/all-expert PPL, 2,048-token oracle smoke, 100,000-token local oracle, then rollout oracle and both relative-PPL measurements.
- The concrete Qwen loader and actual BF16 parameter verification are hard requirements because the generic automatic loader may drop the requested dtype for this composite checkpoint.
- User requested fewer commits. Phase 1 commit cadence is limited to meaningful milestones: combined design/plan, verified implementation, and first real results.
- All work remains on `exp`; `main` remains unchanged until the first real results are reviewed.

### Phase 1 Task 3 loader-gate correction

- Before downloading weights, a real-class construction probe found the approved design named the wrong concrete loader class.
- The pinned repository declares `Qwen3_5ForConditionalGeneration` and uses a composite `Qwen3_5Config`. Its checkpoint contains 488 indexed tensors under `model.language_model.*`, `model.visual.*`, and MTP prefixes; total indexed weight bytes are `1,746,882,752`.
- `Qwen3_5ForCausalLM` instead declares `Qwen3_5TextConfig`. Passing the repository's composite config to that class reproducibly fails because the top-level config has no flat `vocab_size` field.
- A miniature real `Qwen3_5ForConditionalGeneration` successfully executed text-only inputs without pixel values, exposing text FFNs at `model.language_model.layers[*].mlp` while retaining `model.visual` separately.
- Corrected loader policy: use concrete `Qwen3_5ForConditionalGeneration`, verify actual BF16 parameters after loading, feed text-only inputs, convert only the language-model FFNs, and assert the vision module remains untouched.
- No model weights were downloaded during this investigation; only configuration and the small safetensors index were accessed. Hugging Face again warned that requests are unauthenticated and Windows cache symlinks are unavailable.

### 20:55–22:08 PKT — Phase 1 V0 implementation checkpoint

- User authorized autonomous continuation until a genuinely consequential design fork and requested implementation delegation to one execution agent at a time. Architecture, protocol decisions, review, and integration remained with the primary agent; a single GPT-5.6 Terra agent executed bounded Tasks 5–7 without committing or changing the scientific protocol.
- Implemented the locked Phase 1 V0 pipeline:
  - immutable model/dataset configuration and canonical atomic provenance;
  - deterministic exact-target FineWeb token streams, whole-document-disjoint held-out selection, stable WikiText row IDs, and full raw-test materialization;
  - guarded concrete `Qwen3_5ForConditionalGeneration` loading with actual BF16/class/device checks;
  - deterministic physical complete-neuron permutation and eight-way Qwen text-FFN conversion without parameter-count change or vision conversion;
  - mask-correct FP64 NLL/PPL accumulation with failed-batch invalidation and resume;
  - transactional local exact oracle with FP32 Gram scoring, CPU FP64 aggregation, one-time direct/Gram verification, selection/co-selection, interaction, and cancellation statistics;
  - transactional residual-aware greedy rollout oracle using teacher next-layer targets and the evolving student state;
  - gated `preflight`, `datasets`, `m0`, `local-smoke`, `local`, `rollout`, and `report` CLI stages with signature-bound checkpoints;
  - five required value-preserving CSV/PNG report outputs and output/source hashes.
- Review corrections made before acceptance:
  - rollback all partial local/rollout batch metrics before recording a failure;
  - call only `model.language_model` for oracle passes to avoid unnecessary vocabulary logits;
  - keep BF16/FP32 oracle scoring in FP32 and promote per-token metrics before FP64 reduction;
  - run direct-versus-Gram verification only once per layer/K and persist its coverage;
  - persist the full conversion manifest rather than only its hash;
  - isolate M0 checkpoint directories by model/conversion/token signature;
  - load one model pair for the complete rollout stage rather than reloading for every corpus/K;
  - require exact text-layer coverage `0..23` in reports;
  - separate checkpoint weight bytes from total repository bytes in preflight evidence.
- A causal-NLL unit fixture initially placed logits at label positions rather than preceding causal positions. The observed NLL matched that fixture error; the fixture was corrected without changing production scoring code.
- The first dependency-inclusive test attempt timed out during `uv` resolution. A separate `uv sync --group dev` completed successfully; the locked environment now contains `datasets==4.8.5` and Matplotlib in addition to the existing CUDA Torch/Transformers stack.
- Fresh live read-only preflight succeeded on branch `exp` without loading model weights:
  - CUDA available: true; BF16 available: true;
  - Torch `2.13.0+cu130`; Transformers `5.16.1`; Datasets `4.8.5`;
  - free disk `210,016,460,800` bytes;
  - pinned model, FineWeb-Edu, and WikiText revisions all matched;
  - safetensors weight bytes `1,746,942,600`;
  - total repository bytes `1,769,913,749`;
  - preflight JSON SHA256 `E767720750AFC3612F5ACD6EDE6DDC757F3AB3F00D9EEE27C62BDAB2BE36D296`.
- `artifacts/phase1-v0/` is ignored by Git. Only the preflight artifact exists; Phase 1 datasets and model weights have not yet been downloaded/materialized by the experiment pipeline.
- Gate status remains:
  - `PHASE 1 IMPLEMENTATION: PENDING FINAL COMMIT VERIFICATION`;
  - `GLOBAL M0: PENDING REAL MODEL CONVERSION + PERPLEXITY`;
  - `M1: PENDING 100K LOCAL + ROLLOUT ORACLE RESULTS`.
- `main` remains unchanged. The next milestone is dataset materialization followed by real-model M0 and the 2,048-token local smoke gate on `exp`.

## 2026-08-31

### 10:20–10:40 PKT — Independent implementation review and final verification

- A fresh read-only Terra review inspected the entire uncommitted Phase 1 Tasks 1–7 tree against the approved specification. No reviewer edits were accepted blindly; findings were checked against the actual code and protocol.
- Critical resume finding resolved: batch indexes are only meaningful for one microbatch partition. M0, local, and rollout signatures and checkpoint paths now include `microbatch_size`. Reducing microbatch size after OOM starts a clean signature-isolated attempt; the failed attempt remains preserved as negative evidence and cannot silently omit or double-count contexts.
- Critical loader-evidence finding resolved:
  - production validation requires exactly 24 text layers, hidden size 1024, intermediate size 3584, and the required gate/up/down matrix shapes at every layer;
  - every pair load enforces actual BF16 parameters and at least 4 GiB free CUDA memory after both models are resident;
  - the M0 model manifest persists immutable class/dtype/dimension/shape/parameter evidence and that load's free/total/peak CUDA measurements;
  - later stages compare immutable model identity while independently enforcing fresh memory headroom, rather than incorrectly requiring dynamic free/peak byte values to repeat exactly.
- Conversion evidence strengthened:
  - a deterministic layer-0 actual-weight CPU FP32 probe must have maximum absolute all-expert error `<=1e-5` before full mutation;
  - the conversion manifest now records both physical permutations and expert-group original-neuron indices;
  - production orchestration requires 24 converted layers and expert width 448.
- NLL aggregation correction: causal cross-entropy remains FP32 per token, but per-token losses move to CPU FP64 before reduction and durable addition. Full-vocabulary logits are not expanded to FP64.
- M0 now records the fixed batch index, input-ID hash, attention-mask hash, finite-logit status, maximum absolute logit difference, and cosine in addition to both PPL aggregates and their ratio.
- Report plots now annotate K and token/scored counts; local and rollout plots include zero-error references and PPL retains the dense ratio-1 reference.
- Fresh final verification after all review fixes:
  - `uv run pytest -q`: 122 passed in 21.92 seconds, zero failures;
  - `uv lock --check`: passed with 85 resolved packages;
  - `git diff --check`: passed; only Windows LF-to-CRLF notices were emitted.
- Fresh live preflight after review fixes passed without loading weights:
  - free disk `221,282,013,184` bytes;
  - safetensors weight bytes `1,746,942,600`;
  - total repository bytes `1,769,913,749`;
  - all three pinned remote revisions matched;
  - preflight JSON SHA256 `C793B5443E89593BDD4AE2FCA63E90489B95FCFE5011C2AB955560107807D149`.
- Gate status at this checkpoint:
  - `PHASE 1 IMPLEMENTATION: PASS`;
  - `LIVE PREFLIGHT: PASS`;
  - `GLOBAL M0: PENDING REAL WEIGHT LOAD + CONVERSION + PPL`;
  - `M1: PENDING LOCAL/ROLLOUT RESULTS`.
