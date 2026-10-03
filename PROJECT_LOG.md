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

### 10:42–10:48 PKT — Real Phase 1 dataset materialization

- Executed the committed dataset stage once:
  - `uv run python scripts/run_phase1_v0.py --stage datasets --artifact-root artifacts/phase1-v0 --microbatch-size 1`
  - exit code `0`; elapsed `288.705` seconds; stage status `complete`.
- Deterministic corpus evidence:
  - FineWeb-Edu calibration: exactly `100,000` Qwen tokens, `196 x 512` tensor shape, 130 source document records;
  - FineWeb-Edu held-out: exactly `100,000` Qwen tokens, `196 x 512`, 90 source document records;
  - calibration next row `130` equals held-out start row `130`;
  - calibration/held-out document-ID intersection is empty;
  - WikiText-103 raw test: complete stream, `299,943` Qwen tokens, `586 x 512`, 2,891 nonempty records from 4,358 source rows.
- Independently verified stage signature and all manifest/tensor hashes. Manifest SHA256 values:
  - calibration `FAE56A3EF5E50D03CA2C939B99D04E91DDAEBA0828B6E46CFA1DFFE61E920A92`;
  - held-out `476BC91A0C976835B1FCE96989A863C9F2419664EB7AB21619008F2F94BBA658`;
  - WikiText `23FFA28D98D4FE0F5318F23EE104BABA42330EC46B677B23FCB99D2A08336D91`.
- Artifact-root size increased by `8,655,816` bytes. Hugging Face cache increased by `887,680,934` bytes during tokenizer and streamed dataset access.
- Environment warnings preserved:
  - Hugging Face requests remain unauthenticated;
  - Windows symlink-cache degradation remains active;
  - FineWeb access encountered two read timeouts and one DNS lookup failure; library retries recovered without a manual rerun.
- No raw document text entered Git or the project log. `artifacts/` remains ignored.
- Next gate: real two-model BF16 load, locked-dimension/headroom validation, actual-weight FP32 conversion probe, full conversion, and dense/all-expert M0 perplexity on both corpora.

### 10:52–11:02 PKT — First real M0 attempt stopped at fixed-batch equivalence gate

- Executed exactly once with microbatch size 2:
  - `uv run python scripts/run_phase1_v0.py --stage m0 --artifact-root artifacts/phase1-v0 --microbatch-size 2`
  - exit code `1`; elapsed `622.416` seconds; no retry or parameter change.
- Checkpoint/model gates that passed before the stop:
  - concrete `Qwen3_5ForConditionalGeneration`, actual BF16 parameters on `cuda:0`;
  - 24 text layers, hidden size 1024, intermediate size 3584, and every gate/up/down shape validated;
  - two-model CUDA evidence: free `12,197,036,032`, total `17,102,864,384`, peak `3,481,065,472` bytes; the 4 GiB headroom gate passed;
  - 24 converted FFNs, 8 experts of width 448, and unchanged parameter count `852,985,920`;
  - layer-0 actual-weight CPU FP32 probe maximum absolute difference `7.450580596923828e-08`, below `1e-5`;
  - runtime vision-module identity guard passed.
- Complete WikiText evidence before the stop:
  - dense teacher PPL `20.781730576659253`, NLL `908271.3691734727`;
  - all-expert student PPL `20.782032063092196`, NLL `908275.7119984903`;
  - PPL ratio `1.0000145072823377`, inside the locked `[0.999, 1.001]` interval;
  - each model scored `299,357` causal tokens in `293/293` batches with zero failures.
- Hard stop: fixed-batch logit cosine `0.999959382` was below the locked `0.99999` threshold, raising `M0GateError`.
- Because discrepancy evidence was assembled only after the cosine gate, the failed attempt did not persist the fixed-batch input/mask hashes, finite flag, or maximum absolute logit difference. FineWeb held-out evaluation did not start and `stages/m0.json` was not written.
- Preserved evidence:
  - model manifest SHA256 `1B970B3FCD8B11E0553A7CE8B1743E0F0DE1EF15391906DDAA719CAB4D8707A5`;
  - conversion manifest SHA256 `20153B0229F69C8C380D608C7E757EAF29454C6A068A685572A4A51563284EA4`;
  - teacher checkpoint SHA256 `F088DA67ACAAAF21F30757AF1D703E9ECA40F9C067EB112E48FD1408DB03CC16`;
  - student checkpoint SHA256 `072508790AAFBAAC9AA55049E51D90589E4861491F596D0EFDEEBEBA587B84CE`.
- Diagnostic ruling: do not relax the cosine threshold. First isolate whether the discrepancy comes from the current eight-GEMM expert-sum all-expert path, physical-permutation accumulation order, or another conversion defect.

### 11:09–11:14 PKT — M0 BF16 accumulation-order diagnosis

- The first diagnostic instrumentation attempt was preserved as a failure artifact rather than silently retried:
  - `artifacts/phase1-v0/diagnostics/m0-accumulation.json`;
  - SHA256 `9072943B6FA67DDC79451BDD36D5B5D4F4D111A9B5111589119D8E7A9FCBC64C`;
  - it stopped before persisting comparisons because it queried PyTorch's internal tensor version counter inside `inference_mode` (`Inference tensors do not track version counter`).
- A corrected one-shot diagnostic used the same persisted WikiText batch 0 at microbatch 2 and transactionally persisted:
  - `artifacts/phase1-v0/diagnostics/m0-accumulation-v2.json`;
  - SHA256 `8065DD21BD6D399B6E2ACF1F11D5B3CBABFEAD11D7621498EE0DF7E87D23078B`;
  - input shape `[2, 512]`, 1,024 valid tokens; input-ID SHA256 `75FDFF750A22A8D923E251688FAA1225D185FEF1A4D6868F59E4088A55EFDF17`; mask SHA256 `E70C49B36DB298C56E85CFEA6CAB6E96432682F410729A73470A52813E3526E9`.
- Full-model fixed-batch comparison:
  - fresh unconverted teacher/student control: cosine `1.0`, zero max/RMS/NMSE difference;
  - current eight-expert down-GEMM sum: cosine `0.9999593822093301`, max absolute difference `0.5`, RMS `0.0355147380`, NMSE `8.1248690e-05`;
  - physically permuted monolithic down GEMM: cosine `0.9999615374834674`, max absolute difference `0.5`, RMS `0.0345793568`, NMSE `7.7025222e-05`.
- Layer-local comparisons used identical captured teacher MLP inputs for all 24 layers:
  - current expert sum worst cosine `0.9999963744972646` at layer 14; maximum absolute difference `0.03125` at layer 23;
  - monolithic permuted path worst cosine `0.9999999891801782` at layer 20; maximum absolute difference `0.015625` at layer 23.
- The temporary diagnostic changed no weights: 473 tensors and `852,985,920` elements retained identical metadata and sampled-value hashes before/after; methods were restored.
- Root-cause status: the checkpoint and conversion mapping are correct. BF16 neuron-order changes introduce very small per-layer differences that compound through the 24-layer autoregressive stack; eight separate expert GEMMs add some error but are not the sole cause. Next diagnostic: test an all-expert path that restores original neuron order at execution time while retaining physically permuted stored parameters.

### 11:17–11:32 PKT — Exact all-expert path confirmed and implemented

- V3 tested execution-time restoration of original neuron order while keeping all stored gate/up rows and down columns physically permuted:
  - artifact SHA256 `EB8D0E312132406CEF4894FE2DE50DE060BF2756FBE23E6CAEC66CCC94983926`;
  - full-model fixed-batch logits were bit-exact: cosine `1.0`, maximum/RMS/NMSE difference `0`;
  - all 24 layer-local MLP outputs had maximum/RMS/NMSE difference `0` on identical teacher inputs;
  - FP32 summation of already-rounded expert outputs did not improve the original failure.
- V4 tested independent FP32 expert down projections plus FP32 accumulation:
  - artifact SHA256 `69E95DA53CA0C2D1EA6B433CD1549CBE0FB119C2C71B5BC2B9DB6788A3C4E0E3`;
  - full-model cosine `0.9999621624291856`, below `0.99999`;
  - worst layer-local K=8 NMSE `5.345445260689922e-08`, still above the locked `1e-8` local gate.
- V5 additionally restored original neuron order within each expert's FP32 GEMM:
  - artifact SHA256 `63281695D9DA999AAEB291EA296E5C51FE84E39D6D5C27746D508B550C75610F`;
  - full-model cosine `0.9999607041815973`, below `0.99999`;
  - 0/24 layers met local NMSE `<=1e-8`; worst NMSE `5.3146541930675825e-08`.
- Conclusion: independent expert GEMMs are not bitwise decomposable from one dense BF16 GEMM because each partial result is rounded before summation. This is a numerical execution effect, not parameter corruption. The exact all-expert correctness path must globally restore original neuron order and use one down-projection GEMM; sparse selected-expert paths must retain their real independent-expert arithmetic so the oracle curve does not hide that cost.
- Implemented with strict TDD:
  - RED regression failed at `torch.equal` against the previous expert-sum path;
  - GREEN restores activation and down-column order through a non-persistent inverse-permutation buffer only when no selection or oracle transform is active;
  - selected-expert execution and oracle expert-output capture remain unchanged;
  - targeted regression passed; conversion module 6/6 passed; full suite 123/123 passed; `git diff --check` passed.
- Operational trade-off: the M0 correctness path materializes reordered activation and down-weight views/copies per FFN, increasing all-expert evaluation allocation/latency. It adds no parameters and is not used by selected sparse execution.
- Resume ruling: do not reuse the failed microbatch-2 M0 checkpoint because its student PPL was produced by the superseded expert-sum default path and the current checkpoint signature does not contain a code-version fingerprint. Preserve it as negative evidence and run the corrected M0 at microbatch 1, which has a clean signature-isolated checkpoint path.

### 11:39–11:43 PKT — Corrected real M0 PASS

- Fresh pre-run verification: `uv run pytest -q` passed 123 tests in 12.46 seconds; `git diff --check` exited zero.
- Executed exactly once in a clean signature-isolated attempt:
  - `uv run python scripts/run_phase1_v0.py --stage m0 --artifact-root artifacts/phase1-v0 --microbatch-size 1`;
  - exit code `0`; elapsed `176.5` seconds;
  - stage SHA256 `B9BE2ECFE2B73A906682B652D65CE1C9DB094E0D23ECC825AB00E677BD3A40D0`;
  - stage input signature `6fe6f78c9102389dc30f9f351b14af665358453b3948fcba23103a7a2c78abe2`.
- WikiText-103 raw test:
  - dense and converted all-expert PPL both `20.783743340940465`;
  - dense and student NLL both `908300.3612681422`;
  - PPL ratio `1.0`;
  - each scored `299,357` causal tokens in `586/586` completed microbatches with zero failures;
  - fixed-batch logits finite, cosine `1.0`, maximum absolute difference `0.0`.
- FineWeb-Edu held-out:
  - dense and converted all-expert PPL both `16.096086699454688`;
  - dense and student NLL both `277313.0170986966`;
  - PPL ratio `1.0`;
  - each scored `99,804` causal tokens in `196/196` completed microbatches with zero failures;
  - fixed-batch logits finite, cosine `1.0`, maximum absolute difference `0.0`.
- Evaluation peak CUDA allocation was `5,039,206,400` bytes for each corpus/model pass. Load-time evidence remained: free `12,197,036,032`, total `17,102,864,384`, peak `3,481,065,472` bytes.
- Parameter/conversion evidence remained exact: 24 text layers, 8 experts x 448 neurons, `852,985,920` parameters before/after, BF16 on `cuda:0`, layer-0 FP32 probe `7.450580596923828e-08`, and runtime vision-module identity guard passed.
- Clean checkpoint evidence:
  - WikiText signature `2bba489dc298633cf610e92c322ddd6cd901cd493357ee2820e1e1b2b5d00e1c`, identical teacher/student state SHA256 `70096E0B9396AD2EF1F04B840F01CA93C6B254E8990F27B6A59BE24F7860F0D7`;
  - FineWeb signature `9566b873ab64363458b3776f53bf7dfbb895b2cd88abaae733361580847dce9f`, identical teacher/student state SHA256 `D6EFDC609C96AEAE81B28E8CA2307D28660354183A25D9ADECB5DBB20092B58E`.
- The failed microbatch-2 checkpoint hashes and timestamps remained unchanged. Hugging Face cache delta was zero; only the known unauthenticated-request warning recurred.
- Gate decision: `GLOBAL M0: PASS`. This is the first real Phase 1 result: the fully converted parameter-preserving Qwen text stack matches the dense teacher exactly under the locked external and in-domain PPL pipelines.
- Next gate remains the 2,048-token local-oracle smoke. The K=8 local expert-sum numerical floor is deliberately not hidden by the exact M0 correctness path and must be handled as measured BF16 execution behavior, not parameter-conversion failure.

### 2026-09-01 — First-results checkpoint review closure

- A fresh read-only Terra review found no Critical implementation issues and confirmed:
  - physical gate/up/down permutation remains intact;
  - inverse permutation is a non-persistent long buffer with correct device migration and unchanged serialization;
  - selected-expert and oracle `output_transform` paths remain unchanged;
  - the restored-order default path is autograd-safe and adds no parameters.
- The review's sole Important finding was missing committed coverage at the real BF16/CUDA FFN dimensions. Added a focused production-shape test using hidden size 1024 and intermediate size 3584.
- Fresh local verification after the test addition:
  - production-shape CUDA/BF16 regression: 1 passed in 14.01 seconds;
  - Qwen conversion module: 7 passed in 3.09 seconds;
  - full suite: 124 passed in 14.61 seconds;
  - `uv lock --check`: 85 packages resolved, exit zero;
  - `git diff --check`: exit zero.
- Reviewer-noted pre-existing risks remain open for the next slice, not hidden by this checkpoint:
  - stage/checkpoint signatures do not yet include a source-code revision;
  - the local K=8 gate still assumes NMSE `<=1e-8` despite measured independent-expert BF16 arithmetic above that floor.

### 2026-09-01 — Local-oracle smoke gate corrected and passed

- The first CUDA smoke attempt exposed `NotImplementedError: "addmm_cuda" not implemented for 'Long'` in integer co-selection counting. The failed checkpoint preserved zero completed batches and zero direct checks.
- Replaced the unsupported integer CUDA matrix multiplication with boolean pairwise co-selection followed by a token reduction. A CUDA regression reproduced the failure before the fix and passed afterward.
- A complete two-batch smoke then measured K=8 dense-target NMSE between `3.7478717472499317e-06` and `5.77891082239999e-06` across all 24 layers. The old `1e-8` dense-target threshold rejected this otherwise complete result.
- Methodological correction: the written smoke requirement is consistency between K=8 selection and the literal all-expert independent-expert sum. It is not exact equality between that BF16 sum and the monolithic dense GEMM. The gate now:
  - checks literal K=8 expert selection and output equality token by token;
  - records a durable `all_expert_check_count` in checkpoints and requires it during report validation;
  - requires exact converted-layer x configured-K coverage, positive token counts, finite accumulated errors/norms/NMSE, a complete run, and positive direct/all-expert checks;
  - preserves the measured K=8 dense-target NMSE instead of thresholding or rewriting it.
- TDD and verification:
  - RED failures identified the missing result counter and obsolete `1e-8` gate;
  - local-oracle tests passed 10/10, reporting tests 9/9, and the full suite passed 125/125;
  - `uv lock --check` resolved 85 packages and `git diff --check` passed.
- Fresh clean production smoke:
  - command: `uv run python scripts/run_phase1_v0.py --stage local-smoke --artifact-root artifacts/phase1-v0 --microbatch-size 4`;
  - exit code `0`, elapsed approximately 9 seconds;
  - stage: `artifacts/phase1-v0/stages/local-smoke.json`;
  - checkpoint signature: `68d295cb74e2e15eded893cbbf816f9efe4370b72ecefa69ebb5ace8a21a0931`;
  - 1 completed batch, 0 failed batches;
  - 1,920 direct checks and 49,152 literal all-expert checks;
  - exact 24-layer x 5-K coverage (120 metrics);
  - K=8 dense-target NMSE remained `3.7478717472499317e-06` to `5.77891082239999e-06`.
- The stage did not record a peak-CUDA field. An unauthenticated Hugging Face Hub rate-limit warning recurred but did not affect cached data or execution.
- Decision: local-oracle smoke passes. Proceed to the locked 100,000-token calibration run without tuning a post-hoc numerical tolerance.

### 2026-09-01 — Full 100K-token local oracle PASS

- Executed `uv run python scripts/run_phase1_v0.py --stage local --artifact-root artifacts/phase1-v0 --microbatch-size 4`.
- Exit code `0`; elapsed `32.5` seconds; no CUDA/OOM failure.
- Stage: `artifacts/phase1-v0/stages/local.json`.
- Checkpoint signature: `68d295cb74e2e15eded893cbbf816f9efe4370b72ecefa69ebb5ace8a21a0931` under the full-local checkpoint namespace.
- Completion evidence: 49/49 batches, zero failures, exact 24-layer x 5-K coverage, and 100,000 tokens for every layer/K metric.
- Validation evidence: 1,920 direct checks and 2,400,000 literal all-expert checks.
- Aggregate normalized error over all layers:
  - K=8: `4.464879674258729e-06`;
  - K=6: `0.09733733576993434`;
  - K=4: `0.2485731050105692`;
  - K=3: `0.35453659422814027`;
  - K=2: `0.49269984905605735`.
- Interpretation: the all-expert point remains at the measured BF16 independent-sum floor, while removing 25%, 50%, 62.5%, and 75% of experts produces progressively larger local reconstruction error. These are measurements, not yet end-to-end sparse-model quality claims.
- Peak CUDA was not recorded by this stage. The unauthenticated Hugging Face warning recurred without affecting execution.

### 2026-09-01 — Full rollout oracle PASS with negative sparsity result

- Executed `uv run python scripts/run_phase1_v0.py --stage rollout --artifact-root artifacts/phase1-v0 --microbatch-size 4`.
- Exit code `0`; elapsed `542.4` seconds. Peak CUDA allocation was `8,699,603,456` bytes; no OOM occurred.
- Stage: `artifacts/phase1-v0/stages/rollout.json`.
- All 10 dataset/K settings completed with zero failures and complete 24-layer intervention coverage.
- WikiText-103 raw test: 147 batches and 299,357 scored tokens per K.
  - K=8: PPL `20.7820`, aggregate hidden NMSE `0.000132`;
  - K=6: PPL `29.0723`, aggregate hidden NMSE `0.119446`;
  - K=4: PPL `54.1177`, aggregate hidden NMSE `0.278846`;
  - K=3: PPL `103.0113`, aggregate hidden NMSE `0.393997`;
  - K=2: PPL `515.6701`, aggregate hidden NMSE `0.547125`.
- FineWeb-Edu held-out: 49 batches and 99,804 scored tokens per K.
  - K=8: PPL `16.0964`, aggregate hidden NMSE `0.000119`;
  - K=6: PPL `22.2418`, aggregate hidden NMSE `0.111854`;
  - K=4: PPL `39.6923`, aggregate hidden NMSE `0.266938`;
  - K=3: PPL `69.5593`, aggregate hidden NMSE `0.380337`;
  - K=2: PPL `282.0449`, aggregate hidden NMSE `0.533937`.
- Interpretation: the rollout mechanism and coverage gate pass, but the scientific result is negative for naive fixed-K local-oracle sparsification. Even K=6 materially degrades both external and in-domain PPL; degradation accelerates at lower K. This result must be preserved and should constrain any learned-router work rather than be tuned away.
- The unauthenticated Hugging Face warning recurred without affecting cached inputs or the completed run.

### 2026-09-01 — Phase 1A.5 direction approved for specification

- The Phase 1A random-partition rollout result triggered the plan's decomposition-improvement fork rather than router training.
- A proposed sparse reconstruction oracle was reviewed and approved with methodological corrections:
  - separate monolithic dense D0, literal eight-expert E8, and reconstructed R8 references;
  - scale ridge regularization by subset Gram trace;
  - use stable, cutoff-recorded free reconstruction with sampled direct FP64 validation;
  - include unit, ridge, bounded `[0,2]`, and free oracle families at K in `{4,6}`;
  - select conditions using calibration only and freeze them before WikiText/FineWeb-heldout evaluation;
  - use teacher-next-state correction Reconstruction-H during rollout;
  - preserve the free solution as a non-deployable mathematical upper bound;
  - measure D0/E8/R8 distances so BF16 decomposition error is not mislabeled as sparsification error.
- The approved architecture is recorded in `docs/superpowers/specs/2026-09-01-phase1a5-sparse-reconstruction-oracle-design.md`. Implementation remains gated on final review of the written spec.

### 2026-09-01 — Phase 1A.5 approved for implementation

- Final spec review approved implementation after four refinements, all incorporated before planning:
  - renamed the numerical diagnostic to `Registered Free-Reconstruction Candidate Bound` and stated the direction of its error/quality bound explicitly;
  - unit regression uses relative tolerance `1e-10` only for identical stored sufficient statistics and `1e-8` for recomputed model outputs, while retaining exact untied subset matching on the smoke set;
  - ridge near-zero scale uses registered `epsilon_zero=1e-15` relative to `max(||y||^2,1)`;
  - bounded `[0,2]` reports lower/upper saturation rates at `1e-6` and is practical independently of the negative-rate diagnostic.
- Added free-reconstruction gain recovery as an analysis-only metric; it cannot affect the pre-registered 30% eligibility gate.
- Implementation plan written at `docs/superpowers/plans/2026-09-01-phase1a5-sparse-reconstruction-oracle.md`. It separates protocol provenance, solver core, certified bounded QP, local calibration/selection, orchestration, Reconstruction-H rollout, and hashed reporting into major reviewed checkpoints.

### 2026-09-03 — Phase 1A.5 Task 7 source complete; 2,048-token smoke stopped before scoring

- Reporting implementation checkpoint: `88d0e8e feat: add phase 1a5 reporting`. Fresh verification before this production attempt: `uv run pytest -q` passed `230/230`; `uv lock --check` resolved the locked 85 packages; `git diff --check` passed.
- Source command: `uv run python scripts/run_phase1a5.py --stage source --artifact-root artifacts/phase1-v0_5 --microbatch-size 1`.
  - Exit code `0`; source-stage signature `7d41b9700d33c9151b1cc1aca325187acd41514d5ceada847f50a4fcb52d832b`.
  - `artifacts/phase1-v0_5/stages/source.json` records code revision `88d0e8ebe4469d0ece42932c877df361dd1244c5` with `dirty: false` and the frozen Phase 1 source/dataset/model hashes.
- Smoke command (one permitted execution): `uv run python scripts/run_phase1a5.py --stage smoke --artifact-root artifacts/phase1-v0_5 --microbatch-size 1`.
  - Exit code `1`; wall time approximately `9.6` seconds including cached model loading. The environment emitted an unauthenticated Hugging Face Hub warning but loaded all 473 cached weight shards for each model.
  - Failure occurred before unit regression completed and before any family timing, CUDA peak, SVD, bounded-KKT, feasibility, completed-batch, or failed-batch aggregate could be persisted: `RuntimeError: Inference tensors cannot be saved for backward` in `PartitionedQwen35MLP.expert_outputs` while `_default_unit_regression` fed a tensor captured under `torch.inference_mode()` into a student linear layer outside an inference/no-grad context.
  - Durable artifact state after failure: only `source-manifest.json` and `stages/source.json` exist under `artifacts/phase1-v0_5`; no smoke stage or smoke checkpoint exists. Therefore completed/failed batch counts and solver/KKT/SVD evidence are unavailable, rather than zero or passing.
- Decision: the smoke gate is **not passed**. No retry, tolerance change, microbatch change, 100K local calibration, selection, rollout, or report execution was performed. A separately reviewed regression fix must make the unit-regression capture/consumer autograd modes compatible and prove the failure path with a test before a new source signature and smoke attempt.

### 2026-09-03 — Phase 1A.5 Task 7 smoke-failure hardening (no production rerun)

- Root cause correction: `_default_unit_regression` now keeps the captured teacher inference tensors, student `expert_outputs`, and both unit-oracle calls inside one `torch.inference_mode()` scope. A regression test uses a captured inference tensor with a trainable `torch.nn.Linear`; it reproduces the old autograd-save failure outside the scope and passes with the corrected scope.
- Smoke failures are now durable, source-native evidence at `artifacts/phase1-v0_5/failures/smoke/<execution-signature>.json`. The immutable artifact records status, source/execution signatures, source hashes, exception type/message, elapsed time, peak CUDA when available, and known batch counts; a failed attempt never creates a successful smoke-stage file and a matching later invocation cannot overwrite the evidence.
- Smoke/report gates now reject incomplete family measurements, invalid or missing bounded per-K KKT diagnostics, nonphysical metrics/counts, missing/duplicate/unreconciled local metrics, unobserved selected conditions, and unsupported bounded selection. Reporting also refuses to fabricate subset stability and requires persisted D0/E8/R8 distance evidence.
- Verification only (no source/smoke/local/select/rollout/report production command): focused Task 7/orchestration tests passed `26/26`; full `uv run pytest -q` passed `237/237`; `uv lock --check` resolved 85 locked packages; `git diff --check` passed.
- The original smoke failure above remains the authoritative execution evidence. This correction has not rerun the frozen protocol and does not change the current smoke-gate status.

### 2026-09-08 — Phase 1A.5 isolated fresh source and 2,048-token smoke PASS

- Authority and provenance: the fresh attempt used the previously absent explicit run root `artifacts/phase1-v0_5/runs/a8ab03b`, beneath the registered Phase 1A.5 artifact tree. This was necessary because the fixed root retains superseded source/smoke/failure evidence. The log is a post-evidence checkpoint only: the source manifest was created from a clean `exp` worktree at code revision `a8ab03b97d2244ef9965396b7858719d76586d09`, before this log edit.
- Prior fixed-root evidence was rehashed before and after the run and remained byte-identical: source manifest `72091594791EA7E25335B493B7642EFE9673ED08B7A920B7F0A31AD594CD238C`; source stage `31D237BC3B0491BB25DC5D455F5C6C1BEB40A6AEDF0FD97F27F4A422E1D7721C`; smoke stage `6969A48CDDE8ECFF53F1E442F016A70BF108CA20EAA9BA3D7108EAACB097653F`; prior failure `45B5003E2E935E614C1537BDA2CD468AA8021C20E1F2EC78DB625B2D12688A80`.
- Commands: `uv run python scripts/run_phase1a5.py --stage source --artifact-root artifacts/phase1-v0_5/runs/a8ab03b --microbatch-size 1`, followed once by the identical smoke command. The source input signature was `e72e8ae835c99d68fe38b06f094daf022fa2ccc673852a975a01c1408e51a891`; the smoke input signature was `538d2ff4601db7f047a31fd03fa66a3bb15768a0c0418e2d41067af3b09f8645`.
- Environment: Python `3.11.9`, Torch `2.13.0+cu130`, CUDA runtime `13.0`, NVIDIA GeForce RTX 5060 Ti with BF16 support. The smoke wrapper exited zero in `90.3` seconds. Its four persisted family measurements each processed exactly 2,048 tokens in four completed batches with zero failures: unit elapsed `5.692862` seconds, peak CUDA `3,717,508,608` bytes, projected 100K `277.971777` seconds; ridge `35.042496`, `3,724,376,064`, `1711.059375`; bounded `16.091246`, `5,751,445,504`, `785.705347`; Registered Free-Reconstruction Candidate Bound `25.165008`, `5,763,927,040`, `1228.760181`.
- Unit regression passed exactly for 2,048 tokens / 49,152 token-layers. Bounded feasibility passed with all candidates certified: K4 `3,440,640/3,440,640`; K6 `1,376,256/1,376,256`; zero bounded solver failures, uncertified candidates, and coefficient-bound violations; zero R8 face-enumeration fallbacks. Free-bound SVD evidence passed all required identities: sparse K4, sparse K6, and R8 each had 384 checks, 16 for each of 24 layers, literal rank agreement, and finite maximum relative objective error respectively `1.4686326730896531e-16`, `6.934876318317636e-17`, and `2.873959379286303e-19` (all <= `1e-9`).
- Persisted artifact-contract audit passed. New-run SHA256: source manifest `A82B6BC7E50BD8627852829B6C838419DF6F47E7E79512A048E12D03B09A6060`; source stage `264E7D7FA264CF25A4BA565DE2D68CDAA827EF7864D7EC62D3068FAE1B78A618`; smoke stage `713CFE235AA460C66E5DF6F13EDD3D85D948180CA836478913F73592FF8564D0`.
- Gate decision: source and 2,048-token smoke pass the frozen unit, runtime, bounded-certification, and free-bound SVD gates. This does not run or decide the 100K calibration, selection, rollout, or report stages; no later production stage was invoked.

### 2026-09-08 — Phase 1A.5 100K local complete after one recovered CUDA fault; registered no-go

- The first local command was `uv run python scripts/run_phase1a5.py --stage local --artifact-root artifacts/phase1-v0_5/runs/a8ab03b --microbatch-size 1`. It durably completed batches `0..73` (37,888 valid tokens), then batch 74 stopped with `AcceleratorError: CUDA error: misaligned address`. No local stage was published by the failed process.
- Recovery implementation `20cd44b87ef3ccc14ef9f4ea9016cd8d758ef7fa` added fresh-process-only retry semantics: unresolved failures remain recorded, a successful atomic retry clears only the unresolved marker, and immutable `failure_history` remains in the result. Focused tests passed `27/27`; the full suite passed `306/306`; lock and diff checks passed.
- The continuation used the same exact local command in a fresh detached process with isolated logs. It loaded the existing run signature/checkpoint, recovered batch 74, and continued through batch 195. The authoritative local stage contains exactly 100,000 calibration source tokens, 196 completed batches, zero unresolved failed batches, one recovered failure attempt, and the original CUDA exception in `failure_history`.
- Provenance qualification: batches `0..73` were produced by the source-manifest implementation `a8ab03b`; batches `74..195` and finalization used recovery revision `20cd44b`. The scoring/oracle mathematics and frozen inputs were unchanged; only checkpoint failure-state recovery changed. This mixed recovery revision is explicitly recorded and must not be presented as a clean single-revision rerun.
- Local-stage SHA256: `5C48266C4CB5429E9B8AA3544B7AFE112EF3B198C6084CC02921AC0C7DA472B1`. SVD evidence remained complete: K4, K6, and R8 each recorded 384 checks, 16 per layer, literal rank agreement, and maximum relative objective errors `1.4686326730896531e-16`, `6.934876318317636e-17`, and `2.873959379286303e-19`.
- Exact pooled calibration results relative to the same unit condition:
  - K4 unit NMSE `0.248572297616`; best free-bound NMSE `0.207160322401`, relative improvement `0.166599317834` (16.66%). Bounded improvement was `0.166596436069`.
  - K6 unit NMSE `0.0973403259296`; bounded/free-bound NMSE `0.0877054923237`, relative improvement `0.0989809055377` (9.90%).
  - Coefficient diagnostics were sane: negative rate was zero for all conditions; maximum registered above-two rate was `3.78125e-05` for K4 free-bound and zero for K6.
- The immutable selection stage returned `scientifically_stopped` with reason `selection_no_eligible_conditions`. No practical or free-bound condition met the frozen 30% pooled-NMSE reduction gate. Selection manifest SHA256: `628E3E9996C2A525825C818ABD27D7273397C039A11663816DEB9AB66F579CE0`; select-stage SHA256: `0A493DD63829961DF1BC8CA4AB602C7A0D710CDC442AD0F4357509FD03CBB739`.
- Per protocol, no WikiText/FineWeb-heldout rollout or PPL evaluation ran after the no-go selection.
- Report generation exposed two representation-only validator bugs: uppercase/lowercase hexadecimal digest comparison and whole-valued float encodings for bounded maximum counters. Both were corrected with focused reporting tests (`51/51`) without changing any experiment artifact.
- Final report decision: `No registered numerical reconstruction qualifies; proceed to Phase 1B.` Report manifest SHA256: `E54A438A03ADECA3FC24F8011847174A57D17EAE3BEA2028A480E64450CFFCCA`; report Markdown SHA256: `2A52DF6442ED3C938EA4C3AE82BD9C8A8B2379815E3D71E821314889BDC34D3D`.

### 2026-09-08 — Research direction narrowed; recurrence planning only

- User requested selecting the most promising directions, updating a detailed implementable plan and adding a shorter todo list. Their latest objective is affordable improvement of a pretrained model's capability, without prompt manipulations; offloading is assigned to ExpertFlow.
- Read current plans, code interfaces, logs and persisted scientific results. The workspace initially had clean main and exp at the same commit `0c7ed38`. Switched to existing exp before document edits; main was not edited. No experiment or source implementation ran during this planning revision.
- Selected one bounded recurrence study with two linked components: fixed frozen-loop diagnostic controls, followed by shared-block LoRA adaptation compared against ordinary LoRA and an extra-training-time control. Conventional Qwen3-1.7B is specified instead of the hybrid Qwen3.5 model to simplify repeated-layer state semantics. This is a new capability track; additional inference computation is explicitly allowed and measured.
- Archived the original roadmap intact apart from its historical-status banner at `docs/archive/2026-08-30-dense-to-sparse-plan.md`. Replaced root `plan.md` with explicit interfaces, task steps, data-generation rules, source pinning, causality/gradient tests, runtime bounds, failure semantics, six-arm maximum, final-test sealing and quantitative decision gates. Added matching `docs/superpowers/specs/2026-09-08-pretrained-recurrence-design.md` and concise `todo.md`.
- The Phase 1A.5 scientific no-go and all historical artifacts remain unchanged. Its historical recommendation to proceed to Phase 1B is superseded only as the active project queue: clustering and sparse recovery are now deferred. This is not a claim that sparse conversion is impossible.
- No native-MoE sweep, expert merging, specialist assembly, custom pretraining, token-level loop selector, quantization or serving work is part of the new queue. The specialist-first proposal is discussed separately as a possible future shared-backbone adapter study; not selected for implementation.
- New protocol choices are planning assumptions, not measured optima or reproductions of published headline results. The 24-hour cap, model size, rank, loop range, synthetic tasks and promotion thresholds must be checked through the specified gates. No model download, source revision pin, training-memory feasibility, recurrence quality or runtime gain has yet been established.
- Next step when execution is requested: implement Task 1 of the new plan, then sequential correctness/resource gates. Scientific failure or a resource stop ends this study and produces a report; it does not automatically activate another research idea.
- Documentation verification: archived roadmap matches HEAD's original plan after removing only the added historical banner; local Markdown links and code fences pass; `git diff --check` passes. Recomputed the four principal Phase 1A/1A.5 stage hashes and confirmed the registered values. Reviewed spec/plan agreement, added an explicit requirement to beat the original F0 model as well as adapted controls, and specified causal label shifting. No application tests were rerun because this change contains documentation only. Changes remain uncommitted on exp.

### 2026-09-11 — Recurrence execution order expanded and review gates hardened

- User approved a separate exhaustive execution plan while retaining `plan.md` as the scientific protocol and `todo.md` as the short status view. Added `docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md` with a strict Task 0-23 order from protocol freeze through implementation, G0-G6 production execution, evidence reporting and closeout.
- Execution is inline and automatically advances through passing persisted gates. It pauses only for genuinely human-only blockers such as credentials/license acceptance, physical intervention, unavailable external resources or conflicting user-owned edits. Scientific, retention, timing, budget and repeated-environment stops produce an honest report and end the study without ad-hoc retuning.
- User selected milestone commits between per-task and monolithic extremes: protocol freeze; foundation; model/training; evaluation/orchestration; and observed-evidence closeout. Generated artifacts, weights, scores and checkpoints remain uncommitted.
- Incorporated the prior review findings before protocol freeze. G3 now requires at least two informative task families, where F0 family accuracy is strictly between 5% and 95% and invalid-format rate is at most 10%. All families remain scored and reported.
- FineWeb retention now persists per-chunk NLL sums and uses 10,000 paired chunk-bootstrap replicates with seed 9157. Discovery and confirmation require both the A2/F0 PPL point estimate and the 97.5 percentile ratio to be at most 1.02; a failed interval is reported as retention uncertainty rather than hidden or retuned.
- No source implementation, model download, training or evaluation was performed while writing this execution plan. All new checkboxes and commands remain planned work until their specified artifacts and verification results exist.

### 2026-09-19 — Recurrence Task 16 G1 smoke binding implemented; literal run blocked before CUDA

- Implemented the smallest production Task 16 binding on `exp`: `default_dependencies()` now registers the local pinned-source smoke worker and the frozen fresh-process dispatcher `ppmoe.recurrence.smoke:run_child_smoke`. The worker uses only the finalized source snapshot, eight deterministic dev prompts (3 expressions, 3 lists, 2 graphs), BF16 eager/cacheless Qwen3, stock-wrapper identity thresholds, adapter-off A1/F0 and A2/F1 checks, shared-layer/call-count/causality/cache checks, disposable LoRA gradient/suffix checks, synchronized CUDA timing, VRAM measurements, and the append-only CUDA ledger.
- TDD evidence: the new smoke binding/retry contract passed; focused recurrence suites passed `59`, with `1` expected skip. The existing recurrence model/LoRA/orchestration/CLI tests remained green. No sealed `data/test.jsonl` content was opened.
- Exact production command: `uv run python scripts/run_recurrence_v0.py --stage smoke --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0 --run-id 27b3645f7761`; exit `1`, before worker launch, with `predecessor signature mismatch with current config/code`.
- Immutable predecessor evidence remains unchanged: `stages/data.json` is `passed` with code SHA256 `fbad310b761cc009d5bc438d4fdb983e34382b5548c95d9ce777b16b5ea6e029`. The current implementation code SHA256 is `12f7302f7328c1cb40a9f0230b4fd388e28c36b29d5d19d63ea31ddc1b3596ea`; it cannot impersonate the sealed Task 15 revision. No CUDA execution occurred and no ledger entry was fabricated. Detailed attempt evidence is `artifacts/recurrence-v0/27b3645f7761/task16-report.json`.
- Decision: Task 16 reaches its registered implementation/provenance terminal branch before G1 measurements. Do not relax signatures or overwrite the existing smoke failure record; a fresh source/data finalization under the new code signature is required before a real-model smoke can run.

### 2026-09-19 — Task 16 implementation review blockers corrected (no production rerun)

- Corrected the four review blockers without touching the literal-run signature-fail evidence: smoke CUDA-fault events now carry and verify a canonical event ID; disposable empty-checkpoint proofs use identical empty fields before/after the child; model loading and all CUDA setup are inside the one-fresh-process fault boundary with synchronized elapsed time and peak allocated/reserved/total VRAM persisted on failures; and retry postflight uses authenticated child VRAM measurements for smoke.
- Added focused tests for exact emitted fault events, canonical-ID rejection, end-to-end disposable-child proof reconciliation, pre-model-load CUDA recovery, and child VRAM resource gating. Focused recurrence suites: `62 passed, 1 skipped`; full suite: `637 passed, 2 skipped`; compileall and `git diff --check` passed.
- Current implementation code SHA256 is `7bf2b83865485744adeadab2145059270e6e40f288d8703f9fed3ec128ca4552`; sealed Task 15 remains `fbad310b...`. No GPU production command was rerun, no sealed test data was opened, and no commit/push was performed.

### 2026-09-19 — Task 16 rereview follow-ups closed (literal run remains signature-blocked)

- Closed the rereview requirements in the uncommitted Task 16 implementation: descriptor/reparse-safe retry evidence reads; exact eight dev IDs/content bound to the immutable manifest and consumed-byte hash; authenticated full child payload including total/peak VRAM; a frozen non-empty production dispatcher digest; final success-ledger ordering plus correctness-failed attempt records; explicit selected layer/projection/parameter IDs; per-adapter finite and B-gradient-nonzero evidence with the LoRA B-zero initialization nuance; suffix finite/nonzero gradients; unchanged adapter hash and no-checkpoint evidence; strict zero-tolerance causality; and strict smoke payload validation.
- Added a production-shape fake-child regression that computes a valid payload digest and then tampers `peak_reserved_bytes`; parent validation rejects it before postflight. Immutable dev-row mutation and payload-schema regressions are also covered.
- Focused recurrence tests: `67 passed, 1 skipped`; full suite: `642 passed, 2 skipped`; compileall and `git diff --check` passed. No production GPU command was rerun, no sealed `data/test.jsonl` was opened, and no commit/push was performed.
- Current implementation code SHA256 is `4794dbb25b2be74ebd2717f5fd7f4a8f54b273e155dd347f6ee943e55fcc95c6`; current config SHA256 is unchanged at `483fb1bdce6adaa26f5ad5a314fb3b4ded2cf4277e7c19890f1ffed320f10e11`; frozen child dispatcher identity is `6d16e6aa7480b62edb7d829155415fe04fbbd9032011280ad59686380d462877`. Sealed Task 15 code remains `fbad310b...`, so the literal run stays at the registered predecessor-signature terminal branch. Tasks 17–21 remain mapped to unavailable production workers in `task16-rereview-report.json`.

### 2026-09-19 — Task 16 rereview-2 report contract closed (no production rerun)

- Registered `stage_complete` in the report event contract with the exact `{index,timestamp,kind,stage,attempt,payload,outcome}` shape, preserving existing G0 source/data events and Task 16 smoke success. Smoke CUDA and correctness failures now use registered `stage_attempt` records; canonical event IDs, CUDA fault flags, timing, VRAM, error type, and messages are nested inside the payload, and retry reconciliation reads those nested diagnostics.
- Added report-loader and `build_report` integration coverage for source/data/smoke stage completion, CUDA-fault and correctness-failure attempts, plus malformed extra-field rejection. Added explicit selected-projection invocation identity checks across single/repeat recurrence passes.
- Focused recurrence suites: `111 passed, 1 skipped`; full suite: `648 passed, 2 skipped`; compileall and `git diff --check` passed. No production GPU command was rerun, no sealed `data/test.jsonl` was opened, and no commit/push was performed.
- Current implementation code SHA256 is `1fb2966ff0946214421804bbf9faf17a61294e4c2319eefc5a4da985e571f329`; frozen child dispatcher identity remains `6d16e6aa7480b62edb7d829155415fe04fbbd9032011280ad59686380d462877`. Literal RUN_ID `27b3645f7761` still requires a regenerated G0 under this new code hash; prior signature-fail evidence and Task 14 notes remain preserved.

### 2026-09-19 — Task 16 rereview-3 vacuity blocker closed (no production rerun)

- Strengthened `_validate_smoke_payload` from boolean/key presence checks to structural and numeric recomputation: eight finite identity rows and exact thresholds/aggregates; exact 28/32 layer maps and totals; four selected layers, 24 projections, 48 parameters, and 24/48 invocation IDs proving reuse; 224 cache observations; three zero causality metrics; adapter-off finite/diff measurements; all 24 per-adapter gradient entries with finite/nonzero B norms and the documented A-zero nuance; suffix norm; and valid equal adapter hashes/no-checkpoint proof.
- Added TDD regressions rejecting a payload with every gate `True` but empty/fabricated evidence, plus a complete structural fixture accepted by the validator. The authenticated child path now cannot seal a vacuous smoke payload.
- Focused recurrence suites: `113 passed, 1 skipped`; full suite: `650 passed, 2 skipped`; compileall and `git diff --check` passed. No production GPU command was rerun, no sealed `data/test.jsonl` was opened, and no commit/push was performed.
- Current implementation code SHA256 is `8f63aa74d2281780e4acf7938ceb3a8323adb99d01080f5c9697d0702764187c`; frozen child dispatcher identity remains `6d16e6aa7480b62edb7d829155415fe04fbbd9032011280ad59686380d462877`. Literal RUN_ID `27b3645f7761` remains predecessor-signature blocked and requires regenerated G0 evidence.

### 2026-09-19 — Task 16 rereview-4 projection identity split closed (no production rerun)

- Separated the genuine smoke payload's two identity domains: `selected_base_projection_ids` and `selected_projection_invocation_ids` now describe the original pre-LoRA `nn.Linear` objects and exact module-name/object-ID reuse across single and repeat passes; `selected_projection_ids` now describes post-install LoRA wrapper IDs plus their original base IDs, while parameter IDs remain deterministically bound to the same layer/projection names.
- Strengthened validation to recompute exact 24-row base/post-install maps, 24/48 invocation cardinalities and counts, base-object reuse, wrapper/base distinction, unique base-weight and adapter parameter identities, and the derived sharing gate. Added TDD regressions rejecting swapped pre-LoRA invocation IDs and fabricated post-LoRA base identities.
- Focused recurrence suites: `115 passed, 1 skipped`; full suite: `652 passed, 2 skipped`; compileall and `git diff --check` passed. No production GPU command was rerun, no sealed `data/test.jsonl` was opened, and no commit/push was performed.
- Current config SHA256 is `483fb1bdce6adaa26f5ad5a314fb3b4ded2cf4277e7c19890f1ffed320f10e11`; current implementation code SHA256 is `c29803d3535b0d0652e078546394a4946200708ef10f485cd292e6c5dafdb863`; frozen child dispatcher identity remains `6d16e6aa7480b62edb7d829155415fe04fbbd9032011280ad59686380d462877`. Literal RUN_ID `27b3645f7761` remains predecessor-signature blocked; updated evidence is `artifacts/recurrence-v0/27b3645f7761/task16-rereview-report.json`.

### 2026-09-25 — Recurrence V0 production recovery and registered G3 stop

- Implementation milestones were completed and pushed on `exp`; the final scientific run used committed implementation `64c80f7`, code SHA256 `687dd74e9fb942a06dc0efe6017b1852891e759890e175e4e4ddce16c221658e`, configuration SHA256 `483fb1bdce6adaa26f5ad5a314fb3b4ded2cf4277e7c19890f1ffed320f10e11`, data SHA256 `ea68abe5aef3d5023625c149f75c39f80a3b685720c3385c4fef2fb6a9f220bc`, and pinned Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Model source, data, score, and report artifacts remain ignored, local, and immutable; no merge to `main` was performed.
- Earlier signed run `c4e4f1950a58` stopped at G1 because the smoke validator expected 224 cache observations instead of the actual 244. Its G6 attempt then failed because the report resolved source-manifest paths from the run root instead of the pinned snapshot. That report inventory also opened `data/test.jsonl` bytes for hashing before confirmation, although no answer was displayed or evaluated. This was a protocol deviation, not a valid sealed evaluation. Commit `fc01f4f` fixed the count; `a64f2dd` fixed source-path resolution and made reports exclude sealed test rows entirely.
- Recovery run `05b9deb9c2a1` passed G0/G1 but stopped at G2: the runtime passed the recurrence wrapper, not its stock Qwen model, to the LoRA installer. Its G6 report passed with `failed_environment`; commit `f0d0673` corrected the two call sites with real tiny-Qwen3 CPU regressions. Recovery run `ac55e11744fa` passed G0/G1/G2 and completed raw F0/F1/F2 diagnostics, but G3 failed parent validation after the runtime moved F1/F2 summary sidecars to `.evidence`; its G6 attempt also failed because partial F0 scores lacked an authorized completed G3 producer. Commit `64c80f7` retained G3 sidecars and made failed-stage reporting hash/reconcile partial scores without treating them as scientific aggregates. None of those terminal stage records was overwritten.
- Final literal run `129d39646933`: G0 source/data, G1 real-model smoke, and G2 pilot passed; the pilot froze `n3=1264` with peak reserved VRAM `7023362048/17074421760` bytes. G3 completed all three 600-example development score sets and 40 retention chunks per mode, then sealed `stopped_scientific: uninformative_benchmark`. F0 was 4/600 correct: expressions 0/200, graphs 4/200 (128/200 invalid format), lists 0/200. No family met the registered informativeness gate. F1 was 0/600 and F2 9/600, but these frozen diagnostics do not establish adapted-recurrence efficacy. G4 discovery, G5 seed-48 training, and sealed test evaluation were not run.
- G6 for `129d39646933` passed and preserved `report.json`, `report.md`, and `report-sources.json`. Independent read-only checks matched all 74 inventoried file sizes and SHA256 hashes (sealed `data/test.jsonl` absent), exact-score counts/family rates, F1/F2-vs-F0 transitions, 40-chunk perplexity ratios, and 10,000 paired-bootstrap 95% intervals. F1/F0 PPL ratio was `1.1384609262` (CI `1.1211337864–1.1569256203`); F2/F0 was `1.0113661479` (CI `1.0071742670–1.0155922018`).
- The immutable G6 Markdown conclusion overclaims that recurrent LoRA missed a quality gate even though LoRA arms never trained. The authoritative interpretation is **benchmark calibration failed; recurrence efficacy is not evaluated**. A TDD-tested report-template correction and a separate erratum are recorded in `docs/recurrence-v0-report-erratum.md`; neither changes the signed G3 decision or rewrites G6 artifacts. The registered stop ends this V0 study without prompt/threshold retuning or automatic activation of another direction.

### 2026-09-25 — Authorized longer-output and model-review diagnostic

- User authorized increasing the answer allowance and an editable good/bad review file, then explicitly authorized GPT-6 Luna to perform the first-pass answer review. Luna labels are model judgments, not human judgments. The detailed bounded addendum is `docs/recurrence-review-plan.md`; V0 and its stopped gates remain unchanged.
- Read-only reinspection of F0's 600 raw development records found 468 `missing_eos_at_cap`, 128 `invalid_format`, and 4 `correct`. The 32-token allowance assumed compact canonical answers; actual generations often explain first. The saved development prompts have maximum 119 tokens, so 256 output tokens fit the existing 512 total bound without changing the wrapper or training limits.
- Implemented a separate blank Markdown review export and grade-only validator using Luna, with full original-record provenance, source hash checks, development-only input restrictions, and explicit reviewer identity in summaries. A real 600-row legacy export caught a trailing-whitespace roundtrip bug; regression and fix preserve the original response exactly. Independent Luna code review approved the revised tool after provenance-chain checks. A separate Luna copy will be graded only after longer outputs exist; no human grades are inferred.
- The new probe is baseline F0 only, ten deterministic development examples per family, 256 output tokens, and a 3600-second deadline. It reuses the pinned local model and unchanged prompts. This is benchmark calibration on reused development data, not an independent performance improvement result, new training, or a replacement V0 gate.
- Verification before inference: focused probe/evaluator/review tests `56 passed, 1 skipped`; full project suite `721 passed, 3 skipped` in 63.32 seconds; compileall and diff checks passed. Actual-source read-only preflight verified the pinned snapshot and 600 development records and selected 30 examples. No sealed test rows were opened by this diagnostic.
- Probe implementation was committed as `2c6904f` on `exp`. The bounded real-model probe completed at `artifacts/recurrence-review-v1/baseline-256` in 104.0746 seconds: 30/30 selected development rows, 3,982 generated tokens, 24 EOS completions and 6 length-capped completions. All six capped responses were list problems; peak reserved GPU memory was 8,573,157,376 of 17,074,421,760 bytes. The saved `answers.jsonl` SHA256 is `839d644adac0a9749cc1f66883b61cd21226084d9bbf7575ed5f7de6df6c7e9d`.
- Every sampled V0 answer's first 32 token IDs matched the new answer's prefix (30/30), supporting the controlled output-length comparison. On these same IDs, V0 had 23 `missing_eos_at_cap` and 7 `invalid_format`. With 256 output tokens, all 10 expressions, all 10 graphs, and 4 lists reached EOS, but **all 30 still fail the registered strict parser** due to verbose formatting. The new diagnostic did not apply V0's gate or relabel its old run.
- GPT-6 Luna (`/root/luna_answer_review`) graded a separate editable copy as model review: **18 good, 12 bad, 0 pending**. By family: expressions 10/10 good, graphs 8/10, lists 0/10. All six length-capped responses were bad. The blank human-review copy still has 30 unreviewed rows. The sheet validator confirmed that only grade cells changed; `luna-review/model-review-summary.json` binds the answers, review, manifest, and blank human sheet to SHA256 hashes, with reasoning notes in `model-review-notes.md`. One graph row had a correct final yes/no answer but an invalid path explanation and was graded good under the stated final-answer criterion; its note flags the criterion sensitivity.
- Interpretation: the 32-token cap and strict output format substantially understated answer-level performance on this small development sample, while list tasks remain poor and still truncate. This finding motivates a fresh, objectively gradable benchmark and prespecified revised gates before any LoRA training. It is not a human evaluation, an independent test, or evidence of recurrence efficacy; V0's scientific stop remains in force.

### 2026-09-26 — Recurrence V1 authorized direction and frozen pre-execution design

- User clarified that “continue with these” covers repaired answer scoring, fresh bounded benchmark calibration, frozen controls, conditional paired LoRA training, retention/cost checks, and Luna auditing. This is a separate V1 protocol, not a reopening of V0.
- Luna's read-only audit of all 30 longer-output answers found a conservative final-answer grammar can parse 23 unambiguously: 18 match gold and five are wrong; seven are rejected, all on bad/incomplete answers. These old development labels are used only to define and test the parser before collecting new V1 evidence. `docs/superpowers/specs/2026-09-26-recurrence-v1-design.md` freezes the grammar, two candidate difficulties per family, independent split seeds, gates and resource caps; `docs/superpowers/plans/2026-09-26-recurrence-v1-execution.md` sequences implementation and stage execution. Neither document is a V1 experiment result.
- Task 1 implemented the separate V1 scorer in `a594892` and fixed independent-review edge cases in `26cae21`: later incomplete final list cues now invalidate an answer, list extraction derives length from the input rather than consulting the answer oracle, and EOS graph-path explanations are checked independently even when the final answer is ambiguous. Adversarial focused tests: 20 passed; the earlier full suite before the review fixes: 736 passed, 3 skipped. The corrected scorer SHA256 `2f138e6fe106c81dd03712d0a49a3a916cec2596a4acfa8f4e111f6463bab719` is frozen in the spec before any V1 GPU generation. Its read-only old-30 audit is 23 parsed, 18 correct, five wrong, seven ungradable; no new V1 data or LoRA training has run at this point. Independent Luna review approved the follow-up. `exp` was pushed at `581e90f` with this protocol milestone; no merge to `main`.
- Task 2 implemented strict V1 config and deterministic pilot/development/sealed generators in `b079072`; independent Luna review found two split-writer hardening cases, fixed and regression-tested in `f0ef253`. The freezer refuses preexisting output, catches existing ancestor symlinks/reparse points, and cleans only files it created after a handled write failure. Focused suite: 25 passed; full suite before follow-up: 763 passed, three skipped. The exact prompt+256<=512 bound and separate seeds are validated. This is implementation evidence, not a calibrated benchmark result; no V1 model generation or LoRA training has run yet. Before pilot outputs, the V1 spec resolved the two-of-three eligibility case with candidate 1 as the nonqualifying family's fixed fallback, without choosing by pilot score.

### 2026-09-26 — Recurrence V1 six-pilot calibration and terminal no-go

- The V1 calibration implementation and frozen pre-execution documentation were committed and pushed on `exp` through `88b7f3b` before GPU inference. The last pre-run full suite passed `791`, with `3` skipped; Ruff, compileall, and a read-only V0 source preflight passed. Run ID `4f1ddbedf7b4cbd11d73699d579a3e299a049aa8fb2b036e9124842d695eb2ca` passed `prepare`, then completed all six 30-row F0 pilots in `calibrate` under scorer SHA256 `2f138e6fe106c81dd03712d0a49a3a916cec2596a4acfa8f4e111f6463bab719`.
- Raw correct/invalid counts were expressions candidate 1 `28/2`, candidate 2 `28/2`; lists `0/16`, `0/18`; graphs `20/7`, `21/6`, each of 30. The registered >5%, <95% baseline accuracy and <=10% invalid-format gate selected expressions candidate 1 only. Fewer than two families qualified, so `calibrate` sealed `stopped_scientific`, and the terminal `report` sealed `next_stage=complete`. Calibration stage SHA256 `605e63abb58a9f901f2141357f20e75d369676afba4092b6a2448f47a06fae10`; report stage SHA256 `7bf0cb016e0daae3f9f7dead5fe42d7a898d1ff753ad72c015ecb301aaa12daa`.
- Independent read-only checks matched the six pilot split/manifest and score hashes, 30-row counts, and correct/invalid tallies. No selected development split, sealed split, frozen controls, training checkpoint, or LoRA run exists. Calibration consumed 666.724 CUDA-active seconds and reached 8,547,991,552 peak reserved bytes of 17,074,421,760 total VRAM.
- Post-hoc inspection found graph final-answer grammar misses several normally terminated natural/Markdown answers, but also a conflicting `Yes` opening followed by `No` final answer; naive last-answer extraction is unsafe. List responses were capped at 256 output tokens in 16/30 cases for each candidate, and inspected non-capped examples suggest confusion from templates listing operations beyond the requested sequence. These observations cannot change V1's frozen scores. The detailed [V1 calibration report](docs/recurrence-v1-calibration-report.md) records the terminal interpretation and the conditional case for a separately preregistered V2. No claim about recurrence efficacy or post-training performance improvement is supported by V1.

### 2026-09-26 — Recurrence V2 three-family design written for review

- User approved in conversation a new three-family measurement-repair design. V2 is a response to V1's benchmark no-go, not a fallback that guarantees a recurrence gain. The written [V2 design](docs/superpowers/specs/2026-09-26-recurrence-v2-design.md) retains all three families and V1's quantitative gates, repairs graph final-answer extraction while rejecting conflicts, clarifies list operation wording, and applies that wording symmetrically to future evaluation and a deterministic view of the original 8,000 training rows. It specifies new pilot/development/sealed seeds, cross-run ID exclusion, frozen hashes, resource caps, and conditional controls/LoRA stages.
- This is a specification awaiting user review, not an implemented protocol or V2 GPU result. V0/V1 evidence remains unchanged; no V2 data generation, model inference, or training has begun. The next authorized action after written-spec approval is a detailed implementation plan, not immediate GPU execution.

### 2026-09-26 — V2 design approved; calibration-first implementation contract

- User's subsequent "proceed" approved the written V2 design and directed autonomous continuation until a real project blocker or registered gate failure. The [V2 calibration plan](docs/superpowers/plans/2026-09-26-recurrence-v2-calibration.md) sequences scorer, prompt/config/train-view, fresh split, authenticated source/prepare, runtime/calibration/report, and one production calibration. It defers conditional controls/LoRA implementation until the benchmark passes, avoiding training-code spend on another calibration no-go.
- This plan is not execution evidence. V0/V1 artifacts remain untouched, and no V2 GPU output exists at this checkpoint. Luna implementers/reviewers will be used for bounded coding tasks as previously requested; the parent rules on scientific gates.

### 2026-09-27 — V2 calibration implementation reviewed and frozen before output

- Tasks 1–5 of the V2 calibration plan were implemented on `exp` through `5b9a680`, each independently reviewed by Luna with test-first fixes. Final focused Task 5 suite: 37 passed; full suite: 897 passed, 4 skipped. No V2 prepare, calibration, report, or GPU inference had run at this freeze point. The unrelated untracked `task-14-review.md` and rereview notes remain untouched.
- Frozen SHA-256: scorer `answer_v2.py` `127d67489ba75b2c620257d79a9960b276d46ddd041e9ce2aae78b320e6c2b02`; config parser `config_v2.py` `7f20086b8b303d4fdad6895fed5915445f2e2b7e5525bd15546fca33401d262e`; config TOML `84e38457e3c32bcdc466c8190251c11ee0690376ae300d77a72bb17fda80ebb7`; generator/view `data_v2.py` `ecc3d4c0382c923273f54c2a353753efd826a7f84f67b9e77cf7e778ed78ad82`; runtime `v2_runtime.py` `61ee932b9f0eb84abd3b05d609496e958d4b38b8758afa7405edf8382092a76b`; controller `study_v2.py` `3fc903f8eb62d5abdb2cfc76c7b5f11e125692c63504240231107fa2772c435c`; CLI `run_recurrence_v2.py` `cae55534bdfc1cf34a00c0b277accd43286e88a1e5d32e86603dd913b51c3ba`.
- Frozen test SHA-256: `test_recurrence_answer_v2.py` `54ea51658d16bb4d64ba5629bfa22d8aa02a5368ecd1f3701fea0ddea1d77a6d`; `test_recurrence_data_v2.py` `4af2031957ec6733cf7ea8a28a9a29f7e9e3151b2a7c28438d2657936e0171d5`; `test_recurrence_study_v2.py` `24caa8e2fbec36a36c1177d1c67f482f8a83e8b7a2616b702cb8f5aa1e0de721`; `test_recurrence_v2_runtime.py` `d9fb17af1d9d661efe9f33c83e51712d7bbca22f9fa4c4326bc59be0990fc7f5`. The run ID additionally binds current V0/V1 sources, tokenizer, reused code, V2 code, config bytes, and the prepared training-view hash.
- Read-only production preflight: branch `exp`, no tracked modifications, no existing `artifacts/recurrence-v2` root. V0 source audit authenticated 8,000 train and 600 dev rows, plus 180 V1 pilot IDs (8,780 exclusions); tokenizer SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`. CUDA is available on NVIDIA GeForce RTX 5060 Ti (17,074,421,760 bytes VRAM), BF16 supported; `nvidia-smi` reported 14,002 MiB free and 1% utilization. C: had 146,307,956,736 bytes free. The next action is one `prepare`, one `calibrate`, and terminal `report`, respecting all registered stops.

### 2026-09-27 — V2 production calibration stopped scientifically

- One `prepare` passed, one `calibrate` completed, and `report` sealed `stopped_scientific` for run `18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d`. Pilot expressions c1 and graphs c1 qualified; lists used the prespecified c1 fallback. Development F0: expressions 94/100, lists 32/100, graphs 72/100, macro 0.66, invalid 93/300 = 0.31, only expressions individually informative. The development invalid-rate and two-family gates failed. Details and independent hash/cost audit: [V2 calibration report](docs/recurrence-v2-calibration-report.md).
- Calibration used 1,280.428839 CUDA-active seconds and 1,283.547 wall seconds. The sealed split was generated but never opened/scored; no frozen controls, LoRA, retention, confirmation, or model-improvement claim followed. V2 is terminal under its registered protocol. Raw artifacts remain ignored, local, and unchanged.

### 2026-09-27 — Separate answer-audit diagnostic authorized

- The user authorized a proper follow-up to the V2 measurement concern. The [answer-audit design](docs/superpowers/specs/2026-09-27-recurrence-answer-audit-design.md) and [implementation plan](docs/superpowers/plans/2026-09-27-recurrence-answer-audit.md) specify authenticated development-only inputs, gold-free Luna review packets, independent deterministic reconciliation, and a report before any V3 scorer or new inference. This is planned diagnostic work, not a change to V2's signed `stopped_scientific` result or evidence that recurrence works.

### 2026-09-27 — Blind V2 answer audit completed

- The authenticated 123-row development-only audit, three separate Luna first-pass reviews, deterministic reconciliation, and four-row independent blind graph re-review are recorded in the [answer-audit report](docs/recurrence-v2-answer-audit-report.md). The audit found 60 correct final answers among 93 V2-invalid responses, but also 23 wrong answers after graph adjudication, three contradictory graph finals, and seven length-capped graph responses. The 30 accepted controls agreed with V2; 177 accepted rows remain unaudited. The final diagnostic SHA-256 is `9f1c32a1d664742d511d12a04650a374a467baca163f9424368aaa7ef44ae30b`. Earlier diagnostic versions are preserved and explicitly superseded after audit-validator repairs.
- V2's signed scientific stop is unchanged. No frozen controls, LoRA training, sealed scoring, or recurrence-performance claim followed. The audit supports writing a separate V3 scorer/benchmark protocol; it does not authorize a retrospective V2 pass or select a winning model.

### 2026-09-27 — V3 qualification protocol drafted, before GPU output

- The [V3 design](docs/superpowers/specs/2026-09-27-recurrence-v3-design.md) and [calibration-first plan](docs/superpowers/plans/2026-09-27-recurrence-v3-calibration.md) preserve the three objective task families, V2 prompts/difficulties/training view, Qwen3 snapshot, 256/512 token budget, gate thresholds, and recurrence/LoRA comparison. They change only the prespecified answer grammar and fresh split seeds. This is planned work, not evidence of a passing benchmark or model-performance gain. No V3 inference or training has run.

### 2026-09-27 — V3 scorer control-boundary incident, before freeze or GPU

- The V3 scorer implementer had written scorer/tests and run synthetic verification, then accidentally displayed the combined V2 audit `diagnostic.json` while looking for the invalid-only diagnostic schema. That file also contains some of the original 30 accepted controls. The implementer stopped, reported the breach, and made no subsequent scorer/test edits. No V3 model output or training occurred. The original 30 cannot serve as blind V3 qualification.
- The V3 design/plan now transparently specify a replacement, nonoverlapping 30-row accepted-control set with fixed selection seed and blind independent Luna review. Its selector and scorer must be frozen before replacement answers are opened. This is a pre-output protocol amendment, not a retrospective change to V2 or evidence of V3 success.

### 2026-09-27 — V3 scorer freeze and terminal blind-control qualification

- The V3 scorer and replacement-control harness passed independent Luna review, 1,036 CPU tests (4 skipped, 3 deselected), Ruff, compileall and diff checks. Frozen commit `01c9744` was pushed to `exp` before opening replacement controls. The reviewed harness binds transitive recurrence code, test/dependency files, three review sessions, and a durable one-shot attempt.
- Replacement control export `5715e1b5771f4c159b329113347fc3481a7924245910b04349c63cae181ca983` selected 10 previously unused V2-accepted development rows per family; separate gold-free packets went to three independent GPT-6 Luna reviews. All three reviewers agreed on extracted values for all 30, but the one permitted reconciliation stopped on the first evidence-span validation failure before calling the full V3 scorer. Read-only diagnosis found 38 invalid review citations across 18 unique rows (eight expression, ten list); this is an audit/procedure gate failure, not a V3 accuracy or recurrence result.
- The exclusive attempt SHA256 is `d0c6caac22098b4ae74c9296fbdc07e058e0d8bac2258e85c4c9c3eff26454fb`; stopped qualification SHA256 is `cd307499bf3ecdc948c1988ef51f0ba986fe0f2a5a011f7b5ee624aeb1c72e48`. Both content hashes and all currently available input hashes verified. See [V3 scorer qualification report](docs/recurrence-v3-scorer-qualification.md). No V3 GPU inference, calibration, controls, LoRA, sealed scoring or improvement claim followed. V3 cannot be rerun under the frozen protocol; next work requires a separate decision.

### 2026-09-27 — Option 2 verified-distillation QLoRA plan drafted

- Wrote [a separate, planning-only execution contract](docs/superpowers/plans/2026-09-27-verified-distillation-qlora.md) for Qwen3-4B teacher-rationale distillation into Qwen3-1.7B through QLoRA. It compares verified teacher solutions against ordinary gold-solution QLoRA on identical GSM8K train IDs, with teacher-quality, scorer, retention, transfer, cost and sealed-confirmation gates. This is a narrow post-training capability hypothesis, not dense-to-MoE conversion or a guaranteed improvement.
- No Option 2 code, dependency installation, model download, inference, or training was run. The draft awaits user review/approval. V0–V3 terminal evidence and artifacts remain unchanged.

### 2026-10-03 — Option 2 revision 2 authorized; feasibility started

- User requested revision of the four identified plan defects and continuation. The [revision 2 plan](docs/superpowers/plans/2026-09-27-verified-distillation-qlora.md) separates practical gain vs F0 from teacher-specific gain vs A1, chooses output budget on 60 separate questions using only length/EOS, fixes the primary comparator before sealed scoring, defines identical padded tensor work and baseline quantization, and permits bounded recovery from validated checkpoints/rows while retaining failures/cost.
- Read-only preflight: RTX 5060 Ti, CUDA available, BF16 supported, Python 3.11.9, torch 2.13.0+cu130, Transformers 5.16.1. Initial CPU baseline: 1,036 passed, 4 skipped, 3 deselected (81.58 seconds). PEFT/bitsandbytes were absent; feasibility is being implemented with Luna. Pinned Qwen3-4B metadata reports 8,044,982,000 bytes of weights/Apache-2.0; GSM8K pinned metadata is MIT; SVAMP HEAD equals its registered pin. Existing retention chunk SHA matches its 40-chunk/19,960-scored-token manifest. Full source/token authentication remains required before training.
- [Execution state](docs/distillation-v1-execution-state.md) records the recovery handoff. No benchmark pilot, teacher corpus, or adapter training has yet run; synthetic GPU compatibility work is Task 0.

### 2026-10-03 — Option 2 synthetic GPU feasibility passed after reviewed recovery

- Pinned Accelerate 1.15.0, bitsandbytes 0.50.2 and PEFT 0.21.2 in `uv.lock`, preserving torch 2.13.0+cu130/Transformers 5.16.1. Post-dependency CPU suite: 1,050 passed, 4 skipped, 3 deselected. The original smoke downloaded the teacher but failed on the runner's unproven 64-MiB residual-allocation check after the student update; metrics were not persisted and the attempt is not a pass.
- Reviewed fix `50164c1` uses sequential fresh GPU child processes, durable per-phase measurements, explicit authenticated source inventory, early-failure environment evidence, and allocated/reserved enforcement below physical capacity minus 2 GiB. Its 28 covering tests and scoped independent Luna review passed before recovery.
- Recovery1 exited 0: synthetic padded-1,536 student update loss 1.262591, gradient norm 4.516547, 392 changed adapter tensors, 14,222,884,864-byte peak reserved below the 14,926,938,112-byte cap; BF16 teacher generated 16 synthetic tokens, peak reserved 8,090,812,416 bytes. The synthetic response hit its small compatibility token limit; this is not an answer-quality result. Result SHA-256 `118a6c6733f5a7fd8fedcc47d2a4952bea6cfe7885059f87816703b560d06f57`. Original failure artifacts remain unchanged; failed work is charged conservatively. See [execution state](docs/distillation-v1-execution-state.md) for the command and budget.
- Data/config preparation is in progress. No calibration/pilot answers, teacher training corpus, or study adapter training exist yet. This establishes local recipe compatibility, not model improvement.

### 2026-10-03 — Option 2 source preparation passed; evaluator implementation started

- Source/config code `323080d` and reviewed fixes `4bd6d6a` are pushed on `exp`. The fixes implement the registered positional CLI, require an explicit materialization root, pin historical retention/tokenizer provenance, and verify the run-ID digest on reopening. Root CPU suite 1,087 passed/5 skipped/3 deselected; covering fix tests 21 passed/1 skip. Focused tests, exact dedup versus exhaustive reference, pinned Ruff, compile and diff checks passed. Scoped Luna re-review closed all findings.
- Root final preflight/prepare passed for `acd8dfc400c8f079d0403da5d4b981b8aac8e94226db8760f632c8307ff785c2`, with exact 60/120/200/1,200/400/200/500 splits and zero dedup exclusions. Source counts remain 7,473/1,319 GSM8K and 1,000 SVAMP. Root authenticated the manifest, stage, run ID and all output checksums without printing/scoring sealed labels; copied retention chunks match the original pin. Manifest SHA-256 `bf0233bc65cd7f6554c7ab5269f17e051c31ff3903751ef7a24297ee203370ed`. Full provenance/command in [execution state](docs/distillation-v1-execution-state.md).
- Luna now implements the scorer/budget-calibration stage. Its code must pass independent review/freeze before the first calibration response. No teacher viability result, corpus or study adapter training exists yet; old recurrence stops remain unchanged.
