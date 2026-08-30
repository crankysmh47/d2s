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
