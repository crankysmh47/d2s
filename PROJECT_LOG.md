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
