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
