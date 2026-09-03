# Task 7 report — Phase 1A.5 hashed reporting and gated execution

## Implementation checkpoint

- Commit: `88d0e8e feat: add phase 1a5 reporting` on `exp`.
- Added `render_phase1a5_report(artifact_root, config)` with hash-addressed plot sources, PNGs, a report manifest, and conservative decision text.
- The validation layer rejects source-provenance drift, missing local conditions, mutable selection, evaluation leakage, non-finite values, bad token/layer coverage, failed batches, invalid bounded KKT evidence, and existing output hash mismatch.
- The report labels the Registered Free-Reconstruction Candidate Bound as a candidate bound only. It explicitly excludes claims of an exact-span optimum, a trained router, or inference speedup; gain recovery is analysis-only.

## Verification before production

| Check | Observed result |
| --- | --- |
| Focused reporting/orchestration tests | 19 passed |
| Full test suite | 230 passed |
| `uv lock --check` | locked 85 packages resolved |
| `git diff --check` | passed |

## Production evidence

Source command:

```text
uv run python scripts/run_phase1a5.py --stage source --artifact-root artifacts/phase1-v0_5 --microbatch-size 1
```

- Exit code: 0.
- Source signature: `7d41b9700d33c9151b1cc1aca325187acd41514d5ceada847f50a4fcb52d832b`.
- Code provenance: clean commit `88d0e8ebe4469d0ece42932c877df361dd1244c5`.

One smoke command was executed:

```text
uv run python scripts/run_phase1a5.py --stage smoke --artifact-root artifacts/phase1-v0_5 --microbatch-size 1
```

- Exit code: 1 after about 9.6 seconds.
- Failure: `RuntimeError: Inference tensors cannot be saved for backward` during the unit-regression path. A tensor captured under `torch.inference_mode()` was consumed by a student linear layer outside an inference/no-grad context.
- No smoke stage/checkpoint was written. No family reached timing, peak-CUDA, completed/failed batch, unit-regression completion, SVD, KKT, or bounded-feasibility evidence.

## Gate and next decision

The smoke gate did not pass. Per the frozen protocol, this task did not retry, tune thresholds, alter microbatch size, run the 100K calibration, select conditions, inspect evaluation PPL, execute rollout, or render a production report.

The next task must add a regression test for the inference-tensor/autograd-mode boundary, make the execution modes compatible without changing the scientific protocol, recommit, generate a new clean source signature, and seek independent review before another smoke attempt.
