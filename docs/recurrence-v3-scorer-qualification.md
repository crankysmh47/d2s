# V3 scorer qualification: terminal pre-GPU stop

Date: 2026-09-27. Branch: `exp`. Frozen scorer/control commit: `01c9744`. This is a qualification result, **not** a V3 model evaluation or a recurrence-performance result.

## Decision

The one permitted replacement-control reconciliation stopped before any full-response V3 scorer call. The registered rule requires every blind review's cited span to independently yield its claimed final value. The first failure was expression audit ID `00e3bebd43d2f227`, reviewer 3: its citation omitted the closing display-math delimiter. The immutable qualification is `status=stopped`, `failure_type=ValueError`, `failure_detail=review evidence does not independently yield a final candidate`. Do not patch reviews/scorer and retry this V3 candidate. No V3 GPU inference, split calibration, recurrence controls, LoRA training, or sealed scoring has run.

The replacement export ID is `5715e1b5771f4c159b329113347fc3481a7924245910b04349c63cae181ca983`. The original 30 accepted V2 audit controls were retired after the documented blinding-boundary incident; the replacement selected 10 previously unused accepted V2 development rows per family under the frozen Python-3.11.9 seeded selector. The three distinct GPT-6 Luna sessions each reviewed all 30 gold-free packets, without gold/V2 labels in their assigned input. Session identity is root-verified process provenance, not cryptographic attestation.

## Read-only postmortem

All 30 reviewer triplets had unanimous normalized extracted values; all responses were reviewed as conclusive. This is **not** 30/30 scorer agreement: the evidence preflight stopped before full-response scoring and before comparing V3 values/correctness with V2.

Read-only replay of only the frozen evidence-candidate validation identified 38 failing review records over 18 unique rows: eight expression rows (reviewer 3 only), all ten list rows (all three reviewers), and no graph rows. The expression citations contain the final cue and opening `$$`/boxed scalar but omit closing `$$` within the cited substring. Nine list rows cite backtick-wrapped vectors or extra words not accepted by the frozen V3 evidence grammar. The remaining list row cites four elements where the source task requires six. Each citation is an exact substring of its response; the problem is incomplete or unsupported final-evidence form, not a fabricated quote. These observations are diagnostic only and do not retroactively qualify V3.

The audit cannot distinguish a true full-response V3 parser miss from a reviewer-citation miss on each affected row without running a new diagnostic scorer pass. That pass is unnecessary for the terminal V3 decision and was not run. The repeated benchmark-calibration difficulties across V1/V2/V3 argue against spending training compute under this protocol.

## Provenance and preservation

- Frozen code was pushed before replacement export. Root verification: `uv run pytest -q -m "not cuda"` yielded 1,036 passed, 4 skipped, 3 deselected; changed-file Ruff, compileall and diff checks passed.
- Private manifest internal content hash: `853b968c71a53f599b919d61601954125412bd55eb9e907a9dcdeee0e5a94ee7`; raw `private-manifest.json` SHA-256: `cb6ec0221ffdbc7854a94d94756b68910358afdddf0dfc49ad72d8c3fcc30d4f`. Blind review raw SHA-256 values: reviewer 1 `eef74f71ec932c3bd4285aa3e5cc817657ee6b309ae467cb7f6e062f40eba400`; reviewer 2 `e68dada577adaf50a6a272ae6576a0d1e9e3c04d5fec6e540ac4d5466111eb66`; reviewer 3 `2c937ff0c5a090e816825ba9c85d9a5b7988e588c0d82675038277a4b2e16786`.
- Exclusive attempt SHA-256: `d0c6caac22098b4ae74c9296fbdc07e058e0d8bac2258e85c4c9c3eff26454fb`; stopped qualification SHA-256: `cd307499bf3ecdc948c1988ef51f0ba986fe0f2a5a011f7b5ee624aeb1c72e48`. Both records' internal content hashes validate. Available input hashes, source/stage hashes, public/private packet copies, session file and frozen-code map matched in read-only postmortem. The artifacts remain local and are not OS write-protected; immutable creation plus hashes makes later changes detectable rather than physically impossible.
- V0/V1/V2 terminal evidence was not changed, and no sealed split was opened. Unrelated `task-14*.md` files remain untouched.

Next decision requires a separately reviewed new protocol or a strategic pivot. V3 cannot be resumed as a passing calibration or used to justify recurrence/LoRA training.
