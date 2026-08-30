import math

import pytest
import torch

from ppmoe.phase0 import run_phase0_probe


@pytest.mark.cuda
def test_fp32_expert_accumulation_is_not_worse_than_native_bf16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")

    report = run_phase0_probe(
        device="cuda",
        dtype="bfloat16",
        seed=41,
        hidden_size=128,
        intermediate_size=512,
        num_experts=8,
    )
    native = report["all_expert"]
    fp32_accumulation = report["fp32_accumulation"]

    assert all(math.isfinite(value) for value in native.values())
    assert all(math.isfinite(value) for value in fp32_accumulation.values())
    assert (
        fp32_accumulation["normalized_mse"]
        <= native["normalized_mse"] + 1e-7
    )
    assert report["environment"]["bf16_supported"] is True
