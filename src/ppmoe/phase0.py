"""Deterministic Phase 0 numerical verification probe."""

from dataclasses import asdict

import torch

from ppmoe.moefication.partition import balanced_partition
from ppmoe.moefication.reference import DenseSwiGLU, PartitionedSwiGLU
from ppmoe.moefication.verify import compare_tensors
from ppmoe.oracle.exact import exact_direct_oracle, exact_gram_oracle


_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def run_phase0_probe(
    *,
    device: str,
    dtype: str,
    seed: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
) -> dict[str, object]:
    """Run one seeded dense/partitioned equivalence and oracle probe."""

    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    if hidden_size <= 0 or intermediate_size <= 0 or num_experts <= 0:
        raise ValueError("hidden_size, intermediate_size, and num_experts must be positive")
    if intermediate_size % num_experts != 0:
        raise ValueError("intermediate_size must be divisible by num_experts")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    target_dtype = _DTYPES[dtype]
    if target_device.type == "cuda":
        device_index = (
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        )
        fork_devices = [device_index]
        torch.cuda.reset_peak_memory_stats(device_index)
    else:
        fork_devices = []

    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        dense = DenseSwiGLU(
            hidden_size,
            intermediate_size,
            device=target_device,
            dtype=target_dtype,
        )
        permutation = torch.randperm(intermediate_size, device=target_device)
        partitions = balanced_partition(permutation, num_experts)
        partitioned = PartitionedSwiGLU.from_dense(dense, partitions)
        inputs = torch.randn(
            2,
            3,
            hidden_size,
            device=target_device,
            dtype=target_dtype,
        )

        with torch.no_grad():
            dense_output = dense(inputs)
            all_expert_output = partitioned(inputs)
            fp32_accumulated_output = partitioned(
                inputs,
                accumulation_dtype=torch.float32,
            )
            expert_outputs = partitioned.expert_outputs(inputs).reshape(
                -1,
                num_experts,
                hidden_size,
            )
            oracle_target = dense_output.reshape(-1, hidden_size)
            subsets_equal = True
            maximum_oracle_error_delta = 0.0
            for k in range(1, num_experts + 1):
                direct = exact_direct_oracle(oracle_target, expert_outputs, k)
                gram = exact_gram_oracle(oracle_target, expert_outputs, k)
                subsets_equal = subsets_equal and torch.equal(
                    direct.subsets,
                    gram.subsets,
                )
                maximum_oracle_error_delta = max(
                    maximum_oracle_error_delta,
                    (direct.squared_errors - gram.squared_errors)
                    .abs()
                    .max()
                    .item(),
                )

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        peak_cuda_memory = torch.cuda.max_memory_allocated(target_device)
        device_name = torch.cuda.get_device_name(target_device)
        bf16_supported = torch.cuda.is_bf16_supported()
    else:
        peak_cuda_memory = 0
        device_name = "cpu"
        bf16_supported = False

    return {
        "schema_version": 1,
        "seed": seed,
        "dimensions": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "expert_width": intermediate_size // num_experts,
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(target_device),
            "device_name": device_name,
            "dtype": dtype,
            "bf16_supported": bf16_supported,
            "peak_cuda_memory_bytes": peak_cuda_memory,
        },
        "parameter_counts": {
            "dense": sum(parameter.numel() for parameter in dense.parameters()),
            "partitioned": sum(
                parameter.numel() for parameter in partitioned.parameters()
            ),
        },
        "all_expert": asdict(compare_tensors(dense_output, all_expert_output)),
        "fp32_accumulation": asdict(
            compare_tensors(dense_output, fp32_accumulated_output)
        ),
        "oracle": {
            "k_values": list(range(1, num_experts + 1)),
            "direct_gram_subsets_equal": subsets_equal,
            "maximum_squared_error_delta": maximum_oracle_error_delta,
        },
    }
