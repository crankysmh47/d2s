from copy import deepcopy

import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

from ppmoe.moefication.qwen35 import (
    PartitionedQwen35MLP,
    convert_qwen35_text_ffns,
    probe_qwen35_ffn_conversion,
)
from tests.helpers import tiny_qwen35


def test_conversion_physically_permutes_triplets_and_preserves_model_budget() -> None:
    torch.manual_seed(41)
    model = tiny_qwen35()
    original_gate = model.model.language_model.layers[0].mlp.gate_proj.weight.detach().clone()
    original_up = model.model.language_model.layers[0].mlp.up_proj.weight.detach().clone()
    original_down = model.model.language_model.layers[0].mlp.down_proj.weight.detach().clone()
    original_parameters = sum(parameter.numel() for parameter in model.parameters())
    visual = model.model.visual

    manifest = convert_qwen35_text_ffns(model, num_experts=8, seed_base=20_260_830)

    converted = model.model.language_model.layers[0].mlp
    assert isinstance(converted, PartitionedQwen35MLP)
    permutation = converted.permutation
    torch.testing.assert_close(converted.gate_proj.weight, original_gate[permutation])
    torch.testing.assert_close(converted.up_proj.weight, original_up[permutation])
    torch.testing.assert_close(converted.down_proj.weight, original_down[:, permutation])
    assert converted.expert_width == 4
    assert model.model.visual is visual
    assert sum(parameter.numel() for parameter in model.parameters()) == original_parameters
    assert manifest.converted_layers == 2
    assert manifest.parameter_count_before == manifest.parameter_count_after
    assert manifest.probe.layer_index == 0
    assert manifest.probe.seed == 20_260_830
    assert manifest.probe.tokens == 16
    assert manifest.probe.max_abs_difference <= 1e-5
    assert manifest.records[0].expert_original_indices == tuple(
        tuple(int(value) for value in permutation[start:start + converted.expert_width].tolist())
        for start in range(0, converted.intermediate_size, converted.expert_width)
    )


def test_fp32_conversion_probe_uses_layer_zero_weights_without_mutating_model() -> None:
    torch.manual_seed(59)
    model = tiny_qwen35(dtype=torch.bfloat16)
    original = model.model.language_model.layers[0].mlp.gate_proj.weight.detach().clone()

    evidence = probe_qwen35_ffn_conversion(model, num_experts=8, seed_base=211)

    assert evidence.layer_index == 0
    assert evidence.seed == 211
    assert evidence.tokens == 16
    assert evidence.max_abs_difference <= 1e-5
    torch.testing.assert_close(model.model.language_model.layers[0].mlp.gate_proj.weight, original)


def test_all_expert_and_selected_execution_match_literal_reconstruction() -> None:
    torch.manual_seed(43)
    model = tiny_qwen35()
    dense = deepcopy(model.model.language_model.layers[0].mlp)
    convert_qwen35_text_ffns(model, num_experts=8, seed_base=7)
    converted = model.model.language_model.layers[0].mlp
    inputs = torch.randn(2, 3, 16)

    dense_output = dense(inputs)
    all_expert = converted(inputs)
    expert_outputs = converted.expert_outputs(inputs)
    global_selection = converted(inputs, selected_experts=(1, 3, 5))
    token_selection = torch.tensor(
        [[[0, 1], [2, 3], [4, 5]], [[6, 7], [0, 2], [1, 4]]]
    )
    selected = converted(inputs, selected_experts=token_selection)
    literal = torch.gather(
        expert_outputs,
        -2,
        token_selection[..., None].expand(*token_selection.shape, 16),
    ).sum(dim=-2)

    assert (dense_output - all_expert).abs().max().item() <= 1e-5
    torch.testing.assert_close(global_selection, expert_outputs[..., [1, 3, 5], :].sum(-2))
    torch.testing.assert_close(selected, literal)


def test_default_all_expert_restores_original_reduction_order_exactly() -> None:
    torch.manual_seed(47)
    model = tiny_qwen35()
    dense = deepcopy(model.model.language_model.layers[0].mlp)
    inputs = torch.randn(2, 3, 16)

    convert_qwen35_text_ffns(model, num_experts=8, seed_base=29)
    converted = model.model.language_model.layers[0].mlp

    assert torch.equal(converted(inputs), dense(inputs))
    assert torch.equal(converted.inverse_permutation, torch.argsort(converted.permutation))
    assert "model.language_model.layers.0.mlp.inverse_permutation" not in model.state_dict()


@pytest.mark.cuda
def test_production_shape_bf16_default_all_expert_is_exact_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")

    torch.manual_seed(53)
    config = Qwen3_5TextConfig(hidden_size=1_024, intermediate_size=3_584)
    dense = Qwen3_5MLP(config, 3_584).to(device="cuda", dtype=torch.bfloat16).eval()
    original_parameters = sum(parameter.numel() for parameter in dense.parameters())
    permutation = torch.roll(torch.arange(3_584), shifts=137)
    converted = PartitionedQwen35MLP(
        dense,
        permutation=permutation,
        num_experts=8,
    ).eval()
    inputs = torch.randn(
        1,
        4,
        1_024,
        generator=torch.Generator(device="cuda").manual_seed(59),
        device="cuda",
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        dense_output = dense(inputs)
        converted_output = converted(inputs)

    assert not torch.equal(converted.permutation, torch.arange(3_584, device="cuda"))
    assert torch.equal(converted.gate_proj.weight, dense.gate_proj.weight[converted.permutation])
    assert torch.equal(converted.up_proj.weight, dense.up_proj.weight[converted.permutation])
    assert torch.equal(converted.down_proj.weight, dense.down_proj.weight[:, converted.permutation])
    assert torch.equal(converted_output, dense_output)
    assert {parameter.dtype for parameter in converted.parameters()} == {torch.bfloat16}
    assert sum(parameter.numel() for parameter in converted.parameters()) == original_parameters
    assert torch.equal(converted.inverse_permutation, torch.argsort(converted.permutation))
    assert "inverse_permutation" not in converted.state_dict()


def test_layer_permutations_are_deterministic_and_nonpersistent() -> None:
    first = tiny_qwen35()
    second = deepcopy(first)

    first_manifest = convert_qwen35_text_ffns(first, num_experts=8, seed_base=101)
    second_manifest = convert_qwen35_text_ffns(second, num_experts=8, seed_base=101)

    assert first_manifest.records == second_manifest.records
    assert "model.language_model.layers.0.mlp.permutation" not in first.state_dict()


def test_output_transform_is_temporary_validated_and_parameter_free() -> None:
    model = tiny_qwen35()
    convert_qwen35_text_ffns(model, num_experts=8, seed_base=17)
    converted = model.model.language_model.layers[0].mlp
    inputs = torch.randn(1, 2, 16)
    original_parameters = sum(parameter.numel() for parameter in converted.parameters())
    original_state_keys = tuple(converted.state_dict())
    experts = converted.expert_outputs(inputs)
    default_output = converted(inputs)

    with converted.output_transform(lambda _x, outputs: outputs[..., 0, :]):
        torch.testing.assert_close(converted(inputs), experts[..., 0, :])
        with pytest.raises(RuntimeError, match="already active"):
            with converted.output_transform(lambda _x, outputs: outputs.sum(dim=-2)):
                pass
    torch.testing.assert_close(converted(inputs), default_output)

    with pytest.raises(ValueError, match="shape"):
        with converted.output_transform(lambda _x, outputs: outputs[..., 0, :1]):
            converted(inputs)
    with pytest.raises(RuntimeError, match="transform failure"):
        with converted.output_transform(
            lambda _x, _outputs: (_ for _ in ()).throw(RuntimeError("transform failure"))
        ):
            converted(inputs)
    torch.testing.assert_close(converted(inputs), default_output)
    assert sum(parameter.numel() for parameter in converted.parameters()) == original_parameters
    assert tuple(converted.state_dict()) == original_state_keys
