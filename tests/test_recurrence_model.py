"""CPU-only contracts for explicit, cacheless Qwen3 block recurrence."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from ppmoe.recurrence.lora import install_adapters
from ppmoe.recurrence.model import RecurrentQwen3, apply_block, chunked_causal_loss


def _tiny_stock_model() -> Qwen3ForCausalLM:
    """Return a local six-layer eager Qwen3 with no downloaded weights."""
    torch.manual_seed(47)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=None,
        _attn_implementation="eager",
        use_cache=False,
        pad_token_id=0,
    )
    return Qwen3ForCausalLM(config).eval()


@pytest.fixture
def stock() -> Qwen3ForCausalLM:
    return _tiny_stock_model()


@pytest.fixture
def wrapper(stock: Qwen3ForCausalLM) -> RecurrentQwen3:
    return RecurrentQwen3(stock, layer_start=2, layer_stop=4).eval()


def test_recurrence_algebra() -> None:
    h = torch.tensor([2.0])
    block = lambda x: 2 * x

    assert torch.equal(apply_block(block, h, "single"), torch.tensor([4.0]))
    assert torch.equal(apply_block(block, h, "repeat"), torch.tensor([8.0]))
    assert torch.equal(apply_block(block, h, "damped"), torch.tensor([4.5]))
    with pytest.raises(ValueError, match="unsupported mode"):
        apply_block(block, h, "other")


def test_single_is_stock_equivalent_and_stock_is_registered_once(
    stock: Qwen3ForCausalLM, wrapper: RecurrentQwen3
) -> None:
    """Catches an omitted Qwen input, layer, norm, or output-head operation."""
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    expected = stock(input_ids=ids, use_cache=False).logits

    actual = wrapper(ids, mode="single")

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert tuple(wrapper.named_children()) == (("stock_model", stock),)
    assert {id(parameter) for parameter in wrapper.parameters()} == {
        id(parameter) for parameter in stock.parameters()
    }
    assert all(
        wrapper.stock_model.model.layers[index] is stock.model.layers[index]
        for index in range(len(stock.model.layers))
    )


def test_repeat_reuses_the_same_parameters_and_decoder_layers(
    stock: Qwen3ForCausalLM, wrapper: RecurrentQwen3
) -> None:
    """Catches a copied block or a pass-specific parameter registration."""
    ids = torch.tensor([[1, 2, 3, 4]])
    before = {id(parameter) for parameter in wrapper.parameters()}
    layer_ids = tuple(id(layer) for layer in stock.model.layers)

    wrapper(ids, mode="repeat")

    assert {id(parameter) for parameter in wrapper.parameters()} == before
    assert tuple(id(layer) for layer in wrapper.stock_model.model.layers) == layer_ids


@pytest.mark.parametrize(
    ("mode", "expected_calls"),
    (("single", 1), ("repeat", 2), ("damped", 2)),
)
def test_layers_and_terminal_modules_have_exact_call_counts(
    wrapper: RecurrentQwen3, mode: str, expected_calls: int
) -> None:
    """Catches terminal computation inside a recurrence pass or a skipped layer."""
    counts: Counter[str] = Counter()
    hooks = []
    for index, layer in enumerate(wrapper.stock_model.model.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _args, _output, index=index: counts.update([f"layer-{index}"])
            )
        )
    hooks.extend(
        (
            wrapper.stock_model.model.norm.register_forward_hook(
                lambda _module, _args, _output: counts.update(["norm"])
            ),
            wrapper.stock_model.lm_head.register_forward_hook(
                lambda _module, _args, _output: counts.update(["head"])
            ),
        )
    )
    try:
        wrapper(torch.tensor([[1, 2, 3, 4]]), mode=mode)
    finally:
        for hook in hooks:
            hook.remove()

    for index in range(6):
        assert counts[f"layer-{index}"] == (expected_calls if 2 <= index < 4 else 1)
    assert counts["norm"] == 1
    assert counts["head"] == 1


def test_repeat_reuses_identical_causal_inputs_for_both_passes(
    wrapper: RecurrentQwen3,
) -> None:
    """Catches regenerated positions, rotary embeddings, masks, or a KV cache."""
    captured: list[dict[str, Any]] = []

    def capture(_module: nn.Module, _args: tuple[object, ...], kwargs: dict[str, Any]) -> None:
        captured.append(kwargs)

    hook = wrapper.stock_model.model.layers[2].register_forward_pre_hook(
        capture, with_kwargs=True
    )
    try:
        wrapper(torch.tensor([[1, 2, 3, 4]]), mode="repeat")
    finally:
        hook.remove()

    assert len(captured) == 2
    first, second = captured
    for name in ("attention_mask", "position_ids", "position_embeddings"):
        assert first[name] is second[name]
    assert first["past_key_values"] is None
    assert second["past_key_values"] is None
    assert first["use_cache"] is False
    assert second["use_cache"] is False


@pytest.mark.parametrize("mode", ["single", "repeat", "damped"])
def test_suffix_tokens_cannot_change_prefix_logits(
    wrapper: RecurrentQwen3, mode: str
) -> None:
    left = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    right = torch.tensor([[1, 2, 3, 4, 5, 8, 9]])

    torch.testing.assert_close(
        wrapper(left, mode)[:, :5],
        wrapper(right, mode)[:, :5],
        rtol=1e-5,
        atol=1e-6,
    )


def test_hidden_can_return_each_recurrent_pass(wrapper: RecurrentQwen3) -> None:
    """Catches observability that accidentally re-executes a pass or skips normalization."""
    ids = torch.tensor([[1, 2, 3, 4]])

    hidden, passes = wrapper.hidden(ids, "damped", return_pass_hidden_states=True)

    assert hidden.shape == (1, 4, 32)
    assert len(passes) == 2
    assert all(pass_hidden.shape == (1, 4, 32) for pass_hidden in passes)
    torch.testing.assert_close(hidden, wrapper.hidden(ids, "damped"), rtol=0, atol=0)


def test_answer_loss_executes_the_model_once(wrapper: RecurrentQwen3) -> None:
    """Catches answer_loss materializing logits via forward before projecting hidden chunks."""
    calls: Counter[str] = Counter()
    hooks = [
        layer.register_forward_hook(
            lambda _module, _args, _output, index=index: calls.update([f"layer-{index}"])
        )
        for index, layer in enumerate(wrapper.stock_model.model.layers)
    ]
    try:
        loss = wrapper.answer_loss(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[-100, -100, 3, 4]]),
            mode="repeat",
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert loss.ndim == 0
    for index in range(6):
        assert calls[f"layer-{index}"] == (2 if 2 <= index < 4 else 1)


def test_answer_loss_rejects_multiple_supervised_eos_labels(wrapper: RecurrentQwen3) -> None:
    """Catches malformed answer targets that include EOS more than once."""
    wrapper.stock_model.config.eos_token_id = 4

    with pytest.raises(ValueError, match="multiple EOS"):
        wrapper.answer_loss(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[-100, 4, 3, 4]]),
            mode="single",
        )


def test_training_keeps_frozen_suffix_differentiable_for_repeat_adapters(
    wrapper: RecurrentQwen3,
) -> None:
    """Catches no_grad around the frozen suffix, which would sever adapter gradients."""
    adapters = install_adapters(wrapper.stock_model, layer_start=2, layer_stop=4, seed=47)
    wrapper.train()
    captured_inputs: list[Tensor] = []

    def capture_suffix_input(_module: nn.Module, args: tuple[object, ...]) -> None:
        tensor = args[0]
        assert isinstance(tensor, Tensor)
        tensor.retain_grad()
        captured_inputs.append(tensor)

    hook = wrapper.stock_model.model.layers[4].register_forward_pre_hook(capture_suffix_input)
    try:
        loss = wrapper.answer_loss(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[-100, -100, 3, 4]]),
            mode="repeat",
        )
        loss.backward()
    finally:
        hook.remove()

    assert captured_inputs and captured_inputs[0].grad is not None
    assert any(adapter.grad is not None for adapter in adapters)
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for layer in wrapper.stock_model.model.layers[4:]
        for parameter in layer.parameters()
    )


def test_repeat_wrapper_adapter_gradients_match_two_direct_shared_block_calls(
    wrapper: RecurrentQwen3,
) -> None:
    """Catches an adapter-gradient path that omits one of repeat's shared passes."""
    adapters = install_adapters(wrapper.stock_model, layer_start=2, layer_stop=4, seed=47)
    wrapper.train()
    captured: list[tuple[Tensor, dict[str, Any]]] = []

    def capture(
        _module: nn.Module, args: tuple[object, ...], kwargs: dict[str, Any]
    ) -> None:
        tensor = args[0]
        assert isinstance(tensor, Tensor)
        captured.append((tensor, kwargs))

    hook = wrapper.stock_model.model.layers[2].register_forward_pre_hook(capture, with_kwargs=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    try:
        wrapper.answer_loss(ids, labels, mode="repeat").backward()
    finally:
        hook.remove()
    wrapper_gradients = [adapter.grad.detach().clone() for adapter in adapters]
    # The first two entries are forward passes; non-reentrant checkpointing
    # recomputes both during backward and therefore appends two more hooks.
    assert len(captured) >= 2

    for adapter in adapters:
        adapter.grad = None
    prefix_output, layer_arguments = captured[0]

    def direct_block(current: Tensor) -> Tensor:
        for layer_index in range(wrapper.layer_start, wrapper.layer_stop):
            current = wrapper._run_layer(
                wrapper.stock_model.model.layers[layer_index],
                current,
                layer_arguments["attention_mask"],
                layer_arguments["position_ids"],
                layer_arguments["position_embeddings"],
            )
        return current

    direct_hidden = direct_block(direct_block(prefix_output.detach().clone().requires_grad_(True)))
    for layer_index in range(wrapper.layer_stop, len(wrapper.stock_model.model.layers)):
        direct_hidden = wrapper._run_layer(
            wrapper.stock_model.model.layers[layer_index],
            direct_hidden,
            layer_arguments["attention_mask"],
            layer_arguments["position_ids"],
            layer_arguments["position_embeddings"],
        )
    direct_hidden = wrapper.stock_model.model.norm(direct_hidden)
    chunked_causal_loss(direct_hidden, labels, wrapper.stock_model.lm_head).backward()

    for actual, expected in zip((adapter.grad for adapter in adapters), wrapper_gradients, strict=True):
        assert actual is not None
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_repeat_shared_adapter_fp64_gradient_matches_finite_difference() -> None:
    """Catches a repeat path whose shared adapter receives the wrong derivative."""
    adapter = nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
    initial = torch.tensor([2.0], dtype=torch.float64)

    def block(value: Tensor) -> Tensor:
        return adapter * value

    loss = apply_block(block, initial, "repeat").square().sum()
    loss.backward()
    analytic = adapter.grad.detach().clone()
    epsilon = 1e-6
    with torch.no_grad():
        adapter.add_(epsilon)
    plus = apply_block(block, initial, "repeat").square().sum()
    with torch.no_grad():
        adapter.sub_(2 * epsilon)
    minus = apply_block(block, initial, "repeat").square().sum()
    with torch.no_grad():
        adapter.add_(epsilon)
    finite_difference = (plus - minus) / (2 * epsilon)

    torch.testing.assert_close(analytic, finite_difference, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    ("input_ids", "error"),
    (
        (torch.tensor([1, 2, 3]), "shape"),
        (torch.empty((1, 0), dtype=torch.long), "length"),
        (torch.ones((2, 4), dtype=torch.long), "batch size"),
        (torch.ones((1, 513), dtype=torch.long), "length"),
        (torch.tensor([[1, 0, 2]], dtype=torch.long), "padding"),
    ),
)
def test_invalid_input_ids_fail_before_layer_execution(
    wrapper: RecurrentQwen3, input_ids: Tensor, error: str
) -> None:
    calls = 0

    def count(_module: nn.Module, _args: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    hook = wrapper.stock_model.model.layers[0].register_forward_pre_hook(count)
    try:
        with pytest.raises(ValueError, match=error):
            wrapper.hidden(input_ids, "single")
    finally:
        hook.remove()
    assert calls == 0


def test_unsupported_runtime_state_is_rejected(wrapper: RecurrentQwen3) -> None:
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="attention_mask"):
        wrapper.hidden(ids, "single", attention_mask=torch.ones_like(ids))
    with pytest.raises(ValueError, match="past_key_values"):
        wrapper.hidden(ids, "single", past_key_values=object())
    with pytest.raises(ValueError, match="use_cache"):
        wrapper.hidden(ids, "single", use_cache=True)
    with pytest.raises(ValueError, match="unsupported mode"):
        wrapper.hidden(ids, "not-a-mode")


def test_invalid_loop_range_and_sliding_attention_are_rejected(stock: Qwen3ForCausalLM) -> None:
    with pytest.raises(ValueError, match="layer range"):
        RecurrentQwen3(stock, layer_start=4, layer_stop=4)
    with pytest.raises(ValueError, match="layer range"):
        RecurrentQwen3(stock, layer_start=-1, layer_stop=2)

    stock.model.has_sliding_layers = True
    with pytest.raises(ValueError, match="sliding attention"):
        RecurrentQwen3(stock, layer_start=2, layer_stop=4)
