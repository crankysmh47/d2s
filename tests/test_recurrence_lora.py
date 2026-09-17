from __future__ import annotations

import pytest
import torch
from torch import nn

from ppmoe.recurrence import lora
from ppmoe.recurrence.lora import (
    LoRALinear,
    adapter_sha256,
    adapter_state_dict,
    install_adapters,
    set_adapters_enabled,
)


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8)
        self.up_proj = nn.Linear(4, 8)
        self.down_proj = nn.Linear(8, 4)


class _TinyQwenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyAttention()
        self.mlp = _TinyMLP()


class _TinyBackbone(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_TinyQwenLayer() for _ in range(layers))


class _TinyQwen(nn.Module):
    def __init__(self, layers: int = 6) -> None:
        super().__init__()
        self.model = _TinyBackbone(layers)


def test_zero_b_adapter_matches_base_exactly() -> None:
    """Catches a nonzero initial B matrix or unintended adapter arithmetic."""
    base = nn.Linear(3, 2, bias=False).to(dtype=torch.bfloat16)
    layer = LoRALinear(base, rank=8, alpha=16)
    x = torch.randn(4, 3, dtype=torch.bfloat16)

    assert torch.equal(layer(x), base(x))
    assert layer.a.dtype == torch.float32
    assert layer.b.dtype == torch.float32


def test_nonzero_adapter_uses_fp32_low_rank_update() -> None:
    """Catches a transposed, unscaled, or low-precision LoRA update."""
    base = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    layer = LoRALinear(base, rank=1, alpha=2)
    with torch.no_grad():
        layer.a.copy_(torch.tensor([[1.0, 2.0]]))
        layer.b.copy_(torch.tensor([[3.0], [4.0]]))
    x = torch.tensor([[2.0, 1.0]])

    expected = base(x) + 2 * ((x.float() @ layer.a.T) @ layer.b.T)

    torch.testing.assert_close(layer(x), expected.to(dtype=base(x).dtype), rtol=0, atol=0)


def test_installation_wraps_only_targets_and_freezes_base_parameters() -> None:
    """Catches incomplete target selection or accidentally trainable dense weights."""
    model = _TinyQwen()

    candidates = install_adapters(model, layer_start=2, layer_stop=4, seed=47)
    wrapped = [module for module in model.modules() if isinstance(module, LoRALinear)]
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)

    assert len(wrapped) == 12
    assert not isinstance(model.model.layers[2].self_attn.k_proj, LoRALinear)
    assert not isinstance(model.model.layers[3].self_attn.k_proj, LoRALinear)
    assert all(not parameter.requires_grad for layer in model.model.layers[:2] for parameter in layer.parameters())
    assert all(not parameter.requires_grad for layer in model.model.layers[4:] for parameter in layer.parameters())
    assert all(not wrapper.base.weight.requires_grad for wrapper in wrapped)
    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
        if parameter not in set(candidates)
    )
    assert all(parameter.requires_grad and parameter.dtype == torch.float32 for parameter in candidates)
    assert set(candidates) == set(trainable)
    assert set(candidates) == {parameter for wrapper in wrapped for parameter in (wrapper.a, wrapper.b)}


def test_identical_seeded_installations_have_stable_hash_and_state_order() -> None:
    """Catches unstable installation order or a hash that omits adapter contents."""
    first = _TinyQwen()
    second = _TinyQwen()
    install_adapters(first, layer_start=2, layer_stop=4, seed=47)
    install_adapters(second, layer_start=2, layer_stop=4, seed=47)

    assert tuple(adapter_state_dict(first)) == tuple(adapter_state_dict(second))
    assert adapter_sha256(first) == adapter_sha256(second)
    with torch.no_grad():
        next(iter(adapter_state_dict(second).values())).add_(1)
    assert adapter_sha256(first) != adapter_sha256(second)


def test_disable_then_enable_restores_adapter_contribution() -> None:
    """Catches an adapter enable flag that does not control every wrapped projection."""
    model = _TinyQwen()
    install_adapters(model, layer_start=2, layer_stop=4, seed=47)
    projection = model.model.layers[2].self_attn.q_proj
    assert isinstance(projection, LoRALinear)
    x = torch.ones(1, 4)
    with torch.no_grad():
        projection.b.fill_(1)

    adapted = projection(x)
    set_adapters_enabled(model, False)
    assert torch.equal(projection(x), projection.base(x))
    set_adapters_enabled(model, True)
    assert torch.equal(projection(x), adapted)


def test_installation_rejects_an_already_wrapped_projection() -> None:
    """Catches a second installation silently nesting an adapter around an adapter."""
    model = _TinyQwen()
    install_adapters(model, layer_start=2, layer_stop=4, seed=47)

    with pytest.raises(ValueError, match="already wrapped"):
        install_adapters(model, layer_start=2, layer_stop=4, seed=47)


def test_generator_cache_uses_full_device_and_constructs_on_that_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches collapsing cuda:0 and cuda:1 into one device-type generator."""
    created: list[torch.device] = []

    class _Generator:
        def __init__(self, *, device: torch.device) -> None:
            created.append(device)

        def manual_seed(self, seed: int) -> _Generator:
            assert seed == 47
            return self

    monkeypatch.setattr(lora.torch, "Generator", _Generator)
    cache: dict[torch.device, torch.Generator] = {}
    first = lora._generator_for_device(cache, torch.device("cuda", 0), 47)
    second = lora._generator_for_device(cache, torch.device("cuda", 1), 47)

    assert first is cache[torch.device("cuda", 0)]
    assert second is cache[torch.device("cuda", 1)]
    assert first is not second
    assert created == [torch.device("cuda", 0), torch.device("cuda", 1)]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_installation_preserves_a_non_current_cuda_device() -> None:
    """Catches constructing a cuda:0 generator for projections resident on cuda:1."""
    current = torch.cuda.current_device()
    target = 1 if current == 0 else 0
    device = torch.device("cuda", target)
    model = _TinyQwen().to(device)

    candidates = install_adapters(model, layer_start=2, layer_stop=4, seed=47)

    assert all(parameter.device == device for parameter in candidates)


def test_late_preflight_failure_leaves_modules_and_grad_flags_unchanged() -> None:
    """Catches replacement/freezing before a later invalid target is rejected."""
    model = _TinyQwen()
    late = model.model.layers[3].self_attn.v_proj
    model.model.layers[3].self_attn.v_proj = LoRALinear(late)
    targets = [
        getattr(getattr(model.model.layers[layer_index], parent), name)
        for layer_index in range(2, 4)
        for parent, name in lora._TARGET_PATHS
    ]
    identities = tuple(id(target) for target in targets)
    grad_flags = {id(parameter): parameter.requires_grad for parameter in model.parameters()}

    with pytest.raises(ValueError, match="already wrapped"):
        install_adapters(model, layer_start=2, layer_stop=4, seed=47)

    after_targets = [
        getattr(getattr(model.model.layers[layer_index], parent), name)
        for layer_index in range(2, 4)
        for parent, name in lora._TARGET_PATHS
    ]
    assert tuple(id(target) for target in after_targets) == identities
    assert {id(parameter): parameter.requires_grad for parameter in model.parameters()} == grad_flags


def test_production_range_independently_contains_twenty_four_wrapped_projections() -> None:
    """Catches a production installation that reports success without 24 replacements."""
    model = _TinyQwen(layers=16)

    install_adapters(model, layer_start=12, layer_stop=16, seed=47)
    wrapped = [module for module in model.modules() if isinstance(module, LoRALinear)]

    assert len(wrapped) == 24
