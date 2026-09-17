"""Small, deterministic FP32 LoRA adapters for the recurrence experiment."""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_TARGET_PATHS = (
    ("mlp", "down_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("self_attn", "o_proj"),
    ("self_attn", "q_proj"),
    ("self_attn", "v_proj"),
)


class LoRALinear(nn.Module):
    """A frozen linear projection plus a trainable FP32 low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        *,
        _generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.a = nn.Parameter(
            torch.empty(
                (rank, base.in_features), device=base.weight.device, dtype=torch.float32
            )
        )
        self.b = nn.Parameter(
            torch.zeros(
                (base.out_features, rank), device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.a, a=sqrt(5), generator=_generator)
        self.register_buffer("_adapters_enabled", torch.tensor(True), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base(x)
        if not bool(self._adapters_enabled):
            return base_output
        context = (
            nullcontext()
            if x.device.type == "cpu"
            else torch.autocast(device_type=x.device.type, enabled=False)
        )
        with context:
            hidden = F.linear(x.float(), self.a)
            delta = F.linear(hidden, self.b) * self.scale
        return base_output + delta.to(dtype=base_output.dtype)


def _layers(model: nn.Module) -> nn.ModuleList:
    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise ValueError("model must expose Qwen-style model.layers") from exc
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("model.model.layers must be an nn.ModuleList")
    return layers


def _projection_parent(layer: nn.Module, path: tuple[str, str]) -> tuple[nn.Module, str]:
    parent_name, projection_name = path
    try:
        parent = getattr(layer, parent_name)
    except AttributeError as exc:
        raise ValueError(f"layer is missing {parent_name}") from exc
    if not isinstance(parent, nn.Module):
        raise ValueError(f"layer {parent_name} is not a module")
    return parent, projection_name


def _generator_for_device(
    generators: dict[torch.device, torch.Generator], device: torch.device, seed: int
) -> torch.Generator:
    """Return the seeded generator belonging to this exact tensor device."""
    generator = generators.get(device)
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        generators[device] = generator
    return generator


def install_adapters(
    model: nn.Module, layer_start: int, layer_stop: int, seed: int
) -> tuple[nn.Parameter, ...]:
    """Freeze ``model`` and wrap the six approved projections in each selected layer."""
    layers = _layers(model)
    if not 0 <= layer_start < layer_stop <= len(layers):
        raise ValueError("adapter layer range is outside model.model.layers")

    planned: list[tuple[int, nn.Module, str, nn.Linear]] = []
    for layer_index in range(layer_start, layer_stop):
        layer = layers[layer_index]
        for path in _TARGET_PATHS:
            parent, name = _projection_parent(layer, path)
            try:
                projection = getattr(parent, name)
            except AttributeError as exc:
                raise ValueError(f"layer {layer_index} is missing {'.'.join(path)}") from exc
            if isinstance(projection, LoRALinear):
                raise ValueError(f"projection {layer_index}.{'.'.join(path)} is already wrapped")
            if not isinstance(projection, nn.Linear):
                raise TypeError(f"projection {layer_index}.{'.'.join(path)} must be nn.Linear")
            planned.append((layer_index, parent, name, projection))

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    generators: dict[torch.device, torch.Generator] = {}
    adapters: list[nn.Parameter] = []
    for _, parent, name, projection in planned:
        generator = _generator_for_device(generators, projection.weight.device, seed)
        wrapped = LoRALinear(projection, _generator=generator)
        setattr(parent, name, wrapped)
        adapters.extend((wrapped.a, wrapped.b))

    wrapped_count = sum(
        isinstance(getattr(parent, name), LoRALinear) for _, parent, name, _ in planned
    )
    if wrapped_count != len(planned) or len(adapters) != 2 * wrapped_count:
        raise AssertionError("adapter installation did not wrap every requested projection")
    if (layer_start, layer_stop) == (12, 16) and wrapped_count != 24:
        raise AssertionError("production adapter installation must wrap 24 projections")
    return tuple(adapters)


def adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return the adapter tensors in a stable, name-sorted order."""
    state: dict[str, Tensor] = {}
    for module_name, module in sorted(model.named_modules(), key=lambda item: item[0]):
        if isinstance(module, LoRALinear):
            state[f"{module_name}.a"] = module.a
            state[f"{module_name}.b"] = module.b
    return state


def adapter_sha256(model: nn.Module) -> str:
    """Hash adapter names, dtypes, shapes, and canonical contiguous CPU bytes."""
    digest = sha256()
    for name, tensor in adapter_state_dict(model).items():
        contiguous = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(dimension) for dimension in contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def set_adapters_enabled(model: nn.Module, enabled: bool) -> None:
    """Enable or bypass every LoRA update without changing dense projections."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module._adapters_enabled.fill_(enabled)
