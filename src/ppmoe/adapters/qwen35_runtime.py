"""Guarded runtime loading for the composite Qwen3.5 checkpoint."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers import Qwen3_5ForConditionalGeneration

from ppmoe.config import Phase1Config


@dataclass(frozen=True, slots=True)
class MLPShapeEvidence:
    layer_index: int
    gate_proj: tuple[int, int]
    up_proj: tuple[int, int]
    down_proj: tuple[int, int]


@dataclass(frozen=True, slots=True)
class LoadedModelReport:
    model_class: str
    text_layers: int
    floating_dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    parameter_count: int
    validated_num_hidden_layers: int
    validated_hidden_size: int
    validated_intermediate_size: int
    mlp_shapes: tuple[MLPShapeEvidence, ...]


@dataclass(frozen=True, slots=True)
class LoadedModelPair:
    teacher: Qwen3_5ForConditionalGeneration
    student: Qwen3_5ForConditionalGeneration
    teacher_report: LoadedModelReport
    student_report: LoadedModelReport
    peak_cuda_bytes: int
    free_cuda_bytes: int | None
    total_cuda_bytes: int | None


def _text_layers(model: Qwen3_5ForConditionalGeneration) -> nn.ModuleList:
    try:
        layers = model.model.language_model.layers
    except AttributeError as error:
        raise ValueError("Qwen3.5 checkpoint is missing model.language_model.layers") from error
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("Qwen3.5 text layers must be an nn.ModuleList")
    expected = int(model.config.text_config.num_hidden_layers)
    if len(layers) != expected:
        raise ValueError(f"expected {expected} text layers; found {len(layers)}")
    return layers


def iter_text_mlps(
    model: Qwen3_5ForConditionalGeneration,
) -> Iterator[tuple[int, nn.Module]]:
    for index, layer in enumerate(_text_layers(model)):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, nn.Module):
            raise ValueError(f"text layer {index} is missing an MLP")
        yield index, mlp


def validate_loaded_qwen35(
    model: nn.Module,
    *,
    expected_dtype: torch.dtype,
    expected_num_hidden_layers: int,
    expected_hidden_size: int,
    expected_intermediate_size: int,
) -> LoadedModelReport:
    if not isinstance(model, Qwen3_5ForConditionalGeneration):
        raise TypeError("model must be Qwen3_5ForConditionalGeneration")
    layers = _text_layers(model)
    if len(layers) != expected_num_hidden_layers:
        raise ValueError(
            f"expected {expected_num_hidden_layers} locked text layers; found {len(layers)}"
        )
    text_config = model.config.text_config
    if int(text_config.hidden_size) != expected_hidden_size:
        raise ValueError(
            f"expected locked hidden size {expected_hidden_size}; found {text_config.hidden_size}"
        )
    if int(text_config.intermediate_size) != expected_intermediate_size:
        raise ValueError(
            "expected locked intermediate size "
            f"{expected_intermediate_size}; found {text_config.intermediate_size}"
        )
    mlp_shapes: list[MLPShapeEvidence] = []
    for layer_index, mlp in iter_text_mlps(model):
        projections: dict[str, Tensor] = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight = getattr(getattr(mlp, name, None), "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                raise ValueError(f"text layer {layer_index} {name} projection must have a matrix weight")
            projections[name] = weight
        expected_shapes = {
            "gate_proj": (expected_intermediate_size, expected_hidden_size),
            "up_proj": (expected_intermediate_size, expected_hidden_size),
            "down_proj": (expected_hidden_size, expected_intermediate_size),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(int(value) for value in projections[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"text layer {layer_index} {name.replace('_proj', '')} projection "
                    f"must have shape {expected_shape}; found {actual_shape}"
                )
        mlp_shapes.append(
            MLPShapeEvidence(
                layer_index=layer_index,
                gate_proj=tuple(int(value) for value in projections["gate_proj"].shape),
                up_proj=tuple(int(value) for value in projections["up_proj"].shape),
                down_proj=tuple(int(value) for value in projections["down_proj"].shape),
            )
        )
    floating_parameters = [
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ]
    dtypes = tuple(sorted({str(parameter.dtype) for parameter in floating_parameters}))
    if dtypes != (str(expected_dtype),):
        raise ValueError(
            f"parameter dtypes must be only {expected_dtype}; observed {dtypes}"
        )
    devices = tuple(sorted({str(parameter.device) for parameter in model.parameters()}))
    if not floating_parameters:
        raise ValueError("Qwen3.5 model has no floating-point parameters")
    return LoadedModelReport(
        model_class=type(model).__name__,
        text_layers=len(layers),
        floating_dtypes=dtypes,
        devices=devices,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        validated_num_hidden_layers=expected_num_hidden_layers,
        validated_hidden_size=expected_hidden_size,
        validated_intermediate_size=expected_intermediate_size,
        mlp_shapes=tuple(mlp_shapes),
    )


def load_qwen35_pair(
    config: Phase1Config,
    *,
    device: torch.device,
    loader: Callable[..., Any] = Qwen3_5ForConditionalGeneration.from_pretrained,
) -> LoadedModelPair:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but unavailable")
        torch.cuda.reset_peak_memory_stats(device)
    kwargs: dict[str, object] = {
        "revision": config.model.revision,
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
    }
    teacher = loader(config.model.repo_id, **kwargs)
    student = loader(config.model.repo_id, **kwargs)
    if teacher is student:
        raise ValueError("teacher and student loader results must be distinct")
    if not isinstance(teacher, nn.Module) or not isinstance(student, nn.Module):
        raise TypeError("model loader must return torch modules")
    teacher = teacher.to(device).eval()
    student = student.to(device).eval()
    teacher_report = validate_loaded_qwen35(
        teacher,
        expected_dtype=torch.bfloat16,
        expected_num_hidden_layers=config.num_hidden_layers,
        expected_hidden_size=config.hidden_size,
        expected_intermediate_size=config.intermediate_size,
    )
    student_report = validate_loaded_qwen35(
        student,
        expected_dtype=torch.bfloat16,
        expected_num_hidden_layers=config.num_hidden_layers,
        expected_hidden_size=config.hidden_size,
        expected_intermediate_size=config.intermediate_size,
    )
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    free: int | None = None
    total: int | None = None
    if device.type == "cuda":
        free, total = (int(value) for value in torch.cuda.mem_get_info(device))
        if free < 4 * 1024**3:
            raise RuntimeError("at least 4 GiB free CUDA memory is required after loading both models")
    return LoadedModelPair(
        teacher=teacher,
        student=student,
        teacher_report=teacher_report,
        student_report=student_report,
        peak_cuda_bytes=peak,
        free_cuda_bytes=free,
        total_cuda_bytes=total,
    )
