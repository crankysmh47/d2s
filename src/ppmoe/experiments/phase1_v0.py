"""Gated orchestration for the locked Phase 1 V0 oracle experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping

import torch
from transformers import AutoTokenizer

from ppmoe.adapters.qwen35_runtime import iter_text_mlps, load_qwen35_pair
from ppmoe.config import DatasetSpec, Phase1Config
from ppmoe.data.huggingface import load_wikitext_test, stream_fineweb
from ppmoe.data.token_stream import (
    iter_evaluation_batches,
    load_token_tensors,
    materialize_token_stream,
    with_stable_row_ids,
)
from ppmoe.evaluation.perplexity import evaluate_m0
from ppmoe.moefication.qwen35 import convert_qwen35_text_ffns
from ppmoe.oracle.local import run_local_oracle
from ppmoe.oracle.rollout import run_rollout_oracle
from ppmoe.provenance import atomic_write_json, sha256_file


class StageError(RuntimeError):
    """A stage failed a locked safety or numerical gate."""


class StagePrerequisiteError(StageError):
    """A required prior stage is absent, incomplete, or incompatible."""


RuntimeProbe = Callable[[], Mapping[str, object]]
RemoteProbe = Callable[[Phase1Config], Mapping[str, object]]


@dataclass(slots=True)
class Phase1Dependencies:
    """Replaceable boundaries for offline tests; defaults perform real work."""

    runtime_probe: RuntimeProbe | None = None
    remote_probe: RemoteProbe | None = None
    disk_free_probe: Callable[[], int] | None = None
    tokenizer_loader: Callable[..., Any] | None = None
    fineweb_loader: Callable[[DatasetSpec], Iterable[Mapping[str, Any]]] | None = None
    wikitext_loader: Callable[[DatasetSpec], Iterable[Mapping[str, Any]]] | None = None
    model_pair_loader: Callable[[Phase1Config], Any] | None = None
    m0_evaluator: Callable[..., Any] | None = None
    local_runner: Callable[..., Any] | None = None
    rollout_runner: Callable[..., Any] | None = None
    enforce_locked_model_dimensions: bool = True


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return value


def _canonical_hash(value: Mapping[str, object]) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _config_signature(config: Phase1Config) -> str:
    return _canonical_hash({"config": asdict(config)})


def _stage_path(artifact_root: Path, stage: str) -> Path:
    return artifact_root / "stages" / f"{stage}.json"


def _write_stage(
    artifact_root: Path,
    stage: str,
    config_signature: str,
    input_signature: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    state = {
        "config_signature": config_signature,
        "input_signature": input_signature,
        "output_hashes": {},
        "stage": stage,
        "status": "complete",
        **_json_safe(payload),
    }
    atomic_write_json(_stage_path(artifact_root, stage), state)
    return state


def _read_stage(artifact_root: Path, stage: str, config_signature: str) -> dict[str, Any]:
    path = _stage_path(artifact_root, stage)
    if not path.exists():
        raise StagePrerequisiteError(f"required stage {stage} is absent")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise StagePrerequisiteError(f"required stage {stage} is incomplete")
    if state.get("config_signature") != config_signature:
        raise StagePrerequisiteError(f"required stage {stage} signature does not match")
    return state


def _default_runtime_probe() -> Mapping[str, object]:
    import datasets
    import transformers

    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    free_disk = shutil.disk_usage(Path.cwd()).free
    cuda = torch.cuda.is_available()
    return {
        "bf16_available": bool(cuda and torch.cuda.is_bf16_supported()),
        "branch": branch,
        "cuda_available": cuda,
        "datasets": datasets.__version__,
        "free_disk_bytes": free_disk,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _default_remote_probe(config: Phase1Config) -> Mapping[str, object]:
    from huggingface_hub import HfApi

    api = HfApi()
    model = api.model_info(
        config.model.repo_id,
        revision=config.model.revision,
        files_metadata=True,
    )
    fineweb = api.dataset_info(config.fineweb.repo_id, revision=config.fineweb.revision)
    wikitext = api.dataset_info(config.wikitext.repo_id, revision=config.wikitext.revision)
    siblings = getattr(model, "siblings", ()) or ()
    sibling_sizes = [
        (str(getattr(item, "rfilename", "")), int(getattr(item, "size", 0) or 0))
        for item in siblings
    ]
    return {
        "expected_weight_bytes": sum(
            size for filename, size in sibling_sizes if filename.endswith(".safetensors")
        ),
        "expected_repository_bytes": sum(size for _, size in sibling_sizes),
        "fineweb_revision": getattr(fineweb, "sha", config.fineweb.revision),
        "model_revision": getattr(model, "sha", config.model.revision),
        "wikitext_revision": getattr(wikitext, "sha", config.wikitext.revision),
    }


def _dependencies(dependencies: Phase1Dependencies | None) -> Phase1Dependencies:
    values = dependencies or Phase1Dependencies()
    if values.runtime_probe is None:
        values.runtime_probe = _default_runtime_probe
    if values.remote_probe is None:
        values.remote_probe = _default_remote_probe
    if values.disk_free_probe is None:
        values.disk_free_probe = lambda: shutil.disk_usage(Path.cwd()).free
    if values.tokenizer_loader is None:
        values.tokenizer_loader = AutoTokenizer.from_pretrained
    if values.fineweb_loader is None:
        values.fineweb_loader = stream_fineweb
    if values.wikitext_loader is None:
        values.wikitext_loader = load_wikitext_test
    if values.model_pair_loader is None:
        values.model_pair_loader = lambda config: load_qwen35_pair(config, device=torch.device("cuda"))
    if values.m0_evaluator is None:
        values.m0_evaluator = evaluate_m0
    if values.local_runner is None:
        values.local_runner = run_local_oracle
    if values.rollout_runner is None:
        values.rollout_runner = run_rollout_oracle
    return values


def _preflight(
    config: Phase1Config,
    artifact_root: Path,
    config_signature: str,
    dependencies: Phase1Dependencies,
) -> dict[str, object]:
    assert dependencies.runtime_probe is not None and dependencies.remote_probe is not None
    runtime = dict(dependencies.runtime_probe())
    remote = dict(dependencies.remote_probe(config))
    if runtime.get("branch") != "exp":
        raise StageError("Phase 1 work must run on branch exp")
    if not runtime.get("cuda_available") or not runtime.get("bf16_available"):
        raise StageError("CUDA with BF16 support is required")
    expected_weight_bytes = int(remote.get("expected_weight_bytes", 0))
    expected_repository_bytes = int(remote.get("expected_repository_bytes", 0))
    if expected_weight_bytes <= 0:
        raise StageError("expected weight byte count must be positive")
    if expected_repository_bytes < expected_weight_bytes:
        raise StageError("expected repository byte count must cover the weight files")
    for key, expected in (
        ("model_revision", config.model.revision),
        ("fineweb_revision", config.fineweb.revision),
        ("wikitext_revision", config.wikitext.revision),
    ):
        if remote.get(key) != expected:
            raise StageError(f"remote {key} does not match the pinned protocol")
    return _write_stage(
        artifact_root,
        "preflight",
        config_signature,
        _canonical_hash({"config": config_signature, "runtime": runtime, "remote": remote}),
        {**runtime, **remote},
    )


def _manifest_reference(output_dir: Path, manifest: Any) -> dict[str, object]:
    return {
        "attention_mask_sha256": manifest.attention_mask_sha256,
        "final_context_length": manifest.final_context_length,
        "input_ids_sha256": manifest.input_ids_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "path": output_dir.as_posix(),
        "total_tokens": manifest.total_tokens,
    }


def _datasets(
    config: Phase1Config,
    artifact_root: Path,
    config_signature: str,
    dependencies: Phase1Dependencies,
) -> dict[str, object]:
    _read_stage(artifact_root, "preflight", config_signature)
    assert dependencies.tokenizer_loader is not None
    assert dependencies.fineweb_loader is not None and dependencies.wikitext_loader is not None
    tokenizer = dependencies.tokenizer_loader(
        config.model.repo_id,
        revision=config.model.revision,
        trust_remote_code=False,
    )
    token_root = artifact_root / "tokens"
    calibration = materialize_token_stream(
        dependencies.fineweb_loader(config.fineweb),
        tokenizer,
        dataset=config.fineweb,
        tokenizer_repo_id=config.model.repo_id,
        tokenizer_revision=config.model.revision,
        target_tokens=config.calibration_tokens,
        context_length=config.context_length,
        output_dir=token_root / "calibration",
    )
    heldout = materialize_token_stream(
        dependencies.fineweb_loader(config.fineweb),
        tokenizer,
        dataset=config.fineweb,
        tokenizer_repo_id=config.model.repo_id,
        tokenizer_revision=config.model.revision,
        target_tokens=config.heldout_tokens,
        context_length=config.context_length,
        output_dir=token_root / "fineweb-heldout",
        start_row_index=calibration.next_row_index,
        excluded_document_ids=frozenset(record.id for record in calibration.documents),
    )
    overlap = sorted(
        {record.id for record in calibration.documents}
        & {record.id for record in heldout.documents}
    )
    if overlap:
        raise StageError(f"FineWeb calibration/heldout overlap: {overlap}")
    wikitext = materialize_token_stream(
        with_stable_row_ids(dependencies.wikitext_loader(config.wikitext), "wikitext-test"),
        tokenizer,
        dataset=config.wikitext,
        tokenizer_repo_id=config.model.repo_id,
        tokenizer_revision=config.model.revision,
        target_tokens=None,
        context_length=config.context_length,
        output_dir=token_root / "wikitext",
    )
    if calibration.total_tokens != config.calibration_tokens or heldout.total_tokens != config.heldout_tokens:
        raise StageError("FineWeb materialization did not meet the locked token counts")
    streams = {
        "calibration": _manifest_reference(token_root / "calibration", calibration),
        "fineweb_heldout": _manifest_reference(token_root / "fineweb-heldout", heldout),
        "wikitext": _manifest_reference(token_root / "wikitext", wikitext),
    }
    return _write_stage(
        artifact_root,
        "datasets",
        config_signature,
        _canonical_hash({"config": config_signature, "streams": streams}),
        {
            "counts": {name: ref["total_tokens"] for name, ref in streams.items()},
            "overlap": overlap,
            "output_hashes": {name: ref["manifest_sha256"] for name, ref in streams.items()},
            "streams": streams,
        },
    )


def _disk_guard(dependencies: Phase1Dependencies) -> None:
    assert dependencies.disk_free_probe is not None
    if int(dependencies.disk_free_probe()) < 50 * 1024**3:
        raise StageError("at least 50 GiB free disk space is required before weight stages")


def _load_stream(reference: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    output_dir = Path(str(reference["path"]))
    if sha256_file(output_dir / "manifest.json") != reference["manifest_sha256"]:
        raise StageError("manifest hash does not match datasets stage")
    if sha256_file(output_dir / "input_ids.pt") != reference["input_ids_sha256"]:
        raise StageError("input_ids hash does not match datasets stage")
    if sha256_file(output_dir / "attention_mask.pt") != reference["attention_mask_sha256"]:
        raise StageError("attention mask hash does not match datasets stage")
    input_ids = torch.load(output_dir / "input_ids.pt", map_location="cpu", weights_only=True)
    attention_mask = torch.load(output_dir / "attention_mask.pt", map_location="cpu", weights_only=True)
    if int(attention_mask.sum()) != int(reference["total_tokens"]):
        raise StageError("persisted token count does not match datasets stage")
    return input_ids, attention_mask


def _converted_pair(
    config: Phase1Config,
    dependencies: Phase1Dependencies,
) -> tuple[Any, Any, Any, str, Any]:
    assert dependencies.model_pair_loader is not None
    pair = dependencies.model_pair_loader(config)
    if dependencies.enforce_locked_model_dimensions and (
        not hasattr(pair, "teacher_report") or not hasattr(pair, "student_report")
    ):
        raise StageError("model pair loader did not provide locked validation evidence")
    conversion = convert_qwen35_text_ffns(
        pair.student,
        num_experts=config.num_experts,
        seed_base=config.seed_base,
    )
    if dependencies.enforce_locked_model_dimensions and (
        conversion.converted_layers != config.num_hidden_layers
        or conversion.expert_width != config.expert_width
    ):
        raise StageError(
            "conversion dimensions do not match the locked 24-layer, 448-width protocol"
        )
    conversion_hash = _canonical_hash({"conversion": asdict(conversion)})
    return pair.teacher, pair.student, conversion, conversion_hash, pair


def _fallback_model_report(model: Any) -> dict[str, object]:
    parameters = tuple(model.parameters())
    floating = tuple(parameter for parameter in parameters if parameter.is_floating_point())
    return {
        "model_class": type(model).__name__,
        "text_layers": len(tuple(iter_text_mlps(model))),
        "floating_dtypes": sorted({str(parameter.dtype) for parameter in floating}),
        "devices": sorted({str(parameter.device) for parameter in parameters}),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
    }


def _model_manifest_payload(pair: Any) -> dict[str, object]:
    def report(role: str) -> dict[str, object]:
        loaded = getattr(pair, f"{role}_report", None)
        if loaded is not None and is_dataclass(loaded):
            return _json_safe(asdict(loaded))
        return _fallback_model_report(getattr(pair, role))

    return {
        "cuda": {
            "free_cuda_bytes": getattr(pair, "free_cuda_bytes", None),
            "peak_cuda_bytes": int(getattr(pair, "peak_cuda_bytes", 0)),
            "total_cuda_bytes": getattr(pair, "total_cuda_bytes", None),
        },
        "student": report("student"),
        "teacher": report("teacher"),
    }


def _persist_model_manifest(artifact_root: Path, pair: Any) -> str:
    return atomic_write_json(
        artifact_root / "manifests" / "model.json",
        _model_manifest_payload(pair),
    )


def _immutable_model_report(report: Mapping[str, Any]) -> dict[str, Any]:
    immutable_fields = (
        "model_class",
        "text_layers",
        "floating_dtypes",
        "parameter_count",
        "validated_num_hidden_layers",
        "validated_hidden_size",
        "validated_intermediate_size",
        "mlp_shapes",
    )
    return {field: report[field] for field in immutable_fields if field in report}


def _verify_model_manifest(artifact_root: Path, m0: Mapping[str, Any], pair: Any) -> None:
    path = artifact_root / "manifests" / "model.json"
    expected_file_hash = m0.get("model_file_sha256")
    if not path.exists() or not isinstance(expected_file_hash, str) or sha256_file(path) != expected_file_hash:
        raise StageError("model manifest file hash differs from M0")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fresh = _model_manifest_payload(pair)
    persisted_identity = {
        role: _immutable_model_report(payload[role]) for role in ("teacher", "student")
    }
    fresh_identity = {
        role: _immutable_model_report(fresh[role]) for role in ("teacher", "student")
    }
    if persisted_identity != fresh_identity:
        raise StageError("model manifest immutable identity differs from M0")


def _persist_conversion_manifest(artifact_root: Path, conversion: Any) -> str:
    return atomic_write_json(
        artifact_root / "manifests" / "conversion.json",
        _json_safe(asdict(conversion)),
    )


def _verify_conversion_manifest(
    artifact_root: Path,
    m0: Mapping[str, Any],
    conversion: Any,
    conversion_hash: str,
) -> None:
    path = artifact_root / "manifests" / "conversion.json"
    expected_file_hash = m0.get("conversion_file_sha256")
    if not isinstance(expected_file_hash, str) or sha256_file(path) != expected_file_hash:
        raise StageError("conversion manifest file hash differs from M0")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_payload = _json_safe(asdict(conversion))
    if payload != expected_payload or conversion_hash != m0.get("conversion_hash"):
        raise StageError("conversion manifest content differs from M0")


def _m0_checkpoint_dir(artifact_root: Path, corpus: str, signature: str) -> Path:
    return artifact_root / "checkpoints" / "m0" / corpus / signature


def _model_signature(config: Phase1Config) -> str:
    return _canonical_hash({"repo_id": config.model.repo_id, "revision": config.model.revision})


def _m0(
    config: Phase1Config,
    artifact_root: Path,
    config_signature: str,
    microbatch_size: int,
    dependencies: Phase1Dependencies,
) -> dict[str, object]:
    preflight = _read_stage(artifact_root, "preflight", config_signature)
    datasets = _read_stage(artifact_root, "datasets", config_signature)
    _disk_guard(dependencies)
    teacher, student, conversion, conversion_hash, pair = _converted_pair(config, dependencies)
    conversion_file_hash = _persist_conversion_manifest(artifact_root, conversion)
    model_file_hash = _persist_model_manifest(artifact_root, pair)
    assert dependencies.m0_evaluator is not None
    results: dict[str, Any] = {}
    for name in ("wikitext", "fineweb_heldout"):
        stream_hash = datasets["streams"][name]["manifest_sha256"]
        evaluation_signature = _canonical_hash({
            "config": config_signature,
            "conversion": conversion_hash,
            "microbatch_size": microbatch_size,
            "model": _model_signature(config),
            "tokens": stream_hash,
        })
        input_ids, attention_mask = _load_stream(datasets["streams"][name])
        result = dependencies.m0_evaluator(
            teacher,
            student,
            iter_evaluation_batches(input_ids, attention_mask, microbatch_size=microbatch_size),
            _m0_checkpoint_dir(artifact_root, name, evaluation_signature),
        )
        if not getattr(result, "passed", False):
            raise StageError(f"M0 gate failed on {name}")
        results[name] = _json_safe(result)
    signature = _canonical_hash({
        "config": config_signature,
        "conversion": conversion_hash,
        "microbatch_size": microbatch_size,
        "model": _model_signature(config),
        "streams": datasets["output_hashes"],
    })
    return _write_stage(
        artifact_root, "m0", config_signature, signature,
        {
            "conversion_file_sha256": conversion_file_hash,
            "conversion_hash": conversion_hash,
            "model_file_sha256": model_file_hash,
            "execution_microbatch_size": microbatch_size,
            "results": results,
            "output_hashes": {
                "conversion": conversion_file_hash,
                "model": model_file_hash,
            },
        },
    )


def _first_valid_tokens(input_ids: Tensor, attention_mask: Tensor, token_count: int) -> tuple[Tensor, Tensor]:
    values = input_ids[attention_mask.bool()]
    if values.numel() < token_count:
        raise StageError("calibration stream is shorter than the smoke target")
    values = values[:token_count]
    contexts = (token_count + input_ids.shape[1] - 1) // input_ids.shape[1]
    ids = torch.zeros((contexts, input_ids.shape[1]), dtype=input_ids.dtype)
    mask = torch.zeros_like(ids)
    ids.view(-1)[:token_count] = values
    mask.view(-1)[:token_count] = 1
    return ids, mask


def _local_stage(
    config: Phase1Config,
    artifact_root: Path,
    config_signature: str,
    microbatch_size: int,
    dependencies: Phase1Dependencies,
    *,
    smoke: bool,
) -> dict[str, object]:
    required = "m0" if smoke else "local-smoke"
    preflight = _read_stage(artifact_root, "preflight", config_signature)
    datasets = _read_stage(artifact_root, "datasets", config_signature)
    m0 = _read_stage(artifact_root, "m0", config_signature)
    _read_stage(artifact_root, required, config_signature)
    _disk_guard(dependencies)
    teacher, student, conversion, conversion_hash, pair = _converted_pair(config, dependencies)
    _verify_conversion_manifest(artifact_root, m0, conversion, conversion_hash)
    _verify_model_manifest(artifact_root, m0, pair)
    input_ids, attention_mask = _load_stream(datasets["streams"]["calibration"])
    if smoke:
        input_ids, attention_mask = _first_valid_tokens(input_ids, attention_mask, config.smoke_tokens)
    assert dependencies.local_runner is not None
    token_hash = datasets["streams"]["calibration"]["manifest_sha256"]
    signature = _canonical_hash({
        "config": config_signature, "conversion": conversion_hash,
        "microbatch_size": microbatch_size,
        "model": _model_signature(config), "tokens": token_hash,
    })
    result = dependencies.local_runner(
        teacher, student,
        iter_evaluation_batches(input_ids, attention_mask, microbatch_size=microbatch_size),
        k_values=config.k_values,
        checkpoint_dir=(
            artifact_root / "checkpoints" / ("local-smoke" if smoke else "local") / signature
        ),
        run_signature=signature,
    )
    expected_pairs = {
        (layer, k)
        for layer in range(conversion.converted_layers)
        for k in config.k_values
    }
    metrics_are_valid = all(
        metric.token_count > 0
        and isfinite(metric.squared_error_sum)
        and isfinite(metric.target_squared_norm_sum)
        and isfinite(metric.normalized_mse)
        for metric in result.layer_k.values()
    )
    if (
        not result.complete
        or result.direct_check_count <= 0
        or result.all_expert_check_count <= 0
        or set(result.layer_k) != expected_pairs
        or not metrics_are_valid
    ):
        raise StageError("local oracle smoke/completion gate failed")
    stage = "local-smoke" if smoke else "local"
    return _write_stage(
        artifact_root, stage, config_signature, signature,
        {
            "execution_microbatch_size": microbatch_size,
            "result": _json_safe(result),
            "output_hashes": {"tokens": token_hash},
        },
    )


def _rollout(
    config: Phase1Config,
    artifact_root: Path,
    config_signature: str,
    microbatch_size: int,
    dependencies: Phase1Dependencies,
) -> dict[str, object]:
    preflight = _read_stage(artifact_root, "preflight", config_signature)
    datasets = _read_stage(artifact_root, "datasets", config_signature)
    m0 = _read_stage(artifact_root, "m0", config_signature)
    _read_stage(artifact_root, "local", config_signature)
    _disk_guard(dependencies)
    assert dependencies.rollout_runner is not None
    results: dict[str, Any] = {}
    teacher, student, conversion, conversion_hash, pair = _converted_pair(config, dependencies)
    _verify_conversion_manifest(artifact_root, m0, conversion, conversion_hash)
    _verify_model_manifest(artifact_root, m0, pair)
    for corpus in ("wikitext", "fineweb_heldout"):
        input_ids, attention_mask = _load_stream(datasets["streams"][corpus])
        for k in config.k_values:
            signature = _canonical_hash({
                "config": config_signature, "conversion": conversion_hash,
                "k": k, "microbatch_size": microbatch_size, "model": _model_signature(config),
                "tokens": datasets["streams"][corpus]["manifest_sha256"],
            })
            result = dependencies.rollout_runner(
                teacher, student,
                iter_evaluation_batches(input_ids, attention_mask, microbatch_size=microbatch_size),
                k=k,
                checkpoint_dir=(
                    artifact_root / "checkpoints" / "rollout" / corpus / f"k{k}" / signature
                ),
                run_signature=signature,
            )
            if not result.complete:
                raise StageError(f"rollout is incomplete for {corpus} K={k}")
            results[f"{corpus}:k{k}"] = _json_safe(result)
    signature = _canonical_hash({
        "config": config_signature,
        "m0": m0["input_signature"],
        "microbatch_size": microbatch_size,
        "streams": datasets["output_hashes"],
    })
    return _write_stage(
        artifact_root,
        "rollout",
        config_signature,
        signature,
        {
            "execution_microbatch_size": microbatch_size,
            "output_hashes": {"results": _canonical_hash({"results": results})},
            "results": results,
        },
    )


def _report(
    config: Phase1Config,
    artifact_root: Path,
    config_signature: str,
) -> dict[str, object]:
    """Render only after the rollout stage has completed compatibly."""

    rollout = _read_stage(artifact_root, "rollout", config_signature)
    from ppmoe.reporting.phase1 import render_phase1_report

    manifest = render_phase1_report(artifact_root, config)
    input_signature = _canonical_hash({
        "config": config_signature,
        "rollout": rollout["input_signature"],
        "sources": manifest.source_stage_hashes,
    })
    return _write_stage(
        artifact_root,
        "report",
        config_signature,
        input_signature,
        {
            "manifest": _json_safe(manifest),
            "output_hashes": dict(manifest.output_hashes),
        },
    )


def run_phase1(
    config: Phase1Config,
    stage: str,
    resume: bool,
    artifact_root: Path,
    microbatch_size: int,
    *,
    dependencies: Phase1Dependencies | None = None,
) -> Mapping[str, object]:
    """Execute exactly one gated Phase 1 stage."""

    if stage not in {"preflight", "datasets", "m0", "local-smoke", "local", "rollout", "report"}:
        raise ValueError(f"unsupported Phase 1 stage: {stage}")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    artifact_root = Path(artifact_root)
    config_signature = _config_signature(config)
    if resume and _stage_path(artifact_root, stage).exists():
        resumed = _read_stage(artifact_root, stage, config_signature)
        if stage not in {"m0", "local-smoke", "local", "rollout"} or resumed.get(
            "execution_microbatch_size"
        ) == microbatch_size:
            return resumed
    values = _dependencies(dependencies)
    if stage == "preflight":
        return _preflight(config, artifact_root, config_signature, values)
    if stage == "datasets":
        return _datasets(config, artifact_root, config_signature, values)
    if stage == "m0":
        return _m0(config, artifact_root, config_signature, microbatch_size, values)
    if stage == "local-smoke":
        return _local_stage(config, artifact_root, config_signature, microbatch_size, values, smoke=True)
    if stage == "local":
        return _local_stage(config, artifact_root, config_signature, microbatch_size, values, smoke=False)
    if stage == "report":
        return _report(config, artifact_root, config_signature)
    return _rollout(config, artifact_root, config_signature, microbatch_size, values)
