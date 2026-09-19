"""The registered Task 16 real-model G1 correctness smoke."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .gates import CudaCostLedger
from .lora import adapter_sha256, install_adapters, set_adapters_enabled
from .model import RecurrentQwen3
from .provenance import append_event, canonical_json, sha256_payload
from .train import FreshProcessRequired, _is_cuda_process_fault

if TYPE_CHECKING:
    from .experiment import StageContext


_SMOKE_SCHEMA_VERSION = 1
_FAMILIES = ("expressions", "lists", "graphs")
_PROMPT_COUNTS = {"expressions": 3, "lists": 3, "graphs": 2}
_DEV_EXPECTED_IDS = (
    "80a57b0a7fbbfb52358af4be489158547b41e0dc72c470c3bfd477b10d1e370b",
    "1a58e59f7a7ff169ff98eeeca9eaf66e1172c11694c2be5339a923a7cd9a72a2",
    "7a4fceaf912fdc66ff9cd9eb58f381c0f8bddf7028fdc23047e2c337108b119c",
    "20af5310d856a01d57bd621d3e922b81c6c8ab7b87e8705ef630743074fc5053",
    "7cef763be607ea74252b9dfb72d32eb3ec181d22c76d90ee1ff81ed88c4d89e1",
    "5d4bf4b1e024b00531c571bb6c5c00d7ef674259bfd5881645980b9d4c6ca526",
    "1327bca00f4e22d3ce05f7cd2b0e7ba1ed7dd9ffe492d53b90969fbf9149e4c4",
    "79c84ed449c458a86ac809add5e23a5aab554c516c44a1335bd055cbdc4ff099",
)
_DEV_BYTES_SHA256 = "63af435fc3148ba344d4be63f7598276404cacb4103a0da0db8f7d0f537afbca"
_DEV_MANIFEST_SHA256 = "aafff1f7acffd594c5629fb2e2573a57c7ed3ae776aae61758a77a746a8c291d"
_DEV_SELECTED_SHA256 = "8856a0c164d566bfeb035135c4ae92df47e7db26af310ef6a0b3726be33a2c54"
_MODEL_LAYER_COUNT = 28
_SELECTED_LAYER_START = 12
_SELECTED_LAYER_STOP = 16
_TARGET_PROJECTIONS = ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.o_proj", "self_attn.q_proj", "self_attn.v_proj")
_SMOKE_REQUIRED_PAYLOAD_KEYS = frozenset({
    "schema_version", "model_class", "prompt_count", "prompt_ids",
    "dev_bytes_sha256", "dev_manifest_sha256", "selected_rows_sha256",
    "identity_prompts", "identity_max_abs_diff", "identity_min_cosine",
    "identity_passed", "wrapper_stock_model_shared", "selected_layer_ids",
    "selected_base_projection_ids", "selected_projection_ids", "selected_parameter_ids", "layer_call_counts",
    "selected_projection_invocation_ids", "selected_projection_objects_shared",
    "layer_call_counts_passed", "cache_disabled", "causality_passed",
    "causality_max_abs_diff", "causality_max_abs_diff_by_mode", "cache_observations", "shared_layer_objects", "adapter_off_f0_finite",
    "adapter_off_f1_finite", "adapter_off_f0_nonfinite_count", "adapter_off_f1_nonfinite_count", "adapter_off_a1_matches_f0", "adapter_off_a2_matches_f1", "adapter_off_a1_max_abs_diff", "adapter_off_a2_max_abs_diff",
    "adapter_gradients_finite", "adapter_gradients_nonzero", "adapter_gradient_evidence",
    "lora_zero_init_nuance",
    "suffix_input_gradient", "suffix_gradient_finite", "suffix_gradient_nonzero", "suffix_gradient_abs_sum",
    "gradient_loss", "adapter_initial_hash", "adapter_final_hash",
    "adapter_hash_unchanged", "adapter_checkpoint_written", "disposable_adapter",
    "checkpoint_artifacts_unchanged",
    "cuda_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "total_vram_bytes",
    "download_bytes",
})


def _validate_smoke_payload(payload: Mapping[str, object]) -> None:
    missing = sorted(_SMOKE_REQUIRED_PAYLOAD_KEYS.difference(payload))
    if missing:
        raise RuntimeError(f"smoke payload missing required fields: {', '.join(missing)}")
    unexpected = sorted(set(payload).difference(_SMOKE_REQUIRED_PAYLOAD_KEYS))
    if unexpected:
        raise RuntimeError(f"smoke payload has unexpected fields: {', '.join(unexpected)}")
    if payload.get("schema_version") != _SMOKE_SCHEMA_VERSION or payload.get("model_class") != "Qwen3ForCausalLM":
        raise RuntimeError("smoke payload schema is invalid")
    if payload.get("prompt_count") != 8 or payload.get("prompt_ids") != list(_DEV_EXPECTED_IDS):
        raise RuntimeError("smoke payload prompt binding is invalid")
    if payload.get("dev_bytes_sha256") != _DEV_BYTES_SHA256 or payload.get("dev_manifest_sha256") != _DEV_MANIFEST_SHA256 or payload.get("selected_rows_sha256") != _DEV_SELECTED_SHA256:
        raise RuntimeError("smoke payload data binding is invalid")
    for name in ("identity_passed", "layer_call_counts_passed", "cache_disabled", "causality_passed", "shared_layer_objects", "wrapper_stock_model_shared", "adapter_gradients_finite", "adapter_gradients_nonzero", "suffix_input_gradient", "suffix_gradient_finite", "suffix_gradient_nonzero", "adapter_hash_unchanged", "disposable_adapter", "checkpoint_artifacts_unchanged"):
        if payload.get(name) is not True:
            raise RuntimeError(f"smoke payload gate is false: {name}")
    if payload.get("lora_zero_init_nuance") != "LoRA B is zero-initialized; A-gradient zero at first backward is permitted, while every B-gradient must be finite and nonzero.":
        raise RuntimeError("smoke payload LoRA initialization note is invalid")
    for name in ("adapter_checkpoint_written",):
        if payload.get(name) is not False:
            raise RuntimeError(f"smoke payload disposable contract is invalid: {name}")
    for name in ("cuda_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "total_vram_bytes", "download_bytes", "gradient_loss"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
            raise RuntimeError(f"smoke payload numeric field is invalid: {name}")
    if int(payload["total_vram_bytes"]) <= 0 or int(payload["peak_reserved_bytes"]) < 0:
        raise RuntimeError("smoke payload VRAM fields are invalid")
    if not isinstance(payload.get("selected_layer_ids"), list) or not isinstance(payload.get("selected_base_projection_ids"), list) or not isinstance(payload.get("selected_projection_ids"), list) or not isinstance(payload.get("selected_parameter_ids"), list):
        raise RuntimeError("smoke payload parameter identity is invalid")
    if not isinstance(payload.get("selected_projection_invocation_ids"), Mapping) or payload.get("selected_projection_objects_shared") is not True:
        raise RuntimeError("smoke payload projection invocation identity is invalid")
    def finite(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    identity = payload.get("identity_prompts")
    if not isinstance(identity, list) or len(identity) != 8:
        raise RuntimeError("smoke payload identity evidence is incomplete")
    for row in identity:
        if not isinstance(row, Mapping) or set(row) != {"max_abs_diff", "cosine"} or not finite(row["max_abs_diff"]) or not finite(row["cosine"]):
            raise RuntimeError("smoke payload identity evidence is malformed")
        if float(row["max_abs_diff"]) > 1e-3 or float(row["cosine"]) < 0.999999:
            raise RuntimeError("smoke payload identity threshold failed")
    max_diff = max(float(row["max_abs_diff"]) for row in identity)
    min_cosine = min(float(row["cosine"]) for row in identity)
    if float(payload["identity_max_abs_diff"]) != max_diff or float(payload["identity_min_cosine"]) != min_cosine:
        raise RuntimeError("smoke payload identity aggregates are inconsistent")
    counts = payload.get("layer_call_counts")
    if not isinstance(counts, Mapping) or set(counts) != {"single", "single_expected", "repeat", "repeat_expected", "single_total", "repeat_total"}:
        raise RuntimeError("smoke payload layer-count evidence is malformed")
    expected_single = {str(index): 1 for index in range(_MODEL_LAYER_COUNT)}
    expected_repeat = {str(index): (2 if _SELECTED_LAYER_START <= index < _SELECTED_LAYER_STOP else 1) for index in range(_MODEL_LAYER_COUNT)}
    for name, expected in (("single", expected_single), ("single_expected", expected_single), ("repeat", expected_repeat), ("repeat_expected", expected_repeat)):
        value = counts[name]
        if value != expected:
            raise RuntimeError(f"smoke payload {name} layer map is invalid")
    if counts["single_total"] != 28 or counts["repeat_total"] != 32 or sum(counts["single"].values()) != counts["single_total"] or sum(counts["repeat"].values()) != counts["repeat_total"]:
        raise RuntimeError("smoke payload layer totals are invalid")
    layer_ids = payload["selected_layer_ids"]
    if len(layer_ids) != 4 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in layer_ids) or len(set(layer_ids)) != 4:
        raise RuntimeError("smoke payload selected layer identity is invalid")
    expected_projection_rows = [(layer, projection) for layer in range(_SELECTED_LAYER_START, _SELECTED_LAYER_STOP) for projection in _TARGET_PROJECTIONS]
    base_projections = payload["selected_base_projection_ids"]
    projections = payload["selected_projection_ids"]
    parameters = payload["selected_parameter_ids"]
    if len(base_projections) != 24 or len(projections) != 24 or len(parameters) != 24:
        raise RuntimeError("smoke payload selected projection cardinality is invalid")
    base_object_ids: list[int] = []
    base_by_path: dict[tuple[int, str], int] = {}
    for row, (layer, projection) in zip(base_projections, expected_projection_rows):
        if not isinstance(row, Mapping) or set(row) != {"layer", "projection", "object_id"} or row["layer"] != layer or row["projection"] != projection:
            raise RuntimeError("smoke payload selected base projection identity is invalid")
        if isinstance(row["object_id"], bool) or not isinstance(row["object_id"], int) or row["object_id"] <= 0:
            raise RuntimeError("smoke payload selected base projection object ID is invalid")
        base_object_ids.append(row["object_id"])
        base_by_path[(layer, projection)] = row["object_id"]
    if len(set(base_object_ids)) != 24:
        raise RuntimeError("smoke payload selected base projections are not distinct")
    wrapper_object_ids: list[int] = []
    post_base_object_ids: list[int] = []
    for row, (layer, projection) in zip(projections, expected_projection_rows):
        if not isinstance(row, Mapping) or set(row) != {"layer", "projection", "wrapper_object_id", "base_object_id"} or row["layer"] != layer or row["projection"] != projection:
            raise RuntimeError("smoke payload selected projection identity is invalid")
        if any(isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] <= 0 for field in ("wrapper_object_id", "base_object_id")):
            raise RuntimeError("smoke payload selected projection object ID is invalid")
        if row["wrapper_object_id"] == row["base_object_id"] or row["base_object_id"] != base_by_path[(layer, projection)]:
            raise RuntimeError("smoke payload selected projection identity is invalid")
        wrapper_object_ids.append(row["wrapper_object_id"])
        post_base_object_ids.append(row["base_object_id"])
    if len(set(wrapper_object_ids)) != 24 or len(set(post_base_object_ids)) != 24:
        raise RuntimeError("smoke payload selected projections are not distinct")
    parameter_ids: list[int] = []
    base_weight_ids: list[int] = []
    for row, (layer, projection) in zip(parameters, expected_projection_rows):
        if not isinstance(row, Mapping) or set(row) != {"layer", "projection", "a_id", "b_id", "base_weight_id"} or row["layer"] != layer or row["projection"] != projection:
            raise RuntimeError("smoke payload selected parameter identity is invalid")
        if any(isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] <= 0 for field in ("a_id", "b_id", "base_weight_id")):
            raise RuntimeError("smoke payload selected parameter ID is invalid")
        parameter_ids.extend((row["a_id"], row["b_id"]))
        base_weight_ids.append(row["base_weight_id"])
    if len(set(parameter_ids)) != 48 or len(set(base_weight_ids)) != 24:
        raise RuntimeError("smoke payload selected parameters are not distinct")
    invocations = payload["selected_projection_invocation_ids"]
    if set(invocations) != {"single", "repeat"} or not isinstance(invocations["single"], list) or not isinstance(invocations["repeat"], list) or len(invocations["single"]) != 24 or len(invocations["repeat"]) != 48:
        raise RuntimeError("smoke payload projection invocation cardinality is invalid")
    for mode, expected_count in (("single", 1), ("repeat", 2)):
        values = invocations[mode]
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"layer", "projection", "object_id"}
            or isinstance(item["layer"], bool)
            or not isinstance(item["layer"], int)
            or not isinstance(item["projection"], str)
        for item in values):
            raise RuntimeError("smoke payload projection invocation identity is invalid")
        paths = [(item["layer"], item["projection"]) for item in values]
        if any(path not in base_by_path or isinstance(item["object_id"], bool) or not isinstance(item["object_id"], int) or item["object_id"] <= 0 for item, path in zip(values, paths)):
            raise RuntimeError("smoke payload projection invocation identity is invalid")
        if any(item["object_id"] != base_by_path[path] for item, path in zip(values, paths)):
            raise RuntimeError("smoke payload projection invocation identity is invalid")
        if set(paths) != set(expected_projection_rows) or any(paths.count(path) != expected_count for path in set(paths)):
            raise RuntimeError("smoke payload projection invocation reuse is invalid")
    if payload["selected_projection_objects_shared"] is not (
        set(base_object_ids) == {item["object_id"] for item in invocations["single"]}
        == {item["object_id"] for item in invocations["repeat"]}
    ):
        raise RuntimeError("smoke payload projection invocation reuse is invalid")
    observations = payload["cache_observations"]
    if not isinstance(observations, list) or len(observations) != 224 or any(not isinstance(item, Mapping) or set(item) != {"past_key_values", "use_cache"} or item["past_key_values"] is not None or item["use_cache"] is not False for item in observations):
        raise RuntimeError("smoke payload cache evidence is invalid")
    causal_by_mode = payload["causality_max_abs_diff_by_mode"]
    if not isinstance(causal_by_mode, Mapping) or set(causal_by_mode) != {"single", "repeat", "damped"} or any(not finite(value) or float(value) != 0.0 for value in causal_by_mode.values()) or float(payload["causality_max_abs_diff"]) != 0.0:
        raise RuntimeError("smoke payload causality evidence is invalid")
    if payload["adapter_off_f0_nonfinite_count"] != 0 or payload["adapter_off_f1_nonfinite_count"] != 0 or float(payload["adapter_off_a1_max_abs_diff"]) != 0.0 or float(payload["adapter_off_a2_max_abs_diff"]) != 0.0:
        raise RuntimeError("smoke payload adapter-off numeric evidence is invalid")
    evidence = payload["adapter_gradient_evidence"]
    if not isinstance(evidence, list) or len(evidence) != 24:
        raise RuntimeError("smoke payload adapter-gradient evidence is incomplete")
    for index, row in enumerate(evidence):
        fields = {"adapter_index", "a_finite", "a_nonzero", "b_finite", "b_nonzero", "a_zero_at_init_allowed", "a_abs_sum", "b_abs_sum"}
        if not isinstance(row, Mapping) or set(row) != fields or row["adapter_index"] != index or any(not isinstance(row[field], bool) for field in ("a_finite", "a_nonzero", "b_finite", "b_nonzero", "a_zero_at_init_allowed")) or not finite(row["a_abs_sum"]) or not finite(row["b_abs_sum"]):
            raise RuntimeError("smoke payload adapter-gradient evidence is malformed")
        if row["a_finite"] is not True or row["b_finite"] is not True or row["b_nonzero"] is not True or float(row["b_abs_sum"]) <= 0.0 or row["a_nonzero"] != (float(row["a_abs_sum"]) > 0.0) or row["a_zero_at_init_allowed"] is not True:
            raise RuntimeError("smoke payload adapter-gradient evidence failed")
    if payload["adapter_gradients_finite"] is not all(row["a_finite"] and row["b_finite"] for row in evidence) or payload["adapter_gradients_nonzero"] is not all(row["b_nonzero"] for row in evidence):
        raise RuntimeError("smoke payload adapter-gradient aggregates are inconsistent")
    if not finite(payload["suffix_gradient_abs_sum"]) or float(payload["suffix_gradient_abs_sum"]) <= 0.0 or payload["suffix_input_gradient"] is not True or payload["suffix_gradient_finite"] is not True or payload["suffix_gradient_nonzero"] is not True:
        raise RuntimeError("smoke payload suffix-gradient evidence is invalid")
    if not isinstance(payload["adapter_initial_hash"], str) or not isinstance(payload["adapter_final_hash"], str) or __import__("re").fullmatch(r"[0-9a-f]{64}", payload["adapter_initial_hash"]) is None or __import__("re").fullmatch(r"[0-9a-f]{64}", payload["adapter_final_hash"]) is None or payload["adapter_initial_hash"] != payload["adapter_final_hash"] or payload["adapter_hash_unchanged"] is not True:
        raise RuntimeError("smoke payload adapter hash evidence is invalid")


def _smoke_failure_event_id(event: Mapping[str, object]) -> str:
    """Canonical identity for smoke failure diagnostics inside stage payload."""
    raw_payload = event.get("payload")
    payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else dict(event)
    payload.pop("event_id", None)
    return sha256_payload({
        "kind": event.get("kind", "stage_attempt"),
        "stage": event.get("stage"),
        "attempt": event.get("attempt"),
        "outcome": event.get("outcome"),
        "payload": payload,
    })


def _smoke_retry_artifact(run_root: Path) -> dict[str, object]:
    """Authenticate a disposable smoke retry without inventing a checkpoint."""
    events_path = run_root / "events.jsonl"
    from .experiment import _read_test_bytes_once
    if not events_path.is_file():
        raise RuntimeError("smoke retry has no durable CUDA failure event")
    failures = []
    for line in _read_test_bytes_once(events_path, trusted_root=run_root).decode("utf-8").splitlines():
        value = json.loads(line)
        if (
            isinstance(value, Mapping)
            and value.get("kind") == "stage_attempt"
            and value.get("stage") == "smoke"
            and value.get("attempt") == 1
            and value.get("outcome") == "fresh_process_required"
            and isinstance(value.get("payload"), Mapping)
            and value["payload"].get("cuda_process_fault") is True
        ):
            failures.append(value)
    if not failures:
        raise RuntimeError("smoke retry has no durable CUDA failure event")
    failure = failures[-1]
    failure_payload = failure["payload"]
    event_id = failure_payload.get("event_id")
    event_index = failure.get("index")
    if (
        not isinstance(event_id, str)
        or __import__("re").fullmatch(r"[0-9a-f]{64}", event_id) is None
        or event_id != _smoke_failure_event_id(failure)
        or not isinstance(event_index, int)
    ):
        raise RuntimeError("smoke retry failure event identity is invalid")
    return {
        "path": "", "checkpoint_sha256": "", "manifest_digest": "",
        "attempt": 1, "update": 0, "event_id": event_id,
        "event_index": event_index, "arm": "smoke", "seed": 0,
        "relative_path": "", "adapter_schema": None,
    }


def run_production_smoke(context: "StageContext") -> Mapping[str, object]:
    """Load the pinned local checkpoint and execute the immutable G1 checks."""
    from .experiment import _source_snapshot_path, _stage_path, _lazy_transformers_loader

    def load() -> tuple[nn.Module, object]:
        source_record = _safe_stage_json(context.run_root, _stage_path(context.run_root, "source"))
        source_payload = source_record.get("payload")
        if not isinstance(source_payload, Mapping):
            raise RuntimeError("source record has no production payload")
        snapshot = _source_snapshot_path(context.run_root, source_payload)
        revision = source_payload.get("revision")
        if not snapshot.is_dir() or not isinstance(revision, str):
            raise RuntimeError("source snapshot is missing or unpinned")
        tokenizer_cls, model_cls = _lazy_transformers_loader()
        tokenizer = tokenizer_cls.from_pretrained(str(snapshot), revision=revision, local_files_only=True)
        stock = model_cls.from_pretrained(
            str(snapshot), revision=revision, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager",
        )
        return stock, tokenizer

    return _run_cuda_attempt(context, load)


def run_child_smoke(request: Mapping[str, object]) -> Mapping[str, object]:
    """Registered fresh-process entry point.

    Task 16 uses disposable adapters and therefore has no training checkpoint
    to restore.  The normal smoke path never needs this entry point; it is
    registered so a future CUDA-fault retry cannot silently fall back to a
    test helper or an unbound dispatcher.
    """
    from .config import RecurrenceConfig
    from .experiment import StageContext, _lazy_transformers_loader

    config_path = Path(str(request["config_path"])).resolve()
    run_root = Path(str(request["run_root"])).resolve()
    config = RecurrenceConfig.load(config_path)
    signatures = request.get("signatures")
    if not isinstance(signatures, Mapping):
        raise RuntimeError("fresh smoke request has invalid signatures")
    context = StageContext(
        "smoke", config, config_path, Path(str(request["artifact_root"])).resolve(),
        run_root, dict(signatures), 2, lambda path: path.read_bytes(),
    )
    from .experiment import _source_snapshot_path, _stage_path
    def load() -> tuple[nn.Module, object]:
        source_payload = _safe_stage_json(run_root, _stage_path(run_root, "source")).get("payload")
        if not isinstance(source_payload, Mapping):
            raise RuntimeError("source record has no production payload")
        snapshot = _source_snapshot_path(run_root, source_payload)
        revision = source_payload.get("revision")
        if not isinstance(revision, str):
            raise RuntimeError("source revision is invalid")
        tokenizer_cls, model_cls = _lazy_transformers_loader()
        tokenizer = tokenizer_cls.from_pretrained(str(snapshot), revision=revision, local_files_only=True)
        stock = model_cls.from_pretrained(
            str(snapshot), revision=revision, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager",
        )
        return stock, tokenizer

    payload = dict(_run_cuda_attempt(context, load))
    payload.update({
        "worker_pid": __import__("os").getpid(),
        "attempt": request["attempt"],
        "checkpoint_path": request["checkpoint_path"],
        "checkpoint_digest": request["checkpoint_digest"],
        "checkpoint_sha256": request["checkpoint_sha256"],
        "checkpoint_attempt": request["checkpoint_attempt"],
        "checkpoint_update": request["checkpoint_update"],
        "checkpoint_event_id": request["checkpoint_event_id"],
        "checkpoint_event_index": request["checkpoint_event_index"],
        "checkpoint_arm": request["checkpoint_arm"],
        "checkpoint_seed": request["checkpoint_seed"],
        "checkpoint_relative_path": request["checkpoint_relative_path"],
        "checkpoint_signature": request["checkpoint_signature"],
        "dispatcher_stage": request["stage"],
        "dispatcher_ref": request["worker_ref"],
        "dispatcher_identity": request["worker_identity"],
        "checkpoint_proof": request["checkpoint_proof"],
    })
    return payload


def _safe_stage_json(run_root: Path, path: Path) -> dict[str, object]:
    from .experiment import _json_canonical, _read_test_bytes_once
    return _json_canonical(path, _read_test_bytes_once(path, trusted_root=run_root))


def _run_cuda_attempt(context: "StageContext", loader) -> Mapping[str, object]:
    """Wrap model loading and every CUDA operation in one retry boundary."""
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Task 16 requires CUDA")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        stock, tokenizer = loader()
        return _execute_smoke(context, stock, tokenizer, started=started)
    except FreshProcessRequired:
        raise
    except BaseException as error:
        if not _is_cuda_process_fault(error):
            raise
        _record_cuda_failure(context, error, started)
        raise FreshProcessRequired(str(error)) from error


def _record_cuda_failure(context: "StageContext", error: BaseException, started: float) -> None:
    try:
        torch.cuda.synchronize()
    except BaseException:
        pass
    try:
        allocated = int(torch.cuda.max_memory_allocated())
    except BaseException:
        allocated = 0
    try:
        reserved = int(torch.cuda.max_memory_reserved())
    except BaseException:
        reserved = 0
    try:
        total = int(torch.cuda.get_device_properties(0).total_memory)
    except BaseException:
        total = 0
    payload = {
        "cuda_process_fault": True,
        "chargeable_cuda_seconds": max(0.0, time.perf_counter() - started),
        "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
        "total_vram_bytes": total,
        "message": str(error), "error_type": type(error).__name__,
    }
    event = {
        "kind": "stage_attempt", "stage": "smoke", "attempt": context.attempt,
        "outcome": "fresh_process_required", "payload": payload,
    }
    payload["event_id"] = _smoke_failure_event_id(event)
    append_event(context.run_root / "events.jsonl", event)


def _execute_smoke(context: "StageContext", stock: nn.Module, tokenizer: object, *, started: float | None = None) -> Mapping[str, object]:
    if not hasattr(stock, "model") or not hasattr(stock, "lm_head"):
        raise RuntimeError("pinned model is not a causal Qwen3 model")
    device = torch.device("cuda")
    stock = stock.to(device=device, dtype=torch.bfloat16).eval()
    if hasattr(stock, "config"):
        stock.config.use_cache = False
        stock.config._attn_implementation = "eager"
    rows = _fixed_dev_rows(context)
    data_evidence = _dev_selection_evidence(context, rows)
    input_ids = tuple(_prompt_ids(tokenizer, str(row["prompt"]), device) for row in rows)
    if len(input_ids) != 8 or any(ids.shape[1] < 7 for ids in input_ids):
        raise RuntimeError("fixed smoke prompts are incomplete")

    started = time.perf_counter() if started is None else started
    torch.cuda.synchronize()
    try:
        wrapper = RecurrentQwen3(
            stock,
            layer_start=context.config.loop.layer_start,
            layer_stop=context.config.loop.layer_stop,
        ).eval()
        identity = _stock_wrapper_identity(stock, wrapper, input_ids)
        semantics = _semantic_checks(context, wrapper, input_ids[0])
        before_checkpoints = {str(path.relative_to(context.run_root)) for path in (context.run_root / "checkpoints").rglob("*") if path.is_file()} if (context.run_root / "checkpoints").is_dir() else set()
        gradients = _gradient_checks(context, wrapper, input_ids[0])
        after_checkpoints = {str(path.relative_to(context.run_root)) for path in (context.run_root / "checkpoints").rglob("*") if path.is_file()} if (context.run_root / "checkpoints").is_dir() else set()
        gradients["checkpoint_artifacts_unchanged"] = before_checkpoints == after_checkpoints
        checks = {**identity, **semantics, **gradients}
        required = (
            "identity_passed", "layer_call_counts_passed", "cache_disabled",
            "causality_passed", "shared_layer_objects", "wrapper_stock_model_shared",
            "adapter_off_f0_finite", "adapter_off_f1_finite",
            "adapter_off_a1_matches_f0", "adapter_off_a2_matches_f1",
            "adapter_gradients_finite", "adapter_gradients_nonzero", "suffix_input_gradient",
            "suffix_gradient_finite", "suffix_gradient_nonzero", "adapter_hash_unchanged",
            "checkpoint_artifacts_unchanged",
        )
        failed = [name for name in required if checks.get(name) is not True]
        if failed:
            raise RuntimeError(f"G1 smoke correctness gate failed: {', '.join(failed)}")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        total_vram = int(torch.cuda.get_device_properties(device).total_memory)
        payload: dict[str, object] = {
            "schema_version": _SMOKE_SCHEMA_VERSION,
            "model_class": "Qwen3ForCausalLM",
            "prompt_count": len(input_ids),
            "prompt_ids": [str(row["id"]) for row in rows],
            **data_evidence,
            **identity,
            **semantics,
            **gradients,
            "cuda_seconds": elapsed,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "total_vram_bytes": total_vram,
            "download_bytes": 0,
            "adapter_checkpoint_written": False,
            "disposable_adapter": True,
        }
        _validate_smoke_payload(payload)
        # Finalize only after every CUDA-sensitive payload and VRAM read.  A
        # later CUDA fault therefore cannot coexist with a passed interval.
        CudaCostLedger(
            context.run_root / "cuda-cost.jsonl",
            context.config.resources.maximum_cuda_seconds,
        ).record(f"smoke-attempt-{context.attempt}", "smoke", elapsed, "passed")
        append_event(context.run_root / "events.jsonl", {
            "kind": "stage_complete", "stage": "smoke", "attempt": context.attempt,
            "outcome": "passed", "payload": payload,
        })
        return payload
    except BaseException as error:
        try:
            torch.cuda.synchronize()
            elapsed = max(0.0, time.perf_counter() - started)
        except BaseException:
            elapsed = max(0.0, time.perf_counter() - started)
        if _is_cuda_process_fault(error):
            _record_cuda_failure(context, error, started)
            raise FreshProcessRequired(str(error)) from error
        _record_correctness_failure(context, error, started)
        raise


def _record_correctness_failure(context: "StageContext", error: BaseException, started: float) -> None:
    elapsed = max(0.0, time.perf_counter() - started)
    try:
        torch.cuda.synchronize()
    except BaseException:
        pass
    try:
        reserved = int(torch.cuda.max_memory_reserved())
    except BaseException:
        reserved = 0
    try:
        total = int(torch.cuda.get_device_properties(0).total_memory)
    except BaseException:
        total = 0
    ledger_path = context.run_root / "cuda-cost.jsonl"
    ledger = CudaCostLedger.load(ledger_path, context.config.resources.maximum_cuda_seconds) if ledger_path.exists() else CudaCostLedger(ledger_path, context.config.resources.maximum_cuda_seconds)
    attempt_id = f"smoke-attempt-{context.attempt}"
    if not any(entry.attempt_id == attempt_id for entry in ledger.entries):
        ledger.record(attempt_id, "smoke", elapsed, "failed_correctness")
    payload = {
        "cuda_process_fault": False, "chargeable_cuda_seconds": elapsed,
        "peak_allocated_bytes": 0, "peak_reserved_bytes": reserved,
        "total_vram_bytes": total, "message": str(error),
        "error_type": type(error).__name__,
    }
    event = {
        "kind": "stage_attempt", "stage": "smoke", "attempt": context.attempt,
        "outcome": "failure", "payload": payload,
    }
    payload["event_id"] = _smoke_failure_event_id(event)
    append_event(context.run_root / "events.jsonl", event)


def _fixed_dev_rows(context: "StageContext") -> list[Mapping[str, object]]:
    from .experiment import _read_test_bytes_once
    path = context.run_root / "data" / "dev.jsonl"
    manifest_path = context.run_root / "data" / "manifest.json"
    raw = _read_test_bytes_once(path, trusted_root=context.run_root)
    manifest_raw = _read_test_bytes_once(manifest_path, trusted_root=context.run_root)
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if not isinstance(manifest, Mapping):
        raise RuntimeError("data manifest is invalid")
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_hash != _DEV_MANIFEST_SHA256:
        raise RuntimeError("data manifest is not the immutable Task 15 manifest")
    split_hash = manifest.get("split_sha256")
    if not isinstance(split_hash, Mapping) or split_hash.get("dev") != hashlib.sha256(raw).hexdigest() or split_hash.get("dev") != _DEV_BYTES_SHA256:
        raise RuntimeError("dev bytes do not match the immutable data manifest")
    rows: list[Mapping[str, object]] = []
    for line in raw.decode("utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, Mapping) and value.get("family") in _FAMILIES:
            rows.append(value)
    selected = []
    for family in _FAMILIES:
        family_rows = [row for row in rows if row.get("family") == family]
        selected.extend(family_rows[: _PROMPT_COUNTS[family]])
    if [str(row.get("id")) for row in selected] != list(_DEV_EXPECTED_IDS) or any(row.get("split") != "dev" for row in selected):
        raise RuntimeError("fixed smoke rows do not match the immutable selection")
    selected_bytes = b"".join(canonical_json(row) for row in selected)
    if hashlib.sha256(selected_bytes).hexdigest() != _DEV_SELECTED_SHA256:
        raise RuntimeError("fixed smoke row content does not match the immutable selection")
    return selected


def _dev_selection_evidence(context: "StageContext", rows: list[Mapping[str, object]]) -> dict[str, str]:
    from .experiment import _read_test_bytes_once
    raw = _read_test_bytes_once(context.run_root / "data" / "dev.jsonl", trusted_root=context.run_root)
    manifest = _read_test_bytes_once(context.run_root / "data" / "manifest.json", trusted_root=context.run_root)
    return {
        "dev_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "dev_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "selected_rows_sha256": hashlib.sha256(b"".join(canonical_json(row) for row in rows)).hexdigest(),
    }


def _prompt_ids(tokenizer: object, prompt: str, device: torch.device) -> Tensor:
    template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(template):
        raise RuntimeError("pinned tokenizer has no chat template")
    encoded = template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt",
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if isinstance(encoded, list):
        encoded = torch.tensor([encoded], dtype=torch.long)
    if not isinstance(encoded, Tensor) or encoded.ndim != 2:
        raise RuntimeError("chat template did not return [1,T] token IDs")
    return encoded.to(device=device, dtype=torch.long)


def _stock_wrapper_identity(stock: nn.Module, wrapper: RecurrentQwen3, inputs: tuple[Tensor, ...]) -> dict[str, object]:
    per_prompt: list[dict[str, float]] = []
    for ids in inputs:
        with torch.no_grad():
            stock_logits = stock(input_ids=ids, use_cache=False, return_dict=True).logits
            wrapper_logits = wrapper(ids, mode="single")
        if not bool(torch.isfinite(stock_logits).all()) or not bool(torch.isfinite(wrapper_logits).all()):
            raise RuntimeError("non-finite stock or wrapper logits")
        delta = (stock_logits.float() - wrapper_logits.float()).abs()
        cosine = float(F.cosine_similarity(stock_logits.float().flatten(), wrapper_logits.float().flatten(), dim=0).item())
        per_prompt.append({"max_abs_diff": float(delta.max().item()), "cosine": cosine})
    return {
        "identity_prompts": per_prompt,
        "identity_max_abs_diff": max(row["max_abs_diff"] for row in per_prompt),
        "identity_min_cosine": min(row["cosine"] for row in per_prompt),
        "identity_passed": all(row["max_abs_diff"] <= 1e-3 and row["cosine"] >= 0.999999 for row in per_prompt),
        "wrapper_stock_model_shared": wrapper.stock_model is stock,
    }


def _semantic_checks(context: "StageContext", wrapper: RecurrentQwen3, ids: Tensor) -> dict[str, object]:
    def count_layers(mode: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        hooks = [
            layer.register_forward_hook(
                lambda _m, _a, _o, index=index: counts.__setitem__(
                    str(index), counts.get(str(index), 0) + 1
                )
            )
            for index, layer in enumerate(wrapper.stock_model.model.layers)
        ]
        try:
            with torch.no_grad():
                wrapper(ids, mode=mode)
        finally:
            for hook in hooks:
                hook.remove()
        return counts

    single_counts = count_layers(context.config.loop.frozen_mode)
    repeat_counts = count_layers(context.config.loop.repeat_mode)
    target_paths = (("mlp", "down_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("self_attn", "o_proj"), ("self_attn", "q_proj"), ("self_attn", "v_proj"))
    selected_base_projection_ids: list[dict[str, object]] = []
    selected_projection_invocations: dict[str, list[dict[str, object]]] = {"single": [], "repeat": []}
    for layer_index in range(context.config.loop.layer_start, context.config.loop.layer_stop):
        layer = wrapper.stock_model.model.layers[layer_index]
        for parent_name, projection_name in target_paths:
            projection = getattr(getattr(layer, parent_name), projection_name)
            selected_base_projection_ids.append({
                "layer": layer_index,
                "projection": f"{parent_name}.{projection_name}",
                "object_id": id(projection),
            })
    for mode_name, mode in (("single", context.config.loop.frozen_mode), ("repeat", context.config.loop.repeat_mode)):
        hooks = []
        for layer_index in range(context.config.loop.layer_start, context.config.loop.layer_stop):
            layer = wrapper.stock_model.model.layers[layer_index]
            for parent_name, projection_name in target_paths:
                projection = getattr(getattr(layer, parent_name), projection_name)
                projection_name_full = f"{parent_name}.{projection_name}"
                hooks.append(projection.register_forward_hook(
                    lambda module, _a, _o, mode_name=mode_name, layer_index=layer_index, projection_name_full=projection_name_full:
                    selected_projection_invocations[mode_name].append({
                        "layer": layer_index,
                        "projection": projection_name_full,
                        "object_id": id(module),
                    })
                ))
        try:
            with torch.no_grad():
                wrapper(ids, mode=mode)
        finally:
            for hook in hooks:
                hook.remove()
    hooks = []
    cache_calls: list[dict[str, object]] = []
    for layer in wrapper.stock_model.model.layers:
        hooks.append(layer.register_forward_pre_hook(
            lambda _m, _a, kwargs: cache_calls.append({"past_key_values": kwargs.get("past_key_values"), "use_cache": kwargs.get("use_cache")}),
            with_kwargs=True,
        ))
    try:
        with torch.no_grad():
            f0 = wrapper(ids, mode=context.config.loop.frozen_mode)
            f1 = wrapper(ids, mode=context.config.loop.repeat_mode)
            left = ids.clone(); right = ids.clone(); right[:, -1] = (right[:, -1] + 1) % wrapper.stock_model.config.vocab_size
            causal_diffs = [
                float((wrapper(left, mode)[:, :5].float() - wrapper(right, mode)[:, :5].float()).abs().max().item())
                for mode in ("single", "repeat", "damped")
            ]
            causal = all(diff == 0.0 for diff in causal_diffs)
    finally:
        for hook in hooks:
            hook.remove()
    expected_single = {
        str(index): 1 for index in range(len(wrapper.stock_model.model.layers))
    }
    expected_repeat = {
        str(index): (2 if context.config.loop.layer_start <= index < context.config.loop.layer_stop else 1)
        for index in range(len(wrapper.stock_model.model.layers))
    }
    layer_call_counts_passed = single_counts == expected_single and repeat_counts == expected_repeat
    return {
        "adapter_off_f0_finite": bool(torch.isfinite(f0).all()),
        "adapter_off_f1_finite": bool(torch.isfinite(f1).all()),
        "layer_call_counts": {
            "single": single_counts, "single_expected": expected_single,
            "repeat": repeat_counts, "repeat_expected": expected_repeat,
            "single_total": sum(single_counts.values()),
            "repeat_total": sum(repeat_counts.values()),
        },
        "layer_call_counts_passed": layer_call_counts_passed,
        "cache_disabled": bool(cache_calls) and all(call["past_key_values"] is None and call["use_cache"] is False for call in cache_calls),
            "causality_passed": causal,
            "causality_max_abs_diff": max(causal_diffs),
            "causality_max_abs_diff_by_mode": {mode: diff for mode, diff in zip(("single", "repeat", "damped"), causal_diffs)},
            "cache_observations": cache_calls,
        "shared_layer_objects": len({id(layer) for layer in wrapper.stock_model.model.layers}) == len(wrapper.stock_model.model.layers),
        "selected_layer_ids": [id(wrapper.stock_model.model.layers[index]) for index in range(context.config.loop.layer_start, context.config.loop.layer_stop)],
        "selected_base_projection_ids": selected_base_projection_ids,
        "selected_projection_invocation_ids": selected_projection_invocations,
        "selected_projection_objects_shared": bool(
            selected_projection_invocations["single"] and selected_projection_invocations["repeat"]
            and {item["object_id"] for item in selected_projection_invocations["single"]}
            == {item["object_id"] for item in selected_projection_invocations["repeat"]}
        ),
    }


def _gradient_checks(context: "StageContext", wrapper: RecurrentQwen3, ids: Tensor) -> dict[str, object]:
    adapters = install_adapters(wrapper.stock_model, context.config.loop.layer_start, context.config.loop.layer_stop, seed=47)
    initial_hash = adapter_sha256(wrapper.stock_model)
    selected_projection_ids: list[dict[str, object]] = []
    selected_parameter_ids: list[dict[str, object]] = []
    target_paths = (("mlp", "down_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("self_attn", "o_proj"), ("self_attn", "q_proj"), ("self_attn", "v_proj"))
    for layer_index in range(context.config.loop.layer_start, context.config.loop.layer_stop):
        layer = wrapper.stock_model.model.layers[layer_index]
        for parent_name, projection_name in target_paths:
            projection = getattr(getattr(layer, parent_name), projection_name)
            selected_projection_ids.append({"layer": layer_index, "projection": f"{parent_name}.{projection_name}", "wrapper_object_id": id(projection), "base_object_id": id(projection.base)})
            selected_parameter_ids.append({"layer": layer_index, "projection": f"{parent_name}.{projection_name}", "a_id": id(projection.a), "b_id": id(projection.b), "base_weight_id": id(projection.base.weight)})
    set_adapters_enabled(wrapper.stock_model, False)
    with torch.no_grad():
        f0 = wrapper(ids, "single")
        f1 = wrapper(ids, "repeat")
    set_adapters_enabled(wrapper.stock_model, True)
    wrapper.train()
    labels = torch.full_like(ids, -100)
    labels[:, -1] = ids[:, -1]
    suffix_inputs: list[Tensor] = []
    hook = wrapper.stock_model.model.layers[context.config.loop.layer_stop].register_forward_pre_hook(
        lambda _m, args: (args[0].retain_grad(), suffix_inputs.append(args[0]))[1]
    )
    try:
        loss = wrapper.answer_loss(ids, labels, context.config.loop.repeat_mode)
        loss.backward()
    finally:
        hook.remove()
    grads = [parameter.grad for parameter in adapters]
    finite = all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in grads)
    evidence = []
    for index in range(0, len(grads), 2):
        a_grad, b_grad = grads[index:index + 2]
        evidence.append({
            "adapter_index": index // 2,
            "a_finite": a_grad is not None and bool(torch.isfinite(a_grad).all()),
            "a_nonzero": a_grad is not None and bool(a_grad.detach().abs().sum() > 0),
            "b_finite": b_grad is not None and bool(torch.isfinite(b_grad).all()),
            "b_nonzero": b_grad is not None and bool(b_grad.detach().abs().sum() > 0),
            "a_zero_at_init_allowed": True,
            "a_abs_sum": float(a_grad.detach().abs().sum().item()) if a_grad is not None else float("nan"),
            "b_abs_sum": float(b_grad.detach().abs().sum().item()) if b_grad is not None else float("nan"),
        })
    nonzero = bool(evidence) and all(item["b_nonzero"] for item in evidence)
    suffix_grad = suffix_inputs[0].grad if suffix_inputs else None
    suffix = suffix_grad is not None
    set_adapters_enabled(wrapper.stock_model, False)
    with torch.no_grad():
        off_f0 = wrapper(ids, "single")
        off_f1 = wrapper(ids, "repeat")
    return {
        "adapter_initial_hash": initial_hash,
        "adapter_off_a1_matches_f0": bool(torch.equal(off_f0, f0)),
        "adapter_off_a2_matches_f1": bool(torch.equal(off_f1, f1)),
        "adapter_off_f0_nonfinite_count": int((~torch.isfinite(off_f0)).sum().item()),
        "adapter_off_f1_nonfinite_count": int((~torch.isfinite(off_f1)).sum().item()),
        "adapter_off_a1_max_abs_diff": float((off_f0.float() - f0.float()).abs().max().item()),
        "adapter_off_a2_max_abs_diff": float((off_f1.float() - f1.float()).abs().max().item()),
        "adapter_gradients_finite": finite,
        "adapter_gradients_nonzero": nonzero,
        "adapter_gradient_evidence": evidence,
        "lora_zero_init_nuance": "LoRA B is zero-initialized; A-gradient zero at first backward is permitted, while every B-gradient must be finite and nonzero.",
        "suffix_input_gradient": suffix,
        "suffix_gradient_finite": suffix_grad is not None and bool(torch.isfinite(suffix_grad).all()),
        "suffix_gradient_nonzero": suffix_grad is not None and bool(suffix_grad.detach().abs().sum() > 0),
        "suffix_gradient_abs_sum": float(suffix_grad.detach().abs().sum().item()) if suffix_grad is not None else float("nan"),
        "gradient_loss": float(loss.detach().item()),
        "adapter_final_hash": adapter_sha256(wrapper.stock_model),
        "adapter_hash_unchanged": adapter_sha256(wrapper.stock_model) == initial_hash,
        "selected_projection_ids": selected_projection_ids,
        "selected_parameter_ids": selected_parameter_ids,
    }
