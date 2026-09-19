from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from ppmoe.recurrence.experiment import _child_worker_ref, default_dependencies


def test_default_production_dependencies_bind_real_smoke_worker() -> None:
    dependencies = default_dependencies()

    assert dependencies.stage_functions["smoke"].__name__ != "unavailable"
    reference, identity = _child_worker_ref("smoke", dependencies)
    assert reference == "ppmoe.recurrence.smoke:run_child_smoke"
    assert identity
    assert len(identity) == 64


def test_disposable_smoke_retry_has_no_training_checkpoint(tmp_path: Path) -> None:
    from ppmoe.recurrence.smoke import _smoke_failure_event_id, _smoke_retry_artifact

    root = tmp_path / "run"
    root.mkdir()
    event = {"index": 4, "kind": "stage_attempt", "stage": "smoke", "attempt": 1, "outcome": "fresh_process_required", "payload": {"cuda_process_fault": True, "message": "fault", "chargeable_cuda_seconds": 1.25}}
    event["payload"]["event_id"] = _smoke_failure_event_id(event)
    (root / "events.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )
    artifact = _smoke_retry_artifact(root)
    assert artifact["path"] == ""
    assert artifact["event_id"] == event["payload"]["event_id"]
    assert artifact["checkpoint_sha256"] == ""


def test_smoke_retry_rejects_a_noncanonical_failure_event_id(tmp_path: Path) -> None:
    from ppmoe.recurrence.smoke import _smoke_failure_event_id, _smoke_retry_artifact

    root = tmp_path / "run"
    root.mkdir()
    event = {"index": 0, "kind": "stage_attempt", "stage": "smoke", "attempt": 1, "outcome": "fresh_process_required", "payload": {"cuda_process_fault": True, "message": "fault", "chargeable_cuda_seconds": 1.0, "event_id": "f" * 64}}
    assert event["payload"]["event_id"] != _smoke_failure_event_id(event)
    (root / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity"):
        _smoke_retry_artifact(root)


def test_fixed_dev_rows_bind_manifest_bytes_and_exact_selection(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from ppmoe.recurrence.smoke import _fixed_dev_rows

    source = Path("artifacts/recurrence-v0/27b3645f7761/data")
    root = tmp_path / "run"
    (root / "data").mkdir(parents=True)
    for name in ("dev.jsonl", "manifest.json"):
        (root / "data" / name).write_bytes((source / name).read_bytes())
    context = SimpleNamespace(run_root=root, read_path=lambda path: path.read_bytes())
    rows = _fixed_dev_rows(context)
    assert [row["id"] for row in rows] == [
        "80a57b0a7fbbfb52358af4be489158547b41e0dc72c470c3bfd477b10d1e370b",
        "1a58e59f7a7ff169ff98eeeca9eaf66e1172c11694c2be5339a923a7cd9a72a2",
        "7a4fceaf912fdc66ff9cd9eb58f381c0f8bddf7028fdc23047e2c337108b119c",
        "20af5310d856a01d57bd621d3e922b81c6c8ab7b87e8705ef630743074fc5053",
        "7cef763be607ea74252b9dfb72d32eb3ec181d22c76d90ee1ff81ed88c4d89e1",
        "5d4bf4b1e024b00531c571bb6c5c00d7ef674259bfd5881645980b9d4c6ca526",
        "1327bca00f4e22d3ce05f7cd2b0e7ba1ed7dd9ffe492d53b90969fbf9149e4c4",
        "79c84ed449c458a86ac809add5e23a5aab554c516c44a1335bd055cbdc4ff099",
    ]


def test_fixed_dev_rows_reject_mutated_consumed_bytes(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from ppmoe.recurrence.smoke import _fixed_dev_rows

    source = Path("artifacts/recurrence-v0/27b3645f7761/data")
    root = tmp_path / "run"
    (root / "data").mkdir(parents=True)
    (root / "data" / "manifest.json").write_bytes((source / "manifest.json").read_bytes())
    raw = (source / "dev.jsonl").read_bytes()
    (root / "data" / "dev.jsonl").write_bytes(raw.replace(b'"answer":"0"', b'"answer":"1"', 1))
    context = SimpleNamespace(run_root=root, read_path=lambda path: path.read_bytes())
    with pytest.raises(RuntimeError, match="manifest|bytes|content"):
        _fixed_dev_rows(context)


def test_smoke_payload_schema_rejects_missing_vram_and_identity_fields() -> None:
    from ppmoe.recurrence.smoke import _validate_smoke_payload

    with pytest.raises(RuntimeError, match="payload"):
        _validate_smoke_payload({"schema_version": 1})


def test_smoke_payload_rejects_vacuous_true_gates() -> None:
    import ppmoe.recurrence.smoke as smoke
    from ppmoe.recurrence.smoke import _validate_smoke_payload

    payload = {
        key: [] for key in smoke._SMOKE_REQUIRED_PAYLOAD_KEYS
        if key in {"prompt_ids", "identity_prompts", "selected_layer_ids", "selected_projection_ids", "selected_parameter_ids", "selected_projection_invocation_ids", "adapter_gradient_evidence"}
    }
    payload.update({
        "schema_version": 1, "model_class": "Qwen3ForCausalLM", "prompt_count": 8,
        "prompt_ids": list(smoke._DEV_EXPECTED_IDS), "dev_bytes_sha256": smoke._DEV_BYTES_SHA256,
        "dev_manifest_sha256": smoke._DEV_MANIFEST_SHA256, "selected_rows_sha256": smoke._DEV_SELECTED_SHA256,
        "identity_max_abs_diff": 0.0, "identity_min_cosine": 1.0,
        "identity_passed": True, "wrapper_stock_model_shared": True,
        "layer_call_counts": {}, "layer_call_counts_passed": True,
        "selected_projection_objects_shared": True, "cache_disabled": True,
        "causality_passed": True, "causality_max_abs_diff": 0.0,
        "shared_layer_objects": True, "adapter_off_f0_finite": True,
        "adapter_off_f1_finite": True, "adapter_off_a1_matches_f0": True,
        "adapter_off_a2_matches_f1": True, "adapter_gradients_finite": True,
        "adapter_gradients_nonzero": True, "lora_zero_init_nuance": "LoRA B is zero-initialized; A-gradient zero at first backward is permitted, while every B-gradient must be finite and nonzero.",
        "suffix_input_gradient": True, "suffix_gradient_finite": True,
        "suffix_gradient_nonzero": True, "gradient_loss": 1.0,
        "adapter_initial_hash": "a" * 64, "adapter_final_hash": "a" * 64,
        "adapter_hash_unchanged": True, "adapter_checkpoint_written": False,
        "disposable_adapter": True, "checkpoint_artifacts_unchanged": True,
        "cuda_seconds": 1.0, "peak_allocated_bytes": 1,
        "peak_reserved_bytes": 2, "total_vram_bytes": 32 << 30,
        "download_bytes": 0,
        "selected_projection_invocation_ids": {},
    })
    with pytest.raises(RuntimeError, match="identity|layer|projection|gradient|causal|cache"):
        _validate_smoke_payload(payload)


def test_smoke_payload_validator_accepts_complete_structural_evidence() -> None:
    import ppmoe.recurrence.smoke as smoke
    from ppmoe.recurrence.smoke import _validate_smoke_payload

    projections = []
    parameters = []
    base_ids = []
    invocation_records = []
    for layer in range(12, 16):
        for offset, name in enumerate(smoke._TARGET_PROJECTIONS):
            number = len(projections) + 1
            object_id = 1000 + number
            base_ids.append(object_id)
            invocation_records.append({"layer": layer, "projection": name, "object_id": object_id})
            projections.append({"layer": layer, "projection": name, "wrapper_object_id": 2000 + number, "base_object_id": object_id})
            parameters.append({"layer": layer, "projection": name, "a_id": 3000 + number * 2, "b_id": 3001 + number * 2, "base_weight_id": 5000 + number})
    single = invocation_records
    repeat = [dict(item) for item in invocation_records for _ in range(2)]
    payload = {
        "schema_version": 1, "model_class": "Qwen3ForCausalLM", "prompt_count": 8, "prompt_ids": list(smoke._DEV_EXPECTED_IDS),
        "dev_bytes_sha256": smoke._DEV_BYTES_SHA256, "dev_manifest_sha256": smoke._DEV_MANIFEST_SHA256, "selected_rows_sha256": smoke._DEV_SELECTED_SHA256,
        "identity_prompts": [{"max_abs_diff": 0.0, "cosine": 1.0} for _ in range(8)], "identity_max_abs_diff": 0.0, "identity_min_cosine": 1.0, "identity_passed": True, "wrapper_stock_model_shared": True,
        "selected_layer_ids": [12, 13, 14, 15], "selected_base_projection_ids": invocation_records, "selected_projection_ids": projections, "selected_parameter_ids": parameters,
        "selected_projection_invocation_ids": {"single": single, "repeat": repeat}, "selected_projection_objects_shared": True,
        "layer_call_counts": {"single": {str(i): 1 for i in range(28)}, "single_expected": {str(i): 1 for i in range(28)}, "repeat": {str(i): (2 if 12 <= i < 16 else 1) for i in range(28)}, "repeat_expected": {str(i): (2 if 12 <= i < 16 else 1) for i in range(28)}, "single_total": 28, "repeat_total": 32},
        "cache_observations": [{"past_key_values": None, "use_cache": False} for _ in range(224)], "layer_call_counts_passed": True, "cache_disabled": True,
        "causality_passed": True, "causality_max_abs_diff": 0.0, "causality_max_abs_diff_by_mode": {"single": 0.0, "repeat": 0.0, "damped": 0.0}, "shared_layer_objects": True,
        "adapter_off_f0_finite": True, "adapter_off_f1_finite": True, "adapter_off_f0_nonfinite_count": 0, "adapter_off_f1_nonfinite_count": 0, "adapter_off_a1_matches_f0": True, "adapter_off_a2_matches_f1": True, "adapter_off_a1_max_abs_diff": 0.0, "adapter_off_a2_max_abs_diff": 0.0,
        "adapter_gradients_finite": True, "adapter_gradients_nonzero": True, "adapter_gradient_evidence": [{"adapter_index": i, "a_finite": True, "a_nonzero": False, "b_finite": True, "b_nonzero": True, "a_zero_at_init_allowed": True, "a_abs_sum": 0.0, "b_abs_sum": 1.0} for i in range(24)],
        "lora_zero_init_nuance": "LoRA B is zero-initialized; A-gradient zero at first backward is permitted, while every B-gradient must be finite and nonzero.", "suffix_input_gradient": True, "suffix_gradient_finite": True, "suffix_gradient_nonzero": True, "suffix_gradient_abs_sum": 1.0, "gradient_loss": 1.0,
        "adapter_initial_hash": "a" * 64, "adapter_final_hash": "a" * 64, "adapter_hash_unchanged": True, "adapter_checkpoint_written": False, "disposable_adapter": True, "checkpoint_artifacts_unchanged": True,
        "cuda_seconds": 1.0, "peak_allocated_bytes": 1, "peak_reserved_bytes": 2, "total_vram_bytes": 32 << 30, "download_bytes": 0,
    }
    _validate_smoke_payload(payload)


def test_smoke_payload_rejects_swapped_pre_lora_invocation_identity() -> None:
    import ppmoe.recurrence.smoke as smoke
    from ppmoe.recurrence.smoke import _validate_smoke_payload

    # Keep the complete fixture's construction in one place while changing one
    # invocation's base object identity: module-name/ID binding must be exact.
    projections = []
    base = []
    parameters = []
    for layer in range(12, 16):
        for number, name in enumerate(smoke._TARGET_PROJECTIONS, start=len(base) + 1):
            base_id = 1000 + number
            base.append({"layer": layer, "projection": name, "object_id": base_id})
            projections.append({"layer": layer, "projection": name, "wrapper_object_id": 2000 + number, "base_object_id": base_id})
            parameters.append({"layer": layer, "projection": name, "a_id": 3000 + number * 2, "b_id": 3001 + number * 2, "base_weight_id": 5000 + number})
    payload = _complete_smoke_payload_for_test(smoke, base, projections, parameters)
    payload["selected_projection_invocation_ids"]["single"][0]["object_id"] = base[1]["object_id"]
    with pytest.raises(RuntimeError, match="invocation"):
        _validate_smoke_payload(payload)


def test_smoke_payload_rejects_fabricated_post_lora_base_identity() -> None:
    import ppmoe.recurrence.smoke as smoke
    from ppmoe.recurrence.smoke import _validate_smoke_payload

    base = []
    projections = []
    parameters = []
    for layer in range(12, 16):
        for number, name in enumerate(smoke._TARGET_PROJECTIONS, start=len(base) + 1):
            base_id = 1000 + number
            base.append({"layer": layer, "projection": name, "object_id": base_id})
            projections.append({"layer": layer, "projection": name, "wrapper_object_id": 2000 + number, "base_object_id": base_id})
            parameters.append({"layer": layer, "projection": name, "a_id": 3000 + number * 2, "b_id": 3001 + number * 2, "base_weight_id": 5000 + number})
    payload = _complete_smoke_payload_for_test(smoke, base, projections, parameters)
    payload["selected_projection_ids"][0]["base_object_id"] = 999999
    with pytest.raises(RuntimeError, match="projection identity"):
        _validate_smoke_payload(payload)


def _complete_smoke_payload_for_test(smoke, base, projections, parameters):
    identity = [{"layer": row["layer"], "projection": row["projection"], "object_id": row["object_id"]} for row in base]
    payload = {
        "schema_version": 1, "model_class": "Qwen3ForCausalLM", "prompt_count": 8, "prompt_ids": list(smoke._DEV_EXPECTED_IDS),
        "dev_bytes_sha256": smoke._DEV_BYTES_SHA256, "dev_manifest_sha256": smoke._DEV_MANIFEST_SHA256, "selected_rows_sha256": smoke._DEV_SELECTED_SHA256,
        "identity_prompts": [{"max_abs_diff": 0.0, "cosine": 1.0} for _ in range(8)], "identity_max_abs_diff": 0.0, "identity_min_cosine": 1.0, "identity_passed": True, "wrapper_stock_model_shared": True,
        "selected_layer_ids": [12, 13, 14, 15], "selected_base_projection_ids": base, "selected_projection_ids": projections, "selected_parameter_ids": parameters,
        "selected_projection_invocation_ids": {"single": identity, "repeat": [dict(row) for row in identity for _ in range(2)]}, "selected_projection_objects_shared": True,
        "layer_call_counts": {"single": {str(i): 1 for i in range(28)}, "single_expected": {str(i): 1 for i in range(28)}, "repeat": {str(i): (2 if 12 <= i < 16 else 1) for i in range(28)}, "repeat_expected": {str(i): (2 if 12 <= i < 16 else 1) for i in range(28)}, "single_total": 28, "repeat_total": 32},
        "cache_observations": [{"past_key_values": None, "use_cache": False} for _ in range(224)], "layer_call_counts_passed": True, "cache_disabled": True,
        "causality_passed": True, "causality_max_abs_diff": 0.0, "causality_max_abs_diff_by_mode": {"single": 0.0, "repeat": 0.0, "damped": 0.0}, "shared_layer_objects": True,
        "adapter_off_f0_finite": True, "adapter_off_f1_finite": True, "adapter_off_f0_nonfinite_count": 0, "adapter_off_f1_nonfinite_count": 0, "adapter_off_a1_matches_f0": True, "adapter_off_a2_matches_f1": True, "adapter_off_a1_max_abs_diff": 0.0, "adapter_off_a2_max_abs_diff": 0.0,
        "adapter_gradients_finite": True, "adapter_gradients_nonzero": True, "adapter_gradient_evidence": [{"adapter_index": i, "a_finite": True, "a_nonzero": False, "b_finite": True, "b_nonzero": True, "a_zero_at_init_allowed": True, "a_abs_sum": 0.0, "b_abs_sum": 1.0} for i in range(24)],
        "lora_zero_init_nuance": "LoRA B is zero-initialized; A-gradient zero at first backward is permitted, while every B-gradient must be finite and nonzero.", "suffix_input_gradient": True, "suffix_gradient_finite": True, "suffix_gradient_nonzero": True, "suffix_gradient_abs_sum": 1.0, "gradient_loss": 1.0,
        "adapter_initial_hash": "a" * 64, "adapter_final_hash": "a" * 64, "adapter_hash_unchanged": True, "adapter_checkpoint_written": False, "disposable_adapter": True, "checkpoint_artifacts_unchanged": True,
        "cuda_seconds": 1.0, "peak_allocated_bytes": 1, "peak_reserved_bytes": 2, "total_vram_bytes": 32 << 30, "download_bytes": 0,
    }
    return payload


def test_smoke_payload_digest_covers_child_vram_measurements() -> None:
    from ppmoe.recurrence.experiment import _child_payload_digest

    payload = {"schema_version": 1, "total_vram_bytes": 32 << 30, "peak_reserved_bytes": 123}
    digest = _child_payload_digest(payload)
    assert digest != _child_payload_digest({**payload, "peak_reserved_bytes": 124})
    assert digest == _child_payload_digest({**payload, "payload_sha256": digest})


def test_production_retry_rejects_tampered_child_vram_payload(tmp_path: Path) -> None:
    import dataclasses
    import hashlib
    import os
    import sys
    from ppmoe.recurrence.experiment import _run_child_retry, default_dependencies, EnvironmentStop
    from ppmoe.recurrence.provenance import canonical_json
    import ppmoe.recurrence.smoke as smoke

    root = tmp_path / "abcdefabcdef"
    root.mkdir()
    failure = {"kind": "stage_attempt", "stage": "smoke", "attempt": 1, "outcome": "fresh_process_required", "payload": {"cuda_process_fault": True, "message": "fault"}}
    failure["payload"]["event_id"] = smoke._smoke_failure_event_id(failure)
    (root / "events.jsonl").write_bytes(canonical_json({**failure, "index": 0, "timestamp": "x"}))
    ids = list(smoke._DEV_EXPECTED_IDS)
    child = (
        "import hashlib,json,os,sys;"
        "r=json.load(open(sys.argv[1]));"
        "p={k:r[k] for k in ('checkpoint_path','checkpoint_digest','checkpoint_sha256','checkpoint_attempt','checkpoint_update','checkpoint_event_id','checkpoint_event_index','checkpoint_arm','checkpoint_seed','checkpoint_relative_path','checkpoint_signature','checkpoint_proof')};"
        "p.update({'worker_pid':os.getpid(),'attempt':r['attempt'],'dispatcher_stage':r['stage'],'dispatcher_ref':r['worker_ref'],'dispatcher_identity':r['worker_identity'],'schema_version':1,'model_class':'Qwen3ForCausalLM','prompt_count':8,'prompt_ids':" + repr(ids).replace("'", '"') + ",'dev_bytes_sha256':'" + smoke._DEV_BYTES_SHA256 + "','dev_manifest_sha256':'" + smoke._DEV_MANIFEST_SHA256 + "','selected_rows_sha256':'" + smoke._DEV_SELECTED_SHA256 + "','identity_prompts':[],'identity_max_abs_diff':0.0,'identity_min_cosine':1.0,'identity_passed':True,'wrapper_stock_model_shared':True,'selected_layer_ids':[],'selected_projection_ids':[],'selected_parameter_ids':[],'selected_projection_invocation_ids':{},'selected_projection_objects_shared':True,'layer_call_counts':{},'layer_call_counts_passed':True,'cache_disabled':True,'causality_passed':True,'causality_max_abs_diff':0.0,'shared_layer_objects':True,'adapter_off_f0_finite':True,'adapter_off_f1_finite':True,'adapter_off_a1_matches_f0':True,'adapter_off_a2_matches_f1':True,'adapter_gradients_finite':True,'adapter_gradients_nonzero':True,'adapter_gradient_evidence':[],'lora_zero_init_nuance':'LoRA B is zero-initialized; A-gradient zero at first backward is permitted, while every B-gradient must be finite and nonzero.','suffix_input_gradient':True,'suffix_gradient_finite':True,'suffix_gradient_nonzero':True,'gradient_loss':1.0,'adapter_initial_hash':'a'*64,'adapter_final_hash':'a'*64,'adapter_hash_unchanged':True,'adapter_checkpoint_written':False,'disposable_adapter':True,'checkpoint_artifacts_unchanged':True,'cuda_seconds':1.0,'peak_allocated_bytes':1,'peak_reserved_bytes':2,'total_vram_bytes':32<<30,'download_bytes':0});"
        "q=json.dumps(p,sort_keys=True,separators=(',',':'))+'\\n';"
        "p['payload_sha256']=hashlib.sha256(q.encode()).hexdigest();p['peak_reserved_bytes']=3;"
        "x={'request_sha256':hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(),'nonce':r['nonce'],'stage':r['stage'],'run_root':r['run_root'],'attempt':2,'pid':os.getpid(),'status':'passed','payload':p};"
        "open(sys.argv[2],'wb').write((json.dumps(x,sort_keys=True,separators=(',',':'))+'\\n').encode())"
    )
    deps = dataclasses.replace(default_dependencies(), child_command=(getattr(sys, "_base_executable", sys.executable), "-c", child))
    with pytest.raises(EnvironmentStop, match="payload digest"):
        _run_child_retry("smoke", Path("configs/recurrence-v0.toml").resolve(), tmp_path, root, "abcdefabcdef", {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64}, deps)


def test_empty_smoke_retry_proof_is_stable_before_and_after_child() -> None:
    from ppmoe.recurrence.experiment import _checkpoint_proof_for_artifact

    artifact = {
        "path": "", "checkpoint_sha256": "", "manifest_digest": "",
        "attempt": 1, "update": 0, "event_id": "a" * 64,
        "event_index": 4, "arm": "smoke", "seed": 0,
        "relative_path": "", "adapter_schema": None,
    }
    assert _checkpoint_proof_for_artifact("worker", artifact) == _checkpoint_proof_for_artifact("worker", dict(artifact))


def test_cuda_fault_during_model_load_is_durable_and_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ppmoe.recurrence.smoke as smoke
    from ppmoe.recurrence.train import FreshProcessRequired

    class Cuda:
        def is_available(self): return True
        def reset_peak_memory_stats(self, _device): pass
        def synchronize(self): pass
        def max_memory_allocated(self): return 11
        def max_memory_reserved(self): return 22

    monkeypatch.setattr(smoke.torch, "cuda", Cuda())
    context = type("Context", (), {"run_root": tmp_path, "attempt": 1})()
    with pytest.raises(FreshProcessRequired):
        smoke._run_cuda_attempt(context, lambda: (_ for _ in ()).throw(RuntimeError("CUDA setup fault")))
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert event["payload"]["event_id"] == smoke._smoke_failure_event_id(event)
    assert event["payload"]["peak_reserved_bytes"] == 22
    assert smoke._smoke_retry_artifact(tmp_path)["event_id"] == event["payload"]["event_id"]


def test_production_disposable_retry_reconciles_empty_proof_end_to_end(
    tmp_path: Path,
) -> None:
    import dataclasses
    import hashlib
    import ppmoe.recurrence.smoke as smoke
    from ppmoe.recurrence.experiment import _run_child_retry, default_dependencies
    from ppmoe.recurrence.provenance import canonical_json

    root = tmp_path / ("a" * 12)
    root.mkdir()
    failure = {"kind": "stage_attempt", "stage": "smoke", "attempt": 1, "outcome": "fresh_process_required", "payload": {"cuda_process_fault": True, "message": "fault", "chargeable_cuda_seconds": 1.0}}
    failure["payload"]["event_id"] = smoke._smoke_failure_event_id(failure)
    (root / "events.jsonl").write_bytes(canonical_json({**failure, "index": 0, "timestamp": "x"}))
    child = (
        "import hashlib,json,os,sys;"
        "r=json.load(open(sys.argv[1]));"
        "p={k:r[k] for k in ('checkpoint_path','checkpoint_digest','checkpoint_sha256','checkpoint_attempt','checkpoint_update','checkpoint_event_id','checkpoint_event_index','checkpoint_arm','checkpoint_seed','checkpoint_relative_path','checkpoint_signature','checkpoint_proof')};"
        "p.update({'worker_pid':os.getpid(),'attempt':r['attempt'],'dispatcher_stage':r['stage'],'dispatcher_ref':r['worker_ref'],'dispatcher_identity':r['worker_identity']});"
        "x={'request_sha256':hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(),'nonce':r['nonce'],'stage':r['stage'],'run_root':r['run_root'],'attempt':2,'pid':os.getpid(),'status':'passed','payload':p};"
        "open(sys.argv[2],'wb').write((json.dumps(x,sort_keys=True,separators=(',',':'))+'\\n').encode())"
    )
    deps = dataclasses.replace(
        default_dependencies(),
        child_command=(getattr(sys, "_base_executable", sys.executable), "-c", child),
    )
    result = _run_child_retry(
        "smoke", Path("configs/recurrence-v0.toml").resolve(), tmp_path,
        root, "a" * 12,
        {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64},
        deps,
    )
    assert result.payload["checkpoint_path"] == ""
    assert result.payload["checkpoint_proof"]
