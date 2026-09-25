from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
from pathlib import Path
import re

import pytest
import torch

from ppmoe.recurrence.config import RecurrenceConfig
from ppmoe.recurrence.experiment import (
    StageContext,
    StagePrerequisiteError,
    default_dependencies,
    _child_worker_ref,
    _validate_selection,
)
from ppmoe.recurrence.provenance import write_immutable_json
from ppmoe.recurrence.data import RetentionChunk


SIGNATURES = {name: character * 64 for name, character in zip(("source", "data", "config", "code"), "abcd")}


def _context(tmp_path: Path, stage: str) -> StageContext:
    config_path = Path(__file__).parents[1] / "configs" / "recurrence-v0.toml"
    return StageContext(
        stage=stage,
        config=RecurrenceConfig.load(config_path),
        config_path=config_path,
        artifact_root=tmp_path,
        run_root=tmp_path / "0123456789ab",
        signatures=SIGNATURES,
        attempt=1,
        read_path=lambda path: path.read_bytes(),
    )


def test_production_defaults_bind_every_runtime_worker_and_retry_dispatcher() -> None:
    """Catches shipping G2-G5 as the old unavailable placeholder."""
    dependencies = default_dependencies()

    assert set(dependencies.stage_functions) == {
        "source", "data", "smoke", "pilot", "frozen", "discover", "confirm", "report"
    }
    assert dependencies.child_worker_refs is None
    for stage in ("pilot", "frozen", "discover", "confirm"):
        worker = dependencies.stage_functions[stage]
        assert worker.__module__ == "ppmoe.recurrence.production"
        assert worker.__name__ == f"run_production_{stage}"
    assert callable(dependencies.checkpoint_schema)
    assert dependencies.dispose_pilot.__module__ == "ppmoe.recurrence.production"
    assert dependencies.restore_initialization.__module__ == "ppmoe.recurrence.production"
    assert dependencies.remaining_work_projection.__module__ == "ppmoe.recurrence.production"
    for stage in ("smoke", "pilot", "frozen", "discover", "confirm"):
        reference, identity = _child_worker_ref(stage, dependencies)
        assert reference == f"ppmoe.recurrence.{('smoke' if stage == 'smoke' else 'production')}:run_child_{stage}"
        assert re.fullmatch(r"[0-9a-f]{64}", identity)


def test_production_checkpoint_schema_binds_registered_shapes_and_seed_initialization(
    tmp_path: Path,
) -> None:
    """Catches retry validation trusting shapes or initialization from a checkpoint itself."""
    from ppmoe.recurrence.production import production_checkpoint_schema

    root = tmp_path / "run"
    initialization = root / "initializations" / "47.pt"
    initialization.parent.mkdir(parents=True)
    torch.save(
        {
            "schema_version": 1,
            "seed": 47,
            "initialization_hash": "e" * 64,
            "adapter_state": {},
        },
        initialization,
    )

    schema = production_checkpoint_schema(root, "a2", 47, SIGNATURES)

    assert schema["initialization_hash"] == "e" * 64
    assert len(schema["parameters"]) == 48
    assert schema["parameters"]["stock_model.model.layers.12.self_attn.q_proj.a"] == {
        "shape": [8, 2048], "dtype": "torch.float32"
    }
    assert schema["parameters"]["stock_model.model.layers.12.self_attn.v_proj.b"] == {
        "shape": [1024, 8], "dtype": "torch.float32"
    }
    assert schema["parameters"]["stock_model.model.layers.15.mlp.down_proj.a"] == {
        "shape": [8, 6144], "dtype": "torch.float32"
    }

    with pytest.raises(StagePrerequisiteError, match="seed|initialization"):
        production_checkpoint_schema(root, "a2", 48, SIGNATURES)


def test_remaining_projection_counts_only_unfinished_registered_work(tmp_path: Path) -> None:
    """Catches the unsafe all-zero projection and charging already completed arms again."""
    from ppmoe.recurrence.production import production_remaining_work_projection

    context = _context(tmp_path, "discover")
    context.run_root.mkdir(parents=True)
    _passed_pilot_fixture(context, n3=1250)
    # A1 seed 47 is complete; the other five production arms remain.
    _complete_training_fixture(context, arm="a1", seed=47)

    projection = production_remaining_work_projection(context)

    assert set(projection) == {
        "production_arms", "longest_prompt_generation", "retention_chunks",
        "retry_allowance", "report_cuda",
    }
    # Remaining: A2/A3@47 and A1/A2/A3@48.
    assert projection["production_arms"] == pytest.approx(2 * 1000 + 3 * 2 * 1000 + 2 * 2 * 1250)
    assert projection["longest_prompt_generation"] > 0
    assert projection["retention_chunks"] > 0
    assert projection["retry_allowance"] > 0
    assert projection["report_cuda"] == 0.0


def _selection_payload(test_bytes: bytes, dev_hashes: Mapping[str, str] | None = None, data_manifest_sha256: str = "f" * 64) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "passed",
        "selected_condition": "A2",
        "signatures": SIGNATURES,
        "test_sha256": hashlib.sha256(test_bytes).hexdigest(),
        "n3": 1250,
        "data_manifest_sha256": data_manifest_sha256,
        "dev_score_hashes": dev_hashes or {"A1": "1" * 64, "A2": "2" * 64, "A3": "3" * 64, "F0": "0" * 64},
    }


def _write_anchored_selection(
    context: StageContext, test_bytes: bytes, dev_hashes: Mapping[str, str]
) -> None:
    manifest = b'{"split_sha256":{"test":"' + hashlib.sha256(test_bytes).hexdigest().encode() + b'"}}\n'
    path = context.run_root / "data" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(manifest)
    digest = hashlib.sha256(manifest).hexdigest()
    selection = _selection_payload(test_bytes, dev_hashes, digest)
    selection_hash = write_immutable_json(context.run_root / "selection.json", selection)
    write_immutable_json(context.run_root / "stages" / "data.json", {
        "stage": "data", "status": "passed", "signatures": SIGNATURES,
        "payload": {"manifest_sha256": digest}, "attempts": 1,
    })
    write_immutable_json(context.run_root / "stages" / "discover.json", {
        "stage": "discover", "status": "passed", "signatures": SIGNATURES,
        "payload": {"selection_sha256": selection_hash, "data_manifest_sha256": digest},
        "attempts": 1,
    })


def test_orchestrator_accepts_only_the_rich_immutable_selection_schema() -> None:
    payload = _selection_payload(b"sealed\n")
    _validate_selection(payload, SIGNATURES)

    for field, bad in (("selected_condition", "A1"), ("n3", 999), ("dev_score_hashes", {})):
        changed = dict(payload)
        changed[field] = bad
        with pytest.raises(StagePrerequisiteError, match="selection"):
            _validate_selection(changed, SIGNATURES)


def test_confirm_refuses_fake_checkpoint_bytes_after_all_six_manifests_exist(
    tmp_path: Path,
) -> None:
    """A complete manifest matrix cannot authorize fake checkpoint payloads."""
    from ppmoe.recurrence.production import authorize_confirmation_test_read

    context = _context(tmp_path, "confirm")
    root = context.run_root
    root.mkdir(parents=True)
    _passed_pilot_fixture(context)
    (root / "data").mkdir(parents=True)
    test_bytes = b'{"sealed":true}\n'
    (root / "data" / "test.jsonl").write_bytes(test_bytes)
    dev_hashes: dict[str, str] = {}
    for arm, relative in {
        "F0": "frozen-f0.jsonl", "A1": "a1-47-dev.jsonl",
        "A2": "a2-47-dev.jsonl", "A3": "a3-47-dev.jsonl",
    }.items():
        path = root / "scores" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(arm.encode())
        dev_hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_anchored_selection(context, test_bytes, dev_hashes)
    opened: list[Path] = []
    guarded = replace(context, read_path=lambda path: opened.append(path) or path.read_bytes())

    for seed in (47, 48):
        for arm in ("a1", "a2", "a3"):
            if (seed, arm) == (48, "a3"):
                continue
            _complete_training_fixture(context, arm=arm, seed=seed)

    with pytest.raises(StagePrerequisiteError, match="six|checkpoint|A3.*48"):
        authorize_confirmation_test_read(guarded)
    assert opened == []

    _complete_training_fixture(context, arm="a3", seed=48)
    with pytest.raises(StagePrerequisiteError, match="Task-9 checkpoint artifact"):
        authorize_confirmation_test_read(guarded)
    assert opened == []


def test_pilot_cleanup_removes_only_disposable_tree_and_restores_both_initializations(
    tmp_path: Path,
) -> None:
    """Catches pilot adapters surviving as production state or mutated seed snapshots."""
    from ppmoe.recurrence.production import dispose_pilot_state, restore_production_initializations

    context = _context(tmp_path, "pilot")
    disposable = context.run_root / "pilot-disposable"
    disposable.mkdir(parents=True)
    (disposable / "adapter.pt").write_bytes(b"pilot")
    hashes: dict[str, str] = {}
    for seed in (47, 48):
        path = context.run_root / "initializations" / f"{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"seed-{seed}".encode())
        hashes[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "disposable_relative_path": "pilot-disposable",
        "initialization_sha256": hashes,
    }

    dispose_pilot_state(context, payload)
    restore_production_initializations(context, payload)

    assert not disposable.exists()
    assert (context.run_root / "initializations" / "47.pt").read_bytes() == b"seed-47"
    assert (context.run_root / "initializations" / "48.pt").read_bytes() == b"seed-48"


def test_training_manifest_cannot_link_another_arms_completion_event(tmp_path: Path) -> None:
    from ppmoe.recurrence.production import _validate_training_events
    from ppmoe.recurrence.experiment import StagePrerequisiteError

    root = tmp_path / "run"
    root.mkdir()
    event = {
        "kind": "training_completed", "arm": "a1", "seed": 48, "attempt": 2,
        "update": 1000, "cursor": 8000, "index": 17, "timestamp": "2026-09-25T00:00:00+00:00",
    }
    (root / "events.jsonl").write_text(__import__("json").dumps(event) + "\n", encoding="utf-8")
    manifest = {
        "arm": "a2", "seed": 48, "attempts": [2], "completed_updates": 1000,
        "cursor": 8000, "event_indices": [17], "failure_event_indices": [], "failures": [],
    }

    with pytest.raises(StagePrerequisiteError, match="event indices"):
        _validate_training_events(root, manifest)


def test_authenticated_confirmation_retry_accepts_only_ordered_evaluator_prefix(
    tmp_path: Path,
) -> None:
    from ppmoe.recurrence.production import _validate_partial_confirmation_scores
    from ppmoe.recurrence.experiment import StagePrerequisiteError

    context = _context(tmp_path, "confirm")
    score_path = context.run_root / "scores" / "a1-48-test.jsonl"
    score_path.parent.mkdir(parents=True)
    rows = [
        {"example_id": "e-1", "arm": "A1", "seed": 48, "split": "test", "correct": True,
         "signatures": dict(context.signatures)},
        {"example_id": "e-2", "arm": "A1", "seed": 48, "split": "test", "correct": False,
         "signatures": dict(context.signatures)},
    ]
    score_path.write_text(__import__("json").dumps(rows[0]) + "\n", encoding="utf-8")
    test_bytes = b'{"id":"e-1"}\n{"id":"e-2"}\n'

    _validate_partial_confirmation_scores(context, [score_path], test_bytes)
    score_path.write_text(__import__("json").dumps(rows[1]) + "\n", encoding="utf-8")
    with pytest.raises(StagePrerequisiteError, match="ID prefix"):
        _validate_partial_confirmation_scores(context, [score_path], test_bytes)


def test_parent_frozen_gate_rejects_raw_rows_from_a_foreign_producer(
    tmp_path: Path,
) -> None:
    from ppmoe.recurrence.gates import summarize_exact_scores
    from ppmoe.recurrence.production import _validate_persisted_stage_evidence
    from ppmoe.recurrence.runtime_engine import _frozen_diagnostic

    context = _context(tmp_path, "frozen")
    context.run_root.mkdir(parents=True)
    write_immutable_json(context.run_root / "stages" / "data.json", {
        "stage": "data", "status": "passed", "signatures": SIGNATURES,
        "payload": {"retention_chunk_count": 1},
    })
    summaries: dict[str, object] = {}
    exact_by_arm: dict[str, list[dict[str, object]]] = {}
    retention_by_arm: dict[str, list[dict[str, object]]] = {}
    for arm in ("F0", "F1", "F2"):
        exact_rows = [
            {
                "example_id": f"{family}-{index}", "family": family,
                    "correct": index % 2 == 0, "reason_code": "correct" if index % 2 == 0 else "wrong_value",
                    "arm": arm, "seed": 47, "split": "dev", "signatures": dict(SIGNATURES),
                    "latency_seconds": 0.01,
            }
            for family in ("expressions", "lists", "graphs")
            for index in range(200)
        ]
        exact_path = context.run_root / "scores" / f"frozen-{arm.lower()}.jsonl"
        exact_path.parent.mkdir(parents=True, exist_ok=True)
        exact_bytes = b"".join(__import__("json").dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in exact_rows)
        exact_path.write_bytes(exact_bytes)
        exact_digest = hashlib.sha256(exact_bytes).hexdigest()
        write_immutable_json(exact_path.with_name(f"{exact_path.stem}.summary.json"), {
            "complete": True, "arm": arm, "seed": 47, "split": "dev",
            "record_count": 600, "expected_count": 600, "failures": 0,
            "signatures": dict(SIGNATURES), "records_sha256": exact_digest,
        })
        retention_rows = [{
            "chunk_index": 0, "arm": arm, "seed": 47, "split": "dev",
            "signatures": dict(SIGNATURES), "nll_sum": 1.0, "scored_tokens": 1,
            "latency_seconds": 0.01,
        }]
        retention_path = context.run_root / "retention-scores" / f"frozen-{arm.lower()}.jsonl"
        retention_path.parent.mkdir(parents=True, exist_ok=True)
        retention_bytes = b"".join(__import__("json").dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in retention_rows)
        retention_path.write_bytes(retention_bytes)
        retention_digest = hashlib.sha256(retention_bytes).hexdigest()
        write_immutable_json(retention_path.with_name(f"{retention_path.stem}.summary.json"), {
                "complete": True, "arm": arm, "seed": 47, "split": "dev",
                "record_count": 1, "expected_count": 1, "failures": 0,
                "signatures": dict(SIGNATURES), "records_sha256": retention_digest,
                "nll_sum": 1.0, "scored_tokens": 1,
            })
        exact_by_arm[arm] = exact_rows
        retention_by_arm[arm] = retention_rows
        summaries[arm.lower()] = {
            "exact": summarize_exact_scores(exact_rows),
            "retention": {
                "complete": True, "record_count": 1, "records_sha256": retention_digest,
                "nll_sum": 1.0, "scored_tokens": 1,
            },
        }
    payload = {
        "summaries": summaries,
        "diagnostics": {
            arm.lower(): _frozen_diagnostic(
                exact_by_arm[arm], exact_by_arm["F0"], retention_by_arm[arm],
                retention_by_arm["F0"], {"peak_allocated_bytes": 1, "peak_reserved_bytes": 2, "total_vram_bytes": 32 << 30},
            )
            for arm in ("F1", "F2")
        },
    }
    _validate_persisted_stage_evidence("frozen", payload, context)

    tampered = dict(exact_rows[0])
    tampered["arm"] = "A2"
    exact_rows[0] = tampered
    exact_path = context.run_root / "scores" / "frozen-f2.jsonl"
    exact_bytes = b"".join(__import__("json").dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in exact_rows)
    exact_path.write_bytes(exact_bytes)
    sidecar = exact_path.with_name(f"{exact_path.stem}.summary.json")
    sidecar_value = __import__("json").loads(sidecar.read_text(encoding="utf-8"))
    sidecar_value["records_sha256"] = hashlib.sha256(exact_bytes).hexdigest()
    sidecar.unlink()
    write_immutable_json(sidecar, sidecar_value)
    with pytest.raises(ValueError, match="raw evidence"):
        _validate_persisted_stage_evidence("frozen", payload, context)


def test_frozen_gate_requires_visible_f1_f2_summary_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """G3 diagnostic sidecars must remain available to retry validation."""
    from ppmoe.recurrence import runtime_engine

    context = _context(tmp_path, "frozen")
    summary_paths: list[Path] = []

    def score(*_args: object, prefix: str, **_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        exact_summary = context.run_root / "scores" / f"{prefix}.summary.json"
        retention_summary = context.run_root / "retention-scores" / f"{prefix}.summary.json"
        for path in (exact_summary, retention_summary):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
            summary_paths.append(path)
        exact_rows = context.run_root / "scores" / f"{prefix}.jsonl"
        retention_rows = context.run_root / "retention-scores" / f"{prefix}.jsonl"
        exact_rows.write_text("{}\n", encoding="utf-8")
        retention_rows.write_text("{}\n", encoding="utf-8")
        return (
            {"records_path": str(exact_rows), "summary_path": str(exact_summary)},
            {"records_path": str(retention_rows), "summary_path": str(retention_summary)},
        )

    monkeypatch.setattr(runtime_engine, "_unadapted", lambda _context: (object(), object()))
    monkeypatch.setattr(runtime_engine, "_rows", lambda *_args: [])
    monkeypatch.setattr(runtime_engine, "load_retention_chunks", lambda _path: [])
    monkeypatch.setattr(runtime_engine, "_score", score)
    monkeypatch.setattr(runtime_engine, "_records", lambda _path: [])
    monkeypatch.setattr(runtime_engine, "_vram", lambda: {})
    monkeypatch.setattr(runtime_engine, "_hide_diagnostic_summary", lambda _result: pytest.fail("G3 hid a retry summary"))
    monkeypatch.setattr(runtime_engine, "classify_informative_families", lambda _summary: {"passes": True})
    monkeypatch.setattr(runtime_engine, "summarize_exact_scores", lambda *_args: {"families": {}, "macro_accuracy": 0.5, "global_invalid_rate": 0.0})
    monkeypatch.setattr(runtime_engine, "_frozen_diagnostic", lambda *_args: {})
    runtime_engine._frozen(context)
    assert len(summary_paths) == 6
    assert all(path.is_file() and path.suffix == ".json" for path in summary_paths)


def test_retention_raw_records_bind_their_evaluation_split() -> None:
    """Catches discovery retention being mistaken for sealed confirmation evidence."""
    from ppmoe.recurrence.evaluate import _score_chunk

    row = _score_chunk(
        object(), RetentionChunk(0, (1, 2), ("doc",), 1), "single", "F0", 47,
        SIGNATURES, "test",
    )
    assert row["split"] == "test"


def test_stage_payload_validation_rejects_unknown_or_incomplete_evidence() -> None:
    from ppmoe.recurrence.production import validate_stage_payload
    from ppmoe.recurrence.gates import classify_informative_families

    exact = {
        "record_count": 600, "macro_accuracy": 0.5, "global_invalid_rate": 0.0,
        "families": {
            family: {"accuracy": 0.5, "invalid_rate": 0.0}
            for family in ("expressions", "lists", "graphs")
        },
    }
    gates = classify_informative_families(exact)
    valid = {
        "schema_version": 1,
        "decision": {"status": "passed", "reason": "informative", "passed": True, "gates": gates},
        "summaries": {
            arm: {"exact": exact, "retention": {"complete": True, "record_count": 1, "records_sha256": "a" * 64}}
            for arm in ("f0", "f1", "f2")
        },
        "diagnostics": {"f1": {"macro_accuracy_delta": 0.0}, "f2": {"macro_accuracy_delta": 0.0}},
        "peak_allocated_bytes": 1,
        "peak_reserved_bytes": 2,
        "total_vram_bytes": 32 << 30,
    }
    validate_stage_payload("frozen", valid)
    with pytest.raises(ValueError, match="schema|payload"):
        validate_stage_payload("frozen", {**valid, "invented": True})
    with pytest.raises(ValueError, match="decision"):
        validate_stage_payload("frozen", {**valid, "decision": {"status": "passed"}})


@pytest.mark.parametrize("stage", ["frozen", "discover", "confirm"])
def test_normal_and_child_stage_results_share_scientific_terminal_mapping(stage: str) -> None:
    """Catches a retry payload bypassing the registered scientific decision."""
    from ppmoe.recurrence.experiment import _stage_result_status

    payload = {"decision": {"status": "stopped_scientific", "reason": "no-go", "passed": False}}
    assert _stage_result_status(stage, payload) == ("stopped_scientific", "no-go")
    assert _stage_result_status(stage, {"decision": {"status": "passed", "reason": "passed", "passed": True}}) == ("passed", "passed")
    assert _stage_result_status(stage, {"decision": {"status": "retention_uncertain", "reason": "retention_uncertain", "passed": False}}) == ("stopped_scientific", "retention_uncertain")


def _passed_pilot_fixture(context: StageContext, *, n3: int = 1250) -> dict[str, object]:
    hashes: dict[str, str] = {}
    for seed in (47, 48):
        path = context.run_root / "initializations" / f"{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"init-{seed}".encode())
        hashes[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1, "n3": n3,
        "initialization_sha256": hashes,
        "initialization_hashes": {"47": "7" * 64, "48": "8" * 64},
        "seconds_per_training_update": {"a1": 2.0, "a2": 3.0},
        "seconds_per_generation_example": 4.0,
        "seconds_per_retention_chunk": 5.0,
        "forced_generation_seconds": 6.0,
        "longest_prompt_tokens": 100,
    }
    write_immutable_json(context.run_root / "pilot.json", payload)
    write_immutable_json(context.run_root / "stages" / "pilot.json", {
        "stage": "pilot", "status": "passed", "signatures": SIGNATURES,
        "payload": payload, "attempts": 1,
    })
    return payload


def _complete_training_fixture(
    context: StageContext,
    *,
    arm: str,
    seed: int,
    n3: int = 1250,
    seconds: float | None = None,
) -> None:
    checkpoint = context.run_root / "checkpoints" / arm / str(seed) / "latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"{arm}-{seed}".encode())
    pilot = __import__("json").loads((context.run_root / "pilot.json").read_text())
    updates = n3 if arm == "a3" else 1000
    write_immutable_json(
        context.run_root / "training" / f"{arm}-{seed}.json",
        {
            "schema_version": 1,
            "status": "complete",
            "arm": arm,
            "seed": seed,
            "signatures": SIGNATURES,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "completed_updates": updates,
            "requested_updates": updates,
            "total_example_presentations": updates * 8,
            "cursor": updates * 8,
            "unique_example_count": 8000,
            "ordered_id_sha256": "9" * 64,
            "initialization_hash": pilot["initialization_hashes"][str(seed)],
            "final_adapter_hash": "a" * 64,
            "trainable_parameter_count": 1_146_880,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 2,
            "synchronized_cuda_seconds": seconds if seconds is not None else 100.0,
            "checkpoint_hashes": [hashlib.sha256(checkpoint.read_bytes()).hexdigest()],
            "attempts": [1],
            "failures": [],
            "completion_status": "completed",
            "determinism_notes": [],
            "event_indices": [1],
            "failure_event_indices": [],
        },
    )


def test_downstream_pilot_load_binds_n3_and_initialization_files(tmp_path: Path) -> None:
    from ppmoe.recurrence.production import load_passed_pilot

    context = _context(tmp_path, "discover")
    payload = _passed_pilot_fixture(context)
    assert load_passed_pilot(context)["n3"] == payload["n3"]

    (context.run_root / "initializations" / "47.pt").write_bytes(b"replacement")
    with pytest.raises(StagePrerequisiteError, match="initialization"):
        load_passed_pilot(context)


def test_projection_uses_registered_matrix_without_duplicate_f0(tmp_path: Path) -> None:
    """Catches budgeting an extra F0 test pass or repeated discovery retention."""
    from ppmoe.recurrence.production import production_remaining_work_projection

    context = _context(tmp_path, "confirm")
    context.run_root.mkdir(parents=True)
    _passed_pilot_fixture(context)
    projection = production_remaining_work_projection(context)

    assert projection["longest_prompt_generation"] == 4.0 * (1800 + 1800 + 16800)
    assert projection["retention_chunks"] == 5.0 * (120 + 120 + 120)


def test_prepilot_projection_is_derived_from_passed_smoke_measurement(tmp_path: Path) -> None:
    """Catches arbitrary one-second placeholders before G2 evidence exists."""
    from ppmoe.recurrence.production import production_remaining_work_projection

    context = _context(tmp_path, "pilot")
    context.run_root.mkdir(parents=True)
    write_immutable_json(context.run_root / "stages" / "smoke.json", {
        "stage": "smoke", "status": "passed", "signatures": SIGNATURES,
        "payload": {"cuda_seconds": 2.5}, "attempts": 1,
    })
    projection = production_remaining_work_projection(context)
    assert projection["production_arms"] == 125.0
    assert projection["longest_prompt_generation"] == 92.5
    assert projection["retention_chunks"] == 10.0


def test_confirm_authorization_rejects_fake_checkpoint_artifacts_before_sealed_test(
    tmp_path: Path,
) -> None:
    """Catches hash-only authorization of fake checkpoint bytes."""
    from ppmoe.recurrence.production import authorize_confirmation_test_read

    context = _context(tmp_path, "confirm")
    context.run_root.mkdir(parents=True)
    _passed_pilot_fixture(context)
    test_bytes = b'{"sealed":true}\n'
    (context.run_root / "data").mkdir()
    (context.run_root / "data" / "test.jsonl").write_bytes(test_bytes)
    dev_hashes: dict[str, str] = {}
    for arm, name in {"F0": "frozen-f0.jsonl", "A1": "a1-47-dev.jsonl", "A2": "a2-47-dev.jsonl", "A3": "a3-47-dev.jsonl"}.items():
        path = context.run_root / "scores" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(arm.encode())
        dev_hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_anchored_selection(context, test_bytes, dev_hashes)
    for seed in (47, 48):
        _complete_training_fixture(context, arm="a1", seed=seed, seconds=100.0)
        _complete_training_fixture(context, arm="a2", seed=seed, seconds=100.0)
        _complete_training_fixture(context, arm="a3", seed=seed, seconds=100.0)

    opened: list[Path] = []
    guarded = replace(context, read_path=lambda path: opened.append(path) or path.read_bytes())
    with pytest.raises(StagePrerequisiteError, match="Task-9 checkpoint artifact"):
        authorize_confirmation_test_read(guarded)
    assert opened == []


def test_frozen_diagnostics_include_registered_deltas_transitions_and_resources() -> None:
    """Catches G3 persisting independent summaries without F1/F2-vs-F0 evidence."""
    from ppmoe.recurrence.runtime_engine import _frozen_diagnostic

    def row(identifier: str, family: str, correct: bool, reason: str, latency: float) -> dict[str, object]:
        return {"example_id": identifier, "family": family, "correct": correct, "reason_code": reason, "latency_seconds": latency}

    baseline = [
        row("e1", "expressions", False, "wrong_value", 1.0),
        row("l1", "lists", True, "correct", 1.0),
        row("g1", "graphs", False, "invalid_format", 1.0),
    ]
    candidate = [
        row("e1", "expressions", True, "correct", 2.0),
        row("l1", "lists", False, "wrong_value", 2.0),
        row("g1", "graphs", False, "wrong_value", 2.0),
    ]
    retention_base = [{"chunk_index": 0, "nll_sum": 10.0, "scored_tokens": 10, "latency_seconds": 3.0}]
    retention_candidate = [{"chunk_index": 0, "nll_sum": 9.0, "scored_tokens": 10, "latency_seconds": 4.0}]

    result = _frozen_diagnostic(
        candidate, baseline, retention_candidate, retention_base,
        {"peak_allocated_bytes": 10, "peak_reserved_bytes": 20, "total_vram_bytes": 30},
    )
    assert result["transitions"] == {"wrong_to_right": 1, "right_to_wrong": 1, "unchanged": 1}
    assert result["macro_accuracy_delta"] == pytest.approx(0.0)
    assert result["global_invalid_rate_delta"] == pytest.approx(-1 / 3)
    assert set(result["family_accuracy_deltas"]) == {"expressions", "lists", "graphs"}
    assert result["latency_seconds"] == {"generation": 6.0, "retention": 4.0}
    assert result["memory"]["peak_reserved_bytes"] == 20
    assert result["retention_ratio"]["replicates"] == 10000
