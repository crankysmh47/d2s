"""Task 14 evidence-derived report contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import numpy as np

from ppmoe.recurrence.data import RetentionChunk
from ppmoe.recurrence.evaluate import evaluate_retention
from ppmoe.recurrence.event_identity import task9_failure_event_id
from ppmoe.recurrence.gates import paired_accuracy_bootstrap, paired_retention_bootstrap
from ppmoe.recurrence.report import _decision, _derive_bootstrap_comparisons, _derive_observed_comparison_points, _has_registered_bootstrap, _has_timing_gate, _load_events, _read_bootstrap_matrix, _read_bootstrap_vector, _read_inputs, _validate_summary_schema, build_report


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _stage(root: Path, name: str, status: str = "passed", payload: dict | None = None) -> None:
    signatures = {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64}
    if name == "source":
        signatures["data"] = ""
    _write(root / "stages" / f"{name}.json", {
        "stage": name,
        "status": status,
        "signatures": signatures,
        "payload": payload or {},
        "attempts": 1,
    })


def _root(tmp_path: Path, *, decision: str = "stopped_scientific") -> Path:
    root = tmp_path / "run"
    _stage(root, "source", payload={"source_manifest": "source-manifest.json", "revision": "e" * 40})
    _stage(root, "data", payload={"data_hash": "b" * 64})
    _stage(root, "smoke")
    _stage(root, "pilot", payload={"a3_valid": True, "training_steps": {"a1": 1000}})
    _stage(root, "frozen", payload={"decision": {"status": decision, "reason": "uninformative"}})
    config = root / "config.json"
    config.write_text("{}", encoding="utf-8")
    _write(root / "source-manifest.json", {
        "schema_version": 1,
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": "e" * 40,
        "files": [{"relative_path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}],
        "config_sha256": "c" * 64,
        "code_sha256": "d" * 64,
        "protocol_revision": "1" * 40,
    })
    return root


def test_report_marks_missing_downstream_evaluation_not_evaluated(tmp_path: Path) -> None:
    report = build_report(_root(tmp_path))
    assert report["evaluations"]["exact_scores"]["status"] == "not_evaluated"
    assert report["evaluations"]["retention"]["status"] == "not_evaluated"
    assert report["decision"]["status"] == "stopped_scientific"
    assert "not_evaluated" in (tmp_path / "run" / "report.md").read_text(encoding="utf-8")


def test_bare_a3_valid_flag_is_not_timing_evidence(tmp_path: Path) -> None:
    _root(tmp_path)
    assert not _has_timing_gate({"pilot": {"payload": {"a3_valid": True}}})


def test_f0_test_summary_is_authorized_only_by_confirm_producer() -> None:
    signatures = {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64}
    summary = {
        "schema_version": 1, "kind": "exact_scores", "record_count": 1,
        "expected_count": 1, "complete": True, "records_path": "scores/f0-test.jsonl",
        "records_sha256": "e" * 64, "arm": "F0", "seed": 47, "failures": 0,
        "signatures": signatures, "families": ["expressions", "lists", "graphs"],
        "split": "test",
    }
    with pytest.raises(ValueError, match="confirm"):
        _validate_summary_schema(
            summary, "exact_scores",
            {"frozen": {"status": "passed", "signatures": signatures}},
            "scores/f0-test.summary.json",
        )
    _validate_summary_schema(
        summary, "exact_scores",
        {"confirm": {"status": "passed", "signatures": signatures}},
        "scores/f0-test.summary.json",
    )


def test_terminal_g3_report_accepts_frozen_f1_f2_dev_summaries(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _stage(root, "frozen", status="stopped_scientific", payload={
        "decision": {"status": "stopped_scientific", "reason": "uninformative_benchmark"},
    })
    signatures = {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64}
    for arm in ("F1", "F2"):
        raw = root / "scores" / f"{arm.lower()}-47.jsonl"
        raw.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"example_id": f"{family}-0", "family": family, "correct": True,
             "reason_code": "correct", "arm": arm, "seed": 47, "split": "dev",
             "signatures": signatures}
            for family in ("expressions", "lists", "graphs")
        ]
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        _write(raw.with_name(f"{raw.stem}.summary.json"), {
            "schema_version": 1, "kind": "exact_scores", "complete": True,
            "record_count": 3, "expected_count": 3, "records_path": raw.relative_to(root).as_posix(),
            "records_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "arm": arm, "seed": 47, "failures": 0,
            "families": ["expressions", "lists", "graphs"], "signatures": signatures,
            "split": "dev",
        })

    report = build_report(root)
    assert report["decision"]["status"] == "stopped_scientific"
    assert {item["arm"] for item in report["evaluations"]["exact_scores"]["aggregates"]} == {"F1", "F2"}


def test_failed_environment_g3_report_inventories_but_does_not_aggregate_scores(tmp_path: Path) -> None:
    root = _root(tmp_path)
    signatures = {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64}
    _stage(root, "frozen", status="failed_environment", payload={"reason": "persisted evidence check failed"})
    paths: list[tuple[Path, Path]] = []
    for arm in ("F0", "F1", "F2"):
        raw = root / "scores" / f"frozen-{arm.lower()}.jsonl"
        rows = [
            {"example_id": f"{family}-0", "family": family, "correct": True,
             "reason_code": "correct", "arm": arm, "seed": 47, "split": "dev",
             "signatures": signatures}
            for family in ("expressions", "lists", "graphs")
        ]
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        _write(raw.with_name(f"{raw.stem}.summary.json"), {
            "schema_version": 1, "kind": "exact_scores", "complete": True,
            "record_count": 3, "expected_count": 3, "records_path": raw.relative_to(root).as_posix(),
            "records_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "arm": arm, "seed": 47, "failures": 0,
            "families": ["expressions", "lists", "graphs"], "signatures": signatures,
            "split": "dev",
        })
        retention = root / "retention-scores" / f"frozen-{arm.lower()}.jsonl"
        retention_rows = [{"chunk_index": 0, "arm": arm, "seed": 47, "split": "dev",
                           "signatures": signatures, "nll_sum": 1.0, "scored_tokens": 1}]
        retention.parent.mkdir(parents=True, exist_ok=True)
        retention.write_text("".join(json.dumps(row) + "\n" for row in retention_rows), encoding="utf-8")
        _write(retention.with_name(f"{retention.stem}.summary.json"), {
            "schema_version": 1, "kind": "retention_scores", "complete": True,
            "record_count": 1, "expected_count": 1, "records_path": retention.relative_to(root).as_posix(),
            "records_sha256": hashlib.sha256(retention.read_bytes()).hexdigest(),
            "arm": arm, "seed": 47, "failures": 0, "families": ["retention"],
            "signatures": signatures, "split": "dev", "nll_sum": 1.0, "scored_tokens": 1,
        })
        paths.extend(((raw, raw.with_name(f"{raw.stem}.summary.json")),
                      (retention, retention.with_name(f"{retention.stem}.summary.json"))))

    report = build_report(root)
    assert report["decision"]["status"] == "failed_environment"
    assert report["evaluations"]["exact_scores"] == {"status": "not_evaluated", "aggregates": [], "examples_scored": 0, "failures": 0}
    assert report["evaluations"]["retention"]["status"] == "not_evaluated"
    inventory = json.loads((root / "report-sources.json").read_text(encoding="utf-8"))["files"]
    inventory_hashes = {item["path"]: item["sha256"] for item in inventory}
    assert all(inventory_hashes[summary.relative_to(root).as_posix()] == hashlib.sha256(summary.read_bytes()).hexdigest()
               for _raw, summary in paths)


def test_one_sample_timing_control_is_not_registered(tmp_path: Path) -> None:
    _root(tmp_path)
    def timing(seed: int) -> dict[str, object]:
        return {"seed": seed, "sample_count": 1, "metadata": {"units": "seconds", "synchronized": True}, "a2_training_times": [1.0], "a3_training_times": [1.0]}
    assert not _has_timing_gate({"discover": {"payload": timing(47)}, "confirm": {"payload": timing(48)}})


def test_actual_evaluator_retention_summary_is_reportable(tmp_path: Path) -> None:
    root = _root(tmp_path)

    class RetentionModel:
        def __call__(self, input_ids: torch.Tensor, mode: str) -> torch.Tensor:
            logits = torch.zeros((1, input_ids.shape[1], 3), dtype=torch.float64)
            logits[..., 1] = 1.0
            logits[..., 2] = 2.0
            return logits

    signatures = {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64}
    result = evaluate_retention(
        RetentionModel(),
        [RetentionChunk(0, (0, 1, 2), ("doc-0",), 2)],
        "single",
        root / "retention" / "f0-47.jsonl",
        arm="F0",
        seed=47,
        signatures=signatures,
    )
    assert result["complete"] is True
    report = build_report(root)
    aggregate = report["evaluations"]["retention"]["aggregates"][0]
    assert aggregate["mean_nll"] == pytest.approx(result["mean_nll"])
    assert aggregate["scored_tokens"] == result["scored_tokens"]


def test_bootstrap_matrix_rejects_unregistered_columns(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / "bootstrap.json"
    payload = {"families": {family: {"ids": [f"{family}-0"], "indices": [[0]] * 10000} for family in ("expressions", "lists", "graphs")}}
    _write(path, payload)
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    assert _read_bootstrap_matrix(root, inventory, "bootstrap.json", hashlib.sha256(path.read_bytes()).hexdigest(), {family: [f"{family}-0", f"{family}-1"] for family in ("expressions", "lists", "graphs")}) is None


def test_bootstrap_matrix_rejects_biased_shape_valid_registered_seed_matrix(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / "bootstrap.json"
    payload = {"families": {family: {"ids": [f"{family}-0", f"{family}-1"], "indices": [[0, 0]] * 10000} for family in ("expressions", "lists", "graphs")}}
    _write(path, payload)
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    assert _read_bootstrap_matrix(root, inventory, "bootstrap.json", hashlib.sha256(path.read_bytes()).hexdigest(), {family: [f"{family}-0", f"{family}-1"] for family in ("expressions", "lists", "graphs")}, registered_seed=9047) is None


def test_event_without_append_event_identity_is_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "events.jsonl").write_text(json.dumps({"stage": "pilot", "attempt": 1, "outcome": "error"}) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    with pytest.raises(ValueError, match="index|kind"):
        _load_events(root, inventory)


def test_stopped_gate_cannot_smuggle_a_passed_payload() -> None:
    assert _decision({"confirm": {"status": "stopped_scientific", "payload": {"decision": {"status": "passed"}}}}, []) ["status"] == "stopped_scientific"


def test_event_timestamp_is_parsed_not_suffix_checked(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "events.jsonl").write_text(json.dumps({"index": 0, "timestamp": "garbage+00:00", "kind": "training_started", "arm": "A1", "seed": 47, "attempt": 1, "update": 0}) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    with pytest.raises(ValueError, match="timestamp"):
        _load_events(root, inventory)


@pytest.mark.parametrize("stage", ["source", "data", "smoke"])
def test_registered_stage_complete_event_loads_and_builds_report(tmp_path: Path, stage: str) -> None:
    root = _root(tmp_path)
    event = {"index": 0, "timestamp": "2026-01-01T00:00:00+00:00", "kind": "stage_complete", "stage": stage, "attempt": 1, "outcome": "passed", "payload": {"schema_version": 1}}
    (root / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    assert _load_events(root, inventory)[0]["kind"] == "stage_complete"
    report = build_report(root)
    assert report["event_history"][0]["stage"] == stage


@pytest.mark.parametrize("outcome", ["fresh_process_required", "failure"])
def test_smoke_stage_attempt_diagnostics_load_and_build_report(tmp_path: Path, outcome: str) -> None:
    root = _root(tmp_path)
    payload = {
        "event_id": "a" * 64, "cuda_process_fault": outcome == "fresh_process_required",
        "chargeable_cuda_seconds": 1.5, "peak_allocated_bytes": 10,
        "peak_reserved_bytes": 20, "total_vram_bytes": 30,
        "message": "smoke fault", "error_type": "RuntimeError",
    }
    event = {"index": 0, "timestamp": "2026-01-01T00:00:00+00:00", "kind": "stage_attempt", "stage": "smoke", "attempt": 1, "outcome": outcome, "payload": payload}
    (root / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    assert _load_events(root, inventory)[0]["payload"]["total_vram_bytes"] == 30
    assert build_report(root)["failure_history"]


def test_stage_complete_and_stage_attempt_reject_malformed_extra_fields(tmp_path: Path) -> None:
    root = _root(tmp_path)
    event = {"index": 0, "timestamp": "2026-01-01T00:00:00+00:00", "kind": "stage_complete", "stage": "smoke", "attempt": 1, "outcome": "passed", "payload": {}, "unexpected": True}
    (root / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    with pytest.raises(ValueError, match="event"):
        _load_events(root, inventory)


def test_training_failure_requires_full_identity_and_failed_environment_linkage(tmp_path: Path) -> None:
    root = _root(tmp_path)
    malformed = {"index": 0, "timestamp": "2026-01-01T00:00:00+00:00", "kind": "training_failure", "attempt": 1, "message": "x", "error_type": "E", "cuda_process_fault": False}
    (root / "events.jsonl").write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    with pytest.raises(ValueError, match="required"):
        _load_events(root, inventory)


def test_failed_environment_cannot_cross_link_failure_identity(tmp_path: Path) -> None:
    root = _root(tmp_path)
    failure = {"index": 0, "timestamp": "2026-01-01T00:00:00+00:00", "kind": "training_failure", "stage": "pilot", "arm": "A1", "seed": 47, "attempt": 1, "update": 0, "event_id": "a" * 64, "message": "x", "error_type": "E", "cuda_process_fault": False, "payload": {"message": "x"}}
    terminal = {"index": 1, "timestamp": "2026-01-01T00:00:01+00:00", "kind": "failed_environment", "stage": "pilot", "arm": "A2", "seed": 48, "attempts": [1], "failure_event_indices": [0], "failure_event_ids": ["a" * 64], "completion_status": "failed_environment", "payload": {"attempts": [1], "failure_event_indices": [0], "failure_event_ids": ["a" * 64]}}
    (root / "events.jsonl").write_text("\n".join(json.dumps(item) for item in (failure, terminal)) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    with pytest.raises(ValueError, match="identity|stage|arm|seed"):
        _load_events(root, inventory)


def test_training_failure_id_must_match_canonical_task9_tuple(tmp_path: Path) -> None:
    root = _root(tmp_path)
    failure = {"index": 0, "timestamp": "2026-01-01T00:00:00+00:00", "kind": "training_failure", "stage": "pilot", "arm": "A1", "seed": 47, "attempt": 1, "update": 0, "message": "x", "error_type": "E", "cuda_process_fault": False, "payload": {"message": "x"}}
    failure["event_id"] = "a" * 64
    (root / "events.jsonl").write_text(json.dumps(failure) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    with pytest.raises(ValueError, match="canonical|identity"):
        _load_events(root, inventory)
    failure["event_id"] = task9_failure_event_id(failure)
    (root / "events.jsonl").write_text(json.dumps(failure) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    assert _load_events(root, inventory)[0]["event_id"] == failure["event_id"]


def test_observed_points_are_not_bootstrap_means_for_nonconstant_producers(tmp_path: Path) -> None:
    accuracy_records = {
        (arm, seed): [
            {"example_id": "expressions-0", "family": "expressions", "correct": arm == "A2"},
            {"example_id": "expressions-1", "family": "expressions", "correct": arm == "A2" and seed == 47},
            {"example_id": "lists-0", "family": "lists", "correct": arm == "A2"},
            {"example_id": "graphs-0", "family": "graphs", "correct": arm == "A2"},
        ]
        for arm in ("F0", "A1", "A2", "A3")
        for seed in ((47,) if arm == "F0" else (47, 48))
    }
    accuracy = paired_accuracy_bootstrap(accuracy_records, replicates=10000, seed=9047)
    observed = float(accuracy["comparisons"]["A2-F0"]["point"])
    sampled = np.asarray(accuracy["comparisons"]["A2-F0"]["bootstrap"], dtype=float)
    assert not np.isclose(observed, float(sampled.mean()))

    aggregates = {}
    for (arm, seed), rows in accuracy_records.items():
        path = tmp_path / "run" / "scores" / f"{arm}-{seed}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        aggregates[(arm, seed)] = {"raw_path": path.relative_to(tmp_path / "run").as_posix()}
    root = tmp_path / "run"
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    points = _derive_observed_comparison_points(root, inventory, "accuracy", {"expressions": ["expressions-0", "expressions-1"], "lists": ["lists-0"], "graphs": ["graphs-0"]}, aggregates)
    assert points is not None and points["A2-F0"] == pytest.approx(observed)

    candidate = [{"chunk_index": 0, "scored_tokens": 1, "nll_sum": 0.0}, {"chunk_index": 1, "scored_tokens": 9, "nll_sum": 9.0}]
    baseline = [{"chunk_index": 0, "scored_tokens": 1, "nll_sum": 0.0}, {"chunk_index": 1, "scored_tokens": 9, "nll_sum": 0.0}]
    retention = paired_retention_bootstrap(candidate, baseline, replicates=10000, seed=9157)
    retention_point = float(retention["point"])
    retention_vector = np.asarray(retention["bootstrap"], dtype=float)
    assert not np.isclose(retention_point, float(retention_vector.mean()))


def test_registered_bootstrap_reconciles_accuracy_point_to_raw_producer_semantics(tmp_path: Path) -> None:
    root = _root(tmp_path, decision="passed")
    families = {family: [f"{family}-0", f"{family}-1"] for family in ("expressions", "lists", "graphs")}
    aggregates = {}
    for arm in ("F0", "A1", "A2", "A3"):
        for seed in ((47,) if arm == "F0" else (47, 48)):
            rows = []
            for family, identifiers in families.items():
                for identifier in identifiers:
                    rows.append({"example_id": identifier, "family": family, "correct": arm == "A2" and (identifier.endswith("-0") or seed == 47)})
            path = root / "scores" / f"{arm}-{seed}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            aggregates[(arm, seed)] = {"raw_path": path.relative_to(root).as_posix()}
    matrix_path = root / "bootstrap" / "matrix.json"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9047)
    matrix_payload = {"families": {family: {"ids": identifiers, "indices": rng.integers(0, len(identifiers), size=(10000, len(identifiers))).tolist()} for family, identifiers in families.items()}}
    _write(matrix_path, matrix_payload)
    matrix_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    derived = _derive_bootstrap_comparisons(root, inventory, "accuracy", families, {family: item["indices"] for family, item in matrix_payload["families"].items()}, {}, aggregates)
    points = _derive_observed_comparison_points(root, inventory, "accuracy", families, aggregates)
    assert derived is not None and points is not None
    comparisons = {}
    for name, vector in derived.items():
        path = root / "bootstrap" / f"{name}.json"
        data = json.dumps({"values": vector, "comparison": name}, separators=(",", ":")).encode()
        path.write_bytes(data)
        comparisons[name] = {"vector_path": path.relative_to(root).as_posix(), "vector_sha256": hashlib.sha256(data).hexdigest(), "matrix_sha256": matrix_hash, "evaluation_seeds": [47, 48], "point": points[name], "ci_low": float(np.percentile(vector, 2.5)), "ci_high": float(np.percentile(vector, 97.5)), "passed": True}
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    evidence = {"accuracy": "registered", "seed": 9047, "replicates": 10000, "bootstrap": [0.0] * 10000, "bootstrap_sha256": "a" * 64, "paired_sample_indices_path": matrix_path.relative_to(root).as_posix(), "paired_sample_indices_sha256": matrix_hash, "comparisons": comparisons}
    stages = {"discover": {"status": "passed", "payload": {"accuracy": evidence}}}
    assert _read_bootstrap_matrix(root, inventory, matrix_path.relative_to(root).as_posix(), matrix_hash, families, registered_seed=9047) is not None
    for name, item in comparisons.items():
        assert _read_bootstrap_vector(root, inventory, item["vector_path"], item["vector_sha256"]) == derived[name]
        assert np.isclose(item["point"], points[name])
        assert np.isclose(item["ci_low"], np.percentile(derived[name], 2.5))
        assert np.isclose(item["ci_high"], np.percentile(derived[name], 97.5))
    assert _has_registered_bootstrap(root, inventory, stages, 9047, "accuracy", families, aggregates)
    incorrect = json.loads(json.dumps(stages))
    for item in incorrect["discover"]["payload"]["accuracy"]["comparisons"].values():
        item["point"] = float(np.mean([float(value) for value in next(iter(derived.values()))]))
    assert not _has_registered_bootstrap(root, inventory, incorrect, 9047, "accuracy", families, aggregates)


@pytest.mark.parametrize(
    "event",
    [
        {"index": 0, "timestamp": "2026-01-01T00:00:00+00:00", "kind": "training_started", "arm": "A1", "seed": "47", "attempt": 1, "update": 0},
        {"index": 0, "timestamp": "2026-01-01T00:00:00+00:00", "kind": "training_started", "arm": "A1", "seed": 47, "attempt": 1, "update": 0, "message": "unexpected"},
    ],
)
def test_event_registered_schema_rejects_wrong_types_and_extra_fields(tmp_path: Path, event: dict) -> None:
    root = _root(tmp_path)
    (root / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    with pytest.raises(ValueError, match="event"):
        _load_events(root, inventory)


def test_accuracy_bootstrap_rejects_negative_unweighted_seed_macro(tmp_path: Path) -> None:
    root = _root(tmp_path)
    families = ("expressions", "lists", "graphs")
    aggregates = {}
    for arm in ("F0", "A1", "A2", "A3"):
        for seed in (47, 48):
            path = root / "scores" / f"{arm}-{seed}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [{"example_id": family, "family": family, "correct": arm == "A2" and seed == 47} for family in families]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            aggregates[(arm, seed)] = {"raw_path": path.relative_to(root).as_posix()}
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    matrix = {family: [[0]] * 10000 for family in families}
    assert _derive_bootstrap_comparisons(root, inventory, "accuracy", {family: [family] for family in families}, matrix, {}, aggregates) is None


def test_retention_bootstrap_rejects_full_data_ratio_despite_favorable_matrix(tmp_path: Path) -> None:
    root = _root(tmp_path)
    aggregates = {}
    for arm in ("F0", "A2"):
        for seed in ((47,) if arm == "F0" else (47, 48)):
            path = root / "retention" / f"{arm}-{seed}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            nll = 0.0 if arm == "F0" else 1.0
            path.write_text("".join(json.dumps({"chunk_index": index, "scored_tokens": 1, "nll_sum": nll}) + "\n" for index in range(2)), encoding="utf-8")
            aggregates[(arm, seed)] = {"raw_path": path.relative_to(root).as_posix()}
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    matrix = {"retention": [[0]] * 10000}
    assert _derive_bootstrap_comparisons(root, inventory, "retention", {"retention": ["0", "1"]}, matrix, {}, aggregates) is None


def test_retention_bootstrap_derives_distinct_seed_vectors(tmp_path: Path) -> None:
    root = _root(tmp_path)
    aggregates = {}
    for arm in ("F0", "A2"):
        for seed in ((47,) if arm == "F0" else (47, 48)):
            path = root / "retention" / f"{arm}-{seed}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps({"chunk_index": index, "scored_tokens": 1, "nll_sum": 0.0}) + "\n" for index in range(2)), encoding="utf-8")
            aggregates[(arm, seed)] = {"raw_path": path.relative_to(root).as_posix()}
    inventory = {str(item["path"]): item for item in _read_inputs(root)}
    matrix = {"retention": [[0, 1]] * 10000}
    derived = _derive_bootstrap_comparisons(root, inventory, "retention", {"retention": ["0", "1"]}, matrix, {}, aggregates)
    assert derived is not None
    assert set(derived) == {"A2-F0@47", "A2-F0@48"}


def test_report_hashes_sources_and_writes_immutable_outputs(tmp_path: Path) -> None:
    root = _root(tmp_path)
    report = build_report(root)
    manifest = json.loads((root / "report-sources.json").read_text(encoding="utf-8"))
    source = next(item for item in manifest["files"] if item["path"] == "source-manifest.json")
    assert source["sha256"] == hashlib.sha256((root / "source-manifest.json").read_bytes()).hexdigest()
    assert report["source_manifest"]["revision"] == "e" * 40
    before = (root / "report.json").read_bytes()
    assert build_report(root)["report_sha256"] == report["report_sha256"]
    assert (root / "report.json").read_bytes() == before


def test_preconfirmation_report_does_not_open_sealed_rows_or_test_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ppmoe.recurrence.report as report_module

    root = _root(tmp_path)
    sealed_paths = (
        root / "data" / "test.jsonl",
        root / "scores" / "a1-47-test.jsonl",
        root / "scores" / "a1-47-test.summary.json",
        root / "scores" / "a1-48-confirm.jsonl",
        root / "retention-scores" / "a1-48-test.jsonl",
        root / "retention-scores" / "a1-48-confirm.jsonl",
        root / "retention-scores" / "a1-48-confirm.summary.json",
    )
    for path in sealed_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed fixture bytes")

    original_read = report_module._descriptor_read

    def guarded_read(path: Path, trusted_root: Path | None = None):
        relative = path.relative_to(root).as_posix()
        assert relative != "data/test.jsonl", "report opened sealed test rows"
        is_score_path = relative.startswith(("scores/", "retention-scores/"))
        assert not (is_score_path and any(token in path.name for token in ("-test", "-confirm"))), (
            "pre-confirm report opened a sealed score path"
        )
        return original_read(path, trusted_root)

    monkeypatch.setattr(report_module, "_descriptor_read", guarded_read)

    report = build_report(root)

    assert report["decision"]["status"] == "stopped_scientific"


def test_confirmed_report_reads_test_scores_but_never_sealed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ppmoe.recurrence.report as report_module
    from ppmoe.recurrence.provenance import canonical_json

    root = _root(tmp_path)
    signatures = {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64}
    test_hash = "e" * 64
    data_manifest = {"split_sha256": {"test": test_hash}}
    manifest_bytes = canonical_json(data_manifest)
    (root / "data" / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "data" / "manifest.json").write_bytes(manifest_bytes)
    selection = {
        "schema_version": 1,
        "status": "passed",
        "selected_condition": "A2",
        "signatures": signatures,
        "test_sha256": test_hash,
        "n3": 1000,
        "dev_score_hashes": {arm: "f" * 64 for arm in ("F0", "A1", "A2", "A3")},
        "data_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    selection_bytes = canonical_json(selection)
    (root / "selection.json").write_bytes(selection_bytes)
    _stage(root, "data", payload={"manifest_sha256": selection["data_manifest_sha256"]})
    _stage(root, "discover", payload={
        "decision": {"status": "stopped_scientific", "reason": "confirmation_fixture"},
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "data_manifest_sha256": selection["data_manifest_sha256"],
    })
    _stage(root, "confirm", payload={
        "decision": {"status": "stopped_scientific", "reason": "confirmation_fixture"},
    })
    test_score = root / "scores" / "a1-47-test.jsonl"
    retention_score = root / "retention-scores" / "a1-48-confirm.jsonl"
    sealed_rows = root / "data" / "test.jsonl"
    for path in (test_score, retention_score, sealed_rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque test fixture")

    opened: list[str] = []
    original_read = report_module._descriptor_read

    def guarded_read(path: Path, trusted_root: Path | None = None):
        relative = path.relative_to(root).as_posix()
        assert relative != "data/test.jsonl", "report opened sealed test rows"
        opened.append(relative)
        return original_read(path, trusted_root)

    monkeypatch.setattr(report_module, "_descriptor_read", guarded_read)

    report = build_report(root)

    assert report["decision"]["status"] == "stopped_scientific"
    assert "scores/a1-47-test.jsonl" in opened
    assert "retention-scores/a1-48-confirm.jsonl" in opened
    assert "data/test.jsonl" not in opened
    report_sources = json.loads((root / "report-sources.json").read_text(encoding="utf-8"))
    assert "scores/a1-47-test.jsonl" in {entry["path"] for entry in report_sources["files"]}


def test_report_rejects_summary_without_raw_records(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write(root / "scores" / "a2-47.summary.json", {
        "schema_version": 1, "kind": "exact_scores", "complete": True, "record_count": 1,
        "expected_count": 1, "arm": "A2", "seed": 47, "failures": 0,
        "families": ["expressions", "lists", "graphs"], "signatures": {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64},
    })
    with pytest.raises(ValueError, match="raw score"):
        build_report(root)


def test_incomplete_summary_is_not_evaluated_and_not_an_aggregate(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write(root / "scores" / "a2-47.summary.json", {
        "kind": "exact_scores", "complete": False, "record_count": 0,
        "expected_count": 1, "arm": "A2", "seed": 47,
    })
    report = build_report(root)
    assert report["evaluations"]["exact_scores"]["status"] == "not_evaluated"
    assert report["evaluations"]["exact_scores"]["aggregates"] == []


def test_complete_summary_requires_declared_raw_hash(tmp_path: Path) -> None:
    root = _root(tmp_path)
    raw = root / "scores" / "a2-47.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"example_id": "x", "family": "expressions", "correct": True}) + "\n", encoding="utf-8")
    _write(root / "scores" / "a2-47.summary.json", {
        "kind": "exact_scores", "complete": True, "record_count": 1,
        "expected_count": 1, "records_path": "scores/a2-47.jsonl", "arm": "A2", "seed": 47,
    })
    with pytest.raises(ValueError, match="declared raw"):
        build_report(root)


def test_report_rejects_record_count_expected_count_disagreement(tmp_path: Path) -> None:
    root = _root(tmp_path, decision="passed")
    _stage(root, "discover", payload={})
    raw = root / "scores" / "a2-47.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"example_id": "x", "family": "expressions", "correct": True}) + "\n", encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    _write(root / "scores" / "a2-47.summary.json", {
        "schema_version": 1, "kind": "exact_scores", "complete": True, "record_count": 999,
        "expected_count": 1, "records_path": "scores/a2-47.jsonl", "records_sha256": digest,
        "arm": "A2", "seed": 47, "failures": 0, "families": ["expressions", "lists", "graphs"], "signatures": {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64},
        "split": "dev",
    })
    with pytest.raises(ValueError, match="count"):
        build_report(root)


def test_source_manifest_reconciles_current_file_bytes(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest"):
        build_report(root)


def test_source_manifest_paths_resolve_inside_authenticated_snapshot(tmp_path: Path) -> None:
    root = _root(tmp_path)
    revision = "e" * 40
    snapshot = root / "source" / revision
    source_file = snapshot / ".cache" / "config.bin"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"pinned source")
    (root / "config.json").unlink()
    _stage(root, "source", payload={
        "source_manifest": "source-manifest.json",
        "revision": revision,
        "snapshot_relative_path": f"source/{revision}",
    })
    _write(root / "source-manifest.json", {
        "schema_version": 1,
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": revision,
        "files": [{
            "relative_path": ".cache/config.bin",
            "bytes": len(b"pinned source"),
            "sha256": hashlib.sha256(b"pinned source").hexdigest(),
        }],
    })

    report = build_report(root)

    assert report["source_manifest"]["revision"] == revision


def test_source_manifest_revision_must_match_authenticated_snapshot(tmp_path: Path) -> None:
    root = _root(tmp_path)
    revision = "e" * 40
    _stage(root, "source", payload={
        "source_manifest": "source-manifest.json",
        "revision": revision,
        "snapshot_relative_path": f"source/{revision}",
    })
    manifest = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
    manifest["revision"] = "f" * 40
    _write(root / "source-manifest.json", manifest)

    with pytest.raises(ValueError, match="revision.*source stage|source stage.*revision"):
        build_report(root)


@pytest.mark.parametrize("relative_path", ["../config.json", "..\\config.json", "C:/config.json"])
def test_source_manifest_paths_cannot_escape_authenticated_snapshot(
    tmp_path: Path, relative_path: str
) -> None:
    root = _root(tmp_path)
    revision = "e" * 40
    snapshot = root / "source" / revision
    snapshot.mkdir(parents=True)
    _stage(root, "source", payload={
        "source_manifest": "source-manifest.json",
        "revision": revision,
        "snapshot_relative_path": f"source/{revision}",
    })
    _write(root / "source-manifest.json", {
        "schema_version": 1,
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": revision,
        "files": [{
            "relative_path": relative_path,
            "bytes": 2,
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        }],
    })

    with pytest.raises(ValueError, match="source manifest path"):
        build_report(root)


def test_splitless_exact_summary_cannot_bind_to_a_sealed_producer() -> None:
    from ppmoe.recurrence.report import _validate_summary_schema

    signatures = {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64}
    summary = {
        "schema_version": 1, "kind": "exact_scores", "record_count": 6,
        "expected_count": 6, "complete": True, "records_path": "scores/a1-47.jsonl",
        "records_sha256": "e" * 64, "arm": "A1", "seed": 47,
        "failures": 0, "signatures": signatures,
        "families": ["expressions", "lists", "graphs"], "split": None,
    }
    stages = {"discover": {"status": "passed", "signatures": signatures}}
    with pytest.raises(ValueError, match="invalid split"):
        _validate_summary_schema(summary, "exact_scores", stages, "scores/a1-47.summary.json")


def test_complete_registered_success_fixture_requires_and_accepts_full_matrix(tmp_path: Path) -> None:
    root = _root(tmp_path, decision="passed")
    signatures = {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64}
    families = ("expressions", "lists", "graphs")
    ids = [f"{family}-{index}" for family in families for index in range(2)]
    for arm in ("F0", "A1", "A2", "A3"):
        for seed in ((47,) if arm == "F0" else (47, 48)):
            raw = root / "scores" / f"{arm.lower()}-{seed}.jsonl"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("".join(json.dumps({"example_id": identifier, "family": identifier.rsplit("-", 1)[0], "correct": arm == "A2", "split": "test"}) + "\n" for identifier in ids), encoding="utf-8")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            _write(root / "scores" / f"{arm.lower()}-{seed}.summary.json", {
                "schema_version": 1, "kind": "exact_scores", "complete": True, "record_count": 6, "expected_count": 6,
                "records_path": raw.relative_to(root).as_posix(), "records_sha256": digest,
                "arm": arm, "seed": seed, "failures": 0, "families": ["expressions", "lists", "graphs"], "signatures": signatures, "split": "test",
            })
    for arm in ("F0", "A1", "A2", "A3"):
        for seed in ((47,) if arm == "F0" else (47, 48)):
            raw = root / "retention" / f"{arm.lower()}-{seed}.jsonl"
            raw.parent.mkdir(parents=True, exist_ok=True)
            split = "dev" if seed == 47 else "test"
            raw.write_text("".join(json.dumps({"chunk_index": index, "scored_tokens": 2, "nll_sum": 1.0, "split": split}) + "\n" for index in range(2)), encoding="utf-8")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            _write(root / "retention" / f"{arm.lower()}-{seed}.summary.json", {
                "schema_version": 1, "kind": "retention_scores", "complete": True, "record_count": 2, "expected_count": 2,
                "records_path": raw.relative_to(root).as_posix(), "records_sha256": digest,
                "arm": arm, "seed": seed, "failures": 0, "families": ["retention"], "signatures": signatures, "nll_sum": 2.0, "scored_tokens": 4, "split": split,
            })
    bootstrap = root / "bootstrap"
    vector = json.dumps({"values": [1.0] * 10000}, separators=(",", ":")).encode()
    retention_vector = json.dumps({"values": [1.0] * 10000}, separators=(",", ":")).encode()
    accuracy_rng = np.random.default_rng(9047)
    accuracy_indices = json.dumps({"families": {family: {"ids": [f"{family}-0", f"{family}-1"], "indices": accuracy_rng.integers(0, 2, size=(10000, 2)).tolist()} for family in families}}, separators=(",", ":")).encode()
    retention_rng = np.random.default_rng(9157)
    retention_indices = json.dumps({"families": {"retention": {"ids": ["0", "1"], "indices": retention_rng.integers(0, 2, size=(10000, 2)).tolist()}}}, separators=(",", ":")).encode()
    bootstrap.mkdir(parents=True, exist_ok=True)
    (bootstrap / "retention-9157.json").write_bytes(retention_vector)
    (bootstrap / "9047-indices.json").write_bytes(accuracy_indices)
    (bootstrap / "9157-indices.json").write_bytes(retention_indices)
    def _bootstrap(seed: int, label: str) -> dict:
        vector_hash = hashlib.sha256(vector).hexdigest()
        matrix = accuracy_indices if label == "accuracy" else retention_indices
        indices_hash = hashlib.sha256(matrix).hexdigest()
        names = ["A2-F0", "A2-A1", "A2-A3"] if label == "accuracy" else ["A2-F0@47", "A2-F0@48"]
        comparisons = {}
        for name in names:
            if label == "accuracy":
                path = bootstrap / f"accuracy-{name}.json"
                data = json.dumps({"values": [1.0] * 10000, "comparison": name}, separators=(",", ":")).encode()
                path.write_bytes(data)
            else:
                seed_value = int(name.rsplit("@", 1)[1])
                path = bootstrap / f"retention-{seed_value}.json"
                data = json.dumps({"values": [1.0] * 10000, "comparison": name, "seed": seed_value}, separators=(",", ":")).encode()
                path.write_bytes(data)
            comparisons[name] = {"vector_path": path.relative_to(root).as_posix(), "vector_sha256": hashlib.sha256(data).hexdigest(), "matrix_sha256": indices_hash, "evaluation_seeds": [47, 48], "point": 1.0, "ci_low": 1.0, "ci_high": 1.0, "passed": True}
            if label == "retention":
                comparisons[name]["seed"] = seed_value
                comparisons[name]["evaluation_seeds"] = [seed_value]
        return {"seed": seed, "replicates": 10000, "bootstrap_sha256": vector_hash,
                "paired_sample_indices_sha256": indices_hash, "vector_path": f"bootstrap/{seed}.json",
                "paired_sample_indices_path": f"bootstrap/{seed}-indices.json",
                "comparisons": comparisons}
    timing47 = {"seed": 47, "sample_count": 20, "metadata": {"units": "seconds", "synchronized": True, "warmup_samples": 5, "arms": ["A2", "A3"]}, "a2_training_times": [1.0] * 20, "a3_training_times": [1.0] * 20}
    timing48 = {"seed": 48, "sample_count": 20, "metadata": {"units": "seconds", "synchronized": True, "warmup_samples": 5, "arms": ["A2", "A3"]}, "a2_training_times": [1.0] * 20, "a3_training_times": [1.0] * 20}
    _stage(root, "discover", payload={"decision": {"status": "passed"}, "bootstrap": _bootstrap(9047, "accuracy"), "timing": timing47})
    _stage(root, "confirm", payload={"decision": {"status": "passed"}, "bootstrap": _bootstrap(9157, "retention"), "timing": timing48})
    report = build_report(root)
    assert report["decision"]["status"] == "passed"
    assert "fixed two-seed, three-family protocol" in report["conclusion"]


def test_source_decision_cannot_authorize_a_passed_report(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _stage(root, "source", payload={"decision": {"status": "passed"}, "source_manifest": "source-manifest.json", "revision": "e" * 40})
    report = build_report(root)
    assert report["decision"]["status"] != "passed"


def test_source_to_data_signature_progression_is_registered(tmp_path: Path) -> None:
    root = tmp_path / "run"
    signatures = {"source": "a" * 64, "data": "", "config": "c" * 64, "code": "d" * 64}
    _stage(root, "source", payload={})
    _write(root / "stages" / "source.json", {"stage": "source", "status": "passed", "signatures": signatures, "payload": {}, "attempts": 1})
    _write(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": {**signatures, "data": "b" * 64}, "payload": {}, "attempts": 1})
    report = build_report(root)
    assert report["signatures"]["data"] == "b" * 64


def test_cuda_ledger_is_reportable(tmp_path: Path) -> None:
    root = _root(tmp_path)
    from ppmoe.recurrence.gates import CudaCostLedger
    ledger = CudaCostLedger(root / "cuda-cost.jsonl", 86400.0)
    ledger.record("x", "frozen", 1.0, "passed")
    report = build_report(root)
    assert report["resources"]["cuda"]["status"] == "evaluated"


def test_summary_signature_must_match_registered_stage(tmp_path: Path) -> None:
    root = _root(tmp_path, decision="passed")
    _stage(root, "discover", payload={"decision": {"status": "stopped_scientific"}})
    raw = root / "scores" / "a2-47.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"example_id": "x", "family": "expressions", "correct": True}) + "\n", encoding="utf-8")
    _write(root / "scores" / "a2-47.summary.json", {
        "schema_version": 1, "kind": "exact_scores", "complete": True, "record_count": 1,
        "expected_count": 1, "records_path": "scores/a2-47.jsonl", "records_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "arm": "A2", "seed": 47, "failures": 0, "families": ["expressions", "lists", "graphs"], "signatures": {"source": "0" * 64, "data": "0" * 64, "config": "0" * 64, "code": "0" * 64},
        "split": "dev",
    })
    with pytest.raises(ValueError, match="signature"):
        build_report(root)


def test_negative_retention_nll_is_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    raw = root / "retention" / "f0-47.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"chunk_index": 0, "scored_tokens": 1, "nll_sum": -1.0}) + "\n", encoding="utf-8")
    _write(root / "retention" / "f0-47.summary.json", {
        "schema_version": 1, "kind": "retention_scores", "complete": True, "record_count": 1, "families": ["retention"], "signatures": {"source": "a" * 64, "data": "b" * 64, "config": "c" * 64, "code": "d" * 64},
        "expected_count": 1, "records_path": "retention/f0-47.jsonl", "records_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "arm": "F0", "seed": 47, "failures": 0, "nll_sum": -1.0, "scored_tokens": 1,
        "split": "dev",
    })
    with pytest.raises(ValueError, match="non-negative"):
        build_report(root)


def test_g4_accuracy_no_go_terminal_fixture_is_reported_narrowly(tmp_path: Path) -> None:
    root = _root(tmp_path, decision="passed")
    _stage(root, "discover", status="stopped_scientific", payload={"decision": {"status": "stopped_scientific", "reason": "accuracy_no_go"}})
    report = build_report(root)
    assert report["decision"]["reason"] == "accuracy_no_go"
    assert "did not pass the registered quality gate" in report["conclusion"]


def test_g5_confirmation_failure_terminal_fixture_is_reported_narrowly(tmp_path: Path) -> None:
    root = _root(tmp_path, decision="passed")
    _stage(root, "discover", payload={"decision": {"status": "passed"}})
    _stage(root, "confirm", status="stopped_scientific", payload={"decision": {"status": "stopped_scientific", "reason": "confirmation_no_go"}})
    report = build_report(root)
    assert report["decision"]["reason"] == "confirmation_no_go"
    assert "did not pass the registered quality gate" in report["conclusion"]


def test_report_rejects_success_missing_registered_arm(tmp_path: Path) -> None:
    root = _root(tmp_path, decision="passed")
    _stage(root, "discover", payload={"decision": {"status": "passed"}})
    _stage(root, "confirm", payload={"decision": {"status": "passed"}})
    with pytest.raises(ValueError, match="arm"):
        build_report(root)


@pytest.mark.parametrize(
    ("status", "reason", "phrase"),
    [
        ("stopped_scientific", "retention_uncertain", "retention bound was not established"),
        ("stopped_scientific", "timing_inconclusive", "timing-controlled comparison was inconclusive"),
        ("stopped_budget", None, "no scientific conclusion"),
        ("failed_environment", None, "execution evidence is incomplete"),
    ],
)
def test_report_uses_narrow_status_conclusion(tmp_path: Path, status: str, reason: str | None, phrase: str) -> None:
    root = _root(tmp_path, decision=status)
    payload = {"decision": {"status": status}}
    if reason:
        payload["decision"]["reason"] = reason
    _stage(root, "frozen", payload=payload)
    report = build_report(root)
    assert phrase in report["conclusion"]
    assert "general intelligence" in report["conclusion"]
