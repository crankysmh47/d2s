from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from hashlib import sha256
import json
import math
import multiprocessing

import pytest
import torch

from ppmoe.recurrence import gates
from ppmoe.recurrence.config import RecurrenceConfig
from ppmoe.recurrence.gates import (
    CudaCostLedger,
    classify_informative_families,
    confirmation_decision,
    derive_a3_steps,
    discovery_decision,
    is_informative_family,
    measure_training_arms,
    measure_training_step,
    paired_accuracy_bootstrap,
    paired_retention_bootstrap,
    resource_decision,
    summarize_exact_scores,
)


def _row(identifier: str, family: str, correct: bool, reason: str | None = None) -> dict[str, object]:
    return {
        "example_id": identifier,
        "family": family,
        "correct": correct,
        "reason_code": reason or ("correct" if correct else "wrong_value"),
    }


def test_summary_uses_unweighted_family_macro_and_transition_counts() -> None:
    rows = [
        _row("e0", "expressions", True),
        _row("e1", "expressions", False),
        _row("l0", "lists", True),
    ]
    summary = summarize_exact_scores(rows)
    assert summary["macro_accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert summary["pooled_accuracy"] == pytest.approx(2 / 3)
    assert summary["families"]["expressions"]["accuracy"] == pytest.approx(0.5)

    candidate = [_row("e0", "expressions", True), _row("e1", "expressions", True), _row("l0", "lists", True)]
    transitions = summarize_exact_scores(candidate, baseline_records=rows)["transitions"]
    assert transitions == {"wrong_to_right": 1, "right_to_wrong": 0, "unchanged": 2}


@pytest.mark.parametrize(
    ("accuracy", "invalid_rate", "informative"),
    [(0.05, 0.00, False), (0.05001, 0.10, True), (0.94999, 0.10, True), (0.95, 0.00, False), (0.50, 0.10001, False)],
)
def test_family_informativeness_boundaries(accuracy: float, invalid_rate: float, informative: bool) -> None:
    assert is_informative_family(accuracy, invalid_rate) is informative


def test_family_classification_keeps_all_families() -> None:
    summary = summarize_exact_scores(
        [_row("e", "expressions", True), _row("l", "lists", False), _row("g", "graphs", True)]
    )
    result = classify_informative_families(summary)
    assert set(result["families"]) == {"expressions", "lists", "graphs"}
    assert result["minimum_informative_families_passed"] is False


def test_accuracy_bootstrap_is_reproducible_and_pairs_ids() -> None:
    def rows(arm: str) -> list[dict[str, object]]:
        values = {"e0": True, "e1": False, "e2": True, "l0": False}
        if arm == "A2":
            values["e1"] = True
            values["l0"] = True
        return [_row(identifier, "expressions" if identifier.startswith("e") else "lists", value) for identifier, value in values.items()]

    records = {("F0", 47): rows("F0"), ("A2", 47): rows("A2")}
    first = paired_accuracy_bootstrap(records)
    second = paired_accuracy_bootstrap(records)
    assert first == second
    result = first["comparisons"]["A2-F0"]
    assert result["replicates"] == 10_000
    assert result["point"] == pytest.approx((1 / 3 + 1) / 2)
    assert result["wrong_to_right"] == 2
    assert result["right_to_wrong"] == 0
    assert result["ci_low"] <= result["ci_high"]

    bad = dict(records)
    bad[("A2", 47)] = rows("A2")[:-1]
    with pytest.raises(ValueError, match="ID set"):
        paired_accuracy_bootstrap(bad)


def test_accuracy_bootstrap_reuses_one_common_sample_matrix() -> None:
    def rows(arm: str) -> list[dict[str, object]]:
        return [
            _row(f"e{i}", "expressions", arm == "A2" or i == 0)
            for i in range(4)
        ]

    records = {
        ("F0", 47): rows("F0"),
        ("A1", 47): rows("A1"),
        ("A2", 47): rows("A2"),
    }
    result = paired_accuracy_bootstrap(records)
    assert result["comparisons"]["A2-F0"]["sample_indices"] == result["comparisons"]["A2-A1"]["sample_indices"]


def test_accuracy_bootstrap_rejects_family_universe_drift() -> None:
    base = [_row("e", "expressions", True), _row("l", "lists", True), _row("g", "graphs", True)]
    records = {("F0", 47): base, ("A2", 47): [*base[:-1], _row("g", "lists", True)]}
    with pytest.raises(ValueError, match="family universe"):
        paired_accuracy_bootstrap(records)


def test_retention_bootstrap_recomputes_token_weighted_ratio() -> None:
    candidate = [
        {"chunk_index": 0, "nll_sum": 2.0, "scored_tokens": 2},
        {"chunk_index": 1, "nll_sum": 9.0, "scored_tokens": 3},
    ]
    f0 = [
        {"chunk_index": 0, "nll_sum": 4.0, "scored_tokens": 2},
        {"chunk_index": 1, "nll_sum": 6.0, "scored_tokens": 3},
    ]
    result = paired_retention_bootstrap(candidate, f0)
    assert result["replicates"] == 10_000
    assert result["point"] == pytest.approx(math.exp(11 / 5 - 10 / 5))
    assert result == paired_retention_bootstrap(candidate, f0)
    with pytest.raises(ValueError, match="chunk index"):
        paired_retention_bootstrap(candidate, f0[:1])
    with pytest.raises(ValueError, match="token counts"):
        paired_retention_bootstrap(candidate, [{**row, "scored_tokens": row["scored_tokens"] + 1} for row in f0])


def test_discovery_decision_reports_named_reasons() -> None:
    scores = {
        "a2_minus_f0_macro": 0.03,
        "a2_minus_a1_macro": 0.03,
        "a2_minus_a3_macro": 0.03,
        "a2_family_deltas_vs_f0": {"expressions": -0.01, "lists": 0.1, "graphs": 0.0},
        "a2_family_deltas_vs_a1": {"expressions": -0.01, "lists": 0.1, "graphs": 0.0},
    }
    retention = {"point": 1.01, "ci_high": 1.02}
    timing = {"a3_a2_realized_training_time_ratio": 1.0}
    decision = discovery_decision(scores, retention, timing)
    assert decision["status"] == "passed"
    assert all(value["passed"] for value in decision["gates"].values())

    failed = discovery_decision({**scores, "a2_minus_f0_macro": 0.01}, retention, timing)
    assert failed["status"] == "stopped_scientific"
    assert failed["reason"] == "scientific_quality_gate_failed"
    assert discovery_decision(scores, {"point": 1.03, "ci_high": 1.04}, timing)["reason"] == "retention_uncertain"
    assert discovery_decision(scores, retention, {"a3_a2_realized_training_time_ratio": 1.2})["reason"] == "timing_inconclusive"


def test_family_and_ci_gates_reject_missing_or_nonfinite_values() -> None:
    scores = {
        "a2_minus_f0_macro": 0.03,
        "a2_minus_a1_macro": 0.03,
        "a2_minus_a3_macro": 0.03,
        "a2_family_deltas_vs_f0": {"expressions": 0.1, "lists": 0.1, "graphs": float("nan")},
        "a2_family_deltas_vs_a1": {"expressions": 0.1, "lists": 0.1, "graphs": 0.1},
    }
    retention = {"point": 1.0, "ci_high": 1.0}
    timing = {"a3_a2_realized_training_time_ratio": 1.0}
    decision = discovery_decision(scores, retention, timing)
    assert decision["status"] == "stopped_scientific"

    scores["a2_family_deltas_vs_f0"] = {"expressions": 0.1, "lists": 0.1}
    assert discovery_decision(scores, retention, timing)["status"] == "stopped_scientific"


def test_confirmation_uses_strict_ci_and_inclusive_thresholds() -> None:
    scores = {
        "per_seed": {
            47: {
                "a2_minus_f0_macro": 0.01,
                "a2_minus_a1_macro": 0.02,
                "a2_minus_a3_macro": 0.01,
                "a2_family_deltas_vs_f0": {"expressions": -0.02, "lists": 0.1, "graphs": 0.0},
                "a2_family_deltas_vs_a1": {"expressions": -0.02, "lists": 0.1, "graphs": 0.0},
            },
            48: {
                "a2_minus_f0_macro": 0.03,
                "a2_minus_a1_macro": 0.02,
                "a2_minus_a3_macro": 0.03,
                "a2_family_deltas_vs_f0": {"expressions": -0.02, "lists": 0.1, "graphs": 0.0},
                "a2_family_deltas_vs_a1": {"expressions": -0.02, "lists": 0.1, "graphs": 0.0},
            },
        },
        "a2_minus_f0_macro": 0.02,
        "a2_minus_a1_macro": 0.02,
        "a2_minus_a3_macro": 0.02,
        "a2_family_deltas_vs_f0": {"expressions": -0.02, "lists": 0.1, "graphs": 0.0},
        "a2_family_deltas_vs_a1": {"expressions": -0.02, "lists": 0.1, "graphs": 0.0},
        "accuracy_ci_lows": {"a2-f0": 0.0001, "a2-a1": 0.1, "a2-a3": 0.1},
    }
    retention = {47: {"point": 1.02, "ci_high": 1.02}, 48: {"point": 1.02, "ci_high": 1.02}}
    timing = {47: {"a3_a2_realized_training_time_ratio": 0.9}, 48: {"a3_a2_realized_training_time_ratio": 1.1}}
    assert confirmation_decision(scores, retention, timing)["status"] == "passed"
    scores["accuracy_ci_lows"]["a2-f0"] = 0.0
    assert confirmation_decision(scores, retention, timing)["status"] == "stopped_scientific"


def test_confirmation_derives_family_means_from_each_seed() -> None:
    scores = {
        "per_seed": {
            47: {
                "a2_minus_f0_macro": 0.03,
                "a2_minus_a1_macro": 0.03,
                "a2_minus_a3_macro": 0.03,
                "a2_family_deltas_vs_f0": {"expressions": -0.10, "lists": 0.1, "graphs": 0.1},
                "a2_family_deltas_vs_a1": {"expressions": -0.10, "lists": 0.1, "graphs": 0.1},
            },
            48: {
                "a2_minus_f0_macro": 0.03,
                "a2_minus_a1_macro": 0.03,
                "a2_minus_a3_macro": 0.03,
                "a2_family_deltas_vs_f0": {"expressions": -0.05, "lists": 0.1, "graphs": 0.1},
                "a2_family_deltas_vs_a1": {"expressions": -0.05, "lists": 0.1, "graphs": 0.1},
            },
        },
        # Contradictory aggregates must not override the per-seed evidence.
        "a2_family_deltas_vs_f0": {"expressions": 0.0, "lists": 0.0, "graphs": 0.0},
        "a2_family_deltas_vs_a1": {"expressions": 0.0, "lists": 0.0, "graphs": 0.0},
        "accuracy_ci_lows": {"a2-f0": 0.1, "a2-a1": 0.1, "a2-a3": 0.1},
    }
    retention = {47: {"point": 1.0, "ci_high": 1.0}, 48: {"point": 1.0, "ci_high": 1.0}}
    timing = {47: {"a3_a2_realized_training_time_ratio": 1.0}, 48: {"a3_a2_realized_training_time_ratio": 1.0}}
    decision = confirmation_decision(scores, retention, timing)
    assert decision["status"] == "stopped_scientific"
    assert decision["gates"]["every_two_seed_a2_f0_family_delta"]["value"]["expressions"] == pytest.approx(-0.075)


@pytest.mark.parametrize("bad_map", [{"expressions": 0.0, "lists": 0.0}, {"expressions": float("nan"), "lists": 0.0, "graphs": 0.0}])
def test_confirmation_rejects_missing_or_nonfinite_seed_family_maps(bad_map: dict[str, float]) -> None:
    valid = {"expressions": 0.0, "lists": 0.0, "graphs": 0.0}
    scores = {
        "per_seed": {
            47: {
                "a2_minus_f0_macro": 0.03,
                "a2_minus_a1_macro": 0.03,
                "a2_minus_a3_macro": 0.03,
                "a2_family_deltas_vs_f0": bad_map,
                "a2_family_deltas_vs_a1": valid,
            },
            48: {
                "a2_minus_f0_macro": 0.03,
                "a2_minus_a1_macro": 0.03,
                "a2_minus_a3_macro": 0.03,
                "a2_family_deltas_vs_f0": valid,
                "a2_family_deltas_vs_a1": valid,
            },
        },
        "a2_family_deltas_vs_f0": valid,
        "a2_family_deltas_vs_a1": valid,
        "accuracy_ci_lows": {"a2-f0": 0.1, "a2-a1": 0.1, "a2-a3": 0.1},
    }
    retention = {47: {"point": 1.0, "ci_high": 1.0}, 48: {"point": 1.0, "ci_high": 1.0}}
    timing = {47: {"a3_a2_realized_training_time_ratio": 1.0}, 48: {"a3_a2_realized_training_time_ratio": 1.0}}
    decision = confirmation_decision(scores, retention, timing)
    assert decision["status"] == "stopped_scientific"
    assert decision["gates"]["every_two_seed_a2_f0_family_delta"]["passed"] is False


def test_confirmation_requires_exactly_both_registered_seeds() -> None:
    scores = {
        "per_seed": {47: {"a2_minus_f0_macro": 0.03, "a2_minus_a1_macro": 0.03, "a2_minus_a3_macro": 0.03}},
        "a2_minus_f0_macro": 0.03,
        "a2_minus_a1_macro": 0.03,
        "a2_minus_a3_macro": 0.03,
        "a2_family_deltas_vs_f0": {"expressions": 0.0, "lists": 0.0, "graphs": 0.0},
        "a2_family_deltas_vs_a1": {"expressions": 0.0, "lists": 0.0, "graphs": 0.0},
        "accuracy_ci_lows": {"a2-f0": 0.1, "a2-a1": 0.1, "a2-a3": 0.1},
    }
    retention = {47: {"point": 1.0, "ci_high": 1.0}}
    timing = {47: {"a3_a2_realized_training_time_ratio": 1.0}}
    assert confirmation_decision(scores, retention, timing)["status"] == "stopped_scientific"


def test_transition_summary_rejects_family_drift() -> None:
    baseline = [_row("x", "expressions", True)]
    candidate = [_row("x", "lists", True)]
    with pytest.raises(ValueError, match="family"):
        summarize_exact_scores(candidate, baseline_records=baseline)


def test_ledger_charges_completed_failed_and_retried_cuda_attempts_once(tmp_path: Path) -> None:
    """Catches losing failed/retried CUDA intervals or charging them twice."""
    path = tmp_path / "cuda-costs.jsonl"
    ledger = CudaCostLedger.load(path, cap_seconds=86_400.0)
    ledger.record("pilot-a1-1", "pilot", 2.0, "completed")
    ledger.record("train-a2-1", "frozen", 3.0, "failed_environment")
    ledger.record("train-a2-2", "frozen", 4.0, "completed")

    assert ledger.total_seconds == pytest.approx(9.0)
    assert [entry.outcome for entry in ledger.entries] == [
        "completed",
        "failed_environment",
        "completed",
    ]
    assert CudaCostLedger.load(path, cap_seconds=86_400.0).total_seconds == pytest.approx(9.0)

    with pytest.raises(ValueError, match="duplicate"):
        ledger.record("pilot-a1-1", "pilot", 1.0, "completed")
    with pytest.raises(FrozenInstanceError):
        ledger.entries[-1].elapsed_seconds = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("elapsed", [-0.1, float("nan"), float("inf")])
def test_ledger_rejects_invalid_cuda_intervals(elapsed: float, tmp_path: Path) -> None:
    """Catches corrupt attempt entries being allowed into the cumulative cap."""
    ledger = CudaCostLedger.load(tmp_path / "cuda-costs.jsonl", cap_seconds=86_400.0)
    with pytest.raises(ValueError, match="elapsed"):
        ledger.record("attempt", "pilot", elapsed, "completed")


def test_ledger_rejects_decreasing_or_mutated_cumulative_history(tmp_path: Path) -> None:
    """Catches a non-append-only ledger history being trusted after reload."""
    path = tmp_path / "cuda-costs.jsonl"
    ledger = CudaCostLedger.load(path, cap_seconds=86_400.0)
    ledger.record("a1", "pilot", 2.0, "completed")
    ledger.record("a2", "pilot", 1.0, "completed")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[1]["cumulative_seconds"] = 1.0
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        CudaCostLedger.load(path, cap_seconds=86_400.0)


@pytest.mark.parametrize(("field", "replacement"), [("stage", "forged-stage"), ("outcome", "forged-outcome")])
def test_ledger_rejects_finalized_provenance_tampering(
    field: str, replacement: str, tmp_path: Path
) -> None:
    """Catches a finalized attempt's stage/outcome being rewritten on disk."""
    path = tmp_path / "cuda-costs.jsonl"
    ledger = CudaCostLedger.load(path, cap_seconds=86_400.0)
    ledger.record("a1", "pilot", 2.0, "completed")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0][field] = replacement
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        CudaCostLedger.load(path, cap_seconds=86_400.0)


def test_ledger_rejects_coordinated_elapsed_and_cumulative_tampering(tmp_path: Path) -> None:
    """Catches an arithmetic-consistent rewrite of a finalized charged interval."""
    path = tmp_path / "cuda-costs.jsonl"
    ledger = CudaCostLedger.load(path, cap_seconds=86_400.0)
    ledger.record("a1", "pilot", 2.0, "completed")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["elapsed_seconds"] = 7.0
    records[0]["cumulative_seconds"] = 7.0
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        CudaCostLedger.load(path, cap_seconds=86_400.0)


@pytest.mark.parametrize("integrity", [None, 17])
def test_ledger_fails_closed_for_missing_or_malformed_integrity(
    integrity: object, tmp_path: Path
) -> None:
    """Catches legacy or malformed durable records being accepted without integrity evidence."""
    path = tmp_path / "cuda-costs.jsonl"
    ledger = CudaCostLedger.load(path, cap_seconds=86_400.0)
    ledger.record("a1", "pilot", 2.0, "completed")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if integrity is None:
        records[0].pop("integrity", None)
    else:
        records[0]["integrity"] = integrity
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        CudaCostLedger.load(path, cap_seconds=86_400.0)


def test_ledger_rejects_recomputed_jsonl_chain_without_its_immutable_anchor(tmp_path: Path) -> None:
    """Catches a full unkeyed-chain recomputation after an on-disk rewrite."""
    path = tmp_path / "cuda-costs.jsonl"
    ledger = CudaCostLedger.load(path, cap_seconds=86_400.0)
    ledger.record("a1", "pilot", 2.0, "completed")
    ledger.record("a2", "frozen", 3.0, "failed_environment")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["stage"] = "forged-stage"
    previous = records[0]["previous_integrity"]
    for record in records:
        record["previous_integrity"] = previous
        payload = {
            "domain": record["format"],
            "kind": "entry",
            "index": record["index"],
            "previous_integrity": record["previous_integrity"],
            "attempt_id": record["attempt_id"],
            "stage": record["stage"],
            "elapsed_seconds": record["elapsed_seconds"],
            "outcome": record["outcome"],
            "cumulative_seconds": record["cumulative_seconds"],
        }
        record["integrity"] = sha256(
            (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        ).hexdigest()
        previous = record["integrity"]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="anchor"):
        CudaCostLedger.load(path, cap_seconds=86_400.0)


def test_ledger_fails_closed_when_an_immutable_entry_anchor_is_missing(tmp_path: Path) -> None:
    """Catches truncation of the external immutable-anchor set."""
    path = tmp_path / "cuda-costs.jsonl"
    ledger = CudaCostLedger.load(path, cap_seconds=86_400.0)
    ledger.record("a1", "pilot", 2.0, "completed")
    (tmp_path / "cuda-costs.jsonl.anchors" / "00000001.json").unlink()

    with pytest.raises(ValueError, match="anchor"):
        CudaCostLedger.load(path, cap_seconds=86_400.0)


def test_ledger_load_completes_anchor_published_before_append_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an anchor-only normal crash being mistaken for hostile corruption."""
    path = tmp_path / "cuda-costs.jsonl"
    ledger = CudaCostLedger.load(path, cap_seconds=86_400.0)

    def crash_after_anchor(path: Path, entry: object) -> None:
        raise RuntimeError("injected crash after anchor publication")

    monkeypatch.setattr(gates, "_append_ledger_record", crash_after_anchor, raising=False)
    with pytest.raises(RuntimeError, match="after anchor"):
        ledger.record("a1", "pilot", 2.0, "completed")
    monkeypatch.undo()

    recovered = CudaCostLedger.load(path, cap_seconds=86_400.0)
    assert [(entry.attempt_id, entry.index) for entry in recovered.entries] == [("a1", 1)]
    recovered.record("a2", "frozen", 3.0, "completed")
    assert [entry.attempt_id for entry in CudaCostLedger.load(path, 86_400.0).entries] == ["a1", "a2"]


def test_ledger_load_completes_durable_pending_marker_before_anchor_publication(tmp_path: Path) -> None:
    """Catches a crash after the durable commit claim losing already-measured CUDA time."""
    path = tmp_path / "cuda-costs.jsonl"
    entry = gates._new_cuda_cost_entry(
        (), attempt_id="a1", stage="pilot", elapsed_seconds=2.0, outcome="completed"
    )
    gates._write_pending_transaction(path, entry)

    recovered = CudaCostLedger.load(path, cap_seconds=86_400.0)
    assert [(item.attempt_id, item.index) for item in recovered.entries] == [("a1", 1)]


def test_ledger_load_recovers_post_append_crash_before_pending_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a durable appended entry being rejected solely for a stale commit marker."""
    path = tmp_path / "cuda-costs.jsonl"
    ledger = CudaCostLedger.load(path, cap_seconds=86_400.0)

    def crash_before_cleanup(path: Path) -> None:
        raise RuntimeError("injected crash after append")

    monkeypatch.setattr(gates, "_clear_pending_transaction", crash_before_cleanup, raising=False)
    with pytest.raises(RuntimeError, match="after append"):
        ledger.record("a1", "pilot", 2.0, "completed")
    monkeypatch.undo()

    recovered = CudaCostLedger.load(path, cap_seconds=86_400.0)
    assert [(entry.attempt_id, entry.index) for entry in recovered.entries] == [("a1", 1)]


def test_stale_ledger_owner_revalidates_before_claiming_duplicate_attempt(tmp_path: Path) -> None:
    """Catches two stale owners accepting the same attempt and corrupting the tail."""
    path = tmp_path / "cuda-costs.jsonl"
    first = CudaCostLedger.load(path, cap_seconds=86_400.0)
    stale = CudaCostLedger.load(path, cap_seconds=86_400.0)
    first.record("a1", "pilot", 2.0, "completed")

    with pytest.raises(ValueError, match="duplicate"):
        stale.record("a1", "pilot", 2.0, "completed")
    assert [(entry.attempt_id, entry.index) for entry in stale.entries] == [("a1", 1)]
    assert [(entry.attempt_id, entry.index) for entry in CudaCostLedger.load(path, 86_400.0).entries] == [("a1", 1)]


def _record_from_stale_process(
    path_text: str,
    attempt_id: str,
    ready: object,
    start: object,
    results: object,
) -> None:
    ledger = CudaCostLedger.load(Path(path_text), cap_seconds=86_400.0)
    ready.put(attempt_id)
    if not start.wait(15):
        results.put((attempt_id, "timeout"))
        return
    try:
        ledger.record(attempt_id, "pilot", 2.0, "completed")
    except Exception as error:  # pragma: no cover - assertion happens in parent process.
        results.put((attempt_id, type(error).__name__, str(error)))
    else:
        results.put((attempt_id, "ok"))


def test_concurrent_processes_serialize_index_allocation_and_append(tmp_path: Path) -> None:
    """Catches process-local locking that lets distinct stale writers collide at index one."""
    path = tmp_path / "cuda-costs.jsonl"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_record_from_stale_process, args=(str(path), attempt_id, ready, start, results))
        for attempt_id in ("a1", "a2")
    ]
    for process in processes:
        process.start()
    assert {ready.get(timeout=15), ready.get(timeout=15)} == {"a1", "a2"}
    start.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert sorted(results.get(timeout=15) for _ in processes) == [("a1", "ok"), ("a2", "ok")]
    entries = CudaCostLedger.load(path, 86_400.0).entries
    assert {entry.attempt_id for entry in entries} == {"a1", "a2"}
    assert [entry.index for entry in entries] == [1, 2]


def test_concurrent_processes_charge_one_duplicate_attempt_once(tmp_path: Path) -> None:
    """Catches a cross-process retry charging one attempt twice after stale loads."""
    path = tmp_path / "cuda-costs.jsonl"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_record_from_stale_process, args=(str(path), "a1", ready, start, results))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    ready.get(timeout=15)
    ready.get(timeout=15)
    start.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=15) for _ in processes]
    assert sum(outcome[1] == "ok" for outcome in outcomes) == 1
    assert sum(outcome[1] == "ValueError" and "duplicate" in outcome[2] for outcome in outcomes) == 1
    assert [(entry.attempt_id, entry.index) for entry in CudaCostLedger.load(path, 86_400.0).entries] == [("a1", 1)]


class _TimingModel(torch.nn.Module):
    def __init__(self, events: list[str], arm: str) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.events = events
        self.arm = arm

    def answer_loss(self, input_ids: torch.Tensor, labels: torch.Tensor, mode: str) -> torch.Tensor:
        del labels, mode
        self.events.append(f"{self.arm}:forward")
        loss = self.weight * input_ids.float().mean()
        loss.register_hook(lambda gradient: self.events.append(f"{self.arm}:backward"))
        return loss


class _RecordingSgd(torch.optim.SGD):
    def __init__(self, parameters: object, events: list[str], arm: str) -> None:
        super().__init__(parameters, lr=0.01)
        self.events = events
        self.arm = arm

    def step(self, closure: object = None) -> object:
        self.events.append(f"{self.arm}:step")
        return super().step(closure=closure)


def _timing_batch() -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[2]], dtype=torch.long),
        "labels": torch.tensor([[2]], dtype=torch.long),
        "mode": "single",
    }


def test_measure_training_step_excludes_warmups_and_brackets_full_update_with_sync() -> None:
    """Catches pilot samples that include warmups or omit a synchronized update boundary."""
    events: list[str] = []
    model = _TimingModel(events, "a1")
    optimizer = _RecordingSgd(model.parameters(), events, "a1")
    clock_values = iter(float(index) for index in range(50))

    samples = measure_training_step(
        model,
        _timing_batch(),
        optimizer,
        synchronizer=lambda: events.append("sync"),
        clock=lambda: next(clock_values),
        clipper=lambda parameters: (
            events.append("a1:clip"),
            torch.nn.utils.clip_grad_norm_(parameters, 1.0),
        )[1],
    )

    assert samples == (1.0,) * 20
    assert len([event for event in events if event == "sync"]) == 50
    assert model.weight.item() != pytest.approx(1.0)
    for offset in range(0, len(events), 6):
        assert events[offset : offset + 6] == [
            "sync",
            "a1:forward",
            "a1:backward",
            "a1:clip",
            "a1:step",
            "sync",
        ]


def test_measure_training_arms_alternates_a1_a2_sample_order() -> None:
    """Catches biased all-A1-then-all-A2 pilot measurement ordering."""
    events: list[str] = []
    a1 = _TimingModel(events, "a1")
    a2 = _TimingModel(events, "a2")
    clock_values = iter(float(index) for index in range(100))
    arms = {
        "a1": (a1, _timing_batch(), _RecordingSgd(a1.parameters(), events, "a1")),
        "a2": (a2, _timing_batch(), _RecordingSgd(a2.parameters(), events, "a2")),
    }

    result = measure_training_arms(
        arms,
        synchronizer=lambda: None,
        clock=lambda: next(clock_values),
    )

    assert {name: len(samples) for name, samples in result.items()} == {"a1": 20, "a2": 20}
    forwards = [event for event in events if event.endswith(":forward")]
    assert forwards[:10] == ["a1:forward"] * 5 + ["a2:forward"] * 5
    assert forwards[10:] == [item for _ in range(20) for item in ("a1:forward", "a2:forward")]


def test_a3_steps_uses_registered_ceiling_rule() -> None:
    """Catches floor/mean timing derivations that underfund the A3 time control."""
    assert derive_a3_steps([2.0] * 20, [2.5] * 20) == 1250
    assert derive_a3_steps([3.0] * 20, [2.0] * 20) == 1000


@pytest.mark.parametrize("samples", [[2.0] * 19, [2.0] * 19 + [0.0], [2.0] * 19 + [float("nan")]])
def test_a3_step_derivation_rejects_nonregistered_pilot_samples(samples: list[float]) -> None:
    """Catches a non-comparable pilot being used to freeze A3 training length."""
    with pytest.raises(ValueError):
        derive_a3_steps(samples, [2.0] * 20)


def _registered_resource_measurements(config: RecurrenceConfig, *, projected_remaining_seconds: float) -> dict[str, object]:
    total_vram = 10 * 1024**3
    return {
        "peak_reserved_bytes": total_vram - config.resources.reserved_vram_bytes,
        "total_vram_bytes": total_vram,
        "download_bytes": config.resources.maximum_download_bytes,
        "artifact_bytes": config.resources.maximum_artifact_bytes,
        "projected_remaining_seconds": projected_remaining_seconds,
    }


def test_resource_decision_uses_inclusive_registered_limits_and_pessimistic_cuda_projection(tmp_path: Path) -> None:
    """Catches a resource gate that rounds up to a pass after crossing a registered limit."""
    config = RecurrenceConfig.load(Path("configs/recurrence-v0.toml"))
    ledger = CudaCostLedger.load(tmp_path / "cuda-costs.jsonl", config.resources.maximum_cuda_seconds)
    ledger.record("spent", "pilot", 86_399.5, "completed")
    equal_measurements = _registered_resource_measurements(config, projected_remaining_seconds=1 / 3)

    passed = resource_decision(equal_measurements, ledger, config)
    assert passed["status"] == "passed"
    assert all(gate["passed"] for gate in passed["gates"].values())

    assert resource_decision(
        {**equal_measurements, "peak_reserved_bytes": equal_measurements["peak_reserved_bytes"] + 1}, ledger, config
    )["status"] == "stopped_budget"
    assert resource_decision(
        {**equal_measurements, "download_bytes": config.resources.maximum_download_bytes + 1}, ledger, config
    )["status"] == "stopped_budget"
    assert resource_decision(
        {**equal_measurements, "artifact_bytes": config.resources.maximum_artifact_bytes + 1}, ledger, config
    )["status"] == "stopped_budget"
    assert resource_decision(
        {**equal_measurements, "projected_remaining_seconds": 1.0}, ledger, config
    )["status"] == "stopped_budget"
