"""Synthetic qualification tests for the frozen V3 replacement controls."""

from __future__ import annotations

import hashlib
import json
import random
import os
import stat
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ppmoe.recurrence import audit_v2
from ppmoe.recurrence.provenance import canonical_json, sha256_payload


FAMILIES = ("expressions", "lists", "graphs")
ROWS_PER_FAMILY = 100


def _synthetic_source(*, graph_intermediate: bool = False) -> dict[str, object]:
    dev_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for index in range(ROWS_PER_FAMILY):
            identifier = f"{family}-{index:03d}"
            if family == "expressions":
                latent = {"leaves": [0, 1], "operators": ["+"], "shape": "chain"}
                prompt, response, parsed = "Compute 0 + 1.", "Final answer: 1", 1
            elif family == "lists":
                latent = {"input": [0, 1], "ops": ["ADD1"]}
                prompt, response, parsed = "Apply ADD1 to [0,1].", "Final list: [1,2]", [1, 2]
            else:
                latent = {"n": 2, "edges": [], "start": "A", "end": "B"}
                prompt, response, parsed = "Is there a path from A to B?", "no", "no"
                if graph_intermediate:
                    response = "Reasoning says no.\nAnswer: no"
            dev_rows.append({
                "id": identifier,
                "family": family,
                "split": "dev",
                "structure_id": f"synthetic-{identifier}",
                "prompt": prompt,
                "answer": str(parsed),
                "latent_spec": latent,
            })
            accepted = index >= 31
            score_rows.append({
                "example_id": identifier,
                "family": family,
                "completion_text": response,
                "stop_reason": "eos",
                "parsed_value": parsed if accepted else None,
                "correct": accepted,
                "reason_code": "correct" if accepted else "invalid_format",
            })
    return {
        "run_id": audit_v2.RUN_ID,
        "dev_rows": dev_rows,
        "score_rows": score_rows,
        "hashes": {"dev_split": "1" * 64, "dev_sidecar": "2" * 64, "dev_score": "3" * 64},
        "stage_hashes": {"prepare": "4" * 64, "calibrate": "5" * 64, "report": "6" * 64},
    }


def _write_original_audit(tmp_path: Path, source: dict[str, object]) -> Path:
    selected = audit_v2.select_v2_audit_rows(source)
    grouped = {family: [row for row in selected if row["family"] == family] for family in FAMILIES}
    packet_raw = {
        family: b"".join(canonical_json(row) for row in grouped[family])
        for family in FAMILIES
    }
    source_id_by_audit_id = {
        hashlib.sha256(f"v2-audit:{row['id']}".encode("utf-8")).hexdigest()[:16]: row["id"]
        for row in source["dev_rows"]
    }
    scores_by_id = {row["example_id"]: row for row in source["score_rows"]}
    source_ids = {
        row["audit_id"]: source_id_by_audit_id[row["audit_id"]]
        for family in FAMILIES for row in grouped[family]
    }
    selection_reasons = {
        audit_id: scores_by_id[source_id]["reason_code"]
        for audit_id, source_id in source_ids.items()
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": source["run_id"],
        "seed": audit_v2.DEFAULT_SEED,
        "source_hashes": source["hashes"],
        "stage_hashes": source["stage_hashes"],
        "audit_ids": {family: [row["audit_id"] for row in grouped[family]] for family in FAMILIES},
        "packet_sha256": {family: hashlib.sha256(packet_raw[family]).hexdigest() for family in FAMILIES},
        "source_ids": source_ids,
        "selection_reasons": selection_reasons,
    }
    manifest["audit_id"] = sha256_payload({
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "source_hashes": manifest["source_hashes"],
        "stage_hashes": manifest["stage_hashes"],
        "seed": manifest["seed"],
    })
    root = tmp_path / str(manifest["audit_id"])
    (root / "packets").mkdir(parents=True)
    for family in FAMILIES:
        (root / "packets" / f"{family}.jsonl").write_bytes(packet_raw[family])
    (root / "manifest.json").write_bytes(canonical_json(manifest))
    return root


def _write_reviews(
    controls_root: Path,
    packets_by_family: dict[str, list[dict[str, str]]],
    *,
    evidence_overrides: dict[str, str] | None = None,
) -> None:
    review_dir = controls_root / "reviews"
    review_dir.mkdir()
    all_packets = [packet for family in FAMILIES for packet in packets_by_family[family]]
    tasks = ("/root/v3_replacement_luna_1", "/root/v3_replacement_luna_2", "/root/v3_replacement_luna_3")
    names = ("alpha", "beta", "gamma")
    reviewers: list[dict[str, str]] = []
    for task, name, serial in zip(tasks, names, (1, 2, 3), strict=True):
        reviewer_id = f"gpt-6-luna:{task}"
        rows = []
        for packet in all_packets:
            value: object
            if packet["family"] == "expressions":
                value = 1
            elif packet["family"] == "lists":
                value = [1, 2]
            else:
                value = "no"
            rows.append({
                "audit_id": packet["audit_id"],
                "family": packet["family"],
                "status": "conclusive",
                "claimed_value": value,
                "evidence_span": (evidence_overrides or {}).get(packet["family"], packet["response"]),
                "reason": "One explicit final answer is present.",
                "reviewer_id": reviewer_id,
                "explanation_status": "indeterminate" if packet["family"] == "graphs" else "not_applicable",
                "explanation_note": "The response states no path without an explanation." if packet["family"] == "graphs" else "",
            })
        review_path = review_dir / f"{name}.jsonl"
        review_raw = b"".join(canonical_json(row) for row in rows)
        review_path.write_bytes(review_raw)
        reviewers.append({
            "task": task,
            "model": "gpt-6-luna",
            "root_attested_session_id": f"00000000-0000-4000-8000-{serial:012d}",
            "review_file": f"reviews/{name}.jsonl",
            "review_sha256": hashlib.sha256(review_raw).hexdigest(),
            "reviewer_id": reviewer_id,
        })
    (controls_root / "review-sessions.json").write_bytes(canonical_json({"schema_version": 1, "reviewers": reviewers}))


def _packets_by_family(root: Path) -> dict[str, list[dict[str, str]]]:
    return {
        family: [json.loads(line) for line in (root / f"{family}.jsonl").read_bytes().splitlines()]
        for family in FAMILIES
    }


def test_selector_uses_one_python_3119_random_stream_and_excludes_original_accepted_ids(tmp_path: Path) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    original_raw = (original_root / "manifest.json").read_bytes()
    original = audit_v2._parse_object(original_raw, "synthetic original manifest")
    audit_v2._authenticate_audit_export(source, original_root, original)

    selected = audit_v3_controls._select_from_authenticated_source(source, original)

    assert selected["python_version"] == "3.11.9"
    assert selected["seed"] == 927002
    assert [len(selected["selected_ids"][family]) for family in FAMILIES] == [10, 10, 10]
    flat_ids = [identifier for family in FAMILIES for identifier in selected["selected_ids"][family]]
    assert len(set(flat_ids)) == 30
    old_accepted = {
        original["source_ids"][audit_id]
        for audit_id, reason in original["selection_reasons"].items()
        if reason in {"correct", "wrong_value"}
    }
    assert set(flat_ids).isdisjoint(old_accepted)
    for family in FAMILIES:
        pool = sorted(
            row["id"] for row, score in zip(source["dev_rows"], source["score_rows"], strict=True)
            if row["family"] == family and score["reason_code"] in {"correct", "wrong_value"}
            and row["id"] not in old_accepted
        )
        assert selected["eligible_count"][family] == len(pool)
        assert selected["eligible_ids_sha256"][family] == hashlib.sha256(canonical_json(pool)).hexdigest()
    rng = random.Random(927002)
    expected = {
        family: [row["id"] for row in rng.sample(
            sorted(
                (row for row, score in zip(source["dev_rows"], source["score_rows"], strict=True)
                 if row["family"] == family and row["id"] not in old_accepted
                 and score["reason_code"] in {"correct", "wrong_value"}),
                key=lambda row: row["id"],
            ),
            10,
        )]
        for family in FAMILIES
    }
    assert selected["selected_ids"] == expected
    assert len(selected["selector_sha256"]) == 64


def test_export_writes_gold_free_packets_and_private_selection_manifest_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    original_reader = audit_v2._read_audit_file
    source_paths: list[Path] = []

    def reject_sealed_reads(root, relative, description):
        source_paths.append(Path(relative))
        assert "sealed" not in Path(relative).as_posix().casefold()
        return original_reader(root, relative, description)

    monkeypatch.setattr(audit_v2, "_read_audit_file", reject_sealed_reads)
    destination = audit_v3_controls.export_replacement_controls(
        run_root, original_root, tmp_path / "replacement-controls",
    )
    controls_root = Path(destination["controls_root"])
    reviewer_root = Path(destination["reviewer_packet_root"])
    private_manifest = json.loads((controls_root / "private-manifest.json").read_bytes())

    assert private_manifest["python_version"] == "3.11.9"
    assert private_manifest["incident"] == "original_30_accepted_controls_retired_after_blinding_boundary_breach"
    original_manifest = json.loads((original_root / "manifest.json").read_bytes())
    assert private_manifest["original_manifest_sha256"] == hashlib.sha256((original_root / "manifest.json").read_bytes()).hexdigest()
    assert private_manifest["original_packet_sha256"] == original_manifest["packet_sha256"]
    frozen_paths = set(private_manifest["frozen_code_sha256"])
    assert frozen_paths == {
        path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
        for path in (Path(__file__).resolve().parents[1] / "src/ppmoe/recurrence").glob("*.py")
    } | {
        "pyproject.toml", "uv.lock", "tests/test_recurrence_audit_v3_controls.py",
        "tests/test_recurrence_answer_v3.py",
    }
    assert {path.parts[0] for path in source_paths} == {"manifest.json", "packets"}
    assert [len(private_manifest["selected_ids"][family]) for family in FAMILIES] == [10, 10, 10]
    assert not reviewer_root.resolve().is_relative_to(controls_root.resolve())
    assert not controls_root.resolve().is_relative_to(reviewer_root.resolve())
    assert not (reviewer_root / "private-manifest.json").exists()
    assert set(path.name for path in reviewer_root.iterdir()) == {f"{family}.jsonl" for family in FAMILIES}
    assert reviewer_root.resolve() != controls_root.resolve()
    for family in FAMILIES:
        packets = [
            json.loads(line)
            for line in (reviewer_root / f"{family}.jsonl").read_bytes().splitlines()
        ]
        assert len(packets) == 10
        assert all(set(packet) == {"audit_id", "family", "question", "response", "stop_reason"} for packet in packets)
        assert all("latent_spec" not in packet and "parsed_value" not in packet for packet in packets)
    with pytest.raises(FileExistsError):
        audit_v3_controls.export_replacement_controls(
            run_root, original_root, tmp_path / "replacement-controls",
        )


def test_export_rechecks_reparse_and_containment_at_creation_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    output_root = tmp_path / "safe-output"
    output_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    original_lstat = os.lstat

    def pretend_reparse(path):
        if Path(path) == output_root:
            return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0)
        return original_lstat(path)

    monkeypatch.setattr(audit_v3_controls.study_v2.os, "lstat", pretend_reparse)
    with pytest.raises(ValueError, match="reparse|symlink"):
        audit_v3_controls.export_replacement_controls(run_root, original_root, output_root)


def test_post_export_scorer_hash_drift_stops_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    packets = _packets_by_family(Path(result["reviewer_packet_root"]))
    _write_reviews(controls_root, packets)
    expected = audit_v3_controls._frozen_code_hashes()
    scorer_path = "src/ppmoe/recurrence/answer_v3.py"
    drifted = {**expected, scorer_path: "0" * 64}
    monkeypatch.setattr(audit_v3_controls, "_frozen_code_hashes", lambda _captured=None: drifted)
    calls: list[str] = []
    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", lambda example, *_: calls.append(example.id))

    with pytest.raises(ValueError, match="code|hash|drift"):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)

    assert calls == []
    assert not (controls_root / "attempt.json").exists()
    assert json.loads((controls_root / "qualification.json").read_bytes())["status"] == "stopped"


def test_code_hash_drift_after_scoring_locks_stop_with_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    _write_reviews(controls_root, _packets_by_family(Path(result["reviewer_packet_root"])))
    expected = audit_v3_controls._frozen_code_hashes()
    scorer_path = "src/ppmoe/recurrence/answer_v3.py"
    mutated = {**expected, scorer_path: "0" * 64}
    snapshots = iter((expected, expected, mutated))
    monkeypatch.setattr(audit_v3_controls, "_frozen_code_hashes", lambda _captured=None: next(snapshots))
    calls: list[str] = []
    real_scorer = audit_v3_controls.answer_v3.score_completion_v3

    def score(example, *args):
        calls.append(example.id)
        return real_scorer(example, *args)

    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", score)
    with pytest.raises(ValueError, match="changed|drift|hash"):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)
    assert len(calls) == 30
    assert (controls_root / "attempt.json").is_file()
    assert json.loads((controls_root / "qualification.json").read_bytes())["status"] == "stopped"


def test_transitive_helper_hash_is_rechecked_after_scoring_without_reenumerating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    _write_reviews(controls_root, _packets_by_family(Path(result["reviewer_packet_root"])))
    manifest = json.loads((controls_root / "private-manifest.json").read_bytes())
    captured = manifest["frozen_code_sha256"]
    helper_path = "src/ppmoe/recurrence/study_v2.py"
    assert helper_path in captured

    changed = False
    calls: list[str] = []
    real_hash_file = audit_v3_controls._sha256_file
    real_scorer = audit_v3_controls.answer_v3.score_completion_v3

    def hash_file(path: Path) -> str:
        if changed and Path(path).resolve() == Path("src/ppmoe/recurrence/study_v2.py").resolve():
            return "0" * 64
        return real_hash_file(path)

    def score(example, *args):
        nonlocal changed
        calls.append(example.id)
        changed = True
        return real_scorer(example, *args)

    monkeypatch.setattr(audit_v3_controls, "_sha256_file", hash_file)
    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", score)
    with pytest.raises(ValueError, match="changed|drift|hash"):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)

    assert len(calls) == 30
    assert (controls_root / "attempt.json").is_file()
    assert json.loads((controls_root / "qualification.json").read_bytes())["status"] == "stopped"


def test_review_session_manifest_mutation_during_scoring_stops_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    _write_reviews(controls_root, _packets_by_family(Path(result["reviewer_packet_root"])))
    session_path = controls_root / "review-sessions.json"
    calls: list[str] = []
    real_scorer = audit_v3_controls.answer_v3.score_completion_v3

    def score(example, *args):
        calls.append(example.id)
        if len(calls) == 1:
            session_path.write_bytes(session_path.read_bytes() + b" ")
        return real_scorer(example, *args)

    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", score)
    with pytest.raises(ValueError, match="session|manifest|changed|hash"):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)

    assert len(calls) == 30
    assert (controls_root / "attempt.json").is_file()
    qualification = json.loads((controls_root / "qualification.json").read_bytes())
    assert qualification["status"] == "stopped"


def test_public_reconciliation_race_loser_does_not_write_stop_over_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    _write_reviews(controls_root, _packets_by_family(Path(result["reviewer_packet_root"])))

    barrier = __import__("threading").Barrier(2)
    real_reserve = audit_v3_controls._reserve_reconciliation_attempt
    real_write_stop = audit_v3_controls._write_stopped_qualification
    stopped_errors: list[str] = []

    def synchronized_reserve(root, payload):
        barrier.wait()
        return real_reserve(root, payload)

    def track_stopped(root, error):
        stopped_errors.append(str(error))
        return real_write_stop(root, error)

    monkeypatch.setattr(audit_v3_controls, "_reserve_reconciliation_attempt", synchronized_reserve)
    monkeypatch.setattr(audit_v3_controls, "_write_stopped_qualification", track_stopped)

    def reconcile() -> str:
        try:
            return str(audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)["status"])
        except FileExistsError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reconcile(), range(2)))

    assert sorted(results) == ["qualified", "refused"]
    assert stopped_errors == []
    assert json.loads((controls_root / "qualification.json").read_bytes())["status"] == "qualified"


def test_public_race_loser_paused_before_inner_guard_does_not_write_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    _write_reviews(controls_root, _packets_by_family(Path(result["reviewer_packet_root"])))

    threading = __import__("threading")
    both_past_outer_guard = threading.Barrier(2)
    owner_reserved = threading.Event()
    loser_finished = threading.Event()
    assignment_lock = threading.Lock()
    owner_assigned = False
    real_run = audit_v3_controls._run_reconciliation_once
    real_reserve = audit_v3_controls._reserve_reconciliation_attempt
    real_scorer = audit_v3_controls.answer_v3.score_completion_v3
    real_write_stop = audit_v3_controls._write_stopped_qualification
    stopped_errors: list[str] = []

    def coordinated_run(run, original, controls):
        nonlocal owner_assigned
        both_past_outer_guard.wait()
        with assignment_lock:
            is_owner = not owner_assigned
            owner_assigned = True
        if is_owner:
            return real_run(run, original, controls)
        assert owner_reserved.wait(timeout=10)
        return real_run(run, original, controls)

    def signal_reservation(root, payload):
        raw = real_reserve(root, payload)
        owner_reserved.set()
        return raw

    def hold_owner_in_scorer(example, *args):
        assert loser_finished.wait(timeout=10)
        return real_scorer(example, *args)

    def track_stopped(root, error):
        stopped_errors.append(str(error))
        return real_write_stop(root, error)

    monkeypatch.setattr(audit_v3_controls, "_run_reconciliation_once", coordinated_run)
    monkeypatch.setattr(audit_v3_controls, "_reserve_reconciliation_attempt", signal_reservation)
    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", hold_owner_in_scorer)
    monkeypatch.setattr(audit_v3_controls, "_write_stopped_qualification", track_stopped)

    def reconcile() -> str:
        try:
            return str(audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)["status"])
        except FileExistsError:
            loser_finished.set()
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reconcile(), range(2)))

    assert sorted(results) == ["qualified", "refused"]
    assert stopped_errors == []
    assert json.loads((controls_root / "qualification.json").read_bytes())["status"] == "qualified"


@pytest.mark.parametrize("family", ["expressions", "lists", "graphs"])
def test_intermediate_only_evidence_is_rejected_before_full_response_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source(graph_intermediate=family == "graphs")
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    packets = _packets_by_family(Path(result["reviewer_packet_root"]))
    evidence = {
        "expressions": "1",
        "lists": "[1,2]",
        "graphs": "Reasoning says no.",
    }
    _write_reviews(controls_root, packets, evidence_overrides={family: evidence[family]})
    calls: list[str] = []
    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", lambda example, *_: calls.append(example.id))

    with pytest.raises(ValueError, match="evidence|candidate|final"):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)

    assert calls == []
    assert (controls_root / "attempt.json").is_file()
    assert json.loads((controls_root / "qualification.json").read_bytes())["status"] == "stopped"


def test_session_manifest_is_casefolded_but_review_identity_must_match_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    packets = _packets_by_family(Path(result["reviewer_packet_root"]))
    _write_reviews(controls_root, packets)
    sessions_path = controls_root / "review-sessions.json"
    sessions = json.loads(sessions_path.read_bytes())
    sessions["reviewers"][0]["task"] = sessions["reviewers"][0]["task"].upper()
    sessions_path.write_bytes(canonical_json(sessions))

    qualified = audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)

    assert qualified["status"] == "qualified"


def test_session_identity_spoof_is_terminal_before_any_scorer_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    packets = _packets_by_family(Path(result["reviewer_packet_root"]))
    _write_reviews(controls_root, packets)
    sessions_path = controls_root / "review-sessions.json"
    sessions = json.loads(sessions_path.read_bytes())
    sessions["reviewers"][0]["reviewer_id"] = "gpt-6-luna:spoofed"
    sessions_path.write_bytes(canonical_json(sessions))
    calls: list[str] = []
    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", lambda example, *_: calls.append(example.id))

    with pytest.raises(ValueError, match="reviewer|identity|session"):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)

    assert calls == []
    assert not (controls_root / "attempt.json").exists()
    assert json.loads((controls_root / "qualification.json").read_bytes())["status"] == "stopped"


def test_reconciliation_requires_three_distinct_complete_reviewers_and_scores_each_row_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    destination = audit_v3_controls.export_replacement_controls(
        run_root, original_root, tmp_path / "replacement-controls",
    )
    controls_root = Path(destination["controls_root"])
    reviewer_root = Path(destination["reviewer_packet_root"])
    packets_by_family = _packets_by_family(reviewer_root)
    _write_reviews(controls_root, packets_by_family)
    calls: list[str] = []
    real_scorer = audit_v3_controls.answer_v3.score_completion_v3

    def score(example, completion_text, stop_reason):
        calls.append(example.id)
        assert (controls_root / "attempt.json").is_file()
        return real_scorer(example, completion_text, stop_reason)

    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", score)
    result = audit_v3_controls.reconcile_replacement_controls(
        run_root, original_root, controls_root,
    )

    assert result["status"] == "qualified"
    assert result["row_count"] == 30
    assert len(calls) == 30 and len(set(calls)) == 30
    qualification = json.loads((controls_root / "qualification.json").read_bytes())
    assert qualification["reviewer_ids"] == [
        "gpt-6-luna:/root/v3_replacement_luna_1",
        "gpt-6-luna:/root/v3_replacement_luna_2",
        "gpt-6-luna:/root/v3_replacement_luna_3",
    ]
    assert qualification["review_session_provenance"] == "process provenance; not cryptographic attestation"
    assert qualification["attempt_sha256"] == hashlib.sha256((controls_root / "attempt.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)


def test_attempt_reservation_is_exclusive_under_concurrent_claims(tmp_path: Path) -> None:
    from ppmoe.recurrence import audit_v3_controls

    controls_root = tmp_path / "synthetic-controls"
    controls_root.mkdir()
    barrier = __import__("threading").Barrier(2)

    def reserve() -> str:
        barrier.wait()
        try:
            audit_v3_controls._reserve_reconciliation_attempt(controls_root, {"test": "synthetic"})
            return "reserved"
        except FileExistsError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reserve(), range(2)))

    assert sorted(results) == ["refused", "reserved"]
    attempt = json.loads((controls_root / "attempt.json").read_bytes())
    assert attempt["status"] == "reserved"
    assert attempt["manifest_sha256"] == sha256_payload({key: value for key, value in attempt.items() if key != "manifest_sha256"})


def test_crash_after_reservation_leaves_terminal_attempt_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    _write_reviews(controls_root, _packets_by_family(Path(result["reviewer_packet_root"])))
    calls: list[str] = []

    def simulated_crash(example, *_args):
        calls.append(example.id)
        assert (controls_root / "attempt.json").is_file()
        raise KeyboardInterrupt("synthetic process interruption")

    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", simulated_crash)
    with pytest.raises(KeyboardInterrupt):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)
    assert len(calls) == 1
    assert (controls_root / "attempt.json").is_file()
    assert not (controls_root / "qualification.json").exists()
    with pytest.raises(FileExistsError):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)
    assert len(calls) == 1


def test_attempt_blocks_retry_if_final_qualification_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    result = audit_v3_controls.export_replacement_controls(run_root, original_root, tmp_path / "replacement-controls")
    controls_root = Path(result["controls_root"])
    _write_reviews(controls_root, _packets_by_family(Path(result["reviewer_packet_root"])))
    calls: list[str] = []
    real_scorer = audit_v3_controls.answer_v3.score_completion_v3
    real_write = audit_v3_controls._write_exclusive

    def count_score(example, *args):
        calls.append(example.id)
        return real_scorer(example, *args)

    def fail_qualification(path, raw, **kwargs):
        if Path(path).name == "qualification.json":
            raise OSError("synthetic final record failure")
        return real_write(path, raw, **kwargs)

    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", count_score)
    monkeypatch.setattr(audit_v3_controls, "_write_exclusive", fail_qualification)
    with pytest.raises(OSError, match="final record failure"):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)
    assert len(calls) == 30
    assert (controls_root / "attempt.json").is_file()
    assert not (controls_root / "qualification.json").exists()
    with pytest.raises(FileExistsError):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)
    assert len(calls) == 30


@pytest.mark.parametrize("failure", ["reviewer_duplicate", "review_reordered", "ambiguous", "value_conflict", "evidence_missing", "packet_tamper"])
def test_reconciliation_stops_before_scorer_or_record_on_review_or_packet_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    destination = audit_v3_controls.export_replacement_controls(
        run_root, original_root, tmp_path / "replacement-controls",
    )
    controls_root = Path(destination["controls_root"])
    reviewer_root = Path(destination["reviewer_packet_root"])
    packets_by_family = _packets_by_family(reviewer_root)
    _write_reviews(controls_root, packets_by_family)
    review_path = controls_root / "reviews" / "alpha.jsonl"
    rows = [json.loads(line) for line in review_path.read_bytes().splitlines()]
    if failure == "reviewer_duplicate":
        for row in rows:
            row["reviewer_id"] = "gpt-6-luna:/root/v3_replacement_luna_2"
    elif failure == "review_reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif failure == "ambiguous":
        rows[0]["status"] = "ambiguous"
        rows[0]["claimed_value"] = None
    elif failure == "value_conflict":
        rows[0]["claimed_value"] = 3
    elif failure == "evidence_missing":
        rows[0]["evidence_span"] = "not in response"
    elif failure == "packet_tamper":
        packet_path = reviewer_root / "lists.jsonl"
        packet_path.write_bytes(packet_path.read_bytes() + b" ")
    review_path.write_bytes(b"".join(canonical_json(row) for row in rows))
    calls: list[str] = []
    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", lambda *args: calls.append("called"))

    with pytest.raises(ValueError):
        audit_v3_controls.reconcile_replacement_controls(
            run_root, original_root, controls_root,
        )

    assert calls == []
    assert json.loads((controls_root / "qualification.json").read_bytes())["status"] == "stopped"


@pytest.mark.parametrize("tamper", ["audit_id", "source_hash", "stage_hash", "packet_hash", "packet_count"])
def test_export_authenticates_original_manifest_before_selecting_or_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    manifest_path = original_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if tamper == "audit_id":
        manifest["audit_id"] = "0" * 64
    elif tamper == "source_hash":
        manifest["source_hashes"]["dev_split"] = "9" * 64
    elif tamper == "stage_hash":
        manifest["stage_hashes"]["prepare"] = "9" * 64
    elif tamper == "packet_hash":
        manifest["packet_sha256"]["expressions"] = "9" * 64
    elif tamper == "packet_count":
        manifest["audit_ids"]["expressions"].pop()
    if tamper in {"source_hash", "stage_hash"}:
        manifest["audit_id"] = sha256_payload({
            "schema_version": 1,
            "run_id": manifest["run_id"],
            "source_hashes": manifest["source_hashes"],
            "stage_hashes": manifest["stage_hashes"],
            "seed": manifest["seed"],
        })
        new_root = tmp_path / str(manifest["audit_id"])
        original_root.rename(new_root)
        original_root = new_root
        manifest_path = original_root / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    output_root = tmp_path / "should-not-exist"
    selector_calls: list[bool] = []
    monkeypatch.setattr(audit_v3_controls, "_select_from_authenticated_source", lambda *_args: selector_calls.append(True))

    with pytest.raises(ValueError, match="identity|audit ID|hash|differs|count"):
        audit_v3_controls.export_replacement_controls(
            run_root, original_root, output_root,
        )

    assert not output_root.exists()
    assert selector_calls == []


def test_selector_rejects_duplicate_ids_and_insufficient_family_pool(tmp_path: Path) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    original = json.loads((original_root / "manifest.json").read_bytes())
    audit_v2._authenticate_audit_export(source, original_root, original)
    duplicate_source = {**source, "dev_rows": list(source["dev_rows"])}
    duplicate_source["dev_rows"][-1] = dict(duplicate_source["dev_rows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        audit_v3_controls._select_from_authenticated_source(duplicate_source, original)

    sparse_scores = [dict(row) for row in source["score_rows"]]
    for score in sparse_scores:
        if score["family"] == "lists":
            score["reason_code"] = "invalid_format"
            score["parsed_value"] = None
            score["correct"] = False
    sparse_source = {**source, "score_rows": sparse_scores}
    with pytest.raises(ValueError, match="fewer than ten eligible"):
        audit_v3_controls._select_from_authenticated_source(sparse_source, original)


def test_v3_disagreement_finishes_single_pass_then_locks_terminal_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ppmoe.recurrence import audit_v3_controls

    source = _synthetic_source()
    original_root = _write_original_audit(tmp_path, source)
    run_root = tmp_path / "synthetic-v2-run"
    run_root.mkdir()
    monkeypatch.setattr(audit_v2, "verify_v2_audit_source", lambda _root: source)
    destination = audit_v3_controls.export_replacement_controls(
        run_root, original_root, tmp_path / "replacement-controls",
    )
    controls_root = Path(destination["controls_root"])
    reviewer_root = Path(destination["reviewer_packet_root"])
    packets_by_family = _packets_by_family(reviewer_root)
    _write_reviews(controls_root, packets_by_family)
    calls: list[str] = []

    def wrong_scorer(example, completion_text, stop_reason):
        calls.append(example.id)
        return {
            "parsed_value": 2 if example.family == "expressions" else [1, 2] if example.family == "lists" else "no",
            "correct": False if example.family == "expressions" else True,
            "reason_code": "wrong_value" if example.family == "expressions" else "correct",
        }

    monkeypatch.setattr(audit_v3_controls.answer_v3, "score_completion_v3", wrong_scorer)
    result = audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)
    assert len(calls) == 30 and len(set(calls)) == 30
    qualification = json.loads((controls_root / "qualification.json").read_bytes())
    assert result["status"] == qualification["status"] == "stopped"
    assert qualification["row_count"] == len(qualification["rows"]) == 30
    assert set(qualification["review_sha256"]) == {
        "reviews/alpha.jsonl", "reviews/beta.jsonl", "reviews/gamma.jsonl",
    }
    assert set(qualification["packet_sha256"]) == set(FAMILIES)
    with pytest.raises(FileExistsError):
        audit_v3_controls.reconcile_replacement_controls(run_root, original_root, controls_root)
