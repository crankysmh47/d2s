from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ppmoe.recurrence.provenance import canonical_json, sha256_payload


RUN_ID = "18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d"
FAMILIES = ("expressions", "lists", "graphs")
TRUSTED_STAGE_HASHES = {
    "prepare": "b1bd10a2d019907fdf222d092879b9c5cdad68861072915407632d6eeaec4a46",
    "calibrate": "da11072965258f088322ecc567981078adf0895ba8f1a947a076b18acd1fae7d",
    "report": "0799e0c3d34bacac86cd0c942693a29d18bd502ec6d83713feebf975ef0fcbcf",
}
EXPECTED_ACCEPTED_IDS = [
    "expressions-068", "expressions-033", "expressions-071", "expressions-065", "expressions-056",
    "expressions-082", "expressions-098", "expressions-094", "expressions-054", "expressions-091",
    "lists-050", "lists-042", "lists-033", "lists-066", "lists-069", "lists-079", "lists-083",
    "lists-096", "lists-034", "lists-054",
    "graphs-069", "graphs-066", "graphs-038", "graphs-092", "graphs-056", "graphs-076",
    "graphs-042", "graphs-098", "graphs-071", "graphs-095",
]


def _record(stage: str, status: str, payload: dict[str, object], prerequisites: dict[str, str]) -> dict[str, object]:
    unsigned = {
        "schema_version": 1,
        "stage": stage,
        "status": status,
        "prerequisite_stage_sha256": prerequisites,
        "payload": payload,
    }
    return {**unsigned, "stage_sha256": sha256_payload(unsigned)}


@pytest.fixture
def audit_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[dict[str, object]], list[dict[str, object]]]:
    root = tmp_path / RUN_ID
    for folder in ("data", "scores", "stages"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    dev_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for index in range(100):
            example_id = f"{family}-{index:03d}"
            dev_rows.append({
                "id": example_id,
                "family": family,
                "split": "dev",
                "prompt": f"Question for {example_id}?",
                "answer": index,
                "latent_spec": {"answer": index},
                "structure_id": f"structure-{example_id}",
            })
            reason = "invalid_format" if index < 31 else ("wrong_value" if index < 61 else "correct")
            score_rows.append({
                "example_id": example_id,
                "family": family,
                "completion_text": f"Final answer: {index}",
                "stop_reason": "eos",
                "reason_code": reason,
                "correct": reason == "correct",
                "parsed_value": index,
            })
    split_raw = b"".join(canonical_json(row) for row in dev_rows)
    score_raw = b"".join(canonical_json(row) for row in score_rows)
    ordered_ids = [str(row["id"]) for row in dev_rows]
    split_hash = hashlib.sha256(split_raw).hexdigest()
    score_hash = hashlib.sha256(score_raw).hexdigest()
    sidecar_raw = canonical_json({
        "split": "dev", "sha256": split_hash, "count": 300, "ordered_ids": ordered_ids,
    })
    (root / "data/dev.jsonl").write_bytes(split_raw)
    (root / "data/dev.jsonl.manifest.json").write_bytes(sidecar_raw)
    (root / "scores/dev-f0.jsonl").write_bytes(score_raw)
    (root / "data/sealed.jsonl").write_bytes(b"poisoned sealed source; must never be read")

    prepare = _record("prepare", "passed", {"run_id": RUN_ID}, {})
    calibration = _record("calibrate", "stopped_scientific", {
        "run_id": RUN_ID,
        "next_stage": "report",
        "pilot_scores": {
            "dev-f0": {
                "split_path": "data/dev.jsonl",
                "split_manifest_path": "data/dev.jsonl.manifest.json",
                "score_path": "scores/dev-f0.jsonl",
                "split_sha256": split_hash,
                "split_manifest_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
                "score_sha256": score_hash,
                "split_count": 300,
                "score_count": 300,
            },
        },
    }, {"prepare": str(prepare["stage_sha256"])})
    report = _record("report", "stopped_scientific", {
        "run_id": RUN_ID, "status": "stopped_scientific", "next_stage": "complete",
        "prepare_stage_sha256": str(prepare["stage_sha256"]),
        "calibration_stage_sha256": str(calibration["stage_sha256"]),
        "calibration": calibration["payload"],
    }, {
        "prepare": str(prepare["stage_sha256"]),
        "calibrate": str(calibration["stage_sha256"]),
    })
    for stage, record in (("prepare", prepare), ("calibrate", calibration), ("report", report)):
        (root / f"stages/{stage}.json").write_bytes(canonical_json(record))
    monkeypatch.setattr(_api(), "_TRUSTED_STAGE_HASHES", {
        "prepare": str(prepare["stage_sha256"]),
        "calibrate": str(calibration["stage_sha256"]),
        "report": str(report["stage_sha256"]),
    }, raising=False)
    return root, dev_rows, score_rows


def _api():
    from ppmoe.recurrence import audit_v2

    return audit_v2


def _trust_fixture_stage_bytes(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hashes = {
        stage: json.loads((root / f"stages/{stage}.json").read_bytes())["stage_sha256"]
        for stage in ("prepare", "calibrate", "report")
    }
    monkeypatch.setattr(_api(), "_TRUSTED_STAGE_HASHES", hashes)


def test_live_v2_run_matches_the_registered_stage_hash_anchors() -> None:
    root = Path.cwd() / "artifacts/recurrence-v2" / RUN_ID
    source = _api().verify_v2_audit_source(root)
    assert source["stage_hashes"] == TRUSTED_STAGE_HASHES


def test_source_verification_authenticates_nonsealed_evidence_and_never_reads_sealed(
    audit_run: tuple[Path, list[dict[str, object]], list[dict[str, object]]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, _ = audit_run
    sealed = root / "data/sealed.jsonl"
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*") if path.is_file() and path.name != "sealed.jsonl"
    }
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name == "sealed.jsonl":
            raise AssertionError("sealed V2 data was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    source = _api().verify_v2_audit_source(root)
    assert source["run_id"] == RUN_ID
    assert len(source["dev_rows"]) == 300
    assert len(source["score_rows"]) == 300
    assert source["hashes"] == {
        "dev_split": hashlib.sha256((root / "data/dev.jsonl").read_bytes()).hexdigest(),
        "dev_sidecar": hashlib.sha256((root / "data/dev.jsonl.manifest.json").read_bytes()).hexdigest(),
        "dev_score": hashlib.sha256((root / "scores/dev-f0.jsonl").read_bytes()).hexdigest(),
    }
    assert source["dev_rows"][0]["id"] == "expressions-000"
    assert source["score_rows"][-1]["example_id"] == "graphs-099"
    assert sealed.exists()
    assert before == {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*") if path.is_file() and path.name != "sealed.jsonl"
    }


def test_source_verification_rejects_stage_hash_drift(audit_run) -> None:
    root, _, _ = audit_run
    calibration_path = root / "stages/calibrate.json"
    calibration = json.loads(calibration_path.read_bytes())
    calibration["payload"]["next_stage"] = "complete"
    calibration_path.write_bytes(canonical_json(calibration))
    with pytest.raises(ValueError, match="calibrate stage hash"):
        _api().verify_v2_audit_source(root)


def test_source_verification_rejects_report_prerequisite_drift(audit_run, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _, _ = audit_run
    report_path = root / "stages/report.json"
    report = json.loads(report_path.read_bytes())
    report["prerequisite_stage_sha256"]["calibrate"] = "0" * 64
    report_path.write_bytes(canonical_json(_record(
        "report", report["status"], report["payload"], report["prerequisite_stage_sha256"],
    )))
    _trust_fixture_stage_bytes(root, monkeypatch)
    with pytest.raises(ValueError, match="report prerequisite"):
        _api().verify_v2_audit_source(root)


def test_coordinated_dev_and_stage_rewrite_with_same_run_id_fails_registered_anchors(audit_run) -> None:
    root, dev_rows, _ = audit_run
    dev_rows[0]["prompt"] = "Rewritten prompt with unchanged registered run ID"
    split_raw = b"".join(canonical_json(row) for row in dev_rows)
    split_path = root / "data/dev.jsonl"
    split_path.write_bytes(split_raw)
    sidecar_path = root / "data/dev.jsonl.manifest.json"
    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar["sha256"] = hashlib.sha256(split_raw).hexdigest()
    sidecar_raw = canonical_json(sidecar)
    sidecar_path.write_bytes(sidecar_raw)

    prepare_path = root / "stages/prepare.json"
    prepare = json.loads(prepare_path.read_bytes())
    prepare_payload = prepare["payload"]
    prepare_payload["rewrite_marker"] = "same run ID, newly signed prepare"
    prepare = _record("prepare", prepare["status"], prepare_payload, {})
    prepare_path.write_bytes(canonical_json(prepare))

    calibrate_path = root / "stages/calibrate.json"
    calibrate = json.loads(calibrate_path.read_bytes())
    calibrate_payload = calibrate["payload"]
    dev_evidence = calibrate_payload["pilot_scores"]["dev-f0"]
    dev_evidence["split_sha256"] = hashlib.sha256(split_raw).hexdigest()
    dev_evidence["split_manifest_sha256"] = hashlib.sha256(sidecar_raw).hexdigest()
    calibrate = _record("calibrate", calibrate["status"], calibrate_payload, {
        "prepare": str(prepare["stage_sha256"]),
    })
    calibrate_path.write_bytes(canonical_json(calibrate))

    report_path = root / "stages/report.json"
    report = json.loads(report_path.read_bytes())
    report_payload = report["payload"]
    report_payload["prepare_stage_sha256"] = str(prepare["stage_sha256"])
    report_payload["calibration_stage_sha256"] = str(calibrate["stage_sha256"])
    report_payload["calibration"] = calibrate_payload
    report = _record("report", report["status"], report_payload, {
        "prepare": str(prepare["stage_sha256"]),
        "calibrate": str(calibrate["stage_sha256"]),
    })
    report_path.write_bytes(canonical_json(report))

    signed_prepare = _api().study_v2.read_v2_stage(prepare_path, "prepare")
    signed_calibrate = _api().study_v2.read_v2_stage(calibrate_path, "calibrate")
    signed_report = _api().study_v2.read_v2_stage(report_path, "report")
    assert signed_prepare["payload"]["run_id"] == RUN_ID
    assert signed_calibrate["prerequisite_stage_sha256"] == {"prepare": signed_prepare["stage_sha256"]}
    assert signed_report["prerequisite_stage_sha256"] == {
        "prepare": signed_prepare["stage_sha256"], "calibrate": signed_calibrate["stage_sha256"],
    }
    with pytest.raises(ValueError, match="trusted.*stage anchor"):
        _api().verify_v2_audit_source(root)


def test_selection_includes_all_invalid_and_ten_seeded_accepted_rows_per_family(audit_run) -> None:
    root, _, _ = audit_run
    source = _api().verify_v2_audit_source(root)
    packets = _api().select_v2_audit_rows(source)
    assert len(packets) == 123
    assert sum(packet["family"] == "expressions" for packet in packets) == 41
    assert sum(packet["family"] == "lists" for packet in packets) == 41
    assert sum(packet["family"] == "graphs" for packet in packets) == 41
    source_ids = source["score_rows"]
    score_by_id = {row["example_id"]: row for row in source_ids}
    audit_id_to_source = {
        hashlib.sha256(f"v2-audit:{row['example_id']}".encode()).hexdigest()[:16]: row["example_id"]
        for row in source_ids
    }
    selected_accepted = [
        audit_id_to_source[packet["audit_id"]]
        for packet in packets
        if score_by_id[audit_id_to_source[packet["audit_id"]]]["reason_code"] != "invalid_format"
    ]
    assert selected_accepted == EXPECTED_ACCEPTED_IDS
    assert len({packet["audit_id"] for packet in packets}) == 123
    assert all(set(packet) == {"audit_id", "family", "question", "response", "stop_reason"} for packet in packets)
    assert all(not ({"answer", "latent_spec", "parsed_value", "correct", "reason_code"} & set(packet)) for packet in packets)


def test_reordered_score_file_is_rejected_even_when_row_set_is_unchanged(
    audit_run, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, score_rows = audit_run
    path = root / "scores/dev-f0.jsonl"
    path.write_bytes(b"".join(canonical_json(row) for row in reversed(score_rows)))
    calibrate_path = root / "stages/calibrate.json"
    calibrate = json.loads(calibrate_path.read_bytes())
    calibrate_payload = calibrate["payload"]
    calibrate_payload["pilot_scores"]["dev-f0"]["score_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    calibrate = _record("calibrate", calibrate["status"], calibrate_payload, calibrate["prerequisite_stage_sha256"])
    calibrate_path.write_bytes(canonical_json(calibrate))
    report_path = root / "stages/report.json"
    report = json.loads(report_path.read_bytes())
    report_payload = report["payload"]
    report_payload["calibration_stage_sha256"] = str(calibrate["stage_sha256"])
    report_payload["calibration"] = calibrate_payload
    report = _record("report", report["status"], report_payload, {
        "prepare": report["prerequisite_stage_sha256"]["prepare"],
        "calibrate": str(calibrate["stage_sha256"]),
    })
    report_path.write_bytes(canonical_json(report))
    _trust_fixture_stage_bytes(root, monkeypatch)
    with pytest.raises(ValueError, match="score ordered IDs"):
        _api().verify_v2_audit_source(root)


def test_export_authenticates_before_output_creation_and_refuses_existing_root(audit_run, tmp_path: Path) -> None:
    root, _, _ = audit_run
    score_path = root / "scores/dev-f0.jsonl"
    score_path.write_bytes(score_path.read_bytes() + b" ")
    output = tmp_path / "new-audit"
    with pytest.raises(ValueError):
        _api().export_v2_audit(root, output)
    assert not output.exists()

    score_path.write_bytes(score_path.read_bytes()[:-1])
    output.mkdir()
    legacy_manifest = output / "manifest.json"
    legacy_manifest.write_bytes(b"legacy base-level export")
    (output / "packets").mkdir()
    legacy_packet = output / "packets/expressions.jsonl"
    legacy_packet.write_bytes(b"legacy packet bytes")
    result = _api().export_v2_audit(root, output)
    export_root = Path(result["output_root"])
    assert export_root == output / result["audit_id"]
    assert legacy_manifest.read_bytes() == b"legacy base-level export"
    assert legacy_packet.read_bytes() == b"legacy packet bytes"
    assert {path.relative_to(export_root).as_posix() for path in export_root.rglob("*") if path.is_file()} == {
        "manifest.json", "packets/expressions.jsonl", "packets/lists.jsonl", "packets/graphs.jsonl",
    }
    manifest = json.loads((export_root / "manifest.json").read_bytes())
    assert len(manifest["audit_ids"]["expressions"]) == 41
    assert len(manifest["source_ids"]) == 123
    assert len(manifest["selection_reasons"]) == 123
    with pytest.raises(FileExistsError):
        _api().export_v2_audit(root, output)
    with pytest.raises(ValueError, match="outside the read-only source run"):
        _api().export_v2_audit(root, root / "audit-output")
    assert not (root / "audit-output").exists()


def test_packet_ids_use_required_hash_prefix_and_family_files_contain_only_blind_fields(audit_run, tmp_path: Path) -> None:
    root, _, _ = audit_run
    output = tmp_path / "export"
    result = _api().export_v2_audit(root, output)
    assert result["audit_id"]
    export_root = Path(result["output_root"])
    assert export_root == output / result["audit_id"]
    manifest = json.loads((export_root / "manifest.json").read_bytes())
    for family in FAMILIES:
        raw = (export_root / f"packets/{family}.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == 41
        assert hashlib.sha256(raw).hexdigest() == manifest["packet_sha256"][family]
        for packet in rows:
            assert packet["audit_id"] == hashlib.sha256(
                f"v2-audit:{manifest['source_ids'][packet['audit_id']]}".encode()
            ).hexdigest()[:16]
            assert set(packet) == {"audit_id", "family", "question", "response", "stop_reason"}


def test_audit_id_changes_when_authenticated_source_hashes_change(audit_run, tmp_path: Path) -> None:
    root, _, _ = audit_run
    first = _api().export_v2_audit(root, tmp_path / "first")
    second = _api().export_v2_audit(root, tmp_path / "second")
    assert first["audit_id"] == second["audit_id"]
    manifest = json.loads((Path(second["output_root"]) / "manifest.json").read_bytes())
    assert manifest["seed"] == 927001
    assert manifest["source_hashes"]
    assert manifest["stage_hashes"]
    third = _api().export_v2_audit(root, tmp_path / "third", seed=927002)
    assert third["audit_id"] != first["audit_id"]


def test_cli_export_prints_compact_json(audit_run, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _, _ = audit_run
    output = tmp_path / "cli-export"
    result = _api().main([
        "--stage", "export", "--run-root", str(root), "--output-root", str(output),
    ])
    printed = capsys.readouterr().out
    assert result == 0
    assert printed.startswith('{"audit_id":"')
    assert " " not in printed.strip()
    result_payload = json.loads(printed)
    assert result_payload["output_root"] == str(output / result_payload["audit_id"])
