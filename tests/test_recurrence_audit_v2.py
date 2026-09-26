from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ppmoe.recurrence.provenance import canonical_json, sha256_payload


RUN_ID = "18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d"
FAMILIES = ("expressions", "lists", "graphs")
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
            latent_spec, expected = _fixture_answer(family, index)
            dev_rows.append({
                "id": example_id,
                "family": family,
                "split": "dev",
                "prompt": f"Question for {example_id}?",
                "answer": expected,
                "latent_spec": latent_spec,
                "structure_id": f"structure-{example_id}",
            })
            reason = "invalid_format" if index < 31 else ("wrong_value" if index < 61 else "correct")
            if reason == "invalid_format":
                completion_text, parsed_value = f"Final answer: {_fixture_render(expected)}", None
                if family == "lists" and index == 0:
                    completion_text = "Final list: [0,1]"
                if family == "expressions" and index == 1:
                    completion_text = "Final answer: 2\nFinal answer: 3"
                if family == "expressions" and index == 2:
                    completion_text = "Intermediate result: 3\n**Final answer: 4.**"
            elif reason == "wrong_value":
                wrong_value = _fixture_wrong_value(family, expected)
                completion_text, parsed_value = f"Final answer: {_fixture_render(wrong_value)}", wrong_value
            else:
                completion_text, parsed_value = f"Final answer: {_fixture_render(expected)}", expected
                if family == "expressions" and index == 68:
                    completion_text = "Reasoning: 69\nFinal answer: 70"
            score_rows.append({
                "example_id": example_id,
                "family": family,
                "completion_text": completion_text,
                "stop_reason": "length" if family == "expressions" and index == 0 else "eos",
                "reason_code": reason,
                "correct": reason == "correct",
                "parsed_value": parsed_value,
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


def _fixture_answer(family: str, index: int) -> tuple[dict[str, object], object]:
    if family == "expressions":
        return {"leaves": [index, 1], "operators": ["+"], "shape": "chain"}, index + 1
    if family == "lists":
        values = [index % 10, (index + 1) % 10]
        return {"input": values, "ops": []}, values
    edges = ["A->B"] if index % 2 == 0 else ["A->B", "B->C"]
    return {"n": 3, "edges": edges, "start": "A", "end": "C"}, "yes" if index % 2 else "no"


def _fixture_render(value: object) -> str:
    return json.dumps(value, separators=(",", ":")) if isinstance(value, list) else str(value)


def _fixture_wrong_value(family: str, expected: object) -> object:
    if family == "expressions":
        return int(expected) + 1
    if family == "lists":
        assert isinstance(expected, list)
        return [(int(expected[0]) + 1) % 10, expected[1]]
    return "no" if expected == "yes" else "yes"


def _write_review_files(
    audit_root: Path,
    *,
    overrides: dict[str, dict[str, object]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    manifest = json.loads((audit_root / "manifest.json").read_bytes())
    overrides = overrides or {}
    reviews: dict[str, list[dict[str, object]]] = {}
    for family in FAMILIES:
        packet_path = audit_root / f"packets/{family}.jsonl"
        packets = [json.loads(line) for line in packet_path.read_bytes().splitlines()]
        rows: list[dict[str, object]] = []
        for packet in packets:
            source_id = manifest["source_ids"][packet["audit_id"]]
            value = _fixture_answer(family, int(source_id.rsplit("-", 1)[1]))[1]
            claim_value = value
            if manifest["selection_reasons"][packet["audit_id"]] == "wrong_value":
                claim_value = _fixture_wrong_value(family, value)
            row: dict[str, object] = {
                "audit_id": packet["audit_id"],
                "family": family,
                "status": "conclusive" if packet["stop_reason"].casefold() == "eos" else "truncated",
                "claimed_value": claim_value if packet["stop_reason"].casefold() == "eos" else None,
                "evidence_span": packet["response"] if packet["stop_reason"].casefold() == "eos" else "",
                "reason": "" if packet["stop_reason"].casefold() == "eos" else "The completion did not end with EOS.",
                "reviewer_id": f"gpt-6-luna-{family}",
                "explanation_status": "indeterminate" if family == "graphs" else "not_applicable",
                "explanation_note": "No rationale was offered." if family == "graphs" else "",
            }
            row.update(overrides.get(packet["audit_id"], {}))
            rows.append(row)
        reviews[family] = rows
        review_path = audit_root / f"reviews/{family}-luna.jsonl"
        review_path.parent.mkdir(exist_ok=True)
        review_path.write_bytes(b"".join(canonical_json(row) for row in rows))
    return reviews


def _export_fixture_audit(audit_run, tmp_path: Path) -> Path:
    source_root, _, _ = audit_run
    result = _api().export_v2_audit(source_root, tmp_path / "audit")
    return Path(result["output_root"])


def _trust_fixture_stage_bytes(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hashes = {
        stage: json.loads((root / f"stages/{stage}.json").read_bytes())["stage_sha256"]
        for stage in ("prepare", "calibrate", "report")
    }
    monkeypatch.setattr(_api(), "_TRUSTED_STAGE_HASHES", hashes)


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


def _source_audit_id(audit_root: Path, source_id: str) -> str:
    manifest = json.loads((audit_root / "manifest.json").read_bytes())
    return next(audit_id for audit_id, original_id in manifest["source_ids"].items() if original_id == source_id)


def _replace_review_row(audit_root: Path, family: str, audit_id: str, changes: dict[str, object]) -> None:
    path = audit_root / f"reviews/{family}-luna.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    for row in rows:
        if row["audit_id"] == audit_id:
            row.update(changes)
            break
    else:
        raise AssertionError(f"review row {audit_id} was not found")
    path.write_bytes(b"".join(canonical_json(row) for row in rows))


def test_reconcile_preserves_wrong_reviewer_claim_and_flags_v2_disagreement(audit_run, tmp_path: Path) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    _write_review_files(audit_root)
    audit_id = _source_audit_id(audit_root, "expressions-068")
    _replace_review_row(audit_root, "expressions", audit_id, {
        "claimed_value": 70, "evidence_span": "Final answer: 70", "reason": "",
    })

    score_path = audit_run[0] / "scores/dev-f0.jsonl"
    score_bytes = score_path.read_bytes()
    review_bytes = (audit_root / "reviews/expressions-luna.jsonl").read_bytes()
    result = _api().reconcile_v2_audit(audit_run[0], audit_root)

    diagnostic = json.loads(Path(result["diagnostic_path"]).read_bytes())
    row = next(item for item in diagnostic["rows"] if item["audit_id"] == audit_id)
    assert row["model_claim"]["claimed_value"] == 70
    assert row["classification"] == "wrong"
    assert row["reviewer_v2_disagreement"] is True
    assert row["v2_reason_code"] == "correct"
    assert row["v2_parsed_value"] == 69
    assert score_path.read_bytes() == score_bytes
    assert (audit_root / "reviews/expressions-luna.jsonl").read_bytes() == review_bytes
    assert diagnostic["review_file_sha256"]["expressions"] == hashlib.sha256(review_bytes).hexdigest()
    assert set(diagnostic["counts_by_family"]["expressions"]) == {
        "correct", "wrong", "ambiguous", "missing", "truncated",
    }


def test_reconcile_sends_unsubstantiated_claim_to_adjudication(audit_run, tmp_path: Path) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    _write_review_files(audit_root)
    audit_id = _source_audit_id(audit_root, "expressions-068")
    _replace_review_row(audit_root, "expressions", audit_id, {"claimed_value": 999})

    result = _api().reconcile_v2_audit(audit_run[0], audit_root)

    diagnostic = json.loads(Path(result["diagnostic_path"]).read_bytes())
    row = next(item for item in diagnostic["rows"] if item["audit_id"] == audit_id)
    assert row["model_claim"]["claimed_value"] == 999
    assert row["adjudication_required"] is True
    assert row["classification"] == "ambiguous"
    assert diagnostic["counts_by_family"]["expressions"]["ambiguous"] == 3


@pytest.mark.parametrize("evidence_span", ["**Final answer: 4.**", "Intermediate result: 3"])
def test_reconcile_rejects_claim_not_supported_by_its_final_evidence(
    audit_run, tmp_path: Path, evidence_span: str,
) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    _write_review_files(audit_root)
    audit_id = _source_audit_id(audit_root, "expressions-002")
    _replace_review_row(audit_root, "expressions", audit_id, {
        "claimed_value": 3, "evidence_span": evidence_span,
    })

    result = _api().reconcile_v2_audit(audit_run[0], audit_root)

    diagnostic = json.loads(Path(result["diagnostic_path"]).read_bytes())
    row = next(item for item in diagnostic["rows"] if item["audit_id"] == audit_id)
    assert row["model_claim"]["claimed_value"] == 3
    assert row["adjudication_required"] is True
    assert row["classification"] == "ambiguous"
    assert row["parser_miss"] is False


def test_reconcile_accepts_claim_tied_to_unambiguous_final_cue(audit_run, tmp_path: Path) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    _write_review_files(audit_root)
    audit_id = _source_audit_id(audit_root, "expressions-002")
    _replace_review_row(audit_root, "expressions", audit_id, {
        "claimed_value": 4, "evidence_span": "**Final answer: 4.**",
    })

    result = _api().reconcile_v2_audit(audit_run[0], audit_root)

    diagnostic = json.loads(Path(result["diagnostic_path"]).read_bytes())
    row = next(item for item in diagnostic["rows"] if item["audit_id"] == audit_id)
    assert row["classification"] == "wrong"
    assert row["adjudication_required"] is False


@pytest.mark.parametrize("manifest_encoding", ["whitespace", "reordered_keys"])
def test_reconcile_rejects_noncanonical_manifest_bytes_before_output(
    audit_run, tmp_path: Path, manifest_encoding: str,
) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    _write_review_files(audit_root)
    manifest_path = audit_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest_encoding == "whitespace":
        raw = json.dumps(manifest, indent=2).encode("utf-8")
    else:
        raw = json.dumps(dict(reversed(list(manifest.items()))), separators=(",", ":")).encode("utf-8")
    manifest_path.write_bytes(raw)

    with pytest.raises(ValueError, match="canonical|manifest"):
        _api().reconcile_v2_audit(audit_run[0], audit_root)

    assert not (audit_root / "diagnostic.json").exists()


def test_reconcile_classifies_correct_final_list_as_parser_miss(audit_run, tmp_path: Path) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    source_id = "lists-000"
    score_path = audit_run[0] / "scores/dev-f0.jsonl"
    score_rows = [json.loads(line) for line in score_path.read_bytes().splitlines()]
    # The export fixture's invalid_format score models V2 missing the Final list cue.
    assert next(row for row in score_rows if row["example_id"] == source_id)["reason_code"] == "invalid_format"
    _write_review_files(audit_root)
    packets = [json.loads(line) for line in (audit_root / "packets/lists.jsonl").read_bytes().splitlines()]
    assert next(packet for packet in packets if packet["audit_id"] == _source_audit_id(audit_root, source_id))["response"] == "Final list: [0,1]"
    audit_id = _source_audit_id(audit_root, source_id)
    _replace_review_row(audit_root, "lists", audit_id, {
        "evidence_span": "Final list: [0,1]",
    })

    result = _api().reconcile_v2_audit(audit_run[0], audit_root)

    diagnostic = json.loads(Path(result["diagnostic_path"]).read_bytes())
    row = next(item for item in diagnostic["rows"] if item["audit_id"] == audit_id)
    assert row["model_claim"]["claimed_value"] == [0, 1]
    assert row["classification"] == "correct"
    assert row["parser_miss"] is True
    assert row["reviewer_v2_disagreement"] is False


def test_reconcile_keeps_explicit_conflicting_cues_ambiguous(audit_run, tmp_path: Path) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    _write_review_files(audit_root)
    audit_id = _source_audit_id(audit_root, "expressions-001")
    _replace_review_row(audit_root, "expressions", audit_id, {
        "status": "ambiguous", "claimed_value": None, "evidence_span": "Final answer: 2",
        "reason": "The response gives two conflicting final values.",
    })

    result = _api().reconcile_v2_audit(audit_run[0], audit_root)

    diagnostic = json.loads(Path(result["diagnostic_path"]).read_bytes())
    row = next(item for item in diagnostic["rows"] if item["audit_id"] == audit_id)
    assert row["classification"] == "ambiguous"
    assert row["model_claim"]["claimed_value"] is None
    assert diagnostic["counts_by_family"]["expressions"]["ambiguous"] == 3
    packets = [json.loads(line) for line in (audit_root / "packets/expressions.jsonl").read_bytes().splitlines()]
    assert next(packet for packet in packets if packet["audit_id"] == audit_id)["response"] == "Final answer: 2\nFinal answer: 3"


def test_reconcile_validates_all_reviews_before_exclusive_diagnostic_write(audit_run, tmp_path: Path) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    _write_review_files(audit_root)
    review_path = audit_root / "reviews/graphs-luna.jsonl"
    original = [json.loads(line) for line in review_path.read_bytes().splitlines()]
    original[0]["unrecognized"] = "extra field"
    review_path.write_bytes(b"".join(canonical_json(row) for row in original))
    before = {path: path.read_bytes() for path in (audit_root / "reviews").glob("*.jsonl")}

    with pytest.raises(ValueError, match="exactly|field"):
        _api().reconcile_v2_audit(audit_run[0], audit_root)

    assert not (audit_root / "diagnostic.json").exists()
    assert before == {path: path.read_bytes() for path in (audit_root / "reviews").glob("*.jsonl")}


@pytest.mark.parametrize("malformation", [
    "missing_id", "duplicate_id", "extra_id", "wrong_family", "invalid_status", "unhashable_status",
    "wrong_expression_type", "false_eos_conclusive", "bad_graph_explanation", "human_reviewer",
])
def test_reconcile_rejects_malformed_review_rows_before_writing(audit_run, tmp_path: Path, malformation: str) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    reviews = _write_review_files(audit_root)
    rows = reviews["expressions"]
    if malformation == "missing_id":
        rows.pop()
    elif malformation == "duplicate_id":
        rows[1]["audit_id"] = rows[0]["audit_id"]
    elif malformation == "extra_id":
        rows[0]["audit_id"] = "not-in-packet"
    elif malformation == "wrong_family":
        rows[0]["family"] = "lists"
    elif malformation == "invalid_status":
        rows[0]["status"] = "certain"
    elif malformation == "unhashable_status":
        rows[1]["status"] = ["conclusive"]
    elif malformation == "wrong_expression_type":
        rows[1]["claimed_value"] = True
    elif malformation == "false_eos_conclusive":
        rows[0].update({"status": "conclusive", "claimed_value": 1, "evidence_span": "1", "reason": ""})
    elif malformation == "bad_graph_explanation":
        rows = reviews["graphs"]
        rows[0]["explanation_status"] = "not_applicable"
    else:
        rows[0]["reviewer_id"] = "human-reviewer-1"
    if malformation in {"missing_id", "duplicate_id", "extra_id", "wrong_family", "invalid_status", "unhashable_status", "wrong_expression_type", "false_eos_conclusive", "human_reviewer"}:
        (audit_root / "reviews/expressions-luna.jsonl").write_bytes(b"".join(canonical_json(row) for row in rows))
    else:
        (audit_root / "reviews/graphs-luna.jsonl").write_bytes(b"".join(canonical_json(row) for row in rows))

    with pytest.raises(ValueError):
        _api().reconcile_v2_audit(audit_run[0], audit_root)
    assert not (audit_root / "diagnostic.json").exists()


def test_reconcile_rejects_packet_hash_drift_and_cli_prints_compact_json(audit_run, tmp_path: Path, capsys) -> None:
    audit_root = _export_fixture_audit(audit_run, tmp_path)
    _write_review_files(audit_root)
    packet_path = audit_root / "packets/lists.jsonl"
    packet_path.write_bytes(packet_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="packet.*hash|hash.*packet"):
        _api().reconcile_v2_audit(audit_run[0], audit_root)
    assert not (audit_root / "diagnostic.json").exists()

    packet_path.write_bytes(packet_path.read_bytes()[:-1])
    result = _api().main(["--stage", "reconcile", "--run-root", str(audit_run[0]), "--audit-root", str(audit_root)])
    printed = capsys.readouterr().out
    assert result == 0
    assert " " not in printed.strip()
    assert json.loads(printed)["diagnostic_path"] == str(audit_root / "diagnostic.json")
