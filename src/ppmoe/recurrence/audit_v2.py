"""Blind, authenticated export of selected V2 development answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Mapping

from .data import FAMILIES, interpret
from .provenance import canonical_json, sha256_payload
from . import study_v2


RUN_ID = "18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d"
_TRUSTED_STAGE_HASHES = {
    "prepare": "b1bd10a2d019907fdf222d092879b9c5cdad68861072915407632d6eeaec4a46",
    "calibrate": "da11072965258f088322ecc567981078adf0895ba8f1a947a076b18acd1fae7d",
    "report": "0799e0c3d34bacac86cd0c942693a29d18bd502ec6d83713feebf975ef0fcbcf",
}
DEFAULT_SEED = 927001
_DEV_SPLIT = Path("data/dev.jsonl")
_DEV_SIDECAR = Path("data/dev.jsonl.manifest.json")
_DEV_SCORE = Path("scores/dev-f0.jsonl")
_REVIEW_FIELDS = {
    "audit_id", "family", "status", "claimed_value", "evidence_span", "reason",
    "reviewer_id", "explanation_status", "explanation_note",
}
_REVIEW_STATUSES = {"conclusive", "ambiguous", "missing", "truncated"}
_EXPLANATION_STATUSES = {"valid", "invalid", "indeterminate"}
_AUDIT_FINAL_CUE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:[*_]{1,3})?(?:Final[ \t]+(?:answer|result|list)|Answer|"
    r"So[ \t]+the[ \t]+final[ \t]+answer[ \t]+is)\b(?:[*_]{1,3})?[ \t]*:?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
_AUDIT_BOXED_INTEGER = re.compile(r"^\\boxed\s*\{\s*([+-]?\d+)\s*\}")
_AUDIT_BARE_INTEGER = re.compile(r"^(?:is\b[ \t]*)?(?:=[ \t]*)?([+-]?\d+)(?![\w+-]|\.\d)", re.IGNORECASE)
_AUDIT_VECTOR = re.compile(r"^\[\s*[+-]?\d+(?:\s*,\s*[+-]?\d+)*\s*\]")
_AUDIT_GRAPH_VALUE = re.compile(r"^(?:is\b[ \t]*)?(yes|no)\b", re.IGNORECASE)
_AUDIT_GRAPH_LEADING = re.compile(r"^[ \t]*(yes|no)\s*\.", re.IGNORECASE)
_AUDIT_GRAPH_CONCLUSION = re.compile(
    r"\bthere\s+is\s+(?P<negative>no\s+)?(?:(?:direct(?:\s+or\s+indirect)?|indirect)\s+)?path\s+"
    r"from\s+[A-Z]\s+to\s+[A-Z]\b",
    re.IGNORECASE,
)


def verify_v2_audit_source(run_root: Path) -> dict[str, object]:
    """Authenticate V2 stages and development evidence without writing or reading sealed data."""
    root = Path(run_root).resolve(strict=True)
    if root.name != RUN_ID or not root.is_dir():
        raise ValueError("V2 audit source must be the registered run ID")

    prepare = study_v2.read_v2_stage(study_v2._safe_v2_run_file(root, Path("stages/prepare.json")), "prepare")
    calibrate = study_v2.read_v2_stage(study_v2._safe_v2_run_file(root, Path("stages/calibrate.json")), "calibrate")
    report = study_v2.read_v2_stage(study_v2._safe_v2_run_file(root, Path("stages/report.json")), "report")
    prepare_hash = str(prepare["stage_sha256"])
    calibrate_hash = str(calibrate["stage_sha256"])
    stage_hashes = {
        "prepare": prepare_hash,
        "calibrate": calibrate_hash,
        "report": str(report["stage_sha256"]),
    }
    if stage_hashes != _TRUSTED_STAGE_HASHES:
        raise ValueError("V2 stage hashes do not match trusted registered stage anchors")
    if prepare.get("status") != "passed":
        raise ValueError("V2 prepare stage is not passed")
    if calibrate.get("prerequisite_stage_sha256") != {"prepare": prepare_hash}:
        raise ValueError("V2 calibrate prerequisite does not bind prepare")
    if report.get("prerequisite_stage_sha256") != {"prepare": prepare_hash, "calibrate": calibrate_hash}:
        raise ValueError("V2 report prerequisite does not bind prepare and calibrate")

    prepare_payload = _mapping(prepare.get("payload"), "V2 prepare payload")
    calibrate_payload = _mapping(calibrate.get("payload"), "V2 calibrate payload")
    report_payload = _mapping(report.get("payload"), "V2 report payload")
    for payload in (prepare_payload, calibrate_payload, report_payload):
        if payload.get("run_id") != RUN_ID:
            raise ValueError("V2 audit stage run ID differs from the registered run")
    if calibrate.get("status") != "stopped_scientific" or report.get("status") != "stopped_scientific":
        raise ValueError("V2 audit requires matching stopped_scientific calibration and report stages")
    if report_payload.get("status") != "stopped_scientific" or report_payload.get("next_stage") != "complete":
        raise ValueError("V2 report is not the completed scientific stop")
    if report_payload.get("prepare_stage_sha256") != prepare_hash or report_payload.get("calibration_stage_sha256") != calibrate_hash:
        raise ValueError("V2 report payload does not bind the authenticated stages")
    if report_payload.get("calibration") != dict(calibrate_payload):
        raise ValueError("V2 report calibration payload differs from the authenticated calibration stage")

    score_manifest = _mapping(calibrate_payload.get("pilot_scores"), "V2 calibration scores")
    dev_evidence = _mapping(score_manifest.get("dev-f0"), "V2 dev-f0 evidence")
    if dev_evidence.get("split_count") != 300 or dev_evidence.get("score_count") != 300:
        raise ValueError("V2 dev-f0 evidence must bind 300 split and score rows")

    split_raw = _read(root, _DEV_SPLIT)
    sidecar_raw = _read(root, _DEV_SIDECAR)
    score_raw = _read(root, _DEV_SCORE)
    split_hash = hashlib.sha256(split_raw).hexdigest()
    sidecar_hash = hashlib.sha256(sidecar_raw).hexdigest()
    score_hash = hashlib.sha256(score_raw).hexdigest()
    if split_hash != dev_evidence.get("split_sha256"):
        raise ValueError("V2 dev-f0 split hash differs from calibration evidence")
    if sidecar_hash != dev_evidence.get("split_manifest_sha256"):
        raise ValueError("V2 dev-f0 sidecar hash differs from calibration evidence")
    if score_hash != dev_evidence.get("score_sha256"):
        raise ValueError("V2 dev-f0 score hash differs from calibration evidence")

    dev_rows = _parse_jsonl(split_raw, "V2 dev-f0 split")
    score_rows = _parse_jsonl(score_raw, "V2 dev-f0 score")
    if len(dev_rows) != 300 or len(score_rows) != 300:
        raise ValueError("V2 dev-f0 files must each contain exactly 300 JSONL rows")
    ids = [_nonempty_string(row.get("id"), "V2 dev-f0 ID") for row in dev_rows]
    score_ids = [_nonempty_string(row.get("example_id"), "V2 dev-f0 score ID") for row in score_rows]
    if len(set(ids)) != 300:
        raise ValueError("V2 dev-f0 split IDs must be distinct")
    if score_ids != ids:
        raise ValueError("V2 dev-f0 score ordered IDs differ from the authenticated split")
    if any(row.get("split") != "dev" for row in dev_rows):
        raise ValueError("V2 dev-f0 split contains a non-dev row")
    if any(row.get("family") != dev.get("family") for row, dev in zip(score_rows, dev_rows, strict=True)):
        raise ValueError("V2 dev-f0 score family order differs from the authenticated split")

    sidecar = _parse_object(sidecar_raw, "V2 dev-f0 sidecar")
    if sidecar.get("split") != "dev" or sidecar.get("sha256") != split_hash or sidecar.get("count") != 300:
        raise ValueError("V2 dev-f0 sidecar content does not bind the split bytes and count")
    if sidecar.get("ordered_ids") != ids:
        raise ValueError("V2 dev-f0 sidecar ordered IDs differ from the split")

    return {
        "run_id": RUN_ID,
        "dev_rows": dev_rows,
        "score_rows": score_rows,
        "hashes": {
            "dev_split": split_hash,
            "dev_sidecar": sidecar_hash,
            "dev_score": score_hash,
        },
        "stage_hashes": stage_hashes,
    }


def select_v2_audit_rows(source: Mapping[str, object], seed: int = DEFAULT_SEED) -> list[dict[str, str]]:
    """Select every invalid-format answer and ten seeded accepted answers per family."""
    dev_rows = _mapping_rows(source.get("dev_rows"), "authenticated dev rows")
    score_rows = _mapping_rows(source.get("score_rows"), "authenticated score rows")
    score_by_id: dict[str, Mapping[str, object]] = {}
    for score in score_rows:
        example_id = _nonempty_string(score.get("example_id"), "score example ID")
        if example_id in score_by_id:
            raise ValueError("authenticated V2 scores contain duplicate IDs")
        score_by_id[example_id] = score
    if len(dev_rows) != 300 or len(score_by_id) != 300:
        raise ValueError("V2 audit selection requires exactly 300 authenticated rows")

    by_family: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = {family: [] for family in FAMILIES}
    for row in dev_rows:
        example_id = _nonempty_string(row.get("id"), "dev example ID")
        score = score_by_id.get(example_id)
        if score is None or score.get("family") != row.get("family"):
            raise ValueError("V2 audit scores do not match development rows")
        family = row.get("family")
        if family not in by_family:
            raise ValueError("V2 audit source contains an unknown family")
        by_family[str(family)].append((row, score))

    invalid = [pair for family in FAMILIES for pair in by_family[family] if pair[1].get("reason_code") == "invalid_format"]
    if len(invalid) != 93:
        raise ValueError("V2 audit requires exactly 93 invalid_format development answers")
    rng = random.Random(seed)
    selected: list[tuple[Mapping[str, object], Mapping[str, object]]] = list(invalid)
    for family in FAMILIES:
        accepted = sorted(
            (pair for pair in by_family[family] if pair[1].get("reason_code") in {"correct", "wrong_value"}),
            key=lambda pair: str(pair[0]["id"]),
        )
        if len(accepted) < 10:
            raise ValueError(f"V2 audit has fewer than ten accepted {family} answers")
        selected.extend(rng.sample(accepted, 10))

    packets: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for row, score in selected:
        source_id = str(row["id"])
        audit_id = hashlib.sha256(f"v2-audit:{source_id}".encode("utf-8")).hexdigest()[:16]
        if audit_id in seen_ids:
            raise ValueError("V2 audit ID collision")
        seen_ids.add(audit_id)
        question, response = row.get("prompt"), score.get("completion_text")
        if not isinstance(question, str) or not isinstance(response, str):
            raise ValueError("V2 audit question and response must be text")
        stop_reason = score.get("stop_reason")
        if not isinstance(stop_reason, str):
            raise ValueError("V2 audit stop reason must be text")
        packets.append({
            "audit_id": audit_id,
            "family": str(row["family"]),
            "question": question,
            "response": response,
            "stop_reason": stop_reason,
        })
    if len(packets) != 123:
        raise ValueError("V2 audit selection must contain 123 distinct rows")
    return packets


def export_v2_audit(run_root: Path, output_root: Path, seed: int = DEFAULT_SEED) -> dict[str, str]:
    """Verify, then exclusively write the blind family packets and private manifest."""
    source = verify_v2_audit_source(run_root)
    packets = select_v2_audit_rows(source, seed)
    audit_id = sha256_payload({
        "schema_version": 1,
        "run_id": source["run_id"],
        "source_hashes": source["hashes"],
        "stage_hashes": source["stage_hashes"],
        "seed": seed,
    })
    dev_by_id = {str(row["id"]): row for row in _mapping_rows(source["dev_rows"], "authenticated dev rows")}
    scores_by_id = {str(row["example_id"]): row for row in _mapping_rows(source["score_rows"], "authenticated score rows")}
    source_ids: dict[str, str] = {}
    selection_reasons: dict[str, str] = {}
    packet_groups: dict[str, list[dict[str, str]]] = {family: [] for family in FAMILIES}
    audit_ids: dict[str, list[str]] = {family: [] for family in FAMILIES}
    source_id_by_audit_id = {
        hashlib.sha256(f"v2-audit:{identifier}".encode("utf-8")).hexdigest()[:16]: identifier
        for identifier in dev_by_id
    }
    for packet in packets:
        family, audit_row_id = packet["family"], packet["audit_id"]
        source_id = source_id_by_audit_id[audit_row_id]
        if audit_row_id in source_ids:
            raise ValueError("V2 audit ID collision")
        source_ids[audit_row_id] = source_id
        selection_reasons[audit_row_id] = str(scores_by_id[source_id]["reason_code"])
        packet_groups[family].append(packet)
        audit_ids[family].append(audit_row_id)

    output_base = Path(output_root).resolve(strict=False)
    destination = output_base / audit_id
    try:
        destination.relative_to(Path(run_root).resolve(strict=True))
    except ValueError:
        pass
    else:
        raise ValueError("V2 audit output must be outside the read-only source run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    packet_hashes: dict[str, str] = {}
    for family in FAMILIES:
        raw = b"".join(canonical_json(packet) for packet in packet_groups[family])
        _write_exclusive(destination / "packets" / f"{family}.jsonl", raw, make_parent=True)
        packet_hashes[family] = hashlib.sha256(raw).hexdigest()
    manifest = {
        "schema_version": 1,
        "audit_id": audit_id,
        "run_id": source["run_id"],
        "seed": seed,
        "source_hashes": source["hashes"],
        "stage_hashes": source["stage_hashes"],
        "audit_ids": audit_ids,
        "packet_sha256": packet_hashes,
        "source_ids": source_ids,
        "selection_reasons": selection_reasons,
    }
    _write_exclusive(destination / "manifest.json", canonical_json(manifest))
    return {"audit_id": audit_id, "output_root": str(destination)}


def reconcile_v2_audit(run_root: Path, audit_root: Path) -> dict[str, object]:
    """Validate blind reviews against authenticated V2 rows, then exclusively write diagnostics."""
    source = verify_v2_audit_source(run_root)
    root = Path(audit_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("V2 audit root must be a directory")
    source_root = Path(run_root).resolve(strict=True)
    try:
        root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("V2 audit root must be outside the read-only source run")

    manifest_raw = (root / "manifest.json").read_bytes()
    manifest = _parse_object(manifest_raw, "V2 audit manifest")
    manifest, packets, source_ids = _authenticate_audit_export(source, root, manifest)
    if canonical_json(manifest) != manifest_raw:
        raise ValueError("V2 audit manifest bytes are not canonical")

    reviews: dict[str, list[dict[str, object]]] = {}
    review_hashes: dict[str, str] = {}
    for family in FAMILIES:
        review_path = root / "reviews" / f"{family}-luna.jsonl"
        try:
            review_raw = review_path.read_bytes()
        except OSError as error:
            raise ValueError(f"missing V2 {family} review file") from error
        review_hashes[family] = hashlib.sha256(review_raw).hexdigest()
        review_rows = _parse_jsonl(review_raw, f"V2 {family} reviews")
        reviews[family] = _validate_review_rows(family, review_rows, packets[family])
    assigned_reviewers = {
        family: str(reviews[family][0]["reviewer_id"])
        for family in FAMILIES
    }
    if len(set(assigned_reviewers.values())) != len(FAMILIES):
        raise ValueError("each family review must identify a distinct GPT-6 Luna reviewer")

    dev_by_id = {str(row["id"]): row for row in _mapping_rows(source["dev_rows"], "authenticated dev rows")}
    scores_by_id = {str(row["example_id"]): row for row in _mapping_rows(source["score_rows"], "authenticated score rows")}
    counts_by_family = {
        family: {key: 0 for key in ("correct", "wrong", "ambiguous", "missing", "truncated")}
        for family in FAMILIES
    }
    explanation_counts = {key: 0 for key in sorted(_EXPLANATION_STATUSES)}
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        review_by_id = {str(row["audit_id"]): row for row in reviews[family]}
        for packet in packets[family]:
            audit_id = packet["audit_id"]
            review = review_by_id[audit_id]
            source_id = source_ids[audit_id]
            dev = dev_by_id[source_id]
            score = scores_by_id[source_id]
            expected = interpret(_mapping(dev.get("latent_spec"), "V2 latent specification"), family)
            status = str(review["status"])
            claimed = review["claimed_value"]
            adjudication_required = bool(
                status == "conclusive"
                and not _claim_has_final_evidence(claimed, family, review["evidence_span"], packet["response"])
            )
            if status == "conclusive":
                classification = "ambiguous" if adjudication_required else ("correct" if claimed == expected else "wrong")
            else:
                classification = status
            counts_by_family[family][classification] += 1
            if family == "graphs":
                explanation_counts[str(review["explanation_status"])] += 1

            reason_code = score.get("reason_code")
            v2_parsed = score.get("parsed_value")
            v2_correct = score.get("correct")
            reviewer_v2_disagreement = bool(
                status == "conclusive"
                and not adjudication_required
                and reason_code in {"correct", "wrong_value"}
                and (claimed != v2_parsed or (claimed == expected) != (v2_correct is True))
            )
            parser_miss = bool(
                status == "conclusive" and not adjudication_required
                and claimed == expected and reason_code == "invalid_format"
            )
            rows.append({
                "audit_id": audit_id,
                "family": family,
                "source_id": source_id,
                "classification": classification,
                "model_claim": {
                    "status": status,
                    "claimed_value": claimed,
                    "evidence_span": review["evidence_span"],
                    "reason": review["reason"],
                    "reviewer_id": review["reviewer_id"],
                    "explanation_status": review["explanation_status"],
                    "explanation_note": review["explanation_note"],
                },
                "v2_reason_code": reason_code,
                "v2_parsed_value": v2_parsed,
                "v2_correct": v2_correct,
                "v2_failure_detail": score.get("failure_detail"),
                "reference_value": expected,
                "adjudication_required": adjudication_required,
                "reviewer_v2_disagreement": reviewer_v2_disagreement,
                "parser_miss": parser_miss,
            })

    diagnostic: dict[str, object] = {
        "schema_version": 1,
        "audit_id": manifest["audit_id"],
        "run_id": source["run_id"],
        "source_hashes": source["hashes"],
        "stage_hashes": source["stage_hashes"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "packet_sha256": manifest["packet_sha256"],
        "review_file_sha256": review_hashes,
        "counts_by_family": counts_by_family,
        "graph_explanation_counts": explanation_counts,
        "rows": rows,
    }
    diagnostic_path = root / "diagnostic.json"
    _write_exclusive(diagnostic_path, canonical_json(diagnostic))
    return {
        "audit_id": manifest["audit_id"],
        "diagnostic_path": str(diagnostic_path),
        "row_count": len(rows),
        "counts_by_family": counts_by_family,
    }


def _authenticate_audit_export(
    source: Mapping[str, object], root: Path, manifest: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, list[dict[str, str]]], dict[str, str]]:
    expected_manifest_fields = {
        "schema_version", "audit_id", "run_id", "seed", "source_hashes", "stage_hashes",
        "audit_ids", "packet_sha256", "source_ids", "selection_reasons",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("V2 audit manifest has unexpected or missing fields")
    seed = manifest["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("V2 audit manifest seed must be an integer")
    audit_id = sha256_payload({
        "schema_version": 1,
        "run_id": source["run_id"],
        "source_hashes": source["hashes"],
        "stage_hashes": source["stage_hashes"],
        "seed": seed,
    })
    if manifest.get("schema_version") != 1 or manifest.get("audit_id") != audit_id:
        raise ValueError("V2 audit manifest identity does not match authenticated source hashes")
    if manifest.get("run_id") != source.get("run_id"):
        raise ValueError("V2 audit manifest run ID differs from the authenticated source")
    if manifest.get("source_hashes") != source.get("hashes") or manifest.get("stage_hashes") != source.get("stage_hashes"):
        raise ValueError("V2 audit manifest source hashes differ from authenticated V2 evidence")
    if root.name != audit_id:
        raise ValueError("V2 audit root name does not match its manifest identity")

    selected = select_v2_audit_rows(source, seed)
    packets: dict[str, list[dict[str, str]]] = {family: [] for family in FAMILIES}
    for packet in selected:
        packets[packet["family"]].append(packet)
    expected_audit_ids = {family: [packet["audit_id"] for packet in packets[family]] for family in FAMILIES}
    source_id_by_audit_id = {
        hashlib.sha256(f"v2-audit:{row['id']}".encode("utf-8")).hexdigest()[:16]: str(row["id"])
        for row in _mapping_rows(source["dev_rows"], "authenticated dev rows")
    }
    scores_by_id = {
        str(score["example_id"]): score
        for score in _mapping_rows(source["score_rows"], "authenticated score rows")
    }
    expected_source_ids: dict[str, str] = {}
    expected_reasons: dict[str, str] = {}
    for family in FAMILIES:
        for packet in packets[family]:
            source_id = source_id_by_audit_id[packet["audit_id"]]
            expected_source_ids[packet["audit_id"]] = source_id
            expected_reasons[packet["audit_id"]] = str(scores_by_id[source_id]["reason_code"])
    expected_packet_hashes: dict[str, str] = {}
    for family in FAMILIES:
        raw = b"".join(canonical_json(packet) for packet in packets[family])
        expected_packet_hashes[family] = hashlib.sha256(raw).hexdigest()
        packet_path = root / "packets" / f"{family}.jsonl"
        try:
            packet_raw = packet_path.read_bytes()
        except OSError as error:
            raise ValueError(f"missing V2 {family} packet file") from error
        if hashlib.sha256(packet_raw).hexdigest() != expected_packet_hashes[family] or packet_raw != raw:
            raise ValueError(f"V2 {family} packet hash or content differs from authenticated selection")
    expected_manifest = {
        "schema_version": 1,
        "audit_id": audit_id,
        "run_id": source["run_id"],
        "seed": seed,
        "source_hashes": source["hashes"],
        "stage_hashes": source["stage_hashes"],
        "audit_ids": expected_audit_ids,
        "packet_sha256": expected_packet_hashes,
        "source_ids": expected_source_ids,
        "selection_reasons": expected_reasons,
    }
    if dict(manifest) != expected_manifest:
        raise ValueError("V2 audit manifest hashes, packet IDs, or selection mapping drifted")
    return expected_manifest, packets, expected_source_ids


def _validate_review_rows(
    family: str, rows: list[dict[str, object]], packets: list[dict[str, str]],
) -> list[dict[str, object]]:
    packet_by_id = {packet["audit_id"]: packet for packet in packets}
    if len(rows) != len(packet_by_id):
        raise ValueError(f"V2 {family} review must contain each packet ID exactly once")
    seen: set[str] = set()
    reviewer_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        if set(row) != _REVIEW_FIELDS:
            raise ValueError(f"V2 {family} review row {line_number} must have exactly the required fields")
        audit_id = _nonempty_string(row.get("audit_id"), f"V2 {family} review audit ID")
        if audit_id not in packet_by_id:
            raise ValueError(f"V2 {family} review contains an extra audit ID")
        if audit_id in seen:
            raise ValueError(f"V2 {family} review contains a duplicate audit ID")
        seen.add(audit_id)
        packet = packet_by_id[audit_id]
        if row.get("family") != family or packet.get("family") != family:
            raise ValueError(f"V2 {family} review row has the wrong family")
        status = row.get("status")
        if not isinstance(status, str) or status not in _REVIEW_STATUSES:
            raise ValueError(f"V2 {family} review row has an invalid status")
        reviewer_id = _nonempty_string(row.get("reviewer_id"), f"V2 {family} reviewer ID")
        reviewer_id_folded = reviewer_id.casefold()
        if "gpt-6" not in reviewer_id_folded or "luna" not in reviewer_id_folded or "human" in reviewer_id_folded:
            raise ValueError(f"V2 {family} reviewer ID must identify a GPT-6 Luna model")
        reviewer_ids.add(reviewer_id)
        evidence_span = row.get("evidence_span")
        reason = row.get("reason")
        explanation_note = row.get("explanation_note")
        if not isinstance(evidence_span, str) or len(evidence_span) > 1000:
            raise ValueError(f"V2 {family} review evidence span must be short text")
        if not isinstance(reason, str):
            raise ValueError(f"V2 {family} review reason must be text")
        if not isinstance(explanation_note, str):
            raise ValueError(f"V2 {family} explanation note must be text")
        claimed = row.get("claimed_value")
        if status == "conclusive":
            if family == "expressions" and (isinstance(claimed, bool) or not isinstance(claimed, int)):
                raise ValueError("conclusive expression review must claim an integer")
            if family == "lists" and (
                not isinstance(claimed, list)
                or any(isinstance(value, bool) or not isinstance(value, int) for value in claimed)
            ):
                raise ValueError("conclusive list review must claim a list of integers")
            if family == "graphs" and (not isinstance(claimed, str) or claimed not in {"yes", "no"}):
                raise ValueError("conclusive graph review must claim yes or no")
            if not evidence_span or evidence_span not in packet["response"]:
                raise ValueError("conclusive review evidence must be a substring of the response")
        else:
            if claimed is not None:
                raise ValueError("nonconclusive review rows must have a null claimed value")
            if not reason.strip() or len(reason) > 500:
                raise ValueError("nonconclusive review rows require a short reason")
        if packet["stop_reason"].casefold() != "eos" and status != "truncated":
            raise ValueError("non-EOS V2 response must be marked truncated")
        explanation_status = row.get("explanation_status")
        if family == "graphs":
            if (
                not isinstance(explanation_status, str)
                or explanation_status not in _EXPLANATION_STATUSES
                or not explanation_note.strip()
                or len(explanation_note) > 500
            ):
                raise ValueError("graph reviews require an explanation status and short note")
        elif explanation_status != "not_applicable" or explanation_note != "":
            raise ValueError("non-graph reviews require not_applicable with an empty note")
    if seen != set(packet_by_id):
        raise ValueError(f"V2 {family} review is missing packet IDs")
    if len(reviewer_ids) != 1:
        raise ValueError(f"V2 {family} review must identify one consistent reviewer")
    return rows


def _claim_has_final_evidence(claimed: object, family: str, evidence: str, response: str) -> bool:
    evidence_values, evidence_complete = _final_assertions(evidence, family)
    response_values, response_complete = _final_assertions(response, family)
    if not evidence_complete or not response_complete or not evidence_values or not response_values:
        return False
    if not all(_same_answer(value, claimed, family) for value in evidence_values):
        return False
    if not all(_same_answer(value, claimed, family) for value in response_values):
        return False
    return True


def _final_assertions(text: str, family: str) -> tuple[list[object], bool]:
    values: list[object] = []
    complete = True
    matches = list(_AUDIT_FINAL_CUE.finditer(text))
    for index, cue in enumerate(matches):
        window_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        payload = text[cue.end() : min(window_end, cue.end() + 256)].lstrip()
        if family == "expressions":
            boxed = _AUDIT_BOXED_INTEGER.match(payload)
            bare = _AUDIT_BARE_INTEGER.match(payload)
            match = boxed or bare
            if match is None:
                complete = False
            else:
                values.append(int(match.group(1)))
        elif family == "lists":
            match = _AUDIT_VECTOR.match(payload)
            if match is None:
                complete = False
            else:
                values.append([int(part.strip()) for part in match.group(0)[1:-1].split(",")])
        else:
            match = _AUDIT_GRAPH_VALUE.match(payload)
            if match is None:
                complete = False
            else:
                values.append(match.group(1).casefold())

    if family == "graphs":
        first_line = next((line for line in text.splitlines() if line.strip()), "")
        leading = _AUDIT_GRAPH_LEADING.match(first_line)
        if leading is not None:
            values.append(leading.group(1).casefold())
        values.extend(
            "no" if match.group("negative") else "yes"
            for match in _AUDIT_GRAPH_CONCLUSION.finditer(text)
        )
    return values, complete


def _same_answer(left: object, right: object, family: str) -> bool:
    if family == "lists":
        return isinstance(left, list) and isinstance(right, list) and left == right
    return left == right


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export or reconcile a blind audit of authenticated V2 development evidence")
    parser.add_argument("--stage", required=True, choices=("export", "reconcile"))
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--audit-root", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "export":
        if args.output_root is None or args.audit_root is not None:
            parser.error("--stage export requires --output-root and does not accept --audit-root")
        result = export_v2_audit(args.run_root, args.output_root)
    else:
        if args.audit_root is None or args.output_root is not None:
            parser.error("--stage reconcile requires --audit-root and does not accept --output-root")
        result = reconcile_v2_audit(args.run_root, args.audit_root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _read(root: Path, relative: Path) -> bytes:
    path = study_v2._safe_v2_run_file(root, relative)
    return study_v2._read_bytes(path, f"V2 audit source {relative.as_posix()}")


def _parse_jsonl(raw: bytes, description: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{description} row {line_number} is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{description} row {line_number} must be an object")
        rows.append(value)
    return rows


def _parse_object(raw: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _mapping_rows(value: object, description: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{description} must be a list of objects")
    return value


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _write_exclusive(path: Path, raw: bytes, *, make_parent: bool = False) -> None:
    if make_parent:
        path.parent.mkdir(exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
