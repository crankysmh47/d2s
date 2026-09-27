"""Authenticated V3 replacement controls.

Reviewer-session records are process provenance, not cryptographic attestations.
Output checks reparse status and containment around directory creation; a
concurrent path mutation between those checks remains a residual race.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import random
from typing import Mapping
from uuid import UUID

from . import answer_v2, answer_v3, audit_v2, study_v2
from .data import Example, interpret
from .provenance import canonical_json, sha256_payload


SELECTOR_SEED = 927002
EXPECTED_PYTHON = "3.11.9"
FAMILY_ORDER = ("expressions", "lists", "graphs")
INCIDENT = "original_30_accepted_controls_retired_after_blinding_boundary_breach"
_REVIEW_FIELDS = {
    "audit_id", "family", "status", "claimed_value", "evidence_span", "reason",
    "reviewer_id", "explanation_status", "explanation_note",
}
_GRAPH_EXPLANATION_STATUSES = {"valid", "invalid", "indeterminate"}


class _AttemptReservationConflict(FileExistsError):
    """The durable attempt file already belongs to another reconciliation call."""


def export_replacement_controls(
    run_root: Path,
    original_audit_root: Path,
    output_root: Path,
) -> dict[str, str]:
    """Authenticate V2 and its retired audit, then freeze and export replacement packets.

    The original audit's source mapping is not inspected until V2 stages, split,
    score and the original packet manifest/files have all passed authentication.
    """
    source_root = Path(run_root).resolve(strict=True)
    source = audit_v2.verify_v2_audit_source(source_root)
    original_root = _safe_external_root(Path(original_audit_root), source_root, "original audit")
    original_manifest_raw = audit_v2._read_audit_file(
        original_root, Path("manifest.json"), "V2 original audit manifest",
    )
    original_manifest = audit_v2._parse_object(original_manifest_raw, "V2 original audit manifest")
    if canonical_json(original_manifest) != original_manifest_raw:
        raise ValueError("V2 original audit manifest bytes are not canonical")
    authenticated_manifest, _packets, _source_ids = audit_v2._authenticate_audit_export(
        source, original_root, original_manifest,
    )
    # The existing V2 authenticator binds identity, source/stage hashes, packet bytes and counts.
    # Derive exclusions only after it returns successfully.
    selection = _select_from_authenticated_source(source, authenticated_manifest)
    _ensure_output_external(Path(output_root), source_root, original_root)
    return _export_authenticated_selection(
        source, authenticated_manifest, selection, Path(output_root),
        original_manifest_sha256=hashlib.sha256(original_manifest_raw).hexdigest(),
        source_root=source_root,
        original_root=original_root,
    )


def reconcile_replacement_controls(
    run_root: Path,
    original_audit_root: Path,
    controls_root: Path,
) -> dict[str, object]:
    """Run one reconciliation attempt and persist a terminal record if it stops."""
    controls_path = Path(controls_root)
    if (controls_path / "attempt.json").exists() or (controls_path / "qualification.json").exists():
        raise FileExistsError("replacement-control reconciliation was already reserved or completed")
    try:
        return _run_reconciliation_once(run_root, original_audit_root, controls_path)
    except _AttemptReservationConflict as error:
        # Never let a losing concurrent caller write a stopped record over the owner.
        raise FileExistsError("another replacement-control attempt owns this run") from error
    except Exception as error:
        _write_stopped_qualification(controls_path, error)
        raise


def _run_reconciliation_once(
    run_root: Path,
    original_audit_root: Path,
    controls_path: Path,
) -> dict[str, object]:
    """Validate three locked blind reviews, run V3 once per row, and write qualification.

    The qualification file is exclusive. Existing output means a prior one-shot
    reconciliation exists and this scorer pass must not be repeated.
    """
    if (controls_path / "attempt.json").exists() or (controls_path / "qualification.json").exists():
        raise _AttemptReservationConflict("another replacement-control attempt owns this run")
    source_root = Path(run_root).resolve(strict=True)
    source = audit_v2.verify_v2_audit_source(source_root)
    original_root = _safe_external_root(Path(original_audit_root), source_root, "original audit")
    original_manifest, original_manifest_sha256 = _read_authenticated_original_manifest(source, original_root)
    selection = _select_from_authenticated_source(source, original_manifest)
    private, packets = _authenticate_replacement_export(
        source, original_manifest, original_manifest_sha256, selection, controls_path,
    )
    review_rows, reviewer_ids, review_hashes, session_hash = _read_and_validate_reviews(controls_path, packets)
    frozen_code_hashes = _frozen_code_hashes(private["frozen_code_sha256"])
    if frozen_code_hashes != private["frozen_code_sha256"]:
        raise ValueError("frozen scorer, selector or test hashes changed after packet export")

    attempt_payload: dict[str, object] = {
        "schema_version": 1,
        "status": "reserved",
        "replacement_audit_id": private["audit_id"],
        "source_hashes": source["hashes"],
        "stage_hashes": source["stage_hashes"],
        "private_manifest_sha256": private["manifest_sha256"],
        "review_sessions_sha256": session_hash,
        "review_sha256": review_hashes,
        "frozen_code_sha256": frozen_code_hashes,
    }
    attempt_raw = _reserve_reconciliation_attempt(controls_path, attempt_payload)
    attempt_hash = hashlib.sha256(attempt_raw).hexdigest()

    dev_by_id = {str(row["id"]): row for row in _mapping_rows(source.get("dev_rows"), "authenticated V2 dev rows")}
    scores_by_id = {
        str(row["example_id"]): row for row in _mapping_rows(source.get("score_rows"), "authenticated V2 score rows")
    }
    # Preflight citations for every row before invoking the full-response scorer once.
    for packet in packets:
        source_id = private["source_ids"][packet["audit_id"]]
        example = _example_from_authenticated_row(dev_by_id[source_id])
        for review in review_rows[packet["audit_id"]]:
            _validate_evidence_candidate(review, example)

    result_rows: list[dict[str, object]] = []
    discrepancies: list[str] = []
    for packet in packets:
        audit_id = packet["audit_id"]
        source_id = private["source_ids"][audit_id]
        dev = dev_by_id[source_id]
        v2_score = scores_by_id[source_id]
        example = _example_from_authenticated_row(dev)
        reviews = review_rows[audit_id]
        unanimous_values = [_normalize_value(row["claimed_value"], packet["family"]) for row in reviews]
        if len(set(_canonical_value(value) for value in unanimous_values)) != 1:
            discrepancies.append(f"{audit_id}: blind reviewers disagree on extracted value")
        luna_value = unanimous_values[0]
        score = answer_v3.score_completion_v3(example, packet["response"], packet["stop_reason"])
        v3_value = _normalize_value(score.get("parsed_value"), packet["family"], allow_none=True)
        v2_value = _normalize_value(v2_score.get("parsed_value"), packet["family"])
        expected = interpret(example.latent_spec, example.family)
        v2_reason = v2_score.get("reason_code")
        v3_reason = score.get("reason_code")
        expected_value = _normalize_value(expected, example.family)
        expected_reason = "correct" if v2_value == expected_value else "wrong_value"
        if v2_reason not in {"correct", "wrong_value"} or v2_reason != expected_reason:
            discrepancies.append(f"{audit_id}: authenticated V2 label is inconsistent with its parsed value")
        if v2_score.get("correct") is not (v2_reason == "correct"):
            discrepancies.append(f"{audit_id}: authenticated V2 correctness flag and reason disagree")
        if score.get("correct") is not (v3_reason == "correct"):
            discrepancies.append(f"{audit_id}: V3 correctness flag and reason disagree")
        if score.get("correct") is not v2_score.get("correct"):
            discrepancies.append(f"{audit_id}: V3 correctness decision differs from V2 correctness flag")
        if v3_reason != v2_reason:
            discrepancies.append(f"{audit_id}: V3 correctness decision differs from V2")
        if _canonical_value(v3_value) != _canonical_value(v2_value):
            discrepancies.append(f"{audit_id}: V3 parsed value differs from V2")
        if any(_canonical_value(value) != _canonical_value(v2_value) for value in unanimous_values):
            discrepancies.append(f"{audit_id}: unanimous blind extraction differs from V2 parsed value")
        result_rows.append({
            "audit_id": audit_id,
            "source_id": source_id,
            "family": packet["family"],
            "v2_reason_code": v2_reason,
            "v2_parsed_value": v2_value,
            "v3_reason_code": v3_reason,
            "v3_parsed_value": v3_value,
            "unanimous_luna_value": luna_value,
            "review_values": unanimous_values,
            "graph_explanation_statuses": [row["explanation_status"] for row in reviews]
            if packet["family"] == "graphs" else ["not_applicable"] * 3,
        })
    failure_detail = "; ".join(discrepancies) if discrepancies else None

    if _frozen_code_hashes(frozen_code_hashes) != frozen_code_hashes:
        raise ValueError("frozen scorer, selector or test hashes changed during the one-shot scorer pass")
    _verify_review_files_unchanged(controls_path, review_hashes)
    _verify_packet_files_unchanged(
        controls_path, private["packet_sha256"], Path(str(private["reviewer_packet_root"])),
    )
    _verify_review_sessions_unchanged(controls_path, session_hash)
    scorer_path = Path(answer_v3.__file__).resolve(strict=True)
    tests_path = Path(__file__).resolve().parents[3] / "tests" / "test_recurrence_audit_v3_controls.py"
    qualification_payload: dict[str, object] = {
        "schema_version": 1,
        "status": "qualified" if failure_detail is None else "stopped",
        "incident": INCIDENT,
        "run_id": source["run_id"],
        "replacement_audit_id": private["audit_id"],
        "source_hashes": source["hashes"],
        "stage_hashes": source["stage_hashes"],
        "original_audit_id": original_manifest["audit_id"],
        "original_manifest_sha256": original_manifest_sha256,
        "original_packet_sha256": original_manifest["packet_sha256"],
        "selector_sha256": private["selector_sha256"],
        "frozen_code_sha256": frozen_code_hashes,
        "selector_module_sha256": _sha256_file(Path(__file__)),
        "scorer_sha256": _sha256_file(scorer_path),
        "test_sha256": _sha256_file(tests_path),
        "private_manifest_sha256": private["manifest_sha256"],
        "packet_sha256": private["packet_sha256"],
        "packet_counts": private["packet_counts"],
        "review_sha256": review_hashes,
        "review_sessions_sha256": session_hash,
        "review_session_provenance": "process provenance; not cryptographic attestation",
        "reviewer_ids": reviewer_ids,
        "attempt_sha256": attempt_hash,
        "row_count": len(result_rows),
        "rows": result_rows,
    }
    if failure_detail is not None:
        qualification_payload["failure_detail"] = failure_detail
    qualification_raw = _with_content_hash(qualification_payload)
    _write_exclusive(controls_path / "qualification.json", qualification_raw)
    return {
        "status": str(qualification_payload["status"]),
        "replacement_audit_id": private["audit_id"],
        "qualification_path": str(controls_path / "qualification.json"),
        "row_count": len(result_rows),
    }


def _select_from_authenticated_source(
    source: Mapping[str, object], original_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Build the frozen selection record from already authenticated V2 evidence."""
    _require_python_version()
    dev_rows = _mapping_rows(source.get("dev_rows"), "authenticated V2 dev rows")
    score_rows = _mapping_rows(source.get("score_rows"), "authenticated V2 score rows")
    if len(dev_rows) != 300 or len(score_rows) != 300:
        raise ValueError("replacement selection requires exactly 300 authenticated development rows")
    score_by_id: dict[str, Mapping[str, object]] = {}
    for score in score_rows:
        identifier = _nonempty_string(score.get("example_id"), "V2 score example ID")
        if identifier in score_by_id:
            raise ValueError("authenticated V2 scores contain duplicate IDs")
        score_by_id[identifier] = score
    dev_by_id: dict[str, Mapping[str, object]] = {}
    for row in dev_rows:
        identifier = _nonempty_string(row.get("id"), "V2 development ID")
        if identifier in dev_by_id:
            raise ValueError("authenticated V2 development rows contain duplicate IDs")
        if identifier not in score_by_id or score_by_id[identifier].get("family") != row.get("family"):
            raise ValueError("authenticated V2 score rows do not match development rows")
        if row.get("split") != "dev" or row.get("family") not in FAMILY_ORDER:
            raise ValueError("authenticated V2 development row has an invalid split or family")
        dev_by_id[identifier] = row
    if set(dev_by_id) != set(score_by_id):
        raise ValueError("authenticated V2 score IDs differ from development IDs")

    old_source_ids = _original_accepted_source_ids(original_manifest)
    eligible_by_family: dict[str, list[Mapping[str, object]]] = {}
    eligible_count: dict[str, int] = {}
    eligible_hash: dict[str, str] = {}
    for family in FAMILY_ORDER:
        pool = sorted(
            (
                row for row in dev_rows
                if row["family"] == family
                and row["id"] not in old_source_ids
                and score_by_id[str(row["id"])].get("reason_code") in {"correct", "wrong_value"}
            ),
            key=lambda row: str(row["id"]),
        )
        if len(pool) < 10:
            raise ValueError(f"fewer than ten eligible replacement rows for {family}")
        eligible_by_family[family] = pool
        ordered_ids = [str(row["id"]) for row in pool]
        eligible_count[family] = len(pool)
        eligible_hash[family] = hashlib.sha256(canonical_json(ordered_ids)).hexdigest()

    rng = random.Random(SELECTOR_SEED)
    selected_rows: dict[str, list[Mapping[str, object]]] = {}
    selected_ids: dict[str, list[str]] = {}
    seen: set[str] = set()
    for family in FAMILY_ORDER:
        picked = rng.sample(eligible_by_family[family], 10)
        ids = [str(row["id"]) for row in picked]
        if len(ids) != 10 or len(set(ids)) != 10 or seen.intersection(ids):
            raise ValueError("replacement selector produced duplicate or overlapping IDs")
        if set(ids).intersection(old_source_ids):
            raise ValueError("replacement selector overlaps retired accepted controls")
        seen.update(ids)
        selected_rows[family] = picked
        selected_ids[family] = ids
    if len(seen) != 30:
        raise ValueError("replacement selector must produce exactly thirty unique rows")
    return {
        "seed": SELECTOR_SEED,
        "python_version": platform.python_version(),
        "selector_sha256": _sha256_file(Path(__file__)),
        "eligible_count": eligible_count,
        "eligible_ids_sha256": eligible_hash,
        "selected_ids": selected_ids,
        "selected_rows": selected_rows,
        "old_accepted_source_ids": sorted(old_source_ids),
    }


def _export_authenticated_selection(
    source: Mapping[str, object],
    original_manifest: Mapping[str, object],
    selection: Mapping[str, object],
    output_root: Path,
    *,
    original_manifest_sha256: str,
    source_root: Path,
    original_root: Path,
) -> dict[str, str]:
    dev_by_id = {str(row["id"]): row for row in _mapping_rows(source["dev_rows"], "authenticated V2 dev rows")}
    score_by_id = {str(row["example_id"]): row for row in _mapping_rows(source["score_rows"], "authenticated V2 score rows")}
    packet_groups: dict[str, list[dict[str, str]]] = {family: [] for family in FAMILY_ORDER}
    source_ids: dict[str, str] = {}
    v2_labels: dict[str, dict[str, object]] = {}
    for family in FAMILY_ORDER:
        for source_id in selection["selected_ids"][family]:
            row, score = dev_by_id[source_id], score_by_id[source_id]
            question, response, stop_reason = row.get("prompt"), score.get("completion_text"), score.get("stop_reason")
            if not all(isinstance(value, str) for value in (question, response, stop_reason)):
                raise ValueError("selected V2 question, response and stop reason must be text")
            packet_id = hashlib.sha256(f"v3-replacement:{source_id}".encode("utf-8")).hexdigest()[:16]
            if packet_id in source_ids:
                raise ValueError("replacement packet ID collision")
            source_ids[packet_id] = source_id
            packet_groups[family].append({
                "audit_id": packet_id,
                "family": family,
                "question": question,
                "response": response,
                "stop_reason": stop_reason,
            })
            v2_labels[packet_id] = {
                "reason_code": score.get("reason_code"),
                "parsed_value": score.get("parsed_value"),
                "correct": score.get("correct"),
            }
    packet_bytes = {
        family: b"".join(canonical_json(packet) for packet in packet_groups[family])
        for family in FAMILY_ORDER
    }
    packet_hashes = {family: hashlib.sha256(packet_bytes[family]).hexdigest() for family in FAMILY_ORDER}
    packet_counts = {family: len(packet_groups[family]) for family in FAMILY_ORDER}
    payload: dict[str, object] = {
        "schema_version": 1,
        "incident": INCIDENT,
        "run_id": source["run_id"],
        "source_hashes": source["hashes"],
        "stage_hashes": source["stage_hashes"],
        "original_audit_id": original_manifest["audit_id"],
        "original_manifest_sha256": original_manifest_sha256,
        "original_packet_sha256": original_manifest["packet_sha256"],
        "seed": selection["seed"],
        "python_version": selection["python_version"],
        "selector_sha256": selection["selector_sha256"],
        "frozen_code_sha256": _frozen_code_hashes(),
        "eligible_count": selection["eligible_count"],
        "eligible_ids_sha256": selection["eligible_ids_sha256"],
        "selected_ids": selection["selected_ids"],
        "old_accepted_source_ids": selection["old_accepted_source_ids"],
        "source_ids": source_ids,
        "v2_labels": v2_labels,
        "packet_sha256": packet_hashes,
        "packet_counts": packet_counts,
    }
    replacement_audit_id = sha256_payload(payload)
    payload["audit_id"] = replacement_audit_id
    private_raw = _with_content_hash(payload)
    base = Path(output_root).resolve(strict=False)
    _ensure_output_external(base, source_root, original_root)
    base.mkdir(parents=True, exist_ok=True)
    _ensure_output_external(base, source_root, original_root)
    private_destination = base / replacement_audit_id
    public_parent = base / "reviewer-packets"
    public_destination = public_parent / replacement_audit_id
    _ensure_output_external(private_destination, source_root, original_root)
    _assert_disjoint_roots(private_destination, public_destination)
    _assert_no_reparse(private_destination)
    private_destination.mkdir(exist_ok=False)
    _ensure_output_external(private_destination, source_root, original_root)
    _assert_no_reparse(private_destination)
    _ensure_output_external(public_parent, source_root, original_root)
    _assert_no_reparse(public_parent)
    public_parent.mkdir(parents=True, exist_ok=True)
    _ensure_output_external(public_parent, source_root, original_root)
    _assert_no_reparse(public_parent)
    _ensure_output_external(public_destination, source_root, original_root)
    _assert_no_reparse(public_destination)
    _assert_disjoint_roots(private_destination, public_destination)
    public_destination.mkdir(exist_ok=False)
    _ensure_output_external(public_destination, source_root, original_root)
    _assert_no_reparse(public_destination)
    _assert_disjoint_roots(private_destination, public_destination)
    payload["reviewer_packet_root"] = str(public_destination.resolve(strict=True))
    # The manifest self-hash binds this path; the stable audit ID avoids a path-hash cycle.
    private_raw = _with_content_hash(payload)
    destination = base / replacement_audit_id
    try:
        _write_exclusive(destination / "private-manifest.json", private_raw)
        for family in FAMILY_ORDER:
            _write_exclusive(destination / "packets" / f"{family}.jsonl", packet_bytes[family], make_parent=True)
            _write_exclusive(public_destination / f"{family}.jsonl", packet_bytes[family])
    except Exception:
        # Preserve partial evidence; the destination stays occupied and cannot be silently reused.
        raise
    private = json.loads(private_raw)
    return {
        "audit_id": replacement_audit_id,
        "controls_root": str(destination),
        "reviewer_packet_root": str(public_destination),
        "private_manifest_sha256": private["manifest_sha256"],
        "selector_sha256": str(selection["selector_sha256"]),
    }


def _authenticate_replacement_export(
    source: Mapping[str, object],
    original_manifest: Mapping[str, object],
    original_manifest_sha256: str,
    selection: Mapping[str, object],
    controls_root: Path,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    root = _safe_controls_root(controls_root)
    if (root / "qualification.json").exists():
        raise FileExistsError("replacement-control qualification already exists")
    raw = _read_regular(root, Path("private-manifest.json"), "replacement private manifest")
    private = audit_v2._parse_object(raw, "replacement private manifest")
    if canonical_json(private) != raw:
        raise ValueError("replacement private manifest is not canonical")
    authenticated = _verify_content_hash(private, "replacement private manifest")
    audit_identity_payload = {
        key: value for key, value in authenticated.items()
        if key not in {"audit_id", "reviewer_packet_root"}
    }
    if authenticated.get("audit_id") != sha256_payload(audit_identity_payload):
        raise ValueError("replacement audit identity does not match its private manifest")
    captured_code_hashes = authenticated.get("frozen_code_sha256")
    if not isinstance(captured_code_hashes, Mapping) or _frozen_code_hashes(captured_code_hashes) != captured_code_hashes:
        raise ValueError("frozen scorer, selector or test code hash differs from the exported private manifest")
    expected_selected: dict[str, list[str]] = selection["selected_ids"]
    expected_public_root = root.parent / "reviewer-packets" / root.name
    public_root = _safe_public_packet_root(authenticated.get("reviewer_packet_root"), root)
    if public_root != expected_public_root.resolve(strict=True):
        raise ValueError("reviewer-facing packet root is not the expected separate sibling root")
    _assert_disjoint_roots(root, public_root)
    public_names = {path.name for path in public_root.iterdir()}
    if public_names != {f"{family}.jsonl" for family in FAMILY_ORDER}:
        raise ValueError("reviewer-facing packet root contains unexpected files")
    if (
        authenticated.get("incident") != INCIDENT
        or authenticated.get("run_id") != source.get("run_id")
        or authenticated.get("source_hashes") != source.get("hashes")
        or authenticated.get("stage_hashes") != source.get("stage_hashes")
        or authenticated.get("original_audit_id") != original_manifest.get("audit_id")
        or authenticated.get("original_manifest_sha256") != original_manifest_sha256
        or authenticated.get("original_packet_sha256") != original_manifest.get("packet_sha256")
        or authenticated.get("seed") != SELECTOR_SEED
        or authenticated.get("python_version") != EXPECTED_PYTHON
        or authenticated.get("selector_sha256") != selection.get("selector_sha256")
            or authenticated.get("eligible_count") != selection.get("eligible_count")
        or authenticated.get("eligible_ids_sha256") != selection.get("eligible_ids_sha256")
        or authenticated.get("selected_ids") != expected_selected
        or root.name != authenticated.get("audit_id")
    ):
        raise ValueError("replacement private manifest differs from authenticated deterministic selection")
    expected_packets, expected_source_ids, expected_v2_labels = _expected_packets(source, selection)
    if authenticated.get("source_ids") != expected_source_ids or authenticated.get("v2_labels") != expected_v2_labels:
        raise ValueError("replacement private source mapping or V2 labels differ from authenticated rows")
    packets: list[dict[str, str]] = []
    expected_hashes: dict[str, str] = {}
    expected_counts: dict[str, int] = {}
    for family in FAMILY_ORDER:
        raw_packets = b"".join(canonical_json(packet) for packet in expected_packets[family])
        packet_raw = _read_regular(root, Path("packets") / f"{family}.jsonl", f"{family} replacement packets")
        public_raw = _read_regular(public_root, Path(f"{family}.jsonl"), f"public {family} replacement packets")
        if packet_raw != raw_packets:
            raise ValueError(f"{family} replacement packet bytes differ from authenticated selection")
        if public_raw != raw_packets:
            raise ValueError(f"public {family} packet bytes differ from authenticated gold-free selection")
        packet_rows = audit_v2._parse_jsonl(packet_raw, f"{family} replacement packets")
        if len(packet_rows) != 10:
            raise ValueError(f"{family} replacement packet count is not ten")
        if any(set(row) != {"audit_id", "family", "question", "response", "stop_reason"} for row in packet_rows):
            raise ValueError(f"{family} replacement packet has non-blind fields")
        packet_hash = hashlib.sha256(packet_raw).hexdigest()
        if authenticated.get("packet_sha256", {}).get(family) != packet_hash:
            raise ValueError(f"{family} replacement packet hash mismatch")
        expected_hashes[family] = packet_hash
        expected_counts[family] = len(packet_rows)
        packets.extend(packet_rows)  # Fixed family order, with the sampled order preserved.
    if authenticated.get("packet_counts") != expected_counts:
        raise ValueError("replacement packet counts differ from the private manifest")
    return {**authenticated, "manifest_sha256": private["manifest_sha256"]}, packets


def _expected_packets(
    source: Mapping[str, object], selection: Mapping[str, object],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str], dict[str, dict[str, object]]]:
    dev_by_id = {str(row["id"]): row for row in _mapping_rows(source["dev_rows"], "authenticated V2 dev rows")}
    scores_by_id = {
        str(row["example_id"]): row for row in _mapping_rows(source["score_rows"], "authenticated V2 score rows")
    }
    groups: dict[str, list[dict[str, str]]] = {family: [] for family in FAMILY_ORDER}
    source_ids: dict[str, str] = {}
    labels: dict[str, dict[str, object]] = {}
    for family in FAMILY_ORDER:
        for source_id in selection["selected_ids"][family]:
            row, score = dev_by_id[source_id], scores_by_id[source_id]
            packet_id = hashlib.sha256(f"v3-replacement:{source_id}".encode("utf-8")).hexdigest()[:16]
            question, response, stop_reason = row.get("prompt"), score.get("completion_text"), score.get("stop_reason")
            if not all(isinstance(value, str) for value in (question, response, stop_reason)):
                raise ValueError("selected V2 question, response and stop reason must be text")
            groups[family].append({
                "audit_id": packet_id,
                "family": family,
                "question": question,
                "response": response,
                "stop_reason": stop_reason,
            })
            source_ids[packet_id] = source_id
            labels[packet_id] = {
                "reason_code": score.get("reason_code"),
                "parsed_value": score.get("parsed_value"),
                "correct": score.get("correct"),
            }
    return groups, source_ids, labels


def _read_and_validate_reviews(
    controls_root: Path, packets: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, object]]], list[str], dict[str, str], str]:
    session_raw = _read_regular(controls_root, Path("review-sessions.json"), "root-attested review sessions")
    session = audit_v2._parse_object(session_raw, "root-attested review sessions")
    if canonical_json(session) != session_raw or set(session) != {"schema_version", "reviewers"}:
        raise ValueError("review-sessions.json must be canonical and have the registered schema")
    session_rows = session.get("reviewers")
    expected_tasks = (
        "/root/v3_replacement_luna_1",
        "/root/v3_replacement_luna_2",
        "/root/v3_replacement_luna_3",
    )
    if session.get("schema_version") != 1 or not isinstance(session_rows, list) or len(session_rows) != 3:
        raise ValueError("review-sessions.json must register exactly three Luna sessions")
    session_by_task: dict[str, Mapping[str, object]] = {}
    session_ids: set[str] = set()
    reviewer_ids_registered: set[str] = set()
    for row in session_rows:
        required = {
            "task", "model", "root_attested_session_id", "review_file", "review_sha256", "reviewer_id",
        }
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("review session row has unexpected or missing fields")
        raw_task = row.get("task")
        if not isinstance(raw_task, str):
            raise ValueError("review session task must be text")
        task = raw_task.casefold()
        if task not in expected_tasks or task in session_by_task:
            raise ValueError("review sessions must use the three canonical collaboration tasks")
        if row.get("model") != "gpt-6-luna":
            raise ValueError("review session model must be exactly gpt-6-luna")
        session_id = row.get("root_attested_session_id")
        if not isinstance(session_id, str):
            raise ValueError("root-attested session ID must be a UUID")
        try:
            parsed_session_id = UUID(session_id)
        except (ValueError, AttributeError) as error:
            raise ValueError("root-attested session ID must be a UUID") from error
        if str(parsed_session_id) != session_id.casefold() or session_id.casefold() in session_ids:
            raise ValueError("root-attested session IDs must be canonical and unique")
        session_ids.add(session_id.casefold())
        registered_identity = f"gpt-6-luna:{task}"
        if row.get("reviewer_id") != registered_identity or registered_identity in reviewer_ids_registered:
            raise ValueError("review session identity does not match its canonical task")
        reviewer_ids_registered.add(registered_identity)
        review_file = row.get("review_file")
        if not isinstance(review_file, str) or not review_file.startswith("reviews/"):
            raise ValueError("review file must be relative to the private reviews directory")
        relative_review = Path(review_file)
        if relative_review.is_absolute() or ".." in relative_review.parts or len(relative_review.parts) != 2:
            raise ValueError("review file path is unsafe")
        if relative_review.suffix != ".jsonl" or relative_review.name != f"{relative_review.stem}.jsonl":
            raise ValueError("review file must be a JSONL file")
        digest = row.get("review_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("review session must bind a lowercase SHA-256")
        session_by_task[task] = {**row, "task": task, "review_file": relative_review.as_posix()}
    if set(session_by_task) != set(expected_tasks):
        raise ValueError("review sessions are missing a canonical collaboration task")

    review_dir = controls_root / "reviews"
    try:
        study_v2._assert_no_reparse(review_dir)
        resolved = review_dir.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"replacement review directory is unsafe or missing: {error}") from error
    if not resolved.is_dir():
        raise ValueError("replacement reviews must be stored in a directory")
    expected_files = {Path(row["review_file"]).name for row in session_by_task.values()}
    paths = sorted(resolved.glob("*.jsonl"), key=lambda path: path.name)
    if {path.name for path in paths} != expected_files:
        raise ValueError("review files differ from root-attested session registrations")
    expected_ids = [packet["audit_id"] for packet in packets]
    packet_by_id = {packet["audit_id"]: packet for packet in packets}
    validated_by_reviewer: list[list[dict[str, object]]] = []
    reviewer_ids: list[str] = []
    review_hashes: dict[str, str] = {}
    assigned_reviewers: set[str] = set()
    review_meta_by_file = {Path(row["review_file"]).name: (task, row) for task, row in session_by_task.items()}
    paths = sorted(paths, key=lambda path: expected_tasks.index(review_meta_by_file[path.name][0]))
    session_reviewer_ids: list[str] = []
    for path in paths:
        raw = _read_regular(resolved, Path(path.name), "replacement blind review")
        task, session_row = review_meta_by_file[path.name]
        relative_name = f"reviews/{path.name}"
        if hashlib.sha256(raw).hexdigest() != session_row["review_sha256"]:
            raise ValueError("review bytes do not match their root-attested session hash")
        rows = audit_v2._parse_jsonl(raw, "replacement blind review")
        if len(rows) != 30:
            raise ValueError("each Luna reviewer must review all thirty replacement rows")
        if [row.get("audit_id") for row in rows] != expected_ids:
            raise ValueError("review packet IDs are missing, duplicated, extra or reordered")
        normalized_rows: list[dict[str, object]] = []
        current_reviewer: str | None = None
        for line_number, row in enumerate(rows, start=1):
            if set(row) != _REVIEW_FIELDS:
                raise ValueError(f"review row {line_number} has unexpected or missing fields")
            packet = packet_by_id[row["audit_id"]]
            if row.get("family") != packet["family"]:
                raise ValueError("review family does not match its gold-free packet")
            if row.get("status") != "conclusive":
                raise ValueError("every replacement review must be conclusive")
            reviewer_id = row.get("reviewer_id")
            if not isinstance(reviewer_id, str) or not reviewer_id.strip():
                raise ValueError("reviewer identity is missing")
            registered_identity = f"gpt-6-luna:{task}"
            if reviewer_id != registered_identity:
                raise ValueError("review row identity differs from its root-attested session")
            current_reviewer = reviewer_id
            if normalized_rows and reviewer_id != normalized_rows[0]["reviewer_id"]:
                raise ValueError("a reviewer identity changed within one review file")
            family = packet["family"]
            normalized_value = _normalize_value(row.get("claimed_value"), family)
            evidence = row.get("evidence_span")
            reason = row.get("reason")
            if not isinstance(evidence, str) or not evidence or len(evidence) > 1000 or evidence not in packet["response"]:
                raise ValueError("review evidence must be a response substring of at most 1000 characters")
            if not isinstance(reason, str) or len(reason) > 500:
                raise ValueError("review reason must be short text")
            explanation_status, explanation_note = row.get("explanation_status"), row.get("explanation_note")
            if family == "graphs":
                if (
                    explanation_status not in _GRAPH_EXPLANATION_STATUSES
                    or not isinstance(explanation_note, str)
                    or not explanation_note.strip()
                    or len(explanation_note) > 500
                ):
                    raise ValueError("graph explanation status and note must be recorded separately")
            elif explanation_status != "not_applicable" or explanation_note != "":
                raise ValueError("non-graph review must mark explanation as not_applicable")
            normalized_rows.append({**row, "claimed_value": normalized_value})
        assert current_reviewer is not None
        if current_reviewer in assigned_reviewers:
            raise ValueError("the three review sessions must use distinct Luna reviewers")
        assigned_reviewers.add(current_reviewer)
        reviewer_ids.append(current_reviewer)
        session_reviewer_ids.append(str(session_row["reviewer_id"]))
        validated_by_reviewer.append(normalized_rows)
        review_hashes[relative_name] = hashlib.sha256(raw).hexdigest()
    by_audit_id: dict[str, list[dict[str, object]]] = {}
    for row_index, packet in enumerate(packets):
        reviews = [reviewer_rows[row_index] for reviewer_rows in validated_by_reviewer]
        values = [_canonical_value(row["claimed_value"]) for row in reviews]
        if len(set(values)) != 1:
            raise ValueError(f"three Luna reviewers do not unanimously agree for {packet['audit_id']}")
        by_audit_id[packet["audit_id"]] = reviews
    if reviewer_ids != session_reviewer_ids:
        raise ValueError("review files are not in canonical collaboration-task order")
    return by_audit_id, reviewer_ids, review_hashes, hashlib.sha256(session_raw).hexdigest()


def _verify_review_files_unchanged(controls_root: Path, review_hashes: Mapping[str, str]) -> None:
    for name, expected_hash in review_hashes.items():
        raw = _read_regular(controls_root, Path(name), "replacement blind review")
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError("blind review bytes changed during one-shot reconciliation")


def _verify_review_sessions_unchanged(controls_root: Path, expected_hash: str) -> None:
    raw = _read_regular(controls_root, Path("review-sessions.json"), "root-attested review sessions")
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("root-attested review session manifest changed during one-shot reconciliation")


def _verify_packet_files_unchanged(
    controls_root: Path, expected_hashes: Mapping[str, str], public_root: Path,
) -> None:
    for family in FAMILY_ORDER:
        raw = _read_regular(controls_root, Path("packets") / f"{family}.jsonl", f"{family} replacement packets")
        if hashlib.sha256(raw).hexdigest() != expected_hashes.get(family):
            raise ValueError("replacement packet bytes changed during one-shot reconciliation")
        public_raw = _read_regular(public_root, Path(f"{family}.jsonl"), f"public {family} replacement packets")
        if public_raw != raw:
            raise ValueError("public replacement packet bytes changed during one-shot reconciliation")


def _write_stopped_qualification(controls_root: Path, error: Exception) -> None:
    """Lock failed review/provenance attempts so the V3 scorer cannot be silently rerun."""
    destination = controls_root / "qualification.json"
    if destination.exists():
        return
    try:
        root = _safe_controls_root(controls_root)
        if (root / "qualification.json").exists():
            return
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": "stopped",
            "incident": INCIDENT,
            "controls_id": root.name,
            "failure_type": type(error).__name__,
            "failure_detail": str(error),
            "selector_module_sha256": _sha256_file(Path(__file__)),
            "scorer_sha256": _sha256_file(Path(answer_v3.__file__).resolve(strict=True)),
            "available_input_sha256": _available_control_input_hashes(root),
        }
        test_path = Path(__file__).resolve().parents[3] / "tests" / "test_recurrence_audit_v3_controls.py"
        if test_path.is_file():
            payload["test_sha256"] = _sha256_file(test_path)
        _write_exclusive(root / "qualification.json", _with_content_hash(payload))
    except (OSError, ValueError, FileExistsError):
        # The initial failure remains authoritative if a safe terminal write is impossible.
        return


def _available_control_input_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in (
        Path("private-manifest.json"),
        *(Path("packets") / f"{family}.jsonl" for family in FAMILY_ORDER),
    ):
        try:
            raw = _read_regular(root, relative, "available replacement input")
        except ValueError:
            continue
        hashes[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
    review_root = root / "reviews"
    try:
        study_v2._assert_no_reparse(review_root)
        resolved = review_root.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return hashes
    if resolved.is_dir():
        for path in sorted(resolved.glob("*.jsonl"), key=lambda item: item.name):
            try:
                raw = _read_regular(resolved, Path(path.name), "available replacement review")
            except ValueError:
                continue
            hashes[f"reviews/{path.name}"] = hashlib.sha256(raw).hexdigest()
    return hashes


def _read_authenticated_original_manifest(
    source: Mapping[str, object], original_root: Path,
) -> tuple[dict[str, object], str]:
    raw = audit_v2._read_audit_file(original_root, Path("manifest.json"), "V2 original audit manifest")
    manifest = audit_v2._parse_object(raw, "V2 original audit manifest")
    if canonical_json(manifest) != raw:
        raise ValueError("V2 original audit manifest bytes are not canonical")
    authenticated, _packets, _source_ids = audit_v2._authenticate_audit_export(source, original_root, manifest)
    return authenticated, hashlib.sha256(raw).hexdigest()


def _original_accepted_source_ids(manifest: Mapping[str, object]) -> set[str]:
    source_ids = manifest.get("source_ids")
    reasons = manifest.get("selection_reasons")
    if not isinstance(source_ids, Mapping) or not isinstance(reasons, Mapping):
        raise ValueError("authenticated original manifest lacks private source mappings")
    if set(source_ids) != set(reasons):
        raise ValueError("authenticated original manifest source IDs and reasons differ")
    by_family: dict[str, int] = {family: 0 for family in FAMILY_ORDER}
    accepted: set[str] = set()
    for audit_id, source_id in source_ids.items():
        reason = reasons[audit_id]
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("original manifest source ID is invalid")
        if reason in {"correct", "wrong_value"}:
            accepted.add(source_id)
            # Family is authenticated by the original V2 packet list; infer from the ID only nowhere.
    audit_ids = manifest.get("audit_ids")
    if not isinstance(audit_ids, Mapping) or set(audit_ids) != set(FAMILY_ORDER):
        raise ValueError("original manifest does not contain all registered families")
    for family in FAMILY_ORDER:
        family_ids = audit_ids[family]
        if not isinstance(family_ids, list) or any(not isinstance(identifier, str) for identifier in family_ids):
            raise ValueError("original manifest family IDs are invalid")
        family_accepted = [identifier for identifier in family_ids if reasons.get(identifier) in {"correct", "wrong_value"}]
        if len(family_accepted) != 10:
            raise ValueError("original manifest must contain exactly ten accepted controls per family")
        by_family[family] = len(family_accepted)
    if sum(len(audit_ids[family]) for family in FAMILY_ORDER) != len(source_ids):
        raise ValueError("original manifest packet ID count differs from its private mapping")
    if len(accepted) != 30 or len(set(source_ids.values())) != len(source_ids):
        raise ValueError("original accepted controls must map to thirty distinct development IDs")
    return accepted


def _normalize_value(value: object, family: str, *, allow_none: bool = False) -> object:
    if value is None and allow_none:
        return None
    if family == "expressions":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("expression extraction must be an integer")
        return value
    if family == "lists":
        if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("list extraction must be an integer vector")
        return list(value)
    if family == "graphs":
        if not isinstance(value, str) or value.casefold() not in {"yes", "no"}:
            raise ValueError("graph extraction must be yes or no")
        return value.casefold()
    raise ValueError(f"unknown replacement-control family {family!r}")


def _canonical_value(value: object) -> bytes:
    return canonical_json({"value": value})


def _example_from_authenticated_row(row: Mapping[str, object]) -> Example:
    required = {"id", "family", "split", "structure_id", "prompt", "answer", "latent_spec"}
    if set(row) != required:
        raise ValueError("authenticated V2 development row does not match the Example schema")
    try:
        return Example(**row)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("authenticated V2 development row cannot form an Example") from error


def _safe_external_root(path: Path, source_root: Path, description: str) -> Path:
    try:
        study_v2._assert_no_reparse(path)
        root = path.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} root is missing or unsafe: {error}") from error
    try:
        root.relative_to(source_root)
    except ValueError:
        return root
    raise ValueError(f"{description} root must be outside the read-only V2 source")


def _safe_controls_root(path: Path) -> Path:
    try:
        study_v2._assert_no_reparse(path)
        root = path.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"replacement controls root is unsafe or missing: {error}") from error
    if not root.is_dir():
        raise ValueError("replacement controls root must be a directory")
    return root


def _ensure_output_external(output_root: Path, source_root: Path, original_root: Path) -> None:
    try:
        study_v2._assert_no_reparse(output_root)
    except (OSError, ValueError) as error:
        raise ValueError(f"replacement-control output is unsafe or contains a symlink/reparse point: {error}") from error
    output = output_root.resolve(strict=False)
    for protected in (source_root, original_root):
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        raise ValueError("replacement-control output must be outside authenticated V2 evidence")


def _assert_no_reparse(path: Path) -> None:
    """Use the V2 path walker for static symlink/reparse checks (not race-proof)."""
    study_v2._assert_no_reparse(path)


def _safe_public_packet_root(value: object, private_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("private manifest does not bind a reviewer-facing packet root")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("reviewer-facing packet root must be absolute")
    try:
        _assert_no_reparse(path)
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("reviewer-facing packet root must be a directory")
        private = private_root.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"reviewer-facing packet root is unsafe or missing: {error}") from error
    _assert_disjoint_roots(private, resolved)
    return resolved


def _assert_disjoint_roots(left: Path, right: Path) -> None:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    for first, second in ((left_resolved, right_resolved), (right_resolved, left_resolved)):
        try:
            second.relative_to(first)
        except ValueError:
            continue
        raise ValueError("private manifest and reviewer packet roots must be physically disjoint")


def _frozen_code_hashes(captured: Mapping[str, object] | None = None) -> dict[str, str]:
    """Hash either the export-time closed file set or exactly its captured names."""
    repo_root = Path(__file__).resolve().parents[3]
    if captured is None:
        relative_names = {
            *(path.relative_to(repo_root).as_posix() for path in (repo_root / "src/ppmoe/recurrence").glob("*.py")),
            "pyproject.toml",
            "uv.lock",
            "tests/test_recurrence_audit_v3_controls.py",
            "tests/test_recurrence_answer_v3.py",
        }
    else:
        if not isinstance(captured, Mapping) or not captured:
            raise ValueError("frozen code manifest must be a non-empty path-to-hash mapping")
        relative_names = set(captured)

    hashes: dict[str, str] = {}
    for name in sorted(relative_names):
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or Path(name).is_absolute()
            or Path(name).as_posix() != name
            or ".." in Path(name).parts
        ):
            raise ValueError("frozen code manifest contains an unsafe relative path")
        path = repo_root / Path(name)
        try:
            study_v2._assert_no_reparse(path)
            resolved = path.resolve(strict=True)
            resolved.relative_to(repo_root.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise ValueError(f"frozen code path is unsafe or missing: {name}") from error
        if not resolved.is_file():
            raise ValueError(f"frozen code path is not a regular file: {name}")
        if captured is not None:
            expected_hash = captured[name]
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
            ):
                raise ValueError(f"frozen code manifest has an invalid SHA-256 for {name}")
        hashes[name] = _sha256_file(resolved)
    return hashes


def _reserve_reconciliation_attempt(root: Path, payload: Mapping[str, object]) -> bytes:
    """Durably claim the sole scoring pass; an interrupted claim remains terminal."""
    safe_root = _safe_controls_root(root)
    destination = safe_root / "attempt.json"
    if (safe_root / "qualification.json").exists():
        raise FileExistsError("replacement-control qualification already exists")
    attempt_payload = dict(payload)
    attempt_payload.setdefault("status", "reserved")
    attempt_payload["manifest_sha256"] = sha256_payload(attempt_payload)
    raw = canonical_json(attempt_payload)
    try:
        _write_exclusive(destination, raw)
    except FileExistsError as error:
        raise _AttemptReservationConflict("another reconciliation attempt already reserved this run") from error
    return raw


def _validate_evidence_candidate(review: Mapping[str, object], example: Example) -> None:
    """Require each cited response span to independently extract the claimed candidate."""
    family = example.family
    evidence = review.get("evidence_span")
    if not isinstance(evidence, str) or not evidence:
        raise ValueError("review evidence is empty")
    claimed = _normalize_value(review.get("claimed_value"), family)
    if family == "expressions":
        extracted, _span, failure = answer_v3._expression_candidate(evidence)
    elif family == "lists":
        extracted, _span, failure = answer_v3._list_candidate(evidence, example)
    elif family == "graphs":
        candidates = answer_v2._graph_candidates(evidence, example)
        if not candidates:
            raise ValueError("graph evidence does not contain a final candidate for the queried endpoints")
        values = {value for value, _ in candidates}
        if len(values) != 1:
            raise ValueError("graph evidence contains ambiguous final candidates")
        extracted, failure = next(iter(values)), None
    else:
        raise ValueError(f"unknown replacement-control family {family!r}")
    if failure is not None or extracted is None:
        raise ValueError("review evidence does not independently yield a final candidate")
    normalized_extracted = _normalize_value(extracted, family)
    if _canonical_value(normalized_extracted) != _canonical_value(claimed):
        raise ValueError("review evidence candidate differs from the claimed V3 final candidate")


def _read_regular(root: Path, relative: Path, description: str) -> bytes:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} path is unsafe")
    path = root / relative
    try:
        study_v2._assert_no_reparse(path)
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} path is unsafe or missing: {error}") from error
    if not resolved.is_file():
        raise ValueError(f"{description} must be a regular file")
    try:
        return resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {description}") from error


def _write_exclusive(path: Path, raw: bytes, *, make_parent: bool = False) -> None:
    if make_parent:
        path.parent.mkdir(exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _with_content_hash(payload: Mapping[str, object]) -> bytes:
    content = dict(payload)
    content["manifest_sha256"] = sha256_payload(payload)
    return canonical_json(content)


def _verify_content_hash(content: Mapping[str, object], description: str) -> dict[str, object]:
    if "manifest_sha256" not in content:
        raise ValueError(f"{description} is missing its content hash")
    payload = {key: value for key, value in content.items() if key != "manifest_sha256"}
    if content["manifest_sha256"] != sha256_payload(payload):
        raise ValueError(f"{description} content hash mismatch")
    return payload


def _require_python_version() -> None:
    if platform.python_version() != EXPECTED_PYTHON:
        raise RuntimeError(f"replacement selector requires Python {EXPECTED_PYTHON}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping_rows(value: object, description: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{description} must be a list of objects")
    return value


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be non-empty text")
    return value
