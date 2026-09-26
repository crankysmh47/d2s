"""Blind, authenticated export of selected V2 development answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Mapping

from .data import FAMILIES
from .provenance import canonical_json, sha256_payload
from . import study_v2


RUN_ID = "18433b1f62c799ac06bfab7250511e4b06c1d936a4174ecfb9d7b21ff834451d"
DEFAULT_SEED = 927001
_DEV_SPLIT = Path("data/dev.jsonl")
_DEV_SIDECAR = Path("data/dev.jsonl.manifest.json")
_DEV_SCORE = Path("scores/dev-f0.jsonl")


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
        "stage_hashes": {
            "prepare": prepare_hash,
            "calibrate": calibrate_hash,
            "report": str(report["stage_sha256"]),
        },
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

    destination = Path(output_root).resolve(strict=False)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a blind audit packet from authenticated V2 development evidence")
    parser.add_argument("--stage", required=True, choices=("export",))
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = export_v2_audit(args.run_root, args.output_root)
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
