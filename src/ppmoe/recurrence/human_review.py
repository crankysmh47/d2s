"""Blind, editable human review exports for exact-answer completions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import sys

from .provenance import canonical_json, sha256_file, sha256_payload


_MANIFEST_VERSION = 1
_HEADER = "| Row ID | Question | Reference answer | Full response | good/bad |"
_DIVIDER = "|---|---|---|---|---|"
_INSTRUCTIONS = (
    "Review provenance: labels entered by a human reviewer are human review; labels entered by a model are "
    "model review and must not be described as human review. Mark each response good or bad in the last column only. "
    "Good means the final answer is correct, "
    "including when it has an explanation. Bad means wrong, contradictory, or no completed answer. "
    "Leave the cell blank if unreviewed."
)


def export_review(
    records: Sequence[Mapping[str, object]], output_dir: Path, seed: int = 9158
) -> Mapping[str, Path]:
    """Write a blinded shuffled Markdown sheet and its immutable reconciliation manifest."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if output_dir.exists():
        raise FileExistsError(f"review output already exists: {output_dir}")
    prepared: list[dict[str, object]] = []
    seen: set[tuple[object, str]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("review records must be mappings")
        identifier = record.get("example_id")
        arm = record.get("arm")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("review records require non-empty example_id values")
        if arm is not None and not isinstance(arm, str):
            raise ValueError("review record arm must be text or null")
        key = (arm, identifier)
        if key in seen:
            raise ValueError("review records require unique arm/example_id pairs")
        if record.get("split") != "dev":
            raise ValueError("human review export accepts dev rows only")
        seen.add(key)
        original_record = _json_compatible(dict(record))
        completion = record.get("completion_text")
        if completion is None:
            completion = ""
        if not isinstance(completion, str):
            raise ValueError("completion_text must be text or null")
        displayed = {
            "question": _required_text(record, "prompt"),
            "reference_answer": _required_text(record, "answer"),
            "response": completion,
        }
        prepared.append({
            "arm": arm,
            "example_id": identifier,
            "displayed": displayed,
            "displayed_sha256": sha256_payload(displayed),
            "record": original_record,
            "record_sha256": sha256_payload(original_record),
        })
    order = list(range(len(prepared)))
    random.Random(seed).shuffle(order)
    rows: list[dict[str, object]] = []
    for position, index in enumerate(order, 1):
        row = prepared[index]
        rows.append({"row_id": f"r{position:06d}", **row})
    document = _render(rows, grades=[""] * len(rows))
    manifest: dict[str, object] = {
        "schema_version": _MANIFEST_VERSION,
        "row_count": len(rows),
        "review_template_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "rows": rows,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    # Exclusive creation ensures a concurrent exporter cannot replace another review.
    output_dir.mkdir(parents=False, exist_ok=False)
    try:
        _write_new(output_dir / "review.md", document.encode("utf-8"))
        _write_new(output_dir / "manifest.json", canonical_json(manifest))
    except BaseException:
        # Do not remove partial user-visible artifacts automatically; the caller can inspect them.
        raise
    return {"review_path": output_dir / "review.md", "manifest_path": output_dir / "manifest.json"}


def summarize_review(
    output_dir: Path, *, reviewer_kind: str = "unassigned", reviewer_id: str | None = None
) -> Mapping[str, object]:
    """Validate that only grade cells changed, then count labels without inferring missing ones."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    manifest_path = output_dir / "manifest.json"
    review_path = output_dir / "review.md"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        text = review_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("review export is missing or unreadable") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _MANIFEST_VERSION:
        raise ValueError("unsupported review manifest schema")
    expected_hash = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not isinstance(expected_hash, str) or expected_hash != sha256_payload(unsigned):
        raise ValueError("review manifest hash mismatch")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or manifest.get("row_count") != len(rows):
        raise ValueError("review manifest row schema is invalid")
    row_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("row_id"), str):
            raise ValueError("review manifest row schema is invalid")
        displayed = row.get("displayed")
        if not isinstance(displayed, dict) or row.get("displayed_sha256") != sha256_payload(displayed):
            raise ValueError("review manifest displayed-field hash mismatch")
        if set(displayed) != {"question", "reference_answer", "response"} or any(
            not isinstance(value, str) for value in displayed.values()
        ):
            raise ValueError("review manifest displayed-field schema is invalid")
        # Extra record fields are optional so earlier schema-1 exports remain readable.
        if "record" in row:
            original = row.get("record")
            if not isinstance(original, dict) or row.get("record_sha256") != sha256_payload(original):
                raise ValueError("review manifest original-record hash mismatch")
            if original.get("example_id") != row.get("example_id") or original.get("arm") != row.get("arm"):
                raise ValueError("review manifest original record does not match its row mapping")
        row_ids.append(row["row_id"])
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("review manifest contains duplicate row IDs")
    lines = text.splitlines()
    expected_prefix = ["# Answer review", "", _INSTRUCTIONS, "", _HEADER, _DIVIDER]
    if lines[:len(expected_prefix)] != expected_prefix:
        raise ValueError("review instructions or table header were edited")
    table_lines = lines[len(expected_prefix):]
    if len(table_lines) != len(rows):
        raise ValueError("review contains missing or extra rows")
    grades: list[str] = []
    for line, row in zip(table_lines, rows, strict=True):
        cells = line.split("|")
        if len(cells) != 7 or cells[0] != "" or cells[-1] != "":
            raise ValueError("review table row is malformed")
        raw_cells = cells[1:-1]
        displayed_cells = [_remove_render_padding(cell) for cell in raw_cells[:4]]
        expected_cells = [
            row["row_id"],
            _escape_cell(row["displayed"]["question"]),
            _escape_cell(row["displayed"]["reference_answer"]),
            _escape_cell(row["displayed"]["response"]),
        ]
        if displayed_cells != expected_cells:
            raise ValueError("review row ID or displayed fields were edited")
        grade = raw_cells[4].strip().casefold()
        if grade not in {"", "good", "bad"}:
            raise ValueError("grade labels must be blank, good, or bad")
        grades.append(grade)
    blank_template = _render(rows, grades=[""] * len(rows))
    if hashlib.sha256(blank_template.encode("utf-8")).hexdigest() != manifest.get("review_template_sha256"):
        raise ValueError("review template hash mismatch")
    good = grades.count("good")
    bad = grades.count("bad")
    pending = grades.count("")
    if reviewer_kind not in {"human", "model", "unassigned"}:
        raise ValueError("reviewer_kind must be human, model, or unassigned")
    if reviewer_id is not None and (not isinstance(reviewer_id, str) or not reviewer_id.strip()):
        raise ValueError("reviewer_id must be a non-empty string when supplied")
    if reviewer_kind == "model" and reviewer_id is None:
        raise ValueError("model review requires --reviewer-id")
    return {
        "reviewed": good + bad,
        "pending": pending,
        "good": good,
        "bad": bad,
        "complete": pending == 0,
        "reviewer_kind": reviewer_kind,
        "reviewer_id": reviewer_id,
    }


def _required_text(record: Mapping[str, object], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    return value


def _json_compatible(value: object) -> object:
    """Copy JSON-compatible values into plain containers for durable provenance."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("original review record mapping keys must be strings")
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("original review record must contain only JSON-compatible values")


def _escape_cell(value: str) -> str:
    escaped = html.escape(value, quote=True).replace("|", "&#124;")
    return re.sub(r"\r\n|\r|\n", "<br>", escaped)


def _remove_render_padding(cell: str) -> str:
    """Remove exactly the one space added around a Markdown cell's content."""
    if len(cell) < 2 or not cell.startswith(" ") or not cell.endswith(" "):
        raise ValueError("review table cell padding is malformed")
    return cell[1:-1]


def _render(rows: Sequence[Mapping[str, object]], grades: Sequence[str]) -> str:
    lines = ["# Answer review", "", _INSTRUCTIONS, "", _HEADER, _DIVIDER]
    for row, grade in zip(rows, grades, strict=True):
        displayed = row["displayed"]
        assert isinstance(displayed, Mapping)
        lines.append(
            "| " + " | ".join((
                str(row["row_id"]),
                _escape_cell(str(displayed["question"])),
                _escape_cell(str(displayed["reference_answer"])),
                _escape_cell(str(displayed["response"])),
                grade,
            )) + " |"
        )
    return "\n".join(lines) + "\n"


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _assert_no_reparse(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError(f"symlink or reparse point is not allowed: {current}")


def _safe_score_paths(run_root: Path, score_args: Sequence[str]) -> list[Path]:
    _assert_no_reparse(run_root.absolute())
    root = run_root.resolve(strict=True)
    _assert_no_reparse(root)
    paths: list[Path] = []
    for value in score_args:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        # Reject lexical sealed paths before touching the candidate.
        try:
            relative = candidate.absolute().relative_to(root)
        except ValueError as error:
            raise ValueError("score path must be inside --run-root") from error
        if (
            not relative.parts
            or any(part in {".", ".."} for part in relative.parts)
            or relative.parts[0].casefold() != "scores"
            or any("test" in part.casefold() for part in relative.parts)
        ):
            raise ValueError("test or non-score paths are sealed; only explicit dev scores are allowed")
        _assert_no_reparse(candidate.absolute())
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError("score path escapes --run-root") from error
        if not resolved.is_file():
            raise ValueError("score path must name a regular file")
        paths.append(resolved)
    return paths


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _cli_export(run_root: Path, score_args: Sequence[str], output_dir: Path, seed: int) -> Mapping[str, Path]:
    paths = _safe_score_paths(run_root, score_args)
    root = run_root.resolve(strict=True)
    dev_path = root / "data" / "dev.jsonl"
    data_manifest_path = root / "data" / "manifest.json"
    _assert_no_reparse(dev_path)
    examples = _load_dev_examples(root)
    dev_data_sha256 = sha256_file(dev_path)
    data_manifest_sha256 = sha256_file(data_manifest_path)
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    tokenizer: object | None = None
    for path in paths:
        summary_path = path.with_name(f"{path.stem}.summary.json")
        _assert_no_reparse(summary_path)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError) as error:
            raise ValueError(f"completed dev score summary is required: {summary_path}") from error
        if (
            not isinstance(summary, dict)
            or summary.get("schema_version") != 1
            or summary.get("kind") != "exact_scores"
            or summary.get("complete") is not True
            or summary.get("split") != "dev"
            or summary.get("records_sha256") != sha256_file(path)
        ):
            raise ValueError(f"score summary is incomplete, not dev, or hash-mismatched: {summary_path}")
        score_sha256 = str(summary["records_sha256"])
        summary_sha256 = sha256_file(summary_path)
        for score in _load_jsonl(path):
            if score.get("split") != "dev":
                raise ValueError("score file contains a non-dev row")
            identifier = score.get("example_id")
            if not isinstance(identifier, str) or identifier not in examples:
                raise ValueError("score row ID is absent from data/dev.jsonl")
            key = (str(score.get("arm", "")), identifier)
            if key in seen:
                raise ValueError("duplicate arm/example score row")
            seen.add(key)
            example = examples[identifier]
            if example.get("split") != "dev":
                raise ValueError("matched data row is not dev")
            completion = score.get("completion_text")
            if completion is None:
                if tokenizer is None:
                    tokenizer = _load_local_tokenizer(root)
                completion = _decode_token_ids(tokenizer, score.get("completion_token_ids"))
            records.append({
                **score,
                "example_id": identifier,
                "family": score.get("family", example.get("family")),
                "prompt": example.get("prompt"),
                "answer": example.get("answer"),
                "completion_text": completion,
                "completion_token_ids": score.get("completion_token_ids"),
                "reason_code": score.get("reason_code"),
                "arm": score.get("arm"),
                "split": score.get("split"),
                "original_score_record": score,
                "original_dev_record": example,
                "source_provenance": {
                    "run_root": str(root),
                    "dev_data_path": str(dev_path.resolve()),
                    "dev_data_sha256": dev_data_sha256,
                    "data_manifest_path": str(data_manifest_path.resolve()),
                    "data_manifest_sha256": data_manifest_sha256,
                    "score_path": str(path.resolve()),
                    "score_sha256": score_sha256,
                    "score_summary_path": str(summary_path.resolve()),
                    "score_summary_sha256": summary_sha256,
                },
            })
    return export_review(records, output_dir, seed)


def _load_local_tokenizer(run_root: Path) -> object:
    source_path = run_root / "stages" / "source.json"
    _assert_no_reparse(source_path)
    if not source_path.is_file():
        raise ValueError("pinned local model source is required to decode completion token IDs")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_payload = source.get("payload")
    if not isinstance(source_payload, dict):
        raise ValueError("source stage record has no pinned source payload")
    snapshot_relative = source_payload.get("snapshot_relative_path")
    if (
        not isinstance(snapshot_relative, str)
        or not snapshot_relative
        or "\\" in snapshot_relative
        or Path(snapshot_relative).is_absolute()
        or any(part in {".", ".."} for part in Path(snapshot_relative).parts)
    ):
        raise ValueError("source record has no pinned snapshot path")
    candidate_snapshot = run_root / snapshot_relative
    _assert_no_reparse(candidate_snapshot)
    snapshot = candidate_snapshot.resolve(strict=True)
    snapshot.relative_to(run_root.resolve(strict=True))
    _assert_no_reparse(snapshot)
    tokenizer_file = snapshot / "tokenizer.json"
    _assert_no_reparse(tokenizer_file)
    expected_tokenizer_hash = source_payload.get("tokenizer_sha256")
    if (
        not isinstance(expected_tokenizer_hash, str)
        or not tokenizer_file.is_file()
        or sha256_file(tokenizer_file) != expected_tokenizer_hash
    ):
        raise ValueError("pinned tokenizer.json is missing or its source hash does not match")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)


def _decode_token_ids(tokenizer: object, token_ids: object) -> str:
    if not isinstance(token_ids, list) or any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids):
        raise ValueError("completion token IDs are unavailable for local decoding")
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise ValueError("local tokenizer cannot decode completion token IDs")
    return str(decode(token_ids, skip_special_tokens=True))


def _load_dev_examples(run_root: Path) -> dict[str, dict[str, object]]:
    dev_path = run_root / "data" / "dev.jsonl"
    manifest_path = run_root / "data" / "manifest.json"
    data_stage_path = run_root / "stages" / "data.json"
    _assert_no_reparse(dev_path)
    _assert_no_reparse(manifest_path)
    _assert_no_reparse(data_stage_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen data manifest is missing or unreadable") from error
    try:
        data_stage = json.loads(data_stage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("passed data stage record is missing or unreadable") from error
    stage_payload = data_stage.get("payload") if isinstance(data_stage, dict) else None
    manifest_sha256 = sha256_file(manifest_path)
    if (
        not isinstance(data_stage, dict)
        or data_stage.get("stage") != "data"
        or data_stage.get("status") != "passed"
        or not isinstance(stage_payload, dict)
        or stage_payload.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError("passed data stage does not bind data/manifest.json")
    split_hashes = manifest.get("split_sha256") if isinstance(manifest, dict) else None
    if not isinstance(split_hashes, dict) or split_hashes.get("dev") != sha256_file(dev_path):
        raise ValueError("data/dev.jsonl does not match frozen manifest split_sha256.dev")
    rows = _load_jsonl(dev_path)
    examples: dict[str, dict[str, object]] = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or row.get("split") != "dev" or identifier in examples:
            raise ValueError("frozen dev data contains invalid split or duplicate IDs")
        examples[identifier] = row
    return examples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export or summarize blind human review of dev completions.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--run-root", type=Path, required=True)
    export_parser.add_argument("--scores", nargs="+", required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--seed", type=int, default=9158)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    summary_parser.add_argument("--reviewer-kind", choices=("human", "model", "unassigned"), default="unassigned")
    summary_parser.add_argument("--reviewer-id")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = _cli_export(args.run_root, args.scores, args.output_dir, args.seed)
        else:
            result = summarize_review(args.output_dir, reviewer_kind=args.reviewer_kind, reviewer_id=args.reviewer_id)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
