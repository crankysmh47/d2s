"""Human-readable exact-answer review export and validation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from ppmoe.recurrence.human_review import (
    _cli_export,
    _load_dev_examples,
    _safe_score_paths,
    export_review,
    summarize_review,
)
from ppmoe.recurrence.provenance import sha256_file


def _records() -> list[dict[str, object]]:
    return [
        {
            "example_id": f"dev-{index}",
            "family": "expressions",
            "prompt": f"Compute {index} + 1",
            "answer": str(index + 1),
            "completion_text": f"{index + 1}",
            "completion_token_ids": [index + 1],
            "reason_code": "correct",
            "arm": "F0" if index == 0 else "A1",
            "split": "dev",
            "automatic_correct": index == 0,
        }
        for index in range(3)
    ]


def _grade_rows(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("| r")]


def _set_grades(path: Path, grades: list[str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    indices = [index for index, line in enumerate(lines) if line.startswith("| r")]
    assert len(indices) == len(grades)
    for index, grade in zip(indices, grades, strict=True):
        cells = lines[index].split("|")
        cells[-2] = f" {grade} "
        lines[index] = "|".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_export_round_trip_hides_automatic_metadata_and_reports_pending(tmp_path: Path) -> None:
    output = tmp_path / "review"
    result = export_review(_records(), output, seed=12)
    assert result["review_path"] == output / "review.md"
    assert result["manifest_path"] == output / "manifest.json"
    review = (output / "review.md").read_text(encoding="utf-8")
    assert "automatic_correct" not in review
    assert "F0" not in review and "A1" not in review
    assert "| good/bad |" in review

    _set_grades(output / "review.md", ["good", "", "BAD"])
    summary = summarize_review(output)
    assert summary == {
        "reviewed": 2, "pending": 1, "good": 1, "bad": 1, "complete": False,
        "reviewer_kind": "unassigned", "reviewer_id": None,
    }

    _set_grades(output / "review.md", ["GOOD", "bad", "bad"])
    summary = summarize_review(output)
    assert summary["reviewed"] == 3
    assert summary["pending"] == 0
    assert summary["good"] == 1
    assert summary["bad"] == 2
    assert summary["complete"] is True


def test_manifest_preserves_full_original_records_without_revealing_them_in_review(tmp_path: Path) -> None:
    records = _records()
    output = tmp_path / "review"
    export_review(records, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    originals = [row["record"] for row in manifest["rows"]]
    assert sorted(original["example_id"] for original in originals) == sorted(
        record["example_id"] for record in records
    )
    for original in originals:
        source = next(record for record in records if record["example_id"] == original["example_id"])
        assert original == source
    review = (output / "review.md").read_text(encoding="utf-8")
    assert "automatic_correct" not in review
    assert "F0" not in review and "A1" not in review


def test_cli_manifest_records_verified_input_paths_and_hashes(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    data_dir = run_root / "data"
    scores_dir = run_root / "scores"
    data_dir.mkdir(parents=True)
    scores_dir.mkdir()
    dev_path = data_dir / "dev.jsonl"
    dev_path.write_text(json.dumps({"id": "dev-1", "split": "dev", "prompt": "question", "answer": "answer"}) + "\n")
    data_manifest_path = data_dir / "manifest.json"
    data_manifest_path.write_text(json.dumps({"split_sha256": {"dev": sha256_file(dev_path)}}))
    stages_dir = run_root / "stages"
    stages_dir.mkdir()
    (stages_dir / "data.json").write_text(json.dumps({
        "stage": "data", "status": "passed", "payload": {"manifest_sha256": sha256_file(data_manifest_path)},
    }))
    score_path = scores_dir / "f0-dev.jsonl"
    score_path.write_text(json.dumps({
        "example_id": "dev-1", "split": "dev", "arm": "F0", "family": "expressions",
        "completion_text": "answer", "completion_token_ids": [123], "reason_code": "correct",
        "correct": True,
    }) + "\n")
    score_summary_path = scores_dir / "f0-dev.summary.json"
    score_summary_path.write_text(json.dumps({
        "schema_version": 1, "kind": "exact_scores", "complete": True, "split": "dev",
        "records_sha256": sha256_file(score_path),
    }))
    output = tmp_path / "review"
    _cli_export(run_root, ["scores/f0-dev.jsonl"], output, 9158)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["rows"][0]["record"]
    assert record["original_score_record"]["correct"] is True
    assert record["original_dev_record"]["answer"] == "answer"
    provenance = record["source_provenance"]
    assert provenance["dev_data_path"] == str(dev_path.resolve())
    assert provenance["dev_data_sha256"] == sha256_file(dev_path)
    assert provenance["data_manifest_sha256"] == sha256_file(data_manifest_path)
    assert provenance["score_path"] == str(score_path.resolve())
    assert provenance["score_sha256"] == sha256_file(score_path)
    assert provenance["score_summary_path"] == str(score_summary_path.resolve())
    assert provenance["score_summary_sha256"] == sha256_file(score_summary_path)
    review = (output / "review.md").read_text(encoding="utf-8")
    assert "original_score_record" not in review
    assert "True" not in review
    assert "F0" not in review


def test_markdown_escapes_pipes_html_and_newlines_without_truncating(tmp_path: Path) -> None:
    record = _records()[0]
    record["prompt"] = "A | B\n<script>alert(1)</script>"
    record["completion_text"] = "full | response\n<done>"
    record["completion_token_ids"] = list(range(1000))
    export_review([record], tmp_path / "review")
    content = (tmp_path / "review" / "review.md").read_text(encoding="utf-8")
    assert "A &#124; B<br>&lt;script&gt;alert(1)&lt;/script&gt;" in content
    assert "full &#124; response<br>&lt;done&gt;" in content
    assert "999" not in content


def test_untouched_export_preserves_leading_and_trailing_display_whitespace(tmp_path: Path) -> None:
    record = _records()[0]
    record["prompt"] = "  question with boundary spaces  "
    record["answer"] = " answer "
    record["completion_text"] = " response with trailing space "
    output = tmp_path / "whitespace-review"
    export_review([record], output)
    assert summarize_review(output)["pending"] == 1


@pytest.mark.parametrize("mutation", ["field", "id", "duplicate", "missing", "unknown", "grade"])
def test_summary_rejects_review_edits_other_than_grade(tmp_path: Path, mutation: str) -> None:
    output = tmp_path / "review"
    export_review(_records(), output)
    path = output / "review.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    row_indexes = [index for index, line in enumerate(lines) if line.startswith("| r")]
    if mutation == "field":
        lines[row_indexes[0]] = lines[row_indexes[0]].replace("Compute", "Altered", 1)
    elif mutation == "id":
        cells = lines[row_indexes[0]].split("|")
        cells[1] = " changed-id "
        lines[row_indexes[0]] = "|".join(cells)
    elif mutation == "duplicate":
        lines[row_indexes[1]] = lines[row_indexes[0]]
    elif mutation == "missing":
        del lines[row_indexes[0]]
    elif mutation == "unknown":
        cells = lines[row_indexes[0]].split("|")
        cells[1] = " unknown-id "
        lines[row_indexes[0]] = "|".join(cells)
    else:
        cells = lines[row_indexes[0]].split("|")
        cells[-2] = " maybe "
        lines[row_indexes[0]] = "|".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        summarize_review(output)


def test_export_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "review"
    export_review(_records(), output)
    before = (output / "review.md").read_bytes()
    with pytest.raises(FileExistsError):
        export_review(_records(), output)
    assert (output / "review.md").read_bytes() == before


def test_export_requires_explicit_dev_split(tmp_path: Path) -> None:
    record = _records()[0]
    del record["split"]
    with pytest.raises(ValueError, match="dev rows"):
        export_review([record], tmp_path / "review")


def test_multiple_arms_can_review_same_example_but_duplicate_pairs_cannot(tmp_path: Path) -> None:
    first = _records()[0]
    second = dict(first, arm="A2")
    export_review([first, second], tmp_path / "multi-arm")
    with pytest.raises(ValueError, match="arm/example_id"):
        export_review([first, dict(first)], tmp_path / "duplicate")


def test_manifest_hash_tampering_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "review"
    export_review(_records(), output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][0]["displayed"]["answer"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        summarize_review(output)


def test_reviewer_provenance_requires_identity_for_model(tmp_path: Path) -> None:
    output = tmp_path / "review"
    export_review(_records(), output)
    with pytest.raises(ValueError, match="reviewer-id"):
        summarize_review(output, reviewer_kind="model")
    summary = summarize_review(output, reviewer_kind="model", reviewer_id="luna")
    assert summary["reviewer_kind"] == "model"
    assert summary["reviewer_id"] == "luna"
    assert "model review" in (output / "review.md").read_text(encoding="utf-8").lower()


def test_cli_rejects_sealed_score_paths_before_opening(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/review_recurrence_answers.py",
            "export",
            "--run-root",
            str(run_root),
            "--scores",
            "scores/f0-test.jsonl",
            "--output-dir",
            str(tmp_path / "review"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "test" in result.stderr.lower()
    assert not (tmp_path / "review").exists()


def test_cli_rejects_run_root_symlink_alias(tmp_path: Path) -> None:
    real_root = tmp_path / "real-run"
    real_root.mkdir()
    alias_root = tmp_path / "alias-run"
    try:
        alias_root.symlink_to(real_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="symlink or reparse"):
        _safe_score_paths(alias_root, [])


def test_cli_binds_displayed_data_to_frozen_dev_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    data = run_root / "data"
    data.mkdir(parents=True)
    dev_path = data / "dev.jsonl"
    dev_path.write_text(json.dumps({"id": "dev-1", "split": "dev", "prompt": "Q", "answer": "A"}) + "\n")
    manifest_path = data / "manifest.json"
    manifest_path.write_text(json.dumps({"split_sha256": {"dev": "0" * 64}}))
    stages = run_root / "stages"
    stages.mkdir()
    (stages / "data.json").write_text(json.dumps({
        "stage": "data", "status": "passed", "payload": {"manifest_sha256": sha256_file(manifest_path)},
    }))
    with pytest.raises(ValueError, match="split_sha256.dev"):
        _load_dev_examples(run_root)


def test_cli_requires_passed_data_stage_to_bind_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    data = run_root / "data"
    stages = run_root / "stages"
    data.mkdir(parents=True)
    stages.mkdir()
    dev_path = data / "dev.jsonl"
    dev_path.write_text(json.dumps({"id": "dev-1", "split": "dev"}) + "\n")
    manifest_path = data / "manifest.json"
    manifest_path.write_text(json.dumps({"split_sha256": {"dev": sha256_file(dev_path)}}))
    (stages / "data.json").write_text(json.dumps({
        "stage": "data", "status": "passed", "payload": {"manifest_sha256": "0" * 64},
    }))
    with pytest.raises(ValueError, match="does not bind"):
        _load_dev_examples(run_root)
