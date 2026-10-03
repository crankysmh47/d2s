from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json
import random
import string

import pytest

from ppmoe.distill.data import (
    Example,
    PartitionCounts,
    build_splits,
    canonical_id,
    fivegram_jaccard,
    load_split,
    normalized_question,
    _gsm_records,
    _svamp_records,
    validate_source_bytes,
)


def row(dataset: str, split: str, question: str, gold: str = "1") -> Example:
    return Example(
        id=canonical_id(dataset, split, question),
        dataset=dataset,
        source_split=split,
        question=question,
        gold=gold,
        solution=f"work\nFinal answer: {gold}",
    )


def test_canonical_id_binds_exact_question_and_source() -> None:
    first = row("gsm8k", "train", "One question?")
    assert first.id != row("gsm8k", "train", "one question?").id
    assert first.id != row("gsm8k", "test", "One question?").id
    assert len(first.id) == 64


def test_exact_normalization_collapses_case_punctuation_and_spacing() -> None:
    assert normalized_question("  How MANY?\n apples! ") == "how many apples"


def test_gsm8k_parser_canonicalizes_gold_and_removes_calculator_annotations(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    path = tmp_path / "gsm.parquet"
    parquet.write_table(pa.table({"question": ["Six items?"], "answer": ["Multiply <<2*3=6>>.\n#### 6.0"]}), path)

    records = _gsm_records(path, "train")

    assert records[0].gold == "6"
    assert "<<" not in records[0].solution
    assert records[0].solution.endswith("Final answer: 6")


def test_svamp_parser_preserves_decimal_answer_exactly(tmp_path: Path) -> None:
    path = tmp_path / "svamp.json"
    path.write_text(
        '[{"Body":"A short body.","Question":"How much?","Answer":0.125,'
        '"Equation":"1/8","Type":"Addition","ID":"local"}]',
        encoding="utf-8",
    )

    records = _svamp_records(path)

    assert records[0].question == "A short body. How much?"
    assert records[0].source_split == "all"
    assert records[0].gold == "1/8"


def test_dedup_priority_removes_exact_and_near_cross_source_overlap_deterministically() -> None:
    train = [
        row("gsm8k", "train", "How many red apples are left today?"),
        row("gsm8k", "train", "How many red apples are left today?"),
        row("gsm8k", "train", "A unique pool question."),
    ]
    test = [
        row("gsm8k", "test", "How many red apples are left today!"),
        row("gsm8k", "test", "The quick brown fox jumps over the lazy dog 12."),
    ]
    svamp = [
        row("svamp", "all", "How many red apples are left today?"),
        row("svamp", "all", "The quick brown fox jumps over the lazy dog 13."),
    ]
    counts = PartitionCounts(calibration=0, pilot=0, development=0, pool=1, sealed_test=0, transfer_development=0, sealed_transfer=0)

    first = build_splits(train, test, svamp, counts=counts)
    second = build_splits(list(reversed(train)), list(reversed(test)), list(reversed(svamp)), counts=counts)

    assert first.splits == second.splits
    assert first.splits["pool"][0].question == "A unique pool question."
    assert {item.reason for item in first.exclusions} >= {
        "same_source_exact_duplicate",
        "cross_source_exact_duplicate",
        "cross_source_near_duplicate",
    }


def test_near_duplicate_threshold_uses_character_fivegram_jaccard() -> None:
    assert fivegram_jaccard(
        "The quick brown fox jumps over the lazy dog 12.",
        "The quick brown fox jumps over the lazy dog 13.",
    ) >= 0.85
    assert fivegram_jaccard("The quick brown fox jumps.", "A completely different question?") < 0.85


def test_exact_indexed_filter_matches_exhaustive_randomized_reference() -> None:
    rng = random.Random(47)

    def random_question() -> str:
        words = [
            "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
            for _ in range(18)
        ]
        return " ".join(words)

    earlier_questions = [random_question() for _ in range(16)]
    later_questions = [question[:-1] + ("z" if question[-1] != "z" else "y") for question in earlier_questions]
    later_questions += [random_question() for _ in range(12)]
    later_questions += [question[: len(question) // 2] for question in earlier_questions[:4]]
    later_questions += [
        "The quick brown fox jumps over the lazy dog 12.",
        "The quick brown fox jumps over the lazy dog 13.",
        "same question, with punctuation!",
    ]
    earlier_questions += [
        "The quick brown fox jumps over the lazy dog 12.",
        "same question with punctuation",
    ]

    earlier = [row("gsm8k", "test", question) for question in earlier_questions]
    later = [row("svamp", "all", question) for question in later_questions]
    ordered_earlier = sorted(
        earlier,
        key=lambda item: (sha256(f"d2s-distill-v1|47|{item.id}".encode()).hexdigest(), item.id),
    )
    prior: list[Example] = []
    prior_exact: set[str] = set()
    for item in ordered_earlier:
        normalized = normalized_question(item.question)
        if normalized not in prior_exact:
            prior.append(item)
            prior_exact.add(normalized)

    expected_kept: list[str] = []
    expected_exclusions: dict[str, str] = {}
    seen_later: set[str] = set()
    for item in sorted(
        later,
        key=lambda example: (sha256(f"d2s-distill-v1|47|{example.id}".encode()).hexdigest(), example.id),
    ):
        normalized = normalized_question(item.question)
        if normalized in seen_later:
            expected_exclusions[item.id] = "same_source_exact_duplicate"
        elif normalized in prior_exact:
            expected_exclusions[item.id] = "cross_source_exact_duplicate"
        elif any(fivegram_jaccard(item.question, old.question) >= 0.85 for old in prior):
            expected_exclusions[item.id] = "cross_source_near_duplicate"
        else:
            expected_kept.append(item.id)
        seen_later.add(normalized)

    counts = PartitionCounts(
        calibration=0, pilot=0, development=0, pool=0, sealed_test=0,
        transfer_development=len(expected_kept), sealed_transfer=0,
    )
    manifest = build_splits((), earlier, later, counts=counts)

    assert [item.id for item in manifest.splits["transfer_development"]] == expected_kept
    actual_exclusions = {item.id: item.reason for item in manifest.exclusions}
    assert actual_exclusions == expected_exclusions


def test_repeated_duplicate_ids_and_insufficient_source_pool_are_rejected() -> None:
    duplicate = row("gsm8k", "train", "same question")
    conflicting = Example(duplicate.id, duplicate.dataset, duplicate.source_split, duplicate.question, "2", duplicate.solution)
    counts = PartitionCounts(calibration=0, pilot=0, development=0, pool=2, sealed_test=0, transfer_development=0, sealed_transfer=0)
    with pytest.raises(ValueError, match="duplicate.*ID"):
        build_splits([duplicate, conflicting], [], [], counts=counts)
    with pytest.raises(ValueError, match="insufficient"):
        build_splits([duplicate], [], [], counts=counts)


def test_production_partition_counts_are_exact_and_train_test_disjoint() -> None:
    # Synthetic rows exercise production selectors without reading cached labels.
    train = [row("gsm8k", "train", sha256(f"train-{i}".encode()).hexdigest()) for i in range(7473)]
    test = [row("gsm8k", "test", sha256(f"test-{i}".encode()).hexdigest()) for i in range(1319)]
    svamp = [row("svamp", "all", sha256(f"svamp-{i}".encode()).hexdigest()) for i in range(1000)]

    manifest = build_splits(train, test, svamp)
    assert {name: len(rows) for name, rows in manifest.splits.items()} == {
        "calibration": 60,
        "pilot": 120,
        "development": 200,
        "pool": 1200,
        "sealed_test": 400,
        "transfer_development": 200,
        "sealed_transfer": 500,
    }
    assert set(manifest.splits["pool"]).isdisjoint(manifest.splits["sealed_test"])


def test_model_facing_loader_refuses_held_out_splits_before_opening_files(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="sealed"):
        load_split(tmp_path, "sealed_test")
    with pytest.raises(PermissionError, match="sealed"):
        load_split(tmp_path, "sealed_transfer")


def test_model_facing_loader_returns_no_gold_or_solution(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    payload = {
        "id": "a" * 64,
        "dataset": "gsm8k",
        "source_split": "train",
        "question": "Synthetic question?",
        "gold": "123/7",
        "solution": "Synthetic label text",
    }
    (data / "development.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    loaded = load_split(tmp_path, "development")

    assert loaded == [{
        "id": payload["id"],
        "dataset": payload["dataset"],
        "source_split": payload["source_split"],
        "question": payload["question"],
    }]
    assert "gold" not in loaded[0] and "solution" not in loaded[0]


def test_tampered_pinned_source_bytes_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"pinned bytes")
    pin = {"sha256": "0" * 64, "size_bytes": len(b"pinned bytes")}
    with pytest.raises(ValueError, match="SHA-256"):
        validate_source_bytes(source, pin)
