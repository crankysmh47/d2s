from __future__ import annotations

from fractions import Fraction

import pytest

from ppmoe.distill.answers import extract_final_numeric, score_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final answer: -12", "-12"),
        ("Final answer: 1,234.50 dollars", "2469/2"),
        ("Final answer: $1,234 USD", "1234"),
        (r"Final answer: \boxed{3/4}", "3/4"),
        ("Final answer: 0", "0"),
        ("Work 1 + 2. Final answer: **-0.25 kg**", "-1/4"),
    ],
)
def test_extracts_exact_single_numeric_answer(text: str, expected: str) -> None:
    result = extract_final_numeric(text, "eos")
    assert result.status == "valid"
    assert result.value == expected
    assert result.evidence is not None and result.evidence in text


def test_accepts_bold_wrapping_the_entire_final_assertion() -> None:
    text = "**Final answer: 5**"
    result = extract_final_numeric(text, "eos")
    assert (result.status, result.value, result.evidence) == ("valid", "5", text)


def test_ignores_intermediate_numbers_and_accepts_final_cue_after_them() -> None:
    result = extract_final_numeric(r"First 3, then \boxed{9}. Final answer: 12", "eos")
    assert result.value == "12"


@pytest.mark.parametrize(
    "text",
    [
        "Final answer: 2. Final answer: 3",
        "Final answer: 2 then maybe 3",
        "Final answer: 2 or 3",
        "Final answer: 2-3",
        "Final answer: 2/3/4",
        "Final answer: \u00a0",
        r"Final answer: \boxed{2",
        r"Final answer: \boxed{2} and 3",
        "Final answer: 2. Final answer:",
    ],
)
def test_rejects_ambiguous_or_malformed_final_answer(text: str) -> None:
    assert extract_final_numeric(text, "eos").status == "invalid"


@pytest.mark.parametrize("stop_reason", ["length", "error", "cancelled"])
def test_requires_eos_completion(stop_reason: str) -> None:
    result = extract_final_numeric("Final answer: 42", stop_reason)
    assert result.status == "invalid"
    assert result.reason == "not_eos"


def test_unknown_units_do_not_get_mistaken_for_another_value() -> None:
    result = extract_final_numeric("Final answer: 42 widgets", "eos")
    assert result.status == "invalid"


def test_recognized_unit_is_accepted_without_rescaling() -> None:
    result = extract_final_numeric("Final answer: 42 dollars", "eos")
    assert result.status == "valid"
    assert result.value == "42"


def test_scoring_is_separate_from_extraction_and_gold() -> None:
    text = "Final answer: 5"
    extraction = extract_final_numeric(text, "eos")
    assert score_answer(extraction, Fraction(5)).correct
    assert not score_answer(extraction, Fraction(6)).correct
    assert extract_final_numeric(text, "eos") == extraction


def test_missing_final_cue_does_not_infer_from_a_bare_number() -> None:
    result = extract_final_numeric("The computation is 42.", "eos")
    assert result.status == "invalid"
    assert result.reason == "missing_final_cue"


def test_equal_repeated_final_cues_are_accepted() -> None:
    result = extract_final_numeric("Final answer: 2. FINAL ANSWER: 2.0", "eos")
    assert result.status == "valid"
    assert result.value == "2"


def test_later_alternate_assertion_invalidates_an_earlier_numeric_cue() -> None:
    result = extract_final_numeric("Final answer: 5. Actually the answer is seven.", "eos")
    assert result.status == "invalid"


@pytest.mark.parametrize(
    "text",
    [
        "Final answer: 5. Actually, the answer is 7.",
        "Final answer: 5. Actually, the answer is",
        "**Final answer: 5** Actually, 7 is correct.",
    ],
)
def test_rejects_later_conflicting_or_incomplete_alternate_assertion(text: str) -> None:
    assert extract_final_numeric(text, "eos").status == "invalid"
