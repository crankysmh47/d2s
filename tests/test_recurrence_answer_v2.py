from __future__ import annotations

import importlib

import pytest

from ppmoe.recurrence.data import Example


@pytest.fixture
def graph_example() -> Example:
    return Example(
        "id",
        "graphs",
        "dev",
        "n7",
        "Directed edges: A->B. Is there a directed path from A to B? Return only yes or no.",
        "yes",
        {"n": 7, "edges": ["A->B"], "start": "A", "end": "B"},
    )


def _score(example: Example, completion: str | None, stop_reason: str = "eos") -> dict[str, object]:
    module = importlib.import_module("ppmoe.recurrence.answer_v2")
    return module.score_completion_v2(example, completion, stop_reason)


def test_conflicting_graph_cues_are_invalid(graph_example: Example) -> None:
    scored = _score(graph_example, "Yes\n\n**Answer: No**", "eos")

    assert scored["correct"] is False
    assert scored["reason_code"] == "invalid_format"
    assert "ambiguous" in str(scored["failure_detail"])


@pytest.mark.parametrize(
    "completion",
    [
        "Yes",
        "**Answer: yes**",
        "So, the answer is: **Yes**.",
        "there is **a directed path** from **A** to **B**",
    ],
)
def test_graph_extracts_supported_explicit_cues(graph_example: Example, completion: str) -> None:
    scored = _score(graph_example, completion)

    assert scored["parsed_value"] == "yes"
    assert scored["correct"] is True
    assert scored["reason_code"] == "correct"


def test_graph_extracts_bold_negative_path_conclusion(graph_example: Example) -> None:
    scored = _score(graph_example, "There is **no** directed path from **A** to **B**.")

    assert scored["parsed_value"] == "no"
    assert scored["correct"] is False
    assert scored["reason_code"] == "wrong_value"


def test_graph_rejects_wrong_queried_endpoints_without_other_cue(graph_example: Example) -> None:
    scored = _score(graph_example, "There is a path from A to C.")

    assert scored["parsed_value"] is None
    assert scored["reason_code"] == "invalid_format"


def test_graph_does_not_infer_answer_from_incidental_no(graph_example: Example) -> None:
    scored = _score(graph_example, "No, the listed edge is A->B.")

    assert scored["parsed_value"] is None
    assert scored["reason_code"] == "invalid_format"


def test_graph_length_cap_invalidates_explicit_answer(graph_example: Example) -> None:
    scored = _score(graph_example, "Yes", "length")

    assert scored["parsed_value"] is None
    assert scored["correct"] is False
    assert scored["reason_code"] == "invalid_format"


def test_conflicting_path_conclusions_are_invalid(graph_example: Example) -> None:
    scored = _score(graph_example, "There is a path from A to B. There is no path from A to B.")

    assert scored["parsed_value"] is None
    assert scored["correct"] is False
    assert scored["reason_code"] == "invalid_format"
    assert "ambiguous" in str(scored["failure_detail"])


def test_false_path_explanation_is_preserved_for_wrong_answer(graph_example: Example) -> None:
    scored = _score(graph_example, "Answer: no\nA path is A -> C -> B.")

    assert scored["parsed_value"] == "no"
    assert scored["correct"] is False
    assert scored["reason_code"] == "wrong_value"
    assert scored["explanation_status"] == "contradicted"
    assert "A->C" in str(scored["explanation_detail"])


@pytest.mark.parametrize(
    ("completion", "stop_reason"),
    [
        ("Yes, because there is an edge.", "eos"),
        ("**No**, despite the listed edge A->B.", "eos"),
        ("Answer: yes\nFinal answer: no", "eos"),
        ("A->B is listed, but no other edge.", "eos"),
        ("Answer: yes", "length"),
    ],
)
def test_graph_adversarial_non_candidates_and_conflicts(
    graph_example: Example, completion: str, stop_reason: str
) -> None:
    scored = _score(graph_example, completion, stop_reason)

    assert scored["correct"] is False
    assert scored["reason_code"] == "invalid_format"


def test_graph_wrong_well_formed_answer_is_wrong_value(graph_example: Example) -> None:
    scored = _score(graph_example, "No")

    assert scored["parsed_value"] == "no"
    assert scored["correct"] is False
    assert scored["reason_code"] == "wrong_value"


def test_graph_candidate_evidence_preserves_original_text(graph_example: Example) -> None:
    completion = "\n  **Answer: yes**  "
    scored = _score(graph_example, completion)

    assert scored["evidence_span"] == "**Answer: yes**"


def test_graph_rejects_malformed_markdown_decoration(graph_example: Example) -> None:
    scored = _score(graph_example, "Answer: n**o")

    assert scored["parsed_value"] is None
    assert scored["reason_code"] == "invalid_format"


def test_non_graph_families_delegate_to_v1_unchanged() -> None:
    expression = Example(
        "expr", "expressions", "dev", "chain", "prompt", "6",
        {"leaves": [1, 2, 3], "operators": ["+", "+"], "shape": "chain"},
    )

    scored = _score(expression, "Final answer: \\boxed{6}")

    assert scored["parsed_value"] == 6
    assert scored["correct"] is True
    assert set(scored) == {
        "parsed_value",
        "correct",
        "reason_code",
        "failure_detail",
        "evidence_span",
        "explanation_status",
        "explanation_detail",
    }
