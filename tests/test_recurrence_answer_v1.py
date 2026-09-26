from __future__ import annotations

import importlib

from ppmoe.recurrence.data import Example


def _example(family: str, latent_spec: dict[str, object], answer: str) -> Example:
    return Example(
        id=f"fixture-{family}",
        family=family,
        split="test",
        structure_id="hand-derived",
        prompt="Fixture prompt",
        answer=answer,
        latent_spec=latent_spec,
    )


def _expression() -> Example:
    return _example(
        "expressions",
        {"leaves": [1, 2, 3], "operators": ["+", "+"], "shape": "chain"},
        "6",
    )


def _list() -> Example:
    return _example(
        "lists",
        {"input": [1, 2, 3], "ops": ["ROTL1"]},
        "[2,3,1]",
    )


def _graph() -> Example:
    return _example(
        "graphs",
        {"n": 4, "edges": ["A->B", "B->C"], "start": "A", "end": "C"},
        "yes",
    )


def _score(example: Example, completion: str | None, stop_reason: str = "eos"):
    try:
        module = importlib.import_module("ppmoe.recurrence.answer_v1")
    except ModuleNotFoundError as error:
        if error.name != "ppmoe.recurrence.answer_v1":
            raise
        raise AssertionError("V1 completion scorer is not implemented yet") from None
    return module.score_completion(example, completion, stop_reason)


def test_expression_uses_cued_final_answer_instead_of_intermediate_correct_value() -> None:
    result = _score(_expression(), "Intermediate: \\boxed{6}\nFinal answer: \\boxed{7}")

    assert result["parsed_value"] == 7
    assert result["correct"] is False
    assert result["reason_code"] == "wrong_value"


def test_distinct_final_expression_boxes_are_ambiguous() -> None:
    result = _score(_expression(), "Final answer: \\boxed{6} or \\boxed{7}")

    assert result["parsed_value"] is None
    assert result["correct"] is False
    assert result["reason_code"] == "invalid_format"
    assert "ambiguous" in str(result["failure_detail"])


def test_expression_ignores_box_outside_final_cue_window() -> None:
    result = _score(_expression(), "\\boxed{6}\nThe derivation continues.\nFinal answer: 6")

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


def test_expression_accepts_line_anchored_so_the_final_answer_is_cue() -> None:
    result = _score(_expression(), "So the final answer is:\n\\boxed{6}")

    assert result["parsed_value"] == 6
    assert result["correct"] is True


def test_list_ignores_initial_and_intermediate_vectors() -> None:
    result = _score(
        _list(),
        "Initial: [1, 2, 3]\nAfter rotation: [2, 3, 1]\nFinal answer: [2, 3, 1]",
    )

    assert result["parsed_value"] == [2, 3, 1]
    assert result["correct"] is True
    assert result["reason_code"] == "correct"


def test_list_preserves_out_of_range_and_signed_multidigit_values() -> None:
    result = _score(_list(), "Final result: [11, -2, 0]")

    assert result["parsed_value"] == [11, -2, 0]
    assert result["correct"] is False
    assert result["reason_code"] == "wrong_value"


def test_list_accepts_boxed_vector() -> None:
    result = _score(_list(), r"Final answer: \boxed{[2, 3, 1]}")

    assert result["parsed_value"] == [2, 3, 1]
    assert result["correct"] is True


def test_list_rejects_wrong_length_and_malformed_or_partial_vectors() -> None:
    for completion in (
        "Answer: [2, 3]",
        "Answer: [2, three, 1]",
        "Answer: [2, 3,",
    ):
        result = _score(_list(), completion)
        assert result["parsed_value"] is None
        assert result["correct"] is False
        assert result["reason_code"] == "invalid_format"


def test_length_cap_invalidates_previously_complete_looking_answer() -> None:
    result = _score(_expression(), "Final answer: \\boxed{6}", "length")

    assert result["parsed_value"] is None
    assert result["correct"] is False
    assert result["reason_code"] == "invalid_format"
    assert result["failure_detail"]


def test_graph_disagreeing_leading_and_terminal_answers_are_ambiguous() -> None:
    result = _score(_graph(), "Yes. The route is unclear.\nAnswer: No")

    assert result["parsed_value"] is None
    assert result["correct"] is False
    assert result["reason_code"] == "invalid_format"
    assert "ambiguous" in str(result["failure_detail"])


def test_graph_ignores_incidental_negated_mentions_with_other_endpoints() -> None:
    result = _score(_graph(), "There is no path from A to D, but A reaches C.\nAnswer: yes")

    assert result["parsed_value"] == "yes"
    assert result["correct"] is True


def test_graph_accepts_no_direct_or_indirect_path_conclusion() -> None:
    example = _example(
        "graphs",
        {"n": 3, "edges": ["A->B"], "start": "A", "end": "C"},
        "no",
    )
    result = _score(example, "There is no direct or indirect path from A to C.")

    assert result["parsed_value"] == "no"
    assert result["correct"] is True


def test_graph_marks_invalid_claimed_path_contradicted_despite_correct_answer() -> None:
    result = _score(_graph(), "Answer: yes\nA path is A -> D -> C.")

    assert result["parsed_value"] == "yes"
    assert result["correct"] is True
    assert result["explanation_status"] == "contradicted"
    assert result["explanation_detail"]


def test_graph_verifies_wrapped_unicode_arrow_path_claim() -> None:
    example = _example(
        "graphs",
        {"n": 7, "edges": ["C->G", "G->F", "F->B"], "start": "C", "end": "E"},
        "no",
    )
    result = _score(
        example,
        "Answer: no\nThere is a directed path from C to E:\nC → G → F → B → E.",
    )

    assert result["explanation_status"] == "contradicted"
    assert "B->E" in str(result["explanation_detail"])


def test_empty_completion_is_ungradable_and_result_has_full_contract() -> None:
    result = _score(_expression(), "  ")

    assert set(result) == {
        "parsed_value",
        "correct",
        "reason_code",
        "failure_detail",
        "evidence_span",
        "explanation_status",
        "explanation_detail",
    }
    assert result["parsed_value"] is None
    assert result["correct"] is False
    assert result["reason_code"] == "invalid_format"
    assert result["failure_detail"]
