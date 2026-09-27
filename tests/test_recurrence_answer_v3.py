from __future__ import annotations

import importlib

import pytest

from ppmoe.recurrence.data import Example


def expression_example(answer: str = "6", latent_spec: dict[str, object] | None = None) -> Example:
    return Example(
        "expr-v3", "expressions", "dev", "chain", "Calculate the expression.", answer,
        latent_spec or {"leaves": [1, 2, 3], "operators": ["+", "+"], "shape": "chain"},
    )


def list_example(answer: str = "[2, 3, 1]", latent_spec: dict[str, object] | None = None) -> Example:
    return Example(
        "list-v3", "lists", "dev", "rotl", "Apply the operation.", answer,
        latent_spec or {"input": [1, 2, 3], "ops": ["ROTL1"]},
    )


def graph_example(answer: str = "yes", latent_spec: dict[str, object] | None = None) -> Example:
    return Example(
        "graph-v3", "graphs", "dev", "n7", "Is there a path from A to B?", answer,
        latent_spec or {"n": 7, "edges": ["A->B"], "start": "A", "end": "B"},
    )


def score(example: Example, completion: str | None, stop_reason: str = "eos") -> dict[str, object]:
    module = importlib.import_module("ppmoe.recurrence.answer_v3")
    return module.score_completion_v3(example, completion, stop_reason)


@pytest.mark.parametrize(
    "completion",
    [
        "Final answer: -12",
        "Final result: \\boxed{-12}",
        "Answer: $-12$",
        "So the final answer is: $$-12$$",
        "### Final Answer: **-12**",
        "Final answer: $ \\boxed{-12} $",
        "So the final result is:\n$$\n\\boxed{-12}\n$$",
        "__Answer__: -12.",
    ],
)
def test_expression_accepts_registered_cues_and_balanced_scalar_wrappers(completion: str) -> None:
    result = score(expression_example(answer="-12", latent_spec={"leaves": [-12, 0], "operators": ["+"], "shape": "chain"}), completion)

    assert result["parsed_value"] == -12
    assert result["correct"] is True
    assert result["reason_code"] == "correct"


@pytest.mark.parametrize(
    "completion",
    [
        "Result: 3",
        "The answer is Final answer: 3",
        "> Final answer: 3",
        "```text\nFinal answer: 3\n```",
        "Final answerable: 3",
        "Final answer: 30 points",
        "Final answer: 3 and then 4",
        "Final answer: $3",
    ],
)
def test_expression_rejects_noncue_lines_and_non_scalar_payloads(completion: str) -> None:
    result = score(expression_example(), completion)

    assert result["parsed_value"] is None
    assert result["correct"] is False
    assert result["reason_code"] == "invalid_format"


def test_shorter_backtick_fence_line_does_not_close_longer_fence() -> None:
    result = score(expression_example(), "````text\n```\nFinal answer: 6")

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


def test_four_space_indented_fence_line_does_not_close_open_fence() -> None:
    result = score(expression_example(), "```text\n    ```\nFinal answer: 6")

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


@pytest.mark.parametrize("indent", ["    ", "\t"])
def test_overindented_or_tab_indented_line_does_not_open_fence(indent: str) -> None:
    result = score(expression_example(), f"{indent}```\nFinal answer: 6")

    assert result["parsed_value"] == 6
    assert result["correct"] is True


def test_tab_indented_fence_line_does_not_close_open_fence() -> None:
    result = score(expression_example(), "```text\n\t```\nFinal answer: 6")

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


def test_expression_agrees_on_equal_final_cues_and_uses_the_final_not_intermediate_value() -> None:
    result = score(expression_example(), "Intermediate: 6\nFinal answer: 7\nAnswer: $7$")

    assert result["parsed_value"] == 7
    assert result["reason_code"] == "wrong_value"


@pytest.mark.parametrize(
    "completion",
    [
        "Final answer: 6\nAnswer: 7",
        "Final answer: 6\nFinal result: not finished",
    ],
)
def test_expression_rejects_conflicting_or_later_incomplete_final_cue(completion: str) -> None:
    result = score(expression_example(), completion)

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


def test_expression_searches_no_more_than_256_characters_after_cue() -> None:
    result = score(expression_example(), "Final answer: " + (" " * 257) + "6")

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


def test_expression_extraction_does_not_use_gold_answer() -> None:
    completion = "Final answer: 7"
    correct_gold = expression_example(answer="7", latent_spec={"leaves": [7, 0], "operators": ["+"], "shape": "chain"})
    wrong_gold = expression_example(answer="6")

    parsed_correct = score(correct_gold, completion)
    parsed_wrong = score(wrong_gold, completion)

    assert parsed_correct["parsed_value"] == parsed_wrong["parsed_value"] == 7
    assert parsed_correct["correct"] is True
    assert parsed_wrong["correct"] is False


@pytest.mark.parametrize(
    "completion",
    [
        "Answer: [2, 3, 1]",
        "Final answer: \\boxed{[2, 3, 1]}",
        "Final result: $[2, 3, 1]$",
        "Final list: **[2, 3, 1]**",
        "So the final list is: $$\\boxed{[2, 3, 1]}$$",
        "So the final list after applying ADD1 once is:\n$$\n[2, 3, 1]\n$$",
    ],
)
def test_list_accepts_registered_cues_and_wrappers(completion: str) -> None:
    result = score(list_example(), completion)

    assert result["parsed_value"] == [2, 3, 1]
    assert result["correct"] is True


@pytest.mark.parametrize(
    "completion",
    [
        "Final answer: [2, 3,]",
        "Answer: [2, three, 1]",
        "Answer: [2, 3,",
        "Answer: [2, 3]",
        "Answer: [2, 3, 1] extra 9",
        "Final answer: [2, 3, 1]\nAnswer: [2, 3,",
        "Final answer: [2, 3, 1]\nAnswer: [1, 2, 3]",
        "> Answer: [2, 3, 1]",
        "```\nAnswer: [2, 3, 1]\n```",
    ],
)
def test_list_rejects_partial_malformed_wrong_length_conflicting_and_quoted_cues(completion: str) -> None:
    result = score(list_example(), completion)

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


def test_list_parses_integer_ten_as_a_value_then_marks_it_wrong() -> None:
    result = score(list_example(), "Final answer: [10, 3, 1]")

    assert result["parsed_value"] == [10, 3, 1]
    assert result["correct"] is False
    assert result["reason_code"] == "wrong_value"


def test_list_agrees_on_equal_complete_final_cues() -> None:
    result = score(list_example(), "Final answer: [2, 3, 1]\nFinal list: $$[2, 3, 1]$$")

    assert result["parsed_value"] == [2, 3, 1]
    assert result["correct"] is True


def test_list_cue_window_is_limited_to_256_characters() -> None:
    result = score(list_example(), "Answer: " + (" " * 300) + "[2, 3, 1]")

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


def test_list_does_not_take_vector_after_next_final_cue() -> None:
    result = score(list_example(), "Final answer: no complete value\nAnswer: [2, 3, 1]")

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


def test_list_extraction_is_independent_of_gold_answer() -> None:
    completion = "Answer: [2, 3, 1]"
    rotated = list_example(answer="[2, 3, 1]")
    unchanged = list_example(answer="[1, 2, 3]", latent_spec={"input": [1, 2, 3], "ops": []})

    parsed_rotated = score(rotated, completion)
    parsed_unchanged = score(unchanged, completion)

    assert parsed_rotated["parsed_value"] == parsed_unchanged["parsed_value"] == [2, 3, 1]
    assert parsed_rotated["correct"] is True
    assert parsed_unchanged["correct"] is False


@pytest.mark.parametrize("completion", [None, "  "])
def test_empty_or_missing_completion_is_invalid(completion: str | None) -> None:
    result = score(expression_example(), completion)

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"


def test_non_eos_completion_is_invalid_even_when_value_is_complete() -> None:
    result = score(expression_example(), "Final answer: 6", "length")

    assert result["parsed_value"] is None
    assert result["correct"] is False
    assert result["reason_code"] == "invalid_format"


def test_graph_v3_preserves_v2_conflict_and_explanation_contract() -> None:
    result = score(graph_example(), "Answer: yes\nFinal answer: no\nA path is A -> C -> B.")

    assert result["parsed_value"] is None
    assert result["reason_code"] == "invalid_format"
    assert result["explanation_status"] == "contradicted"
    assert "A->C" in str(result["explanation_detail"])


def test_all_families_preserve_v2_score_record_schema() -> None:
    expected_keys = {
        "parsed_value", "correct", "reason_code", "failure_detail", "evidence_span",
        "explanation_status", "explanation_detail",
    }

    for example, completion in (
        (expression_example(), "Final answer: 6"),
        (list_example(), "Final answer: [2, 3, 1]"),
        (graph_example(), "Yes"),
    ):
        assert set(score(example, completion)) == expected_keys
