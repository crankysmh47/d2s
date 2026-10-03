from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from ppmoe.distill.model import (
    GenerationBudgetError,
    build_prompt_ids,
    generate_one,
    generation_stop_reason,
    pinned_eos_token_ids,
    verify_student_is_frozen,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return [10, 11, 12]


class FakeParam:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, *, trainable: bool = False) -> None:
        self._parameters = {"base": FakeParam(trainable)}

    def named_parameters(self):
        return self._parameters.items()

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return {"text": "Final answer: 7", "stop_reason": "eos", "generated_tokens": 4}


@dataclass(frozen=True)
class FakePromptConfig:
    template: str = "Solve: {question}"
    thinking: bool = False


def test_prompt_uses_chat_template_once_without_duplicate_special_tokens() -> None:
    tokenizer = FakeTokenizer()
    ids = build_prompt_ids(tokenizer, FakePromptConfig(), "2+5?")
    assert ids == [10, 11, 12]
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0]["tokenize"] is True
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[0]["enable_thinking"] is False


def test_prompt_overflow_is_rejected_before_generation() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    ids = build_prompt_ids(tokenizer, FakePromptConfig(), "2+5?")
    with pytest.raises(GenerationBudgetError, match="prompt.*budget"):
        generate_one(model, ids, max_prompt_tokens=2, max_new_tokens=3, candidate_total=5)
    assert not hasattr(model, "generation_kwargs")


def test_generation_uses_greedy_settings_and_no_prompt_truncation() -> None:
    model = FakeModel()
    result = generate_one(model, [10, 11], max_prompt_tokens=2, max_new_tokens=4, candidate_total=6)
    assert result.text == "Final answer: 7"
    assert model.generation_kwargs["do_sample"] is False
    assert model.generation_kwargs["use_cache"] is True
    assert model.generation_kwargs["max_new_tokens"] == 4
    assert "truncation" not in model.generation_kwargs


def test_student_base_must_have_no_trainable_parameters() -> None:
    verify_student_is_frozen(FakeModel())
    with pytest.raises(ValueError, match="trainable"):
        verify_student_is_frozen(FakeModel(trainable=True))


def test_effective_alternate_eos_wins_at_exact_generation_budget_boundary() -> None:
    assert generation_stop_reason([10, 11, 99], [2, 99], max_new_tokens=3) == "eos"
    assert generation_stop_reason([10, 11, 12], [2, 99], max_new_tokens=3) == "length"
    assert generation_stop_reason([99], 99, max_new_tokens=3) == "eos"


def test_pinned_generation_config_accepts_multiple_effective_eos_ids(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [9, 3, 9]}), encoding="utf-8")
    assert pinned_eos_token_ids(tmp_path) == (3, 9)
