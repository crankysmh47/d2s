from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppmoe.recurrence.data import Example
from ppmoe.recurrence.v2_runtime import ScoreExecutionError, score_v2_examples


class TinyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [1, 3]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [4]}

    def decode(self, token_ids, skip_special_tokens=True):
        return "**Answer: yes**"


class TinyModel:
    def __call__(self, prefix, mode):
        raise AssertionError("generation is replaced by the CPU fake")


@pytest.fixture
def graph_example() -> Example:
    return Example("graph-1", "graphs", "pilot", "n7", "Directed edges: A->B. Is there a directed path from A to B? Return only yes or no.", "yes", {
        "n": 7, "edges": ["A->B"], "start": "A", "end": "B",
    })


def test_v2_runtime_scores_markdown_graph_answer_and_fsyncs_each_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph_example: Example,
) -> None:
    import ppmoe.recurrence.v2_runtime as runtime

    writes: list[bytes] = []
    real_fsync = runtime.os.fsync

    def track_fsync(fd: int) -> None:
        writes.append(b"fsync")
        real_fsync(fd)

    observed: dict[str, object] = {}
    def generate(_model, input_ids, mode, _eos_id, max_new_tokens, **_kwargs):
        observed.update({"input_ids": input_ids.tolist(), "mode": mode, "max_new_tokens": max_new_tokens})
        return {"token_ids": [4, 2], "completion_token_ids": [4], "stop_reason": "eos", "generated_token_count": 2}

    monkeypatch.setattr(runtime, "generate_answer", generate)
    monkeypatch.setattr(runtime.os, "fsync", track_fsync)
    monkeypatch.setattr(runtime, "_cuda_event", lambda: None)
    monkeypatch.setattr(runtime, "_finish_cuda_events", lambda _end, _start: 1.25)
    output = tmp_path / "scores.jsonl"
    result = score_v2_examples(TinyModel(), TinyTokenizer(), [graph_example], "single", output, {"source": "a" * 64})
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["validated_count"] == 1
    assert rows[0]["completion_text"] == "**Answer: yes**"
    assert rows[0]["reason_code"] == "correct" and rows[0]["parsed_value"] == "yes"
    assert len(writes) == 1
    assert result["cuda_active_seconds"] == 1.25
    assert observed == {"input_ids": [[1, 3]], "mode": "single", "max_new_tokens": 256}


def test_v2_runtime_preserves_partial_rows_after_generation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph_example: Example,
) -> None:
    import ppmoe.recurrence.v2_runtime as runtime

    second = Example("graph-2", "graphs", "pilot", "n7", graph_example.prompt, "yes", graph_example.latent_spec)
    calls = 0

    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic model failure")
        return {"token_ids": [4, 2], "completion_token_ids": [4], "stop_reason": "eos", "generated_token_count": 2}

    monkeypatch.setattr(runtime, "generate_answer", generate)
    output = tmp_path / "partial.jsonl"
    with pytest.raises(ScoreExecutionError) as caught:
        score_v2_examples(TinyModel(), TinyTokenizer(), [graph_example, second], "single", output, {})
    assert caught.value.row_count == 1
    assert [json.loads(line)["example_id"] for line in output.read_text(encoding="utf-8").splitlines()] == ["graph-1"]


def test_v2_runtime_refuses_overwrite_and_non_greedy_modes(
    tmp_path: Path, graph_example: Example,
) -> None:
    output = tmp_path / "scores.jsonl"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        score_v2_examples(TinyModel(), TinyTokenizer(), [graph_example], "single", output, {})
    with pytest.raises(ValueError, match="single"):
        score_v2_examples(TinyModel(), TinyTokenizer(), [graph_example], "sample", tmp_path / "other.jsonl", {})
