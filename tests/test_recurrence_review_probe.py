from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from ppmoe.recurrence.provenance import canonical_json
from ppmoe.recurrence.review_probe import (
    ProbeDependencies,
    load_config,
    run_probe,
    select_balanced_examples,
)


REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FAMILIES = ("expressions", "lists", "graphs")


def _fixture(tmp_path: Path, *, max_output: int = 4, max_total: int = 8) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "source-run"
    snapshot = run_root / "source" / REVISION
    snapshot.mkdir(parents=True)
    model_file = snapshot / "weights.safetensors"
    model_file.write_bytes(b"local test fixture")
    tokenizer_file = snapshot / "tokenizer.json"
    tokenizer_file.write_bytes(b"test tokenizer")
    source_manifest = {
        "schema_version": 1,
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": REVISION,
        "model_class": "Qwen3ForCausalLM",
        "files": [{
            "relative_path": "weights.safetensors",
            "bytes": model_file.stat().st_size,
            "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        }, {
            "relative_path": "tokenizer.json",
            "bytes": tokenizer_file.stat().st_size,
            "sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        }],
        "code_files": [],
    }
    (run_root / "source.json").write_bytes(canonical_json(source_manifest))
    source_hash = hashlib.sha256((run_root / "source.json").read_bytes()).hexdigest()
    (run_root / "stages").mkdir()
    (run_root / "stages" / "source.json").write_bytes(canonical_json({
        "status": "passed",
        "payload": {
            "revision": REVISION,
            "snapshot_relative_path": f"source/{REVISION}",
            "source_manifest_sha256": source_hash,
            "tokenizer_revision": REVISION,
            "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        },
        "signatures": {"source": source_hash},
    }))
    (run_root / "data").mkdir()
    rows = [
        {
            "id": f"{family}-{index}", "family": family, "split": "dev",
            "structure_id": f"s-{index}", "prompt": f"question {index}",
            "answer": str(index), "latent_spec": {"shape": "chain"},
        }
        for family in FAMILIES for index in range(2)
    ]
    dev_path = run_root / "data" / "dev.jsonl"
    dev_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dev_hash = hashlib.sha256(dev_path.read_bytes()).hexdigest()
    data_manifest = {"schema_version": 1, "split_sha256": {"dev": dev_hash}}
    manifest_path = run_root / "data" / "manifest.json"
    manifest_path.write_bytes(canonical_json(data_manifest))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (run_root / "stages" / "data.json").write_bytes(canonical_json({
        "status": "passed",
        "payload": {"manifest_sha256": manifest_hash},
        "signatures": {"source": source_hash, "data": "d" * 64},
    }))
    config_path = tmp_path / "probe.toml"
    config_path.write_text(
        "\n".join((
            "[probe]",
            f'source_run = "{run_root.as_posix()}"',
            f'source_revision = "{REVISION}"',
            f'output_dir = "{(tmp_path / "review-output").as_posix()}"',
            f"max_output_tokens = {max_output}",
            f"max_total_tokens = {max_total}",
            "examples_per_family = 1",
            "seed = 9158",
            "deadline_seconds = 30",
            'families = ["expressions", "lists", "graphs"]',
            "",
        )),
        encoding="utf-8",
    )
    return config_path, run_root, config_path.parent / "review-output"


def test_balanced_selection_is_seeded_and_covers_each_family(tmp_path: Path) -> None:
    config_path, _, _ = _fixture(tmp_path)
    config = load_config(config_path)
    rows = [
        {"id": f"{family}-{index}", "family": family, "split": "dev"}
        for family in FAMILIES for index in range(8)
    ]
    first = select_balanced_examples(rows, config)
    assert [row["id"] for row in first] == [row["id"] for row in select_balanced_examples(rows, config)]
    assert {family: sum(row["family"] == family for row in first) for family in FAMILIES} == {
        family: 1 for family in FAMILIES
    }


def test_checked_in_probe_config_has_frozen_diagnostic_values() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "recurrence-review.toml")
    assert config.max_output_tokens == 256
    assert config.max_total_tokens == 512
    assert config.examples_per_family == 10
    assert config.seed == 9158
    assert config.deadline_seconds == 3600
    assert config.families == FAMILIES


@pytest.mark.parametrize(
    "output,total",
    [(0, 8), (9, 8), (1, 513)],
)
def test_config_rejects_invalid_token_bounds(tmp_path: Path, output: int, total: int) -> None:
    config_path, _, _ = _fixture(tmp_path, max_output=output, max_total=total)
    with pytest.raises(ValueError, match="token"):
        load_config(config_path)


def test_config_rejects_bool_integer_and_incomplete_family_coverage(tmp_path: Path) -> None:
    config_path, _, _ = _fixture(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace("max_output_tokens = 4", "max_output_tokens = true"), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an integer"):
        load_config(config_path)
    config_path.write_text(
        text.replace('families = ["expressions", "lists", "graphs"]', 'families = ["expressions"]'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="family coverage"):
        load_config(config_path)


class _Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, *_: object, **__: object) -> list[int]:
        return [1]

    def __call__(self, answer: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [int(answer) + 3]}

    def decode(self, tokens: list[int], **_: object) -> str:
        return " ".join(str(token - 3) for token in tokens if token != 2)


class _Model:
    config = type("Config", (), {"eos_token_id": 2})()

    def eval(self) -> _Model:
        self.is_eval = True
        return self

    def __call__(self, input_ids: torch.Tensor, mode: str) -> torch.Tensor:
        logits = torch.zeros((1, input_ids.shape[1], 8))
        logits[0, -1, 4] = 1
        return logits


def _dependencies(*, clock=lambda: 10.0) -> ProbeDependencies:
    return ProbeDependencies(
        tokenizer_loader=lambda *_args, **_kwargs: _Tokenizer(),
        model_loader=lambda *_args, **_kwargs: object(),
        model_factory=lambda _stock, _config: _Model(),
        device="cpu",
        hardware_probe=lambda: {"device": "test-cpu", "vram_bytes": 0},
        clock=clock,
        synchronize=lambda: None,
        progress=lambda _message: None,
    )


def test_mocked_probe_writes_all_records_and_blank_review_without_gpu(tmp_path: Path) -> None:
    config_path, _, output_dir = _fixture(tmp_path)
    result = run_probe(config_path, dependencies=_dependencies())
    assert result["status"] == "complete"
    rows = [json.loads(line) for line in (output_dir / "answers.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {row["family"] for row in rows} == set(FAMILIES)
    assert all(row["mode"] == "single" and row["split"] == "dev" for row in rows)
    assert all(row["completion_text"] == "1 1 1 1" for row in rows)
    assert (output_dir / "human-review" / "review.md").is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["record_count"] == 3


def test_probe_refuses_existing_output_directory(tmp_path: Path) -> None:
    config_path, _, output_dir = _fixture(tmp_path)
    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        run_probe(config_path, dependencies=_dependencies())


def test_probe_rejects_dev_hash_mismatch_before_creating_output(tmp_path: Path) -> None:
    config_path, run_root, output_dir = _fixture(tmp_path)
    manifest_path = run_root / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_sha256"]["dev"] = "0" * 64
    manifest_path.write_bytes(canonical_json(manifest))
    stage_path = run_root / "stages" / "data.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["payload"]["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    stage_path.write_bytes(canonical_json(stage))
    with pytest.raises(ValueError, match="dev.*hash|hash.*dev"):
        run_probe(config_path, dependencies=_dependencies())
    assert not output_dir.exists()


def test_probe_rejects_modified_pinned_model_file_before_output(tmp_path: Path) -> None:
    config_path, run_root, output_dir = _fixture(tmp_path)
    snapshot_file = run_root / "source" / REVISION / "weights.safetensors"
    snapshot_file.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="source file hash mismatch"):
        run_probe(config_path, dependencies=_dependencies())
    assert not output_dir.exists()


def test_probe_preserves_capped_answer_text_and_failure_summary(tmp_path: Path) -> None:
    config_path, _, output_dir = _fixture(tmp_path, max_output=2, max_total=8)
    deps = _dependencies()
    # This fixture's model emits a non-EOS token forever, so capped text must survive.
    result = run_probe(config_path, dependencies=deps)
    rows = [json.loads(line) for line in (output_dir / "answers.jsonl").read_text().splitlines()]
    assert result["status"] == "complete"
    assert all(row["stop_reason"] == "length" for row in rows)
    assert all(row["completion_text"] for row in rows)


def test_probe_deadline_keeps_partial_row_and_failure_summary(tmp_path: Path) -> None:
    config_path, _, output_dir = _fixture(tmp_path)
    tick = 0

    def clock() -> float:
        nonlocal tick
        tick += 5
        return float(tick)

    with pytest.raises(TimeoutError, match="during generation"):
        run_probe(config_path, dependencies=_dependencies(clock=clock))
    rows = [json.loads(line) for line in (output_dir / "answers.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["reason_code"] == "deadline_exceeded"
    assert 0 < rows[0]["generated_token_count"] < 4
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["record_count"] == 1
    assert (output_dir / "human-review" / "review.md").is_file()
