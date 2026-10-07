from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ppmoe.distill.config import load_config
from ppmoe.distill.model import HuggingFaceWorker, build_prompt_ids


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDENT_SNAPSHOT = (
    REPO_ROOT
    / "artifacts/recurrence-v0/129d39646933/source/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
STUDENT_INVENTORY = (
    REPO_ROOT
    / "artifacts/distillation-v1-executions/3116768bd90444328e1ec0e4f8b518a18c238ed74e643d24ae0a69affd0683eb"
    / "calibration/model-inventory/student.json"
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
OPTIONAL_TOKENIZER_FILES = ("chat_template.jinja", "special_tokens_map.json")
QUESTION = "What is 1 plus 1?"
EXPECTED_PROMPT_TOKENS = 39
EXPECTED_EOS_IDS = (151643, 151645)


def _authenticate_tokenizer_files(snapshot: Path, inventory: dict) -> None:
    if inventory.get("repo_id") != "Qwen/Qwen3-1.7B" or inventory.get("resolved_sha") != snapshot.name:
        raise ValueError("student tokenizer inventory identity does not match snapshot")
    entries = {entry.get("path"): entry for entry in inventory.get("files", [])}
    present = set(TOKENIZER_FILES)
    present.update(
        name for name in OPTIONAL_TOKENIZER_FILES
        if (snapshot / name).is_file() or name in entries
    )
    for name in sorted(present):
        entry = entries.get(name)
        path = snapshot / name
        if entry is None or not path.is_file():
            raise ValueError(f"student tokenizer inventory is missing {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"student tokenizer inventory hash mismatch for {name}")


def _load_authenticated_tokenizer(snapshot: Path, inventory: dict, loader):
    _authenticate_tokenizer_files(snapshot, inventory)
    return loader(snapshot, local_files_only=True)


def _cached_inputs():
    if not STUDENT_SNAPSHOT.is_dir() or not STUDENT_INVENTORY.is_file():
        pytest.skip("retained pinned student tokenizer cache is unavailable")
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    return transformers, torch, json.loads(STUDENT_INVENTORY.read_text(encoding="utf-8"))


def _prompt_config():
    return load_config(REPO_ROOT / "configs/distillation-v1.toml").prompt


def test_cached_tokenizer_matches_independent_list_and_default_encoding_contract() -> None:
    transformers, _torch, inventory = _cached_inputs()
    tokenizer = _load_authenticated_tokenizer(
        STUDENT_SNAPSHOT,
        inventory,
        lambda path, **kwargs: transformers.AutoTokenizer.from_pretrained(path, **kwargs),
    )
    prompt_config = _prompt_config()
    prompt = prompt_config.template.format(question=QUESTION)
    messages = [{"role": "user", "content": prompt}]
    helper_ids = build_prompt_ids(tokenizer, prompt_config, QUESTION)
    explicit_list_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    default_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert helper_ids == explicit_list_ids == default_encoding.input_ids
    assert len(helper_ids) == EXPECTED_PROMPT_TOKENS


@pytest.mark.parametrize("failure", ["missing-entry", "tampered-hash"])
def test_tokenizer_inventory_failure_precedes_tokenizer_load(failure: str) -> None:
    _transformers, _torch, inventory = _cached_inputs()
    changed = json.loads(json.dumps(inventory))
    if failure == "missing-entry":
        changed["files"] = [entry for entry in changed["files"] if entry["path"] != "tokenizer.json"]
    else:
        next(entry for entry in changed["files"] if entry["path"] == "tokenizer.json")["sha256"] = "0" * 64
    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("tokenizer loader must not run before inventory authentication")

    with pytest.raises(ValueError, match="inventory"):
        _load_authenticated_tokenizer(STUDENT_SNAPSHOT, changed, loader)
    assert calls == []


def test_cpu_worker_preserves_generation_boundary_with_real_tokenizer(monkeypatch) -> None:
    transformers, torch, inventory = _cached_inputs()
    tokenizer = _load_authenticated_tokenizer(
        STUDENT_SNAPSHOT,
        inventory,
        lambda path, **kwargs: transformers.AutoTokenizer.from_pretrained(path, **kwargs),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    prompt_config = _prompt_config()
    prompt_ids = build_prompt_ids(tokenizer, prompt_config, QUESTION)
    content_ids = tokenizer.encode("Final answer: 2", add_special_tokens=False)
    expected_text = "Final answer: 2"

    class ControlledModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = SimpleNamespace(eos_token_id=[151645, 151643])
            self.calls = []
            self.next_generated_ids = content_ids + list(EXPECTED_EOS_IDS)

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            input_ids = kwargs["input_ids"]
            suffix = torch.tensor([self.next_generated_ids], dtype=torch.long, device="cpu")
            return transformers.generation.utils.GenerateDecoderOnlyOutput(
                sequences=torch.cat((input_ids, suffix), dim=1)
            )

    controlled = ControlledModel()
    worker = HuggingFaceWorker(
        "student", controlled, tokenizer, prompt_config, torch, eos_token_ids=EXPECTED_EOS_IDS,
    )
    result = worker.generate({"question": QUESTION}, (256, 768))
    call = controlled.calls[-1]
    assert call["input_ids"].device.type == "cpu"
    assert call["input_ids"].dtype == torch.long
    assert tuple(call["input_ids"].shape) == (1, EXPECTED_PROMPT_TOKENS)
    assert call["input_ids"].tolist() == [prompt_ids]
    assert call["max_new_tokens"] == 256
    assert call["do_sample"] is False
    assert call["num_beams"] == 1
    assert call["use_cache"] is True
    assert call["return_dict_in_generate"] is True
    assert call["eos_token_id"] == list(EXPECTED_EOS_IDS)
    assert result["text"] == expected_text
    assert result["stop_reason"] == "eos"
    assert result["generated_tokens"] == len(content_ids) + 2

    controlled.next_generated_ids = [content_ids[0]] * 256
    capped = worker.generate({"question": QUESTION}, (256, 768))
    assert capped["stop_reason"] == "length"
    assert capped["generated_tokens"] == 256

    before_overflow = len(controlled.calls)
    overflow = worker.generate({"question": "x " * 900}, (256, 768))
    assert overflow["stop_reason"] == "budget_overflow"
    assert len(controlled.calls) == before_overflow
