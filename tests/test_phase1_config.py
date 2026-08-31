from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from ppmoe.config import Phase1Config
from ppmoe.provenance import atomic_write_json, sha256_file


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "phase1-v0.toml"


def test_phase1_config_loads_locked_protocol() -> None:
    config = Phase1Config.load(CONFIG_PATH)

    assert config.name == "qwen35_0p8b_ppmoe_oracle_v0"
    assert config.model.repo_id == "Qwen/Qwen3.5-0.8B-Base"
    assert config.model.revision == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    assert config.fineweb.repo_id == "HuggingFaceFW/fineweb-edu"
    assert config.fineweb.name == "sample-10BT"
    assert config.fineweb.split == "train"
    assert config.fineweb.revision == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert config.wikitext.name == "wikitext-103-raw-v1"
    assert config.wikitext.split == "test"
    assert config.wikitext.revision == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert config.k_values == (8, 6, 4, 3, 2)
    assert config.calibration_tokens == 100_000
    assert config.heldout_tokens == 100_000
    assert config.smoke_tokens == 2_048
    assert config.context_length == 512
    assert config.num_experts == 8
    assert config.num_hidden_layers == 24
    assert config.intermediate_size == 3_584
    assert config.expert_width == 448
    assert config.dtype == "bfloat16"
    assert config.artifact_root == Path("artifacts/phase1-v0")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: replace(
                value,
                model=replace(value.model, revision="wrong"),
            ),
            "model revision",
        ),
        (lambda value: replace(value, k_values=(8, 6, 6)), "k_values"),
        (lambda value: replace(value, intermediate_size=3_583), "divisible"),
        (lambda value: replace(value, num_hidden_layers=23), "num_hidden_layers"),
        (lambda value: replace(value, calibration_tokens=0), "token targets"),
        (
            lambda value: replace(value, artifact_root=Path("C:/outside")),
            "artifact_root",
        ),
    ],
)
def test_phase1_config_rejects_protocol_mutation(mutation, message: str) -> None:
    config = Phase1Config.load(CONFIG_PATH)

    with pytest.raises(ValueError, match=message):
        mutation(config)


def test_atomic_json_is_canonical_hashed_and_replaced(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "manifest.json"

    first_hash = atomic_write_json(output, {"b": 2, "a": 1})

    expected = b'{"a":1,"b":2}\n'
    assert output.read_bytes() == expected
    assert first_hash == sha256(expected).hexdigest()
    assert sha256_file(output) == first_hash

    second_hash = atomic_write_json(output, {"state": "complete"})

    assert output.read_bytes() == b'{"state":"complete"}\n'
    assert second_hash == sha256_file(output)
    assert list(output.parent.glob("*.tmp")) == []
