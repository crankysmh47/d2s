import json
from pathlib import Path
import subprocess
import sys

from transformers import Qwen3_5Config, Qwen3_5TextConfig


def test_qwen35_cli_inspects_local_real_transformers_config(tmp_path: Path):
    model_directory = tmp_path / "model"
    output_path = tmp_path / "report.json"
    text_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["full_attention", "full_attention"],
    )
    Qwen3_5Config(text_config=text_config).save_pretrained(model_directory)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/probe_qwen35.py",
            "--model-id",
            str(model_directory),
            "--num-experts",
            "4",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["config"]["hidden_size"] == 16
    assert report["config"]["intermediate_size"] == 32
    assert report["config"]["expert_width"] == 8
    assert report["mlp"]["gate_weight_shape"] == [32, 16]
    assert report["mlp"]["down_weight_shape"] == [16, 32]
    assert report["mlp"]["gate_bias"] is False
    assert report["weights_loaded"] is False
