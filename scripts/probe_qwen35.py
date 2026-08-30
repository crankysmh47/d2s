"""Inspect Qwen3.5 configuration and a newly initialized text MLP only."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from transformers import AutoConfig, __version__ as transformers_version
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

from ppmoe.adapters.qwen35 import inspect_qwen35_config, inspect_qwen35_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-0.8B-Base",
    )
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = AutoConfig.from_pretrained(
        arguments.model_id,
        trust_remote_code=False,
    )
    config_report = inspect_qwen35_config(
        config,
        num_experts=arguments.num_experts,
    )
    text_config = getattr(config, "text_config", config)
    mlp = Qwen3_5MLP(text_config, text_config.intermediate_size)
    mlp_report = inspect_qwen35_mlp(mlp)
    report = {
        "schema_version": 1,
        "model_id": arguments.model_id,
        "transformers_version": transformers_version,
        "trust_remote_code": False,
        "weights_loaded": False,
        "config": asdict(config_report),
        "mlp": asdict(mlp_report),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
