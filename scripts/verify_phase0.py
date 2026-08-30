"""Write one deterministic Phase 0 numerical report."""

import argparse
import json
from pathlib import Path

from ppmoe.phase0 import run_phase0_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True, choices=("float32", "bfloat16"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--hidden-size", required=True, type=int)
    parser.add_argument("--intermediate-size", required=True, type=int)
    parser.add_argument("--num-experts", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = run_phase0_probe(
        device=arguments.device,
        dtype=arguments.dtype,
        seed=arguments.seed,
        hidden_size=arguments.hidden_size,
        intermediate_size=arguments.intermediate_size,
        num_experts=arguments.num_experts,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
