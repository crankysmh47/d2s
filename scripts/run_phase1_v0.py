"""CLI for the gated Phase 1 V0 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ppmoe.config import Phase1Config
from ppmoe.experiments.phase1_v0 import StageError, run_phase1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase1-v0.toml"))
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "datasets", "m0", "local-smoke", "local", "rollout", "report"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--microbatch-size", type=int, default=1)
    arguments = parser.parse_args(argv)
    try:
        config = Phase1Config.load(arguments.config)
        artifact_root = arguments.artifact_root or config.artifact_root
        result = run_phase1(
            config, arguments.stage, arguments.resume, artifact_root, arguments.microbatch_size
        )
    except (StageError, ValueError, OSError) as error:
        print(f"Phase 1 {arguments.stage} failed: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
