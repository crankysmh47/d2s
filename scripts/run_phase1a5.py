"""CLI for the gated Phase 1A.5 calibration stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ppmoe.experiments.phase1a5 import StageError, run_phase1a5
from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.experiments.phase1a5_provenance import SourceVerificationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase1-v0_5.toml"))
    parser.add_argument(
        "--stage",
        required=True,
        choices=("source", "smoke", "local", "select", "rollout", "report"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=1)
    arguments = parser.parse_args(argv)
    try:
        config = Phase1A5Config.load(arguments.config)
        result = run_phase1a5(
            config=config,
            stage=arguments.stage,
            resume=arguments.resume,
            artifact_root=arguments.artifact_root,
            microbatch_size=arguments.microbatch_size,
        )
    except (StageError, SourceVerificationError, ValueError, OSError) as error:
        print(f"Phase 1A.5 {arguments.stage} failed: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
