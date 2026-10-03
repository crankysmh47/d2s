"""Run one registered source/config stage for verified-distillation v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from ppmoe.distill.study import KNOWN_STAGES, StageError, run_stage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one verified-distillation v1 stage.")
    parser.add_argument("--stage", required=True, choices=sorted(KNOWN_STAGES))
    parser.add_argument("--config", type=Path, default=Path("configs/distillation-v1.toml"))
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args(argv)
    try:
        record = run_stage(args.stage, args.config, args.run_root)
    except (StageError, ValueError, FileExistsError, RuntimeError) as exc:
        print(f"distillation-v1: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
