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
    parser.add_argument("stage", choices=sorted(KNOWN_STAGES))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--execution-root", type=Path)
    parser.add_argument("--attempt-id")
    args = parser.parse_args(argv)
    if args.stage in {"prepare", "calibrate"} and args.run_root is None:
        parser.error(f"--run-root is required for {args.stage}")
    if (args.execution_root is None) != (args.attempt_id is None):
        parser.error("--execution-root and --attempt-id must be supplied together")
    if args.execution_root is not None and args.stage != "calibrate":
        parser.error("separate execution mode currently supports calibrate only")
    try:
        if args.execution_root is None:
            record = run_stage(args.stage, args.config, args.run_root)
        else:
            record = run_stage(
                args.stage, args.config, args.run_root,
                execution_root=args.execution_root, attempt_id=args.attempt_id,
            )
    except (StageError, ValueError, FileExistsError, RuntimeError) as exc:
        print(f"distillation-v1: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
