"""One-stage CLI for the independently versioned recurrence V2 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Sequence

from ppmoe.recurrence.config_v2 import load_v2_config
from ppmoe.recurrence.study_v2 import run_v2_stage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one implemented recurrence V2 stage.")
    parser.add_argument("--stage", choices=("prepare", "calibrate", "report"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/recurrence-v2.toml"))
    parser.add_argument("--run-id", help="full deterministic V2 run ID for calibrate/report")
    args = parser.parse_args(argv)
    config = load_v2_config(args.config)
    root = None
    if args.run_id is not None:
        if re.fullmatch(r"[0-9a-f]{64}", args.run_id) is None:
            parser.error("--run-id must be the full 64-character lowercase deterministic ID")
        output = Path(config.data.output_root)
        if not output.is_absolute():
            output = Path(__file__).resolve().parents[1] / output
        root = output / args.run_id
    elif args.stage != "prepare":
        parser.error(f"--run-id is required for --stage {args.stage}")
    result = run_v2_stage(args.stage, args.config, root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 1 if result.get("status") in {"failed_environment", "stopped_resource"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
