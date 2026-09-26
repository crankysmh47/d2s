"""One-stage CLI for the independently versioned recurrence V1 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Sequence

from ppmoe.recurrence.config_v1 import load_v1_config
from ppmoe.recurrence.study_v1 import run_v1_stage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one implemented recurrence V1 stage.")
    parser.add_argument("--stage", choices=("prepare", "calibrate", "report"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", help="existing deterministic V1 run ID for calibrate/report")
    args = parser.parse_args(argv)
    config = load_v1_config(args.config)
    root = None
    if args.run_id is not None:
        if re.fullmatch(r"[0-9a-f]{64}", args.run_id) is None:
            parser.error("--run-id must be the full 64-character lowercase deterministic ID")
        root_value = Path(config.data.output_root)
        if not root_value.is_absolute():
            root_value = Path(__file__).resolve().parents[1] / root_value
        root = root_value / args.run_id
    elif args.stage != "prepare":
        parser.error(f"--run-id is required for --stage {args.stage}")
    result = run_v1_stage(args.stage, args.config, root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 1 if result.get("status") in {"failed_environment", "stopped_resource"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
