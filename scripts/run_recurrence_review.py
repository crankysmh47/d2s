"""Run the isolated 256-token development review diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded F0 longer-output diagnostic on dev examples only."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/recurrence-review.toml"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from ppmoe.recurrence.review_probe import run_probe

    try:
        result = run_probe(args.config)
    except (FileExistsError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"review probe failed: {error}", file=sys.stderr)
        return 1
    print(f"REVIEW_PROBE_STATUS={result['status']}")
    print(f"REVIEW_PROBE_RECORDS={result['record_count']}")
    print(f"REVIEW_PROBE_OUTPUT={result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
