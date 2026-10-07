"""Register, launch, inspect, or internally supervise a distillation execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from ppmoe.distill.background import execution_status, launch_stage, register_execution_cli, supervise_stage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage a detached verified-distillation execution.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--config", type=Path, required=True)
    register.add_argument("--run-root", type=Path, required=True)
    register.add_argument("--executions-root", type=Path, required=True)
    register.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    launch = subparsers.add_parser("launch")
    launch.add_argument("stage")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--execution-root", type=Path, required=True)
    launch.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    status = subparsers.add_parser("status")
    status.add_argument("--execution-root", type=Path, required=True)
    status.add_argument("--config", type=Path)
    status.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    run = subparsers.add_parser("run", help=argparse.SUPPRESS)
    run.add_argument("stage")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--execution-root", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "register":
            root = register_execution_cli(
                args.run_root, args.executions_root,
                repo_root=args.repo_root, config_path=args.config,
            )
            print(json.dumps({"execution_root": str(root), "status_path": str(root / "status.md")}, sort_keys=True))
            return 0
        if args.action == "launch":
            result = launch_stage(
                args.execution_root, args.stage,
                repo_root=args.repo_root, config_path=args.config,
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if result.get("state") in {"running", "awaiting_review", "completed", "stopped_scientific"} else 2
        if args.action == "status":
            result = execution_status(
                args.execution_root, repo_root=args.repo_root, config_path=args.config,
            )
            print(json.dumps(result, sort_keys=True, indent=2))
            print(f"status.md: {result['status_md_path']}")
            return 0 if result.get("accounting_valid", True) else 2
        return supervise_stage(
            args.execution_root, args.stage, repo_root=args.repo_root,
            config_path=args.config, _attempt_id=args.attempt_id,
        )
    except (OSError, ValueError, RuntimeError, FileExistsError) as exc:
        print(f"distillation-background: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
