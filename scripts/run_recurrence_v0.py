"""Thin, lazy CLI for one registered recurrence stage."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


_STAGES = ("source", "data", "smoke", "pilot", "frozen", "discover", "confirm", "report")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exactly one sealed recurrence-v0 stage.")
    parser.add_argument("--stage", choices=_STAGES, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/recurrence-v0.toml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/recurrence-v0"))
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id is not None and __import__("re").fullmatch(r"[0-9a-f]{12}", args.run_id) is None:
        print("run_id must be exactly 12-hex lowercase characters", file=sys.stderr)
        return 1
    if args.stage not in {"source", "data", "report"} and args.run_id is None:
        print(f"--run-id is required for {args.stage} after data finalization", file=sys.stderr)
        return 1
    # Delayed import is intentional: --help must work on a CPU-only host with
    # neither Torch initialization nor artifact creation.
    from ppmoe.recurrence.experiment import BudgetStop, EnvironmentStop, ScientificStop, StageError, _control_hashes, _staging_id, run_stage
    config = args.config.resolve(); root = args.artifact_root.resolve()
    print(f"CONFIG={config}"); print(f"ARTIFACT_ROOT={root}")
    if args.stage in {"source", "data"} or (args.stage == "report" and args.run_id is None):
        from ppmoe.recurrence.config import RecurrenceConfig
        config_hash, code_hash = _control_hashes(config)
        staging_id = _staging_id(RecurrenceConfig.load(config).model.repo_id, config_hash, code_hash)
        print(f"STAGING_ID={staging_id}"); print(f"STAGING_ROOT={(root / 'staging' / staging_id).resolve()}")
    try:
        result = run_stage(args.stage, config, root, args.run_id)
    except ScientificStop as error:
        print(str(error), file=sys.stderr); return 2
    except BudgetStop as error:
        print(str(error), file=sys.stderr); return 3
    except EnvironmentStop as error:
        print(str(error), file=sys.stderr); return 1
    except StageError as error:
        print(str(error), file=sys.stderr); return 1
    final_id = args.run_id
    if args.stage == "data":
        # Final signatures are immutable evidence; recompute the registered ID.
        from ppmoe.recurrence.provenance import compute_run_id
        signatures = result["signatures"]
        final_id = compute_run_id(signatures["source"], signatures["data"], signatures["config"], signatures["code"])
    if final_id:
        print(f"RUN_ID={final_id}"); print(f"RUN_ROOT={(root / final_id).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
