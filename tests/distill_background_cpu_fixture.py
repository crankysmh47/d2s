"""Test-only detached CPU supervisor fixture; never invokes a model or network."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _execution_fixture(workspace: Path) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "distill_task1_fixture", TESTS / "test_distill_execution.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = REPO_ROOT / "configs" / "distillation-v1.toml"
    source, _ = module._prepared_source(workspace / "source fixture", config)
    from ppmoe.distill import execution, study

    terminal_path = source / "stages" / "calibrate.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["signature"]["code_signatures"] = {
        config.resolve().relative_to(REPO_ROOT.resolve()).as_posix():
            __import__("hashlib").sha256(config.read_bytes()).hexdigest(),
    }
    terminal_path.write_bytes(_canonical(terminal) + b"\n")

    study.TASK1_MANIFEST_SHA256 = __import__("hashlib").sha256((source / "manifest.json").read_bytes()).hexdigest()
    study.TASK1_PREPARE_SHA256 = __import__("hashlib").sha256((source / "stages" / "prepare.json").read_bytes()).hexdigest()
    execution.TASK1_STOP_SHA256 = __import__("hashlib").sha256(terminal_path.read_bytes()).hexdigest()
    root = execution.register_execution(
        source, workspace / "executions", repo_root=REPO_ROOT, config_path=config,
    )
    runner_dir = workspace / "fake stage path with spaces"
    runner_dir.mkdir(parents=True, exist_ok=True)
    runner = runner_dir / "fake stage.py"
    runner.write_text(
        "import json,sys,time\n"
        "from pathlib import Path\n"
        "root=Path(sys.argv[1]); signature=sys.argv[2]\n"
        "print('started', flush=True); time.sleep(float(sys.argv[3]))\n"
        "record={'execution_id':root.name,'execution_stage_signature':signature,'status':'awaiting_review','reason':'CPU fixture complete','review_packet':'calibration/review/fixture.csv','private_review_map':'calibration/review/private-map.json'}\n"
        "target=root/'stages'/'calibrate.json'; target.parent.mkdir(parents=True,exist_ok=True)\n"
        "target.write_bytes(json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\\n')\n"
        "print('finished', flush=True)\n",
        encoding="utf-8",
    )
    fixture = {
        "execution_root": str(root), "repo_root": str(REPO_ROOT),
        "config_path": str(config), "source_root": str(source),
        "manifest_sha256": study.TASK1_MANIFEST_SHA256,
        "prepare_sha256": study.TASK1_PREPARE_SHA256,
        "stop_sha256": execution.TASK1_STOP_SHA256,
        "runner": str(runner),
        "duration_seconds": 2.5,
    }
    (workspace / "fixture.json").write_bytes(_canonical(fixture) + b"\n")
    return fixture


def _load_fixture(workspace: Path) -> dict[str, str]:
    return json.loads((workspace / "fixture.json").read_text(encoding="utf-8"))


def _patch_fixture_constants(fixture: dict[str, str]) -> None:
    from ppmoe.distill import execution, study

    study.TASK1_MANIFEST_SHA256 = fixture["manifest_sha256"]
    study.TASK1_PREPARE_SHA256 = fixture["prepare_sha256"]
    execution.TASK1_STOP_SHA256 = fixture["stop_sha256"]


def _argv_builder(fixture: dict[str, str]):
    def build(_repo_root: Path, _manifest: dict[str, object], _attempt_id: str) -> list[str]:
        from ppmoe.distill.execution import execution_stage_signature

        signature = execution_stage_signature(
            Path(fixture["execution_root"]), "calibrate", repo_root=Path(fixture["repo_root"]),
        )
        return [sys.executable, fixture["runner"], fixture["execution_root"], signature,
                str(fixture.get("duration_seconds", 2.5))]
    return build


def _run_supervisor(workspace: Path, attempt_id: str) -> int:
    fixture = _load_fixture(workspace)
    _patch_fixture_constants(fixture)
    from ppmoe.distill.background import supervise_stage

    return supervise_stage(
        Path(fixture["execution_root"]), "calibrate",
        repo_root=Path(fixture["repo_root"]), config_path=Path(fixture["config_path"]),
        _argv_builder=_argv_builder(fixture), _attempt_id=attempt_id,
    )


def _start(workspace: Path, duration_seconds: float | None = None) -> dict[str, object]:
    workspace.mkdir(parents=True, exist_ok=True)
    fixture = _load_fixture(workspace) if (workspace / "fixture.json").is_file() else _execution_fixture(workspace)
    if duration_seconds is not None:
        fixture["duration_seconds"] = duration_seconds
        (workspace / "fixture.json").write_bytes(_canonical(fixture) + b"\n")
    _patch_fixture_constants(fixture)
    from ppmoe.distill.background import launch_stage

    def command_builder(_root: Path, _stage: str, _config: Path, _repo: Path, attempt_id: str) -> list[str]:
        return [sys.executable, str(Path(__file__).resolve()), "run", "--workspace",
                str(workspace), "--attempt-id", attempt_id]

    result = launch_stage(
        Path(fixture["execution_root"]), "calibrate", repo_root=REPO_ROOT,
        config_path=Path(fixture["config_path"]), _supervisor_argv_builder=command_builder,
    )
    result["workspace"] = str(workspace)
    result["stdout_path"] = str(
        Path(fixture["execution_root"]) / "logs" / f"{result['attempt_id']}.stdout.log"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--workspace", type=Path, required=True)
    start = sub.add_parser("start")
    start.add_argument("--workspace", type=Path, required=True)
    start.add_argument("--duration", type=float)
    status = sub.add_parser("status")
    status.add_argument("--workspace", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(_execution_fixture(args.workspace), sort_keys=True))
        return 0
    if args.action == "start":
        print(json.dumps(_start(args.workspace, args.duration), sort_keys=True))
        return 0
    if args.action == "run":
        return _run_supervisor(args.workspace, args.attempt_id)
    fixture = _load_fixture(args.workspace)
    _patch_fixture_constants(fixture)
    from ppmoe.distill.background import execution_status

    print(json.dumps(execution_status(
        Path(fixture["execution_root"]), repo_root=Path(fixture["repo_root"]),
        config_path=Path(fixture["config_path"]),
    ), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
