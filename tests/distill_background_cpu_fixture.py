"""Test-only detached CPU supervisor fixture; never invokes a model or network."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _execution_fixture(workspace: Path, *, recovery: bool = False, repair: bool = False) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "distill_task1_fixture", TESTS / "test_distill_execution.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = REPO_ROOT / "configs" / "distillation-v1.toml"
    from ppmoe.distill import execution, study
    if repair:
        from ppmoe.distill.background import register_repair_cli
        predecessor_id = "b0c082cba66a37f6f49bacc4bee182102b926ac5ac3af4c0a657823213e1802f"
        source = workspace / "executions" / predecessor_id
        source.parent.mkdir(parents=True)
        shutil.copytree(REPO_ROOT / "artifacts" / "distillation-v1-executions" / predecessor_id, source)
        root = register_repair_cli(source, source.parent, repo_root=REPO_ROOT, config_path=config)
        source_root = Path(json.loads((source / "execution.json").read_text(encoding="utf-8"))["source_root"])
    elif recovery:
        from ppmoe.distill.background import register_recovery_cli
        predecessor_id = "fc901bb0766a9c79fb205302d220fe727a99ebbd0b315e6ee4ad4e66e6733634"
        source = workspace / "executions" / predecessor_id
        source.parent.mkdir(parents=True)
        shutil.copytree(REPO_ROOT / "artifacts" / "distillation-v1-executions" / predecessor_id, source)
        root = register_recovery_cli(source, source.parent, repo_root=REPO_ROOT, config_path=config)
        source_root = Path(json.loads((source / "execution.json").read_text(encoding="utf-8"))["source_root"])
    else:
        source_root, _ = module._prepared_source(workspace / "source fixture", config)
        terminal_path = source_root / "stages" / "calibrate.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["signature"]["code_signatures"] = {
            config.resolve().relative_to(REPO_ROOT.resolve()).as_posix():
                __import__("hashlib").sha256(config.read_bytes()).hexdigest(),
        }
        terminal_path.write_bytes(_canonical(terminal) + b"\n")

        study.TASK1_MANIFEST_SHA256 = __import__("hashlib").sha256((source_root / "manifest.json").read_bytes()).hexdigest()
        study.TASK1_PREPARE_SHA256 = __import__("hashlib").sha256((source_root / "stages" / "prepare.json").read_bytes()).hexdigest()
        execution.TASK1_STOP_SHA256 = __import__("hashlib").sha256(terminal_path.read_bytes()).hexdigest()
        root = execution.register_execution(
            source_root, workspace / "executions", repo_root=REPO_ROOT, config_path=config,
        )
    runner_dir = workspace / "fake stage path with spaces"
    runner_dir.mkdir(parents=True, exist_ok=True)
    runner = runner_dir / "fake stage.py"
    runner.write_text(
        "import hashlib,json,sys,time\n"
        "from pathlib import Path\n"
        "from ppmoe.distill.study import shared_persist_calibration_terminal,_utc_now\n"
        "root=Path(sys.argv[1]); signature=json.loads(sys.argv[2]); manifest=json.loads((root/'execution.json').read_text(encoding='utf-8'))\n"
        "started_at=_utc_now()\n"
        "print('started', flush=True); time.sleep(float(sys.argv[3]))\n"
        "packet=root/'calibration'/'review'/'fixture.csv'; packet.parent.mkdir(parents=True,exist_ok=True)\n"
        "packet.write_text('opaque_id,answer\\nfixture,42\\n',encoding='utf-8'); packet_sha=hashlib.sha256(packet.read_bytes()).hexdigest()\n"
        "from ppmoe.distill.execution import artifact_baseline_bytes\n"
        "record={'execution_id':root.name,'stage':'calibrate','stage_signature':signature['stage_signature'],'signature':signature,'status':'awaiting_review','reason':'CPU fixture complete','resource_ledger':{'cuda_seconds':manifest.get('inherited_costs',{}).get('cuda_seconds',1168),'download_bytes':manifest.get('inherited_costs',{}).get('download_bytes',8066699435),'artifact_bytes':packet.stat().st_size},'resource_caps':{'cuda_seconds':86400,'wall_seconds':172800,'download_bytes':21474836480,'artifact_bytes':21474836480},'inherited_artifact_bytes':artifact_baseline_bytes(manifest),'review_packet':'calibration/review/fixture.csv','review_packet_sha256':packet_sha,'private_review_map':'calibration/review/private-map.json'}\n"
        "record.pop('stage',None)\n"
        "reservation_path=root/'stages'/'calibrate-reservations.jsonl'; reservation_path.parent.mkdir(parents=True,exist_ok=True); reservation_path.touch(exist_ok=True)\n"
        "shared_persist_calibration_terminal(root,reservation_path,record,started_at=started_at,artifact_cap=record['resource_caps']['artifact_bytes'])\n"
        "print('finished', flush=True)\n",
        encoding="utf-8",
    )
    fixture = {
        "execution_root": str(root), "repo_root": str(REPO_ROOT),
        "config_path": str(config), "source_root": str(source_root),
        "schema_version": 3 if repair else (2 if recovery else 1),
        "manifest_sha256": study.TASK1_MANIFEST_SHA256 if not recovery else "",
        "prepare_sha256": study.TASK1_PREPARE_SHA256 if not recovery else "",
        "stop_sha256": execution.TASK1_STOP_SHA256 if not recovery else "",
        "runner": str(runner),
        "duration_seconds": 2.5,
    }
    (workspace / "fixture.json").write_bytes(_canonical(fixture) + b"\n")
    return fixture


def _load_fixture(workspace: Path) -> dict[str, str]:
    return json.loads((workspace / "fixture.json").read_text(encoding="utf-8"))


def _patch_fixture_constants(fixture: dict[str, str]) -> None:
    if fixture.get("schema_version") in {2, 3}:
        return
    from ppmoe.distill import execution, study

    study.TASK1_MANIFEST_SHA256 = fixture["manifest_sha256"]
    study.TASK1_PREPARE_SHA256 = fixture["prepare_sha256"]
    execution.TASK1_STOP_SHA256 = fixture["stop_sha256"]


def _argv_builder(fixture: dict[str, str]):
    def build(_repo_root: Path, _manifest: dict[str, object], _attempt_id: str) -> list[str]:
        from ppmoe.distill.execution import execution_stage_signature
        from ppmoe.distill.study import calibration_stage_signature

        execution_signature = execution_stage_signature(
            Path(fixture["execution_root"]), "calibrate", repo_root=Path(fixture["repo_root"]),
        )
        manifest = json.loads((Path(fixture["execution_root"]) / "execution.json").read_text(encoding="utf-8"))
        signature = calibration_stage_signature(
            Path(fixture["repo_root"]), source_run_id=manifest["source_run_id"],
            manifest_sha256=manifest["source_manifest_sha256"],
            prepare_sha256=manifest["source_prepare_sha256"],
            execution_stage_signature=execution_signature,
            execution_policy_sha256=__import__("hashlib").sha256(_canonical(manifest["policy"])).hexdigest(),
        )
        return [sys.executable, fixture["runner"], fixture["execution_root"],
                json.dumps(signature, sort_keys=True, separators=(",", ":")),
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
    prepare_recovery = sub.add_parser("prepare-recovery")
    prepare_recovery.add_argument("--workspace", type=Path, required=True)
    prepare_repair = sub.add_parser("prepare-repair")
    prepare_repair.add_argument("--workspace", type=Path, required=True)
    start = sub.add_parser("start")
    start.add_argument("--workspace", type=Path, required=True)
    start.add_argument("--duration", type=float)
    status = sub.add_parser("status")
    status.add_argument("--workspace", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    if args.action in {"prepare", "prepare-recovery", "prepare-repair"}:
        print(json.dumps(_execution_fixture(
            args.workspace, recovery=args.action == "prepare-recovery", repair=args.action == "prepare-repair",
        ), sort_keys=True))
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
