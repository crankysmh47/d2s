"""The CLI must parse lazily and reject unsupported scientific controls."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPT = Path("scripts/run_recurrence_v0.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], text=True, capture_output=True, check=False)


def test_help_is_lazy_and_does_not_create_artifacts(tmp_path: Path) -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "--stage" in result.stdout
    assert not (tmp_path / "artifacts").exists()


def test_cli_rejects_all_unknown_and_scientific_override_before_work() -> None:
    for arguments in (("--stage", "all"), ("--stage", "bad"), ("--stage", "smoke"), ("--stage", "source", "--lora-rank", "16")):
        result = _run(*arguments)
        assert result.returncode != 0
        assert "RUN_ID=" not in result.stdout


def test_cli_rejects_malformed_post_data_run_id() -> None:
    result = _run("--stage", "smoke", "--run-id", "not-a-run")
    assert result.returncode != 0
    assert "12-hex" in result.stderr
