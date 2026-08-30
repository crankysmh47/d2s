import json
from pathlib import Path
import subprocess
import sys


def test_verify_phase0_cli_writes_valid_json(tmp_path: Path):
    output_path = tmp_path / "phase0.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_phase0.py",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--seed",
            "43",
            "--hidden-size",
            "8",
            "--intermediate-size",
            "16",
            "--num-experts",
            "4",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["seed"] == 43
    assert report["environment"]["device"] == "cpu"
