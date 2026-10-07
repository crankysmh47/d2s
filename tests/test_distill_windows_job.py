from __future__ import annotations

import os
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ppmoe.distill.windows_job import WindowsJob


pytestmark = pytest.mark.skipif(os.name != "nt", reason="requires native Windows Job Objects")


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def test_suspended_child_is_assigned_before_it_runs_in_path_with_spaces(tmp_path: Path):
    work = tmp_path / "directory with spaces"
    work.mkdir()
    script = work / "child script.py"
    output = work / "child output.txt"
    pid_output = work / "child pid.txt"
    script.write_text(
        "from pathlib import Path\nimport importlib.metadata,json,os,sys,time\n"
        "identity={'pid':os.getpid(),'executable':sys.executable,'prefix':sys.prefix,'packages':{name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate','bitsandbytes','peft')}}\n"
        f"Path({str(pid_output)!r}).write_text(json.dumps(identity))\n"
        "time.sleep(.1)\n"
        f"Path({str(output)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    job = WindowsJob()
    try:
        child = job.start(
            [sys.executable, str(script)], cwd=work,
            stdout_path=work / "stdout.log", stderr_path=work / "stderr.log",
        )
        assert child["pid"] > 0
        assert not output.exists()
        job.resume()
        assert _wait_until(output.exists)
        identity = json.loads(pid_output.read_text(encoding="utf-8"))
        assert identity["pid"] == child["pid"]
        assert os.path.normcase(identity["executable"]) == os.path.normcase(sys.executable)
        assert os.path.normcase(identity["prefix"]) == os.path.normcase(sys.prefix)
        assert identity["packages"] == {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "accelerate", "bitsandbytes", "peft")
        }
        assert _wait_until(lambda: job.poll() is not None)
        assert job.poll() == 0
    finally:
        job.close()


def test_job_termination_waits_for_owned_grandchild(tmp_path: Path):
    work = tmp_path / "parent and grandchild"
    work.mkdir()
    child = work / "tree.py"
    marker = work / "grandchild.txt"
    child.write_text(
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', \"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).write_text('escaped')\"])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    job = WindowsJob()
    try:
        job.start([sys.executable, str(child)], cwd=work,
                  stdout_path=work / "stdout.log", stderr_path=work / "stderr.log")
        job.resume()
        assert _wait_until(lambda: job.active_processes >= 2)
        assert job.terminate_and_wait(timeout_seconds=5)
        time.sleep(2.2)
        assert not marker.exists()
        assert job.active_processes == 0
    finally:
        job.close()


def test_failed_job_assignment_cleans_suspended_child(tmp_path: Path):
    work = tmp_path / "failed assignment"
    work.mkdir()
    child = work / "child.py"
    child.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    job = WindowsJob()
    job._assign_process = lambda *_args: (_ for _ in ()).throw(OSError("assignment rejected"))
    try:
        with pytest.raises(OSError, match="assignment rejected"):
            job.start([sys.executable, str(child)], cwd=work,
                      stdout_path=work / "stdout.log", stderr_path=work / "stderr.log")
        assert job.process_id is None
    finally:
        job.close()


def test_supervisor_process_death_closes_job_and_kills_grandchild(tmp_path: Path):
    work = tmp_path / "supervisor crash with spaces"
    work.mkdir()
    ready = work / "ready.txt"
    marker = work / "grandchild escaped.txt"
    supervisor = work / "supervisor.py"
    child = work / "tree.py"
    child.write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', \"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).write_text('escaped')\"])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    supervisor.write_text(
        "import sys,time\n"
        "from pathlib import Path\n"
        "from ppmoe.distill.windows_job import WindowsJob\n"
        "job=WindowsJob(); work=Path(sys.argv[1]); child=work/'tree.py'\n"
        "job.start([sys.executable,str(child)],cwd=work,stdout_path=work/'child.log',stderr_path=work/'child.err')\n"
        "job.resume()\n"
        "deadline=time.monotonic()+5\n"
        "while job.active_processes<2 and time.monotonic()<deadline: time.sleep(.02)\n"
        "Path(sys.argv[2]).write_text(str(job.active_processes))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(supervisor), str(work), str(ready)], cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_until(ready.exists)
        assert int(ready.read_text(encoding="utf-8")) >= 2
        process.kill()
        assert process.wait(timeout=5) is not None
        time.sleep(2.2)
        assert not marker.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
