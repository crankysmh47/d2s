"""Owned Windows Job Object for one stage process tree."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

if os.name == "nt":
    import _winapi
    import msvcrt
else:  # Keep status and registration modules importable on unsupported platforms.
    _winapi = None
    msvcrt = None


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaults", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


class _NativeProcess:
    def __init__(self, process_handle: int, thread_handle: int, pid: int) -> None:
        self.handle = process_handle
        self.thread_handle = thread_handle
        self.pid = pid


def direct_venv_launch(argv: Sequence[str]) -> tuple[list[str], str | None, dict[str, str] | None]:
    """Bypass Windows' venv redirector while retaining venv identity and paths."""
    effective = list(argv)
    if os.name != "nt" or not effective or sys.prefix == sys.base_prefix:
        return effective, None, None
    selected = os.path.normcase(os.path.abspath(effective[0]))
    venv_python = os.path.normcase(os.path.abspath(sys.executable))
    if selected != venv_python:
        return effective, None, None
    base_executable = str(Path(sys._base_executable).resolve())
    environment = os.environ.copy()
    environment["__PYVENV_LAUNCHER__"] = sys.executable
    effective[0] = base_executable
    return effective, base_executable, environment


class WindowsJob:
    """A kill-on-close Job Object holding a suspended stage before it runs."""

    _CREATE_SUSPENDED = 0x00000004
    _KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("distillation background execution requires Windows Job Objects")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_api()
        self._job_handle = self._kernel32.CreateJobObjectW(None, None)
        if not self._job_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = self._KILL_ON_JOB_CLOSE
        if not self._kernel32.SetInformationJobObject(
            self._job_handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)
        self._process: _NativeProcess | None = None
        self.process_id: int | None = None
        self.created_utc: str | None = None
        self._resumed = False

    def _configure_api(self) -> None:
        from ctypes import wintypes

        kernel32 = self._kernel32
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.ResumeThread.restype = wintypes.DWORD
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

    @property
    def active_processes(self) -> int:
        if not getattr(self, "_job_handle", None):
            return 0
        info = _BasicAccountingInformation()
        if not self._kernel32.QueryInformationJobObject(
            self._job_handle,
            self._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.ActiveProcesses)

    def _assign_process(self, process_handle: Any) -> None:
        if not self._kernel32.AssignProcessToJobObject(self._job_handle, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> dict[str, Any]:
        if self._process is not None:
            raise RuntimeError("WindowsJob can contain only one stage root")
        if not argv or any(not isinstance(item, str) for item in argv):
            raise ValueError("stage command must be a nonempty argument list")
        from .execution import process_creation_utc

        effective_argv, executable_override, environment = direct_venv_launch(argv)

        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        process: _NativeProcess | None = None
        inherited_handles: list[int] = []
        file_streams: list[Any] = []
        try:
            stdin = open(os.devnull, "rb", buffering=0)
            stdout = Path(stdout_path).open("ab", buffering=0)
            stderr = Path(stderr_path).open("ab", buffering=0)
            file_streams.extend([stdin, stdout, stderr])
            handles = [msvcrt.get_osfhandle(stream.fileno()) for stream in file_streams]
            for handle in handles:
                os.set_handle_inheritable(handle, True)
                inherited_handles.append(handle)
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= _winapi.STARTF_USESTDHANDLES
            startup.hStdInput, startup.hStdOutput, startup.hStdError = handles
            startup.lpAttributeList = {"handle_list": handles}
            command_line = subprocess.list2cmdline(effective_argv)
            process_handle, thread_handle, pid, _tid = _winapi.CreateProcess(
                executable_override or str(effective_argv[0]), command_line, None, None, True,
                self._CREATE_SUSPENDED, environment, str(cwd), startup,
            )
            process = _NativeProcess(process_handle, thread_handle, pid)
            self._assign_process(process.handle)
            created = process_creation_utc(process.pid)
            if created is None:
                raise OSError("could not prove live suspended stage creation identity")
            self._process = process
            self.process_id = process.pid
            self.created_utc = created
            return {"pid": process.pid, "created_utc": created}
        except BaseException:
            if process is not None:
                try:
                    self._kernel32.TerminateProcess(process.handle, 1)
                    _winapi.WaitForSingleObject(process.handle, 5000)
                except OSError:
                    pass
                self._kernel32.CloseHandle(process.thread_handle)
                self._kernel32.CloseHandle(process.handle)
            raise
        finally:
            for handle in inherited_handles:
                try:
                    os.set_handle_inheritable(handle, False)
                except OSError:
                    pass
            for stream in file_streams:
                stream.close()

    def resume(self) -> None:
        process = self._require_process()
        if self._resumed:
            raise RuntimeError("stage process has already been resumed")
        thread = process.thread_handle
        if not thread:
            raise RuntimeError("suspended stage thread handle is unavailable")
        result = self._kernel32.ResumeThread(thread)
        if result == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        self._kernel32.CloseHandle(thread)
        process.thread_handle = None
        self._resumed = True

    def poll(self) -> int | None:
        if self._process is None:
            return None
        code = ctypes.c_uint32()
        if not self._kernel32.GetExitCodeProcess(self._process.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return None if code.value == _winapi.STILL_ACTIVE else int(code.value)

    def terminate_and_wait(self, timeout_seconds: float = 10.0) -> bool:
        if not getattr(self, "_job_handle", None):
            return True
        if self.active_processes and not self._kernel32.TerminateJobObject(self._job_handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        if self._process is not None:
            remaining = max(0.0, deadline - time.monotonic())
            milliseconds = min(0xFFFFFFFE, int(remaining * 1000))
            if _winapi.WaitForSingleObject(self._process.handle, milliseconds) != _winapi.WAIT_OBJECT_0:
                return False
        while time.monotonic() < deadline:
            if self.active_processes == 0:
                return True
            time.sleep(0.02)
        return self.active_processes == 0

    def close(self) -> None:
        handle = getattr(self, "_job_handle", None)
        if not handle:
            return
        # Job Object kill-on-close owns crash cleanup as well as normal teardown.
        self._kernel32.CloseHandle(handle)
        self._job_handle = None
        if self._process is not None:
            _winapi.WaitForSingleObject(self._process.handle, 5000)
            if self._process.thread_handle:
                self._kernel32.CloseHandle(self._process.thread_handle)
                self._process.thread_handle = None
            self._kernel32.CloseHandle(self._process.handle)
            self._process = None

    def _require_process(self) -> _NativeProcess:
        if self._process is None:
            raise RuntimeError("stage process has not been started")
        return self._process

    def __enter__(self) -> WindowsJob:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
