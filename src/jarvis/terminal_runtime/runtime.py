from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO
from uuid import uuid4

from .models import (
    ExecutionState,
    ShellKind,
    TerminalError,
    TerminalErrorCode,
    TerminalOutput,
    TerminalResult,
    TerminalSessionSnapshot,
    TerminalSessionState,
    VisibleTerminalResult,
    utc_now,
)


EventSink = Callable[[str, dict[str, Any]], None]
_SECRET_KEY = re.compile(r"(?:secret|token|password|passwd|credential|api[_-]?key)", re.I)


@dataclass(slots=True)
class _Execution:
    execution_id: str
    command: str
    started: float
    timeout_seconds: float | None
    accepts_input: bool
    state: ExecutionState = ExecutionState.PENDING
    exit_code: int | None = None
    finished: float | None = None
    error_code: TerminalErrorCode | None = None
    done: threading.Event = field(default_factory=threading.Event)
    stdout_done: bool = False
    stderr_done: bool = False
    output: deque[TerminalOutput] = field(default_factory=deque)
    retained_chars: int = 0
    truncated: bool = False
    timer: threading.Timer | None = None
    script_path: Path | None = None
    run_id: str | None = None
    correlation_id: str | None = None


@dataclass(slots=True)
class _Session:
    session_id: str
    shell: ShellKind
    cwd: Path
    process: subprocess.Popen[str]
    created_at: Any
    last_activity: Any
    environment_keys: tuple[str, ...]
    state: TerminalSessionState = TerminalSessionState.STARTING
    active: _Execution | None = None
    executions: dict[str, _Execution] = field(default_factory=dict)
    sequence: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)
    reader_threads: list[threading.Thread] = field(default_factory=list)
    boot_token: str = ""
    boot_ready: threading.Event = field(default_factory=threading.Event)
    job_handle: Any = None
    run_id: str | None = None
    correlation_id: str | None = None


class TerminalRuntime:
    """Owned persistent PowerShell/CMD sessions with structured output."""

    def __init__(
        self,
        *,
        event_sink: EventSink | None = None,
        max_output_chars: int = 64_000,
        base_environment: dict[str, str] | None = None,
    ) -> None:
        self._event_sink = event_sink
        self._max_output_chars = max(max_output_chars, 1024)
        self._base_environment = dict(base_environment or os.environ)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._temp_root = Path(tempfile.mkdtemp(prefix="jarvis-terminal-"))

    def start_session(
        self,
        shell: ShellKind | str = ShellKind.POWERSHELL,
        *,
        cwd: str | Path | None = None,
        environment: dict[str, str | None] | None = None,
        venv_path: str | Path | None = None,
        run_id: str | None = None,
        correlation_id: str | None = None,
    ) -> TerminalSessionSnapshot:
        if self._closed:
            raise TerminalError(TerminalErrorCode.SESSION_CLOSED, "terminal runtime is closed")
        kind = ShellKind(shell)
        working_dir = self._resolve_cwd(cwd)
        executable, args = self._resolve_shell(kind)
        env = dict(self._base_environment)
        changed_keys: set[str] = set()
        for key, value in (environment or {}).items():
            changed_keys.add(key)
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        if venv_path is not None:
            venv = Path(venv_path).expanduser().resolve(strict=False)
            scripts = venv / "Scripts"
            python = scripts / "python.exe"
            if not python.is_file():
                raise TerminalError(TerminalErrorCode.INVALID_CWD, "venv Python executable does not exist")
            env["VIRTUAL_ENV"] = str(venv)
            env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
            changed_keys.update(("VIRTUAL_ENV", "PATH"))
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            process = subprocess.Popen(
                [executable, *args],
                cwd=str(working_dir),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=flags,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TerminalError(TerminalErrorCode.SPAWN_FAILED, "could not start requested shell") from exc
        session_id = uuid4().hex
        now = utc_now()
        session = _Session(
            session_id=session_id,
            shell=kind,
            cwd=working_dir,
            process=process,
            created_at=now,
            last_activity=now,
            environment_keys=tuple(sorted(changed_keys)),
            boot_token=uuid4().hex,
            job_handle=self._create_kill_on_close_job(process),
            run_id=run_id,
            correlation_id=correlation_id,
        )
        with self._lock:
            self._sessions[session_id] = session
        assert process.stdout is not None and process.stderr is not None
        for stream_name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            thread = threading.Thread(
                target=self._read_stream,
                args=(session, stream_name, pipe),
                name=f"jarvis-terminal-{session_id[:8]}-{stream_name}",
                daemon=True,
            )
            session.reader_threads.append(thread)
            thread.start()
        assert process.stdin is not None
        if kind is ShellKind.POWERSHELL:
            process.stdin.write(
                "[Console]::InputEncoding=[Text.UTF8Encoding]::new($false); "
                "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
                f"[Console]::Out.WriteLine('__JARVIS_BOOT_{session.boot_token}__')\n"
            )
        else:
            process.stdin.write(
                f"@echo off & prompt __JARVIS_PROMPT__ & chcp 65001>nul & echo __JARVIS_BOOT_{session.boot_token}__\n"
            )
        process.stdin.flush()
        if not session.boot_ready.wait(5.0):
            self._terminate_owned_tree(process, session.job_handle)
            session.job_handle = None
            session.state = TerminalSessionState.FAILED
            raise TerminalError(TerminalErrorCode.SPAWN_FAILED, "shell did not become ready")
        session.state = TerminalSessionState.READY
        self._publish("terminal.session_started", self._session_event(session))
        return self._snapshot(session)

    def run(
        self,
        session_id: str,
        command: str,
        *,
        timeout_seconds: float | None = 30.0,
        wait: bool = True,
        accepts_input: bool = False,
        run_id: str | None = None,
        correlation_id: str | None = None,
    ) -> TerminalResult:
        session = self._get_session(session_id)
        if not command.strip() or "\x00" in command or "\r" in command or "\n" in command:
            raise TerminalError(TerminalErrorCode.PROCESS_FAILED, "command must be one non-empty line")
        with session.lock:
            if session.state in {TerminalSessionState.CLOSED, TerminalSessionState.FAILED, TerminalSessionState.STOPPING}:
                raise TerminalError(TerminalErrorCode.SESSION_CLOSED, "terminal session is closed")
            if session.active is not None and not session.active.done.is_set():
                raise TerminalError(TerminalErrorCode.SESSION_BUSY, "terminal session already has a running command")
            execution = _Execution(
                execution_id=uuid4().hex,
                command=command,
                started=time.perf_counter(),
                timeout_seconds=timeout_seconds,
                accepts_input=accepts_input,
                state=ExecutionState.RUNNING,
                run_id=run_id if run_id is not None else session.run_id,
                correlation_id=correlation_id if correlation_id is not None else session.correlation_id,
            )
            session.active = execution
            session.executions[execution.execution_id] = execution
            session.state = TerminalSessionState.RUNNING
            session.last_activity = utc_now()
            encoded_command = self._encode_command(session.shell, command, execution.execution_id)
            if session.shell is ShellKind.CMD:
                execution.script_path = self._temp_root / f"{execution.execution_id}.cmd"
                execution.script_path.write_text("@echo off\n" + encoded_command + "\n", encoding="utf-8")
                wire = f'call "{execution.script_path}"'
            else:
                wire = encoded_command
            try:
                assert session.process.stdin is not None
                session.process.stdin.write(wire + "\n")
                session.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._finish_abnormally(session, execution, ExecutionState.FAILED, TerminalErrorCode.PROCESS_FAILED)
                raise TerminalError(TerminalErrorCode.PROCESS_FAILED, "shell process is no longer available") from exc
            if timeout_seconds is not None and timeout_seconds > 0:
                execution.timer = threading.Timer(timeout_seconds, self._timeout_execution, args=(session_id, execution.execution_id))
                execution.timer.daemon = True
                execution.timer.start()
        self._publish("terminal.command_started", self._execution_event(session, execution))
        if wait:
            execution.done.wait()
        return self._result(session, execution)

    def wait(self, session_id: str, execution_id: str, timeout_seconds: float | None = None) -> TerminalResult:
        session, execution = self._get_execution(session_id, execution_id)
        execution.done.wait(timeout_seconds)
        return self._result(session, execution)

    def read_output(self, session_id: str, execution_id: str, *, after_sequence: int = 0) -> TerminalResult:
        session, execution = self._get_execution(session_id, execution_id)
        return self._result(session, execution, after_sequence=after_sequence)

    def status(self, session_id: str, execution_id: str | None = None) -> TerminalSessionSnapshot | TerminalResult:
        session = self._get_session(session_id, allow_closed=True)
        if execution_id is None:
            return self._snapshot(session)
        _, execution = self._get_execution(session_id, execution_id)
        return self._result(session, execution)

    def write(self, session_id: str, execution_id: str, text: str, *, newline: bool = True) -> None:
        session, execution = self._get_execution(session_id, execution_id)
        with session.lock:
            if execution.state is not ExecutionState.RUNNING or not execution.accepts_input:
                raise TerminalError(TerminalErrorCode.INPUT_NOT_ACCEPTED, "execution was not declared input-capable")
            try:
                assert session.process.stdin is not None
                session.process.stdin.write(text + ("\n" if newline else ""))
                session.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise TerminalError(TerminalErrorCode.PROCESS_FAILED, "stdin is no longer available") from exc
            session.last_activity = utc_now()

    def cancel(self, session_id: str, execution_id: str) -> TerminalResult:
        session, execution = self._get_execution(session_id, execution_id)
        with session.lock:
            if execution.done.is_set():
                return self._result(session, execution)
            self._finish_abnormally(session, execution, ExecutionState.CANCELLED, TerminalErrorCode.CANCELLED)
        self._terminate_owned_tree(session.process, session.job_handle)
        session.job_handle = None
        self._publish("terminal.command_cancelled", self._execution_event(session, execution))
        return self._result(session, execution)

    def close_session(self, session_id: str) -> TerminalSessionSnapshot:
        session = self._get_session(session_id, allow_closed=True)
        with session.lock:
            if session.state is TerminalSessionState.CLOSED:
                return self._snapshot(session)
            session.state = TerminalSessionState.STOPPING
            if session.active is not None and not session.active.done.is_set():
                self._finish_abnormally(session, session.active, ExecutionState.CANCELLED, TerminalErrorCode.CANCELLED)
        self._terminate_owned_tree(session.process, session.job_handle)
        session.job_handle = None
        for thread in session.reader_threads:
            thread.join(timeout=1.0)
        session.state = TerminalSessionState.CLOSED
        session.last_activity = utc_now()
        self._publish("terminal.session_closed", self._session_event(session))
        return self._snapshot(session)

    def open_visible_terminal(
        self,
        shell: ShellKind | str = ShellKind.POWERSHELL,
        *,
        cwd: str | Path | None = None,
        command: str | None = None,
        run_id: str | None = None,
        correlation_id: str | None = None,
        window_name: str | None = None,
    ) -> VisibleTerminalResult:
        kind = ShellKind(shell)
        working_dir = self._resolve_cwd(cwd)
        wt = self._resolve_windows_terminal()
        if kind is ShellKind.POWERSHELL:
            profile_args = ["powershell.exe", "-NoLogo"]
            if command:
                profile_args.extend(("-NoProfile", "-Command", command))
        else:
            profile_args = ["cmd.exe", "/D"]
            if command:
                profile_args.extend(("/C", command))
        window = window_name or f"jarvis-{uuid4().hex}"
        try:
            process = subprocess.Popen(
                [wt, "-w", window, "new-tab", "-d", str(working_dir), *profile_args],
                cwd=str(working_dir),
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as exc:
            raise TerminalError(TerminalErrorCode.SPAWN_FAILED, "Windows Terminal could not be launched") from exc
        self._publish("terminal.visible_started", {"pid": process.pid, "shell": kind.value, "cwd": str(working_dir), "owned_by_jarvis": True, **self._context_payload(run_id, correlation_id)})
        return VisibleTerminalResult(pid=process.pid, shell=kind, cwd=str(working_dir))

    def shutdown(self) -> None:
        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            try:
                self.close_session(session_id)
            except TerminalError:
                pass
        self._closed = True
        shutil.rmtree(self._temp_root, ignore_errors=True)

    close = shutdown

    def _read_stream(self, session: _Session, stream_name: str, pipe: TextIO) -> None:
        try:
            for line in iter(pipe.readline, ""):
                self._handle_line(session, stream_name, line)
        finally:
            try:
                pipe.close()
            except OSError:
                pass
            with session.lock:
                active = session.active
                if active is not None and not active.done.is_set() and session.process.poll() is not None:
                    self._finish_abnormally(session, active, ExecutionState.FAILED, TerminalErrorCode.PROCESS_FAILED)
                if session.state not in {TerminalSessionState.CLOSED, TerminalSessionState.STOPPING} and session.process.poll() is not None:
                    session.state = TerminalSessionState.FAILED

    def _handle_line(self, session: _Session, stream: str, line: str) -> None:
        with session.lock:
            if stream == "stdout" and f"__JARVIS_BOOT_{session.boot_token}__" in line:
                session.boot_ready.set()
                return
            if session.shell is ShellKind.CMD and stream == "stdout":
                line = line.replace("__JARVIS_PROMPT__", "")
                if not line.strip():
                    return
            execution = session.active
            if execution is None or execution.done.is_set():
                return
            token = execution.execution_id
            stripped = line.rstrip("\r\n ")
            done_marker = f"__JARVIS_DONE_{token}__"
            cwd_marker = f"__JARVIS_CWD_{token}__"
            if stream == "stdout" and done_marker in stripped:
                try:
                    execution.exit_code = int(stripped.split(done_marker, 1)[-1])
                except ValueError:
                    execution.exit_code = 1
                execution.stdout_done = True
                self._maybe_complete(session, execution)
                return
            if stream == "stdout" and cwd_marker in stripped:
                raw = stripped.split(cwd_marker, 1)[-1]
                try:
                    cwd = base64.b64decode(raw).decode("utf-8") if session.shell is ShellKind.POWERSHELL else raw
                    resolved = Path(cwd).resolve(strict=False)
                    if resolved.is_dir():
                        session.cwd = resolved
                except (ValueError, OSError):
                    pass
                return
            if stream == "stderr" and stripped == f"__JARVIS_ERR_DONE_{token}__":
                execution.stderr_done = True
                self._maybe_complete(session, execution)
                return
            session.sequence += 1
            chunk = TerminalOutput(sequence=session.sequence, stream=stream, text=line)
            execution.output.append(chunk)
            execution.retained_chars += len(line)
            while execution.retained_chars > self._max_output_chars and execution.output:
                removed = execution.output.popleft()
                execution.retained_chars -= len(removed.text)
                execution.truncated = True
            session.last_activity = utc_now()
            self._publish("terminal.output", {**self._execution_event(session, execution), "stream": stream, "characters": len(line), "sequence": chunk.sequence})

    def _maybe_complete(self, session: _Session, execution: _Execution) -> None:
        if not (execution.stdout_done and execution.stderr_done) or execution.done.is_set():
            return
        execution.state = ExecutionState.SUCCEEDED if execution.exit_code == 0 else ExecutionState.FAILED
        if execution.state is ExecutionState.FAILED:
            execution.error_code = self._classify_failure(execution)
        execution.finished = time.perf_counter()
        if execution.timer is not None:
            execution.timer.cancel()
        execution.done.set()
        self._remove_script(execution)
        session.active = None
        session.state = TerminalSessionState.READY
        session.last_activity = utc_now()
        suffix = "completed" if execution.state is ExecutionState.SUCCEEDED else "failed"
        self._publish(f"terminal.command_{suffix}", self._execution_event(session, execution))

    def _finish_abnormally(self, session: _Session, execution: _Execution, state: ExecutionState, code: TerminalErrorCode) -> None:
        if execution.done.is_set():
            return
        execution.state = state
        execution.error_code = code
        execution.finished = time.perf_counter()
        if execution.timer is not None:
            execution.timer.cancel()
        execution.done.set()
        self._remove_script(execution)
        session.active = None
        session.state = TerminalSessionState.FAILED
        session.last_activity = utc_now()

    def _timeout_execution(self, session_id: str, execution_id: str) -> None:
        try:
            session, execution = self._get_execution(session_id, execution_id)
        except TerminalError:
            return
        with session.lock:
            if execution.done.is_set():
                return
            self._finish_abnormally(session, execution, ExecutionState.TIMED_OUT, TerminalErrorCode.TIMED_OUT)
        self._terminate_owned_tree(session.process, session.job_handle)
        session.job_handle = None
        self._publish("terminal.command_timed_out", self._execution_event(session, execution))

    def _result(self, session: _Session, execution: _Execution, *, after_sequence: int = 0) -> TerminalResult:
        with session.lock:
            output = [item for item in execution.output if item.sequence > after_sequence]
            stdout = "".join(item.text for item in output if item.stream == "stdout")
            stderr = "".join(item.text for item in output if item.stream == "stderr")
            end = execution.finished or time.perf_counter()
            return TerminalResult(
                session_id=session.session_id,
                execution_id=execution.execution_id,
                shell=session.shell,
                cwd=str(session.cwd),
                status=execution.state,
                exit_code=execution.exit_code,
                stdout=stdout,
                stderr=stderr,
                output=output,
                duration_ms=max((end - execution.started) * 1000.0, 0.0),
                timeout_seconds=execution.timeout_seconds,
                truncated=execution.truncated,
                error_code=execution.error_code,
            )

    def _snapshot(self, session: _Session) -> TerminalSessionSnapshot:
        return TerminalSessionSnapshot(
            session_id=session.session_id,
            shell=session.shell,
            cwd=str(session.cwd),
            state=session.state,
            owned_by_jarvis=True,
            pid=session.process.pid,
            created_at=session.created_at,
            last_activity=session.last_activity,
            active_execution_id=session.active.execution_id if session.active else None,
            environment={key: "<redacted>" if _SECRET_KEY.search(key) else "<set>" for key in session.environment_keys},
        )

    def _get_session(self, session_id: str, *, allow_closed: bool = False) -> _Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise TerminalError(TerminalErrorCode.SESSION_NOT_FOUND, "terminal session does not exist")
        if not allow_closed and session.state is TerminalSessionState.CLOSED:
            raise TerminalError(TerminalErrorCode.SESSION_CLOSED, "terminal session is closed")
        return session

    def _get_execution(self, session_id: str, execution_id: str) -> tuple[_Session, _Execution]:
        session = self._get_session(session_id, allow_closed=True)
        execution = session.executions.get(execution_id)
        if execution is None:
            raise TerminalError(TerminalErrorCode.PROCESS_FAILED, "terminal execution does not exist")
        return session, execution

    @staticmethod
    def _resolve_cwd(cwd: str | Path | None) -> Path:
        try:
            path = Path(cwd or Path.cwd()).expanduser().resolve(strict=False)
        except (OSError, ValueError) as exc:
            raise TerminalError(TerminalErrorCode.INVALID_CWD, "invalid working directory") from exc
        if not path.is_dir():
            raise TerminalError(TerminalErrorCode.INVALID_CWD, "working directory does not exist")
        return path

    @staticmethod
    def _resolve_shell(shell: ShellKind) -> tuple[str, list[str]]:
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        if shell is ShellKind.POWERSHELL:
            candidates = [system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"]
            args = ["-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", "-"]
        else:
            candidates = [system_root / "System32" / "cmd.exe"]
            args = ["/D", "/Q", "/V:ON"]
        executable = next((item for item in candidates if item.is_file()), None)
        if executable is None:
            raise TerminalError(TerminalErrorCode.SHELL_UNAVAILABLE, f"{shell.value} is unavailable")
        return str(executable), args

    @staticmethod
    def _resolve_windows_terminal() -> str:
        local = os.environ.get("LOCALAPPDATA")
        candidates = [Path(local) / "Microsoft" / "WindowsApps" / "wt.exe"] if local else []
        resolved = shutil.which("wt.exe")
        if resolved:
            candidates.append(Path(resolved))
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise TerminalError(TerminalErrorCode.WINDOWS_TERMINAL_UNAVAILABLE, "Windows Terminal is not installed")

    @staticmethod
    def _encode_command(shell: ShellKind, command: str, token: str) -> str:
        if shell is ShellKind.POWERSHELL:
            encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
            return (
                f"$global:LASTEXITCODE=0; $__j=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}')); "
                f"try {{ . ([ScriptBlock]::Create($__j)) | ForEach-Object {{ [Console]::Out.WriteLine(($_ | Out-String -Width 4096).TrimEnd()) }}; $__ok=$?; if ($__ok) {{ $__ec=$global:LASTEXITCODE }} else {{ $__ec=if ($global:LASTEXITCODE -ne 0) {{$global:LASTEXITCODE}} else {{1}} }} }} "
                f"catch {{ [Console]::Error.WriteLine($_.Exception.Message); $__ec=1 }}; "
                f"[Console]::Out.WriteLine('__JARVIS_CWD_{token}__'+[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((Get-Location).Path))); "
                f"[Console]::Out.WriteLine('__JARVIS_DONE_{token}__'+$__ec); "
                f"[Console]::Error.WriteLine('__JARVIS_ERR_DONE_{token}__')"
            )
        return (
            f"{command} & set \"__JARVIS_EC=!errorlevel!\" & "
            f"echo __JARVIS_CWD_{token}__!CD! & "
            f"echo __JARVIS_DONE_{token}__!__JARVIS_EC! & "
            f"echo __JARVIS_ERR_DONE_{token}__ 1>&2"
        )

    @staticmethod
    def _classify_failure(execution: _Execution) -> TerminalErrorCode:
        text = "".join(item.text for item in execution.output).lower()
        if any(marker in text for marker in ("not recognized", "commandnotfoundexception", "is not recognized")):
            return TerminalErrorCode.COMMAND_NOT_FOUND
        if "access is denied" in text or "permission denied" in text:
            return TerminalErrorCode.PERMISSION_DENIED
        return TerminalErrorCode.PROCESS_FAILED

    @staticmethod
    def _remove_script(execution: _Execution) -> None:
        if execution.script_path is None:
            return
        try:
            execution.script_path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _create_kill_on_close_job(process: subprocess.Popen[str]) -> Any:
        handle = None
        try:
            import win32job

            handle = win32job.CreateJobObject(None, "")
            info = win32job.QueryInformationJobObject(
                handle, win32job.JobObjectExtendedLimitInformation
            )
            info["BasicLimitInformation"]["LimitFlags"] |= (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            win32job.SetInformationJobObject(
                handle, win32job.JobObjectExtendedLimitInformation, info
            )
            win32job.AssignProcessToJobObject(handle, process._handle)  # noqa: SLF001
            return handle
        except Exception:
            if handle is not None:
                try:
                    import win32api

                    win32api.CloseHandle(handle)
                except Exception:
                    pass
            return None

    @staticmethod
    def _terminate_owned_tree(process: subprocess.Popen[str], job_handle: Any = None) -> None:
        if job_handle is not None:
            try:
                import win32api

                try:
                    win32api.CloseHandle(job_handle)
                except Exception:  # noqa: BLE001 - handle already closed by a prior timeout/kill path
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
                return
            except (ImportError, OSError):
                pass
        if process.poll() is not None:
            return
        try:
            import psutil

            root = psutil.Process(process.pid)
            children = root.children(recursive=True)
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.Error:
                    pass
            try:
                root.terminate()
            except psutil.Error:
                pass
            _, alive = psutil.wait_procs([*children, root], timeout=1.5)
            for item in alive:
                try:
                    item.kill()
                except psutil.Error:
                    pass
            psutil.wait_procs(alive, timeout=1.5)
        except Exception:
            try:
                process.terminate()
                process.wait(timeout=1.5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=1.5)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def _publish(self, event: str, payload: dict[str, Any]) -> None:
        if self._event_sink is not None:
            self._event_sink(event, payload)

    @staticmethod
    def _session_event(session: _Session) -> dict[str, Any]:
        return {"session_id": session.session_id, "shell": session.shell.value, "cwd": str(session.cwd), "state": session.state.value, "pid": session.process.pid, "owned_by_jarvis": True, **TerminalRuntime._context_payload(session.run_id, session.correlation_id)}

    @staticmethod
    def _execution_event(session: _Session, execution: _Execution) -> dict[str, Any]:
        return {"session_id": session.session_id, "execution_id": execution.execution_id, "shell": session.shell.value, "cwd": str(session.cwd), "status": execution.state.value, "exit_code": execution.exit_code, "error_code": execution.error_code.value if execution.error_code else None, "owned_by_jarvis": True, **TerminalRuntime._context_payload(execution.run_id, execution.correlation_id)}

    @staticmethod
    def _context_payload(run_id: str | None, correlation_id: str | None) -> dict[str, str]:
        return {
            **({"run_id": run_id} if run_id else {}),
            **({"correlation_id": correlation_id} if correlation_id else {}),
        }
