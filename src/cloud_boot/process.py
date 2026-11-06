# ----------------------------------------------------------------------------------------------------
# process.py
# ----------------------------------------------------------------------------------------------------

"""Bounded, injectable host-process execution primitives.

All external commands go through this module.  It intentionally exposes results rather than
interpreting command-specific exit statuses; higher layers choose the appropriate cloud-boot phase and
exit-code mapping.
"""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, Protocol, cast

from .errors import CloudBootError, ErrorPhase, ExitCode

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "DEFAULT_ENV_ALLOWLIST",
    "DEFAULT_TIMEOUT_SECONDS",
    "Clock",
    "CommandResult",
    "CommandRunner",
    "CommandSpec",
    "ManagedProcess",
    "MemorySink",
    "OutputSink",
    "OutputSinks",
    "ProcessExecutionError",
    "ProcessStartError",
    "ProcessTimeoutError",
    "deadline_after",
    "run",
    "start",
]

# ----------------------------------------------------------------------------------------------------
DEFAULT_TIMEOUT_SECONDS = 300.0  # If timeout seconds is unset, the default timeout is 5 minutes
TERMINATION_GRACE_SECONDS = (
    2.0  # After sending SIGTERM, 2 seconds are given for the program to terminate
)
DEFAULT_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "TMPDIR",
        "USER",
    }
)


# ----------------------------------------------------------------------------------------------------
class ProcessStartError(CloudBootError):
    """Raised when a child process cannot be created."""

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        remediation: str | None = None,
        exit_code: int | ExitCode = ExitCode.HOST_UNAVAILABLE,
    ) -> None:
        super().__init__(
            message,
            code="process-start-failed",
            phase=ErrorPhase.PROCESS,
            remediation=remediation,
            context=context,
            exit_code=exit_code,
        )


# ----------------------------------------------------------------------------------------------------
class ProcessExecutionError(CloudBootError):
    """Raised when a caller explicitly requires a zero return code."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        context: Mapping[str, object] | None = None,
        exit_code: int | ExitCode = ExitCode.INTERNAL,
    ) -> None:
        merged = dict(context or {})
        merged["returncode"] = returncode
        super().__init__(
            message,
            code="process-failed",
            phase=ErrorPhase.PROCESS,
            context=merged,
            exit_code=exit_code,
        )
        self.returncode = returncode


# ----------------------------------------------------------------------------------------------------
class ProcessTimeoutError(CloudBootError):
    """Raised after a process deadline and bounded TERM/KILL cleanup."""

    result: CommandResult | None

    def __init__(
        self,
        message: str = "command exceeded its deadline",
        *,
        context: Mapping[str, object] | None = None,
        result: CommandResult | None = None,
    ) -> None:
        super().__init__(
            message,
            code="process-timeout",
            phase=ErrorPhase.PROCESS,
            remediation="Increase the command deadline or inspect the captured output.",
            context=context,
            exit_code=ExitCode.TIMEOUT,
        )
        self.result = result


# ----------------------------------------------------------------------------------------------------
class OutputSink(Protocol):
    """A destination for bytes read from a child stream."""

    def write(self, data: bytes) -> object:
        """Consume a chunk of output."""


type Sink = OutputSink | Callable[[bytes], object] | None
type PopenFactory = Callable[..., subprocess.Popen[bytes]]
type Clock = Callable[[], float]
type Sleeper = Callable[[float], None]


# ----------------------------------------------------------------------------------------------------
class MemorySink:
    """Thread-safe byte sink useful for callers and tests."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def write(self, data: bytes) -> None:
        """Append bytes under a lock so concurrent writers cannot interleave chunks."""
        with self._lock:
            self._buffer.extend(data)

    def getvalue(self) -> bytes:
        """Return an immutable snapshot unaffected by later writes."""
        with self._lock:
            return bytes(self._buffer)


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OutputSinks:
    """Optional destinations receiving live stdout and stderr bytes."""

    stdout: Sink = None
    stderr: Sink = None


# ----------------------------------------------------------------------------------------------------
def _write_sink(sink: Sink, data: bytes) -> None:
    """Forward one chunk of child output to a caller-provided sink.

    A sink is either an object with a `write(byte)` method, such as an open file or
    :class:`MemorySink`, or a plain callable, such as `list.append`. `None` disables forwarding, and
    any other value is silently ignored. The value returned by the sink is discarded.

    This runs on a stream reader thread. A sink shared by stdout and stderr receives writes from two
    threads at once and must be thread-safe. Exceptions raised by the sink are not caught.

    Args:
        sink: Destination for the chunk, or `None` to drop it.
        data: Bytes just read from the child's stdout or stderr.
    """
    if sink is None:
        return
    writer = getattr(sink, "write", None)

    # Prefer write() so an object that is both writable and callable is treated as a stream.
    if callable(writer):
        writer(data)
    elif callable(sink):
        sink(data)


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Validated command input for :class:`CommandRunner`.

    Environment values are merged over the host's allowlisted values.
    A key outside `env_allowlist` is rejected instead of silently inherited.
    """

    argv: tuple[str, ...]
    cwd: Path | str | None = None
    env: Mapping[str, str] | None = None
    env_allowlist: frozenset[str] = field(default_factory=lambda: DEFAULT_ENV_ALLOWLIST)
    stdin: bytes | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        args = tuple(self.argv)

        if not args:
            raise ValueError("argv must contain at least one argument")
        if any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in args):
            raise ValueError("argv entries must be non-empty strings without NUL bytes")
        object.__setattr__(self, "argv", args)

        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))
        if self.env is not None:
            environment = dict(self.env)

            if any(not isinstance(key, str) or not key or "=" in key for key in environment):
                raise ValueError("environment names must be non-empty strings without '='")
            if any(not isinstance(value, str) or "\x00" in value for value in environment.values()):
                raise ValueError("environment values must be strings without NUL bytes")
            object.__setattr__(self, "env", environment)
        allowlist = frozenset(self.env_allowlist)

        if any(not isinstance(key, str) or not key for key in allowlist):
            raise ValueError("environment allowlist names must be non-empty strings")
        if self.env is not None:
            denied = sorted(set(self.env) - allowlist)

            if denied:
                raise ValueError(f"environment key is not allowlisted: {denied[0]}")
        object.__setattr__(self, "env_allowlist", allowlist)

        if self.stdin is not None and not isinstance(self.stdin, bytes):
            raise ValueError("stdin must be bytes when provided")


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CommandResult:
    """Observed result of a child process."""

    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    duration_seconds: float
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        """Whether the child exited normally with status zero."""
        return self.returncode == 0 and not self.timed_out

    def check_returncode(self, *, exit_code: int | ExitCode = ExitCode.INTERNAL) -> None:
        """Raise a typed error if the command did not succeed."""
        if not self.succeeded:
            raise ProcessExecutionError(
                "command exited unsuccessfully",
                returncode=self.returncode,
                context={"argv_length": len(self.argv), "timed_out": self.timed_out},
                exit_code=exit_code,
            )


# ----------------------------------------------------------------------------------------------------
def deadline_after(seconds: float, *, clock: Clock = time.monotonic) -> float:
    """Return an absolute monotonic deadline after ``seconds``."""
    if seconds < 0 or not (seconds < float("inf")):
        raise ValueError("deadline duration must be finite and non-negative")
    return clock() + seconds


# ----------------------------------------------------------------------------------------------------
def _resolve_deadline(
    *,
    timeout: float | None,
    deadline: float | None,
    clock: Clock,
    default: float | None,
) -> float:
    if timeout is not None and deadline is not None:
        raise ValueError("provide either timeout or deadline, not both")
    if timeout is not None:
        if timeout < 0 or not (timeout < float("inf")):
            raise ValueError("timeout must be finite and non-negative")
        return clock() + timeout
    if deadline is not None:
        if not (deadline < float("inf")):
            raise ValueError("deadline must be finite")
        return deadline
    if default is None:
        raise ValueError("a bounded timeout or deadline is required")
    return clock() + default


# ----------------------------------------------------------------------------------------------------
class ManagedProcess:
    """A started process with bounded waiting and group termination."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        spec: CommandSpec,
        sinks: OutputSinks,
        *,
        clock: Clock,
        sleeper: Sleeper,
        termination_grace_seconds: float,
    ) -> None:
        self._process = process
        self.spec = spec
        self._clock = clock
        self._sleeper = sleeper
        self._termination_grace_seconds = termination_grace_seconds
        self._started_at = clock()
        self._stdout_sink = sinks.stdout
        self._stderr_sink = sinks.stderr
        self._stdout = MemorySink()
        self._stderr = MemorySink()
        self._threads: list[threading.Thread] = []
        self._stdin_thread: threading.Thread | None = None
        self._start_reader(process.stdout, self._stdout, self._stdout_sink)  # thread 1, read stdout
        self._start_reader(process.stderr, self._stderr, self._stderr_sink)  # thread 2, read stderr
        self._start_stdin_writer()  # thread 3, write stdin

    @property
    def pid(self) -> int:
        """Return the child PID."""
        return self._process.pid

    def poll(self) -> int | None:
        """Return the current exit status without waiting."""
        return self._process.poll()

    def _start_reader(
        self,
        stream: IO[bytes] | None,
        capture: MemorySink,
        sink: Sink,
    ) -> None:
        if stream is None:
            return

        def read_stream() -> None:
            try:
                read1 = getattr(stream, "read1", None)
                reader = cast(Callable[[int], bytes], read1) if callable(read1) else stream.read
                while True:
                    chunk = reader(64 * 1024)
                    if not chunk:
                        return
                    capture.write(chunk)
                    _write_sink(sink, chunk)
            finally:
                stream.close()

        thread = threading.Thread(target=read_stream, name="cloud-boot-output", daemon=True)
        thread.start()
        self._threads.append(thread)

    def _start_stdin_writer(self) -> None:
        if self.spec.stdin is None:
            return
        stream = self._process.stdin
        if stream is None:
            return

        def write_stdin() -> None:
            try:
                stream.write(self.spec.stdin or b"")
            except (BrokenPipeError, OSError, ValueError):
                pass
            finally:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()

        thread = threading.Thread(target=write_stdin, name="cloud-boot-stdin", daemon=True)
        thread.start()
        self._stdin_thread = thread

    def _signal_group(self, sig: signal.Signals) -> None:
        try:
            killpg = getattr(os, "killpg", None)
            if callable(killpg) and os.name == "posix":
                killpg(self._process.pid, sig)
            elif sig == signal.SIGTERM:
                self._process.terminate()
            else:
                self._process.kill()
        except (AttributeError, OSError, ProcessLookupError):
            try:
                if sig == signal.SIGTERM:
                    self._process.terminate()
                else:
                    self._process.kill()
            except OSError:
                pass

    def terminate(self) -> None:
        """Request graceful process-group termination."""
        self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        """Force process-group termination."""
        self._signal_group(cast(signal.Signals, getattr(signal, "SIGKILL", signal.SIGTERM)))

    def _all_threads(self) -> list[threading.Thread]:
        threads = list(self._threads)
        if self._stdin_thread is not None:
            threads.append(self._stdin_thread)
        return threads

    def _threads_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._all_threads())

    def _join_threads(self, deadline: float | None) -> None:
        for thread in self._all_threads():
            if deadline is None:
                thread.join(timeout=1.0)
                continue
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            thread.join(timeout=remaining)

    def _fallback_returncode(self) -> int:
        returncode = self.poll()
        if returncode is not None:
            return returncode
        return -int(getattr(signal, "SIGKILL", signal.SIGTERM))

    def _finish(
        self,
        returncode: int,
        *,
        timed_out: bool = False,
        join_deadline: float | None = None,
    ) -> CommandResult:
        self._join_threads(join_deadline)
        if self._threads_alive():
            self.kill()
            self._join_threads(join_deadline)
        return CommandResult(
            argv=self.spec.argv,
            returncode=returncode,
            stdout=self._stdout.getvalue(),
            stderr=self._stderr.getvalue(),
            duration_seconds=max(0.0, self._clock() - self._started_at),
            timed_out=timed_out,
        )

    def _wait_until(self, deadline: float) -> int | None:
        while True:
            returncode = self.poll()
            if returncode is not None:
                return returncode
            remaining = deadline - self._clock()
            if remaining <= 0:
                return None
            self._sleeper(min(0.02, remaining))

    def stop(self, *, timeout: float | None = None, deadline: float | None = None) -> CommandResult:
        """Stop gracefully, then forcefully, within a hard deadline."""
        absolute = _resolve_deadline(
            timeout=timeout,
            deadline=deadline,
            clock=self._clock,
            default=self._termination_grace_seconds + 1.0,
        )
        remaining = max(0.0, absolute - self._clock())
        termination_window = min(self._termination_grace_seconds, remaining / 2)
        kill_deadline = absolute - termination_window
        self.terminate()
        returncode = self._wait_until(kill_deadline)
        if returncode is None:
            self.kill()
            returncode = self._wait_until(absolute)
        if returncode is None:
            result = self._finish(
                self._fallback_returncode(),
                timed_out=True,
                join_deadline=self._clock(),
            )
            raise ProcessTimeoutError(
                "process did not terminate after SIGTERM and SIGKILL",
                context={"argv_length": len(self.spec.argv), "pid": self.pid},
                result=result,
            )
        return self._finish(returncode, join_deadline=absolute)

    def wait(self, *, timeout: float | None = None, deadline: float | None = None) -> CommandResult:
        """Wait for completion; an explicit deadline is a hard cleanup boundary."""
        absolute = _resolve_deadline(
            timeout=timeout,
            deadline=deadline,
            clock=self._clock,
            default=DEFAULT_TIMEOUT_SECONDS,
        )
        returncode = self._wait_until(absolute)
        if returncode is None:
            cleanup_deadline = (
                absolute if deadline is not None else absolute + 2 * self._termination_grace_seconds
            )
            timed_out_result: CommandResult | None
            try:
                timed_out_result = self.stop(deadline=cleanup_deadline)
            except ProcessTimeoutError as exc:
                timed_out_result = exc.result
            if timed_out_result is not None:
                timed_out_result = replace(timed_out_result, timed_out=True)
            raise ProcessTimeoutError(
                context={"argv_length": len(self.spec.argv), "pid": self.pid},
                result=timed_out_result,
            )
        self.terminate()
        cleanup_deadline = self._clock() + self._termination_grace_seconds
        return self._finish(returncode, join_deadline=cleanup_deadline)


# ----------------------------------------------------------------------------------------------------
class CommandRunner:
    """Create and run commands with injectable time and process boundaries."""

    def __init__(
        self,
        *,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
        popen_factory: PopenFactory = subprocess.Popen,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        termination_grace_seconds: float = TERMINATION_GRACE_SECONDS,
    ) -> None:
        if default_timeout_seconds <= 0 or termination_grace_seconds <= 0:
            raise ValueError("process timeouts must be positive")
        self._clock = clock
        self._sleeper = sleeper
        self._popen_factory = popen_factory
        self._default_timeout_seconds = default_timeout_seconds
        self._termination_grace_seconds = termination_grace_seconds

    def _environment(self, spec: CommandSpec) -> dict[str, str]:
        environment = {key: value for key, value in os.environ.items() if key in spec.env_allowlist}
        if spec.env:
            environment.update(spec.env)
        return environment

    def start(self, spec: CommandSpec, sinks: OutputSinks | None = None) -> ManagedProcess:
        """Start one command without waiting for it."""
        try:
            process = self._popen_factory(
                list(spec.argv),
                cwd=str(spec.cwd) if spec.cwd is not None else None,
                env=self._environment(spec),
                stdin=subprocess.PIPE if spec.stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=(os.name == "posix"),
            )
        except (OSError, ValueError) as exc:
            raise ProcessStartError(
                "unable to start host command",
                context={
                    "command": spec.name or spec.argv[0],
                    "environment_keys": sorted(spec.env or {}),
                },
                remediation="Install the required host command and verify its path and permissions.",
            ) from exc
        return ManagedProcess(
            process,
            spec,
            sinks or OutputSinks(),
            clock=self._clock,
            sleeper=self._sleeper,
            termination_grace_seconds=self._termination_grace_seconds,
        )

    def run(
        self,
        spec: CommandSpec,
        *,
        timeout: float | None = None,
        deadline: float | None = None,
        sinks: OutputSinks | None = None,
    ) -> CommandResult:
        """Start and wait for one command within a bounded deadline."""
        process = self.start(spec, sinks)
        if timeout is None and deadline is None:
            timeout = self._default_timeout_seconds
        return process.wait(timeout=timeout, deadline=deadline)


# ----------------------------------------------------------------------------------------------------
_DEFAULT_RUNNER = CommandRunner()


def start(spec: CommandSpec, sinks: OutputSinks | None = None) -> ManagedProcess:
    """Start a command using the default runner."""
    return _DEFAULT_RUNNER.start(spec, sinks)


# ----------------------------------------------------------------------------------------------------
def run(
    spec: CommandSpec,
    *,
    timeout: float | None = None,
    deadline: float | None = None,
    sinks: OutputSinks | None = None,
) -> CommandResult:
    """Run a command using the default runner."""
    return _DEFAULT_RUNNER.run(spec, timeout=timeout, deadline=deadline, sinks=sinks)
