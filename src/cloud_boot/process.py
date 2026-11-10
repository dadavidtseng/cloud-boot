# ----------------------------------------------------------------------------------------------------
# process.py
# ----------------------------------------------------------------------------------------------------

"""Bounded, injectable host-process execution: every external command goes through here.

Guarantees for every command:

- No shell: argv is passed as a list with `shell=False`, so nothing is ever shell-parsed.
- Bounded time: every wait has a monotonic deadline; on expiry the whole process group gets
  SIGTERM, then SIGKILL after a grace period.
- Own process group: children start in a new session, so stopping one also stops anything it
  spawned and a terminal Ctrl-C reaches cloud-boot first.
- Minimal environment: only allowlisted variables are inherited from the host.
- Bounded memory: captured stdout/stderr stop growing at a limit; live sinks still see every byte.

This layer reports what happened and does not interpret exit statuses. Callers decide what a
status means and which cloud-boot phase and exit code it maps to.
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
from typing import IO, Literal, Protocol, cast

from .errors import CloudBootError, ErrorPhase, ExitCode

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "DEFAULT_CAPTURE_LIMIT_BYTES",
    "DEFAULT_ENV_ALLOWLIST",
    "DEFAULT_TIMEOUT_SECONDS",
    "ByteWriter",
    "Clock",
    "CommandResult",
    "CommandRunner",
    "CommandSpec",
    "ManagedProcess",
    "MemorySink",
    "ProcessStartError",
    "ProcessTimeoutError",
    "SinkFailure",
    "Sleeper",
    "StreamSinks",
]

# ----------------------------------------------------------------------------------------------------
# Fallback when a caller passes neither timeout nor deadline.
# The coordinator always passes its run deadline, so this only bounds direct use of the runner.
DEFAULT_TIMEOUT_SECONDS = 300.0

# How long a process group gets to exit after SIGTERM before SIGKILL.
# Short, because graceful shutdown happens earlier (the guest is first asked over SSH to power off),
# so only children that are already past their deadline or being cleaned up ever receive signals.
TERMINATION_GRACE_SECONDS = 2.0

# In-memory stdout/stderr capture per stream; live sinks still receive every byte.
DEFAULT_CAPTURE_LIMIT_BYTES = 1024 * 1024

# Host variables a child may inherit: a minimal login environment (tool lookup, user identity,
# locale, temporary directory). Everything else is dropped, so credentials such as cloud tokens
# and loader hooks such as LD_PRELOAD never reach QEMU or ssh. Dropping SSH_AUTH_SOCK also keeps
# ssh from offering the user's agent keys; it uses only the run's own key.
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
    """The child process could not be created at all (missing binary, no permission, bad argv).

    Raised by `CommandRunner.start` when `Popen` itself fails, so no process exists and nothing
    needs cleaning up. The exit code is always `HOST_UNAVAILABLE` because the usual cause is a
    host tool that preflight did not catch, such as a binary that exists but cannot run.
    """

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code="process-start-failed",
            phase=ErrorPhase.PROCESS,
            remediation=remediation,
            context=context,
            exit_code=ExitCode.HOST_UNAVAILABLE,
        )


# ----------------------------------------------------------------------------------------------------
class ProcessTimeoutError(CloudBootError):
    """A deadline expired before a command finished; callers treat it as "out of time".

    Raised in three situations:

    - `ManagedProcess.wait` reached its deadline. The process group has already received SIGTERM and,
    if needed, SIGKILL.
    - `ManagedProcess.stop` sent SIGKILL and the process still had not exited by the deadline. It may
    still be running, for example while blocked in uninterruptible I/O.
    - The coordinator refused to start a guest command because the run deadline had already
      passed. No process exists in this case.

    `result` holds whatever output was captured, or `None` when no process was involved. The
    exit code is always `ExitCode.TIMEOUT`.
    """

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
class ByteWriter(Protocol):
    """Interface for one output destination: anything with a `write(bytes)` method.

    `Sink` is what a single stream accepts (this, a plain callable, or `None`), and
    `StreamSinks` pairs one `Sink` for stdout with one for stderr. The return type is `object`
    so both `file.write`, which returns a byte count, and writers that return `None` qualify.
    """

    def write(self, data: bytes) -> object:
        """Consume a chunk of output."""


# What one stream accepts: a `ByteWriter`, a plain callable such as `file.write`, or `None` to keep
# only the in-memory capture.
type Sink = ByteWriter | Callable[[bytes], object] | None


class PopenLike(Protocol):
    """The part of `subprocess.Popen` that `ManagedProcess` uses; test fakes implement only this."""

    @property
    def pid(self) -> int: ...
    @property
    def stdin(self) -> IO[bytes] | None: ...
    @property
    def stdout(self) -> IO[bytes] | None: ...
    @property
    def stderr(self) -> IO[bytes] | None: ...
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


type PopenFactory = Callable[..., PopenLike]
# Injected time sources, so tests can drive deadlines without real waiting.
type Clock = Callable[[], float]
type Sleeper = Callable[[float], None]


# ----------------------------------------------------------------------------------------------------
class MemorySink:
    """Thread-safe in-memory byte buffer, optionally keeping only the newest `limit` bytes.

    `ManagedProcess` uses one per stream for the output that ends up in `CommandResult`; callers
    can also pass one as a live sink. The lock is needed because the stream's reader thread
    writes while another thread may read.

    Without a `limit`, every byte is kept. With a `limit`, only the most recent `limit` bytes are
    kept, because the end of a child's output usually explains its failure; older bytes are counted
    in `dropped_bytes` so truncation is never silent.
    """

    def __init__(self, limit: int | None = None) -> None:
        if limit is not None and (isinstance(limit, bool) or limit < 0):
            raise ValueError("memory sink limit must be a non-negative integer")
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._limit = limit
        self._dropped_bytes = 0

    def write(self, data: bytes) -> None:
        """Append bytes under a lock so concurrent writers cannot interleave chunks."""
        with self._lock:
            self._buffer.extend(data)
            if self._limit is not None and len(self._buffer) > self._limit:
                excess = len(self._buffer) - self._limit
                del self._buffer[:excess]
                self._dropped_bytes += excess

    def getvalue(self) -> bytes:
        """Return an immutable snapshot unaffected by later writes."""
        with self._lock:
            return bytes(self._buffer)

    @property
    def dropped_bytes(self) -> int:
        """Return how many of the oldest bytes were discarded to stay within the limit."""
        with self._lock:
            return self._dropped_bytes


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StreamSinks:
    """Live destinations for one process: a `Sink` for stdout and one for stderr.

    Bundled into one object so it can pass unchanged through `CommandRunner` down to the stream
    reader threads. These receive every byte, in addition to the bounded in-memory capture;
    `None`, the default, means capture only. For example, `vm.py` streams the QEMU serial console
    to the console log, and `ssh.py` streams a guest file straight to disk.
    """

    stdout: Sink = None
    stderr: Sink = None


# ----------------------------------------------------------------------------------------------------
type StreamName = Literal["stdout", "stderr"]


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SinkFailure:
    """A live output sink that raised and stopped receiving output.

    The reader keeps draining the stream after a failure, so the child never blocks on a full pipe
    or receives `SIGPIPE`; only forwarding to the failed sink stops.

    Failures are reported in `CommandResult.sink_failures`, and the caller decides whether they
    matter: the coordinator logs a console-log failure as a warning, while a guest file copy
    treats one as a failed copy.
    """

    stream: StreamName
    error: Exception


# ----------------------------------------------------------------------------------------------------
def _write_sink(sink: Sink, data: bytes) -> None:
    """Forward one chunk of child output to a caller-provided sink.

    A sink is either an object with a `write(data)` method, such as an open file or
    `MemorySink`, or a plain callable, such as `list.append`. `None` disables forwarding, and
    any other value is silently ignored. The value returned by the sink is discarded.

    This runs on a stream reader thread. A sink shared by stdout and stderr receives writes from two
    threads at once and must be thread-safe. Exceptions raised by the sink propagate to the reader
    thread, which records them as a `SinkFailure` and stops forwarding to that sink.

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
    """What to run and how, validated on construction so a bad command fails before any process.

    Fields are normalized: `argv` becomes a tuple, `cwd` a `Path`, and `env` and `env_allowlist`
    are copied, so the spec cannot change after it is built.

    Args:
        argv: Program and arguments, passed to the OS without a shell. Empty arguments are allowed
            (`ssh-keygen -N ""` needs one); NUL bytes and an empty program name are not.
        cwd: Working directory for the child, or `None` to inherit cloud-boot's.
        env: Extra variables merged over the inherited host variables. Every key must itself be
            in `env_allowlist`, so a typo or a secret is rejected instead of silently passed.
        env_allowlist: Host variables the child may inherit and `env` may set.
        stdin: Bytes written to the child's stdin, which is then closed; `None` connects stdin to
            `/dev/null` so a child can never wait for terminal input.
        name: Short label for diagnostics, such as `"qemu-img"`, used instead of `argv[0]`.
        capture_limit_bytes: Per-stream in-memory capture limit; see `MemorySink`.
    """

    argv: tuple[str, ...]
    cwd: Path | str | None = None
    env: Mapping[str, str] | None = None
    env_allowlist: frozenset[str] = field(default_factory=lambda: DEFAULT_ENV_ALLOWLIST)
    stdin: bytes | None = None
    name: str | None = None
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES

    def __post_init__(self) -> None:
        args = tuple(self.argv)

        if not args:
            raise ValueError("argv must contain at least one argument")
        # Empty arguments are legitimate with shell=False (for example `ssh-keygen -N ""`).
        if any(not isinstance(arg, str) or "\x00" in arg for arg in args):
            raise ValueError("argv entries must be strings without NUL bytes")
        if not args[0]:
            raise ValueError("the program name in argv[0] must not be empty")
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
        limit = self.capture_limit_bytes
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("capture_limit_bytes must be a non-negative integer")


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CommandResult:
    """What a finished child process did. Nothing here is interpreted; callers decide meaning.

    `returncode` follows `subprocess`: a negative value means the child was killed by that signal,
    for example -9 for SIGKILL. `stdout` and `stderr` hold only the newest bytes up to the capture
    limit; `stdout_dropped_bytes` and `stderr_dropped_bytes` count what was cut, so callers can
    tell complete output from a tail. `timed_out` is set when cloud-boot had to stop the child at
    its deadline, even if the child then exited 0. `sink_failures` lists live sinks that failed.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    duration_seconds: float
    timed_out: bool = False
    sink_failures: tuple[SinkFailure, ...] = ()
    stdout_dropped_bytes: int = 0
    stderr_dropped_bytes: int = 0

    @property
    def succeeded(self) -> bool:
        """Whether the child exited 0 on its own, without being stopped at its deadline."""
        return self.returncode == 0 and not self.timed_out


# ----------------------------------------------------------------------------------------------------
def _resolve_deadline(
    *,
    timeout: float | None,
    deadline: float | None,
    clock: Clock,
    default: float,
) -> float:
    """Turn a relative `timeout` or an absolute `deadline` into one absolute monotonic deadline.

    Exactly one of `timeout` and `deadline` may be given; with neither, `default` seconds from
    now is used. A deadline already in the past is allowed and simply means "expired". Infinite
    and NaN values are rejected, so every wait stays bounded.
    """
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
    return clock() + default


# ----------------------------------------------------------------------------------------------------
class ManagedProcess:
    """A running child process: bounded waiting, process-group signals, and live output capture.

    Created by `CommandRunner.start`; most callers use `CommandRunner.run` instead and never see
    one. It is for long-running children such as QEMU, which `vm.py` starts, leaves running, and
    later stops. Reader threads start on construction, so every instance must end with `wait` or
    `stop`, which collect the threads and return the `CommandResult`.
    """

    def __init__(
        self,
        process: PopenLike,
        spec: CommandSpec,
        sinks: StreamSinks,
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
        self._stdout = MemorySink(spec.capture_limit_bytes)
        self._stderr = MemorySink(spec.capture_limit_bytes)
        self._threads: list[threading.Thread] = []
        self._stdin_thread: threading.Thread | None = None
        self._sink_failures: list[SinkFailure] = []
        self._sink_failures_lock = threading.Lock()
        # Drain stdout and stderr and feed stdin on separate threads from the start. A pipe holds
        # only a small kernel buffer, so a child writing to an unread pipe, or cloud-boot writing to
        # an unread stdin, would block forever. The stdin thread exists only when `spec.stdin` is set.
        self._start_reader(process.stdout, "stdout", self._stdout, self._stdout_sink)
        self._start_reader(process.stderr, "stderr", self._stderr, self._stderr_sink)
        self._start_stdin_writer()

    @property
    def pid(self) -> int:
        """Return the child PID, also its process-group ID since the child starts a new session."""
        return self._process.pid

    def poll(self) -> int | None:
        """Return the current exit status without waiting."""
        return self._process.poll()

    @property
    def sink_failures(self) -> tuple[SinkFailure, ...]:
        """Return output sinks that have failed so far; readable while the child still runs."""
        with self._sink_failures_lock:
            return tuple(self._sink_failures)

    def _record_sink_failure(self, name: StreamName, error: Exception) -> None:
        with self._sink_failures_lock:
            self._sink_failures.append(SinkFailure(stream=name, error=error))

    def _start_reader(
        self,
        stream: IO[bytes] | None,
        name: StreamName,
        capture: MemorySink,
        sink: Sink,
    ) -> None:
        """Start a daemon thread that copies one child stream into `capture` and the live `sink`.

        It uses `read1` when available, which returns as soon as any bytes arrive instead of
        waiting for a full 64 KiB block, so live sinks such as the console log stay current. Each
        chunk reaches `capture` first, so the in-memory tail survives a failing sink. The thread
        ends at EOF, which comes only once every process holding the pipe's write end has exited;
        `_finish` handles children whose descendants keep it open.
        """
        if stream is None:
            return

        def read_stream() -> None:
            forward = sink
            try:
                read1 = getattr(stream, "read1", None)
                reader = cast(Callable[[int], bytes], read1) if callable(read1) else stream.read
                while True:
                    chunk = reader(64 * 1024)
                    if not chunk:
                        return
                    capture.write(chunk)
                    if forward is None:
                        continue
                    try:
                        _write_sink(forward, chunk)
                    except Exception as e:
                        # Keep draining: closing the pipe early would SIGPIPE or block the child.
                        self._record_sink_failure(name, e)
                        forward = None
            finally:
                stream.close()

        # Daemon, so a reader stuck on a pipe that never reaches EOF cannot keep cloud-boot alive.
        thread = threading.Thread(target=read_stream, name="cloud-boot-output", daemon=True)
        thread.start()
        self._threads.append(thread)

    def _start_stdin_writer(self) -> None:
        """Start a daemon thread that writes `spec.stdin` once and then closes the pipe.

        Closing sends EOF, which is how `sh -s` on the guest knows the script is complete. If the
        child exits without reading everything, the write fails with `BrokenPipeError` (Python
        ignores SIGPIPE, so cloud-boot itself is not killed). That is not an error here: the
        child's exit status already says what happened.
        """
        data = self.spec.stdin
        if data is None:
            return
        stream = self._process.stdin
        if stream is None:
            return

        def write_stdin() -> None:
            try:
                stream.write(data)
            except (BrokenPipeError, OSError, ValueError):
                pass
            finally:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()

        # Daemon, so a write blocked on a child that never reads stdin cannot keep cloud-boot alive.
        thread = threading.Thread(target=write_stdin, name="cloud-boot-stdin", daemon=True)
        thread.start()
        self._stdin_thread = thread

    def _signal_group(self, sig: signal.Signals) -> None:
        """Send `sig` to the child's whole process group, falling back to the child alone.

        The child leads its own session, so its PID is the group ID and `killpg` also reaches
        anything it spawned, such as a grandchild still holding an output pipe. If the group no
        longer exists, only the child itself is signalled, which `Popen` ignores once the child has
        been reaped. Errors are swallowed: signalling is best effort, and callers check the outcome
        with `poll`.

        POSIX only, like the rest of cloud-boot: Windows would need a Job Object to reach a
        child's descendants, not a branch here.
        """
        try:
            os.killpg(self._process.pid, sig)
        except ProcessLookupError:
            # The whole group is gone; signal the child alone, which Popen skips once reaped.
            with contextlib.suppress(OSError):
                if sig == signal.SIGKILL:
                    self._process.kill()
                else:
                    self._process.terminate()
        except OSError:
            # For example EPERM; signalling is best effort and callers check the outcome with poll.
            pass

    def terminate(self) -> None:
        """Send SIGTERM to the process group, asking it to exit."""
        self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        """Send SIGKILL to the process group; it cannot be caught or ignored."""
        self._signal_group(signal.SIGKILL)

    def _all_threads(self) -> list[threading.Thread]:
        threads = list(self._threads)
        if self._stdin_thread is not None:
            threads.append(self._stdin_thread)
        return threads

    def _threads_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._all_threads())

    def _join_threads(self, deadline: float) -> None:
        """Wait for every helper thread to end, but never past `deadline`."""
        for thread in self._all_threads():
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            thread.join(timeout=remaining)

    def _fallback_returncode(self) -> int:
        """Return the real exit status if the child has exited, else -SIGKILL as a placeholder.

        Used only when the child survived SIGKILL until the deadline, so the result is always
        marked `timed_out` and the placeholder is never mistaken for a real status.
        """
        returncode = self.poll()
        if returncode is not None:
            return returncode
        return -signal.SIGKILL

    def _finish(
        self,
        returncode: int,
        *,
        timed_out: bool = False,
        join_deadline: float,
    ) -> CommandResult:
        """Collect the helper threads and build the result, never waiting past `join_deadline`.

        Readers still running after the first join mean a descendant still holds an output pipe,
        so the whole group is killed to release it. Threads still alive at the deadline are left
        behind (they are daemons), and the result uses whatever output was captured so far.
        """
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
            sink_failures=self.sink_failures,
            stdout_dropped_bytes=self._stdout.dropped_bytes,
            stderr_dropped_bytes=self._stderr.dropped_bytes,
        )

    def _wait_until(self, deadline: float) -> int | None:
        """Poll until the child exits or `deadline` passes; return its status, or `None` on timeout.

        Polls every 20 ms through the injected clock and sleeper rather than `Popen.wait`, which
        uses real time, so tests can drive deadlines with a fake clock.
        """
        while True:
            returncode = self.poll()
            if returncode is not None:
                return returncode
            remaining = deadline - self._clock()
            if remaining <= 0:
                return None
            self._sleeper(min(0.02, remaining))

    def stop(self, *, timeout: float | None = None, deadline: float | None = None) -> CommandResult:
        """Stop the process group: SIGTERM first, then SIGKILL if needed, within one deadline.

        The end of the deadline, up to the grace period but never more than half of the time left,
        is reserved for SIGKILL to take effect; SIGTERM gets the rest. With neither `timeout` nor
        `deadline`, the grace period plus one second is used.

        Raises:
            ProcessTimeoutError: If the child is still running at the deadline even after SIGKILL.
                Its `result` holds the output captured so far.
        """
        absolute = _resolve_deadline(
            timeout=timeout,
            deadline=deadline,
            clock=self._clock,
            default=self._termination_grace_seconds + 1.0,
        )
        remaining = max(0.0, absolute - self._clock())
        # Reserve the end of the deadline for SIGKILL; SIGTERM gets the rest.
        kill_window = min(self._termination_grace_seconds, remaining / 2)
        kill_deadline = absolute - kill_window
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
        """Wait for the child to exit and return its result; on timeout, stop it and raise.

        An explicit `deadline` is a hard boundary: cleanup after a timeout must finish by it too, so
        an overrunning child gets SIGTERM and SIGKILL with no extra time. A relative `timeout`
        allows up to twice the grace period of cleanup after it expires. With neither,
        `DEFAULT_TIMEOUT_SECONDS` applies.

        Raises:
            ProcessTimeoutError: If the deadline passes first. The process group has been
                signalled, and `result` holds the output captured before it stopped, marked
                `timed_out`. `context["survived_sigkill"]` is true when the child may still be
                running.
        """
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
            survived_sigkill = False
            try:
                timed_out_result = self.stop(deadline=cleanup_deadline)
            except ProcessTimeoutError as e:
                timed_out_result = e.result
                survived_sigkill = True
            if timed_out_result is not None:
                timed_out_result = replace(timed_out_result, timed_out=True)
            raise ProcessTimeoutError(
                context={
                    "argv_length": len(self.spec.argv),
                    "pid": self.pid,
                    "survived_sigkill": survived_sigkill,
                },
                result=timed_out_result,
            )
        # The child has exited, but anything it started may still hold the output pipes open. A
        # command owns its whole group, so nothing it started may outlive it; if no descendant is
        # left, the group is gone and this does nothing.
        self.terminate()
        cleanup_deadline = self._clock() + self._termination_grace_seconds
        return self._finish(returncode, join_deadline=cleanup_deadline)


# ----------------------------------------------------------------------------------------------------
class CommandRunner:
    """The only way cloud-boot starts host commands; the module docstring lists its guarantees.

    One instance is shared by every component. Time and process creation are injected, so tests
    can replace `subprocess.Popen` with a fake and drive deadlines with a fake clock.

    Args:
        clock: Monotonic time source for deadlines.
        sleeper: Sleep function used while polling.
        popen_factory: Creates the child process; `subprocess.Popen` in production.
        default_timeout_seconds: Used by `run` when a caller passes neither timeout nor deadline.
        termination_grace_seconds: Time a process group gets after SIGTERM before SIGKILL.
    """

    def __init__(
        self,
        *,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
        popen_factory: PopenFactory = subprocess.Popen,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        termination_grace_seconds: float = TERMINATION_GRACE_SECONDS,
    ) -> None:
        for value in (default_timeout_seconds, termination_grace_seconds):
            if not 0 < value < float("inf"):
                raise ValueError("process timeouts must be positive and finite")
        self._clock = clock
        self._sleeper = sleeper
        self._popen_factory = popen_factory
        self._default_timeout_seconds = default_timeout_seconds
        self._termination_grace_seconds = termination_grace_seconds

    def _environment(self, spec: CommandSpec) -> dict[str, str]:
        """Build the child environment: allowlisted host variables, overridden by `spec.env`."""
        environment = {key: value for key, value in os.environ.items() if key in spec.env_allowlist}
        if spec.env:
            environment.update(spec.env)
        return environment

    def start(self, spec: CommandSpec, sinks: StreamSinks | None = None) -> ManagedProcess:
        """Start one command and return it running; the caller must end it with `wait` or `stop`.

        Raises:
            ProcessStartError: If the OS cannot create the process, for example because the
                program is missing or not executable.
        """
        try:
            process = self._popen_factory(
                list(spec.argv),
                cwd=str(spec.cwd) if spec.cwd is not None else None,
                env=self._environment(spec),
                stdin=subprocess.PIPE if spec.stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                # Own session and process group: `_signal_group` relies on the PID being the group
                # ID, and a terminal Ctrl-C reaches cloud-boot instead of the child.
                start_new_session=True,
            )
        except (OSError, ValueError) as e:
            raise ProcessStartError(
                "unable to start host command",
                context={
                    "command": spec.name or spec.argv[0],
                    "environment_keys": sorted(spec.env or {}),
                },
                remediation="Install the required host command and verify its path and permissions.",
            ) from e
        return ManagedProcess(
            process,
            spec,
            sinks or StreamSinks(),
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
        sinks: StreamSinks | None = None,
    ) -> CommandResult:
        """Run one command to completion within a bounded time and return its result.

        Raises:
            ProcessStartError: If the command cannot be started.
            ProcessTimeoutError: If it does not finish in time; see `ManagedProcess.wait`.
            ValueError: For an invalid `timeout` or `deadline`, before anything is started.
        """
        if timeout is None and deadline is None:
            timeout = self._default_timeout_seconds
        # Validate before starting, so a bad timeout or deadline cannot leave a child running.
        _resolve_deadline(timeout=timeout, deadline=deadline, clock=self._clock, default=0.0)
        process = self.start(spec, sinks)
        return process.wait(timeout=timeout, deadline=deadline)
