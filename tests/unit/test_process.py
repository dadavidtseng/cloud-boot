"""Unit tests for the bounded host process boundary."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from cloud_boot.errors import ExitCode
from cloud_boot.process import (
    DEFAULT_CAPTURE_LIMIT_BYTES,
    CommandRunner,
    CommandSpec,
    ManagedProcess,
    MemorySink,
    ProcessStartError,
    ProcessTimeoutError,
    SinkFailure,
    StreamSinks,
    _resolve_deadline,
)

PYTHON = sys.executable


def python_spec(
    *code: str,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    env_allowlist: frozenset[str] | None = None,
) -> CommandSpec:
    if env_allowlist is None:
        return CommandSpec((PYTHON, "-c", " ".join(code)), cwd=cwd, env=env)
    return CommandSpec(
        (PYTHON, "-c", " ".join(code)),
        cwd=cwd,
        env=env,
        env_allowlist=env_allowlist,
    )


class _FakeProcess:
    def __init__(self, *, exits_after_kill: bool) -> None:
        self.pid = 999_999_999
        self.stdin = None
        self.stdout = io.BytesIO(b"partial stdout")
        self.stderr = io.BytesIO(b"partial stderr")
        self.exits_after_kill = exits_after_kill
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        if self.killed and self.exits_after_kill:
            return -9
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def test_runs_without_shell_and_captures_both_streams() -> None:
    result = CommandRunner().run(
        python_spec("import sys; print('out'); print('err', file=sys.stderr)"),
    )

    assert result.succeeded
    assert result.stdout == b"out\n"
    assert result.stderr == b"err\n"
    assert result.returncode == 0
    assert result.duration_seconds >= 0


def test_streams_are_forwarded_to_injected_sinks() -> None:
    stdout = MemorySink()
    stderr = MemorySink()
    result = CommandRunner().run(
        python_spec("import sys; sys.stdout.write('a'); sys.stderr.write('b')"),
        sinks=StreamSinks(stdout=stdout, stderr=stderr),
    )

    assert result.stdout == b"a"
    assert result.stderr == b"b"
    assert stdout.getvalue() == b"a"
    assert stderr.getvalue() == b"b"


def test_stdin_is_written_without_shell_interpolation() -> None:
    with_input = CommandRunner().run(
        CommandSpec(
            (PYTHON, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"),
            stdin=b"$HOME;\n",
        ),
        timeout=5,
    )
    assert with_input.stdout == b"$HOME;\n"


def test_start_does_not_block_on_large_stdin() -> None:
    started: list[ManagedProcess] = []
    errors: list[BaseException] = []

    def start_process() -> None:
        try:
            started.append(
                CommandRunner().start(
                    CommandSpec(
                        (PYTHON, "-c", "import time; time.sleep(5)"),
                        stdin=b"x" * (1024 * 1024),
                    )
                )
            )
        except BaseException as e:
            errors.append(e)

    thread = threading.Thread(target=start_process, daemon=True)
    thread.start()
    thread.join(timeout=0.5)

    assert not thread.is_alive(), "start() blocked while writing stdin"
    assert not errors
    process = started[0]
    assert hasattr(process, "stop")
    process.stop(timeout=1)


def test_environment_is_allowlisted_and_explicit_values_override() -> None:
    result = CommandRunner().run(
        python_spec(
            "import os; print(os.getenv('PATH', '')); "
            "print(os.getenv('CLOUD_BOOT_TEST', 'missing'));",
            env={"CLOUD_BOOT_TEST": "present"},
            env_allowlist=frozenset({"PATH", "CLOUD_BOOT_TEST"}),
        )
    )

    assert result.succeeded
    assert result.stdout.splitlines()[-1] == b"present"


def test_disallowed_environment_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        CommandSpec((PYTHON, "-c", "pass"), env={"SECRET": "nope"})


def test_missing_command_has_typed_start_error() -> None:
    with pytest.raises(ProcessStartError) as caught:
        CommandRunner().run(CommandSpec(("definitely-not-a-cloud-boot-command",)), timeout=1)

    assert caught.value.code == "process-start-failed"
    assert caught.value.exit_code == ExitCode.HOST_UNAVAILABLE
    assert "environment_keys" in caught.value.context


def test_nonzero_status_is_a_result_not_an_error() -> None:
    result = CommandRunner().run(python_spec("raise SystemExit(7)"))

    assert not result.succeeded
    assert result.returncode == 7


def test_deadline_timeout_terminates_child() -> None:
    started = time.monotonic()
    with pytest.raises(ProcessTimeoutError) as caught:
        CommandRunner(termination_grace_seconds=0.05).run(
            python_spec("import time; time.sleep(30)"),
            timeout=0.1,
        )

    assert caught.value.exit_code == ExitCode.TIMEOUT
    assert time.monotonic() - started < 5


@pytest.mark.skipif(os.name != "posix", reason="process-group signaling is POSIX-specific")
def test_termination_reaches_child_process_group() -> None:
    child = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print(p.pid, flush=True)"
    )
    process = CommandRunner(termination_grace_seconds=0.05).start(CommandSpec((PYTHON, "-c", child)))
    deadline = time.monotonic() + 2
    while not process._stdout.getvalue():
        if time.monotonic() >= deadline:
            process.kill()
            pytest.fail("child process did not announce its PID")
        time.sleep(0.01)
    grandchild_pid = int(process._stdout.getvalue().splitlines()[0])
    while process.poll() is None:
        if time.monotonic() >= deadline:
            process.kill()
            pytest.fail("group leader did not exit")
        time.sleep(0.01)

    try:
        process.stop(timeout=0.5)
        deadline = time.monotonic() + 2
        while True:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                pytest.fail("grandchild survived process-group termination")
            time.sleep(0.01)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(grandchild_pid, 9)


@pytest.mark.skipif(os.name != "posix", reason="process-group signaling is POSIX-specific")
def test_normal_exit_cleans_inherited_pipes_and_stdin() -> None:
    child = (
        "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"
    )
    process = CommandRunner().start(CommandSpec((PYTHON, "-c", child), stdin=b"x" * (1024 * 1024)))
    results: list[object] = []
    errors: list[BaseException] = []

    def wait_process() -> None:
        try:
            results.append(process.wait(timeout=1))
        except BaseException as e:
            errors.append(e)

    waiter = threading.Thread(target=wait_process, daemon=True)
    started = time.monotonic()
    waiter.start()
    waiter.join(timeout=1)
    if waiter.is_alive():
        process.kill()
        waiter.join(timeout=1)

    assert not waiter.is_alive(), "wait() blocked on inherited stdin or output pipes"
    assert not errors
    assert results
    assert time.monotonic() - started < 1.5


def test_timeout_allows_sigterm_handler_and_preserves_output() -> None:
    child = (
        "import signal,sys,time; "
        "exec(\"def handle(*_):\\n print('term', flush=True)\\n raise SystemExit(0)\"); "
        "signal.signal(signal.SIGTERM, handle); time.sleep(30)"
    )
    with pytest.raises(ProcessTimeoutError) as caught:
        CommandRunner(termination_grace_seconds=0.1).run(
            CommandSpec((PYTHON, "-c", child)),
            timeout=0.3,
        )

    assert caught.value.result is not None
    assert caught.value.result.timed_out
    assert b"term" in caught.value.result.stdout


def test_start_is_nonblocking() -> None:
    process = CommandRunner().start(python_spec("import time; time.sleep(.1)"))
    result = process.wait(timeout=5)
    assert result.succeeded


def test_timeout_budget_does_not_cut_short_command() -> None:
    started = time.monotonic()
    result = CommandRunner().run(
        python_spec("import time; time.sleep(1.25)"),
        timeout=5,
    )

    assert result.succeeded
    assert time.monotonic() - started >= 1.0


def test_stop_respects_hard_deadline_with_injected_clock() -> None:
    fake = _FakeProcess(exits_after_kill=False)
    current = 0.0

    def clock() -> float:
        return current

    def sleeper(duration: float) -> None:
        nonlocal current
        current += duration

    runner = CommandRunner(
        clock=clock,
        sleeper=sleeper,
        popen_factory=lambda *_args, **_kwargs: fake,
        termination_grace_seconds=0.01,
    )
    process = runner.start(CommandSpec(("fake-command",)))

    with pytest.raises(ProcessTimeoutError):
        process.stop(deadline=0.1)

    assert current <= 0.1
    assert fake.terminated
    assert fake.killed


def test_stop_default_uses_injected_termination_grace() -> None:
    fake = _FakeProcess(exits_after_kill=False)
    current = 0.0

    def clock() -> float:
        return current

    def sleeper(duration: float) -> None:
        nonlocal current
        current += duration

    runner = CommandRunner(
        clock=clock,
        sleeper=sleeper,
        popen_factory=lambda *_args, **_kwargs: fake,
        termination_grace_seconds=0.01,
    )
    process = runner.start(CommandSpec(("fake-command",)))

    with pytest.raises(ProcessTimeoutError):
        process.stop()

    assert current <= 1.02


def test_timeout_error_retains_timed_out_result() -> None:
    fake = _FakeProcess(exits_after_kill=True)
    current = 0.0

    def clock() -> float:
        return current

    def sleeper(duration: float) -> None:
        nonlocal current
        current += duration

    runner = CommandRunner(
        clock=clock,
        sleeper=sleeper,
        popen_factory=lambda *_args, **_kwargs: fake,
    )
    process = runner.start(CommandSpec(("fake-command",)))
    # The fake clock gives _finish no join time, so drain the readers before timing out.
    for reader in process._threads:
        reader.join(timeout=2)
        assert not reader.is_alive()

    with pytest.raises(ProcessTimeoutError) as caught:
        process.wait(timeout=0.1)

    assert caught.value.result is not None
    assert caught.value.context["survived_sigkill"] is False
    assert caught.value.result.timed_out
    assert caught.value.result.stdout == b"partial stdout"
    assert caught.value.result.stderr == b"partial stderr"


def test_run_rejects_an_unbounded_timeout_before_starting_anything() -> None:
    started: list[object] = []

    def factory(*args: object, **_kwargs: object) -> _FakeProcess:
        started.append(args)
        return _FakeProcess(exits_after_kill=True)

    runner = CommandRunner(popen_factory=factory)

    with pytest.raises(ValueError, match="timeout must be finite"):
        runner.run(CommandSpec(("fake-command",)), timeout=float("inf"))

    assert started == []


@pytest.mark.parametrize("value", [0.0, float("inf"), float("nan")])
def test_runner_rejects_invalid_time_settings(value: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        CommandRunner(default_timeout_seconds=value)
    with pytest.raises(ValueError, match="positive and finite"):
        CommandRunner(termination_grace_seconds=value)


def test_wait_timeout_reports_a_child_that_survives_sigkill() -> None:
    fake = _FakeProcess(exits_after_kill=False)
    current = 0.0

    def clock() -> float:
        return current

    def sleeper(duration: float) -> None:
        nonlocal current
        current += duration

    runner = CommandRunner(
        clock=clock,
        sleeper=sleeper,
        popen_factory=lambda *_args, **_kwargs: fake,
        termination_grace_seconds=0.01,
    )
    process = runner.start(CommandSpec(("fake-command",)))

    with pytest.raises(ProcessTimeoutError) as caught:
        process.wait(timeout=0.1)

    assert fake.killed
    assert caught.value.context["survived_sigkill"] is True
    assert caught.value.result is not None
    assert caught.value.result.timed_out


def test_callable_sinks_receive_output() -> None:
    stdout: list[bytes] = []
    stderr: list[bytes] = []
    result = CommandRunner().run(
        python_spec("import sys; sys.stdout.write('out'); sys.stderr.write('err')"),
        sinks=StreamSinks(stdout=stdout.append, stderr=stderr.append),
    )

    assert result.stdout == b"out"
    assert result.stderr == b"err"
    assert b"out" in stdout
    assert b"err" in stderr


def test_non_allowlisted_host_environment_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD_BOOT_DENIED", "secret")
    result = CommandRunner().run(
        python_spec(
            "import os; print(os.getenv('CLOUD_BOOT_DENIED', 'missing'))",
            env_allowlist=frozenset({"PATH"}),
        )
    )

    assert result.stdout.strip() == b"missing"


def test_command_spec_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CommandSpec(())
    with pytest.raises(ValueError, match="NUL"):
        CommandSpec(("echo\x00",))
    with pytest.raises(ValueError, match="NUL"):
        CommandSpec(("echo", "a\x00b"))
    with pytest.raises(ValueError, match=r"argv\[0\]"):
        CommandSpec(("", "arg"))


def test_empty_arguments_reach_the_child_unchanged() -> None:
    result = CommandRunner().run(
        CommandSpec((PYTHON, "-c", "import sys; print(repr(sys.argv[1:]))", "", "x", "")),
        timeout=10,
    )

    assert result.stdout.strip() == b"['', 'x', '']"


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_deadline_resolution_rejects_unbounded_values(value: float) -> None:
    with pytest.raises(ValueError, match="timeout must be finite"):
        _resolve_deadline(timeout=value, deadline=None, clock=time.monotonic, default=1.0)
    with pytest.raises(ValueError, match="deadline must be finite"):
        _resolve_deadline(timeout=None, deadline=value, clock=time.monotonic, default=1.0)


def test_command_spec_accepts_path_cwd(tmp_path: Path) -> None:
    result = CommandRunner().run(
        python_spec("import os; print(os.getcwd())", cwd=tmp_path),
    )
    assert Path(result.stdout.decode().strip()) == tmp_path


def test_failed_sink_keeps_draining_and_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    unhandled: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: unhandled.append(args.exc_value))
    failing_calls: list[bytes] = []
    stderr_received: list[bytes] = []
    disk_full = OSError(28, "No space left on device")

    def failing_sink(chunk: bytes) -> None:
        failing_calls.append(chunk)
        raise disk_full

    # Far more than a pipe buffer, so closing the pipe after the failure would SIGPIPE the child.
    result = CommandRunner().run(
        python_spec(
            "import sys; sys.stdout.buffer.write(b'x' * 300_000); sys.stdout.flush(); "
            "sys.stderr.write('still forwarded')"
        ),
        timeout=10,
        sinks=StreamSinks(stdout=failing_sink, stderr=stderr_received.append),
    )

    assert result.returncode == 0
    assert result.stdout == b"x" * 300_000
    assert len(failing_calls) == 1
    assert result.sink_failures == (SinkFailure(stream="stdout", error=disk_full),)
    assert b"".join(stderr_received) == b"still forwarded"
    assert unhandled == []


def test_sink_failure_is_visible_while_the_process_runs() -> None:
    def failing_sink(_chunk: bytes) -> None:
        raise RuntimeError("console log unavailable")

    process = CommandRunner(termination_grace_seconds=0.05).start(
        python_spec("import sys, time; sys.stderr.write('boot'); sys.stderr.flush(); time.sleep(30)"),
        StreamSinks(stderr=failing_sink),
    )
    try:
        deadline = time.monotonic() + 2
        while not process.sink_failures:
            if time.monotonic() >= deadline:
                pytest.fail("sink failure was not reported while the process ran")
            time.sleep(0.01)

        assert process.poll() is None
        assert [failure.stream for failure in process.sink_failures] == ["stderr"]
        assert isinstance(process.sink_failures[0].error, RuntimeError)
    finally:
        result = process.stop(timeout=1)
    assert len(result.sink_failures) == 1


def test_capture_keeps_the_tail_while_sinks_receive_everything() -> None:
    stdout_received: list[bytes] = []
    code = (
        "import sys; sys.stdout.write('x' * 3000 + 'END'); sys.stdout.flush(); "
        "sys.stderr.write('e' * 1500)"
    )

    result = CommandRunner().run(
        CommandSpec((PYTHON, "-c", code), capture_limit_bytes=1000),
        timeout=10,
        sinks=StreamSinks(stdout=stdout_received.append),
    )

    assert result.stdout == (b"x" * 3000 + b"END")[-1000:]
    assert result.stdout_dropped_bytes == 2003
    assert result.stderr == b"e" * 1000
    assert result.stderr_dropped_bytes == 500
    assert b"".join(stdout_received) == b"x" * 3000 + b"END"


def test_default_capture_limit_bounds_memory() -> None:
    result = CommandRunner().run(
        python_spec(
            f"import sys; sys.stdout.buffer.write(b'x' * {DEFAULT_CAPTURE_LIMIT_BYTES + 10})"
        ),
        timeout=10,
    )

    assert len(result.stdout) == DEFAULT_CAPTURE_LIMIT_BYTES
    assert result.stdout_dropped_bytes == 10


def test_zero_capture_limit_disables_capture_but_not_sinks() -> None:
    received: list[bytes] = []

    result = CommandRunner().run(
        CommandSpec((PYTHON, "-c", "print('console')"), capture_limit_bytes=0),
        timeout=10,
        sinks=StreamSinks(stdout=received.append),
    )

    assert result.stdout == b""
    assert result.stdout_dropped_bytes == len(b"console\n")
    assert b"".join(received) == b"console\n"


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_rejects_invalid_capture_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="capture_limit_bytes"):
        CommandSpec((PYTHON, "-c", "pass"), capture_limit_bytes=limit)  # type: ignore[arg-type]


def test_memory_sink_limit_keeps_recent_bytes() -> None:
    limited = MemorySink(limit=4)
    unlimited = MemorySink()

    for sink in (limited, unlimited):
        sink.write(b"abc")
        sink.write(b"def")

    assert limited.getvalue() == b"cdef"
    assert limited.dropped_bytes == 2
    assert unlimited.getvalue() == b"abcdef"
    assert unlimited.dropped_bytes == 0
    with pytest.raises(ValueError, match="non-negative"):
        MemorySink(limit=-1)
