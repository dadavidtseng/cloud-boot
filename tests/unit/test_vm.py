"""Unit tests for the process-only QEMU backend."""

from __future__ import annotations

import io
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cloud_boot.errors import ErrorPhase, ExitCode
from cloud_boot.image_store import VerifiedImage
from cloud_boot.models import ResourceSpec
from cloud_boot.process import CommandRunner, CommandSpec, MemorySink, ProcessTimeoutError
from cloud_boot.vm import (
    CONSOLE_CAPTURE_LIMIT_BYTES,
    DiskArtifact,
    OverlayError,
    QemuBackend,
    StopReason,
    VmHandle,
    VmLaunchError,
    VmLaunchSpec,
    allocate_loopback_port,
    build_qemu_argv,
)

PORT_CONFLICT = (
    b"qemu-system-x86_64: -netdev user,id=net0,hostfwd=tcp:127.0.0.1:40001-:22: "
    b"Could not set up host forwarding rule 'tcp:127.0.0.1:40001-:22'"
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakePopen:
    """A child that exits on its own after `exit_after_polls`, or reacts to TERM/KILL."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        exit_after_polls: int | None = None,
        returncode: int = 0,
        on_terminate: int | None = 0,
        on_kill: int | None = -9,
    ) -> None:
        self.pid = 999_999_999  # Above pid_max, so killpg fails and ManagedProcess falls back.
        self.stdin = None
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode: int | None = None
        self.signals_while_running: list[str] = []
        self._polls_left = exit_after_polls
        self._exit_code = returncode
        self._on_terminate = on_terminate
        self._on_kill = on_kill

    def poll(self) -> int | None:
        if self.returncode is None and self._polls_left is not None:
            if self._polls_left <= 0:
                self.returncode = self._exit_code
            else:
                self._polls_left -= 1
        return self.returncode

    def terminate(self) -> None:
        if self.returncode is None:
            self.signals_while_running.append("TERM")
            self.returncode = self._on_terminate

    def kill(self) -> None:
        if self.returncode is None:
            self.signals_while_running.append("KILL")
            self.returncode = self._on_kill


class FakeFactory:
    def __init__(
        self, processes: list[FakePopen], on_call: Callable[[tuple[str, ...]], None] | None = None
    ) -> None:
        self.processes = processes
        self.argvs: list[tuple[str, ...]] = []
        self._on_call = on_call

    def __call__(self, argv: list[str], **_kwargs: Any) -> FakePopen:
        self.argvs.append(tuple(argv))
        if self._on_call is not None:
            self._on_call(tuple(argv))
        return self.processes.pop(0)


def make_backend(factory: FakeFactory, ports: tuple[int, ...] = (40001, 40002, 40003)) -> Any:
    clock = FakeClock()
    runner = CommandRunner(
        clock=clock, sleeper=clock.sleep, popen_factory=factory, termination_grace_seconds=0.5
    )
    allocated = iter(ports)
    backend = QemuBackend(
        runner, port_allocator=lambda: next(allocated), clock=clock, sleeper=clock.sleep
    )
    return backend, runner, clock


def launch_spec(tmp_path: Path, console: Any = None) -> VmLaunchSpec:
    disk = DiskArtifact(
        path=tmp_path / "run" / "overlay.qcow2", backing_image=tmp_path / "base.img", size_gib=8
    )
    return VmLaunchSpec(
        disk=disk,
        seed_image=tmp_path / "run" / "seed.iso",
        resources=ResourceSpec(),
        console=console if console is not None else MemorySink(),
    )


def running_handle(runner: CommandRunner) -> VmHandle:
    """Start the factory's next fake directly, skipping the startup window."""
    managed = runner.start(CommandSpec(("qemu-system-x86_64",)))
    return VmHandle(
        process=managed,
        argv=("qemu-system-x86_64",),
        ssh_host="127.0.0.1",
        ssh_port=40001,
        launch_attempts=1,
    )


def eventually(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 2
    while not condition():
        if time.monotonic() >= deadline:
            pytest.fail("condition was not met in time")
        time.sleep(0.01)


def verified_image(path: Path) -> VerifiedImage:
    return VerifiedImage(
        path=path, digest="0" * 64, size_bytes=1, source_url="https://images.test/x", retrieved_at=0
    )


def test_prepare_creates_overlay_with_absolute_backing_and_explicit_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    factory = FakeFactory([FakePopen(exit_after_polls=0)])
    backend, _, clock = make_backend(factory)

    disk = backend.prepare(
        verified_image(Path("cache/base.img")), tmp_path / "run", disk_gib=8, deadline=30.0
    )

    backing = tmp_path / "cache" / "base.img"
    overlay = tmp_path / "run" / "overlay.qcow2"
    assert factory.argvs == [
        (
            "qemu-img",
            "create",
            "-q",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            str(backing),
            str(overlay),
            "8G",
        )
    ]
    assert disk == DiskArtifact(path=overlay, backing_image=backing, size_gib=8)
    assert clock.now <= 30.0


def test_prepare_never_overwrites_an_existing_overlay(tmp_path: Path) -> None:
    (tmp_path / "overlay.qcow2").write_bytes(b"guest state")
    factory = FakeFactory([])
    backend, _, _ = make_backend(factory)

    with pytest.raises(OverlayError, match="already exists"):
        backend.prepare(verified_image(tmp_path / "base.img"), tmp_path, disk_gib=8, deadline=30.0)

    assert factory.argvs == []
    assert (tmp_path / "overlay.qcow2").read_bytes() == b"guest state"


def test_prepare_failure_is_a_preparation_error_and_removes_partial_overlay(
    tmp_path: Path,
) -> None:
    def leave_partial(argv: tuple[str, ...]) -> None:
        Path(argv[-2]).write_bytes(b"partial")

    stderr = b"qemu-img: Could not open backing file: No such file or directory"
    factory = FakeFactory([FakePopen(stderr=stderr, exit_after_polls=0, returncode=1)], leave_partial)
    backend, _, _ = make_backend(factory)

    with pytest.raises(OverlayError) as raised:
        backend.prepare(verified_image(tmp_path / "base.img"), tmp_path, disk_gib=8, deadline=30.0)

    assert raised.value.phase == ErrorPhase.PREPARATION
    assert raised.value.exit_code == ExitCode.PREPARATION
    assert "Could not open backing file" in str(raised.value.context["stderr"])
    assert not (tmp_path / "overlay.qcow2").exists()


def test_qemu_argv_is_deterministic_and_forwards_only_loopback_ssh(tmp_path: Path) -> None:
    spec = launch_spec(tmp_path)

    argv = build_qemu_argv(spec, 40001)

    assert argv == build_qemu_argv(spec, 40001)
    assert argv == (
        "qemu-system-x86_64", "-name", "cloud-boot", "-nodefaults", "-no-user-config",
        "-machine", "q35,accel=kvm", "-cpu", "host", "-smp", "2", "-m", "2048M",
        "-display", "none", "-serial", "stdio",
        "-drive", f"file={tmp_path}/run/overlay.qcow2,if=virtio,format=qcow2",
        "-drive", f"file={tmp_path}/run/seed.iso,if=virtio,format=raw,readonly=on",
        "-netdev", "user,id=net0,hostfwd=tcp:127.0.0.1:40001-:22",
        "-device", "virtio-net-pci,netdev=net0",
    )  # fmt: skip
    # QEMU only sees the overlay; the base image is reached read-only through its backing file.
    assert not any(str(spec.disk.backing_image) in argument for argument in argv)


def test_qemu_argv_escapes_commas_in_paths(tmp_path: Path) -> None:
    spec = launch_spec(tmp_path / "a,b")

    drive = build_qemu_argv(spec, 40001)[18]

    assert drive == f"file={tmp_path}/a,,b/run/overlay.qcow2,if=virtio,format=qcow2"


@pytest.mark.parametrize(
    ("field", "value"), [("name", "bad,name"), ("name", ""), ("network_mode", "restricted")]
)
def test_launch_spec_rejects_unsafe_name_and_unsupported_network(
    tmp_path: Path, field: str, value: str
) -> None:
    fields: dict[str, Any] = {
        "disk": DiskArtifact(tmp_path / "overlay.qcow2", tmp_path / "base.img", 8),
        "seed_image": tmp_path / "seed.iso",
        "resources": ResourceSpec(),
        "console": MemorySink(),
        field: value,
    }

    with pytest.raises(ValueError, match=r"network mode|VM name"):
        VmLaunchSpec(**fields)


def test_start_streams_console_and_returns_after_the_startup_window(tmp_path: Path) -> None:
    factory = FakeFactory([FakePopen(stdout=b"[    0.000000] Linux version 6.8")])
    backend, _, clock = make_backend(factory)
    console = MemorySink()

    handle = backend.start(launch_spec(tmp_path, console))

    assert (handle.ssh_host, handle.ssh_port, handle.launch_attempts) == ("127.0.0.1", 40001, 1)
    assert handle.argv == factory.argvs[0]
    assert clock.now == pytest.approx(2.0)
    assert handle.process.spec.capture_limit_bytes == CONSOLE_CAPTURE_LIMIT_BYTES
    eventually(lambda: console.getvalue() == b"[    0.000000] Linux version 6.8")


def test_start_retries_port_conflicts_with_fresh_ports(tmp_path: Path) -> None:
    conflict = {"stderr": PORT_CONFLICT, "exit_after_polls": 0, "returncode": 1}
    factory = FakeFactory([FakePopen(**conflict), FakePopen(**conflict), FakePopen()])
    backend, _, _ = make_backend(factory)

    handle = backend.start(launch_spec(tmp_path))

    assert (handle.ssh_port, handle.launch_attempts) == (40003, 3)
    forwards = [argv[argv.index("-netdev") + 1] for argv in factory.argvs]
    assert forwards == [
        f"user,id=net0,hostfwd=tcp:127.0.0.1:{port}-:22" for port in (40001, 40002, 40003)
    ]


def test_start_gives_up_after_three_port_conflicts(tmp_path: Path) -> None:
    conflict = {"stderr": PORT_CONFLICT, "exit_after_polls": 0, "returncode": 1}
    factory = FakeFactory([FakePopen(**conflict) for _ in range(3)])
    backend, _, _ = make_backend(factory)

    with pytest.raises(VmLaunchError, match="after 3 attempts") as raised:
        backend.start(launch_spec(tmp_path))

    assert raised.value.phase == ErrorPhase.VM
    assert raised.value.exit_code == ExitCode.VM
    assert raised.value.context["attempts"] == 3
    assert raised.value.context["port_conflict"] is True


def test_start_does_not_retry_other_startup_failures(tmp_path: Path) -> None:
    stderr = b"qemu-system-x86_64: -accel kvm: failed to initialize kvm: Permission denied"
    factory = FakeFactory([FakePopen(stderr=stderr, exit_after_polls=3, returncode=1)])
    backend, _, _ = make_backend(factory)

    with pytest.raises(VmLaunchError, match="exited during startup") as raised:
        backend.start(launch_spec(tmp_path))

    assert len(factory.argvs) == 1
    assert raised.value.context["attempts"] == 1
    assert raised.value.context["port_conflict"] is False
    assert "Permission denied" in str(raised.value.context["stderr"])


def test_console_sink_failure_keeps_the_vm_running(tmp_path: Path) -> None:
    def failing_console(_chunk: bytes) -> None:
        raise OSError(28, "No space left on device")

    factory = FakeFactory([FakePopen(stdout=b"console output")])
    backend, _, _ = make_backend(factory)

    handle = backend.start(launch_spec(tmp_path, failing_console))

    eventually(lambda: bool(handle.process.sink_failures))
    assert handle.process.poll() is None
    assert handle.process.sink_failures[0].stream == "stdout"


def test_stop_reports_guest_exit_without_sending_signals(tmp_path: Path) -> None:
    fake = FakePopen(exit_after_polls=5)
    backend, runner, clock = make_backend(FakeFactory([fake]))
    handle = running_handle(runner)

    stopped = backend.stop(handle, deadline=30.0)

    assert stopped.reason == StopReason.EXITED
    assert stopped.result.returncode == 0
    assert fake.signals_while_running == []
    assert clock.now < 20.0


def test_stop_escalates_to_sigterm_inside_the_deadline(tmp_path: Path) -> None:
    fake = FakePopen()
    backend, runner, clock = make_backend(FakeFactory([fake]))
    handle = running_handle(runner)

    stopped = backend.stop(handle, deadline=30.0)

    assert stopped.reason == StopReason.TERMINATED
    assert fake.signals_while_running == ["TERM"]
    # The guest had until 10 seconds before the deadline to shut down on its own.
    assert clock.now == pytest.approx(20.0)


def test_stop_escalates_to_sigkill_inside_the_deadline(tmp_path: Path) -> None:
    fake = FakePopen(on_terminate=None)
    backend, runner, clock = make_backend(FakeFactory([fake]))
    handle = running_handle(runner)

    stopped = backend.stop(handle, deadline=30.0)

    assert stopped.reason == StopReason.KILLED
    assert stopped.result.returncode == -9
    assert fake.signals_while_running == ["TERM", "KILL"]
    assert clock.now <= 30.0


def test_stop_raises_when_qemu_survives_sigkill(tmp_path: Path) -> None:
    fake = FakePopen(on_terminate=None, on_kill=None)
    backend, runner, clock = make_backend(FakeFactory([fake]))
    handle = running_handle(runner)

    with pytest.raises(ProcessTimeoutError) as raised:
        backend.stop(handle, deadline=30.0)

    assert raised.value.exit_code == ExitCode.TIMEOUT
    assert fake.signals_while_running == ["TERM", "KILL"]
    assert clock.now <= 30.0


def test_allocate_loopback_port_returns_a_bindable_loopback_port() -> None:
    port = allocate_loopback_port()

    assert 1 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


@pytest.mark.parametrize("window", [0.0, -1.0, float("inf"), float("nan")])
def test_backend_rejects_unbounded_windows(window: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        QemuBackend(CommandRunner(), startup_window_seconds=window)
