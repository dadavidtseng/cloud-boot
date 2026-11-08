# ----------------------------------------------------------------------------------------------------
# vm.py
# ----------------------------------------------------------------------------------------------------

"""Launch and stop one QEMU/KVM guest on a qcow2 overlay of a verified base image."""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import math
import re
import signal
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from .errors import CloudBootError, ErrorPhase, ExitCode
from .image_store import VerifiedImage
from .models import ResourceSpec
from .process import (
    Clock,
    CommandResult,
    CommandRunner,
    CommandSpec,
    ManagedProcess,
    OutputSink,
    OutputSinks,
)

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "CONSOLE_CAPTURE_LIMIT_BYTES",
    "DEFAULT_QEMU_BINARY",
    "DEFAULT_QEMU_IMG_BINARY",
    "MAX_LAUNCH_ATTEMPTS",
    "SSH_HOST",
    "DiskArtifact",
    "OverlayError",
    "QemuBackend",
    "StopReason",
    "StopResult",
    "VmHandle",
    "VmLaunchError",
    "VmLaunchSpec",
    "allocate_loopback_port",
    "build_qemu_argv",
]

# ----------------------------------------------------------------------------------------------------
DEFAULT_QEMU_BINARY = "qemu-system-x86_64"
DEFAULT_QEMU_IMG_BINARY = "qemu-img"
SSH_HOST = "127.0.0.1"
MAX_LAUNCH_ATTEMPTS = 3
DEFAULT_STARTUP_WINDOW_SECONDS = 2.0
DEFAULT_TERMINATION_SECONDS = 10.0
# The full console goes to the caller's sink; memory keeps only a tail for error reports.
CONSOLE_CAPTURE_LIMIT_BYTES = 64 * 1024
_POLL_INTERVAL_SECONDS = 0.05
_STDERR_CONTEXT_BYTES = 2048
# QEMU prints this when the SSH forwarding port was taken between allocation and launch.
_PORT_CONFLICT_MARKER = b"Could not set up host forwarding rule"
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

type ConsoleSink = OutputSink | Callable[[bytes], object]
type PortAllocator = Callable[[], int]
type Sleeper = Callable[[float], None]


# ----------------------------------------------------------------------------------------------------
class OverlayError(CloudBootError):
    """Expected failure while creating the run's qcow2 overlay."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="overlay-failed",
            phase=ErrorPhase.PREPARATION,
            remediation="Check that qemu-img is installed and the run directory is writable.",
            context=context,
            exit_code=ExitCode.PREPARATION,
        )


# ----------------------------------------------------------------------------------------------------
class VmLaunchError(CloudBootError):
    """Expected failure while starting QEMU."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="vm-launch-failed",
            phase=ErrorPhase.VM,
            remediation="Inspect QEMU's stderr in the error context; KVM needs /dev/kvm access.",
            context=context,
            exit_code=ExitCode.VM,
        )


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DiskArtifact:
    """A per-run qcow2 overlay backed read-only by the verified base image."""

    path: Path
    backing_image: Path
    size_gib: int


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VmLaunchSpec:
    """Inputs for one QEMU launch.

    Attributes:
        disk: The run's overlay; the base image is never passed to QEMU directly.
        seed_image: NoCloud seed media, attached read-only.
        resources: vCPU, memory, and disk bounds.
        console: Sink receiving every byte of the serial console.
        name: QEMU guest name; restricted so it cannot inject QEMU option separators.
        network_mode: Only `nat` (QEMU user networking) is supported by this backend.
    """

    disk: DiskArtifact
    seed_image: Path
    resources: ResourceSpec
    console: ConsoleSink
    name: str = "cloud-boot"
    network_mode: Literal["nat"] = "nat"

    def __post_init__(self) -> None:
        if self.network_mode != "nat":
            raise ValueError(f"unsupported network mode: {self.network_mode!r}")
        if not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError("VM name must be 1-64 lowercase letters, digits, '.', '_', or '-'")


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VmHandle:
    """A running guest and its loopback SSH endpoint."""

    process: ManagedProcess
    argv: tuple[str, ...]
    ssh_host: str
    ssh_port: int
    launch_attempts: int


# ----------------------------------------------------------------------------------------------------
class StopReason(StrEnum):
    """How QEMU ended during `QemuBackend.stop`."""

    EXITED = "exited"
    TERMINATED = "terminated"
    KILLED = "killed"


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StopResult:
    """The outcome of stopping a guest, with QEMU's captured stderr and console tail."""

    reason: StopReason
    result: CommandResult


# ----------------------------------------------------------------------------------------------------
def allocate_loopback_port() -> int:
    """Return a TCP port on 127.0.0.1 that was free a moment ago.

    The probe socket closes before QEMU binds the port, so another process can still take it in
    between; `QemuBackend.start` retries when QEMU reports that conflict.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((SSH_HOST, 0))
        return int(probe.getsockname()[1])


# ----------------------------------------------------------------------------------------------------
def build_qemu_argv(
    spec: VmLaunchSpec, ssh_port: int, *, qemu_binary: str = DEFAULT_QEMU_BINARY
) -> tuple[str, ...]:
    """Assemble the deterministic QEMU argument array for one launch.

    The guest gets KVM with the host CPU model, virtio disks, the serial console on stdout, no
    display or monitor, and QEMU user networking whose only host forward is loopback SSH.

    Args:
        spec: Disk, seed, resources, and guest name for the launch.
        ssh_port: Host port on 127.0.0.1 forwarded to guest port 22.
        qemu_binary: The QEMU system emulator to execute.

    Returns:
        The full argument array, starting with `qemu_binary`.

    Raises:
        ValueError: If `ssh_port` is not a valid TCP port.
    """
    if not 1 <= ssh_port <= 65535:
        raise ValueError("ssh_port must be between 1 and 65535")
    resources = spec.resources
    return (
        qemu_binary,
        "-name",
        spec.name,
        "-nodefaults",
        "-no-user-config",
        "-machine",
        "q35,accel=kvm",
        "-cpu",
        "host",
        "-smp",
        str(resources.cpus),
        "-m",
        f"{resources.memory_mib}M",
        "-display",
        "none",
        "-serial",
        "stdio",
        "-drive",
        f"file={_option_path(spec.disk.path)},if=virtio,format=qcow2",
        "-drive",
        f"file={_option_path(spec.seed_image)},if=virtio,format=raw,readonly=on",
        "-netdev",
        f"user,id=net0,hostfwd=tcp:{SSH_HOST}:{ssh_port}-:22",
        "-device",
        "virtio-net-pci,netdev=net0",
    )


# ----------------------------------------------------------------------------------------------------
def _option_path(path: Path) -> str:
    # QEMU separates option values with ","; a literal comma must be written as ",,".
    return str(path.absolute()).replace(",", ",,")


# ----------------------------------------------------------------------------------------------------
def _tail(data: bytes) -> str:
    return data[-_STDERR_CONTEXT_BYTES:].decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------------------------------
class QemuBackend:
    """Process-only QEMU backend: overlay creation, launch with port retry, and bounded stop.

    Pass the same monotonic clock and sleeper to this backend and its `CommandRunner` so startup
    observation and shutdown deadlines are measured on one timeline.
    """

    def __init__(
        self,
        runner: CommandRunner,
        *,
        qemu_binary: str = DEFAULT_QEMU_BINARY,
        qemu_img_binary: str = DEFAULT_QEMU_IMG_BINARY,
        port_allocator: PortAllocator = allocate_loopback_port,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
        startup_window_seconds: float = DEFAULT_STARTUP_WINDOW_SECONDS,
        termination_seconds: float = DEFAULT_TERMINATION_SECONDS,
    ) -> None:
        for value in (startup_window_seconds, termination_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("startup and termination windows must be positive and finite")
        self._runner = runner
        self._qemu_binary = qemu_binary
        self._qemu_img_binary = qemu_img_binary
        self._port_allocator = port_allocator
        self._clock = clock
        self._sleeper = sleeper
        self._startup_window_seconds = startup_window_seconds
        self._termination_seconds = termination_seconds

    def prepare(
        self, image: VerifiedImage, run_dir: str | Path, *, disk_gib: int, deadline: float
    ) -> DiskArtifact:
        """Create `overlay.qcow2` in `run_dir`, backed read-only by the verified image.

        The backing path is absolute because qemu-img resolves a relative one against the
        overlay's directory. An existing overlay is never overwritten.

        Args:
            image: The checksum-verified base image.
            run_dir: The run's directory, which must already exist.
            disk_gib: Virtual size of the guest disk.
            deadline: Monotonic deadline for qemu-img.

        Returns:
            The overlay path, its backing image, and its virtual size.

        Raises:
            OverlayError: If the overlay already exists or qemu-img fails.
        """
        if isinstance(disk_gib, bool) or not isinstance(disk_gib, int) or disk_gib <= 0:
            raise ValueError("disk_gib must be a positive integer")
        overlay = Path(run_dir) / "overlay.qcow2"
        if overlay.exists() or overlay.is_symlink():
            raise OverlayError("overlay already exists", context={"path": str(overlay)})
        backing = image.path.absolute()
        argv = (
            self._qemu_img_binary,
            "create",
            "-q",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            str(backing),
            str(overlay),
            f"{disk_gib}G",
        )
        result = self._runner.run(CommandSpec(argv, name="qemu-img"), deadline=deadline)
        if not result.succeeded:
            overlay.unlink(missing_ok=True)
            raise OverlayError(
                "qemu-img could not create the overlay",
                context={"returncode": result.returncode, "stderr": _tail(result.stderr)},
            )
        return DiskArtifact(path=overlay, backing_image=backing, size_gib=disk_gib)

    def start(self, spec: VmLaunchSpec) -> VmHandle:
        """Launch QEMU and return once it survives the startup window.

        The console streams to `spec.console`; only a bounded tail stays in memory. A forwarding
        port conflict is retried with a new port up to `MAX_LAUNCH_ATTEMPTS` times; any other
        early exit fails immediately.

        Raises:
            VmLaunchError: If QEMU exits during the startup window.
            ProcessStartError: If the QEMU binary cannot be executed.
        """
        attempt = 0
        while True:
            attempt += 1
            port = self._port_allocator()
            argv = build_qemu_argv(spec, port, qemu_binary=self._qemu_binary)
            process = self._runner.start(
                CommandSpec(argv, name="qemu", capture_limit_bytes=CONSOLE_CAPTURE_LIMIT_BYTES),
                OutputSinks(stdout=spec.console),
            )
            try:
                early_exit = self._await_startup(process)
            except BaseException:
                with contextlib.suppress(Exception):
                    process.stop(timeout=self._termination_seconds)
                raise
            if early_exit is None:
                return VmHandle(
                    process=process,
                    argv=argv,
                    ssh_host=SSH_HOST,
                    ssh_port=port,
                    launch_attempts=attempt,
                )
            port_conflict = _PORT_CONFLICT_MARKER in early_exit.stderr
            if port_conflict and attempt < MAX_LAUNCH_ATTEMPTS:
                continue
            message = (
                f"QEMU could not bind an SSH forwarding port after {attempt} attempts"
                if port_conflict
                else "QEMU exited during startup"
            )
            raise VmLaunchError(
                message,
                context={
                    "returncode": early_exit.returncode,
                    "stderr": _tail(early_exit.stderr),
                    "attempts": attempt,
                    "port_conflict": port_conflict,
                },
            )

    def stop(self, handle: VmHandle, *, deadline: float) -> StopResult:
        """Wait for QEMU to exit, then escalate to SIGTERM and SIGKILL within `deadline`.

        The caller requests guest shutdown first (for example over SSH). The last
        `termination_seconds` before `deadline` are reserved for process-group signals, so the
        whole stop never outlasts the caller's deadline.

        Raises:
            ProcessTimeoutError: If QEMU survives SIGKILL until `deadline`.
        """
        if not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        process = handle.process
        signal_from = max(self._clock(), deadline - self._termination_seconds)
        while process.poll() is None:
            remaining = signal_from - self._clock()
            if remaining <= 0:
                break
            self._sleeper(min(_POLL_INTERVAL_SECONDS, remaining))
        if process.poll() is not None:
            return StopResult(StopReason.EXITED, process.wait(deadline=deadline))
        result = process.stop(deadline=deadline)
        killed = result.returncode == -signal.SIGKILL
        return StopResult(StopReason.KILLED if killed else StopReason.TERMINATED, result)

    def _await_startup(self, process: ManagedProcess) -> CommandResult | None:
        end = self._clock() + self._startup_window_seconds
        while True:
            if process.poll() is not None:
                return process.wait(timeout=self._termination_seconds)
            remaining = end - self._clock()
            if remaining <= 0:
                return None
            self._sleeper(min(_POLL_INTERVAL_SECONDS, remaining))
