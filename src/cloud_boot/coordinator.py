# ----------------------------------------------------------------------------------------------------
# coordinator.py
# ----------------------------------------------------------------------------------------------------

"""Milestone 0 run coordinator: one scenario, one boot, one VM, one deterministic exit code."""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .artifacts import ArtifactError, RunWorkspace, boot_directory
from .assertions import AssertionResult, block_all, run_assertions
from .errors import CloudBootError, ErrorPhase, ExitCode
from .image_store import ImageStore, VerifiedImage
from .models import BootPlan, ImageSpec
from .process import Clock, CommandRunner, ProcessTimeoutError
from .report import (
    REPORT_FILENAME,
    BootResult,
    FailureRecord,
    Outcome,
    RunReportV1,
    decide_outcome,
    image_identity,
    scenario_identity,
    seed_digests,
    write_report,
)
from .scenario import LoadedScenario
from .seed import SeedArtifact, build_seed
from .ssh import AccessIdentity, GuestResult, SshTransport, SshTransportError, create_access_identity
from .vm import DiskArtifact, QemuBackend, StopResult, VmHandle, VmLaunchSpec

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "REQUIRED_TOOLS",
    "GuestSession",
    "HostUnavailableError",
    "ImageResolver",
    "RunCoordinator",
    "RunDependencies",
    "RunInterruptedError",
    "RunResult",
    "RunTimeoutError",
    "UnexpectedRunError",
    "VmBackend",
    "check_host",
    "default_artifact_root",
    "default_cache_dir",
    "production_dependencies",
]

# ----------------------------------------------------------------------------------------------------
REQUIRED_TOOLS = ("qemu-system-x86_64", "qemu-img", "ssh", "ssh-keygen")
KVM_DEVICE = Path("/dev/kvm")
# Fixed budgets outside the scenario timeout, so cleanup still runs after the run times out.
PREPARATION_TIMEOUT_SECONDS = 60.0
COLLECTION_TIMEOUT_SECONDS = 30.0
SHUTDOWN_REQUEST_TIMEOUT_SECONDS = 30.0
STOP_TIMEOUT_SECONDS = 60.0
# `cloud-init status --wait` exits non-zero for error and degraded states; assertions judge those.
CLOUD_INIT_WAIT_SCRIPT = "cloud-init status --wait >/dev/null 2>&1\nexit 0\n"
COLLECTED_LOGS = (
    ("/var/log/cloud-init.log", "cloud-init.log"),
    ("/var/log/cloud-init-output.log", "cloud-init-output.log"),
)

type Progress = Callable[[str, str], None]
type WallClock = Callable[[], datetime]


# ----------------------------------------------------------------------------------------------------
class HostUnavailableError(CloudBootError):
    """A required host command or KVM access is missing."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="host-unavailable",
            phase=ErrorPhase.HOST,
            remediation=(
                "Install qemu-system-x86, qemu-utils, and openssh-client, and give your user "
                "read/write access to /dev/kvm (for example, add it to the kvm group and log in "
                "again)."
            ),
            context=context,
            exit_code=ExitCode.HOST_UNAVAILABLE,
        )


# ----------------------------------------------------------------------------------------------------
class RunTimeoutError(CloudBootError):
    """The run exceeded the scenario's time limit."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="run-timeout",
            phase=ErrorPhase.VM,
            remediation="Raise the scenario's timeout_seconds or inspect the console log.",
            context=context,
            exit_code=ExitCode.TIMEOUT,
        )


# ----------------------------------------------------------------------------------------------------
class RunInterruptedError(CloudBootError):
    """The user interrupted the run; cleanup and the report still completed."""

    def __init__(self) -> None:
        super().__init__(
            "the run was interrupted",
            code="run-interrupted",
            phase=ErrorPhase.INTERNAL,
            exit_code=ExitCode.SIGINT,
        )


# ----------------------------------------------------------------------------------------------------
class UnexpectedRunError(CloudBootError):
    """An unexpected exception, wrapped so the run still reports exit code 10."""

    def __init__(self, error: Exception) -> None:
        super().__init__(
            f"unexpected {type(error).__name__}: {error}",
            code="internal-error",
            phase=ErrorPhase.INTERNAL,
            remediation="Report this as a cloud-boot bug and attach the run directory.",
            exit_code=ExitCode.INTERNAL,
        )


# ----------------------------------------------------------------------------------------------------
class ImageResolver(Protocol):
    """Resolves the scenario's image to a verified local file (`ImageStore`)."""

    def resolve(self, spec: ImageSpec | None = ..., *, deadline: float | None = ...) -> VerifiedImage:
        """Return a verified cached or freshly downloaded image."""
        ...


# ----------------------------------------------------------------------------------------------------
class VmBackend(Protocol):
    """Creates the overlay and starts and stops the guest (`QemuBackend`)."""

    def prepare(
        self, image: VerifiedImage, run_dir: str | Path, *, disk_gib: int, deadline: float
    ) -> DiskArtifact:
        """Create the run's overlay."""
        ...

    def start(self, spec: VmLaunchSpec) -> VmHandle:
        """Launch the guest."""
        ...

    def stop(self, handle: VmHandle, *, deadline: float) -> StopResult:
        """Stop the guest within the deadline."""
        ...


# ----------------------------------------------------------------------------------------------------
class GuestSession(Protocol):
    """Talks to the running guest (`SshTransport`)."""

    def wait_ready(self, *, deadline: float) -> int:
        """Wait until the guest accepts commands."""
        ...

    def execute(self, script: str, args: Sequence[str] = ..., *, timeout: float = ...) -> GuestResult:
        """Run a fixed script in the guest."""
        ...

    def copy_from(self, remote: str, local: str | Path, *, timeout: float = ...) -> GuestResult:
        """Copy a guest file to the host."""
        ...

    def shutdown(self, *, timeout: float = ...) -> None:
        """Ask the guest to power off."""
        ...


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RunDependencies:
    """Every port the coordinator uses; production wiring and test fakes both fill this."""

    preflight: Callable[[], None]
    create_workspace: Callable[[], RunWorkspace]
    image_store: ImageResolver
    create_identity: Callable[[Path, float], AccessIdentity]
    build_seed: Callable[[BootPlan, str, Path, Path], SeedArtifact]
    vm_backend: VmBackend
    connect: Callable[[AccessIdentity, VmHandle], GuestSession]
    clock: Clock = time.monotonic
    wall_clock: WallClock = field(default=lambda: datetime.now(UTC))


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RunResult:
    """What the CLI needs: the exit code, where evidence lives, and what went wrong."""

    exit_code: int
    outcome: Outcome | None
    run_dir: Path | None
    report_path: Path | None
    failure: CloudBootError | None
    assertions: tuple[AssertionResult, ...] = ()


# ----------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class _RunState:
    image: VerifiedImage | None = None
    seed: SeedArtifact | None = None
    stop_reason: str | None = None
    assertions: list[AssertionResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------------------------------
class _DeadlineSession:
    """Caps every guest command at the time left before the run deadline."""

    def __init__(self, session: GuestSession, deadline: float, clock: Clock) -> None:
        self._session = session
        self._deadline = deadline
        self._clock = clock

    def execute(self, script: str, args: Sequence[str] = (), *, timeout: float = 60.0) -> GuestResult:
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            raise ProcessTimeoutError("the run deadline has passed")
        return self._session.execute(script, args, timeout=min(timeout, remaining))


# ----------------------------------------------------------------------------------------------------
def check_host(
    *,
    which: Callable[[str], str | None] = shutil.which,
    kvm_device: Path = KVM_DEVICE,
    access: Callable[[Path, int], bool] = os.access,
) -> None:
    """Check the commands and KVM access a run needs, reporting every problem at once.

    Raises:
        HostUnavailableError: If any command is missing or `/dev/kvm` is not usable.
    """
    problems: list[str] = []
    missing = [tool for tool in REQUIRED_TOOLS if which(tool) is None]
    if missing:
        problems.append(f"missing commands: {', '.join(missing)}")
    if not kvm_device.exists():
        problems.append(f"{kvm_device} does not exist, so KVM acceleration is unavailable")
    elif not access(kvm_device, os.R_OK | os.W_OK):
        problems.append(f"no read/write access to {kvm_device}")
    if problems:
        raise HostUnavailableError(
            "; ".join(problems), context={"missing_commands": missing, "kvm_device": str(kvm_device)}
        )


# ----------------------------------------------------------------------------------------------------
def default_cache_dir(environ: Mapping[str, str]) -> Path:
    """Return `$XDG_CACHE_HOME/cloud-boot`, falling back to `~/.cache/cloud-boot`."""
    return _xdg_base(environ, "XDG_CACHE_HOME", Path(".cache")) / "cloud-boot"


# ----------------------------------------------------------------------------------------------------
def default_artifact_root(environ: Mapping[str, str]) -> Path:
    """Return `$XDG_STATE_HOME/cloud-boot`, falling back to `~/.local/state/cloud-boot`."""
    return _xdg_base(environ, "XDG_STATE_HOME", Path(".local") / "state") / "cloud-boot"


# ----------------------------------------------------------------------------------------------------
def _xdg_base(environ: Mapping[str, str], variable: str, fallback: Path) -> Path:
    # The XDG specification says relative values are invalid and must be ignored.
    value = environ.get(variable, "")
    if value and Path(value).is_absolute():
        return Path(value)
    return Path(environ.get("HOME") or Path.home()) / fallback


# ----------------------------------------------------------------------------------------------------
def production_dependencies(
    *, cache_dir: Path, artifact_root: Path, runner: CommandRunner | None = None
) -> RunDependencies:
    """Wire the real host adapters: image cache, QEMU, SSH, and the run workspace."""
    host = runner if runner is not None else CommandRunner()
    return RunDependencies(
        preflight=check_host,
        create_workspace=lambda: RunWorkspace.create(artifact_root),
        image_store=ImageStore(cache_dir, command_runner=host),
        create_identity=lambda directory, deadline: create_access_identity(
            host, directory, deadline=deadline
        ),
        build_seed=lambda plan, key, scenario_dir, output_dir: build_seed(
            plan, ssh_public_key=key, scenario_dir=scenario_dir, output_dir=output_dir
        ),
        vm_backend=QemuBackend(host),
        connect=lambda identity, handle: SshTransport(
            host, identity, host=handle.ssh_host, port=handle.ssh_port
        ),
    )


# ----------------------------------------------------------------------------------------------------
class RunCoordinator:
    """Sequence one Milestone 0 run and always finish with cleanup and a consistent report.

    Every acquired resource registers its cleanup at once; cleanups run in reverse order from a
    `finally` block. The report is written after cleanup, so its warnings include cleanup
    problems, and its exit code is the one the CLI returns.
    """

    def __init__(self, dependencies: RunDependencies, *, progress: Progress | None = None) -> None:
        self._deps = dependencies
        self._progress = progress if progress is not None else (lambda _phase, _message: None)

    def run(self, loaded: LoadedScenario) -> RunResult:
        """Run the scenario's single boot and return its exit code and evidence location."""
        try:
            self._progress("preflight", "checking host commands and KVM access")
            self._deps.preflight()
            workspace = self._deps.create_workspace()
        except CloudBootError as error:
            return RunResult(error.exit_code, "error", None, None, error)
        self._progress("workspace", f"writing evidence to {workspace.path}")
        started_at = self._deps.wall_clock()
        started = self._deps.clock()
        state = _RunState()
        failure: CloudBootError | None = None
        try:
            with contextlib.ExitStack() as cleanup:
                self._execute(loaded, workspace, state, cleanup)
        except CloudBootError as error:
            failure = error
        except KeyboardInterrupt:
            failure = RunInterruptedError()
        except Exception as error:
            failure = UnexpectedRunError(error)
        state.warnings.extend(workspace.cleanup_transient())
        return self._finish(loaded, workspace, state, failure, started_at, started)

    def _execute(
        self,
        loaded: LoadedScenario,
        workspace: RunWorkspace,
        state: _RunState,
        cleanup: contextlib.ExitStack,
    ) -> None:
        scenario = loaded.scenario
        plan = loaded.boots[0]
        deps = self._deps

        self._progress("image", f"resolving {scenario.image.release} {scenario.image.architecture}")
        image = deps.image_store.resolve(scenario.image)
        state.image = image

        self._progress("prepare", "creating the run key, seed media, and overlay")
        identity = deps.create_identity(
            workspace.transient_dir, self._after(PREPARATION_TIMEOUT_SECONDS)
        )
        cleanup.callback(identity.destroy)
        seed = deps.build_seed(
            plan, identity.public_key, loaded.source_path.parent, workspace.transient_dir
        )
        state.seed = seed
        disk = deps.vm_backend.prepare(
            image,
            workspace.transient_dir,
            disk_gib=scenario.resources.disk_gib,
            deadline=self._after(PREPARATION_TIMEOUT_SECONDS),
        )

        boot_dir = boot_directory(1, plan.name)
        console = workspace.open_new(f"{boot_dir}/console.log")
        # Registered before the VM starts, so the console closes only after QEMU has stopped.
        cleanup.callback(console.close)
        run_deadline = deps.clock() + scenario.timeout_seconds
        self._progress("boot", f"starting QEMU ({scenario.timeout_seconds} s limit)")
        handle = deps.vm_backend.start(
            VmLaunchSpec(
                disk=disk,
                seed_image=seed.path,
                resources=scenario.resources,
                console=console,
                name=scenario.name,
            )
        )
        session: dict[str, GuestSession] = {}
        cleanup.callback(self._stop_guest, handle, session, workspace, boot_dir, state)
        workspace.write_text_atomic(
            f"{boot_dir}/qemu-command.json", json.dumps(list(handle.argv), indent=2) + "\n"
        )

        guest = deps.connect(identity, handle)
        self._progress("ssh", f"waiting for SSH on {handle.ssh_host}:{handle.ssh_port}")
        try:
            guest.wait_ready(deadline=run_deadline)
        except SshTransportError as error:
            state.assertions = block_all(
                plan.assertions, boot=plan.name, reason=f"SSH never became ready: {error.message}"
            )
            raise
        session["ready"] = guest
        capped = _DeadlineSession(guest, run_deadline, deps.clock)

        self._progress("cloud-init", "waiting for cloud-init to finish")
        try:
            capped.execute(CLOUD_INIT_WAIT_SCRIPT, (), timeout=float(scenario.timeout_seconds))
        except ProcessTimeoutError as error:
            raise RunTimeoutError(
                "the run reached its time limit while waiting for cloud-init",
                context={"timeout_seconds": scenario.timeout_seconds},
            ) from error
        except SshTransportError as error:
            state.assertions = block_all(
                plan.assertions, boot=plan.name, reason=f"SSH transport unavailable: {error.message}"
            )
            raise

        self._progress("assertions", f"evaluating {len(plan.assertions)} assertions")
        state.assertions = run_assertions(plan.assertions, capped, boot=plan.name, clock=deps.clock)
        if deps.clock() > run_deadline:
            raise RunTimeoutError(
                "the run exceeded its time limit during assertions",
                context={"timeout_seconds": scenario.timeout_seconds},
            )

    def _stop_guest(
        self,
        handle: VmHandle,
        session: Mapping[str, GuestSession],
        workspace: RunWorkspace,
        boot_dir: str,
        state: _RunState,
    ) -> None:
        guest = session.get("ready")
        if guest is not None:
            self._collect(guest, workspace, boot_dir, state)
            self._progress("shutdown", "asking the guest to power off")
            try:
                guest.shutdown(timeout=SHUTDOWN_REQUEST_TIMEOUT_SECONDS)
            except CloudBootError as error:
                state.warnings.append(f"guest shutdown request failed: {error.message}")
        self._progress("shutdown", "stopping QEMU")
        try:
            stopped = self._deps.vm_backend.stop(handle, deadline=self._after(STOP_TIMEOUT_SECONDS))
        except CloudBootError as error:
            state.warnings.append(f"QEMU did not stop cleanly: {error.message}")
            return
        state.stop_reason = str(stopped.reason)
        for failure in stopped.result.sink_failures:
            state.warnings.append(f"console log {failure.stream} write failed: {failure.error}")
        try:
            workspace.write_bytes(f"{boot_dir}/qemu.stderr.log", stopped.result.stderr)
        except ArtifactError as error:
            state.warnings.append(f"could not save QEMU stderr: {error.message}")

    def _collect(
        self, guest: GuestSession, workspace: RunWorkspace, boot_dir: str, state: _RunState
    ) -> None:
        self._progress("collect", "copying cloud-init logs")
        for remote, name in COLLECTED_LOGS:
            try:
                result = guest.copy_from(
                    remote,
                    workspace.artifact_path(f"{boot_dir}/{name}"),
                    timeout=COLLECTION_TIMEOUT_SECONDS,
                )
            except CloudBootError as error:
                state.warnings.append(f"could not collect {remote}: {error.message}")
                continue
            if result.exit_status != 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip()
                state.warnings.append(f"could not collect {remote}: {detail or result.exit_status}")

    def _finish(
        self,
        loaded: LoadedScenario,
        workspace: RunWorkspace,
        state: _RunState,
        failure: CloudBootError | None,
        started_at: datetime,
        started: float,
    ) -> RunResult:
        plan = loaded.boots[0]
        outcome, exit_code = decide_outcome(failure=failure, assertions=state.assertions)
        artifacts = sorted(
            workspace.relative(path)
            for path in workspace.path.rglob("*")
            if path.is_file() and not path.is_symlink() and path.name != REPORT_FILENAME
        )
        report = RunReportV1(
            run_id=workspace.run_id,
            scenario=scenario_identity(loaded),
            image=image_identity(state.image) if state.image is not None else None,
            started_at=started_at,
            finished_at=self._deps.wall_clock(),
            duration_seconds=max(0.0, self._deps.clock() - started),
            outcome=outcome,
            exit_code=exit_code,
            boots=[
                BootResult(
                    name=plan.name,
                    index=1,
                    instance_id=plan.seed.instance_id,
                    seed_digests=seed_digests(state.seed) if state.seed is not None else {},
                    stop_reason=state.stop_reason,
                    assertions=state.assertions,
                )
            ],
            artifacts=artifacts,
            warnings=state.warnings,
            failure=FailureRecord.from_error(failure) if failure is not None else None,
        )
        self._progress("report", f"{outcome} (exit code {exit_code})")
        try:
            report_path = write_report(workspace, report)
        except ArtifactError as error:
            # Never report success without persisted evidence.
            return RunResult(
                error.exit_code, "error", workspace.path, None, error, tuple(state.assertions)
            )
        return RunResult(
            exit_code, outcome, workspace.path, report_path, failure, tuple(state.assertions)
        )

    def _after(self, seconds: float) -> float:
        return self._deps.clock() + seconds
