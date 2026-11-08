"""Contract tests for the Milestone 0 run coordinator and the `run` CLI."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cloud_boot import __version__
from cloud_boot.artifacts import ArtifactError, RunWorkspace
from cloud_boot.assertions import (
    CLOUD_INIT_STATUS_SCRIPT,
    FILE_SCRIPT,
    PACKAGE_SCRIPT,
    AssertionResult,
)
from cloud_boot.cli import main
from cloud_boot.coordinator import (
    CLOUD_INIT_WAIT_SCRIPT,
    HostUnavailableError,
    RunCoordinator,
    RunDependencies,
    RunResult,
    check_host,
    default_artifact_root,
    default_cache_dir,
)
from cloud_boot.errors import ExitCode
from cloud_boot.image_store import ImageStoreError, VerifiedImage
from cloud_boot.models import BootPlan, ImageSpec
from cloud_boot.process import CommandResult, ProcessTimeoutError
from cloud_boot.scenario import LoadedScenario, load_scenario
from cloud_boot.seed import SeedArtifact, SeedBuildError, SeedFile
from cloud_boot.ssh import AccessIdentity, AccessKeyError, GuestResult, SshTransportError
from cloud_boot.vm import (
    DiskArtifact,
    OverlayError,
    StopReason,
    StopResult,
    VmHandle,
    VmLaunchError,
    VmLaunchSpec,
)

STARTED = datetime(2026, 11, 8, 2, 0, 0, tzinfo=UTC)
SCENARIO = """\
scenario_version: 1
name: basic-boot
seed:
  instance_id: cloud-boot-basic-001
  user_data:
    inline: |
      #cloud-config
      {}
assertions:
  - id: cloud-init-done
    type: cloud-init-status
  - id: marker-created
    type: file
    path: /var/tmp/boot-marker
    content_contains: boot-ok
  - id: ssh-installed
    type: package-installed
    name: openssh-server
"""
SUCCESS_EVENTS = [
    "preflight",
    "workspace",
    "image",
    "identity",
    "seed",
    "prepare",
    "start",
    "connect",
    "wait_ready",
    "cloud-init",
    "assert:cloud-init",
    "assert:file",
    "assert:package",
    "copy:cloud-init.log",
    "copy:cloud-init-output.log",
    "shutdown",
    "stop",
    "destroy",
]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeProcess:
    sink_failures = ()


class RecordingIdentity(AccessIdentity):
    """Records when the coordinator destroys the run key."""

    def destroy(self) -> None:
        self.events.append("destroy")  # type: ignore[attr-defined]
        super().destroy()


def guest(exit_status: int = 0, stdout: bytes = b"") -> GuestResult:
    return GuestResult(exit_status=exit_status, stdout=stdout, stderr=b"", duration_seconds=0.1)


class Harness:
    """Fake image store, QEMU backend, and guest session that log calls and fail on demand."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        fail: Mapping[str, BaseException] | None = None,
        file_stdout: bytes = b"file 7\nboot-ok",
        assertion_seconds: float = 0.0,
    ) -> None:
        self.tmp_path = tmp_path
        self.fail = dict(fail or {})
        self.file_stdout = file_stdout
        self.assertion_seconds = assertion_seconds
        self.events: list[str] = []
        self.clock = FakeClock()
        self.workspace: RunWorkspace | None = None
        self.launch: VmLaunchSpec | None = None

    def _step(self, name: str) -> None:
        self.events.append(name)
        if name in self.fail:
            raise self.fail[name]

    def preflight(self) -> None:
        self._step("preflight")

    def create_workspace(self) -> RunWorkspace:
        self._step("workspace")
        self.workspace = RunWorkspace.create(
            self.tmp_path / "state", wall_clock=lambda: STARTED, token_factory=lambda: "a1b2c3d4"
        )
        return self.workspace

    def resolve(
        self, spec: ImageSpec | None = None, *, deadline: float | None = None
    ) -> VerifiedImage:
        self._step("image")
        return VerifiedImage(
            path=self.tmp_path / "base.img",
            digest="0" * 64,
            size_bytes=1,
            source_url="https://images.test/noble-server-cloudimg-amd64.img",
            retrieved_at=0.0,
        )

    def create_identity(self, directory: Path, deadline: float) -> AccessIdentity:
        self._step("identity")
        key = directory / "id_ed25519"
        key.write_text("PRIVATE KEY")
        Path(f"{key}.pub").write_text("ssh-ed25519 AAAA test")
        identity = RecordingIdentity(key, "ssh-ed25519 AAAA test", directory / "known_hosts")
        object.__setattr__(identity, "events", self.events)
        return identity

    def build_seed(
        self, plan: BootPlan, public_key: str, scenario_dir: Path, output_dir: Path
    ) -> SeedArtifact:
        self._step("seed")
        path = output_dir / "seed.iso"
        path.write_bytes(b"iso")
        return SeedArtifact(
            path, "CIDATA", plan.seed.instance_id, (SeedFile("meta-data", "a" * 64, 1),)
        )

    def prepare(
        self, image: VerifiedImage, run_dir: str | Path, *, disk_gib: int, deadline: float
    ) -> DiskArtifact:
        self._step("prepare")
        overlay = Path(run_dir) / "overlay.qcow2"
        overlay.write_bytes(b"qcow2")
        return DiskArtifact(overlay, image.path, disk_gib)

    def start(self, spec: VmLaunchSpec) -> VmHandle:
        self._step("start")
        self.launch = spec
        spec.console.write(b"[    0.000000] Linux version 6.8\n")  # type: ignore[union-attr]
        return VmHandle(
            process=FakeProcess(),  # type: ignore[arg-type]
            argv=("qemu-system-x86_64", "-name", spec.name),
            ssh_host="127.0.0.1",
            ssh_port=40001,
            launch_attempts=1,
        )

    def stop(self, handle: VmHandle, *, deadline: float) -> StopResult:
        self._step("stop")
        result = CommandResult(
            argv=handle.argv, returncode=0, stdout=b"", stderr=b"qemu stderr", duration_seconds=1.0
        )
        return StopResult(StopReason.EXITED, result)

    def connect(self, identity: AccessIdentity, handle: VmHandle) -> Harness:
        self.events.append("connect")
        return self

    def wait_ready(self, *, deadline: float) -> int:
        self._step("wait_ready")
        return 1

    def execute(self, script: str, args: Sequence[str] = (), *, timeout: float = 60.0) -> GuestResult:
        if script == CLOUD_INIT_WAIT_SCRIPT:
            self._step("cloud-init")
            return guest()
        name = {
            CLOUD_INIT_STATUS_SCRIPT: "assert:cloud-init",
            FILE_SCRIPT: "assert:file",
            PACKAGE_SCRIPT: "assert:package",
        }[script]
        self._step(name)
        self.clock.now += self.assertion_seconds
        if name == "assert:cloud-init":
            return guest(stdout=json.dumps({"status": "done", "extended_status": "done"}).encode())
        if name == "assert:file":
            return guest(stdout=self.file_stdout)
        return guest(stdout=b"installed\t1:9.6p1-3ubuntu13.20\n")

    def copy_from(self, remote: str, local: str | Path, *, timeout: float = 60.0) -> GuestResult:
        self._step(f"copy:{Path(remote).name}")
        Path(local).write_bytes(b"log line\n")
        return guest()

    def shutdown(self, *, timeout: float = 30.0) -> None:
        self._step("shutdown")

    def dependencies(self) -> RunDependencies:
        return RunDependencies(
            preflight=self.preflight,
            create_workspace=self.create_workspace,
            image_store=self,
            create_identity=self.create_identity,
            build_seed=self.build_seed,
            vm_backend=self,
            connect=self.connect,
            clock=self.clock,
            wall_clock=lambda: STARTED,
        )

    def run(self, loaded: LoadedScenario) -> RunResult:
        return RunCoordinator(self.dependencies()).run(loaded)

    def report(self) -> dict[str, Any]:
        assert self.workspace is not None
        document: dict[str, Any] = json.loads((self.workspace.path / "report.json").read_text())
        return document


@pytest.fixture
def scenario(tmp_path: Path) -> LoadedScenario:
    path = tmp_path / "basic-boot.yaml"
    path.write_text(SCENARIO)
    return load_scenario(path)


def test_successful_run_exits_0_and_cleans_up_in_reverse_order(
    tmp_path: Path, scenario: LoadedScenario
) -> None:
    harness = Harness(tmp_path)

    result = harness.run(scenario)

    assert harness.events == SUCCESS_EVENTS
    report = harness.report()
    assert (result.exit_code, result.outcome) == (0, "passed")
    assert report["exit_code"] == result.exit_code
    assert report["outcome"] == "passed"
    assert report["boots"][0]["stop_reason"] == "exited"
    assert [entry["status"] for entry in report["boots"][0]["assertions"]] == ["passed"] * 3
    assert report["artifacts"] == [
        "boots/001-boot/cloud-init-output.log",
        "boots/001-boot/cloud-init.log",
        "boots/001-boot/console.log",
        "boots/001-boot/qemu-command.json",
        "boots/001-boot/qemu.stderr.log",
    ]
    assert harness.workspace is not None
    assert result.run_dir == harness.workspace.path
    assert not harness.workspace.transient_dir.exists()
    assert harness.launch is not None
    assert harness.launch.name == "basic-boot"
    assert harness.launch.resources == scenario.scenario.resources


@pytest.mark.parametrize(
    ("step", "error", "exit_code", "vm_stopped", "report_written"),
    [
        ("preflight", HostUnavailableError("missing commands: qemu-img"), 3, False, False),
        ("workspace", ArtifactError("cannot create the run directory"), 9, False, False),
        ("image", ImageStoreError("checksum mismatch"), 4, False, True),
        ("identity", AccessKeyError("ssh-keygen failed"), 4, False, True),
        ("seed", SeedBuildError("payload missing"), 4, False, True),
        ("prepare", OverlayError("qemu-img failed"), 4, False, True),
        ("start", VmLaunchError("QEMU exited during startup"), 5, False, True),
        ("wait_ready", SshTransportError("guest SSH did not become ready"), 6, True, True),
        ("cloud-init", ProcessTimeoutError(), 8, True, True),
        ("prepare", RuntimeError("unexpected bug"), 10, False, True),
        ("wait_ready", KeyboardInterrupt(), 130, True, True),
    ],
)
def test_failures_map_to_exact_exit_codes_and_always_clean_up(
    tmp_path: Path,
    scenario: LoadedScenario,
    step: str,
    error: BaseException,
    exit_code: int,
    vm_stopped: bool,
    report_written: bool,
) -> None:
    harness = Harness(tmp_path, fail={step: error})

    result = harness.run(scenario)

    assert result.exit_code == exit_code
    assert ("stop" in harness.events) is vm_stopped
    if vm_stopped:
        assert harness.events.index("stop") < harness.events.index("destroy")
    # An image or preparation failure never boots the guest.
    if harness.events.index(step) < SUCCESS_EVENTS.index("start"):
        assert "start" not in harness.events
    assert (result.report_path is not None) is report_written
    if report_written:
        report = harness.report()
        assert report["exit_code"] == exit_code
        assert report["outcome"] == "error"
        assert harness.workspace is not None
        assert not harness.workspace.transient_dir.exists()
    if "identity" in harness.events and step != "identity":
        assert "destroy" in harness.events


def test_ssh_failure_blocks_every_assertion_and_skips_guest_requests(
    tmp_path: Path, scenario: LoadedScenario
) -> None:
    harness = Harness(tmp_path, fail={"wait_ready": SshTransportError("unreachable")})

    harness.run(scenario)

    statuses = [entry["status"] for entry in harness.report()["boots"][0]["assertions"]]
    assert statuses == ["blocked"] * 3
    assert "shutdown" not in harness.events
    assert not any(event.startswith("copy:") for event in harness.events)


def test_assertion_failure_is_exit_7_and_still_collects_and_shuts_down(
    tmp_path: Path, scenario: LoadedScenario
) -> None:
    harness = Harness(tmp_path, file_stdout=b"missing\n")

    result = harness.run(scenario)

    assert (result.exit_code, result.outcome) == (7, "failed")
    assert harness.report()["exit_code"] == 7
    assert [entry.status for entry in result.assertions] == ["passed", "failed", "passed"]
    assert harness.events[-5:] == SUCCESS_EVENTS[-5:]


def test_cloud_init_timeout_still_collects_diagnostics(
    tmp_path: Path, scenario: LoadedScenario
) -> None:
    harness = Harness(tmp_path, fail={"cloud-init": ProcessTimeoutError()})

    result = harness.run(scenario)

    assert result.exit_code == ExitCode.TIMEOUT
    assert result.failure is not None
    assert "waiting for cloud-init" in result.failure.message
    assert harness.events[-5:] == SUCCESS_EVENTS[-5:]


def test_run_deadline_bounds_the_assertions(tmp_path: Path, scenario: LoadedScenario) -> None:
    # Each assertion takes 200 s against the scenario's default 300 s limit.
    harness = Harness(tmp_path, assertion_seconds=200.0)

    result = harness.run(scenario)

    assert result.exit_code == ExitCode.TIMEOUT
    report = harness.report()
    assert report["exit_code"] == 8
    statuses = [entry["status"] for entry in report["boots"][0]["assertions"]]
    assert statuses[0] == "passed"
    assert "assert:package" not in harness.events


def test_report_write_failure_is_exit_9_and_never_success(
    tmp_path: Path, scenario: LoadedScenario
) -> None:
    harness = Harness(tmp_path)
    create = harness.create_workspace

    def workspace_with_blocked_report() -> RunWorkspace:
        workspace = create()
        (workspace.path / "report.json").mkdir()
        return workspace

    harness.create_workspace = workspace_with_blocked_report  # type: ignore[method-assign]

    result = harness.run(scenario)

    assert result.exit_code == ExitCode.REPORT
    assert result.report_path is None
    assert isinstance(result.failure, ArtifactError)
    assert "stop" in harness.events
    assert "destroy" in harness.events


def test_cleanup_problems_become_warnings_without_changing_the_exit_code(
    tmp_path: Path, scenario: LoadedScenario
) -> None:
    harness = Harness(
        tmp_path,
        fail={
            "copy:cloud-init-output.log": SshTransportError("connection reset"),
            "stop": ProcessTimeoutError("process did not terminate after SIGTERM and SIGKILL"),
        },
    )

    result = harness.run(scenario)

    assert result.exit_code == 0
    warnings = harness.report()["warnings"]
    assert any("cloud-init-output.log" in warning for warning in warnings)
    assert any("QEMU did not stop cleanly" in warning for warning in warnings)
    assert "destroy" in harness.events


def test_check_host_reports_every_problem_at_once(tmp_path: Path) -> None:
    present = {"qemu-system-x86_64", "ssh-keygen"}

    with pytest.raises(HostUnavailableError) as raised:
        check_host(
            which=lambda tool: f"/usr/bin/{tool}" if tool in present else None,
            kvm_device=tmp_path / "kvm",
        )

    assert raised.value.exit_code == ExitCode.HOST_UNAVAILABLE
    assert "missing commands: qemu-img, ssh" in raised.value.message
    assert "does not exist" in raised.value.message

    (tmp_path / "kvm").touch()
    with pytest.raises(HostUnavailableError, match="no read/write access"):
        check_host(
            which=lambda tool: f"/usr/bin/{tool}",
            kvm_device=tmp_path / "kvm",
            access=lambda _path, _mode: False,
        )
    check_host(which=lambda tool: f"/usr/bin/{tool}", kvm_device=tmp_path / "kvm")


def test_default_directories_follow_xdg_and_ignore_relative_values() -> None:
    absolute = {"HOME": "/home/u", "XDG_CACHE_HOME": "/xdg/cache", "XDG_STATE_HOME": "/xdg/state"}
    relative = {"HOME": "/home/u", "XDG_CACHE_HOME": "cache", "XDG_STATE_HOME": ""}

    assert default_cache_dir(absolute) == Path("/xdg/cache/cloud-boot")
    assert default_artifact_root(absolute) == Path("/xdg/state/cloud-boot")
    assert default_cache_dir(relative) == Path("/home/u/.cache/cloud-boot")
    assert default_artifact_root(relative) == Path("/home/u/.local/state/cloud-boot")


def run_cli(argv: list[str], **kwargs: Any) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = main(argv, stdout=stdout, stderr=stderr, environ={"HOME": "/nonexistent"}, **kwargs)
    return code, stdout.getvalue(), stderr.getvalue()


def test_cli_help_and_version_exit_0() -> None:
    code, stdout, _ = run_cli(["--help"])
    assert code == 0
    assert stdout.startswith("usage: cloud-boot")

    code, stdout, _ = run_cli(["--version"])
    assert (code, stdout) == (0, f"cloud-boot {__version__}\n")


@pytest.mark.parametrize(
    "argv",
    [[], ["run"], ["doctor"], ["compare", "a", "b"], ["run", "s.yaml", "--cleanup"]],
)
def test_cli_rejects_unsupported_usage_with_exit_2(argv: list[str]) -> None:
    code, stdout, stderr = run_cli(argv)

    assert code == ExitCode.VALIDATION
    assert stdout == ""
    assert "error:" in stderr


def test_cli_reports_scenario_validation_errors(tmp_path: Path) -> None:
    code, stdout, stderr = run_cli(["run", str(tmp_path / "missing.yaml")])

    assert code == ExitCode.VALIDATION
    assert stdout == ""
    assert stderr.startswith("ERROR validation:")


def test_cli_prints_only_the_run_directory_on_stdout(tmp_path: Path) -> None:
    scenario_file = tmp_path / "basic-boot.yaml"
    scenario_file.write_text(SCENARIO)
    failed = AssertionResult(
        id="marker-created",
        type="file",
        boot="boot",
        required=True,
        status="failed",
        expected={},
        observed={},
        reason="/var/tmp/boot-marker does not exist",
        duration_seconds=0.1,
        command=None,
    )

    class FakeRunner:
        def __init__(self, progress: Any, environ: Mapping[str, str]) -> None:
            self.progress = progress

        def run(self, loaded: LoadedScenario) -> RunResult:
            self.progress("boot", "starting QEMU")
            return RunResult(
                7, "failed", Path("/runs/r1"), Path("/runs/r1/report.json"), None, (failed,)
            )

    code, stdout, stderr = run_cli(["run", str(scenario_file)], runner_factory=FakeRunner)

    assert code == 7
    assert stdout == "/runs/r1\n"
    assert "INFO boot: starting QEMU" in stderr
    assert "FAIL marker-created (failed): /var/tmp/boot-marker does not exist" in stderr


def test_cli_prints_the_failure_and_its_remediation(tmp_path: Path) -> None:
    scenario_file = tmp_path / "basic-boot.yaml"
    scenario_file.write_text(SCENARIO)
    failure = HostUnavailableError("no read/write access to /dev/kvm")

    def factory(_progress: Any, _environ: Mapping[str, str]) -> Any:
        class Runner:
            def run(self, loaded: LoadedScenario) -> RunResult:
                return RunResult(3, "error", None, None, failure)

        return Runner()

    code, stdout, stderr = run_cli(["run", str(scenario_file)], runner_factory=factory)

    assert (code, stdout) == (3, "")
    assert "ERROR host: no read/write access to /dev/kvm" in stderr
    assert "HINT Install qemu-system-x86" in stderr


def test_cli_runs_the_coordinator_end_to_end_with_fake_ports(
    tmp_path: Path, scenario: LoadedScenario
) -> None:
    harness = Harness(tmp_path)

    code, stdout, stderr = run_cli(
        ["run", str(scenario.source_path)],
        runner_factory=lambda progress, _environ: RunCoordinator(
            harness.dependencies(), progress=progress
        ),
    )

    assert code == 0
    assert harness.workspace is not None
    assert stdout == f"{harness.workspace.path}\n"
    assert harness.report()["exit_code"] == code
    assert "INFO report: passed (exit code 0)" in stderr


def test_cli_unexpected_error_is_exit_10(tmp_path: Path) -> None:
    scenario_file = tmp_path / "basic-boot.yaml"
    scenario_file.write_text(SCENARIO)

    def broken(_progress: Any, _environ: Mapping[str, str]) -> Any:
        raise RuntimeError("wiring bug")

    code, _, stderr = run_cli(["run", str(scenario_file)], runner_factory=broken)

    assert code == ExitCode.INTERNAL
    assert "ERROR internal: unexpected RuntimeError: wiring bug" in stderr
