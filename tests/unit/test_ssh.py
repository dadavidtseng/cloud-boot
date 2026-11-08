"""Unit tests for the ephemeral SSH identity and guest transport."""

from __future__ import annotations

import io
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cloud_boot.errors import ErrorPhase, ExitCode
from cloud_boot.process import CommandRunner, ProcessTimeoutError
from cloud_boot.ssh import (
    AccessIdentity,
    AccessKeyError,
    GuestCopyError,
    SshTransport,
    SshTransportError,
    create_access_identity,
)

PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyMaterialForUnitTests cloud-boot-run"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class CapturingStdin(io.BytesIO):
    """Keeps what the runner wrote even after the runner closes the pipe."""

    def __init__(self) -> None:
        super().__init__()
        self.captured = b""

    def close(self) -> None:
        self.captured = self.getvalue()
        super().close()


class FakePopen:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        exit_after_polls: int | None = 0,
    ) -> None:
        self.pid = 999_999_999  # Above pid_max, so killpg fails and ManagedProcess falls back.
        self.stdin = CapturingStdin()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode: int | None = None
        self._polls_left = exit_after_polls
        self._exit_code = returncode

    def poll(self) -> int | None:
        if self.returncode is None and self._polls_left is not None:
            if self._polls_left <= 0:
                self.returncode = self._exit_code
            else:
                self._polls_left -= 1
        return self.returncode

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = 0

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9


class FakeFactory:
    def __init__(
        self, processes: list[FakePopen], on_call: Callable[[tuple[str, ...]], None] | None = None
    ) -> None:
        self.processes = processes
        self.started: list[FakePopen] = []
        self.argvs: list[tuple[str, ...]] = []
        self._on_call = on_call

    def __call__(self, argv: list[str], **_kwargs: Any) -> FakePopen:
        self.argvs.append(tuple(argv))
        if self._on_call is not None:
            self._on_call(tuple(argv))
        process = self.processes.pop(0)
        self.started.append(process)
        return process


def make_runner(factory: FakeFactory, clock: FakeClock) -> CommandRunner:
    return CommandRunner(
        clock=clock, sleeper=clock.sleep, popen_factory=factory, termination_grace_seconds=0.5
    )


def make_transport(
    tmp_path: Path, processes: list[FakePopen]
) -> tuple[SshTransport, FakeFactory, FakeClock]:
    clock = FakeClock()
    factory = FakeFactory(processes)
    identity = AccessIdentity(
        private_key=tmp_path / "id_ed25519", public_key=PUBLIC_KEY, known_hosts=tmp_path / "kh"
    )
    transport = SshTransport(
        make_runner(factory, clock),
        identity,
        host="127.0.0.1",
        port=40001,
        clock=clock,
        sleeper=clock.sleep,
    )
    return transport, factory, clock


def fake_keygen(public_key: str = PUBLIC_KEY) -> Callable[[tuple[str, ...]], None]:
    def write_keys(argv: tuple[str, ...]) -> None:
        private = Path(argv[argv.index("-f") + 1])
        private.write_text("PRIVATE KEY")
        private.chmod(0o644)
        Path(f"{private}.pub").write_text(public_key + "\n")

    return write_keys


def mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_create_access_identity_generates_a_passphraseless_ed25519_key(tmp_path: Path) -> None:
    factory = FakeFactory([FakePopen()], fake_keygen())

    identity = create_access_identity(make_runner(factory, FakeClock()), tmp_path, deadline=30.0)

    key = tmp_path / "id_ed25519"
    assert factory.argvs == [
        ("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "cloud-boot-run", "-f", str(key))
    ]
    assert identity == AccessIdentity(
        private_key=key, public_key=PUBLIC_KEY, known_hosts=tmp_path / "known_hosts"
    )
    assert mode(key) == 0o600
    assert identity.known_hosts.read_bytes() == b""
    assert mode(identity.known_hosts) == 0o600


@pytest.mark.parametrize("existing", ["id_ed25519", "id_ed25519.pub", "known_hosts"])
def test_create_access_identity_never_reuses_existing_files(tmp_path: Path, existing: str) -> None:
    (tmp_path / existing).write_text("previous run")
    factory = FakeFactory([])

    with pytest.raises(AccessKeyError, match="already exist"):
        create_access_identity(make_runner(factory, FakeClock()), tmp_path, deadline=30.0)

    assert factory.argvs == []
    assert (tmp_path / existing).read_text() == "previous run"


def test_create_access_identity_failure_removes_partial_keys(tmp_path: Path) -> None:
    factory = FakeFactory([FakePopen(returncode=1, stderr=b"disk full")], fake_keygen())

    with pytest.raises(AccessKeyError) as raised:
        create_access_identity(make_runner(factory, FakeClock()), tmp_path, deadline=30.0)

    assert raised.value.phase == ErrorPhase.PREPARATION
    assert raised.value.exit_code == ExitCode.PREPARATION
    assert raised.value.context["stderr"] == "disk full"
    assert list(tmp_path.iterdir()) == []


def test_create_access_identity_rejects_an_unexpected_public_key(tmp_path: Path) -> None:
    factory = FakeFactory([FakePopen()], fake_keygen("ssh-rsa AAAAB3Nza unexpected"))

    with pytest.raises(AccessKeyError, match="unexpected public key"):
        create_access_identity(make_runner(factory, FakeClock()), tmp_path, deadline=30.0)

    assert list(tmp_path.iterdir()) == []


def test_destroy_removes_both_keys_and_keeps_known_hosts(tmp_path: Path) -> None:
    factory = FakeFactory([FakePopen()], fake_keygen())
    identity = create_access_identity(make_runner(factory, FakeClock()), tmp_path, deadline=30.0)

    identity.destroy()
    identity.destroy()

    assert not identity.private_key.exists()
    assert not identity.public_key_path.exists()
    assert identity.known_hosts.exists()


def test_ssh_argv_isolates_user_configuration_identity_and_forwarding(tmp_path: Path) -> None:
    transport, _, _ = make_transport(tmp_path, [])

    assert transport.base_argv == (
        "ssh", "-F", "/dev/null",
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", f'IdentityFile="{tmp_path}/id_ed25519"',
        "-o", f'UserKnownHostsFile="{tmp_path}/kh"',
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ForwardAgent=no",
        "-o", "ForwardX11=no",
        "-o", "ClearAllForwardings=yes",
        "-o", "PermitLocalCommand=no",
        "-o", "ConnectTimeout=5",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        "-o", "LogLevel=ERROR",
        "-p", "40001", "-l", "ubuntu", "127.0.0.1",
    )  # fmt: skip
    user_ssh = os.path.expanduser("~/.ssh")
    assert not any(user_ssh in argument or "~" in argument for argument in transport.base_argv)


def test_ssh_file_options_quote_spaces_and_escape_percent_tokens(tmp_path: Path) -> None:
    transport, _, _ = make_transport(tmp_path / "a b%h", [])

    assert f'IdentityFile="{tmp_path}/a b%%h/id_ed25519"' in transport.base_argv


@pytest.mark.parametrize(
    "overrides",
    [
        {"host": "0.0.0.0"},
        {"host": "localhost"},
        {"port": 0},
        {"port": True},
        {"user": "-oProxyCommand=touch /tmp/x"},
        {"user": "root;id"},
    ],
)
def test_transport_rejects_non_loopback_endpoints_and_unsafe_users(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    options: dict[str, Any] = {"host": "127.0.0.1", "port": 40001, "user": "ubuntu", **overrides}
    identity = AccessIdentity(tmp_path / "id_ed25519", PUBLIC_KEY, tmp_path / "kh")

    with pytest.raises(ValueError, match=r"loopback|port|user"):
        SshTransport(CommandRunner(), identity, **options)


def test_execute_sends_the_script_on_stdin_with_quoted_positional_values(tmp_path: Path) -> None:
    fake = FakePopen(returncode=3, stdout=b"out", stderr=b"err")
    transport, factory, _ = make_transport(tmp_path, [fake])
    script = 'test -f "$1" && printf "%s" "$2"'

    result = transport.execute(script, ["/tmp/a b; rm -rf /", "$(id)"])

    assert factory.argvs[0][-5:] == ("sh", "-s", "--", "'/tmp/a b; rm -rf /'", "'$(id)'")
    assert fake.stdin.captured == script.encode()
    # A non-zero guest status is evidence for the caller, not a transport failure.
    assert (result.exit_status, result.stdout, result.stderr) == (3, b"out", b"err")


def test_execute_rejects_a_bare_string_of_arguments(tmp_path: Path) -> None:
    transport, _, _ = make_transport(tmp_path, [])

    with pytest.raises(TypeError, match="sequence of strings"):
        transport.execute("true", "/etc/hostname")


def test_ssh_exit_255_is_a_transport_failure_even_without_stderr(tmp_path: Path) -> None:
    transport, _, _ = make_transport(tmp_path, [FakePopen(returncode=255)])

    with pytest.raises(SshTransportError) as raised:
        transport.execute("true")

    assert raised.value.phase == ErrorPhase.SSH
    assert raised.value.exit_code == ExitCode.SSH


def test_execute_timeout_is_a_process_timeout(tmp_path: Path) -> None:
    transport, _, clock = make_transport(tmp_path, [FakePopen(exit_after_polls=None)])

    with pytest.raises(ProcessTimeoutError) as raised:
        transport.execute("sleep 600", timeout=1.0)

    assert raised.value.exit_code == ExitCode.TIMEOUT
    assert clock.now <= 1.0 + 2 * 0.5


def test_wait_ready_retries_boot_time_failures_with_exponential_backoff(tmp_path: Path) -> None:
    processes = [
        FakePopen(returncode=255),
        FakePopen(returncode=255, stderr=b"ubuntu@127.0.0.1: Permission denied (publickey)."),
        FakePopen(),
    ]
    transport, factory, clock = make_transport(tmp_path, processes)

    attempts = transport.wait_ready(deadline=60.0)

    assert attempts == 3
    assert clock.sleeps == [0.5, 1.0]
    assert all(argv[-1] == "true" for argv in factory.argvs)


def test_wait_ready_gives_up_at_the_deadline_without_overshooting(tmp_path: Path) -> None:
    refused = b"ssh: connect to host 127.0.0.1 port 40001: Connection refused"
    transport, _, clock = make_transport(
        tmp_path, [FakePopen(returncode=255, stderr=refused) for _ in range(6)]
    )

    with pytest.raises(SshTransportError, match="did not become ready") as raised:
        transport.wait_ready(deadline=5.0)

    assert raised.value.context["attempts"] == 4
    assert "Connection refused" in str(raised.value.context["last_failure"])
    assert clock.sleeps == [0.5, 1.0, 2.0, 1.5]
    assert clock.now == pytest.approx(5.0)


def test_wait_ready_treats_a_hung_attempt_as_a_failed_attempt(tmp_path: Path) -> None:
    transport, _, clock = make_transport(tmp_path, [FakePopen(exit_after_polls=None), FakePopen()])

    assert transport.wait_ready(deadline=60.0) == 2
    assert clock.now < 60.0


def test_copy_from_streams_the_guest_file_to_a_private_host_file(tmp_path: Path) -> None:
    content = b"cloud-init log line\n" * 1000
    fake = FakePopen(stdout=content)
    transport, factory, _ = make_transport(tmp_path, [fake])
    local = tmp_path / "cloud-init.log"

    result = transport.copy_from("/var/log/cloud-init.log", local)

    assert result.exit_status == 0
    assert local.read_bytes() == content
    assert mode(local) == 0o600
    # Copies stream to disk; the result does not repeat the file content.
    assert result.stdout == b""
    assert factory.started[0] is fake
    assert factory.argvs[0][-4:] == ("sh", "-s", "--", "/var/log/cloud-init.log")
    assert fake.stdin.captured == b'exec cat -- "$1"'


def test_copy_from_removes_the_partial_host_file_when_the_guest_read_fails(tmp_path: Path) -> None:
    fake = FakePopen(returncode=1, stdout=b"partial", stderr=b"cat: /missing: No such file")
    transport, _, _ = make_transport(tmp_path, [fake])
    local = tmp_path / "missing.log"

    result = transport.copy_from("/missing", local)

    assert result.exit_status == 1
    assert b"No such file" in result.stderr
    assert not local.exists()


def test_copy_from_refuses_existing_host_files_and_relative_guest_paths(tmp_path: Path) -> None:
    transport, factory, _ = make_transport(tmp_path, [])
    existing = tmp_path / "existing.log"
    existing.write_text("keep")

    with pytest.raises(GuestCopyError):
        transport.copy_from("/var/log/syslog", existing)
    with pytest.raises(ValueError, match="absolute"):
        transport.copy_from("var/log/syslog", tmp_path / "new.log")

    assert existing.read_text() == "keep"
    assert factory.argvs == []


@pytest.mark.parametrize("returncode", [0, 255])
@pytest.mark.parametrize(
    ("operation", "script"),
    [("shutdown", b"sudo -n shutdown -h now"), ("reboot", b"sudo -n reboot")],
)
def test_power_requests_accept_success_or_the_dropped_connection(
    tmp_path: Path, operation: str, script: bytes, returncode: int
) -> None:
    fake = FakePopen(returncode=returncode)
    transport, _, _ = make_transport(tmp_path, [fake])

    getattr(transport, operation)()

    assert fake.stdin.captured == script


def test_power_request_refused_by_the_guest_is_reported(tmp_path: Path) -> None:
    fake = FakePopen(returncode=1, stderr=b"sudo: a password is required")
    transport, _, _ = make_transport(tmp_path, [fake])

    with pytest.raises(SshTransportError, match="refused the shutdown") as raised:
        transport.shutdown()

    assert "password is required" in str(raised.value.context["stderr"])
