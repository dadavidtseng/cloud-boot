# ----------------------------------------------------------------------------------------------------
# ssh.py
# ----------------------------------------------------------------------------------------------------

"""Run-scoped SSH identity and the loopback SSH transport into the guest.

`create_access_identity` makes a fresh Ed25519 key for one run; its public key goes into the
seed's `meta-data`, and `AccessIdentity.destroy` deletes the key pair at cleanup. `SshTransport`
reaches the guest only through QEMU's loopback port forward, never reads the user's SSH
configuration or keys, and trusts only host keys recorded during this run.
"""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

import ipaddress
import math
import os
import re
import shlex
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from .errors import CloudBootError, ErrorPhase, ExitCode
from .process import (
    DEFAULT_CAPTURE_LIMIT_BYTES,
    Clock,
    CommandResult,
    CommandRunner,
    CommandSpec,
    ProcessTimeoutError,
    Sleeper,
    StreamSinks,
)

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "DEFAULT_SSH_USER",
    "SSH_CLIENT_FAILURE",
    "AccessIdentity",
    "AccessKeyError",
    "GuestCopyError",
    "GuestResult",
    "SshTransport",
    "SshTransportError",
    "create_access_identity",
]

# ----------------------------------------------------------------------------------------------------
DEFAULT_SSH_BINARY = "ssh"
DEFAULT_SSH_KEYGEN_BINARY = "ssh-keygen"
DEFAULT_SSH_USER = "ubuntu"  # The Ubuntu cloud image's default user, which gets the run key.
# Whole seconds: ssh's ConnectTimeout accepts only an integer.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0
DEFAULT_POWER_REQUEST_TIMEOUT_SECONDS = 30.0
# ssh exits 255 for its own failures. With LogLevel=ERROR, stderr can be empty for some of them
# (for example a connection closed before the banner), so 255 alone marks a transport failure.
SSH_CLIENT_FAILURE = 255
KEY_COMMENT = "cloud-boot-run"  # Marks the key in the guest's authorized_keys.
# wait_ready backoff: 0.5 s, doubling, capped at 5 s between attempts.
_READY_INITIAL_DELAY_SECONDS = 0.5
_READY_MAX_DELAY_SECONDS = 5.0
# ConnectTimeout covers the TCP connect, handshake, and key exchange, but not authentication or
# running `true`, which a guest still booting can stall; this bounds the rest of each attempt.
_READY_ATTEMPT_SLACK_SECONDS = 10.0
_COPY_CAPTURE_LIMIT_BYTES = 64 * 1024
# A plain account name; it cannot start with "-", so ssh never reads it as an option.
_USER_PATTERN = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


# ----------------------------------------------------------------------------------------------------
class AccessKeyError(CloudBootError):
    """Expected failure while creating the run's SSH identity."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="access-key-failed",
            phase=ErrorPhase.PREPARATION,
            remediation="Check that ssh-keygen is installed and the run directory is writable.",
            context=context,
            exit_code=ExitCode.PREPARATION,
        )


# ----------------------------------------------------------------------------------------------------
class SshTransportError(CloudBootError):
    """The guest could not be reached or controlled over SSH."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="ssh-unavailable",
            phase=ErrorPhase.SSH,
            remediation=(
                "Check the serial console: the guest may not have booted, started sshd, "
                "or installed the run key."
            ),
            context=context,
            exit_code=ExitCode.SSH,
        )


# ----------------------------------------------------------------------------------------------------
class GuestCopyError(CloudBootError):
    """A guest file was read but could not be written on the host."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="guest-copy-failed",
            phase=ErrorPhase.ARTIFACT,
            remediation="Check that the artifact directory is writable and has free space.",
            context=context,
            exit_code=ExitCode.REPORT,
        )


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AccessIdentity:
    """A run-scoped Ed25519 key pair and the run's own known-hosts file.

    Attributes:
        private_key: The private key file, mode 0600.
        public_key: The single-line OpenSSH public key, as written to `meta-data`.
        known_hosts: Host keys accepted during this run only.
    """

    private_key: Path
    public_key: str
    known_hosts: Path

    @property
    def public_key_path(self) -> Path:
        """Return the path of the `.pub` file next to the private key."""
        return self.private_key.with_name(f"{self.private_key.name}.pub")

    def destroy(self) -> None:
        """Delete both key files; safe to call more than once.

        The known-hosts file holds only the guest's public host key and is kept as evidence.
        """
        for path in (self.private_key, self.public_key_path):
            path.unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GuestResult:
    """The exact outcome of one script run in the guest.

    `exit_status` is the script's status in the guest; the ssh client's own failures are raised
    as `SshTransportError` instead.

    Attributes:
        exit_status: The script's exit status.
        stdout: Captured stdout, or only its tail when `stdout_dropped_bytes` is non-zero.
        stderr: Captured stderr, likewise bounded.
        duration_seconds: Wall time of the ssh invocation.
        stdout_dropped_bytes: Bytes of stdout not kept in memory.
        stderr_dropped_bytes: Bytes of stderr not kept in memory.
    """

    exit_status: int
    stdout: bytes
    stderr: bytes
    duration_seconds: float
    stdout_dropped_bytes: int = 0
    stderr_dropped_bytes: int = 0


# ----------------------------------------------------------------------------------------------------
def create_access_identity(
    runner: CommandRunner,
    run_dir: str | Path,
    *,
    deadline: float,
    ssh_keygen_binary: str = DEFAULT_SSH_KEYGEN_BINARY,
) -> AccessIdentity:
    """Generate a passphrase-less Ed25519 key and an empty known-hosts file in `run_dir`.

    The key never touches the user's `~/.ssh`. Create it before building seed media, because the
    public key goes into NoCloud metadata.

    Args:
        runner: Runs `ssh-keygen`.
        run_dir: The run's existing directory.
        deadline: Monotonic deadline for `ssh-keygen`.
        ssh_keygen_binary: The key generator to execute.

    Returns:
        The private key path (mode 0600), the single-line public key, and the known-hosts path.

    Raises:
        AccessKeyError: If identity files already exist, or the key cannot be created or read.
        ProcessStartError: If ssh-keygen cannot be executed.
        ProcessTimeoutError: If ssh-keygen does not finish by `deadline`. Partial key files
            stay in `run_dir` until the run's transient directory is removed.
    """
    directory = Path(run_dir)
    private_key = directory / "id_ed25519"
    public_key_path = directory / "id_ed25519.pub"
    known_hosts = directory / "known_hosts"
    # ssh-keygen would stop at an "Overwrite (y/n)?" prompt (stdin is /dev/null), and it never
    # checks known_hosts, so refuse existing files up front with a clear error.
    for path in (private_key, public_key_path, known_hosts):
        if path.exists() or path.is_symlink():
            raise AccessKeyError("access identity files already exist", context={"path": str(path)})

    def discard_keys() -> None:
        for path in (private_key, public_key_path):
            path.unlink(missing_ok=True)

    argv = (
        ssh_keygen_binary, "-q", "-t", "ed25519", "-N", "", "-C", KEY_COMMENT, "-f", str(private_key),
    )  # fmt: skip
    result = runner.run(CommandSpec(argv, name="ssh-keygen"), deadline=deadline)
    if not result.succeeded:
        discard_keys()
        raise AccessKeyError(
            "ssh-keygen could not create the run key",
            context={"returncode": result.returncode, "stderr": result.stderr_tail()},
        )
    try:
        # ssh-keygen already writes 0600, but ssh refuses a key others can read; do not rely on it.
        os.chmod(private_key, 0o600)
        public_key = public_key_path.read_text(encoding="ascii").strip()
        known_hosts.touch(mode=0o600, exist_ok=False)
    except (OSError, UnicodeDecodeError) as e:
        discard_keys()
        raise AccessKeyError("cannot finish the run key", context={"path": str(directory)}) from e
    # The seed's meta-data needs exactly one line, and we asked for an Ed25519 key.
    if "\n" in public_key or not public_key.startswith("ssh-ed25519 "):
        discard_keys()
        known_hosts.unlink(missing_ok=True)
        raise AccessKeyError("ssh-keygen wrote an unexpected public key")
    return AccessIdentity(private_key=private_key, public_key=public_key, known_hosts=known_hosts)


# ----------------------------------------------------------------------------------------------------
def _config_path(path: Path) -> str:
    """Quote `path` as an ssh `-o` value that ssh reads back unchanged.

    The path is made absolute so it does not depend on ssh's working directory. ssh splits
    unquoted values on whitespace, expands %-tokens, and unescapes backslashes inside quotes, so
    "%" is doubled and characters that cannot be quoted safely are rejected.

    Raises:
        ValueError: If the path contains a double quote, backslash, or line break.
    """
    text = str(path.absolute())
    if any(character in text for character in '"\\\r\n'):
        raise ValueError("SSH file paths must not contain double quotes, backslashes, or line breaks")
    return '"' + text.replace("%", "%%") + '"'


# ----------------------------------------------------------------------------------------------------
def _guest_result(result: CommandResult) -> GuestResult:
    """Return `result` as the guest script's outcome, with ssh's exit status as the script's."""
    return GuestResult(
        exit_status=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_seconds=result.duration_seconds,
        stdout_dropped_bytes=result.stdout_dropped_bytes,
        stderr_dropped_bytes=result.stderr_dropped_bytes,
    )


# ----------------------------------------------------------------------------------------------------
class SshTransport:
    """Run fixed scripts in the guest over loopback SSH with a run-scoped identity.

    Every invocation ignores the user's SSH configuration, uses only the run key, records host keys
    only in the run's known-hosts file, disables agent, X11, and port forwarding, and bounds the
    connection with a timeout and keepalives. Scripts travel on stdin to `sh -s --`; values are
    positional parameters, each quoted because ssh joins the remote command into one string.

    Args:
        runner: Runs every ssh invocation.
        identity: The run's key pair and known-hosts file.
        host: A loopback IP address; QEMU forwards SSH only on loopback.
        port: The host port QEMU forwards to the guest's port 22.
        user: The guest account that holds the run key.
        ssh_binary: The ssh client to execute.
        clock: Monotonic clock for `wait_ready`.
        sleeper: Waits between `wait_ready` attempts.
        connect_timeout_seconds: ssh's ConnectTimeout, in whole seconds.

    Raises:
        ValueError: If `host` is not a loopback address, or another argument is out of range.
    """

    def __init__(
        self,
        runner: CommandRunner,
        identity: AccessIdentity,
        *,
        host: str,
        port: int,
        user: str = DEFAULT_SSH_USER,
        ssh_binary: str = DEFAULT_SSH_BINARY,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
        connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        # An IP literal, not a name: resolving "localhost" could pick ::1 while QEMU listens on
        # 127.0.0.1, and BatchMode with accept-new must never reach another machine.
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError("the SSH transport only connects to a loopback address")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not _USER_PATTERN.fullmatch(user):
            raise ValueError("SSH user must be a plain lowercase account name")
        # ssh's ConnectTimeout rejects fractions, and every attempt would then fail as if the
        # guest were down.
        if (
            isinstance(connect_timeout_seconds, bool)
            or not isinstance(connect_timeout_seconds, int)
            or connect_timeout_seconds <= 0
        ):
            raise ValueError("connect timeout must be a positive whole number of seconds")
        self._runner = runner
        self._clock = clock
        self._sleeper = sleeper
        self._connect_timeout_seconds = connect_timeout_seconds
        self._base_argv = (
            ssh_binary,
            # No user or system ssh_config, and never prompt.
            "-F", "/dev/null",
            "-o", "BatchMode=yes",
            # Only the run key, and only this run's known hosts. The guest is new each run, so its
            # first host key is accepted; a different key later (another process on the port) is
            # refused.
            "-o", "IdentitiesOnly=yes",
            "-o", f"IdentityFile={_config_path(identity.private_key)}",
            "-o", f"UserKnownHostsFile={_config_path(identity.known_hosts)}",
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "StrictHostKeyChecking=accept-new",
            # The guest gets nothing from the host: no agent, X11, port forwarding, or LocalCommand.
            "-o", "ForwardAgent=no",
            "-o", "ForwardX11=no",
            "-o", "ClearAllForwardings=yes",
            "-o", "PermitLocalCommand=no",
            # Give up on a dead connection after about 15 s: 3 keepalives, 5 s apart.
            "-o", f"ConnectTimeout={connect_timeout_seconds}",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=3",
            # Hide "Permanently added ... to known hosts", so stderr holds only the script's output.
            "-o", "LogLevel=ERROR",
            "-p", str(port),
            "-l", user,
            host,
        )  # fmt: skip

    @property
    def base_argv(self) -> tuple[str, ...]:
        """Return the ssh invocation shared by every guest operation, up to the destination."""
        return self._base_argv

    def wait_ready(self, *, deadline: float) -> int:
        """Retry `true` over SSH with bounded exponential backoff until it succeeds.

        Ready means sshd accepts the run key; cloud-init may still be running, and the caller
        waits for it separately. Early in boot, connections are refused before sshd starts and
        the key is refused until cloud-init installs it, so every failure is retried until
        `deadline`. That includes failures that will never clear, such as a changed host key;
        the error's `last_failure` then shows ssh's own message.

        Returns:
            The number of attempts made; the run report does not include it yet.

        Raises:
            ValueError: If `deadline` is not finite.
            SshTransportError: If the guest is not reachable before `deadline`.
        """
        if not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        delay = _READY_INITIAL_DELAY_SECONDS
        attempt = 0
        last_failure = "no attempt finished before the deadline"
        while True:
            now = self._clock()
            if now >= deadline:
                raise SshTransportError(
                    "guest SSH did not become ready before the deadline",
                    context={"attempts": attempt, "last_failure": last_failure},
                )
            attempt += 1
            attempt_deadline = min(
                deadline, now + self._connect_timeout_seconds + _READY_ATTEMPT_SLACK_SECONDS
            )
            try:
                result = self._runner.run(
                    CommandSpec((*self._base_argv, "true"), name="ssh"), deadline=attempt_deadline
                )
            except ProcessTimeoutError:
                last_failure = "ssh attempt timed out"
            else:
                if result.returncode == 0:
                    return attempt
                last_failure = result.stderr_tail() or f"ssh exited with {result.returncode}"
            pause = min(delay, deadline - self._clock())
            if pause > 0:
                self._sleeper(pause)
            delay = min(delay * 2, _READY_MAX_DELAY_SECONDS)

    def execute(
        self,
        script: str,
        args: Sequence[str] = (),
        *,
        timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> GuestResult:
        """Run `script` in the guest with `args` as `$1`, `$2`, ...

        A non-zero guest exit status is returned, not raised; scripts must not exit 255.

        Raises:
            SshTransportError: If ssh itself fails (exit status 255).
            ProcessTimeoutError: If the script outlives `timeout`.
        """
        result = self._run_script(script.encode("utf-8"), args, timeout=timeout)
        self._raise_on_transport_failure(result, "guest command")
        return _guest_result(result)

    def _run_script(
        self,
        script: bytes,
        args: Sequence[str],
        *,
        timeout: float,
        sinks: StreamSinks | None = None,
        capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    ) -> CommandResult:
        """Run `script` in the guest with `sh -s`, reading it from stdin.

        ssh joins the remote command into one string for the guest's shell, so each argument is
        quoted to arrive as exactly one positional parameter. The script itself never passes
        through that shell. Commands in a script must not read stdin, which holds the script.

        Raises:
            TypeError: If `args` is a single string rather than a sequence of strings.
        """
        if isinstance(args, str):
            raise TypeError("args must be a sequence of strings, not a string")
        remote = ("sh", "-s", "--", *(shlex.quote(value) for value in args))
        spec = CommandSpec(
            (*self._base_argv, *remote),
            stdin=script,
            name="ssh",
            capture_limit_bytes=capture_limit_bytes,
        )
        return self._runner.run(spec, timeout=timeout, sinks=sinks)

    @staticmethod
    def _raise_on_transport_failure(result: CommandResult, operation: str) -> None:
        """Raise `SshTransportError` when ssh itself failed, which it reports as exit 255."""
        if result.returncode == SSH_CLIENT_FAILURE:
            raise SshTransportError(
                f"SSH transport failed during {operation}",
                context={"returncode": result.returncode, "stderr": result.stderr_tail()},
            )

    def copy_from(
        self, remote: str, local: str | Path, *, timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
    ) -> GuestResult:
        """Stream the guest file `remote` into a new host file `local` with mode 0600.

        Bytes stream to disk; memory holds only a bounded tail, and the returned stdout is empty.
        If the guest cannot read the file, the partial host file is removed and the non-zero
        result, including the guest's stderr, is returned.

        Raises:
            ValueError: If `remote` is not an absolute guest path.
            SshTransportError: If ssh itself fails.
            GuestCopyError: If the host file cannot be created or written.
        """
        if "\x00" in remote or not PurePosixPath(remote).is_absolute():
            raise ValueError("remote path must be an absolute guest path")
        destination = Path(local)
        try:
            # O_EXCL also refuses a symlink, even a dangling one, so the copy never lands elsewhere.
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as e:
            raise GuestCopyError(
                "cannot create the host copy", context={"path": str(destination)}
            ) from e
        keep = False
        try:
            try:
                # Buffered: a write error can surface only when the file is closed, outside the
                # sink.
                with os.fdopen(descriptor, "wb") as stream:
                    result = self._run_script(
                        b'exec cat -- "$1"',
                        (remote,),
                        timeout=timeout,
                        sinks=StreamSinks(stdout=stream.write),
                        # The limit covers both streams; keep enough for the guest's stderr.
                        capture_limit_bytes=_COPY_CAPTURE_LIMIT_BYTES,
                    )
            except OSError as e:
                raise GuestCopyError(
                    "cannot write the host copy", context={"path": str(destination)}
                ) from e
            self._raise_on_transport_failure(result, "guest file copy")
            if result.sink_failures:
                raise GuestCopyError(
                    "cannot write the host copy",
                    context={"path": str(destination), "error": str(result.sink_failures[0].error)},
                )
            keep = result.returncode == 0
            # The file content lives on disk, so drop the in-memory stdout tail from the result.
            return replace(_guest_result(result), stdout=b"", stdout_dropped_bytes=0)
        finally:
            if not keep:
                destination.unlink(missing_ok=True)

    def shutdown(self, *, timeout: float = DEFAULT_POWER_REQUEST_TIMEOUT_SECONDS) -> None:
        """Ask the guest to power off; follow with `QemuBackend.stop` to wait for QEMU to exit.

        systemd's `shutdown -h` powers off rather than halting, so QEMU exits with the guest.

        Raises:
            SshTransportError: If the guest refuses the request, for example when sudo needs a
                password.
            ProcessTimeoutError: If ssh does not return within `timeout`.
        """
        self._request_power_change("sudo -n shutdown -h now", "shutdown", timeout)

    def reboot(self, *, timeout: float = DEFAULT_POWER_REQUEST_TIMEOUT_SECONDS) -> None:
        """Ask the guest to reboot; call `wait_ready` again before using the transport.

        Raises:
            SshTransportError: If the guest refuses the request.
            ProcessTimeoutError: If ssh does not return within `timeout`.
        """
        self._request_power_change("sudo -n reboot", "reboot", timeout)

    def _request_power_change(self, command: str, operation: str, timeout: float) -> None:
        """Run a privileged power command with `sudo -n`, so a password prompt fails at once."""
        result = self._run_script(command.encode("ascii"), (), timeout=timeout)
        # The guest drops the connection while going down, so ssh's own 255 is expected here. It
        # can also mean the request never reached the guest; `QemuBackend.stop` escalates to
        # signals either way.
        if result.returncode in (0, SSH_CLIENT_FAILURE):
            return
        raise SshTransportError(
            f"guest refused the {operation} request",
            context={"exit_status": result.returncode, "stderr": result.stderr_tail()},
        )
