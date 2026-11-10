# ----------------------------------------------------------------------------------------------------
# assertions.py
# ----------------------------------------------------------------------------------------------------

"""Evaluate Milestone 0 guest assertions over the SSH transport."""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue

from .models import (
    AssertionSpec,
    CloudInitStatusAssertion,
    FileAssertion,
    PackageInstalledAssertion,
)
from .process import Clock, ProcessTimeoutError
from .ssh import GuestResult, SshTransportError

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "MAX_FILE_CONTENT_BYTES",
    "AssertionRegistry",
    "AssertionResult",
    "AssertionStatus",
    "CommandReference",
    "Evaluation",
    "GuestTransport",
    "block_all",
    "boot_passed",
    "default_registry",
    "run_assertions",
]

# ----------------------------------------------------------------------------------------------------
# Content is compared on the host. The limit stays well below the 1 MiB stdout capture so the
# header line plus content can never be truncated.
MAX_FILE_CONTENT_BYTES = 512 * 1024
_STDERR_REASON_BYTES = 512

# Scripts exit 3 when a required guest tool is missing and 4 when `sudo -n` is unavailable.
# They never exit 255, which the SSH transport reserves for its own failures.
_TOOL_MISSING = 3
_SUDO_UNAVAILABLE = 4
_SUDO_GUARD = "sudo -n true 2>/dev/null || exit 4\n"

CLOUD_INIT_STATUS_SCRIPT = (
    _SUDO_GUARD
    + "command -v cloud-init >/dev/null 2>&1 || exit 3\n"
    + "sudo -n cloud-init status --format=json\n"
)
# $1 path, $2 "content" to send the file body, $3 the content limit in bytes.
FILE_SCRIPT = (
    _SUDO_GUARD
    + 'if ! sudo -n test -e "$1"; then printf "missing\\n"; exit 0; fi\n'
    + 'if sudo -n test -f "$1"; then\n'
    + '  size=$(sudo -n stat -c %s -- "$1") || exit 3\n'
    + '  printf "file %s\\n" "$size"\n'
    + '  if [ "$2" = content ] && [ "$size" -le "$3" ]; then sudo -n cat -- "$1" || exit 3; fi\n'
    + 'elif sudo -n test -d "$1"; then printf "directory\\n"\n'
    + 'else printf "other\\n"\n'
    + "fi\n"
)
PACKAGE_SCRIPT = (
    _SUDO_GUARD
    + "command -v dpkg-query >/dev/null 2>&1 || exit 3\n"
    + "sudo -n dpkg-query -W -f='${db:Status-Status}\\t${Version}\\n' -- \"$1\"\n"
)

type AssertionStatus = Literal["passed", "failed", "blocked"]


# ----------------------------------------------------------------------------------------------------
class GuestTransport(Protocol):
    """The part of `SshTransport` that evaluators need."""

    def execute(self, script: str, args: Sequence[str] = (), *, timeout: float = ...) -> GuestResult:
        """Run a fixed script in the guest with positional values."""
        ...


# ----------------------------------------------------------------------------------------------------
class CommandReference(BaseModel):
    """Which fixed guest script ran, and with which positional values, for reproduction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    script: str
    args: tuple[str, ...]


# ----------------------------------------------------------------------------------------------------
class AssertionResult(BaseModel):
    """The uniform outcome of one assertion.

    `failed` means the assertion ran and the guest did not match; `blocked` means it could not run
    because the SSH transport was lost.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: str
    boot: str
    required: bool
    status: AssertionStatus
    expected: JsonValue
    observed: JsonValue
    reason: str
    duration_seconds: float
    command: CommandReference | None


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Evaluation:
    """What one evaluator observed in the guest."""

    passed: bool
    observed: JsonValue
    reason: str
    command: CommandReference


type Evaluator = Callable[[AssertionSpec, GuestTransport], Evaluation]


# ----------------------------------------------------------------------------------------------------
class AssertionRegistry:
    """Map each assertion type to exactly one evaluator."""

    def __init__(self) -> None:
        self._evaluators: dict[str, Evaluator] = {}

    def register(self, type_name: str, evaluator: Evaluator) -> None:
        """Add the evaluator for `type_name`.

        Raises:
            ValueError: If `type_name` already has an evaluator.
        """
        if type_name in self._evaluators:
            raise ValueError(f"an evaluator is already registered for {type_name!r}")
        self._evaluators[type_name] = evaluator

    def evaluate(self, assertion: AssertionSpec, transport: GuestTransport) -> Evaluation:
        """Run the evaluator registered for the assertion's type.

        Raises:
            LookupError: If no evaluator handles the assertion's type.
        """
        evaluator = self._evaluators.get(assertion.type)
        if evaluator is None:
            raise LookupError(f"no evaluator is registered for {assertion.type!r}")
        return evaluator(assertion, transport)


# ----------------------------------------------------------------------------------------------------
def default_registry() -> AssertionRegistry:
    """Return a registry with the three Milestone 0 evaluators."""
    registry = AssertionRegistry()
    registry.register("cloud-init-status", _evaluate_cloud_init_status)
    registry.register("file", _evaluate_file)
    registry.register("package-installed", _evaluate_package)
    return registry


# ----------------------------------------------------------------------------------------------------
def run_assertions(
    assertions: Sequence[AssertionSpec],
    transport: GuestTransport,
    *,
    boot: str,
    registry: AssertionRegistry | None = None,
    clock: Clock = time.monotonic,
) -> list[AssertionResult]:
    """Evaluate assertions in declaration order.

    A per-assertion timeout fails only that assertion. Losing the SSH transport blocks the
    assertion in progress and every assertion after it; nothing is reported as passed without
    running.

    Args:
        assertions: The boot plan's assertions, in declaration order.
        transport: Runs fixed scripts in the guest.
        boot: The boot plan name recorded on every result.
        registry: Evaluators by type; defaults to the Milestone 0 evaluators.
        clock: Monotonic clock for durations.

    Returns:
        One result per assertion, in the same order.
    """
    active = registry if registry is not None else default_registry()
    results: list[AssertionResult] = []
    blocked_reason: str | None = None
    for assertion in assertions:
        if blocked_reason is not None:
            results.append(_blocked(assertion, boot, blocked_reason))
            continue
        started = clock()
        try:
            evaluation = active.evaluate(assertion, transport)
        except SshTransportError as e:
            detail = str(e.context.get("stderr") or "").strip()
            blocked_reason = f"SSH transport unavailable: {e.message}" + (
                f" ({detail})" if detail else ""
            )
            results.append(_blocked(assertion, boot, blocked_reason, clock() - started))
            continue
        except ProcessTimeoutError:
            results.append(
                _result(
                    assertion,
                    boot,
                    status="failed",
                    observed=None,
                    reason=f"guest check exceeded its {assertion.timeout_seconds}-second timeout",
                    duration=clock() - started,
                    command=None,
                )
            )
            continue
        results.append(
            _result(
                assertion,
                boot,
                status="passed" if evaluation.passed else "failed",
                observed=evaluation.observed,
                reason=evaluation.reason,
                duration=clock() - started,
                command=evaluation.command,
            )
        )
    return results


# ----------------------------------------------------------------------------------------------------
def block_all(
    assertions: Sequence[AssertionSpec], *, boot: str, reason: str
) -> list[AssertionResult]:
    """Mark every assertion blocked, for example when SSH never became ready."""
    return [_blocked(assertion, boot, reason) for assertion in assertions]


# ----------------------------------------------------------------------------------------------------
def boot_passed(results: Sequence[AssertionResult]) -> bool:
    """Return whether every required assertion passed; failed or blocked ones fail the boot."""
    return all(result.status == "passed" for result in results if result.required)


# ----------------------------------------------------------------------------------------------------
def _expected(assertion: AssertionSpec) -> JsonValue:
    if isinstance(assertion, CloudInitStatusAssertion):
        return {"extended_status": assertion.expected}
    if isinstance(assertion, FileAssertion):
        return {
            "path": assertion.path,
            "exists": assertion.exists,
            "content_contains": assertion.content_contains,
        }
    return {"name": assertion.name, "status": "installed"}


# ----------------------------------------------------------------------------------------------------
def _result(
    assertion: AssertionSpec,
    boot: str,
    *,
    status: AssertionStatus,
    observed: JsonValue,
    reason: str,
    duration: float,
    command: CommandReference | None,
) -> AssertionResult:
    return AssertionResult(
        id=assertion.id,
        type=assertion.type,
        boot=boot,
        required=assertion.required,
        status=status,
        expected=_expected(assertion),
        observed=observed,
        reason=reason,
        duration_seconds=max(0.0, duration),
        command=command,
    )


# ----------------------------------------------------------------------------------------------------
def _blocked(
    assertion: AssertionSpec, boot: str, reason: str, duration: float = 0.0
) -> AssertionResult:
    return _result(
        assertion,
        boot,
        status="blocked",
        observed=None,
        reason=reason,
        duration=duration,
        command=None,
    )


# ----------------------------------------------------------------------------------------------------
def _unobservable(result: GuestResult, what: str) -> str | None:
    if result.exit_status == _TOOL_MISSING:
        return f"a guest tool needed to check {what} is missing"
    if result.exit_status == _SUDO_UNAVAILABLE:
        return "sudo -n is not available for the SSH user"
    return None


# ----------------------------------------------------------------------------------------------------
def _stderr_text(result: GuestResult) -> str:
    return result.stderr[-_STDERR_REASON_BYTES:].decode("utf-8", errors="replace").strip()


# ----------------------------------------------------------------------------------------------------
def _evaluate_cloud_init_status(assertion: AssertionSpec, transport: GuestTransport) -> Evaluation:
    if not isinstance(assertion, CloudInitStatusAssertion):
        raise TypeError("expected a cloud-init-status assertion")
    command = CommandReference(script="cloud-init-status", args=())
    result = transport.execute(CLOUD_INIT_STATUS_SCRIPT, (), timeout=assertion.timeout_seconds)
    problem = _unobservable(result, "cloud-init status")
    if problem is not None:
        return Evaluation(False, {"exit_status": result.exit_status}, problem, command)
    # cloud-init exits non-zero for error and degraded states but still prints the JSON status.
    try:
        status = json.loads(result.stdout)
        if not isinstance(status, dict):
            raise ValueError("status is not a JSON object")
    except ValueError:
        reason = "could not parse cloud-init status output"
        detail = _stderr_text(result)
        return Evaluation(
            False,
            {"exit_status": result.exit_status},
            f"{reason}: {detail}" if detail else reason,
            command,
        )
    extended = status.get("extended_status", status.get("status"))
    errors = status.get("errors")
    recoverable = status.get("recoverable_errors")
    observed: JsonValue = {
        "status": status.get("status"),
        "extended_status": extended,
        "datasource": status.get("datasource"),
        "errors": len(errors) if isinstance(errors, list) else None,
        "recoverable_errors": (
            sum(len(entries) for entries in recoverable.values() if isinstance(entries, list))
            if isinstance(recoverable, dict)
            else None
        ),
    }
    if extended == assertion.expected:
        return Evaluation(True, observed, f"cloud-init reached {assertion.expected}", command)
    return Evaluation(
        False,
        observed,
        f"cloud-init extended status is {extended!r}, expected {assertion.expected!r}",
        command,
    )


# ----------------------------------------------------------------------------------------------------
def _evaluate_file(assertion: AssertionSpec, transport: GuestTransport) -> Evaluation:
    if not isinstance(assertion, FileAssertion):
        raise TypeError("expected a file assertion")
    wants_content = assertion.content_contains is not None
    args = (assertion.path, "content" if wants_content else "", str(MAX_FILE_CONTENT_BYTES))
    command = CommandReference(script="file", args=args)
    result = transport.execute(FILE_SCRIPT, args, timeout=assertion.timeout_seconds)
    problem = _unobservable(result, "the file")
    if problem is not None or result.exit_status != 0 or result.stdout_dropped_bytes:
        reason = problem or f"file check failed: {_stderr_text(result) or result.exit_status}"
        return Evaluation(False, {"exit_status": result.exit_status}, reason, command)
    header, _, body = result.stdout.partition(b"\n")
    fields = header.decode("ascii", errors="replace").split()
    kind = fields[0] if fields else ""
    if kind == "missing":
        observed: JsonValue = {"exists": False, "type": None, "size": None, "contains": None}
        if assertion.exists:
            return Evaluation(False, observed, f"{assertion.path} does not exist", command)
        return Evaluation(True, observed, f"{assertion.path} does not exist", command)
    if kind != "file":
        observed = {"exists": True, "type": kind, "size": None, "contains": None}
        if not assertion.exists:
            return Evaluation(False, observed, f"{assertion.path} exists", command)
        return Evaluation(False, observed, f"{assertion.path} is a {kind}, not a file", command)
    size = int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else None
    if not assertion.exists:
        observed = {"exists": True, "type": "file", "size": size, "contains": None}
        return Evaluation(False, observed, f"{assertion.path} exists", command)
    if assertion.content_contains is None:
        observed = {"exists": True, "type": "file", "size": size, "contains": None}
        return Evaluation(True, observed, f"{assertion.path} exists", command)
    if size is None or size > MAX_FILE_CONTENT_BYTES:
        observed = {"exists": True, "type": "file", "size": size, "contains": None}
        reason = f"{assertion.path} is larger than the {MAX_FILE_CONTENT_BYTES}-byte content limit"
        return Evaluation(False, observed, reason, command)
    contains = assertion.content_contains.encode("utf-8") in body
    observed = {"exists": True, "type": "file", "size": size, "contains": contains}
    if contains:
        return Evaluation(True, observed, f"{assertion.path} contains the expected text", command)
    return Evaluation(
        False, observed, f"{assertion.path} does not contain the expected text", command
    )


# ----------------------------------------------------------------------------------------------------
def _evaluate_package(assertion: AssertionSpec, transport: GuestTransport) -> Evaluation:
    if not isinstance(assertion, PackageInstalledAssertion):
        raise TypeError("expected a package-installed assertion")
    command = CommandReference(script="package-installed", args=(assertion.name,))
    result = transport.execute(PACKAGE_SCRIPT, (assertion.name,), timeout=assertion.timeout_seconds)
    problem = _unobservable(result, "packages")
    if problem is not None:
        return Evaluation(False, {"exit_status": result.exit_status}, problem, command)
    # A multi-arch package can report one line per installed architecture.
    entries = [
        line.split("\t", 1)
        for line in result.stdout.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if result.exit_status != 0 or not entries:
        observed: JsonValue = {"status": "not-installed", "version": None}
        detail = _stderr_text(result)
        reason = f"{assertion.name} is not installed" + (f": {detail}" if detail else "")
        return Evaluation(False, observed, reason, command)
    installed = [entry for entry in entries if entry[0] == "installed"]
    chosen = installed[0] if installed else entries[0]
    observed = {"status": chosen[0], "version": chosen[1] if len(chosen) > 1 else None}
    if installed:
        return Evaluation(True, observed, f"{assertion.name} is installed", command)
    return Evaluation(
        False, observed, f"{assertion.name} has dpkg status {chosen[0]!r}, not 'installed'", command
    )
