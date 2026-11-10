# ----------------------------------------------------------------------------------------------------
# errors.py
# ----------------------------------------------------------------------------------------------------

"""Shared error contract for cloud-boot components.

This module holds only the base error, its phases, and the exit-code table, and imports nothing
else from the package. Each component defines its own `CloudBootError` subclasses next to the code
that raises them, so callers can handle every expected failure through one stable shape.
"""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum, StrEnum

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "CloudBootError",
    "ErrorPhase",
    "ExitCode",
]


# ----------------------------------------------------------------------------------------------------
class ErrorPhase(StrEnum):
    """Where in a run a failure happened.

    `ErrorPhase` tells a reader where to look; it does not choose the exit code. The two are
    independent on purpose: a timeout can happen in any phase but always exits with
    `ExitCode.TIMEOUT`, and a `PROCESS` failure exits with `HOST_UNAVAILABLE` when a command
    cannot start or `TIMEOUT` when it runs out of time. `INTERNAL` marks a cloud-boot bug and is
    the default for an unclassified error.

    A failed assertion is a run outcome, not an error, so nothing raises `ASSERTION` yet.
    Scenario validation fails before any run exists and is reported by the CLI directly.

    Values appear in `report.json` and on stderr as `ERROR <phase>: ...`, so renaming one is a
    breaking change to the report schema.
    """

    VALIDATION = "validation"
    HOST = "host"
    PROCESS = "process"
    PREPARATION = "preparation"
    VM = "vm"
    SSH = "ssh"
    ASSERTION = "assertion"
    ARTIFACT = "artifact"
    INTERNAL = "internal"


# ----------------------------------------------------------------------------------------------------
class ExitCode(IntEnum):
    """Process exit codes; the public contract scripts and CI rely on.

    Codes 3-9 follow the order of a run, so a larger code means the run got further. Code 1 is
    left unused so that a Python crash outside cloud-boot's own handling stays distinguishable.
    Signal codes follow the shell convention of 128 plus the signal number.

    Only `SIGINT` is produced today (Ctrl-C, after normal cleanup). `SIGHUP` and `SIGTERM` are
    reserved: their handlers do not exist yet, so those signals currently end the process
    without cleanup.
    """

    SUCCESS = 0
    VALIDATION = 2
    HOST_UNAVAILABLE = 3
    PREPARATION = 4
    VM = 5
    SSH = 6
    ASSERTION = 7
    TIMEOUT = 8
    REPORT = 9
    INTERNAL = 10
    SIGHUP = 129
    SIGINT = 130
    SIGTERM = 143


# ----------------------------------------------------------------------------------------------------
class CloudBootError(Exception):
    """Base class for expected failures: anything cloud-boot anticipates and can report.

    Each component subclasses this next to the code that raises it and sets its own `code`,
    `phase`, and `exit_code`. Unexpected exceptions are not subclasses; the coordinator wraps
    them as an internal error instead.

    Args:
        message: Short user-facing text, printed as `ERROR <phase>: <message>`.
        code: Stable kebab-case identifier that scripts can match instead of parsing `message`.
        phase: Where the failure happened; must be an `ErrorPhase` value. An unknown value is a
            programming error and raises `ValueError`.
        remediation: Optional next step for the user, printed as `HINT <remediation>`.
        context: Structured diagnostics for `report.json`. It is copied on construction and
            exposed as a read-only `Mapping`, so code that catches the error cannot edit it. Keep
            it to metadata such as paths, URLs, and return codes, plus bounded tails of command
            output. Never put secrets here: it is written to disk unredacted.
        exit_code: Process exit code for this failure.

    The defaults all mean "internal", so an unclassified error is reported as a cloud-boot bug.
    """

    message: str
    code: str
    phase: str
    remediation: str | None
    context: Mapping[str, object]
    exit_code: int

    def __init__(
        self,
        message: str,
        *,
        code: str = "internal-error",
        phase: str | ErrorPhase = ErrorPhase.INTERNAL,
        remediation: str | None = None,
        context: Mapping[str, object] | None = None,
        exit_code: int | ExitCode = ExitCode.INTERNAL,
    ) -> None:
        try:
            normalized_phase = ErrorPhase(phase)
        except ValueError as e:
            raise ValueError(f"unknown error phase: {phase!r}") from e
        super().__init__(message)
        self.message = message
        self.code = code
        self.phase = normalized_phase.value
        self.remediation = remediation
        self.context = dict(context or {})
        self.exit_code = int(exit_code)
