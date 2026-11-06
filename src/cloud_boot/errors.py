"""Shared error contract for cloud-boot components.

This module holds only the base error, its phases, and the exit-code table, and imports nothing
else from the package. Each component defines its own `CloudBootError` subclasses next to the code
that raises them, so callers can handle every expected failure through one stable shape.
"""

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
    """Stable high-level phases used in diagnostics and reports."""

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
    """Exit-code contract from the approved design."""

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
    """Base class for expected failures.

    The code is a stable machine-readable identifier. The message is concise
    user-facing text; remediation is an optional next action. Context is
    intentionally structured and should contain metadata rather than secrets.
    """

    code: str
    phase: str
    message: str
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
        except ValueError as exc:
            raise ValueError(f"unknown error phase: {phase!r}") from exc
        super().__init__(message)
        self.code = code
        self.phase = normalized_phase.value
        self.message = message
        self.remediation = remediation
        self.context = dict(context or {})
        self.exit_code = int(exit_code)

    def as_dict(self) -> dict[str, object]:
        """Return the stable shape used by reports and CLI adapters."""
        result: dict[str, object] = {
            "code": self.code,
            "phase": self.phase,
            "message": self.message,
            "exit_code": self.exit_code,
            "context": dict(self.context),
        }
        if self.remediation is not None:
            result["remediation"] = self.remediation
        return result
