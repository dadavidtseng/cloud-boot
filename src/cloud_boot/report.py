# ----------------------------------------------------------------------------------------------------
# report.py
# ----------------------------------------------------------------------------------------------------

"""The minimal versioned run report and the single source of a run's exit code."""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue, field_validator, model_validator

from .artifacts import RunWorkspace
from .assertions import AssertionResult, boot_passed
from .errors import CloudBootError, ExitCode
from .image_store import VerifiedImage
from .scenario import MAX_SCENARIO_BYTES, LoadedScenario
from .seed import SeedArtifact

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "REPORT_FILENAME",
    "BootResult",
    "FailureRecord",
    "ImageIdentity",
    "Outcome",
    "RunReportV1",
    "ScenarioIdentity",
    "decide_outcome",
    "image_identity",
    "scenario_identity",
    "seed_digests",
    "write_report",
]

# ----------------------------------------------------------------------------------------------------
REPORT_FILENAME = "report.json"

# passed: every required assertion passed; failed: one was failed or blocked (exit 7);
# error: a CloudBootError ended the run with its own exit code.
type Outcome = Literal["passed", "failed", "error"]


# ----------------------------------------------------------------------------------------------------
class _ReportModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ----------------------------------------------------------------------------------------------------
class ScenarioIdentity(_ReportModel):
    """Which scenario ran, and the digest of the file as read when the report was built."""

    name: str
    source_path: str
    source_sha256: str | None


# ----------------------------------------------------------------------------------------------------
class ImageIdentity(_ReportModel):
    """The base image the guest booted from, with its honest verification level."""

    digest: str
    size_bytes: int
    source_url: str
    checksum_verified: bool
    signature_authenticated: bool


# ----------------------------------------------------------------------------------------------------
class BootResult(_ReportModel):
    """What happened during one boot plan."""

    name: str
    index: int
    instance_id: str | None
    seed_digests: dict[str, str]
    stop_reason: str | None
    assertions: list[AssertionResult]


# ----------------------------------------------------------------------------------------------------
class FailureRecord(_ReportModel):
    """The `CloudBootError` that ended the run, in its stable serialized shape."""

    code: str
    phase: str
    message: str
    remediation: str | None
    exit_code: int
    context: dict[str, JsonValue]

    @classmethod
    def from_error(cls, error: CloudBootError) -> FailureRecord:
        """Build a record whose context holds only JSON values; anything else becomes text."""
        return cls(
            code=error.code,
            phase=error.phase,
            message=error.message,
            remediation=error.remediation,
            exit_code=error.exit_code,
            context={key: _json_value(value) for key, value in error.context.items()},
        )


# ----------------------------------------------------------------------------------------------------
class RunReportV1(_ReportModel):
    """The minimal version-1 run report written as `report.json`.

    Construction fails if the outcome, exit code, failure, and assertion results disagree, so an
    inconsistent report can never be written.
    """

    report_version: Literal[1] = 1
    run_id: str
    scenario: ScenarioIdentity
    image: ImageIdentity | None
    started_at: AwareDatetime
    finished_at: AwareDatetime
    duration_seconds: float
    outcome: Outcome
    exit_code: int
    boots: list[BootResult]
    artifacts: list[str]
    warnings: list[str]
    failure: FailureRecord | None

    @field_validator("duration_seconds")
    @classmethod
    def require_finite_duration(cls, value: float) -> float:
        """Reject negative, infinite, or NaN durations."""
        if not math.isfinite(value) or value < 0:
            raise ValueError("duration must be finite and non-negative")
        return value

    @field_validator("artifacts")
    @classmethod
    def require_relative_artifacts(cls, paths: list[str]) -> list[str]:
        """Keep artifact references portable: run-relative, never absolute or traversing."""
        for path in paths:
            pure = PurePosixPath(path)
            if not path or pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"artifact paths must be run-relative: {path!r}")
        return paths

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> RunReportV1:
        """Tie outcome, exit code, failure, and assertion results together."""
        all_assertions = [result for boot in self.boots for result in boot.assertions]
        if self.failure is not None:
            if self.outcome != "error" or self.exit_code != self.failure.exit_code:
                raise ValueError("a failed run must report outcome 'error' and the failure exit code")
        elif self.outcome == "error":
            raise ValueError("outcome 'error' requires a failure record")
        elif self.outcome == "passed":
            if self.exit_code != ExitCode.SUCCESS or not boot_passed(all_assertions):
                raise ValueError("outcome 'passed' requires exit code 0 and passing assertions")
        elif self.exit_code != ExitCode.ASSERTION or boot_passed(all_assertions):
            raise ValueError(
                "outcome 'failed' requires exit code 7 and a failed or blocked assertion"
            )
        return self


# ----------------------------------------------------------------------------------------------------
def decide_outcome(
    *, failure: CloudBootError | None, assertions: Sequence[AssertionResult]
) -> tuple[Outcome, int]:
    """Return the run outcome and exit code; the CLI and the report must both use this result.

    The most specific terminal condition wins: an error's own exit code beats assertion results.
    """
    if failure is not None:
        return "error", failure.exit_code
    if boot_passed(assertions):
        return "passed", int(ExitCode.SUCCESS)
    return "failed", int(ExitCode.ASSERTION)


# ----------------------------------------------------------------------------------------------------
def write_report(workspace: RunWorkspace, report: RunReportV1) -> Path:
    """Write `report.json` atomically.

    Raises:
        ArtifactError: If the report cannot be persisted; the run must then exit 9, never 0.
    """
    return workspace.write_text_atomic(REPORT_FILENAME, report.model_dump_json(indent=2) + "\n")


# ----------------------------------------------------------------------------------------------------
def scenario_identity(loaded: LoadedScenario) -> ScenarioIdentity:
    """Identify the scenario by name, path, and the SHA-256 of its file (None if unreadable)."""
    try:
        with loaded.source_path.open("rb") as stream:
            data = stream.read(MAX_SCENARIO_BYTES + 1)
        digest: str | None = hashlib.sha256(data).hexdigest()
    except OSError:
        digest = None
    return ScenarioIdentity(
        name=loaded.scenario.name, source_path=str(loaded.source_path), source_sha256=digest
    )


# ----------------------------------------------------------------------------------------------------
def image_identity(image: VerifiedImage) -> ImageIdentity:
    """Copy the verified image's identity, including that its signature was not authenticated."""
    return ImageIdentity(
        digest=image.digest,
        size_bytes=image.size_bytes,
        source_url=image.source_url,
        checksum_verified=image.checksum_verified,
        signature_authenticated=image.signature_authenticated,
    )


# ----------------------------------------------------------------------------------------------------
def seed_digests(seed: SeedArtifact) -> dict[str, str]:
    """Map each NoCloud file on the seed media to its SHA-256."""
    return {entry.name: entry.sha256 for entry in seed.files}


# ----------------------------------------------------------------------------------------------------
def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
