# ----------------------------------------------------------------------------------------------------
# models.py
# ----------------------------------------------------------------------------------------------------

"""Validated scenario and boot-plan models."""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

# BaseModel: https://pydantic.dev/docs/validation/latest/api/pydantic/base_model/
# ConfigDict: https://pydantic.dev/docs/validation/latest/api/pydantic/config/
# Field: https://pydantic.dev/docs/validation/latest/api/pydantic/fields/
# StrictBool: https://pydantic.dev/docs/validation/latest/api/pydantic/types/#pydantic.types.StrictBool
# StrictInt: https://pydantic.dev/docs/validation/latest/api/pydantic/types/#pydantic.types.StrictInt
# StrictStr: https://pydantic.dev/docs/validation/latest/api/pydantic/types/#pydantic.types.StrictStr
# field_validator: https://pydantic.dev/docs/validation/latest/concepts/validators/#field-validators
# model_validator: https://pydantic.dev/docs/validation/latest/concepts/validators/#model-validators
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

# A strict string that must be 1-64 characters, start with a lowercase letter or digit,
# and contain only lowercase letters, digits, dots, underscores, or hyphens.
type Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9._-]*$"),
]

# A strict integer that is >= 1 and <= 3600 (one hour) used for timeout fields.
type PositiveTimeout = Annotated[StrictInt, Field(ge=1, le=3600)]


# -----------------------------------------------------------------------------
class StrictModel(BaseModel):
    """
    Base model for user-controlled schema objects.

    All fields are immutable after construction, unknown fields are rejected,
    and default values are validated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# -----------------------------------------------------------------------------
class ImageSpec(StrictModel):
    """
    Allowlisted Ubuntu image identity for Milestone 0.
    """

    release: Literal["noble"] = "noble"
    architecture: Literal["amd64"] = "amd64"
    variant: Literal["server-cloudimg"] = "server-cloudimg"
    source: Literal["official"] = "official"


# ----------------------------------------------------------------------------------------------------
class ResourceSpec(StrictModel):
    """
    Bounded virtual-machine resources.
    """

    cpus: Annotated[StrictInt, Field(ge=1, le=16)] = 2
    memory_mib: Annotated[StrictInt, Field(ge=512, le=32768)] = 2048
    disk_gib: Annotated[StrictInt, Field(ge=4, le=128)] = 8


# ----------------------------------------------------------------------------------------------------
def _validate_relative_file(value: str) -> str:
    """
    Validate that a payload path stays inside the scenario directory.

    The path is checked as both a POSIX path and a Windows path, so a scenario is rejected if either
    platform would read it as absolute or as escaping the directory. This is a text-only check.
    Symlink are resolved later by `cloud_boot.scenario._validate_payload_containment`.

    Args:
        value: A file path from the scenario, relative to the scenario file's directory.

    Returns:
        The unchanged `value` when it is a safe relative path.

    Raises:
        ValueError: If `value` is empty, contains a NUL byte, is absolute, or contains a `..`
        component.
    """
    # NUL bytes truncate paths at the OS level and make open() raise.
    if "\x00" in value:
        raise ValueError("file path must not contain a NUL byte")
    if not value:
        raise ValueError("file path must not be empty")

    # Parse both ways: "C:\x" is relative in POSIX, and "..\x" only shows ".." in Windows.
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)

    if posix_path.is_absolute() or windows_path.is_absolute():
        raise ValueError("file path must be relative to the scenario directory")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise ValueError("file path must not traverse outside the scenario directory")
    return value


# ----------------------------------------------------------------------------------------------------
class PayloadSpec(StrictModel):
    """
    A UTF-8 payload supplied inline or from a scenario-relative file.
    """

    inline: StrictStr | None = None
    file: StrictStr | None = None

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str | None) -> str | None:
        return None if value is None else _validate_relative_file(value)

    @model_validator(mode="after")
    def require_one_source(self) -> PayloadSpec:
        if (self.inline is None) == (self.file is None):
            raise ValueError("exactly one of 'inline' or 'file' is required")
        return self


# ----------------------------------------------------------------------------------------------------
class SeedSpec(StrictModel):
    """
    Inputs used to construct normal NoCloud seed media.
    """

    instance_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    local_hostname: (
        Annotated[
            StrictStr,
            Field(
                min_length=1,
                max_length=253,
                pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$",
            ),
        ]
        | None
    ) = None
    user_data: PayloadSpec
    vendor_data: PayloadSpec | None = None
    network_config: PayloadSpec | None = None


# ----------------------------------------------------------------------------------------------------
class CloudInitState(StrEnum):
    """
    Cloud-init terminal states needed by the scenario contract.
    """

    DONE = "done"
    DEGRADED = "degraded"
    ERROR = "error"
    DISABLED = "disabled"


# ----------------------------------------------------------------------------------------------------
class AssertionBase(StrictModel):
    """
    Fields common to every Milestone 0 assertion.
    """

    id: Identifier
    required: StrictBool = True
    timeout_seconds: Annotated[StrictInt, Field(ge=1, le=300)] = 30


# ----------------------------------------------------------------------------------------------------
class CloudInitStatusAssertion(AssertionBase):
    """
    Require a cloud-init terminal state.
    """

    type: Literal["cloud-init-status"]
    expected: Literal["done"] = "done"


# ----------------------------------------------------------------------------------------------------
class FileAssertion(AssertionBase):
    """
    Require file existence and, optionally, matching content.
    """

    type: Literal["file"]
    path: StrictStr
    exists: StrictBool = True
    content_contains: StrictStr | None = None

    @field_validator("path")
    @classmethod
    def validate_guest_path(cls, value: str) -> str:
        path = PurePosixPath(value)

        if "\x00" in value:
            raise ValueError("guest path must not contain a NUL byte")
        if not path.is_absolute():
            raise ValueError("guest path must be absolute")
        if ".." in path.parts:
            raise ValueError("guest path must not contain parent traversal")
        return value

    @model_validator(mode="after")
    def validate_content_expectation(self) -> FileAssertion:
        if not self.exists and self.content_contains is not None:
            raise ValueError("content_contains cannot be used when exists is false")
        return self


# ----------------------------------------------------------------------------------------------------
class PackageInstalledAssertion(AssertionBase):
    """
    Require an installed Debian package.
    """

    type: Literal["package-installed"]
    name: Annotated[
        StrictStr,
        Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9+.-]*$"),
    ]


# ----------------------------------------------------------------------------------------------------
type AssertionSpec = Annotated[
    CloudInitStatusAssertion | FileAssertion | PackageInstalledAssertion,
    Field(discriminator="type"),
]


# ----------------------------------------------------------------------------------------------------
class BootPlan(StrictModel):
    """
    Normalized internal plan for one VM boot.
    """

    name: Identifier
    seed: SeedSpec
    wait_for_cloud_init: StrictBool = True
    expected_cloud_init_states: frozenset[CloudInitState] = Field(
        default_factory=lambda: frozenset({CloudInitState.DONE}), min_length=1
    )
    assertions: list[AssertionSpec] = Field(default_factory=list, max_length=100)


# ----------------------------------------------------------------------------------------------------
class ScenarioV1(StrictModel):
    """
    Defines the strict single-boot input schema for Milestone 0.

    Attributes:
        scenario_version: Schema version. Must be 1.
        name: Identifier for the scenario.
        description: Optional description of the scenario.
        image: Container image configuration.
        resources: Resource requirements.
        network_mode: Network mode. Must be "nat".
        timeout_seconds: Maximum execution time in seconds.
        seed: Seed configuration.
        assertions: Assertions to evaluate after execution. Assertion IDs
            must be unique.
    """

    scenario_version: Literal[1]
    name: Identifier
    description: Annotated[StrictStr, Field(max_length=4096)] | None = None
    image: ImageSpec = Field(default_factory=ImageSpec)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    network_mode: Literal["nat"] = "nat"
    timeout_seconds: PositiveTimeout = 300
    seed: SeedSpec
    assertions: list[AssertionSpec] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def require_unique_assertion_ids(self) -> ScenarioV1:
        identifiers = [assertion.id for assertion in self.assertions]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError("assertion ids must be unique")
        return self

    def to_boot_plan(self) -> BootPlan:
        """
        Normalize the single-boot shorthand for downstream components.
        """

        return BootPlan(name="boot", seed=self.seed, assertions=self.assertions)


# ----------------------------------------------------------------------------------------------------
__all__ = [
    "AssertionSpec",
    "BootPlan",
    "CloudInitState",
    "CloudInitStatusAssertion",
    "FileAssertion",
    "ImageSpec",
    "PackageInstalledAssertion",
    "PayloadSpec",
    "ResourceSpec",
    "ScenarioV1",
    "SeedSpec",
]
