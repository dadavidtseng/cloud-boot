# ----------------------------------------------------------------------------------------------------
# models.py
# ----------------------------------------------------------------------------------------------------

"""The scenario schema: what a valid scenario YAML file looks like, as Pydantic models.

These models only describe and validate data; `scenario.py` reads the YAML and payload files.
Every field uses a strict type, so a value of the wrong type is rejected instead of coerced:
YAML and Pydantic would otherwise silently turn `"2"` into 2 or `yes` into True.
`ScenarioV1` is the user-facing schema and `BootPlan` the normalized form the runner consumes.
"""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

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

# Names that may end up in paths, QEMU arguments, or reports, such as scenario, boot, and
# assertion IDs: 1-64 characters, starting with a lowercase letter or digit, then only lowercase
# letters, digits, dots, underscores, or hyphens.
type Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9._-]*$"),
]

# The whole run's time limit: 1 second to 1 hour. Assertions have their own, shorter limit.
type PositiveTimeout = Annotated[StrictInt, Field(ge=1, le=3600)]

# A DNS host name (RFC 1123): dot-separated labels of 1-63 letters, digits, or hyphens, each
# starting and ending with a letter or digit. The 253-character total is checked separately.
_HOSTNAME_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_HOSTNAME_PATTERN = rf"^{_HOSTNAME_LABEL}(?:\.{_HOSTNAME_LABEL})*$"


# ----------------------------------------------------------------------------------------------------
class StrictModel(BaseModel):
    """Base for every scenario model: unknown keys rejected, frozen, defaults validated.

    `extra="forbid"` turns a misspelled key such as `timeout_second` into an error instead of a
    silently ignored setting.
    `frozen=True` keeps a validated scenario from changing while it is passed between components.
    `validate_default=True` checks defaults against the same constraints as user values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# ----------------------------------------------------------------------------------------------------
class ImageSpec(StrictModel):
    """Which Ubuntu cloud image to boot; only the official Noble amd64 server image for now.

    Each field allows a single value today, so the schema already has a place for future
    releases and architectures: widening a `Literal` keeps existing scenario files valid. The
    fields form the image file name, such as `noble-server-cloudimg-amd64.img`, and `source`
    is reserved for non-official images. `ImageStore` re-checks the values, since it can be
    called without going through scenario validation.
    """

    release: Literal["noble"] = "noble"
    architecture: Literal["amd64"] = "amd64"
    variant: Literal["server-cloudimg"] = "server-cloudimg"
    source: Literal["official"] = "official"


# ----------------------------------------------------------------------------------------------------
class ResourceSpec(StrictModel):
    """Virtual hardware for the guest, each value bounded to catch typos before QEMU sees them.

    `cpus` and `memory_mib` are passed to QEMU as the virtual CPU count (`-smp`) and memory size
    in MiB (`-m`). `disk_gib` is the virtual size of the qcow2 overlay, which is sparse, so it
    costs host space only as the guest writes. Its minimum of 4 sits just above the 3.5 GiB
    virtual size of the Noble cloud image, which an overlay must not be smaller than.
    """

    cpus: Annotated[StrictInt, Field(ge=1, le=16)] = 2
    memory_mib: Annotated[StrictInt, Field(ge=512, le=32768)] = 2048
    disk_gib: Annotated[StrictInt, Field(ge=4, le=128)] = 8


# ----------------------------------------------------------------------------------------------------
def _validate_relative_file(value: str) -> str:
    """Validate that a payload path stays inside the scenario directory.

    The path is checked as both a POSIX path and a Windows path, so a scenario is rejected if either
    platform would read it as absolute or as escaping the directory. This text-only check is the
    first of three: `cloud_boot.scenario._validate_payload_containment` resolves symlinks when the
    scenario loads, and `cloud_boot.seed._read_payload_file` checks again when the file is read,
    since it may have changed in between.

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
    """One cloud-init input (user-data, vendor-data, or network-config), inline or from a file.

    Exactly one of `inline` and `file` must be set. Inline text is encoded as UTF-8; a file's
    bytes are passed through unchanged, so they reach cloud-init exactly as written. `file` is
    relative to the scenario file's directory and must stay inside it. Either way, a payload is
    limited to `cloud_boot.seed.MAX_PAYLOAD_BYTES` (16 MiB).
    """

    inline: StrictStr | None = None
    file: StrictStr | None = None

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str | None) -> str | None:
        """Reject absolute or traversing file paths when a source is provided."""
        return None if value is None else _validate_relative_file(value)

    @model_validator(mode="after")
    def require_one_source(self) -> PayloadSpec:
        """Raise ValueError unless exactly one payload source is provided."""
        # True when both are missing or both are set; exactly one is required.
        if (self.inline is None) == (self.file is None):
            raise ValueError("exactly one of 'inline' or 'file' is required")
        return self


# ----------------------------------------------------------------------------------------------------
class SeedSpec(StrictModel):
    """What goes on the NoCloud seed disk that cloud-init reads at boot.

    cloud-boot writes `meta-data` itself from `instance_id`, `local_hostname`, and the run's SSH
    public key, so the run's own key is always installed. `user_data` is required. A missing
    `vendor_data` becomes an empty file, which cloud-init reads as no vendor configuration, and
    a missing `network_config` leaves the file off the disk, so cloud-init applies its default
    networking.

    `instance_id` is how cloud-init tells instances apart: modules that run once per instance
    run again only when it changes. `local_hostname` must be a valid DNS name.
    """

    instance_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    local_hostname: (
        Annotated[
            StrictStr,
            Field(
                min_length=1,
                max_length=253,
                pattern=_HOSTNAME_PATTERN,
            ),
        ]
        | None
    ) = None
    user_data: PayloadSpec
    vendor_data: PayloadSpec | None = None
    network_config: PayloadSpec | None = None


# ----------------------------------------------------------------------------------------------------
class CloudInitState(StrEnum):
    """Terminal cloud-init states a boot may expect; used only by `BootPlan`, not yet read."""

    DONE = "done"
    DEGRADED = "degraded"
    ERROR = "error"
    DISABLED = "disabled"


# ----------------------------------------------------------------------------------------------------
class AssertionBase(StrictModel):
    """Fields every assertion has: its ID, whether it counts, and its own time limit.

    An assertion with `required: false` is still run and reported, but its failure does not fail
    the run. `timeout_seconds` bounds this one check on the guest; the run's own limit still
    applies.
    """

    id: Identifier
    required: StrictBool = True
    timeout_seconds: Annotated[StrictInt, Field(ge=1, le=300)] = 30


# ----------------------------------------------------------------------------------------------------
class CloudInitStatusAssertion(AssertionBase):
    """Require cloud-init to have finished cleanly: its extended status must be exactly `done`.

    A run that finished with warnings reports `degraded done` and fails this check. `expected`
    allows only `done` today; the field leaves room for other terminal states later.
    """

    type: Literal["cloud-init-status"]
    expected: Literal["done"] = "done"


# ----------------------------------------------------------------------------------------------------
class FileAssertion(AssertionBase):
    """Require a guest file to exist, optionally containing some text, or to be absent.

    `content_contains` is a plain substring match on the file's bytes and only works for files up
    to `cloud_boot.assertions.MAX_FILE_CONTENT_BYTES` (512 KiB); a larger file fails the check.
    """

    type: Literal["file"]
    path: StrictStr
    exists: StrictBool = True
    content_contains: StrictStr | None = None

    @field_validator("path")
    @classmethod
    def validate_guest_path(cls, value: str) -> str:
        """Require an absolute guest path without NUL bytes or `..` components.

        Rejecting `..` keeps the reported path the one that was checked. It is not a security
        boundary: checks run with sudo and may read any guest file.
        """
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
        """Reject content matching when the file is expected to be absent."""
        if not self.exists and self.content_contains is not None:
            raise ValueError("content_contains cannot be used when exists is false")
        return self


# ----------------------------------------------------------------------------------------------------
class PackageInstalledAssertion(AssertionBase):
    """Require a Debian package to be installed on the guest, as reported by dpkg.

    `name` follows Debian Policy: at least two characters of lowercase letters, digits, `+`, `-`,
    and `.`, starting with a letter or digit.
    """

    type: Literal["package-installed"]
    name: Annotated[
        StrictStr,
        Field(min_length=2, max_length=128, pattern=r"^[a-z0-9][a-z0-9+.-]*$"),
    ]


# ----------------------------------------------------------------------------------------------------
# One scenario assertion. Pydantic picks the class from the `type` field, so a mistake is
# reported against that one class instead of against every possible assertion type.
type AssertionSpec = Annotated[
    CloudInitStatusAssertion | FileAssertion | PackageInstalledAssertion,
    Field(discriminator="type"),
]


# ----------------------------------------------------------------------------------------------------
class BootPlan(StrictModel):
    """One boot of the guest, in the normalized form every downstream component consumes.

    `ScenarioV1.to_boot_plan` builds it from the single-boot scenario. Scenarios with several
    boots will produce one plan per boot, so components already handle plans rather than raw
    scenarios. `name` names the boot's artifact directory, such as `boots/001-boot`.

    `wait_for_cloud_init` and `expected_cloud_init_states` are reserved for multi-boot
    scenarios. Nothing reads them yet: the runner always waits for cloud-init, and the
    cloud-init status assertion decides what counts as success.
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
    """A version 1 scenario file: one VM boot, its NoCloud seed, and the checks to run.

    Attributes:
        scenario_version: Schema version; must be 1.
        name: Scenario identifier, also used as the QEMU VM name.
        description: Free text for readers; no code reads it.
        image: Which Ubuntu cloud image to boot (a VM disk image, not a container image).
        resources: Virtual CPU, memory, and disk for the guest.
        network_mode: Guest networking; only `nat` (QEMU user networking). It is not yet passed
            to the VM backend, which also defaults to `nat`.
        timeout_seconds: Limit for the whole run: boot, waiting for cloud-init, and assertions.
        seed: Contents of the NoCloud seed disk.
        assertions: Checks run on the guest after cloud-init finishes. IDs must be unique.
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
        """Reject repeated assertion IDs; reports identify each result by its ID."""
        identifiers = [assertion.id for assertion in self.assertions]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError("assertion ids must be unique")
        return self

    def to_boot_plan(self) -> BootPlan:
        """Convert this single-boot scenario into its one `BootPlan`, named `boot`."""
        return BootPlan(name="boot", seed=self.seed, assertions=self.assertions)
