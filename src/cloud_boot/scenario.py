# ----------------------------------------------------------------------------------------------------
# scenario.py
# ----------------------------------------------------------------------------------------------------

"""Safe loading and normalization of cloud-boot scenarios."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node
from yaml.resolver import BaseResolver

from cloud_boot.models import BootPlan, PayloadSpec, ScenarioV1, SeedSpec

__all__ = [
    "MAX_SCENARIO_BYTES",
    "MAX_YAML_ALIASES",
    "MAX_YAML_DEPTH",
    "LoadedScenario",
    "ScenarioIssue",
    "ScenarioValidationError",
    "load_scenario",
]

MAX_SCENARIO_BYTES = 1024 * 1024
MAX_YAML_ALIASES = 50
MAX_YAML_DEPTH = 64
_SECRET_REFERENCE = "${secret:"
_SECRET_UNSUPPORTED = "secret references are not supported in Milestone 0"

LocationPart = str | int


@dataclass(frozen=True, slots=True)
class ScenarioIssue:
    """One location-aware scenario loading or validation problem."""

    location: tuple[LocationPart, ...]
    message: str

    @property
    def field(self) -> str:
        """Render the location as a rooted path such as $.assertions[1].path."""
        rendered = "$"
        for part in self.location:
            rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
        return rendered


class ScenarioValidationError(ValueError):
    """Raised when a scenario cannot be safely loaded and normalized."""

    def __init__(self, issues: list[ScenarioIssue]) -> None:
        if not issues:
            raise ValueError("at least one scenario issue is required")
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.field}: {issue.message}" for issue in self.issues)
        super().__init__(f"invalid scenario: {details}")


@dataclass(frozen=True, slots=True)
class LoadedScenario:
    """Validated source model plus deterministic downstream boot plans."""

    source_path: Path
    scenario: ScenarioV1
    boots: tuple[BootPlan, ...]


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader with duplicate-key detection and an alias ceiling."""

    alias_count: int
    nesting_depth: int

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.alias_count = 0
        self.nesting_depth = 0

    def compose_node(self, parent: Node | None, index: Any) -> Node:
        self.nesting_depth += 1
        if self.nesting_depth > MAX_YAML_DEPTH:
            event = self.peek_event()  # type: ignore[no-untyped-call]
            raise ComposerError(
                None,
                None,
                f"YAML nesting exceeds the {MAX_YAML_DEPTH}-level limit",
                event.start_mark,
            )
        if self.check_event(AliasEvent):
            self.alias_count += 1
            if self.alias_count > MAX_YAML_ALIASES:
                event = self.peek_event()  # type: ignore[no-untyped-call]
                raise ComposerError(
                    None,
                    None,
                    f"more than {MAX_YAML_ALIASES} YAML aliases are not allowed",
                    event.start_mark,
                )
        try:
            return cast(Node, super().compose_node(parent, index))
        finally:
            self.nesting_depth -= 1


def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: MappingNode, deep: bool = False
) -> dict[Hashable, object]:
    loader.flatten_mapping(node)
    mapping: dict[Hashable, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as e:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from e
        hashable_key = cast(Hashable, key)
        if hashable_key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[hashable_key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _yaml_issue(error: yaml.YAMLError) -> ScenarioIssue:
    mark = getattr(error, "problem_mark", None)
    location: tuple[LocationPart, ...] = ("yaml",)
    if mark is not None:
        location += (f"line {mark.line + 1}", f"column {mark.column + 1}")
    problem = getattr(error, "problem", None)
    return ScenarioIssue(location, str(problem or error).strip())


def _find_secret_references(document: object) -> list[ScenarioIssue]:
    issues: list[ScenarioIssue] = []
    active: set[int] = set()  # containers on the current path, to detect recursion
    seen: set[int] = set()  # containers already checked, to skip shared aliases

    def walk(value: object, location: tuple[LocationPart, ...]) -> None:
        if isinstance(value, str):
            if _SECRET_REFERENCE in value:
                issues.append(ScenarioIssue(location, _SECRET_UNSUPPORTED))
            return
        if not isinstance(value, (Mapping, list)):
            return
        identity = id(value)
        if identity in active:
            issues.append(ScenarioIssue(location, "recursive YAML aliases are not allowed"))
            return
        if identity in seen:
            return
        seen.add(identity)
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    key_location = (*location, str(key))
                    if isinstance(key, str) and _SECRET_REFERENCE in key:
                        issues.append(ScenarioIssue(key_location, _SECRET_UNSUPPORTED))
                    walk(item, key_location)
            else:
                for index, item in enumerate(value):
                    walk(item, (*location, index))
        finally:
            active.remove(identity)

    walk(document, ())
    return issues


def _pydantic_issues(error: ValidationError) -> list[ScenarioIssue]:
    return [
        ScenarioIssue(
            tuple(item["loc"]),
            str(item["msg"]),
        )
        for item in error.errors(include_url=False, include_input=False)
    ]


def _payloads(seed: SeedSpec) -> tuple[tuple[str, PayloadSpec], ...]:
    pairs = [("user_data", seed.user_data)]
    if seed.vendor_data is not None:
        pairs.append(("vendor_data", seed.vendor_data))
    if seed.network_config is not None:
        pairs.append(("network_config", seed.network_config))
    return tuple(pairs)


def _validate_payload_containment(
    scenario: ScenarioV1, scenario_directory: Path
) -> list[ScenarioIssue]:
    root = scenario_directory.resolve()
    issues: list[ScenarioIssue] = []

    for field_name, payload in _payloads(scenario.seed):
        if payload.file is None:
            continue
        candidate = (root / payload.file).resolve()
        location = ("seed", field_name, "file")
        if not candidate.is_relative_to(root):
            issues.append(
                ScenarioIssue(location, "file path resolves outside the scenario directory")
            )
        elif not candidate.is_file():
            issues.append(ScenarioIssue(location, "file does not exist or is not a regular file"))
    return issues


def _read_scenario(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            document = stream.read(MAX_SCENARIO_BYTES + 1)
    except OSError as e:
        # The OSError text already repeats the path; keep only the reason.
        message = f"cannot read {path}: {e.strerror or e}"
        raise ScenarioValidationError([ScenarioIssue(("source",), message)]) from e

    if len(document) > MAX_SCENARIO_BYTES:
        raise ScenarioValidationError(
            [
                ScenarioIssue(
                    ("source",),
                    f"scenario exceeds the {MAX_SCENARIO_BYTES}-byte limit",
                )
            ]
        )
    try:
        return document.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ScenarioValidationError(
            [
                ScenarioIssue(
                    ("source",),
                    f"scenario must be UTF-8 (invalid byte at offset {e.start})",
                )
            ]
        ) from e


# ----------------------------------------------------------------------------------------------------
def load_scenario(path: str | Path, environ: Mapping[str, str] | None = None) -> LoadedScenario:
    """Load, validate, and normalize one Milestone 0 scenario.

    ``environ`` is accepted for forward compatibility, but Milestone 0 rejects
    every secret reference and therefore never reads secret values from it.
    """
    del environ
    source_path = Path(path).expanduser().resolve()
    document = _read_scenario(source_path)
    try:
        # SafeLoader subclass adds duplicate-key checks and alias/depth limits.
        raw = yaml.load(document, Loader=_StrictSafeLoader)  # noqa: S506
    except yaml.YAMLError as e:
        raise ScenarioValidationError([_yaml_issue(e)]) from e

    if not isinstance(raw, dict):
        raise ScenarioValidationError([ScenarioIssue((), "document root must be a mapping")])
    if not all(isinstance(key, str) for key in raw):
        raise ScenarioValidationError(
            [ScenarioIssue((), "all top-level field names must be strings")]
        )

    unsupported: list[ScenarioIssue] = []
    if "boots" in raw:
        unsupported.append(
            ScenarioIssue(("boots",), "multi-boot input is not supported in Milestone 0")
        )
    if "fault" in raw:
        unsupported.append(ScenarioIssue(("fault",), "fault input is not supported in Milestone 0"))
    issues = unsupported + _find_secret_references(raw)
    if issues:
        raise ScenarioValidationError(issues)

    try:
        scenario = ScenarioV1.model_validate(raw)
    except ValidationError as e:
        raise ScenarioValidationError(_pydantic_issues(e)) from e

    containment_issues = _validate_payload_containment(scenario, source_path.parent)
    if containment_issues:
        raise ScenarioValidationError(containment_issues)

    return LoadedScenario(
        source_path=source_path,
        scenario=scenario,
        boots=(scenario.to_boot_plan(),),
    )
