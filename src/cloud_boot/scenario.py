# ----------------------------------------------------------------------------------------------------
# scenario.py
# ----------------------------------------------------------------------------------------------------

"""Load a scenario file safely and turn it into validated models.

`load_scenario` reads at most `MAX_SCENARIO_BYTES` of UTF-8, parses it with a strict YAML loader
(no duplicate keys, bounded aliases and nesting), rejects input that Milestone 0 does not support,
validates it against `ScenarioV1`, and checks that payload files stay inside the scenario's
directory. Every problem is reported as a `ScenarioValidationError` carrying the field locations.

`ScenarioValidationError` is deliberately not a `CloudBootError`: validation fails before any run
exists, so there is no report to write, and the CLI prints it and exits with
`ExitCode.VALIDATION`.
"""

# ----------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "MAX_SCENARIO_BYTES",
    "MAX_YAML_ALIASES",
    "MAX_YAML_DEPTH",
    "LoadedScenario",
    "ScenarioIssue",
    "ScenarioValidationError",
    "load_scenario",
]

# Read limit, checked before parsing; one extra byte is read to detect an oversized file.
MAX_SCENARIO_BYTES = 1024 * 1024

# Caps YAML aliases (`*name`). An alias shares one object, so a short document can describe a
# tree exponentially larger than itself ("billion laughs"). The cap bounds that, and code that
# walks the tree must also visit each shared container only once.
MAX_YAML_ALIASES = 50

# PyYAML composes nodes recursively; without a limit, deep nesting raises RecursionError, which
# would surface as an internal error instead of a validation error.
MAX_YAML_DEPTH = 64

# Prefix of the `${secret:NAME}` syntax reserved by the design; Milestone 0 rejects it.
_SECRET_REFERENCE = "${secret:"
_SECRET_UNSUPPORTED = "secret references are not supported in Milestone 0"

# One step of a field location: a mapping key or a list index.
type LocationPart = str | int


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ScenarioIssue:
    """One problem found while loading a scenario, and where it is.

    `location` is the path of mapping keys and list indices to the offending value, as produced
    by Pydantic. The loader adds two pseudo-locations of its own: `("source",)` for problems
    reading the file, and `("yaml", "line N", "column M")` for YAML syntax errors.
    """

    location: tuple[LocationPart, ...]
    message: str

    @property
    def field(self) -> str:
        """The location as a JSONPath-like string, such as `$.assertions[1].path`."""
        rendered = "$"
        for part in self.location:
            rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
        return rendered


# ----------------------------------------------------------------------------------------------------
class ScenarioValidationError(ValueError):
    """The scenario cannot be loaded; carries every issue found, not just the first.

    The message joins all issues as `invalid scenario: <field>: <message>; ...`, and `issues`
    keeps them as structured values for callers that need more than the text.

    Raises:
        ValueError: If `issues` is empty, which would be a loader bug.
    """

    def __init__(self, issues: list[ScenarioIssue]) -> None:
        if not issues:
            raise ValueError("at least one scenario issue is required")
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.field}: {issue.message}" for issue in self.issues)
        super().__init__(f"invalid scenario: {details}")


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LoadedScenario:
    """A validated scenario, where it came from, and the boot plans to run.

    `source_path` is absolute; the report hashes the file it points to, and payload files are
    resolved relative to its directory. `boots` holds one plan per boot, in order, and always
    exactly one in Milestone 0.
    """

    source_path: Path
    scenario: ScenarioV1
    boots: tuple[BootPlan, ...]


# ----------------------------------------------------------------------------------------------------
class _StrictSafeLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` with the extra checks a scenario needs.

    SafeLoader already refuses to build arbitrary Python objects. This subclass also rejects
    duplicate mapping keys (see `_construct_unique_mapping`), more than `MAX_YAML_ALIASES`
    aliases, and nesting deeper than `MAX_YAML_DEPTH`. The limits raise `ComposerError` at the
    offending node, so they are reported as YAML issues with a line and column.

    The counters are per document; `yaml.load` creates a new loader for each one.
    """

    alias_count: int
    nesting_depth: int

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.alias_count = 0
        self.nesting_depth = 0

    def compose_node(self, parent: Node | None, index: Any) -> Node:
        """Count nesting and aliases, then compose the node as SafeLoader does.

        PyYAML calls this once per node and recurses into children, so `nesting_depth` is the
        depth of the node being composed, with the root at 1. An alias reuses a node composed
        earlier instead of recursing, so aliases are counted separately.
        """
        self.nesting_depth += 1
        if self.nesting_depth > MAX_YAML_DEPTH:
            # PyYAML's type stubs leave peek_event unannotated.
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
                # PyYAML's type stubs leave peek_event unannotated.
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


# ----------------------------------------------------------------------------------------------------
def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: MappingNode, deep: bool = False
) -> dict[Hashable, object]:
    """Build a mapping as SafeLoader does, but reject duplicate and unhashable keys.

    PyYAML normally keeps the last of two equal keys, so a repeated `name:` would silently
    replace the first. `flatten_mapping` expands YAML merge keys (`<<: *base`) first, and
    merged keys count as present: overriding one in the same mapping is reported as a
    duplicate. That is stricter than YAML's merge rules, and it keeps every value explicit.

    The mapping is built in one pass, so a mapping that contains itself through an alias fails
    with PyYAML's "unconstructable recursive node" error. Lists still use SafeLoader's
    constructor and can contain themselves; `_find_secret_references` reports those.
    """
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


# Every plain YAML mapping, including the document root, goes through the strict constructor.
_StrictSafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


# ----------------------------------------------------------------------------------------------------
def _yaml_issue(error: yaml.YAMLError) -> ScenarioIssue:
    """Turn a PyYAML error into an issue at `$.yaml`, with a 1-based line and column if known.

    Only the error's one-line `problem` is kept; PyYAML's full message adds context lines and a
    source excerpt. Errors without a position, such as a `ReaderError` for a control
    character, keep their whole message instead.
    """
    mark = getattr(error, "problem_mark", None)
    location: tuple[LocationPart, ...] = ("yaml",)
    if mark is not None:
        location += (f"line {mark.line + 1}", f"column {mark.column + 1}")
    problem = getattr(error, "problem", None)
    return ScenarioIssue(location, str(problem or error).strip())


# ----------------------------------------------------------------------------------------------------
def _find_secret_references(document: object) -> list[ScenarioIssue]:
    """Report every `${secret:...}` in the document's keys or values; Milestone 0 has none.

    This runs before schema validation because a secret in a free-text field such as
    `user_data.inline` would otherwise pass validation and reach the guest unexpanded. It also
    reports lists that contain themselves through an alias.

    Each container is checked once, even when aliases share it, so the work grows with the
    document rather than with its expanded tree, and a secret in a shared container is
    reported at its first location only. Only the dicts and lists that plain YAML produces are
    walked; tagged collections such as `!!set` are skipped, and no schema field accepts them.
    """
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


# ----------------------------------------------------------------------------------------------------
def _pydantic_issues(error: ValidationError) -> list[ScenarioIssue]:
    """Turn each Pydantic error into an issue at the same location.

    Input values and documentation URLs are left out: a value may be a whole `user_data`
    document, and the URLs are aimed at developers rather than scenario authors.
    """
    return [
        ScenarioIssue(
            tuple(item["loc"]),
            str(item["msg"]),
        )
        for item in error.errors(include_url=False, include_input=False)
    ]


# ----------------------------------------------------------------------------------------------------
def _payloads(seed: SeedSpec) -> tuple[tuple[str, PayloadSpec], ...]:
    """The seed's payloads that are set, each with its field name, in seed order."""
    pairs = [("user_data", seed.user_data)]
    if seed.vendor_data is not None:
        pairs.append(("vendor_data", seed.vendor_data))
    if seed.network_config is not None:
        pairs.append(("network_config", seed.network_config))
    return tuple(pairs)


# ----------------------------------------------------------------------------------------------------
def _validate_payload_containment(
    scenario: ScenarioV1, scenario_directory: Path
) -> list[ScenarioIssue]:
    """Check that each payload file, after resolving symlinks, is a regular file in the directory.

    This is the second of the three containment checks described in
    `cloud_boot.models._validate_relative_file`. It runs at load time so that a missing or
    misplaced file fails validation before any image is downloaded. `cloud_boot.seed` checks
    again when it reads the file, since the file can change in between.
    """
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


# ----------------------------------------------------------------------------------------------------
def _read_scenario(path: Path) -> str:
    """Read the scenario as UTF-8 text, refusing anything over `MAX_SCENARIO_BYTES`.

    At most one byte past the limit is read, so even a huge file or a device such as
    `/dev/zero` is rejected without being read in full.

    Raises:
        ScenarioValidationError: At `$.source` if the file cannot be read, is too large, or is
            not valid UTF-8.
    """
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
    """Load, validate, and normalize one Milestone 0 scenario file.

    The checks run in order, and the first stage that finds problems raises with all of that
    stage's issues: reading the file, parsing YAML, rejecting input Milestone 0 does not support
    (`boots`, `fault`, and `${secret:...}` references), schema validation, and payload file
    checks.

    Args:
        path: The scenario file. It is resolved, following symlinks, so payload files are
            relative to the directory of the real scenario file.
        environ: Reserved for expanding `${secret:NAME}` from `CLOUD_BOOT_SECRET_NAME`
            variables in a later milestone. It is not read now: every secret reference is
            rejected.

    Raises:
        ScenarioValidationError: If any check fails.
    """
    del environ  # Reserved; see the docstring.
    source_path = Path(path).expanduser().resolve()
    document = _read_scenario(source_path)
    try:
        # S506 flags yaml.load because unsafe loaders can build arbitrary objects; this
        # loader subclasses SafeLoader, so it cannot.
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
