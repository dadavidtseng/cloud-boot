"""Tests for strict Milestone 0 scenario loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from cloud_boot.models import (
    CloudInitStatusAssertion,
    FileAssertion,
    PackageInstalledAssertion,
)
from cloud_boot.scenario import (
    MAX_SCENARIO_BYTES,
    MAX_YAML_ALIASES,
    MAX_YAML_DEPTH,
    ScenarioValidationError,
    load_scenario,
)

VALID_SCENARIO = """\
scenario_version: 1
name: basic-boot
seed:
  instance_id: cloud-boot-basic-001
  local_hostname: basic-boot
  user_data:
    inline: |
      #cloud-config
      write_files:
        - path: /var/tmp/boot-marker
          content: boot-ok
assertions:
  - id: cloud-init-done
    type: cloud-init-status
    expected: done
  - id: marker-created
    type: file
    path: /var/tmp/boot-marker
    exists: true
    content_contains: boot-ok
  - id: ssh-installed
    type: package-installed
    name: openssh-server
"""


def write_scenario(tmp_path: Path, content: str = VALID_SCENARIO) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_and_normalizes_valid_single_boot(tmp_path: Path) -> None:
    source = write_scenario(tmp_path)

    loaded = load_scenario(source)

    assert loaded.source_path == source.resolve()
    assert loaded.scenario.name == "basic-boot"
    assert loaded.scenario.image.release == "noble"
    assert loaded.scenario.image.architecture == "amd64"
    assert loaded.scenario.resources.cpus == 2
    assert loaded.scenario.resources.memory_mib == 2048
    assert loaded.scenario.resources.disk_gib == 8
    assert len(loaded.boots) == 1
    assert loaded.boots[0].name == "boot"
    assert loaded.boots[0].seed is loaded.scenario.seed
    assert loaded.boots[0].assertions == loaded.scenario.assertions
    assert isinstance(loaded.boots[0].assertions[0], CloudInitStatusAssertion)
    assert isinstance(loaded.boots[0].assertions[1], FileAssertion)
    assert isinstance(loaded.boots[0].assertions[2], PackageInstalledAssertion)


def test_accepts_scenario_relative_payload_file(tmp_path: Path) -> None:
    (tmp_path / "payloads").mkdir()
    (tmp_path / "payloads" / "user-data.yaml").write_text(
        "#cloud-config\n{}\n", encoding="utf-8"
    )
    source = write_scenario(
        tmp_path,
        """\
scenario_version: 1
name: file-payload
seed:
  instance_id: file-payload-001
  user_data:
    file: payloads/user-data.yaml
""",
    )

    loaded = load_scenario(source)

    assert loaded.scenario.seed.user_data.file == "payloads/user-data.yaml"


@pytest.mark.parametrize("assertions", ["", "assertions: []\n"])
def test_accepts_scenario_without_assertions(tmp_path: Path, assertions: str) -> None:
    source = write_scenario(
        tmp_path,
        """\
scenario_version: 1
name: no-assertions
seed:
  instance_id: no-assertions-001
  user_data:
    inline: |
      #cloud-config
      {}
"""
        + assertions,
    )

    loaded = load_scenario(source)

    assert loaded.scenario.assertions == []
    assert loaded.boots[0].assertions == []


@pytest.mark.parametrize(
    ("fragment", "location"),
    [
        ("unknown: value\n", "$.unknown"),
        ("timeout_seconds: 0\n", "$.timeout_seconds"),
        ("network_mode: restricted\n", "$.network_mode"),
    ],
)
def test_reports_invalid_field_location(
    tmp_path: Path, fragment: str, location: str
) -> None:
    source = write_scenario(tmp_path, VALID_SCENARIO + fragment)

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source)

    assert location in str(raised.value)


def test_rejects_duplicate_mapping_keys_with_source_location(tmp_path: Path) -> None:
    source = write_scenario(tmp_path, VALID_SCENARIO + "name: duplicate\n")

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source)

    message = str(raised.value)
    assert "duplicate key 'name'" in message
    assert "line" in message
    assert "column" in message


def test_rejects_malformed_yaml_with_source_location(tmp_path: Path) -> None:
    source = write_scenario(tmp_path, "scenario_version: 1\nseed: [\n")

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source)

    assert "$.yaml" in str(raised.value)
    assert "line" in str(raised.value)


def test_rejects_non_mapping_document(tmp_path: Path) -> None:
    source = write_scenario(tmp_path, "- scenario_version\n- 1\n")

    with pytest.raises(
        ScenarioValidationError, match="document root must be a mapping"
    ):
        load_scenario(source)


def test_rejects_non_utf8_document(tmp_path: Path) -> None:
    source = tmp_path / "scenario.yaml"
    source.write_bytes(b"scenario_version: 1\nname: \xff\n")

    with pytest.raises(ScenarioValidationError, match="must be UTF-8"):
        load_scenario(source)


def test_reports_nonexistent_source_path(tmp_path: Path) -> None:
    source = tmp_path / "missing.yaml"

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source)

    assert raised.value.issues[0].location == ("source",)
    assert f"cannot read {source.resolve()}" in str(raised.value)


def test_rejects_oversized_document_before_parsing(tmp_path: Path) -> None:
    source = tmp_path / "scenario.yaml"
    source.write_bytes(b"x" * (MAX_SCENARIO_BYTES + 1))

    with pytest.raises(ScenarioValidationError, match=r"exceeds the .*byte limit"):
        load_scenario(source)


def test_rejects_excessive_yaml_aliases(tmp_path: Path) -> None:
    aliases = ", ".join("*seed" for _ in range(MAX_YAML_ALIASES + 1))
    source = write_scenario(tmp_path, f"seed: &seed value\nitems: [{aliases}]\n")

    with pytest.raises(ScenarioValidationError, match="YAML aliases"):
        load_scenario(source)


def test_rejects_recursive_yaml_alias(tmp_path: Path) -> None:
    source = write_scenario(tmp_path, "scenario: &scenario [*scenario]\n")

    with pytest.raises(ScenarioValidationError, match="recursive YAML aliases"):
        load_scenario(source)


def test_rejects_excessive_yaml_nesting(tmp_path: Path) -> None:
    source = write_scenario(
        tmp_path,
        "value: " + ("[" * (MAX_YAML_DEPTH + 1)) + ("]" * (MAX_YAML_DEPTH + 1)),
    )

    with pytest.raises(ScenarioValidationError, match="YAML nesting"):
        load_scenario(source)


def test_rejects_secret_reference_without_reading_environment(tmp_path: Path) -> None:
    source = write_scenario(
        tmp_path,
        VALID_SCENARIO.replace("boot-ok", "${secret:BOOT_TOKEN}"),
    )

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source, {"CLOUD_BOOT_SECRET_BOOT_TOKEN": "not-read"})

    assert "$.seed.user_data.inline" in str(raised.value)
    assert "not supported in Milestone 0" in str(raised.value)


@pytest.mark.parametrize(
    "file_path", ["../user-data", "/tmp/user-data", "C:\\user-data"]
)
def test_rejects_payload_path_traversal(tmp_path: Path, file_path: str) -> None:
    source = write_scenario(
        tmp_path,
        f"""\
scenario_version: 1
name: unsafe-path
seed:
  instance_id: unsafe-path-001
  user_data:
    file: {file_path!r}
""",
    )

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source)

    assert "$.seed.user_data.file" in str(raised.value)


def test_rejects_symlink_that_resolves_outside_scenario_directory(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-user-data"
    outside.write_text("#cloud-config\n{}\n", encoding="utf-8")
    link = tmp_path / "user-data-link"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    source = write_scenario(
        tmp_path,
        """\
scenario_version: 1
name: symlink-escape
seed:
  instance_id: symlink-escape-001
  user_data:
    file: user-data-link
""",
    )

    with pytest.raises(ScenarioValidationError, match="resolves outside"):
        load_scenario(source)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("boots: []", "multi-boot input is not supported"),
        ("fault: {type: missing-meta-data}", "fault input is not supported"),
    ],
)
def test_rejects_features_deferred_beyond_milestone_zero(
    tmp_path: Path, field: str, message: str
) -> None:
    source = write_scenario(tmp_path, VALID_SCENARIO + field + "\n")

    with pytest.raises(ScenarioValidationError, match=message):
        load_scenario(source)


def test_rejects_unsupported_assertion_with_precise_location(tmp_path: Path) -> None:
    source = write_scenario(
        tmp_path,
        VALID_SCENARIO.replace("type: package-installed", "type: service"),
    )

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source)

    assert "$.assertions[2]" in str(raised.value)
    assert "union_tag_invalid" not in str(raised.value)


def test_rejects_payload_with_both_inline_and_file(tmp_path: Path) -> None:
    source = write_scenario(
        tmp_path,
        VALID_SCENARIO.replace(
            "    inline: |",
            "    file: user-data.yaml\n    inline: |",
        ),
    )

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source)

    assert "$.seed.user_data" in str(raised.value)
    assert "exactly one" in str(raised.value)


def test_rejects_duplicate_assertion_ids(tmp_path: Path) -> None:
    source = write_scenario(
        tmp_path,
        VALID_SCENARIO.replace("id: marker-created", "id: cloud-init-done"),
    )

    with pytest.raises(ScenarioValidationError, match="assertion ids must be unique"):
        load_scenario(source)


def test_rejects_relative_guest_file_path(tmp_path: Path) -> None:
    source = write_scenario(
        tmp_path, VALID_SCENARIO.replace("/var/tmp/boot-marker", "boot-marker")
    )

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source)

    assert "$.assertions[1].file.path" in str(raised.value)
    assert "must be absolute" in str(raised.value)


def test_rejects_content_expectation_for_absent_file(tmp_path: Path) -> None:
    source = write_scenario(
        tmp_path,
        VALID_SCENARIO.replace(
            "    exists: true\n    content_contains: boot-ok",
            "    exists: false\n    content_contains: boot-ok",
        ),
    )

    with pytest.raises(ScenarioValidationError) as raised:
        load_scenario(source)

    assert "$.assertions[1].file" in str(raised.value)
    assert "content_contains cannot be used when exists is false" in str(raised.value)
