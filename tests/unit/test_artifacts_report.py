"""Unit tests for the run workspace and the version-1 run report."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cloud_boot.artifacts import ArtifactError, RunWorkspace, boot_directory, new_run_id
from cloud_boot.assertions import AssertionResult, AssertionStatus
from cloud_boot.errors import ErrorPhase, ExitCode
from cloud_boot.image_store import VerifiedImage
from cloud_boot.process import ProcessTimeoutError
from cloud_boot.report import (
    BootResult,
    FailureRecord,
    RunReportV1,
    ScenarioIdentity,
    decide_outcome,
    image_identity,
    scenario_identity,
    seed_digests,
    write_report,
)
from cloud_boot.scenario import load_scenario
from cloud_boot.seed import SeedArtifact, SeedFile
from cloud_boot.ssh import SshTransportError

STARTED = datetime(2026, 11, 7, 19, 5, 12, tzinfo=UTC)


def tokens(*values: str) -> Callable[[], str]:
    remaining: Iterator[str] = iter(values)
    return lambda: next(remaining)


def workspace(tmp_path: Path, token: str = "a1b2c3d4") -> RunWorkspace:
    return RunWorkspace.create(tmp_path, wall_clock=lambda: STARTED, token_factory=tokens(token))


def mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def assertion(status: AssertionStatus = "passed", *, required: bool = True) -> AssertionResult:
    return AssertionResult(
        id="marker",
        type="file",
        boot="boot",
        required=required,
        status=status,
        expected={"exists": True},
        observed={"exists": status == "passed"},
        reason="checked",
        duration_seconds=0.1,
        command=None,
    )


def report(**overrides: Any) -> RunReportV1:
    fields: dict[str, Any] = {
        "run_id": "20261107T190512Z-a1b2c3d4",
        "scenario": ScenarioIdentity(name="basic-boot", source_path="/s.yaml", source_sha256=None),
        "image": None,
        "started_at": STARTED,
        "finished_at": STARTED,
        "duration_seconds": 1.5,
        "outcome": "passed",
        "exit_code": 0,
        "boots": [
            BootResult(
                name="boot",
                index=1,
                instance_id="i-1",
                seed_digests={},
                stop_reason="exited",
                assertions=[assertion()],
            )
        ],
        "artifacts": [],
        "warnings": [],
        "failure": None,
    }
    fields.update(overrides)
    return RunReportV1(**fields)


def test_create_allocates_a_private_uniquely_named_run_directory(tmp_path: Path) -> None:
    run = workspace(tmp_path)

    assert run.path == tmp_path / "runs" / "20261107T190512Z-a1b2c3d4"
    assert run.run_id == "20261107T190512Z-a1b2c3d4"
    assert mode(run.path) == 0o700
    assert mode(run.transient_dir) == 0o700
    assert new_run_id(STARTED, "ff") == "20261107T190512Z-ff"
    assert boot_directory(1, "boot") == "boots/001-boot"


def test_create_retries_name_collisions_and_then_gives_up(tmp_path: Path) -> None:
    (tmp_path / "runs" / "20261107T190512Z-taken").mkdir(parents=True)

    run = RunWorkspace.create(
        tmp_path, wall_clock=lambda: STARTED, token_factory=tokens("taken", "fresh")
    )
    assert run.run_id == "20261107T190512Z-fresh"

    with pytest.raises(ArtifactError, match="unique run directory") as raised:
        RunWorkspace.create(tmp_path, wall_clock=lambda: STARTED, token_factory=lambda: "taken")
    assert raised.value.exit_code == ExitCode.REPORT


def test_create_refuses_a_symlinked_runs_directory(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "runs").symlink_to(elsewhere)

    with pytest.raises(ArtifactError, match="real directory"):
        workspace(tmp_path / "root")
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize(
    "relative", ["", "/etc/passwd", "../outside", "boots/../../outside", "a\x00b"]
)
def test_artifact_paths_cannot_escape_the_run_directory(tmp_path: Path, relative: str) -> None:
    run = workspace(tmp_path)

    with pytest.raises(ValueError, match="artifact path"):
        run.artifact_path(relative)


def test_artifact_paths_reject_symlink_components(tmp_path: Path) -> None:
    run = workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (run.path / "boots").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        run.open_new("boots/001-boot/console.log")
    with pytest.raises(ValueError, match="symlink"):
        run.write_text_atomic("boots/report.json", "{}")
    assert list(outside.iterdir()) == []


def test_open_new_creates_private_files_and_never_overwrites(tmp_path: Path) -> None:
    run = workspace(tmp_path)

    with run.open_new("boots/001-boot/console.log") as console:
        console.write(b"[    0.000000] Linux version")
    written = run.write_bytes("boots/001-boot/qemu.stderr.log", b"terminating on signal 15")

    console_path = run.path / "boots" / "001-boot" / "console.log"
    assert console_path.read_bytes() == b"[    0.000000] Linux version"
    assert mode(console_path) == 0o600
    assert mode(console_path.parent) == 0o700
    assert written.read_bytes() == b"terminating on signal 15"
    with pytest.raises(ArtifactError):
        run.open_new("boots/001-boot/console.log")
    with pytest.raises(ArtifactError):
        run.write_bytes("boots/001-boot/qemu.stderr.log", b"replacement")
    assert written.read_bytes() == b"terminating on signal 15"


def test_open_new_refuses_a_planted_symlink(tmp_path: Path) -> None:
    run = workspace(tmp_path)
    target = tmp_path / "target"
    target.write_text("untouched")
    (run.path / "console.log").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        run.open_new("console.log")
    assert target.read_text() == "untouched"


def test_atomic_write_replaces_the_file_and_leaves_no_temporary(tmp_path: Path) -> None:
    run = workspace(tmp_path)

    run.write_text_atomic("report.json", '{"attempt": 1}\n')
    path = run.write_text_atomic("report.json", '{"attempt": 2}\n')

    assert path.read_text() == '{"attempt": 2}\n'
    assert mode(path) == 0o600
    assert not list(run.path.glob(".tmp-*"))


def test_atomic_write_failure_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = workspace(tmp_path)
    run.write_text_atomic("report.json", "previous")

    def disk_full(_source: object, _destination: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", disk_full)

    with pytest.raises(ArtifactError) as raised:
        run.write_text_atomic("report.json", "next")

    assert raised.value.exit_code == ExitCode.REPORT
    assert raised.value.phase == ErrorPhase.ARTIFACT
    assert (run.path / "report.json").read_text() == "previous"
    assert not list(run.path.glob(".tmp-*"))


def test_cleanup_removes_transient_assets_and_keeps_evidence(tmp_path: Path) -> None:
    run = workspace(tmp_path)
    for name in ("seed.iso", "overlay.qcow2", "id_ed25519", "id_ed25519.pub", "known_hosts"):
        (run.transient_dir / name).write_bytes(b"sensitive")
    run.write_bytes("boots/001-boot/console.log", b"console")
    run.write_text_atomic("report.json", "{}")

    assert run.cleanup_transient() == []
    assert run.cleanup_transient() == []

    assert not run.transient_dir.exists()
    assert (run.path / "boots" / "001-boot" / "console.log").read_bytes() == b"console"
    assert (run.path / "report.json").exists()


def test_cleanup_failures_become_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = workspace(tmp_path)
    blocked = run.transient_dir / "overlay.qcow2"

    def failing_rmtree(path: Path, *, onexc: Callable[..., object]) -> None:
        onexc(os.unlink, str(blocked), PermissionError(13, "Permission denied"))

    monkeypatch.setattr(shutil, "rmtree", failing_rmtree)

    warnings = run.cleanup_transient()

    assert warnings == [f"could not remove {blocked}: [Errno 13] Permission denied"]


def test_relative_references_stay_inside_the_run(tmp_path: Path) -> None:
    run = workspace(tmp_path)

    assert (
        run.relative(run.path / "boots" / "001-boot" / "console.log") == "boots/001-boot/console.log"
    )
    with pytest.raises(ValueError, match="subpath"):
        run.relative(tmp_path / "elsewhere.log")


@pytest.mark.parametrize(
    ("failure", "results", "expected"),
    [
        (None, [assertion()], ("passed", 0)),
        (None, [], ("passed", 0)),
        (None, [assertion("failed")], ("failed", 7)),
        (None, [assertion("blocked")], ("failed", 7)),
        (None, [assertion(), assertion("failed", required=False)], ("passed", 0)),
        # The most specific terminal condition wins over assertion results.
        (ProcessTimeoutError(), [assertion()], ("error", 8)),
        (SshTransportError("guest SSH did not become ready"), [assertion("blocked")], ("error", 6)),
    ],
)
def test_decide_outcome_is_the_single_source_of_the_exit_code(
    failure: Any, results: list[AssertionResult], expected: tuple[str, int]
) -> None:
    assert decide_outcome(failure=failure, assertions=results) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"exit_code": 7},
        {"outcome": "failed", "exit_code": 7},
        {"outcome": "failed", "exit_code": 0},
        {"outcome": "error", "exit_code": 10},
        {
            "outcome": "error",
            "exit_code": 4,
            "failure": FailureRecord.from_error(SshTransportError("unreachable")),
        },
        {"started_at": datetime(2026, 11, 7, 19, 5, 12)},  # naive: no timezone
        {"artifacts": ["/var/tmp/console.log"]},
        {"artifacts": ["../console.log"]},
        {"duration_seconds": -1.0},
    ],
)
def test_report_rejects_inconsistent_or_non_portable_content(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        report(**overrides)


def test_report_records_a_failure_with_json_safe_context() -> None:
    error = SshTransportError(
        "guest SSH did not become ready",
        context={"path": Path("/run/x"), "attempts": 4, "items": (1, Path("a"))},
    )

    failed = report(
        outcome="error",
        exit_code=error.exit_code,
        failure=FailureRecord.from_error(error),
        boots=[],
    )

    assert failed.failure is not None
    assert failed.failure.context == {"path": "/run/x", "attempts": 4, "items": [1, "a"]}
    assert failed.exit_code == ExitCode.SSH


def test_write_report_persists_portable_evidence(tmp_path: Path) -> None:
    scenario_file = tmp_path / "basic-boot.yaml"
    scenario_file.write_text(
        "scenario_version: 1\nname: basic-boot\nseed:\n  instance_id: i-1\n"
        "  user_data:\n    inline: |\n      #cloud-config\n      {}\n"
    )
    loaded = load_scenario(scenario_file)
    run = workspace(tmp_path)
    console = run.write_bytes("boots/001-boot/console.log", b"console")
    image = VerifiedImage(
        path=tmp_path / "base.img",
        digest="04e8f2" + "0" * 58,
        size_bytes=621284352,
        source_url="https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img",
        retrieved_at=0.0,
    )
    seed = SeedArtifact(
        path=run.transient_dir / "seed.iso",
        label="CIDATA",
        instance_id="i-1",
        files=(SeedFile("meta-data", "a" * 64, 10), SeedFile("user-data", "b" * 64, 20)),
    )
    results = [assertion()]
    outcome, exit_code = decide_outcome(failure=None, assertions=results)

    path = write_report(
        run,
        report(
            run_id=run.run_id,
            scenario=scenario_identity(loaded),
            image=image_identity(image),
            outcome=outcome,
            exit_code=exit_code,
            boots=[
                BootResult(
                    name="boot",
                    index=1,
                    instance_id=seed.instance_id,
                    seed_digests=seed_digests(seed),
                    stop_reason="exited",
                    assertions=results,
                )
            ],
            artifacts=[run.relative(console)],
        ),
    )

    document = json.loads(path.read_text())
    assert path == run.path / "report.json"
    assert mode(path) == 0o600
    assert document["report_version"] == 1
    assert document["exit_code"] == 0
    assert (
        document["scenario"]["source_sha256"]
        == hashlib.sha256(scenario_file.read_bytes()).hexdigest()
    )
    assert document["image"]["signature_authenticated"] is False
    assert document["boots"][0]["seed_digests"] == {"meta-data": "a" * 64, "user-data": "b" * 64}
    assert document["boots"][0]["assertions"][0]["status"] == "passed"
    assert document["artifacts"] == ["boots/001-boot/console.log"]
    assert document["started_at"].startswith("2026-11-07T19:05:12")
    assert RunReportV1.model_validate(document) == RunReportV1.model_validate_json(path.read_text())


def test_report_persistence_failure_is_exit_code_9(tmp_path: Path) -> None:
    run = workspace(tmp_path)
    (run.path / "report.json").mkdir()

    with pytest.raises(ArtifactError) as raised:
        write_report(run, report())

    assert raised.value.exit_code == ExitCode.REPORT
