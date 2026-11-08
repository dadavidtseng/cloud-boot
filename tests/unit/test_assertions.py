"""Unit tests for Milestone 0 assertion evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from cloud_boot.assertions import (
    CLOUD_INIT_STATUS_SCRIPT,
    FILE_SCRIPT,
    MAX_FILE_CONTENT_BYTES,
    PACKAGE_SCRIPT,
    AssertionRegistry,
    block_all,
    boot_passed,
    default_registry,
    run_assertions,
)
from cloud_boot.models import CloudInitStatusAssertion, FileAssertion, PackageInstalledAssertion
from cloud_boot.process import ProcessTimeoutError
from cloud_boot.ssh import GuestResult, SshTransportError


class FakeTransport:
    """Replays one scripted response per call and records what was sent."""

    def __init__(self, responses: Sequence[GuestResult | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, tuple[str, ...], float]] = []

    def execute(self, script: str, args: Sequence[str] = (), *, timeout: float = 60.0) -> GuestResult:
        self.calls.append((script, tuple(args), timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.25
        return self.now


def guest(exit_status: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> GuestResult:
    return GuestResult(exit_status=exit_status, stdout=stdout, stderr=stderr, duration_seconds=0.1)


def cloud_init_json(extended: str, status: str = "done", **extra: Any) -> bytes:
    document = {
        "status": status,
        "extended_status": extended,
        "datasource": "nocloud",
        "errors": [],
        "recoverable_errors": {},
        **extra,
    }
    return json.dumps(document).encode()


def cloud_init(**fields: Any) -> CloudInitStatusAssertion:
    return CloudInitStatusAssertion(id="cloud-init-done", type="cloud-init-status", **fields)


def file_check(**fields: Any) -> FileAssertion:
    return FileAssertion(id="marker", type="file", path="/var/tmp/boot-marker", **fields)


def package(name: str = "openssh-server", **fields: Any) -> PackageInstalledAssertion:
    return PackageInstalledAssertion(id="pkg", type="package-installed", name=name, **fields)


def evaluate_one(assertion: Any, response: GuestResult) -> tuple[Any, FakeTransport]:
    transport = FakeTransport([response])
    [result] = run_assertions([assertion], transport, boot="boot", clock=FakeClock())
    return result, transport


def test_cloud_init_done_passes_with_observed_status() -> None:
    result, transport = evaluate_one(cloud_init(), guest(stdout=cloud_init_json("done")))

    assert result.status == "passed"
    assert result.expected == {"extended_status": "done"}
    assert result.observed == {
        "status": "done",
        "extended_status": "done",
        "datasource": "nocloud",
        "errors": 0,
        "recoverable_errors": 0,
    }
    assert transport.calls == [(CLOUD_INIT_STATUS_SCRIPT, (), 30)]
    assert result.command is not None
    assert result.command.script == "cloud-init-status"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        # Recoverable errors keep status "done" but mark the extended status degraded.
        (
            guest(
                2,
                cloud_init_json("degraded done", recoverable_errors={"WARNING": ["deprecated key"]}),
            ),
            "'degraded done'",
        ),
        (guest(0, cloud_init_json("running", status="running")), "'running'"),
        (guest(1, cloud_init_json("error", status="error")), "'error'"),
        (guest(0, b"not json", b"Traceback"), "could not parse cloud-init status output"),
        (guest(3), "guest tool needed to check cloud-init status is missing"),
        (guest(4), "sudo -n is not available"),
    ],
)
def test_cloud_init_status_other_than_done_fails(response: GuestResult, reason: str) -> None:
    result, _ = evaluate_one(cloud_init(), response)

    assert result.status == "failed"
    assert reason in result.reason


def test_file_with_expected_content_passes_and_values_travel_as_arguments() -> None:
    result, transport = evaluate_one(
        file_check(content_contains="boot-ok"), guest(stdout=b"file 7\nboot-ok")
    )

    assert result.status == "passed"
    assert result.observed == {"exists": True, "type": "file", "size": 7, "contains": True}
    script, args, _ = transport.calls[0]
    assert script == FILE_SCRIPT
    assert args == ("/var/tmp/boot-marker", "content", str(MAX_FILE_CONTENT_BYTES))
    assert "/var/tmp/boot-marker" not in script
    assert "boot-ok" not in script


@pytest.mark.parametrize(
    ("needle", "status"), [("second\nthird", "passed"), ("first\nthird", "failed")]
)
def test_multiline_content_is_matched_as_one_substring(needle: str, status: str) -> None:
    body = b"first\nsecond\nthird\n"
    response = guest(stdout=b"file %d\n" % len(body) + body)

    result, _ = evaluate_one(file_check(content_contains=needle), response)

    assert result.status == status


@pytest.mark.parametrize(
    ("fields", "stdout", "status", "reason"),
    [
        ({}, b"file 7\n", "passed", "exists"),
        ({}, b"missing\n", "failed", "does not exist"),
        ({"exists": False}, b"missing\n", "passed", "does not exist"),
        ({"exists": False}, b"file 7\n", "failed", "exists"),
        ({}, b"directory\n", "failed", "is a directory, not a file"),
        ({"content_contains": "boot-ok"}, b"file 9\nboot-fail", "failed", "does not contain"),
        (
            {"content_contains": "boot-ok"},
            b"file %d\n" % (MAX_FILE_CONTENT_BYTES + 1),
            "failed",
            "larger than",
        ),
    ],
)
def test_file_states(fields: dict[str, Any], stdout: bytes, status: str, reason: str) -> None:
    result, _ = evaluate_one(file_check(**fields), guest(stdout=stdout))

    assert result.status == status
    assert reason in result.reason


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (guest(4), "sudo -n is not available"),
        (guest(3), "guest tool needed to check the file is missing"),
        (guest(0, b"file 7\n", b""), None),
    ],
)
def test_file_check_never_mistakes_an_unobservable_guest_for_a_missing_file(
    response: GuestResult, reason: str | None
) -> None:
    result, _ = evaluate_one(file_check(), response)

    if reason is None:
        assert result.status == "passed"
    else:
        assert result.status == "failed"
        assert reason in result.reason
        assert result.observed == {"exit_status": response.exit_status}


def test_truncated_file_output_fails_instead_of_matching_partial_content() -> None:
    truncated = GuestResult(
        exit_status=0, stdout=b"boot-ok", stderr=b"", duration_seconds=0.1, stdout_dropped_bytes=10
    )

    result, _ = evaluate_one(file_check(content_contains="boot-ok"), truncated)

    assert result.status == "failed"


def test_installed_package_passes_with_version() -> None:
    result, transport = evaluate_one(package(), guest(stdout=b"installed\t1:9.6p1-3ubuntu13.20\n"))

    assert result.status == "passed"
    assert result.observed == {"status": "installed", "version": "1:9.6p1-3ubuntu13.20"}
    assert transport.calls[0][:2] == (PACKAGE_SCRIPT, ("openssh-server",))
    assert "openssh-server" not in PACKAGE_SCRIPT


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        # Removed packages keep their configuration files and still appear in dpkg's database.
        (guest(0, b"config-files\t1:9.6p1\n"), "has dpkg status 'config-files'"),
        (
            guest(1, b"", b"dpkg-query: no packages found matching nope\n"),
            "is not installed: dpkg-query: no packages found matching nope",
        ),
        (guest(3), "guest tool needed to check packages is missing"),
    ],
)
def test_package_not_installed_fails(response: GuestResult, reason: str) -> None:
    result, _ = evaluate_one(package("nope"), response)

    assert result.status == "failed"
    assert reason in result.reason


def test_any_installed_architecture_satisfies_a_multiarch_package() -> None:
    stdout = b"config-files\t2.39-0ubuntu8\ninstalled\t2.39-0ubuntu8.4\n"

    result, _ = evaluate_one(package("libc6"), guest(stdout=stdout))

    assert result.status == "passed"
    assert result.observed == {"status": "installed", "version": "2.39-0ubuntu8.4"}


def test_transport_loss_blocks_the_current_and_remaining_assertions_in_order() -> None:
    transport = FakeTransport(
        [
            guest(stdout=cloud_init_json("done")),
            SshTransportError(
                "SSH transport failed during guest command",
                context={"stderr": "ssh: connect to host 127.0.0.1 port 40001: Connection refused"},
            ),
        ]
    )
    assertions = [cloud_init(), file_check(), package()]

    results = run_assertions(assertions, transport, boot="boot", clock=FakeClock())

    assert [(result.id, result.status) for result in results] == [
        ("cloud-init-done", "passed"),
        ("marker", "blocked"),
        ("pkg", "blocked"),
    ]
    assert all("SSH transport unavailable" in result.reason for result in results[1:])
    assert all("Connection refused" in result.reason for result in results[1:])
    assert all(result.observed is None for result in results[1:])
    # Nothing is attempted after the transport is lost.
    assert len(transport.calls) == 2
    assert not boot_passed(results)


def test_timeout_fails_only_that_assertion_and_uses_its_own_limit() -> None:
    transport = FakeTransport([ProcessTimeoutError(), guest(stdout=b"installed\t1:9.6p1\n")])
    assertions = [file_check(timeout_seconds=5), package()]

    results = run_assertions(assertions, transport, boot="boot", clock=FakeClock())

    assert [result.status for result in results] == ["failed", "passed"]
    assert "5-second timeout" in results[0].reason
    assert [call[2] for call in transport.calls] == [5, 30]


def test_results_are_uniform_and_record_durations() -> None:
    transport = FakeTransport([guest(stdout=b"missing\n")])

    [result] = run_assertions(
        [file_check(required=False)], transport, boot="first", clock=FakeClock()
    )

    assert result.model_dump() == {
        "id": "marker",
        "type": "file",
        "boot": "first",
        "required": False,
        "status": "failed",
        "expected": {"path": "/var/tmp/boot-marker", "exists": True, "content_contains": None},
        "observed": {"exists": False, "type": None, "size": None, "contains": None},
        "reason": "/var/tmp/boot-marker does not exist",
        "duration_seconds": 0.25,
        "command": {
            "script": "file",
            "args": ("/var/tmp/boot-marker", "", str(MAX_FILE_CONTENT_BYTES)),
        },
    }


def test_boot_outcome_ignores_optional_failures_but_not_required_ones() -> None:
    optional_failure = run_assertions(
        [file_check(required=False), package()],
        FakeTransport([guest(stdout=b"missing\n"), guest(stdout=b"installed\t1\n")]),
        boot="boot",
    )
    blocked = block_all([cloud_init(), package()], boot="boot", reason="SSH never became ready")

    assert boot_passed(optional_failure)
    assert [result.status for result in blocked] == ["blocked", "blocked"]
    assert not boot_passed(blocked)
    assert boot_passed([])


def test_registry_rejects_duplicate_and_unknown_types() -> None:
    registry = default_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register("file", lambda _assertion, _transport: None)  # type: ignore[arg-type,return-value]

    with pytest.raises(LookupError, match="no evaluator"):
        AssertionRegistry().evaluate(file_check(), FakeTransport([]))
