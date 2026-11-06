"""Unit tests for the shared cloud-boot error contract."""

from __future__ import annotations

import pytest

from cloud_boot.errors import CloudBootError, ExitCode


def test_cloud_boot_error_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="unknown error phase"):
        CloudBootError("invalid", phase="not-a-phase")


def test_cloud_boot_error_serializes_stable_shape() -> None:
    context = {"pid": 42}
    error = CloudBootError(
        "image digest mismatch",
        code="image-digest-mismatch",
        phase="preparation",
        remediation="Delete the cached image and retry.",
        context=context,
        exit_code=ExitCode.PREPARATION,
    )
    context["pid"] = 0

    assert error.as_dict() == {
        "code": "image-digest-mismatch",
        "phase": "preparation",
        "message": "image digest mismatch",
        "exit_code": 4,
        "context": {"pid": 42},
        "remediation": "Delete the cached image and retry.",
    }
    assert type(error.as_dict()["exit_code"]) is int


def test_cloud_boot_error_omits_missing_remediation() -> None:
    serialized = CloudBootError("failed").as_dict()

    assert "remediation" not in serialized
    assert serialized["phase"] == "internal"
    assert serialized["exit_code"] == ExitCode.INTERNAL
