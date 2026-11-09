# ----------------------------------------------------------------------------------------------------
# test_errors.py
# ----------------------------------------------------------------------------------------------------

"""Unit tests for the shared cloud-boot error contract."""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

import pytest

from cloud_boot.errors import CloudBootError, ExitCode


# ----------------------------------------------------------------------------------------------------
def test_cloud_boot_error_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="unknown error phase"):
        CloudBootError("invalid", phase="not-a-phase")


# ----------------------------------------------------------------------------------------------------
def test_cloud_boot_error_stores_normalized_fields() -> None:
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

    assert error.code == "image-digest-mismatch"
    assert error.phase == "preparation"
    assert error.message == "image digest mismatch"
    assert error.remediation == "Delete the cached image and retry."
    assert error.context == {"pid": 42}
    assert error.exit_code == 4
    assert type(error.phase) is str
    assert type(error.exit_code) is int


# ----------------------------------------------------------------------------------------------------
def test_cloud_boot_error_defaults_to_internal() -> None:
    error = CloudBootError("failed")

    assert error.code == "internal-error"
    assert error.phase == "internal"
    assert error.remediation is None
    assert error.exit_code == ExitCode.INTERNAL
