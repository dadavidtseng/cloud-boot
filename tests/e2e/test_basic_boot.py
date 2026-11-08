"""Opt-in end-to-end test: boot Ubuntu Noble under KVM with the documented basic scenario.

This test downloads the official cloud image on first use (about 600 MB) and boots a real VM,
so it only runs when `CLOUD_BOOT_E2E=1` is set and `/dev/kvm` is usable. Otherwise it is
reported as skipped with the reason, never as passed.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO = REPO_ROOT / "scenarios" / "basic-boot.yaml"
KVM_DEVICE = Path("/dev/kvm")
# Covers a first-run image download plus the boot itself.
RUN_TIMEOUT_SECONDS = 30 * 60
INTERRUPT_GRACE_SECONDS = 120


def skip_reason() -> str | None:
    if os.environ.get("CLOUD_BOOT_E2E") != "1":
        return "set CLOUD_BOOT_E2E=1 to boot a real VM (downloads about 600 MB on first use)"
    if not KVM_DEVICE.exists():
        return "/dev/kvm is missing, so this host cannot run KVM guests"
    if not os.access(KVM_DEVICE, os.R_OK | os.W_OK):
        return "/dev/kvm is not readable and writable by this user (join the kvm group)"
    return None


def installed_cloud_boot() -> str:
    # The console script installed next to this interpreter, as the README documents.
    command = shutil.which("cloud-boot", path=str(Path(sys.executable).parent))
    if command is None:
        pytest.fail("the cloud-boot console script is not installed; run `uv sync` first")
    return command


def run_cloud_boot(command: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        [command, "run", str(SCENARIO)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=RUN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # SIGINT lets cloud-boot stop QEMU and delete the key; kill only as a last resort.
        process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=INTERRUPT_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        pytest.fail(
            f"cloud-boot did not finish within {RUN_TIMEOUT_SECONDS} s; "
            f"run directory: {stdout.strip() or '<none>'}\nstderr:\n{stderr}"
        )
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def test_basic_boot_scenario_passes_under_kvm(tmp_path: Path) -> None:
    reason = skip_reason()
    if reason is not None:
        pytest.skip(reason)
    command = installed_cloud_boot()
    # Keep run directories out of the user's state directory; reuse the default image cache.
    env = {**os.environ, "XDG_STATE_HOME": str(tmp_path / "state")}

    completed = run_cloud_boot(command, env)

    run_dir = completed.stdout.strip()
    evidence = (
        f"exit code {completed.returncode}; run directory: {run_dir or '<none>'}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert completed.returncode == 0, evidence
    assert completed.stdout.count("\n") == 1, evidence
    run = Path(run_dir)
    report = json.loads((run / "report.json").read_text())
    assert report["exit_code"] == 0, evidence
    assert report["outcome"] == "passed", evidence
    assertions = report["boots"][0]["assertions"]
    assert [(entry["id"], entry["status"]) for entry in assertions] == [
        ("cloud-init-done", "passed"),
        ("marker-created", "passed"),
        ("ssh-installed", "passed"),
    ], evidence
    # Milestone 0 verifies the checksum only and must say so.
    assert report["image"]["checksum_verified"] is True
    assert report["image"]["signature_authenticated"] is False
    boot = run / "boots" / "001-boot"
    for name in ("console.log", "cloud-init.log", "qemu-command.json", "qemu.stderr.log"):
        assert (boot / name).is_file(), f"missing {name}; {evidence}"
    # The guest took its hostname from the seed's meta-data.
    assert b"basic-boot login:" in (boot / "console.log").read_bytes(), evidence
    # Seed media, the overlay, and the private key never outlive the run.
    assert not (run / "transient").exists(), evidence
