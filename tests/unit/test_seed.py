"""Unit tests for NoCloud seed media."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pycdlib
import pytest
import yaml

from cloud_boot import seed as seed_module
from cloud_boot.errors import ErrorPhase, ExitCode
from cloud_boot.models import BootPlan, SeedSpec
from cloud_boot.seed import SeedBuildError, build_seed, render_meta_data

PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleKeyMaterial cloud-boot-run"
USER_DATA = "#cloud-config\nwrite_files:\n  - path: /var/tmp/boot-marker\n    content: boot-ok\n"


def make_plan(**seed: Any) -> BootPlan:
    fields: dict[str, Any] = {
        "instance_id": "cloud-boot-basic-001",
        "user_data": {"inline": USER_DATA},
    }
    fields.update(seed)
    return BootPlan(name="boot", seed=SeedSpec.model_validate(fields))


def build(tmp_path: Path, plan: BootPlan, **kwargs: Any) -> seed_module.SeedArtifact:
    options: dict[str, Any] = {
        "ssh_public_key": PUBLIC_KEY,
        "scenario_dir": tmp_path / "scenario",
        "output_dir": tmp_path / "run",
    }
    options.update(kwargs)
    Path(options["scenario_dir"]).mkdir(parents=True, exist_ok=True)
    return build_seed(plan, **options)


def read_iso(path: Path) -> tuple[str, dict[str, bytes], list[str]]:
    """Independently reopen an ISO, returning its label, Rock Ridge files, and Joliet names."""
    iso = pycdlib.PyCdlib()
    iso.open(str(path))
    try:
        label = iso.pvd.volume_identifier.decode("ascii").rstrip(" ")
        files: dict[str, bytes] = {}
        for child in iso.list_children(rr_path="/"):
            if child.is_dot() or child.is_dotdot():
                continue
            assert child.rock_ridge is not None
            name = child.rock_ridge.name().decode()
            extracted = io.BytesIO()
            iso.get_file_from_iso_fp(extracted, rr_path=f"/{name}")
            files[name] = extracted.getvalue()
        joliet = sorted(
            child.file_identifier().decode("utf-16-be")
            for child in iso.list_children(joliet_path="/")
            if not (child.is_dot() or child.is_dotdot())
        )
    finally:
        iso.close()
    return label, files, joliet


def test_builds_cidata_iso_with_exact_nocloud_files(tmp_path: Path) -> None:
    artifact = build(tmp_path, make_plan())

    label, files, joliet = read_iso(artifact.path)

    assert artifact.path == tmp_path / "run" / "seed.iso"
    assert artifact.label == label == "CIDATA"
    assert artifact.instance_id == "cloud-boot-basic-001"
    assert sorted(files) == joliet == ["meta-data", "user-data", "vendor-data"]
    assert [entry.name for entry in artifact.files] == ["meta-data", "user-data", "vendor-data"]
    for entry in artifact.files:
        assert entry.sha256 == hashlib.sha256(files[entry.name]).hexdigest()
        assert entry.size_bytes == len(files[entry.name])
    assert not list((tmp_path / "run").glob(".seed-*"))


def test_preserves_inline_and_file_payload_bytes(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    vendor_bytes = b"#cloud-config\r\nruncmd: [echo \xe4\xb8\xad\xe6\x96\x87]\r\n\xff"
    (scenario / "vendor.yaml").write_bytes(vendor_bytes)
    inline = "#cloud-config\nruncmd: [echo 中文]"  # No trailing newline.

    artifact = build(
        tmp_path,
        make_plan(user_data={"inline": inline}, vendor_data={"file": "vendor.yaml"}),
    )

    _, files, _ = read_iso(artifact.path)
    assert files["user-data"] == inline.encode("utf-8")
    assert files["vendor-data"] == vendor_bytes


@pytest.mark.parametrize("hostname", [None, "basic-boot"])
def test_meta_data_carries_identity_and_public_key(tmp_path: Path, hostname: str | None) -> None:
    artifact = build(tmp_path, make_plan(local_hostname=hostname))

    _, files, _ = read_iso(artifact.path)
    expected: dict[str, object] = {"instance-id": "cloud-boot-basic-001"}
    if hostname is not None:
        expected["local-hostname"] = hostname
    expected["public-keys"] = [PUBLIC_KEY]
    assert yaml.safe_load(files["meta-data"]) == expected


def test_long_public_key_stays_on_one_line() -> None:
    key = "ssh-rsa " + "A" * 700 + " long-comment"

    rendered = render_meta_data(make_plan().seed, key)

    assert yaml.safe_load(rendered)["public-keys"] == [key]
    assert any(line.endswith(key) for line in rendered.decode().splitlines())


def test_network_config_is_written_only_when_provided(tmp_path: Path) -> None:
    network = "version: 2\nethernets:\n  enp1s0:\n    dhcp4: true\n"

    without = build(tmp_path / "a", make_plan())
    with_network = build(tmp_path / "b", make_plan(network_config={"inline": network}))

    assert "network-config" not in read_iso(without.path)[1]
    assert read_iso(with_network.path)[1]["network-config"] == network.encode()


def test_missing_vendor_data_defaults_to_empty_document(tmp_path: Path) -> None:
    artifact = build(tmp_path, make_plan())

    _, files, _ = read_iso(artifact.path)
    assert files["vendor-data"] == b""
    vendor = next(entry for entry in artifact.files if entry.name == "vendor-data")
    assert vendor.sha256 == hashlib.sha256(b"").hexdigest()


def test_rejects_payload_file_replaced_by_escaping_symlink(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("#cloud-config\n{}\n")
    (scenario / "user-data.yaml").symlink_to(outside)

    with pytest.raises(SeedBuildError, match="outside the scenario directory"):
        build(tmp_path, make_plan(user_data={"file": "user-data.yaml"}))
    assert not (tmp_path / "run" / "seed.iso").exists()


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_unreadable_payload_file_is_a_preparation_failure(tmp_path: Path, kind: str) -> None:
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    if kind == "directory":
        (scenario / "user-data.yaml").mkdir()

    with pytest.raises(SeedBuildError) as raised:
        build(tmp_path, make_plan(user_data={"file": "user-data.yaml"}))

    assert raised.value.phase == ErrorPhase.PREPARATION
    assert raised.value.exit_code == ExitCode.PREPARATION
    assert raised.value.context["field"] == "user_data"


@pytest.mark.parametrize("source", ["inline", "file"])
def test_rejects_oversized_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(seed_module, "MAX_PAYLOAD_BYTES", 8)
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    (scenario / "user-data.yaml").write_bytes(b"x" * 9)
    payload = {"inline": "x" * 9} if source == "inline" else {"file": "user-data.yaml"}

    with pytest.raises(SeedBuildError, match="payload limit"):
        build(tmp_path, make_plan(user_data=payload))


@pytest.mark.parametrize(
    "key",
    ["", "ssh-ed25519", " ssh-ed25519 AAAA", "ssh-ed25519 AAAA\ninstance-id: injected"],
)
def test_rejects_invalid_public_key(tmp_path: Path, key: str) -> None:
    with pytest.raises(SeedBuildError, match="SSH public key"):
        build(tmp_path, make_plan(), ssh_public_key=key)


def test_unwritable_output_directory_is_a_preparation_failure(tmp_path: Path) -> None:
    blocker = tmp_path / "run"
    blocker.write_text("not a directory")

    with pytest.raises(SeedBuildError) as raised:
        build(tmp_path, make_plan(), output_dir=blocker / "seed")

    assert raised.value.exit_code == ExitCode.PREPARATION


def test_verification_failure_publishes_no_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = seed_module._write_iso

    def tampered(contents: dict[str, bytes], stream: Any) -> None:
        original({**contents, "user-data": b"tampered"}, stream)

    monkeypatch.setattr(seed_module, "_write_iso", tampered)

    with pytest.raises(SeedBuildError, match="content does not match"):
        build(tmp_path, make_plan())
    assert not list((tmp_path / "run").iterdir())


def test_rebuild_replaces_previous_seed(tmp_path: Path) -> None:
    first = build(tmp_path, make_plan(instance_id="first-001"))
    second = build(tmp_path, make_plan(instance_id="second-001"))

    _, files, _ = read_iso(second.path)
    assert first.path == second.path
    assert yaml.safe_load(files["meta-data"])["instance-id"] == "second-001"
