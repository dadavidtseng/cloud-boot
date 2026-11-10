# ----------------------------------------------------------------------------------------------------
# seed.py
# ----------------------------------------------------------------------------------------------------

"""Build and verify NoCloud seed media for one boot plan."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pycdlib
import yaml
from pycdlib.pycdlibexception import PyCdlibException

from .errors import CloudBootError, ErrorPhase, ExitCode
from .models import BootPlan, PayloadSpec, SeedSpec

__all__ = [
    "MAX_PAYLOAD_BYTES",
    "SEED_FILENAME",
    "SEED_LABEL",
    "SeedArtifact",
    "SeedBuildError",
    "SeedFile",
    "build_seed",
    "render_meta_data",
]

SEED_LABEL = "CIDATA"
SEED_FILENAME = "seed.iso"
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024

# NoCloud reads these Rock Ridge/Joliet names; each needs an ISO-9660 level-1 (8.3) alias.
_ISO_NAMES: Mapping[str, str] = {
    "meta-data": "/METADATA.;1",
    "user-data": "/USERDATA.;1",
    "vendor-data": "/VENDORDA.;1",
    "network-config": "/NETWORKC.;1",
}


# ----------------------------------------------------------------------------------------------------
class SeedBuildError(CloudBootError):
    """Expected failure while preparing NoCloud seed media."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="seed-build-failed",
            phase=ErrorPhase.PREPARATION,
            remediation="Check the scenario payload files and that the run directory is writable.",
            context=context,
            exit_code=ExitCode.PREPARATION,
        )


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SeedFile:
    """Digest and size of one file stored on the seed media."""

    name: str
    sha256: str
    size_bytes: int


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SeedArtifact:
    """A verified NoCloud seed ISO and the identity of each file it contains."""

    path: Path
    label: str
    instance_id: str
    files: tuple[SeedFile, ...]


# ----------------------------------------------------------------------------------------------------
def render_meta_data(seed: SeedSpec, ssh_public_key: str) -> bytes:
    """Serialize harness-owned NoCloud metadata.

    Args:
        seed: Seed inputs supplying the instance ID and optional hostname.
        ssh_public_key: The run's single-line OpenSSH public key.

    Returns:
        UTF-8 YAML containing `instance-id`, optional `local-hostname`, and `public-keys`.

    Raises:
        SeedBuildError: If the public key is empty or spans more than one line.
    """
    _validate_public_key(ssh_public_key)
    document: dict[str, object] = {"instance-id": seed.instance_id}
    if seed.local_hostname is not None:
        document["local-hostname"] = seed.local_hostname
    document["public-keys"] = [ssh_public_key]
    # A large width keeps PyYAML from folding long keys across lines.
    rendered = yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=1 << 16)
    return rendered.encode("utf-8")


# ----------------------------------------------------------------------------------------------------
def build_seed(
    plan: BootPlan,
    *,
    ssh_public_key: str,
    scenario_dir: str | Path,
    output_dir: str | Path,
) -> SeedArtifact:
    """Build `seed.iso` for one boot plan and verify it before publishing.

    The ISO is written to a temporary file in `output_dir`, reopened, and checked for the
    `CIDATA` label, the exact Rock Ridge and Joliet file sets, and byte-identical contents. Only
    then is it renamed to `seed.iso`, so an unverified seed never appears under the final name.

    Args:
        plan: The boot plan whose seed inputs are materialized.
        ssh_public_key: The run's single-line OpenSSH public key, added to `public-keys`.
        scenario_dir: Directory that scenario-relative payload files resolve against.
        output_dir: Run directory that receives `seed.iso`; created with mode 0700 if missing.

    Returns:
        The verified seed's path, label, instance ID, and per-file SHA-256 digests.

    Raises:
        SeedBuildError: If a payload cannot be read, exceeds the size limit, or the ISO cannot
            be written or fails verification.
    """
    contents = _seed_contents(plan.seed, ssh_public_key, Path(scenario_dir))
    destination = Path(output_dir)
    try:
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".seed-", suffix=".iso", dir=destination)
    except OSError as e:
        raise SeedBuildError(
            "cannot create the seed output directory", context={"path": str(destination)}
        ) from e
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            _write_iso(contents, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            _verify_iso(contents, stream)
        final_path = destination / SEED_FILENAME
        os.replace(temporary_path, final_path)
    except OSError as e:
        raise SeedBuildError("cannot write the seed ISO", context={"path": str(destination)}) from e
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
    return SeedArtifact(
        path=final_path,
        label=SEED_LABEL,
        instance_id=plan.seed.instance_id,
        files=tuple(
            SeedFile(name=name, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
            for name, data in contents.items()
        ),
    )


# ----------------------------------------------------------------------------------------------------
def _validate_public_key(key: str) -> None:
    if not key or key != key.strip() or any(character in key for character in "\r\n\x00"):
        raise SeedBuildError("SSH public key must be one non-empty line")
    if len(key.split()) < 2:
        raise SeedBuildError("SSH public key must contain a key type and key data")


# ----------------------------------------------------------------------------------------------------
def _seed_contents(seed: SeedSpec, ssh_public_key: str, scenario_dir: Path) -> dict[str, bytes]:
    contents = {
        "meta-data": render_meta_data(seed, ssh_public_key),
        "user-data": _payload_bytes("user_data", seed.user_data, scenario_dir),
        # An empty vendor-data file is a valid "no vendor configuration" document.
        "vendor-data": (
            b""
            if seed.vendor_data is None
            else _payload_bytes("vendor_data", seed.vendor_data, scenario_dir)
        ),
    }
    if seed.network_config is not None:
        contents["network-config"] = _payload_bytes(
            "network_config", seed.network_config, scenario_dir
        )
    return contents


# ----------------------------------------------------------------------------------------------------
def _payload_bytes(field: str, payload: PayloadSpec, scenario_dir: Path) -> bytes:
    if payload.inline is not None:
        data = payload.inline.encode("utf-8")
    elif payload.file is not None:
        data = _read_payload_file(field, payload.file, scenario_dir)
    else:
        raise SeedBuildError(f"{field} has no payload source")
    if len(data) > MAX_PAYLOAD_BYTES:
        raise SeedBuildError(
            f"{field} exceeds the {MAX_PAYLOAD_BYTES}-byte payload limit",
            context={"field": field, "size_bytes": len(data)},
        )
    return data


# ----------------------------------------------------------------------------------------------------
def _read_payload_file(field: str, relative: str, scenario_dir: Path) -> bytes:
    # Re-check containment at read time: the file may have changed since the scenario loaded.
    context = {"field": field, "file": relative}
    try:
        root = scenario_dir.resolve(strict=True)
        candidate = (root / relative).resolve(strict=True)
    except OSError as e:
        raise SeedBuildError(f"cannot read {field} payload file", context=context) from e
    if not candidate.is_relative_to(root):
        raise SeedBuildError(
            f"{field} payload file resolves outside the scenario directory", context=context
        )
    try:
        descriptor = os.open(candidate, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise SeedBuildError(f"{field} payload is not a regular file", context=context)
            return stream.read(MAX_PAYLOAD_BYTES + 1)
    except OSError as e:
        raise SeedBuildError(f"cannot read {field} payload file", context=context) from e


# ----------------------------------------------------------------------------------------------------
def _write_iso(contents: Mapping[str, bytes], stream: BinaryIO) -> None:
    iso = pycdlib.PyCdlib()
    try:
        iso.new(interchange_level=1, vol_ident=SEED_LABEL, joliet=3, rock_ridge="1.09")
        for name, data in contents.items():
            iso.add_fp(
                io.BytesIO(data),
                len(data),
                iso_path=_ISO_NAMES[name],
                rr_name=name,
                joliet_path=f"/{name}",
            )
        iso.write_fp(stream)
    except PyCdlibException as e:
        raise SeedBuildError("cannot build the seed ISO") from e
    finally:
        iso.close()


# ----------------------------------------------------------------------------------------------------
def _verify_iso(contents: Mapping[str, bytes], stream: BinaryIO) -> None:
    expected = sorted(contents)
    iso = pycdlib.PyCdlib()
    try:
        iso.open_fp(stream)
        label = iso.pvd.volume_identifier.decode("ascii").rstrip(" ")
        if label != SEED_LABEL:
            raise SeedBuildError("seed ISO has the wrong volume label", context={"label": label})
        rock_ridge_names = sorted(_rock_ridge_names(iso))
        joliet_names = sorted(
            child.file_identifier().decode("utf-16-be")
            for child in iso.list_children(joliet_path="/")
            if not (child.is_dot() or child.is_dotdot())
        )
        if rock_ridge_names != expected or joliet_names != expected:
            raise SeedBuildError(
                "seed ISO file set does not match the expected NoCloud files",
                context={
                    "expected": expected,
                    "rock_ridge": rock_ridge_names,
                    "joliet": joliet_names,
                },
            )
        for name, data in contents.items():
            extracted = io.BytesIO()
            iso.get_file_from_iso_fp(extracted, rr_path=f"/{name}")
            if hashlib.sha256(extracted.getvalue()).digest() != hashlib.sha256(data).digest():
                raise SeedBuildError(
                    "seed ISO content does not match its input", context={"file": name}
                )
    except PyCdlibException as e:
        raise SeedBuildError("cannot reopen the seed ISO for verification") from e
    finally:
        with contextlib.suppress(PyCdlibException):
            iso.close()


# ----------------------------------------------------------------------------------------------------
def _rock_ridge_names(iso: pycdlib.PyCdlib) -> list[str]:
    names: list[str] = []
    for child in iso.list_children(rr_path="/"):
        if child.is_dot() or child.is_dotdot():
            continue
        if child.rock_ridge is None:
            raise SeedBuildError("seed ISO entry is missing its Rock Ridge name")
        names.append(child.rock_ridge.name().decode("utf-8"))
    return names
