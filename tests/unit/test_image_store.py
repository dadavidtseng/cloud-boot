"""Unit tests for checksum-verified image storage."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from cloud_boot.errors import ErrorPhase, ExitCode
from cloud_boot.image_store import ImageStore, ImageStoreError, _sleep_until_cancelled
from cloud_boot.models import ImageSpec
from cloud_boot.process import CommandRunner

IMAGE_NAME = "noble-server-cloudimg-amd64.img"
IMAGE_BYTES = b"cloud-boot image\n"
IMAGE_DIGEST = hashlib.sha256(IMAGE_BYTES).hexdigest()
MANIFEST = f"{IMAGE_DIGEST} *{IMAGE_NAME}\n"


class FakeResponse(io.BytesIO):
    """Closable byte response with HTTP metadata."""

    def __init__(
        self, data: bytes, *, status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class FakeDownloader:
    """URL keyed response factory recording requests."""

    def __init__(self, responses: dict[str, list[FakeResponse | Exception]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, url: str) -> FakeResponse:
        self.calls.append(url)
        value = self.responses[url].pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FailingResponse(FakeResponse):
    """Response that fails after yielding a bounded prefix."""

    def __init__(self, data: bytes, fail_after: int) -> None:
        super().__init__(data)
        self.fail_after = fail_after

    def read(self, size: int | None = -1) -> bytes:
        if self.tell() >= self.fail_after:
            raise OSError("connection interrupted")
        limit = self.fail_after - self.tell()
        return super().read(limit if size is None or size < 0 else min(size, limit))


def store(tmp_path: Path, downloader: FakeDownloader, **kwargs: Any) -> ImageStore:
    return ImageStore(tmp_path / "cache", downloader=downloader, **kwargs)


def urls(base: str = "https://images.test/noble/current/") -> tuple[str, str]:
    return f"{base}SHA256SUMS", f"{base}{IMAGE_NAME}"


def record_sleeps(sleeps: list[float]) -> Callable[[float, threading.Event | None], None]:
    return lambda seconds, _cancelled: sleeps.append(seconds)


def test_downloads_streams_verifies_and_writes_metadata(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {manifest_url: [FakeResponse(MANIFEST.encode())], image_url: [FakeResponse(IMAGE_BYTES)]}
    )
    image = store(tmp_path, downloader, base_url="https://images.test/noble/current/").resolve()

    assert image.path.read_bytes() == IMAGE_BYTES
    assert image.digest == IMAGE_DIGEST
    assert image.size_bytes == len(IMAGE_BYTES)
    assert image.checksum_verified is True
    assert image.signature_authenticated is False
    metadata = json.loads((tmp_path / "cache" / "releases" / "noble.json").read_text())
    assert metadata["digest"] == IMAGE_DIGEST
    assert metadata["signature_authenticated"] is False


def test_reuses_valid_cache_without_downloading(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {manifest_url: [FakeResponse(MANIFEST.encode())], image_url: [FakeResponse(IMAGE_BYTES)]}
    )
    image_store = store(tmp_path, downloader, base_url="https://images.test/noble/current/")
    first = image_store.resolve()
    second = image_store.resolve()

    assert second == first
    assert downloader.calls == [manifest_url, image_url]


def test_rehashes_corrupt_cache_before_redownloading(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(MANIFEST.encode()), FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES), FakeResponse(IMAGE_BYTES)],
        }
    )
    image_store = store(tmp_path, downloader, base_url="https://images.test/noble/current/")
    image_store.resolve().path.write_bytes(b"corrupt")
    image = image_store.resolve()

    assert image.path.read_bytes() == IMAGE_BYTES
    assert downloader.calls == [manifest_url, image_url, manifest_url, image_url]


def test_rejects_digest_mismatch_and_leaves_no_partial(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {manifest_url: [FakeResponse(MANIFEST.encode())], image_url: [FakeResponse(b"wrong")]}
    )
    image_store = store(tmp_path, downloader, base_url="https://images.test/noble/current/")

    with pytest.raises(ImageStoreError, match="checksum") as raised:
        image_store.resolve()
    assert raised.value.phase == ErrorPhase.PREPARATION
    assert raised.value.exit_code == ExitCode.PREPARATION
    assert not list((tmp_path / "cache").glob(".partial-*"))
    assert not (tmp_path / "cache" / "sha256").exists()


@pytest.mark.parametrize(
    "manifest",
    [
        f"{IMAGE_DIGEST}  other.img\n",
        f"{IMAGE_DIGEST}  {IMAGE_NAME}\n{IMAGE_DIGEST}  {IMAGE_NAME}\n",
        f"{IMAGE_DIGEST}  ../{IMAGE_NAME}\n",
        f"{IMAGE_DIGEST} {IMAGE_NAME}\n",
        f"not-a-checksum  {IMAGE_NAME}\n",
    ],
)
def test_rejects_invalid_or_unsafe_manifest(tmp_path: Path, manifest: str) -> None:
    manifest_url, _ = urls()
    downloader = FakeDownloader({manifest_url: [FakeResponse(manifest.encode())]})

    with pytest.raises(ImageStoreError):
        store(tmp_path, downloader, base_url="https://images.test/noble/current/").resolve()


def test_rejects_oversized_image_and_cleans_partial(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {manifest_url: [FakeResponse(MANIFEST.encode())], image_url: [FakeResponse(IMAGE_BYTES)]}
    )

    with pytest.raises(ImageStoreError, match="size limit"):
        store(
            tmp_path,
            downloader,
            base_url="https://images.test/noble/current/",
            max_image_bytes=3,
        ).resolve()
    assert not list((tmp_path / "cache").glob(".partial-*"))


def test_interrupted_image_download_cleans_partial(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(MANIFEST.encode())],
            image_url: [FailingResponse(IMAGE_BYTES, fail_after=4)],
        }
    )

    with pytest.raises(ImageStoreError, match="failed while downloading"):
        store(tmp_path, downloader, base_url="https://images.test/noble/current/").resolve()
    assert not list((tmp_path / "cache").glob(".partial-*"))
    assert not (tmp_path / "cache" / "releases" / "noble.json").exists()


def test_body_shorter_than_content_length_is_a_closed_connection(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    declared = {"Content-Length": str(len(IMAGE_BYTES))}
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES[:5], headers=declared)],
        }
    )

    with pytest.raises(ImageStoreError, match="connection closed before") as raised:
        store(tmp_path, downloader, base_url="https://images.test/noble/current/").resolve()

    assert raised.value.exit_code == ExitCode.PREPARATION
    assert raised.value.context["expected_bytes"] == len(IMAGE_BYTES)
    assert raised.value.context["received_bytes"] == 5
    assert not list((tmp_path / "cache").glob(".partial-*"))


def test_invalid_content_length_is_ignored(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES, headers={"Content-Length": "abc"})],
        }
    )

    image = store(tmp_path, downloader, base_url="https://images.test/noble/current/").resolve()

    assert image.digest == IMAGE_DIGEST


def test_manifest_size_limit_fails_before_image_download(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse((MANIFEST * 2).encode())],
            image_url: [FakeResponse(IMAGE_BYTES)],
        }
    )

    with pytest.raises(ImageStoreError, match="SHA256SUMS exceeds"):
        store(
            tmp_path,
            downloader,
            base_url="https://images.test/noble/current/",
            max_manifest_bytes=len(MANIFEST),
        ).resolve()
    assert downloader.calls == [manifest_url]


def test_failed_refresh_preserves_old_metadata_and_no_final_replacement(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(MANIFEST.encode()), FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES), FailingResponse(b"new", fail_after=1)],
        }
    )
    image_store = store(tmp_path, downloader, base_url="https://images.test/noble/current/")
    image = image_store.resolve()
    metadata_path = tmp_path / "cache" / "releases" / "noble.json"
    original_metadata = metadata_path.read_bytes()
    image.path.write_bytes(b"corrupt")

    with pytest.raises(ImageStoreError, match="failed while downloading"):
        image_store.resolve()
    assert metadata_path.read_bytes() == original_metadata
    assert not list((tmp_path / "cache").glob(".partial-*"))


def test_uses_per_release_lock_file(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {manifest_url: [FakeResponse(MANIFEST.encode())], image_url: [FakeResponse(IMAGE_BYTES)]}
    )

    store(tmp_path, downloader, base_url="https://images.test/noble/current/").resolve()

    assert (tmp_path / "cache" / ".locks" / "noble.lock").is_file()


def test_retries_transient_http_failure_with_bounded_backoff(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    sleeps: list[float] = []
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(b"", status=503), FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES)],
        }
    )
    image = store(
        tmp_path,
        downloader,
        base_url="https://images.test/noble/current/",
        sleeper=record_sleeps(sleeps),
        retry_base_seconds=0.25,
    ).resolve()

    assert image.digest == IMAGE_DIGEST
    assert sleeps == [0.25]


def test_retries_connection_failure_but_not_client_error(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    sleeps: list[float] = []
    downloader = FakeDownloader(
        {
            manifest_url: [ConnectionError("temporary"), FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES)],
        }
    )
    image = store(
        tmp_path,
        downloader,
        base_url="https://images.test/noble/current/",
        sleeper=record_sleeps(sleeps),
    ).resolve()
    assert image.digest == IMAGE_DIGEST
    assert sleeps == [0.5]
    assert downloader.calls == [manifest_url, manifest_url, image_url]


def test_respects_retry_after_header(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    sleeps: list[float] = []
    downloader = FakeDownloader(
        {
            manifest_url: [
                FakeResponse(b"", status=429, headers={"Retry-After": "2"}),
                FakeResponse(MANIFEST.encode()),
            ],
            image_url: [FakeResponse(IMAGE_BYTES)],
        }
    )
    image = store(
        tmp_path,
        downloader,
        base_url="https://images.test/noble/current/",
        sleeper=record_sleeps(sleeps),
    ).resolve()
    assert image.digest == IMAGE_DIGEST
    assert sleeps == [2.0]


def test_does_not_retry_client_error(tmp_path: Path) -> None:
    manifest_url, _ = urls()
    downloader = FakeDownloader({manifest_url: [FakeResponse(b"", status=404)]})

    with pytest.raises(ImageStoreError, match="HTTP 404"):
        store(tmp_path, downloader, base_url="https://images.test/noble/current/").resolve()
    assert downloader.calls == [manifest_url]


def test_injected_clock_and_command_boundary_are_accepted(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {manifest_url: [FakeResponse(MANIFEST.encode())], image_url: [FakeResponse(IMAGE_BYTES)]}
    )
    image = store(
        tmp_path,
        downloader,
        base_url="https://images.test/noble/current/",
        command_runner=CommandRunner(),
        wall_clock=lambda: 123.5,
    ).resolve()

    assert image.retrieved_at == 123.5


def test_rejects_unsupported_image_spec(tmp_path: Path) -> None:
    downloader = FakeDownloader({})
    with pytest.raises(ImageStoreError, match="official Noble"):
        store(tmp_path, downloader).resolve(ImageSpec.model_construct(architecture="arm64"))


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[threading.Event | None] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float, cancelled: threading.Event | None) -> None:
        self.waits.append(cancelled)
        self.advance(seconds)


class AdvancingResponse(FakeResponse):
    def __init__(self, data: bytes, clock: FakeClock) -> None:
        super().__init__(data)
        self.clock = clock

    def read(self, size: int | None = -1) -> bytes:
        self.clock.advance(0.25)
        return super().read(size)


@pytest.mark.parametrize("separator", [" *", "  "])
def test_parses_real_manifest_modes_and_selects_exact_basename(
    tmp_path: Path, separator: str
) -> None:
    # Ubuntu's binary-mode layout, including variants that must not be selected.
    manifest = (
        "00b9709d33192c8cfb987fd9522e9a035c73fd125c5a479b125b5d26ffafec67"
        " *noble-server-cloudimg-amd64.squashfs\n"
        f"{'1' * 64} *noble-server-cloudimg-arm64.img\n"
        f"{IMAGE_DIGEST.upper()}{separator}{IMAGE_NAME}\n"
    )
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(manifest.encode())],
            image_url: [FakeResponse(IMAGE_BYTES)],
        }
    )

    result = store(tmp_path, downloader, base_url="https://images.test/noble/current/").resolve()

    assert result.digest == IMAGE_DIGEST
    assert downloader.calls == [manifest_url, image_url]


@pytest.mark.parametrize("name", ["../bad.img", r"..\bad.img", ".", "..", "/bad.img", "bad name.img"])
def test_rejects_unsafe_binary_manifest_names(tmp_path: Path, name: str) -> None:
    manifest_url, _ = urls()
    manifest = f"{IMAGE_DIGEST} *{name}\n{MANIFEST}"
    downloader = FakeDownloader({manifest_url: [FakeResponse(manifest.encode())]})

    with pytest.raises(ImageStoreError, match=r"(invalid|unsafe) SHA256SUMS"):
        store(tmp_path, downloader, base_url="https://images.test/noble/current/").resolve()


def test_cleans_crash_leftovers_even_on_cache_hit(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES)],
        }
    )
    image_store = store(tmp_path, downloader, base_url="https://images.test/noble/current/")
    expected = image_store.resolve()
    cache = tmp_path / "cache"
    partial = cache / ".partial-stale"
    metadata = cache / "releases" / ".metadata-stale"
    partial.write_bytes(b"incomplete image")
    metadata.write_text("incomplete metadata")
    outside = tmp_path / "keep.img"
    outside.write_bytes(b"do not touch")
    link = cache / ".partial-symlink"
    link.symlink_to(outside)

    assert image_store.resolve() == expected

    assert not partial.exists()
    assert not metadata.exists()
    assert not link.is_symlink()
    assert outside.read_bytes() == b"do not touch"
    assert downloader.calls == [manifest_url, image_url]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_cache_is_a_preparation_failure(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o500)
    try:
        with pytest.raises(ImageStoreError, match="cannot use the image cache") as raised:
            store(tmp_path, FakeDownloader({})).resolve()
    finally:
        cache.chmod(0o700)

    assert raised.value.exit_code == ExitCode.PREPARATION
    assert isinstance(raised.value.__cause__, PermissionError)


def test_undeletable_leftover_is_a_preparation_failure(tmp_path: Path) -> None:
    (tmp_path / "cache" / ".partial-dir").mkdir(parents=True)

    with pytest.raises(ImageStoreError, match="cannot use the image cache") as raised:
        store(tmp_path, FakeDownloader({})).resolve()

    assert raised.value.exit_code == ExitCode.PREPARATION
    assert isinstance(raised.value.__cause__, IsADirectoryError)


@pytest.mark.parametrize("status", [429, 503])
def test_retries_real_urllib_http_errors(tmp_path: Path, status: int) -> None:
    manifest_url, image_url = urls()
    headers = Message()
    headers["Retry-After"] = "2"
    error_body = io.BytesIO(b"busy")
    error = HTTPError(manifest_url, status, "busy", headers, error_body)
    sleeps: list[float] = []
    downloader = FakeDownloader(
        {
            manifest_url: [error, FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES)],
        }
    )

    result = store(
        tmp_path,
        downloader,
        base_url="https://images.test/noble/current/",
        sleeper=record_sleeps(sleeps),
    ).resolve()

    assert result.digest == IMAGE_DIGEST
    assert sleeps == [2.0]
    assert error_body.closed
    assert downloader.calls == [manifest_url, manifest_url, image_url]


def test_real_urllib_404_is_closed_and_not_retried(tmp_path: Path) -> None:
    manifest_url, _ = urls()
    error_body = io.BytesIO(b"missing")
    downloader = FakeDownloader(
        {manifest_url: [HTTPError(manifest_url, 404, "missing", Message(), error_body)]}
    )
    sleeps: list[float] = []

    with pytest.raises(ImageStoreError, match="HTTP 404"):
        store(
            tmp_path,
            downloader,
            base_url="https://images.test/noble/current/",
            sleeper=record_sleeps(sleeps),
        ).resolve()

    assert downloader.calls == [manifest_url]
    assert sleeps == []
    assert error_body.closed


def test_contending_resolvers_download_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloading = threading.Event()
    release = threading.Event()
    contention = threading.Event()
    native_flock = fcntl.flock

    def observe_flock(fd: int, operation: int) -> None:
        try:
            native_flock(fd, operation)
        except BlockingIOError:
            contention.set()
            raise

    class GatedResponse(FakeResponse):
        def read(self, size: int | None = -1) -> bytes:
            downloading.set()
            if not release.wait(3):
                raise TimeoutError("test did not release download")
            return super().read(size)

    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(MANIFEST.encode())],
            image_url: [GatedResponse(IMAGE_BYTES)],
        }
    )
    monkeypatch.setattr(fcntl, "flock", observe_flock)
    first = store(tmp_path, downloader, base_url="https://images.test/noble/current/")
    second = store(tmp_path, downloader, base_url="https://images.test/noble/current/")
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending_first = pool.submit(first.resolve)
        try:
            assert downloading.wait(2)
            pending_second = pool.submit(second.resolve)
            assert contention.wait(2)
            # Acquiring another resolver's lock must not remove its live partial.
            assert len(list((tmp_path / "cache").glob(".partial-*"))) == 1
        finally:
            release.set()
        assert pending_first.result(timeout=3) == pending_second.result(timeout=3)

    assert downloader.calls == [manifest_url, image_url]


def test_lock_deadline_uses_fake_clock_and_preserves_live_partials(tmp_path: Path) -> None:
    clock = FakeClock()
    cache = tmp_path / "cache"
    locks = cache / ".locks"
    locks.mkdir(parents=True)
    partial = cache / ".partial-active"
    partial.write_bytes(b"other owner")
    downloader = FakeDownloader({})
    image_store = store(
        tmp_path,
        downloader,
        monotonic=clock,
        sleeper=clock.sleep,
        resolve_timeout_seconds=0.25,
    )

    with (locks / "noble.lock").open("a+b") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ImageStoreError, match=r"timed out.*lock") as raised:
            image_store.resolve()

    assert raised.value.exit_code == ExitCode.TIMEOUT
    assert clock.now == pytest.approx(0.25)
    # Lock polling has nothing to cancel, so it never passes an event.
    assert clock.waits
    assert all(cancelled is None for cancelled in clock.waits)
    assert partial.read_bytes() == b"other owner"
    assert downloader.calls == []


@pytest.mark.parametrize("phase", ["manifest", "image"])
def test_download_deadline_with_fake_clock_cleans_partial(tmp_path: Path, phase: str) -> None:
    clock = FakeClock()
    manifest_url, image_url = urls()
    manifest = (
        AdvancingResponse(MANIFEST.encode(), clock)
        if phase == "manifest"
        else FakeResponse(MANIFEST.encode())
    )
    image = AdvancingResponse(IMAGE_BYTES, clock) if phase == "image" else FakeResponse(IMAGE_BYTES)
    downloader = FakeDownloader({manifest_url: [manifest], image_url: [image]})

    with pytest.raises(ImageStoreError, match="timed out") as raised:
        store(
            tmp_path,
            downloader,
            base_url="https://images.test/noble/current/",
            monotonic=clock,
            chunk_size=1,
            resolve_timeout_seconds=0.5,
        ).resolve()

    # The timeout crosses the producer-thread boundary without losing its exit code.
    assert raised.value.exit_code == ExitCode.TIMEOUT
    assert not list((tmp_path / "cache").glob(".partial-*"))
    assert not list((tmp_path / "cache" / "sha256").glob("*.img"))
    assert not (tmp_path / "cache" / "releases" / "noble.json").exists()


def test_manifest_and_image_share_one_budget(tmp_path: Path) -> None:
    clock = FakeClock()
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [AdvancingResponse(MANIFEST.encode(), clock)],
            image_url: [AdvancingResponse(IMAGE_BYTES, clock)],
        }
    )

    with pytest.raises(ImageStoreError, match="timed out"):
        store(
            tmp_path,
            downloader,
            base_url="https://images.test/noble/current/",
            monotonic=clock,
            resolve_timeout_seconds=1.0,
        ).resolve()

    assert clock.now == pytest.approx(1.0)
    assert downloader.calls == [manifest_url, image_url]
    assert not list((tmp_path / "cache").glob(".partial-*"))
    assert not (tmp_path / "cache" / "releases" / "noble.json").exists()


def test_retry_backoff_cannot_extend_parent_deadline(tmp_path: Path) -> None:
    clock = FakeClock()
    manifest_url, _ = urls()
    headers = Message()
    headers["Retry-After"] = "60"
    downloader = FakeDownloader({manifest_url: [HTTPError(manifest_url, 503, "busy", headers, None)]})
    sleeps: list[float] = []

    def advance(seconds: float, _cancelled: threading.Event | None) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    with pytest.raises(ImageStoreError, match="timed out"):
        store(
            tmp_path,
            downloader,
            base_url="https://images.test/noble/current/",
            monotonic=clock,
            sleeper=advance,
        ).resolve(deadline=0.25)

    assert sleeps == [0.25]
    assert downloader.calls == [manifest_url]


@pytest.mark.parametrize("digest", ["../../outside", "..\\..\\outside", None, ["invalid"]])
def test_tampered_metadata_digest_is_rejected_before_path_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, digest: object
) -> None:
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(MANIFEST.encode()), FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES), FakeResponse(IMAGE_BYTES)],
        }
    )
    image_store = store(tmp_path, downloader, base_url="https://images.test/noble/current/")
    image_store.resolve()
    pointer = tmp_path / "cache" / "releases" / "noble.json"
    metadata = json.loads(pointer.read_text())
    metadata["digest"] = digest
    pointer.write_text(json.dumps(metadata))
    native_open = Path.open
    opened: list[Path] = []

    def record_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(path)
        return native_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)

    result = image_store.resolve()

    assert result.digest == IMAGE_DIGEST
    assert downloader.calls == [manifest_url, image_url, manifest_url, image_url]
    assert all(".." not in path.parts for path in opened)
    assert all("outside" not in str(path) for path in opened)


def test_cache_rehash_obeys_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    manifest_url, image_url = urls()
    downloader = FakeDownloader(
        {
            manifest_url: [FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES)],
        }
    )
    image_store = store(
        tmp_path,
        downloader,
        base_url="https://images.test/noble/current/",
        monotonic=clock,
        chunk_size=1,
    )
    cached = image_store.resolve()
    native_open = Path.open

    def advancing_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == cached.path and args == ("rb",):
            return AdvancingResponse(IMAGE_BYTES, clock)
        return native_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", advancing_open)

    with pytest.raises(ImageStoreError, match=r"timed out.*cached image"):
        image_store.resolve(deadline=0.5)

    assert downloader.calls == [manifest_url, image_url]


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_unbounded_resolve_timeout(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="resolve timeout must be positive and finite"):
        ImageStore(tmp_path, resolve_timeout_seconds=value)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_rejects_unbounded_explicit_deadline(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="deadline must be finite"):
        ImageStore(tmp_path).resolve(deadline=value)


def test_expired_deadline_does_not_create_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    with pytest.raises(ImageStoreError, match="timed out"):
        ImageStore(cache, monotonic=lambda: 1.0).resolve(deadline=1.0)
    assert not cache.exists()


@pytest.mark.parametrize("phase", ["headers", "manifest", "image"])
def test_default_downloader_bounds_blocked_http_operations(tmp_path: Path, phase: str) -> None:
    blocked = threading.Event()
    release = threading.Event()
    served = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            is_manifest = self.path.endswith("SHA256SUMS")
            data = MANIFEST.encode() if is_manifest else IMAGE_BYTES
            should_block = phase != "image" or not is_manifest
            try:
                if should_block and phase == "headers":
                    blocked.set()
                    assert release.wait(3)
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if should_block and phase != "headers":
                    self.wfile.write(data[:1])
                    self.wfile.flush()
                    blocked.set()
                    assert release.wait(3)
                    self.wfile.write(data[1:])
                else:
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if should_block:
                    served.set()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    server_thread.start()
    image_store = ImageStore(
        tmp_path / "cache",
        base_url=f"http://127.0.0.1:{server.server_port}/",
        resolve_timeout_seconds=0.4,
    )
    start = time.monotonic()
    try:
        with pytest.raises(ImageStoreError, match="timed out"):
            image_store.resolve()
        assert blocked.is_set()
        assert time.monotonic() - start < 1.5
        assert not list((tmp_path / "cache").glob(".partial-*"))
        assert not (tmp_path / "cache" / "releases" / "noble.json").exists()
    finally:
        release.set()
        assert served.wait(3)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)
        assert not server_thread.is_alive()


def test_retry_backoff_receives_the_download_cancellation_event(tmp_path: Path) -> None:
    manifest_url, image_url = urls()
    waits: list[tuple[float, threading.Event | None]] = []
    downloader = FakeDownloader(
        {
            manifest_url: [ConnectionError("temporary"), FakeResponse(MANIFEST.encode())],
            image_url: [FakeResponse(IMAGE_BYTES)],
        }
    )

    store(
        tmp_path,
        downloader,
        base_url="https://images.test/noble/current/",
        sleeper=lambda seconds, cancelled: waits.append((seconds, cancelled)),
    ).resolve()

    assert len(waits) == 1
    seconds, cancelled = waits[0]
    assert seconds == 0.5
    assert isinstance(cancelled, threading.Event)
    # Closing the manifest stream cancels its producer, which wakes a default backoff.
    assert cancelled.is_set()


def test_default_sleeper_returns_early_once_cancelled() -> None:
    cancelled = threading.Event()
    cancelled.set()
    started = time.monotonic()

    _sleep_until_cancelled(5.0, cancelled)

    assert time.monotonic() - started < 1.0


def test_default_sleeper_without_event_sleeps_for_the_full_delay() -> None:
    started = time.monotonic()

    _sleep_until_cancelled(0.05, None)

    assert time.monotonic() - started >= 0.04
