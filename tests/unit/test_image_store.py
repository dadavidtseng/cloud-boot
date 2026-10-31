"""Unit tests for checksum-verified image storage."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from cloud_boot.image_store import ImageStore, ImageStoreError
from cloud_boot.models import ImageSpec

IMAGE_NAME = "noble-server-cloudimg-amd64.img"
IMAGE_BYTES = b"cloud-boot image\n"
IMAGE_DIGEST = hashlib.sha256(IMAGE_BYTES).hexdigest()
MANIFEST = f"{IMAGE_DIGEST}  {IMAGE_NAME}\n"


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

    def read(self, size: int = -1) -> bytes:
        if self.tell() >= self.fail_after:
            raise OSError("connection interrupted")
        limit = self.fail_after - self.tell()
        return super().read(limit if size < 0 else min(size, limit))


def store(tmp_path: Path, downloader: FakeDownloader, **kwargs: object) -> ImageStore:
    return ImageStore(tmp_path / "cache", downloader=downloader, **kwargs)


def urls(base: str = "https://images.test/noble/current/") -> tuple[str, str]:
    return f"{base}SHA256SUMS", f"{base}{IMAGE_NAME}"


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

    with pytest.raises(ImageStoreError, match="checksum"):
        image_store.resolve()
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
        sleeper=sleeps.append,
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
        sleeper=sleeps.append,
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
        sleeper=sleeps.append,
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
        command_runner=object(),
        clock=lambda: 123.5,
    ).resolve()

    assert image.retrieved_at == 123.5


def test_rejects_unsupported_image_spec(tmp_path: Path) -> None:
    downloader = FakeDownloader({})
    with pytest.raises(ImageStoreError, match="official Noble"):
        store(tmp_path, downloader).resolve(ImageSpec.model_construct(architecture="arm64"))
