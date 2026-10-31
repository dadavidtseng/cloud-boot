"""Checksum-verified Ubuntu cloud-image download and cache primitives."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import CloudBootError, ErrorPhase
from .models import ImageSpec

__all__ = ["ImageStore", "ImageStoreError", "VerifiedImage"]

DEFAULT_BASE_URL = "https://cloud-images.ubuntu.com/noble/current/"
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_RETRY_ATTEMPTS = 3
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_LINE_PATTERN = re.compile(r"^([0-9a-fA-F]{64})  (\*?)([^\s]+)$")

type Clock = Callable[[], float]
type Sleeper = Callable[[float], None]
type Downloader = Callable[[str], IO[bytes]]


class ImageStoreError(CloudBootError):
    """Expected failure while resolving or caching a cloud image."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="image-store-error",
            phase=ErrorPhase.PREPARATION,
            context=context,
        )


@dataclass(frozen=True, slots=True)
class VerifiedImage:
    """Immutable identity and provenance for a checksum-verified image."""

    path: Path
    digest: str
    size_bytes: int
    source_url: str
    retrieved_at: float
    checksum_verified: bool = True
    signature_authenticated: bool = False


class ImageStore:
    """Resolve, verify, and atomically cache the official Noble cloud image."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        downloader: Downloader | None = None,
        command_runner: object | None = None,
        clock: Clock = time.time,
        sleeper: Sleeper = time.sleep,
        base_url: str = DEFAULT_BASE_URL,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_base_seconds: float = 0.5,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._downloader = downloader or self._urllib_downloader
        self._command_runner = command_runner
        self._clock = clock
        self._sleeper = sleeper
        self._base_url = base_url.rstrip("/") + "/"
        self._max_image_bytes = max_image_bytes
        self._max_manifest_bytes = max_manifest_bytes
        self._chunk_size = chunk_size
        self._retry_attempts = retry_attempts
        self._retry_base_seconds = retry_base_seconds
        if max_image_bytes <= 0 or max_manifest_bytes <= 0 or chunk_size <= 0:
            raise ValueError("image and manifest limits and chunk size must be positive")
        if retry_attempts <= 0 or retry_base_seconds < 0 or not retry_base_seconds < float("inf"):
            raise ValueError("retry settings must be positive and finite")

    def resolve(self, spec: ImageSpec | None = None) -> VerifiedImage:
        """Return a verified cached image, downloading it when necessary."""
        image_spec = spec or ImageSpec()
        self._validate_spec(image_spec)
        basename = self._basename(image_spec)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_dir = self.cache_dir / ".locks"
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        lock_path = lock_dir / f"{image_spec.release}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                cached = self._read_cached(image_spec, basename)
                if cached is not None:
                    return cached
                return self._download_and_cache(image_spec, basename)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _validate_spec(self, spec: ImageSpec) -> None:
        if (spec.release, spec.architecture, spec.variant, spec.source) != (
            "noble",
            "amd64",
            "server-cloudimg",
            "official",
        ):
            raise ImageStoreError("only the official Noble amd64 server-cloudimg is supported")

    @staticmethod
    def _basename(spec: ImageSpec) -> str:
        return f"{spec.release}-{spec.variant}-{spec.architecture}.img"

    def _read_cached(self, spec: ImageSpec, basename: str) -> VerifiedImage | None:
        pointer = self.cache_dir / "releases" / f"{spec.release}.json"
        try:
            metadata = json.loads(pointer.read_text(encoding="utf-8"))
            digest = metadata["digest"]
            size_bytes = int(metadata["size_bytes"])
            source_url = metadata["source_url"]
            retrieved_at = float(metadata["retrieved_at"])
            cached_path = self.cache_dir / "sha256" / f"{digest}.img"
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if (
            metadata.get("basename") != basename
            or metadata.get("checksum_verified") is not True
            or metadata.get("signature_authenticated") is not False
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or not cached_path.is_file()
        ):
            return None
        actual_digest, actual_size = self._hash_file(cached_path)
        if actual_size != size_bytes or not hmac.compare_digest(actual_digest, digest):
            return None
        return VerifiedImage(
            path=cached_path,
            digest=digest,
            size_bytes=actual_size,
            source_url=source_url,
            retrieved_at=retrieved_at,
        )

    def _download_and_cache(self, spec: ImageSpec, basename: str) -> VerifiedImage:
        manifest_url = f"{self._base_url}SHA256SUMS"
        image_url = f"{self._base_url}{basename}"
        manifest = self._read_manifest(manifest_url)
        expected_digest = self._parse_manifest(manifest, basename)
        partial_path: Path | None = None
        try:
            response = self._open_with_retry(image_url)
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=".partial-", dir=self.cache_dir, delete=False
                ) as partial:
                    partial_path = Path(partial.name)
                    digest, size_bytes = self._stream_image(response, partial)
                    partial.flush()
                    os.fsync(partial.fileno())
                if not hmac.compare_digest(digest, expected_digest):
                    raise ImageStoreError(
                        "downloaded image checksum does not match SHA256SUMS",
                        context={"expected_digest": expected_digest, "actual_digest": digest},
                    )
            finally:
                with contextlib.suppress(Exception):
                    response.close()
            final_dir = self.cache_dir / "sha256"
            final_dir.mkdir(mode=0o700, exist_ok=True)
            final_path = final_dir / f"{digest}.img"
            os.replace(partial_path, final_path)
            partial_path = None
            retrieved_at = self._clock()
            image = VerifiedImage(
                path=final_path,
                digest=digest,
                size_bytes=size_bytes,
                source_url=image_url,
                retrieved_at=retrieved_at,
            )
            self._write_metadata(spec, basename, image)
            return image
        finally:
            if partial_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    partial_path.unlink()

    def _read_manifest(self, url: str) -> bytes:
        response = self._open_with_retry(url)
        try:
            data = response.read(self._max_manifest_bytes + 1)
        except OSError as error:
            raise ImageStoreError("failed to read SHA256SUMS", context={"url": url}) from error
        finally:
            with contextlib.suppress(Exception):
                response.close()
        if len(data) > self._max_manifest_bytes:
            raise ImageStoreError("SHA256SUMS exceeds the configured size limit")
        return data

    def _stream_image(self, response: IO[bytes], destination: IO[bytes]) -> tuple[str, int]:
        hasher = hashlib.sha256()
        size_bytes = 0
        while True:
            try:
                chunk = response.read(self._chunk_size)
            except OSError as error:
                raise ImageStoreError("failed while downloading the image") from error
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > self._max_image_bytes:
                raise ImageStoreError("downloaded image exceeds the configured size limit")
            hasher.update(chunk)
            destination.write(chunk)
        return hasher.hexdigest(), size_bytes

    def _parse_manifest(self, data: bytes, basename: str) -> str:
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError as error:
            raise ImageStoreError("SHA256SUMS is not valid ASCII") from error
        found: str | None = None
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            match = MANIFEST_LINE_PATTERN.fullmatch(line)
            if match is None:
                raise ImageStoreError(f"invalid SHA256SUMS entry on line {line_number}")
            digest, _, name = match.groups()
            if (
                PurePosixPath(name).name != name
                or PureWindowsPath(name).name != name
                or name in {".", ".."}
            ):
                raise ImageStoreError(f"unsafe SHA256SUMS filename on line {line_number}")
            if name == basename:
                if found is not None:
                    raise ImageStoreError(f"duplicate SHA256SUMS entry for {basename}")
                found = digest.lower()
        if found is None:
            raise ImageStoreError(f"SHA256SUMS does not contain the allowlisted image {basename}")
        return found

    def _write_metadata(self, spec: ImageSpec, basename: str, image: VerifiedImage) -> None:
        releases = self.cache_dir / "releases"
        releases.mkdir(mode=0o700, exist_ok=True)
        metadata: dict[str, object] = {
            "schema_version": 1,
            "release": spec.release,
            "architecture": spec.architecture,
            "variant": spec.variant,
            "source": spec.source,
            "basename": basename,
            "digest": image.digest,
            "size_bytes": image.size_bytes,
            "source_url": image.source_url,
            "retrieved_at": image.retrieved_at,
            "checksum_verified": True,
            "signature_authenticated": False,
        }
        pointer = releases / f"{spec.release}.json"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".metadata-", dir=releases, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(metadata, temporary, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.replace(temporary_path, pointer)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary_path.unlink()

    def _hash_file(self, path: Path) -> tuple[str, int]:
        hasher = hashlib.sha256()
        size_bytes = 0
        try:
            with path.open("rb") as source:
                while chunk := source.read(self._chunk_size):
                    size_bytes += len(chunk)
                    if size_bytes > self._max_image_bytes:
                        return "", size_bytes
                    hasher.update(chunk)
        except OSError:
            return "", 0
        return hasher.hexdigest(), size_bytes

    def _open_with_retry(self, url: str) -> IO[bytes]:
        last_error: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                response = self._downloader(url)
                status = int(getattr(response, "status", 200))
                if 200 <= status < 300:
                    return response
                with contextlib.suppress(Exception):
                    response.close()
                if status < 500 and status != 429:
                    raise ImageStoreError(
                        f"image server returned HTTP {status}", context={"url": url}
                    )
                last_error = ImageStoreError(
                    f"image server returned HTTP {status}", context={"url": url}
                )
                delay = self._retry_delay(response, attempt)
            except HTTPError as error:
                last_error = error
                if error.code < 500 and error.code != 429:
                    raise ImageStoreError(
                        f"image server returned HTTP {error.code}", context={"url": url}
                    ) from error
                delay = self._retry_delay(error, attempt)
            except (OSError, TimeoutError, URLError) as error:
                last_error = error
                delay = self._retry_base_seconds * (2**attempt)
            if attempt + 1 < self._retry_attempts:
                self._sleeper(delay)
        raise ImageStoreError(
            "image download failed after bounded retries", context={"url": url}
        ) from last_error

    def _retry_delay(self, response: Any, attempt: int) -> float:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), 60.0))
            except (TypeError, ValueError):
                with contextlib.suppress(TypeError, ValueError, OverflowError):
                    retry_at = parsedate_to_datetime(str(retry_after)).timestamp()
                    return max(0.0, min(retry_at - self._clock(), 60.0))
        return float(self._retry_base_seconds * (2**attempt))

    @staticmethod
    def _urllib_downloader(url: str) -> IO[bytes]:
        request = Request(url, headers={"User-Agent": "cloud-boot/0.1"})
        return cast(IO[bytes], urlopen(request, timeout=30.0))
