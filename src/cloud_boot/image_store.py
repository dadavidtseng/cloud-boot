# ----------------------------------------------------------------------------------------------------
# image_store.py
# ----------------------------------------------------------------------------------------------------

"""Checksum-verified Ubuntu cloud-image download and cache primitives."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import math
import os
import queue
import re
import tempfile
import threading
import time
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import CloudBootError, ErrorPhase, ExitCode
from .models import ImageSpec
from .process import Clock, CommandRunner

__all__ = ["ImageStore", "ImageStoreError", "VerifiedImage"]

DEFAULT_BASE_URL = "https://cloud-images.ubuntu.com/noble/current/"
DEFAULT_MAX_IMAGE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RESOLVE_TIMEOUT_SECONDS = 30 * 60.0
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_LINE_PATTERN = re.compile(r"^([0-9a-fA-F]{64}) [ *]([^\s/\\]+)$")

type CancellableSleeper = Callable[[float, threading.Event | None], object]
type Downloader = Callable[[str], IO[bytes]]


def _sleep_until_cancelled(seconds: float, cancelled: threading.Event | None) -> None:
    """Sleep for `seconds`, returning early once `cancelled` is set."""
    if cancelled is None:
        time.sleep(seconds)
    else:
        cancelled.wait(seconds)


class ImageStoreError(CloudBootError):
    """Expected failure while resolving or caching a cloud image.

    Failures default to the preparation exit code; deadline expiry uses the timeout exit code.
    """

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        exit_code: ExitCode = ExitCode.PREPARATION,
    ) -> None:
        super().__init__(
            message,
            code="image-store-error",
            phase=ErrorPhase.PREPARATION,
            context=context,
            exit_code=exit_code,
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
    """Resolve and cache the checksum-verified Noble image.

    A single monotonic budget covers lock acquisition, cache hashing, network
    reads, and retries. Cached releases stay pinned until their metadata is
    removed. The optional command runner is reserved for signature-verification
    tools; checksum-only resolution never invokes it. The sleeper receives the
    download's cancellation event during retry backoff and `None` while polling
    the cache lock, so the default backoff returns early once a resolve gives up.

    Network operations run in a daemon producer with bounded buffering. On
    expiry the caller cancels consumption and removes its partial file. An
    uncooperative downloader may keep its worker blocked until its I/O returns,
    but the worker never owns cache files or writes after cancellation.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        downloader: Downloader | None = None,
        command_runner: CommandRunner | None = None,
        wall_clock: Clock = time.time,
        monotonic: Clock = time.monotonic,
        sleeper: CancellableSleeper = _sleep_until_cancelled,
        base_url: str = DEFAULT_BASE_URL,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_base_seconds: float = 0.5,
        resolve_timeout_seconds: float = DEFAULT_RESOLVE_TIMEOUT_SECONDS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._downloader = downloader or self._urllib_downloader
        self._command_runner = command_runner
        self._wall_clock = wall_clock
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._base_url = base_url.rstrip("/") + "/"
        self._max_image_bytes = max_image_bytes
        self._max_manifest_bytes = max_manifest_bytes
        self._chunk_size = chunk_size
        self._retry_attempts = retry_attempts
        self._retry_base_seconds = retry_base_seconds
        self._resolve_timeout_seconds = resolve_timeout_seconds
        if max_image_bytes <= 0 or max_manifest_bytes <= 0 or chunk_size <= 0:
            raise ValueError("image and manifest limits and chunk size must be positive")
        if retry_attempts <= 0 or retry_base_seconds < 0 or not math.isfinite(retry_base_seconds):
            raise ValueError("retry settings must be positive and finite")
        if not math.isfinite(resolve_timeout_seconds) or resolve_timeout_seconds <= 0:
            raise ValueError("resolve timeout must be positive and finite")

    def resolve(
        self, spec: ImageSpec | None = None, *, deadline: float | None = None
    ) -> VerifiedImage:
        """Return a rehashed cache hit or download within one overall deadline.

        An explicit deadline uses the injected monotonic clock and may shorten
        the configured timeout. It never extends the store's default budget.
        """
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        end = self._monotonic() + self._resolve_timeout_seconds
        if deadline is not None:
            end = min(end, deadline)
        self._remaining(end, "image resolution")
        image_spec = spec or ImageSpec()
        self._validate_spec(image_spec)
        basename = self._basename(image_spec)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_dir = self.cache_dir / ".locks"
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        lock_path = lock_dir / f"{image_spec.release}.lock"
        with lock_path.open("a+b") as lock_file:
            self._acquire_lock(lock_file.fileno(), end)
            try:
                self._remove_stale_partials(end)
                cached = self._read_cached(image_spec, basename, end)
                if cached is not None:
                    return cached
                return self._download_and_cache(image_spec, basename, end)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _remaining(self, deadline: float, operation: str) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise ImageStoreError(f"timed out during {operation}", exit_code=ExitCode.TIMEOUT)
        return remaining

    def _acquire_lock(self, fd: int, deadline: float) -> None:
        while True:
            self._remaining(deadline, "image cache lock acquisition")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                remaining = self._remaining(deadline, "image cache lock acquisition")
                self._sleeper(min(0.1, remaining), None)

    def _remove_stale_partials(self, deadline: float) -> None:
        # Only Noble is supported, so every temporary here belongs to this lock.
        # Multiple releases must first move temporaries into release-scoped dirs.
        for directory, pattern in (
            (self.cache_dir, ".partial-*"),
            (self.cache_dir / "releases", ".metadata-*"),
        ):
            for stale in directory.glob(pattern):
                self._remaining(deadline, "stale cache cleanup")
                with contextlib.suppress(FileNotFoundError):
                    stale.unlink()

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

    def _read_cached(self, spec: ImageSpec, basename: str, deadline: float) -> VerifiedImage | None:
        self._remaining(deadline, "cached image verification")
        pointer = self.cache_dir / "releases" / f"{spec.release}.json"
        try:
            metadata = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(metadata, dict):
            return None
        digest = metadata.get("digest")
        size_bytes = metadata.get("size_bytes")
        retrieved_at = metadata.get("retrieved_at")
        source_url = metadata.get("source_url")
        if (
            metadata.get("basename") != basename
            or metadata.get("checksum_verified") is not True
            or metadata.get("signature_authenticated") is not False
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or type(size_bytes) is not int
            or not 0 <= size_bytes <= self._max_image_bytes
            or not isinstance(retrieved_at, (int, float))
            or isinstance(retrieved_at, bool)
            or not math.isfinite(retrieved_at)
            or source_url != f"{self._base_url}{basename}"
        ):
            return None
        # Validate the digest before constructing or accessing its filesystem path.
        cached_path = self.cache_dir / "sha256" / f"{digest}.img"
        actual_digest, actual_size = self._hash_file(cached_path, deadline)
        if actual_size != size_bytes or not hmac.compare_digest(actual_digest, digest):
            return None
        return VerifiedImage(
            path=cached_path,
            digest=digest,
            size_bytes=actual_size,
            source_url=source_url,
            retrieved_at=float(retrieved_at),
        )

    def _download_and_cache(self, spec: ImageSpec, basename: str, deadline: float) -> VerifiedImage:
        manifest_url = f"{self._base_url}SHA256SUMS"
        image_url = f"{self._base_url}{basename}"
        manifest = self._read_manifest(manifest_url, deadline)
        expected_digest = self._parse_manifest(manifest, basename)
        partial_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".partial-", dir=self.cache_dir, delete=False
            ) as partial:
                partial_path = Path(partial.name)
                digest, size_bytes = self._stream_image(image_url, partial, deadline)
                partial.flush()
                os.fsync(partial.fileno())
            if not hmac.compare_digest(digest, expected_digest):
                raise ImageStoreError(
                    "downloaded image checksum does not match SHA256SUMS",
                    context={"expected_digest": expected_digest, "actual_digest": digest},
                )
            self._remaining(deadline, "image cache publication")
            final_dir = self.cache_dir / "sha256"
            final_dir.mkdir(mode=0o700, exist_ok=True)
            final_path = final_dir / f"{digest}.img"
            os.replace(partial_path, final_path)
            partial_path = None
            image = VerifiedImage(
                path=final_path,
                digest=digest,
                size_bytes=size_bytes,
                source_url=image_url,
                retrieved_at=self._wall_clock(),
            )
            self._write_metadata(spec, basename, image, deadline)
            return image
        finally:
            if partial_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    partial_path.unlink()

    def _download_chunks(self, url: str, deadline: float) -> Generator[bytes, None, None]:
        # A buffered read (including urllib's header parsing) can perform many
        # socket reads, each resetting the inactivity timeout. Isolate all
        # network I/O, including open/close, from the caller's deadline.
        messages: queue.Queue[bytes | Exception | None] = queue.Queue(maxsize=1)
        cancelled = threading.Event()

        def publish(value: bytes | Exception | None) -> None:
            while not cancelled.is_set():
                try:
                    messages.put(value, timeout=0.05)
                    return
                except queue.Full:
                    continue

        def produce() -> None:
            response: IO[bytes] | None = None
            failure: Exception | None = None
            try:
                response = self._open_with_retry(url, deadline, cancelled)
                while not cancelled.is_set():
                    self._remaining(deadline, "image download")
                    chunk = response.read(self._chunk_size)
                    if not chunk:
                        break
                    if len(chunk) > self._chunk_size:
                        raise ImageStoreError("downloader exceeded the requested chunk size")
                    publish(chunk)
            except Exception as error:
                failure = error
            finally:
                if response is not None:
                    with contextlib.suppress(Exception):
                        response.close()
            publish(failure)

        threading.Thread(target=produce, name="cloud-boot-download", daemon=True).start()
        try:
            while True:
                remaining = self._remaining(deadline, "image download")
                try:
                    message = messages.get(timeout=min(remaining, 0.05))
                except queue.Empty:
                    continue
                self._remaining(deadline, "image download")
                if message is None:
                    return
                if isinstance(message, Exception):
                    if isinstance(message, ImageStoreError):
                        raise message
                    raise ImageStoreError(
                        "failed while downloading image data", context={"url": url}
                    ) from message
                yield message
        finally:
            cancelled.set()

    def _read_manifest(self, url: str, deadline: float) -> bytes:
        data = bytearray()
        with contextlib.closing(self._download_chunks(url, deadline)) as chunks:
            for chunk in chunks:
                if len(data) + len(chunk) > self._max_manifest_bytes:
                    raise ImageStoreError("SHA256SUMS exceeds the configured size limit")
                data.extend(chunk)
        return bytes(data)

    def _stream_image(self, url: str, destination: IO[bytes], deadline: float) -> tuple[str, int]:
        hasher = hashlib.sha256()
        size_bytes = 0
        with contextlib.closing(self._download_chunks(url, deadline)) as chunks:
            for chunk in chunks:
                size_bytes += len(chunk)
                if size_bytes > self._max_image_bytes:
                    raise ImageStoreError("downloaded image exceeds the configured size limit")
                hasher.update(chunk)
                destination.write(chunk)
                self._remaining(deadline, "image download")
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
            digest, name = match.groups()
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

    def _write_metadata(
        self, spec: ImageSpec, basename: str, image: VerifiedImage, deadline: float
    ) -> None:
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
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=".metadata-", dir=releases, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(metadata, temporary, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            self._remaining(deadline, "image metadata publication")
            os.replace(temporary_path, pointer)
        finally:
            if temporary_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    temporary_path.unlink()

    def _hash_file(self, path: Path, deadline: float) -> tuple[str, int]:
        hasher = hashlib.sha256()
        size_bytes = 0
        try:
            with path.open("rb") as source:
                while True:
                    self._remaining(deadline, "cached image verification")
                    chunk = source.read(self._chunk_size)
                    self._remaining(deadline, "cached image verification")
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > self._max_image_bytes:
                        return "", size_bytes
                    hasher.update(chunk)
        except OSError:
            return "", 0
        return hasher.hexdigest(), size_bytes

    def _open_with_retry(self, url: str, deadline: float, cancelled: threading.Event) -> IO[bytes]:
        last_error: Exception | None = None
        for attempt in range(self._retry_attempts):
            self._remaining(deadline, "image download")
            if cancelled.is_set():
                raise ImageStoreError("image download cancelled")
            try:
                response = self._downloader(url)
                status = int(getattr(response, "status", 200))
                if 200 <= status < 300:
                    return response
                try:
                    delay = self._retry_delay(response, attempt)
                finally:
                    response.close()
                if not 500 <= status < 600 and status != 429:
                    raise ImageStoreError(
                        f"image server returned HTTP {status}", context={"url": url}
                    )
                last_error = ImageStoreError(
                    f"image server returned HTTP {status}", context={"url": url}
                )
            except HTTPError as error:
                last_error = error
                try:
                    if not 500 <= error.code < 600 and error.code != 429:
                        raise ImageStoreError(
                            f"image server returned HTTP {error.code}", context={"url": url}
                        ) from error
                    delay = self._retry_delay(error, attempt)
                finally:
                    error.close()
            except (OSError, URLError) as error:
                last_error = error
                delay = self._retry_base_seconds * (2**attempt)
            if attempt + 1 < self._retry_attempts:
                delay = min(delay, self._remaining(deadline, "download retry backoff"))
                self._sleeper(delay, cancelled)
        raise ImageStoreError(
            "image download failed after bounded retries", context={"url": url}
        ) from last_error

    def _retry_delay(self, response: Any, attempt: int) -> float:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
        if retry_after:
            try:
                seconds = float(retry_after)
                if math.isfinite(seconds):
                    return max(0.0, min(seconds, 60.0))
            except (TypeError, ValueError):
                with contextlib.suppress(TypeError, ValueError, OverflowError):
                    retry_at = parsedate_to_datetime(str(retry_after)).timestamp()
                    return max(0.0, min(retry_at - self._wall_clock(), 60.0))
        return float(self._retry_base_seconds * (2**attempt))

    @staticmethod
    def _urllib_downloader(url: str) -> IO[bytes]:
        request = Request(url, headers={"User-Agent": "cloud-boot/0.1"})
        return cast(IO[bytes], urlopen(request, timeout=30.0))
