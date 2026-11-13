# ----------------------------------------------------------------------------------------------------
# image_store.py
# ----------------------------------------------------------------------------------------------------

"""Checksum-verified Ubuntu cloud-image download and cache primitives.

`ImageStore.resolve` returns a `VerifiedImage` whose bytes match the SHA-256 listed in the
release's `SHA256SUMS`. The manifest itself is not yet signature-checked (Task 12 adds `gpgv`),
so results are `checksum_verified` but never `signature_authenticated`.

Cache layout under `cache_dir`::

    .locks/<release>.lock     flock held for a whole resolve
    .partial-*                image download in progress; removed on the next resolve
    sha256/<digest>.img       content-addressed images
    releases/<release>.json   pointer to the release's current image and its provenance
    releases/.metadata-*      pointer write in progress
"""

# ----------------------------------------------------------------------------------------------------
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
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .errors import CloudBootError, ErrorPhase, ExitCode
from .models import ImageSpec
from .process import Clock, CommandRunner

# ----------------------------------------------------------------------------------------------------
__all__ = ["ImageStore", "ImageStoreError", "VerifiedImage"]

# ----------------------------------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://cloud-images.ubuntu.com/noble/current/"
DEFAULT_MAX_IMAGE_BYTES = 2 * 1024 * 1024 * 1024  # Noble images are about 600 MB.
DEFAULT_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_RETRY_ATTEMPTS = 3  # Total attempts, including the first.
DEFAULT_RESOLVE_TIMEOUT_SECONDS = 30 * 60.0
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
# One `sha256sum` output line: digest, then " " (text mode) or "*" (binary mode),
# then a bare filename with no directory part.
MANIFEST_LINE_PATTERN = re.compile(r"^([0-9a-fA-F]{64}) [ *]([^\s/\\]+)$")

# ----------------------------------------------------------------------------------------------------
# Sleeps up to the given seconds, returning early once the event is set; the result is ignored, so
# `threading.Event.wait` fits.
type CancellableSleeper = Callable[[float, threading.Event | None], object]

# Opens a URL for streaming; the result may carry urllib-style `status` and `headers`.
type Downloader = Callable[[str], IO[bytes]]


# ----------------------------------------------------------------------------------------------------
def _sleep_until_cancelled(seconds: float, cancelled: threading.Event | None) -> None:
    """Sleep for `seconds`, returning early once `cancelled` is set."""
    # If `cancelled` is empty, sleep for `seconds`,
    # otherwise, it's a cancellable event which relies on threading.Event
    if cancelled is None:
        time.sleep(seconds)
    else:
        cancelled.wait(seconds)


# ----------------------------------------------------------------------------------------------------
def _content_length(response: object) -> int | None:
    """Return the response's declared body size, or `None` if absent or invalid."""
    get = getattr(getattr(response, "headers", None), "get", None)
    value = get("Content-Length") if callable(get) else None

    if not isinstance(value, str):
        return None
    value = value.strip()

    # HTTP allows only ASCII digits; `isdigit` alone would accept characters like "²".
    return int(value) if value.isascii() and value.isdigit() else None


# ----------------------------------------------------------------------------------------------------
def _is_retryable(status: int) -> bool:
    """Return whether an HTTP status may clear up on retry: server errors and rate limiting."""
    return 500 <= status < 600 or status == 429


# ----------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VerifiedImage:
    """Immutable identity and provenance for a checksum-verified image.

    Attributes:
        path: Content-addressed cache file; the VM uses it as a read-only backing image.
        digest: Lowercase hex SHA-256 of the image bytes.
        size_bytes: Image size in bytes.
        source_url: URL the image was downloaded from.
        retrieved_at: Wall-clock time of the download, in seconds since the epoch.
        checksum_verified: The bytes match the digest listed in `SHA256SUMS`.
        signature_authenticated: `SHA256SUMS` was checked against Ubuntu's signing key;
            always `False` until Task 12.
    """

    path: Path
    digest: str
    size_bytes: int
    source_url: str
    retrieved_at: float
    checksum_verified: bool = True
    signature_authenticated: bool = False


# ----------------------------------------------------------------------------------------------------
class ImageStore:
    """Resolve and cache the checksum-verified Noble image.

    A single monotonic budget covers lock acquisition, cache hashing, network reads, and retries.
    Cached releases stay pinned until their metadata is removed. The sleeper receives the
    download's cancellation event during retry backoff and `None` while polling the cache lock,
    so the default backoff returns early once a resolve gives up.

    Network operations run in a daemon producer with bounded buffering. On expiry the caller
    cancels consumption and removes its partial file. An uncooperative downloader may keep its
    worker blocked until its I/O returns, but the worker never owns cache files or writes after
    cancellation.

    Args:
        cache_dir: Root of the cache; created on first use.
        downloader: Opens a URL for streaming; defaults to urllib with a 30-second socket timeout.
        command_runner: Reserved for signature verification (Task 12); never invoked yet.
        wall_clock: Source of `retrieved_at` and of `Retry-After` dates.
        monotonic: Clock for every deadline.
        sleeper: Waits between lock polls and download retries.
        base_url: Directory holding the image and `SHA256SUMS`.
        max_image_bytes: Largest image accepted, downloaded or cached.
        max_manifest_bytes: Largest `SHA256SUMS` accepted.
        chunk_size: Bytes per read, and the most the downloader may return at once.
        retry_attempts: Total download attempts per URL, including the first.
        retry_base_seconds: First backoff delay; doubles on each attempt.
        resolve_timeout_seconds: Budget for one `resolve` call.

    Raises:
        ValueError: A limit, retry setting, or timeout is out of range.
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
        if max_image_bytes <= 0 or max_manifest_bytes <= 0 or chunk_size <= 0:
            raise ValueError("image and manifest limits and chunk size must be positive")
        if retry_attempts <= 0 or retry_base_seconds < 0 or not math.isfinite(retry_base_seconds):
            raise ValueError(
                "retry attempts must be positive and the retry base delay finite and non-negative"
            )
        if not math.isfinite(resolve_timeout_seconds) or resolve_timeout_seconds <= 0:
            raise ValueError("resolve timeout must be positive and finite")

        self.cache_dir = Path(cache_dir)
        self._downloader = downloader or self._urllib_downloader
        self._command_runner = command_runner
        self._wall_clock = wall_clock
        self._monotonic = monotonic
        self._sleeper = sleeper
        # Exactly one trailing slash, so filenames can be appended directly.
        self._base_url = base_url.rstrip("/") + "/"
        self._max_image_bytes = max_image_bytes
        self._max_manifest_bytes = max_manifest_bytes
        self._chunk_size = chunk_size
        self._retry_attempts = retry_attempts
        self._retry_base_seconds = retry_base_seconds
        self._resolve_timeout_seconds = resolve_timeout_seconds

    def resolve(
        self, spec: ImageSpec | None = None, *, deadline: float | None = None
    ) -> VerifiedImage:
        """Return a verified image, rehashing a cache hit or downloading within one deadline.

        A cached release is reused without checking for a newer upstream image.

        Args:
            spec: Image to resolve; defaults to the official Noble amd64 image.
            deadline: Optional monotonic deadline; it may shorten the store's budget but never
                extends it.

        Returns:
            The verified image in the content-addressed cache.

        Raises:
            ValueError: `deadline` is not finite.
            ImageStoreError: The spec is unsupported, the cache is unusable, the download or its
                verification fails (exit code 4), or the budget runs out (exit code 8).
        """
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        end = self._monotonic() + self._resolve_timeout_seconds

        if deadline is not None:
            end = min(end, deadline)  # A caller's deadline may shorten the budget, never extend it.
        self._remaining(end, "image resolution")  # Fail fast if that deadline already passed.
        image_spec = spec if spec is not None else ImageSpec()
        self._validate_spec(image_spec)
        basename = self._basename(image_spec)

        try:
            return self._resolve_locked(image_spec, basename, end)
        except OSError as e:
            # Unwritable directories, a full disk, and similar host problems are the user's to fix,
            # not cloud-boot bugs.
            raise ImageStoreError(
                f"cannot use the image cache: {e}", context={"cache_dir": str(self.cache_dir)}
            ) from e

    def _resolve_locked(self, spec: ImageSpec, basename: str, deadline: float) -> VerifiedImage:
        """Take the release lock, then return a verified cache hit or a fresh download."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_dir = self.cache_dir / ".locks"
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        lock_path = lock_dir / f"{spec.release}.lock"

        # "a+b" creates the file without truncating it. The lock file is never deleted: removing it
        # would let a newcomer lock a fresh file while another process still holds the old one.
        with lock_path.open("a+b") as lock_file:
            self._acquire_lock(lock_file.fileno(), deadline)
            try:
                self._remove_stale_partials(deadline)
                cached = self._read_cached(spec, basename, deadline)
                if cached is not None:
                    return cached
                return self._download_and_cache(spec, basename, deadline)
            finally:
                # Closing the file would release the lock too; unlocking first makes it explicit.
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _remaining(self, deadline: float, operation: str) -> float:
        """Return seconds left before `deadline`, or raise a timeout naming `operation`."""
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise ImageStoreError(f"timed out during {operation}", exit_code=ExitCode.TIMEOUT)
        return remaining

    def _acquire_lock(self, fd: int, deadline: float) -> None:
        """Poll for the exclusive cache lock until it is free or the deadline passes."""
        # A blocking flock cannot time out, so poll with LOCK_NB instead.
        while True:
            self._remaining(deadline, "image cache lock acquisition")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                remaining = self._remaining(deadline, "image cache lock acquisition")
                # No download is running yet, so there is no cancellation event to pass.
                self._sleeper(min(0.1, remaining), None)

    def _remove_stale_partials(self, deadline: float) -> None:
        """Delete temporaries left by an interrupted resolve; the caller holds the lock."""
        # Only Noble is supported, so every temporary here belongs to this lock.
        # Multiple releases must first move temporaries into release-scoped dirs.
        for directory, pattern in (
            (self.cache_dir, ".partial-*"),
            (self.cache_dir / "releases", ".metadata-*"),
        ):
            for stale in directory.glob(pattern):
                self._remaining(deadline, "stale cache cleanup")
                # Another resolve cannot be writing it, since we hold the lock, but a user may have
                # deleted it by hand.
                with contextlib.suppress(FileNotFoundError):
                    stale.unlink()

    def _validate_spec(self, spec: ImageSpec) -> None:
        """Reject anything but the official Noble amd64 server image.

        `ImageSpec` already allows only these values; this guards callers that bypass scenario
        validation, such as `ImageSpec.model_construct`.
        """
        if (spec.release, spec.architecture, spec.variant, spec.source) != (
            "noble",
            "amd64",
            "server-cloudimg",
            "official",
        ):
            raise ImageStoreError("only the official Noble amd64 server-cloudimg is supported")

    @staticmethod
    def _basename(spec: ImageSpec) -> str:
        """Return the upstream file name, such as `noble-server-cloudimg-amd64.img`."""
        return f"{spec.release}-{spec.variant}-{spec.architecture}.img"

    def _read_cached(self, spec: ImageSpec, basename: str, deadline: float) -> VerifiedImage | None:
        """Return the cached image if its pointer is valid and its bytes still hash correctly.

        Any missing, unreadable, malformed, or mismatched pointer or image is a cache miss rather
        than an error, so the caller downloads a fresh copy, which rewrites both.
        """
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
        source_url = f"{self._base_url}{basename}"

        if (
            metadata.get("basename") != basename
            # Task 12 must change this to require an authenticated signature.
            or metadata.get("checksum_verified") is not True
            or metadata.get("signature_authenticated") is not False
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            # `bool` is a subclass of `int`, so check the exact type.
            or type(size_bytes) is not int
            or not 0 <= size_bytes <= self._max_image_bytes
            or not isinstance(retrieved_at, (int, float))
            or isinstance(retrieved_at, bool)
            or not math.isfinite(retrieved_at)
            # A different `base_url` must not reuse an image fetched from elsewhere.
            or metadata.get("source_url") != source_url
        ):
            return None

        # Validate the digest before constructing or accessing its filesystem path.
        cached_path = self.cache_dir / "sha256" / f"{digest}.img"

        # Rehash on every hit: the file may have changed since it was cached, and QEMU will use it.
        actual_digest, actual_size = self._hash_file(cached_path, deadline)

        # Digests are not secret; the design asks for a constant-time comparison anyway.
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
        """Download, verify, and atomically publish the image, then point the release at it.

        Nothing is written until `SHA256SUMS` names the image. The partial file is removed on any
        failure, so the cache holds only verified images.
        """
        manifest_url = f"{self._base_url}SHA256SUMS"
        image_url = f"{self._base_url}{basename}"
        manifest = self._read_manifest(manifest_url, deadline)
        expected_digest = self._parse_manifest(manifest, basename)
        partial_path: Path | None = None

        try:
            # Same filesystem as `sha256/`, so the final `os.replace` is atomic.
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
            # Readers see the old file or the complete new one, never a partial one. A VM still
            # using the old file keeps its open inode, and the contents are identical anyway.
            # The directory is not fsynced: losing the rename in a crash is only a cache miss.
            os.replace(partial_path, final_path)
            partial_path = None  # Published; nothing left for `finally` to remove.
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
        """Yield the body of `url` in chunks, enforcing `deadline` even if the network stalls.

        urllib's timeout covers each socket operation, and one buffered read (header parsing
        included) can make many of them, so a slow server can stretch a single read indefinitely.
        All network I/O, including open and close, therefore runs in a daemon producer thread,
        while this generator polls a one-slot queue and checks the deadline. Closing the
        generator cancels the producer, which then stops publishing.

        Raises:
            ImageStoreError: The download fails, is cut short, or the deadline passes.
        """
        # One slot bounds memory to the chunk in transit plus the one each side holds.
        messages: queue.Queue[bytes | Exception | None] = queue.Queue(maxsize=1)
        cancelled = threading.Event()

        def publish(value: bytes | Exception | None) -> None:
            # A plain `put` would block forever once the consumer is gone; poll so cancellation
            # can end the wait. Values published after cancellation are dropped.
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
                expected_bytes = _content_length(response)
                received_bytes = 0
                while not cancelled.is_set():
                    self._remaining(deadline, "image download")
                    chunk = response.read(self._chunk_size)
                    if not chunk:
                        # urllib reports a body cut short as a normal end, so compare the byte
                        # count with Content-Length; the digest check would catch it, but would
                        # blame the image instead of the connection.
                        if expected_bytes is not None and received_bytes < expected_bytes:
                            raise ImageStoreError(
                                "connection closed before the download finished",
                                context={
                                    "url": url,
                                    "expected_bytes": expected_bytes,
                                    "received_bytes": received_bytes,
                                },
                            )
                        break
                    if len(chunk) > self._chunk_size:  # Keeps the memory bound honest.
                        raise ImageStoreError("downloader exceeded the requested chunk size")
                    received_bytes += len(chunk)
                    publish(chunk)
            except Exception as e:
                failure = e
            finally:
                if response is not None:
                    with contextlib.suppress(Exception):
                        response.close()
            # Always end with one message: `None` for a complete body, or the failure.
            publish(failure)

        threading.Thread(target=produce, name="cloud-boot-download", daemon=True).start()
        try:
            while True:
                remaining = self._remaining(deadline, "image download")
                try:
                    # A short timeout lets the deadline fire even while the producer is stuck.
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
            # Stops the producer, and wakes it if it is sleeping between retries.
            cancelled.set()

    def _read_manifest(self, url: str, deadline: float) -> bytes:
        """Download `SHA256SUMS` into memory, refusing more than `max_manifest_bytes`."""
        data = bytearray()
        with contextlib.closing(self._download_chunks(url, deadline)) as chunks:
            for chunk in chunks:
                if len(data) + len(chunk) > self._max_manifest_bytes:
                    raise ImageStoreError("SHA256SUMS exceeds the configured size limit")
                data.extend(chunk)
        return bytes(data)

    def _stream_image(self, url: str, destination: IO[bytes], deadline: float) -> tuple[str, int]:
        """Stream the image into `destination`, returning its SHA-256 hex digest and size."""
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
        """Return the lowercase digest that `SHA256SUMS` lists for `basename`.

        The whole manifest must look like `sha256sum` output: one malformed or unsafe line, or a
        duplicate entry for `basename`, rejects it.
        """
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError as e:
            raise ImageStoreError("SHA256SUMS is not valid ASCII") from e
        found: str | None = None
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            match = MANIFEST_LINE_PATTERN.fullmatch(line)
            if match is None:
                raise ImageStoreError(f"invalid SHA256SUMS entry on line {line_number}")
            digest, name = match.groups()
            # Names are only compared with `basename`, never used as paths. These checks reject
            # manifests that do not look like Ubuntu's, and back up the pattern if it is loosened.
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
        """Atomically point the release at `image`, recording its provenance.

        Only the fields `_read_cached` checks are read back; the rest are for people inspecting
        the cache.
        """
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
        """Return the SHA-256 hex digest and size of `path`, read in chunks within the deadline.

        An unreadable or oversized file yields an empty digest, which never matches.
        """
        hasher = hashlib.sha256()
        size_bytes = 0
        try:
            with path.open("rb") as source:
                while True:
                    self._remaining(deadline, "cached image verification")
                    chunk = source.read(self._chunk_size)
                    # Check again after reading: one read can block on slow storage.
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
        """Open `url`, retrying connection failures, HTTP 429, and 5xx with backoff.

        Other HTTP statuses fail at once. Backoff never outlasts the deadline and ends early once
        `cancelled` is set.

        Raises:
            ImageStoreError: A non-retryable status, exhausted attempts, cancellation, or timeout.
        """
        last_error: Exception | None = None

        for attempt in range(self._retry_attempts):
            self._remaining(deadline, "image download")

            if cancelled.is_set():
                raise ImageStoreError("image download cancelled")
            try:
                response = self._downloader(url)
                # Injected downloaders may report a status instead of raising; none means success.
                status = int(getattr(response, "status", 200))
                if 200 <= status < 300:
                    return response
                error = ImageStoreError(f"image server returned HTTP {status}", context={"url": url})
                try:
                    if not _is_retryable(status):
                        raise error
                    delay = self._retry_delay(response, attempt)
                finally:
                    response.close()
                last_error = error
            except HTTPError as e:  # urllib raises this for every non-2xx status.
                last_error = e
                try:
                    if not _is_retryable(e.code):
                        raise ImageStoreError(
                            f"image server returned HTTP {e.code}", context={"url": url}
                        ) from e
                    delay = self._retry_delay(e, attempt)
                finally:
                    e.close()
            except OSError as e:  # Includes `URLError`: refused connections, DNS, and timeouts.
                last_error = e
                delay = self._retry_base_seconds * (2**attempt)
            # No sleep after the last attempt; there is nothing left to wait for.
            if attempt + 1 < self._retry_attempts:
                delay = min(delay, self._remaining(deadline, "download retry backoff"))
                self._sleeper(delay, cancelled)
        raise ImageStoreError(
            "image download failed after bounded retries", context={"url": url}
        ) from last_error

    def _retry_delay(self, response: Any, attempt: int) -> float:
        """Return the backoff before the next attempt, honoring `Retry-After` up to 60 seconds.

        `Retry-After` may be seconds or an HTTP date; anything else falls back to exponential
        backoff from `retry_base_seconds`.
        """
        headers = getattr(response, "headers", {})
        retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
        if retry_after:
            try:
                seconds = float(retry_after)
                if math.isfinite(seconds):  # "inf" and "nan" parse but are not delays.
                    return max(0.0, min(seconds, 60.0))
            except (TypeError, ValueError):
                # An HTTP date is an absolute time, so it needs the wall clock, not the monotonic one.
                with contextlib.suppress(TypeError, ValueError, OverflowError):
                    retry_at = parsedate_to_datetime(str(retry_after)).timestamp()
                    return max(0.0, min(retry_at - self._wall_clock(), 60.0))
        return float(self._retry_base_seconds * (2**attempt))

    @staticmethod
    def _urllib_downloader(url: str) -> IO[bytes]:
        """Open `url` with urllib; the 30-second timeout applies to each socket operation."""
        request = Request(url, headers={"User-Agent": "cloud-boot/0.1"})
        return cast(IO[bytes], urlopen(request, timeout=30.0))
