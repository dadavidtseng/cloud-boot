# ----------------------------------------------------------------------------------------------------
# artifacts.py
# ----------------------------------------------------------------------------------------------------

"""Run workspace: a private, contained directory for one run's evidence and transient assets."""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from io import BufferedWriter
from pathlib import Path, PurePosixPath

from .errors import CloudBootError, ErrorPhase, ExitCode

# ----------------------------------------------------------------------------------------------------
__all__ = [
    "RUNS_DIRNAME",
    "TRANSIENT_DIRNAME",
    "ArtifactError",
    "RunWorkspace",
    "boot_directory",
    "new_run_id",
]

# ----------------------------------------------------------------------------------------------------
RUNS_DIRNAME = "runs"
# Seed media, overlays, keys, and known hosts live here and are removed after collection.
TRANSIENT_DIRNAME = "transient"
DIRECTORY_MODE = 0o700
FILE_MODE = 0o600
_RUN_ID_ATTEMPTS = 3

type WallClock = Callable[[], datetime]
type TokenFactory = Callable[[], str]


# ----------------------------------------------------------------------------------------------------
class ArtifactError(CloudBootError):
    """Run evidence could not be created or written."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(
            message,
            code="artifact-write-failed",
            phase=ErrorPhase.ARTIFACT,
            remediation="Check that the artifact directory is writable and has free space.",
            context=context,
            exit_code=ExitCode.REPORT,
        )


# ----------------------------------------------------------------------------------------------------
def new_run_id(now: datetime, token: str) -> str:
    """Return a sortable run ID such as `20261107T190512Z-a1b2c3d4` from a UTC time and token."""
    if now.utcoffset() is None:
        raise ValueError("run IDs need a timezone-aware time")
    return f"{now.astimezone(UTC):%Y%m%dT%H%M%SZ}-{token}"


# ----------------------------------------------------------------------------------------------------
def boot_directory(index: int, name: str) -> str:
    """Return the run-relative directory for one boot, such as `boots/001-boot`."""
    return f"boots/{index:03d}-{name}"


# ----------------------------------------------------------------------------------------------------
class RunWorkspace:
    """One run's directory under `<artifact root>/runs/`, with mode 0700 throughout.

    Every path handed out stays inside the run directory and has no symlink components, so a
    planted link cannot redirect a write elsewhere. Files are created 0600 and never overwrite an
    existing file, except through `write_text_atomic`, which replaces it in one step.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @classmethod
    def create(
        cls,
        artifact_root: str | Path,
        *,
        wall_clock: WallClock = lambda: datetime.now(UTC),
        token_factory: TokenFactory = lambda: secrets.token_hex(4),
    ) -> RunWorkspace:
        """Allocate a new, uniquely named run directory and its `transient/` subdirectory.

        Raises:
            ArtifactError: If the runs directory is unusable or no unique name could be created.
        """
        runs = Path(artifact_root) / RUNS_DIRNAME
        try:
            runs.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
        except OSError as e:
            raise ArtifactError(
                "cannot create the runs directory", context={"path": str(runs)}
            ) from e
        if runs.is_symlink() or not runs.is_dir():
            raise ArtifactError(
                "the runs directory must be a real directory", context={"path": str(runs)}
            )
        for _ in range(_RUN_ID_ATTEMPTS):
            path = runs / new_run_id(wall_clock(), token_factory())
            try:
                os.mkdir(path, DIRECTORY_MODE)
            except FileExistsError:
                continue
            except OSError as e:
                raise ArtifactError(
                    "cannot create the run directory", context={"path": str(path)}
                ) from e
            # mkdir applies the umask; set the mode explicitly.
            os.chmod(path, DIRECTORY_MODE)
            workspace = cls(path)
            workspace.make_dirs(TRANSIENT_DIRNAME)
            return workspace
        raise ArtifactError("could not allocate a unique run directory", context={"path": str(runs)})

    @property
    def path(self) -> Path:
        """Return the run directory."""
        return self._path

    @property
    def run_id(self) -> str:
        """Return the run ID, which is also the run directory's name."""
        return self._path.name

    @property
    def transient_dir(self) -> Path:
        """Return the directory for seed media, overlays, keys, and known hosts."""
        return self._path / TRANSIENT_DIRNAME

    def artifact_path(self, relative: str) -> Path:
        """Return the path for a run-relative name after checking it cannot escape.

        Raises:
            ValueError: If `relative` is empty, absolute, contains NUL or `..`, or crosses a symlink.
        """
        if not relative or "\x00" in relative:
            raise ValueError("artifact paths must be non-empty and free of NUL bytes")
        parts = PurePosixPath(relative).parts
        if PurePosixPath(relative).is_absolute() or any(part in {"..", "."} for part in parts):
            raise ValueError(f"artifact path must be relative to the run directory: {relative!r}")
        current = self._path
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"artifact path crosses a symlink: {relative!r}")
        return current

    def relative(self, path: str | Path) -> str:
        """Return `path` relative to the run directory, for portable report references.

        Raises:
            ValueError: If `path` is outside the run directory.
        """
        return Path(path).relative_to(self._path).as_posix()

    def make_dirs(self, relative: str) -> Path:
        """Create the run-relative directory and any missing parents with mode 0700.

        Raises:
            ValueError: If the path escapes the run directory or crosses a symlink.
            ArtifactError: If a directory cannot be created.
        """
        self.artifact_path(relative)
        current = self._path
        for part in PurePosixPath(relative).parts:
            current = current / part
            try:
                os.mkdir(current, DIRECTORY_MODE)
                os.chmod(current, DIRECTORY_MODE)
            except FileExistsError:
                pass
            except OSError as e:
                raise ArtifactError(
                    "cannot create an artifact directory", context={"path": str(current)}
                ) from e
            if current.is_symlink() or not current.is_dir():
                raise ValueError(f"artifact directory is not a real directory: {relative!r}")
        return current

    def open_new(self, relative: str) -> BufferedWriter:
        """Create a new 0600 file for streaming, such as the serial console log.

        Raises:
            ValueError: If the path escapes the run directory or crosses a symlink.
            ArtifactError: If the file already exists or cannot be created.
        """
        path = self._prepared_path(relative)
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, FILE_MODE
            )
        except OSError as e:
            raise ArtifactError("cannot create an artifact file", context={"path": relative}) from e
        return os.fdopen(descriptor, "wb")

    def write_bytes(self, relative: str, data: bytes) -> Path:
        """Write a new 0600 file in one call, refusing to overwrite.

        Raises:
            ArtifactError: If the file exists or cannot be written.
        """
        try:
            with self.open_new(relative) as stream:
                stream.write(data)
        except OSError as e:
            raise ArtifactError("cannot write an artifact file", context={"path": relative}) from e
        return self.artifact_path(relative)

    def write_text_atomic(self, relative: str, text: str) -> Path:
        """Replace a structured file in one step so readers never see a partial document.

        The text goes to a 0600 sibling temporary file, which is flushed, synced, and renamed over
        the destination; the directory is then synced so the rename survives a crash.

        Raises:
            ArtifactError: If any step fails; the previous file, if any, is left intact.
        """
        path = self._prepared_path(relative)
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            _sync_directory(path.parent)
        except OSError as e:
            raise ArtifactError("cannot write an artifact file", context={"path": relative}) from e
        finally:
            if temporary is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary)
        return path

    def cleanup_transient(self) -> list[str]:
        """Delete `transient/` with its seed media, overlay, and keys; keep all other evidence.

        Safe to call more than once. Failures are returned as warnings instead of raised, so a
        cleanup problem never replaces the run's primary outcome.
        """
        transient = self.transient_dir
        warnings: list[str] = []
        if transient.is_symlink():
            with contextlib.suppress(FileNotFoundError):
                transient.unlink()
            return warnings
        if not transient.exists():
            return warnings

        def record(_function: Callable[..., object], path: str, error: BaseException) -> None:
            warnings.append(f"could not remove {path}: {error}")

        shutil.rmtree(transient, onexc=record)
        return warnings

    def _prepared_path(self, relative: str) -> Path:
        path = self.artifact_path(relative)
        parent = PurePosixPath(relative).parent
        if str(parent) != ".":
            self.make_dirs(str(parent))
        return path


# ----------------------------------------------------------------------------------------------------
def _sync_directory(directory: Path) -> None:
    # Best effort: the data is already durable; this only persists the rename itself.
    with contextlib.suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
