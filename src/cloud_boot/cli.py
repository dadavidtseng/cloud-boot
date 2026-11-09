# ----------------------------------------------------------------------------------------------------
# cli.py
# ----------------------------------------------------------------------------------------------------

"""Command-line entry point. Milestone 0 offers only `cloud-boot run <scenario>`."""

# ----------------------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import NoReturn, Protocol, TextIO

from . import __version__
from .coordinator import (
    RunCoordinator,
    RunResult,
    default_artifact_root,
    default_cache_dir,
    production_dependencies,
)
from .errors import CloudBootError, ErrorPhase, ExitCode
from .scenario import LoadedScenario, ScenarioValidationError, load_scenario

# ----------------------------------------------------------------------------------------------------
__all__ = ["build_parser", "main"]

type Progress = Callable[[str, str], None]


# ----------------------------------------------------------------------------------------------------
class Runner(Protocol):
    """The coordinator surface the CLI drives."""

    def run(self, loaded: LoadedScenario) -> RunResult:
        """Run one loaded scenario."""
        ...


type RunnerFactory = Callable[[Progress, Mapping[str, str]], Runner]


# ----------------------------------------------------------------------------------------------------
class _ParserExit(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


# ----------------------------------------------------------------------------------------------------
class _Parser(argparse.ArgumentParser):
    """An ArgumentParser that returns control to `main` instead of calling `sys.exit`."""

    def __init__(self, *args: object, stdout: TextIO, stderr: TextIO, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._stdout = stdout
        self._stderr = stderr

    def _print_message(self, message: str, file: object = None) -> None:
        if message:
            (self._stderr if file is sys.stderr else self._stdout).write(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        """Raise instead of exiting, so `main` can return the status."""
        if message:
            self._stderr.write(message)
        raise _ParserExit(status)

    def error(self, message: str) -> NoReturn:
        """Report a usage error with the validation exit code."""
        self._stderr.write(self.format_usage())
        self._stderr.write(f"{self.prog}: error: {message}\n")
        raise _ParserExit(ExitCode.VALIDATION)


# ----------------------------------------------------------------------------------------------------
def build_parser(*, stdout: TextIO, stderr: TextIO) -> argparse.ArgumentParser:
    """Build the Milestone 0 parser: `run <scenario>`, `--help`, and `--version`."""
    parser = _Parser(
        prog="cloud-boot",
        description="Boot an Ubuntu cloud image with QEMU/KVM and verify cloud-init behavior.",
        stdout=stdout,
        stderr=stderr,
    )
    parser.add_argument("--version", action="version", version=f"cloud-boot {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    run = commands.add_parser(
        "run",
        help="run one scenario and write a report",
        description="Run one scenario. The run directory path is printed on stdout.",
        stdout=stdout,
        stderr=stderr,
    )
    run.add_argument("scenario", help="path to the scenario YAML file")
    return parser


# ----------------------------------------------------------------------------------------------------
def _production_runner(progress: Progress, environ: Mapping[str, str]) -> Runner:
    dependencies = production_dependencies(
        cache_dir=default_cache_dir(environ), artifact_root=default_artifact_root(environ)
    )
    return RunCoordinator(dependencies, progress=progress)


# ----------------------------------------------------------------------------------------------------
def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    runner_factory: RunnerFactory = _production_runner,
) -> int:
    """Parse arguments, run the command, and return the process exit code.

    Progress, failures, and remediation go to stderr; stdout carries only the run directory path
    so scripts can capture it.
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    try:
        args = build_parser(stdout=out, stderr=err).parse_args(argv)
    except _ParserExit as exit_request:
        return exit_request.status
    try:
        return _run(
            args.scenario, out, err, environ if environ is not None else os.environ, runner_factory
        )
    except Exception as error:
        err.write(f"ERROR {ErrorPhase.INTERNAL}: unexpected {type(error).__name__}: {error}\n")
        return int(ExitCode.INTERNAL)


# ----------------------------------------------------------------------------------------------------
def _run(
    scenario: str,
    out: TextIO,
    err: TextIO,
    environ: Mapping[str, str],
    runner_factory: RunnerFactory,
) -> int:
    try:
        loaded = load_scenario(scenario)
    except ScenarioValidationError as error:
        err.write(f"ERROR {ErrorPhase.VALIDATION}: {error}\n")
        return int(ExitCode.VALIDATION)

    def progress(phase: str, message: str) -> None:
        err.write(f"INFO {phase}: {message}\n")
        err.flush()

    result = runner_factory(progress, environ).run(loaded)
    for assertion in result.assertions:
        if assertion.status != "passed":
            label = "FAIL" if assertion.required else "WARN"
            err.write(f"{label} {assertion.id} ({assertion.status}): {assertion.reason}\n")
    if result.failure is not None:
        _write_failure(err, result.failure)
    if result.run_dir is not None:
        out.write(f"{result.run_dir}\n")
    return result.exit_code


# ----------------------------------------------------------------------------------------------------
def _write_failure(err: TextIO, failure: CloudBootError) -> None:
    err.write(f"ERROR {failure.phase}: {failure.message}\n")
    if failure.remediation:
        err.write(f"HINT {failure.remediation}\n")
