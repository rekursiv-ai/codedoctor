"""Run Codedoctor stages or configure development tools and Git hooks."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import shutil
import subprocess
import sys

import yaml

from codedoctor import install


_COMMANDS: Final = {
    "lint": "codedoctor.lint.__main__",
    "typecheck": "codedoctor.typecheck.__main__",
    "testing": "codedoctor.testing.__main__",
    "mutation": "codedoctor.mutation.__main__",
    "gate": "codedoctor.gate.__main__",
    "affected": "codedoctor.affected.__main__",
    "coverage-floor": "codedoctor.coverage_floor.__main__",
    "assert-isinstance": "codedoctor.assert_isinstance.__main__",
    "nbstripout": "codedoctor.nbstripout.__main__",
    "yamlcheck": "codedoctor.yamlcheck.__main__",
    "export-leaks": "codedoctor.export_leaks.__main__",
    "codegraph-sync": "codedoctor.affected.codegraph_sync",
    "check-pr-text": "codedoctor.export_leaks.check_pr_text",
}


def main() -> int:
    """Run the Codedoctor command.

    Returns:
      exit_code: Selected stage's status, or zero on successful setup.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    flags, remaining = _parse_args(parser)
    if flags.command in _COMMANDS:
        return _run_stage(flags.command, arguments=remaining)
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    try:
        root = subprocess.run(  # noqa: S603 -- Fixed Git command, no user input.
            [shutil.which("git") or "git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        if flags.command == "check":
            if install.current(Path(root.stdout.strip())):
                sys.stdout.write("Codedoctor configuration is current.\n")
                return 0
            sys.stderr.write(
                "Codedoctor configuration is stale; run `codedoctor install`.\n",
            )
            return 1
        overrides = install.install(Path(root.stdout.strip()))
    except subprocess.CalledProcessError as error:
        sys.stderr.write(f"codedoctor: {cast(object, error.stderr) or error}\n")
        return error.returncode
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        sys.stderr.write(f"codedoctor: {error}\n")
        return 1
    if overrides:
        sys.stderr.write(
            "Codedoctor preserved project overrides: " + ", ".join(overrides) + "\n",
        )
    sys.stdout.write(
        "Codedoctor installed: development tools and pre-commit Git hook.\n"
        "Check existing files: uv run --group pre-commit pre-commit run --all-files\n",
    )
    return 0


class Flags(Protocol):
    """Parsed Codedoctor command."""

    command: str


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register setup commands and stage dispatchers."""
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="Configure tools and install Git hooks.")
    commands.add_parser(
        "check",
        help="Check generated policy freshness without modifying files.",
    )
    for command in _COMMANDS:
        commands.add_parser(command, add_help=False, help=f"Run {command}.")


def _parse_args(parser: argparse.ArgumentParser) -> tuple[Flags, list[str]]:
    """Parse the command while leaving stage arguments to its own parser."""
    _add_arguments(parser)
    flags, remaining = parser.parse_known_args()
    return cast(Flags, flags), remaining


def _run_stage(command: str, *, arguments: list[str]) -> int:
    """Load only the requested stage and restore argv after its parser runs."""
    entry = cast(Callable[[], int], import_module(_COMMANDS[command]).main)
    original = sys.argv
    sys.argv = [f"codedoctor {command}", *arguments]
    try:
        return entry()
    finally:
        sys.argv = original
