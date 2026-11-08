"""Install Codedoctor's development tools and Git hooks in a repository."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import argparse
import shutil
import subprocess
import sys

import yaml

from codedoctor import install


def main() -> int:
    """Run the Codedoctor command.

    Returns:
      exit_code: Zero on successful installation, nonzero on setup failure.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
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
    """Parsed repository setup command."""

    command: str


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register repository setup commands."""
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="Configure tools and install Git hooks.")
    commands.add_parser(
        "check",
        help="Check generated policy freshness without modifying files.",
    )
