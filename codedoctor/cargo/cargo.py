"""Run a cargo command once per crate the supplied files belong to."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import shutil
import subprocess


if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> int:
    """Run the cargo command once for each crate the supplied files belong to.

    Returns:
      exit_code: The first nonzero cargo status, or zero when every crate passed.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    command, *options = {
        "fmt": ["fmt"],
        "clippy": ["clippy", "--all-targets", "--locked", "--", "-D", "warnings"],
    }[flags.command]
    result = 0
    for manifest in manifests(flags.files):
        completed = subprocess.run(  # noqa: S603 -- Fixed cargo arguments.
            [
                shutil.which("cargo") or "cargo",
                command,
                "--manifest-path",
                str(manifest),
                *options,
            ],
            check=False,
        )
        result = result or completed.returncode
    return result


class Flags(Protocol):
    """Cargo command and the files pre-commit supplied."""

    command: str
    files: list[Path]


def manifests(paths: Sequence[Path]) -> list[Path]:
    """Return the nearest ``Cargo.toml`` above each path, each once, in path order.

    Args:
      paths: Files relative to the repository root, as pre-commit supplies them.

    Returns:
      manifests: One manifest per crate; files outside every crate add none.

    """
    found: list[Path] = []
    for path in paths:
        manifest = next(
            (
                folder / "Cargo.toml"
                for folder in path.parents
                if (folder / "Cargo.toml").is_file()
            ),
            None,
        )
        if manifest is not None and manifest not in found:
            found.append(manifest)
    return found


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("command", choices=["fmt", "clippy"])
    parser.add_argument("files", type=Path, nargs="*")
