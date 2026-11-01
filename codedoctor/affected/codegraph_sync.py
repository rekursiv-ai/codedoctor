#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Keep the CodeGraph index current for the gates that walk it.

Runs only when PRECOMMIT_CODEGRAPH_PYTEST or PRECOMMIT_CODEGRAPH_TYPECHECK holds
a positive integer. Installs the CLI and links .codegraph into a cache directory
on first use. Never fails a commit: no network, no npm, or a locked index leaves
the gates on their fallback scope.

Examples:
  python -m codedoctor.affected.codegraph_sync --caches /opt/scratch/caches/codegraph

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import os
import shutil

from codedoctor.gate.gate import positive, run_argv


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def main() -> int:
    """Sync the index when a gate opted in to walking it.

    Returns:
      exit_code: Always 0; a gate without an index falls back to its floor.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    if opted_in(os.environ):
        sync(Path.cwd(), caches=flags.caches, run=run_argv, which=shutil.which)
    return 0


def opted_in(environ: Mapping[str, str]) -> bool:
    """Report whether either graph-walking gate is on."""
    return any(
        positive(environ.get(name, ""))
        for name in ("PRECOMMIT_CODEGRAPH_PYTEST", "PRECOMMIT_CODEGRAPH_TYPECHECK")
    )


def sync(
    root: Path,
    *,
    caches: Path,
    run: Callable[[Sequence[str]], int],
    which: Callable[[str], str | None],
) -> None:
    """Sync ``root``'s index, installing the CLI and index first if missing.

    Args:
      root: The worktree.
      caches: Where the index lives, one directory per worktree name.
      run: Runs one argv quietly; returns its exit code.
      which: Finds an executable on PATH.

    """
    if which("codegraph") is None:
        run(["npm", "install", "-g", "@colbymchenry/codegraph"])
        if which("codegraph") is None:
            return
    # A symlink: CODEGRAPH_DIR resolves relative to the project and silently
    # falls back in-tree on an absolute path, and the index is ~387 MB.
    index = root / ".codegraph"
    if not index.exists() and not index.is_symlink():
        cache = caches / root.name
        try:
            cache.mkdir(parents=True, exist_ok=True)
            index.symlink_to(cache)
        except OSError:
            return
    if not (index / "codegraph.db").is_file():
        run(["codegraph", "init", "--yes"])
    run(["codegraph", "sync", "--quiet"])


class _Flags(Protocol):
    caches: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--caches",
        type=Path,
        required=True,
        help="Where the index lives, one directory per worktree name.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
