#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Keep the shared CodeGraph backbone and this worktree's overlay current.

Hook mode runs only when PRECOMMIT_CODEGRAPH_PYTEST or PRECOMMIT_CODEGRAPH_TYPECHECK
holds a positive integer. It installs the CLI if missing, starts building the
backbone in the background when none exists, and otherwise indexes this
worktree's changes from the backbone into its overlay. It never fails a commit:
a problem is printed, and `affected` independently refuses a stale overlay.

--refresh republishes the backbone at origin/main; a timer runs it with --fetch.

Examples:
  python -m codedoctor.affected.codegraph_sync --caches /opt/scratch/caches/codegraph
  python -m codedoctor.affected.codegraph_sync --caches /opt/scratch/caches/codegraph --refresh --fetch

'''
# fmt: on

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from urllib.request import pathname2url

import argparse
import fcntl
import os
import shutil
import sqlite3
import subprocess
import sys

from codedoctor.affected import backbone
from codedoctor.affected.affected import ensure_derived
from codedoctor.gate.gate import output_argv, positive, run_argv


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def main() -> int:
    """Refresh the backbone, or sync this worktree's overlay when opted in.

    Returns:
      exit_code: 0, except a failed ``--refresh``, which its timer must see.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    if flags.refresh:
        # Group-writable, so whoever refreshes next can replace what this run wrote.
        os.umask(0o002)
        problem = refresh(
            Path.cwd(),
            caches=flags.caches,
            fetch=flags.fetch,
            run=run_argv,
            output=output_argv,
        )
        if problem:
            sys.stderr.write(f"codegraph-sync: {problem}\n")
        return int(bool(problem))
    if opted_in(os.environ) and (
        problem := sync(
            Path.cwd(),
            caches=flags.caches,
            run=run_argv,
            output=output_argv,
            which=shutil.which,
            spawn=spawn,
        )
    ):
        sys.stderr.write(f"codegraph-sync: {problem}\n")
    return 0


def opted_in(environ: Mapping[str, str]) -> bool:
    """Report whether either graph-walking gate is on."""
    return any(
        positive(environ.get(name, ""))
        for name in ("PRECOMMIT_CODEGRAPH_PYTEST", "PRECOMMIT_CODEGRAPH_TYPECHECK")
    )


class Spawn(Protocol):
    """Starts ``argv`` detached in ``cwd``, its output appended to ``log``."""

    def __call__(self, argv: Sequence[str], *, cwd: Path, log: Path) -> object:
        """Start the process; return its handle."""
        ...


def sync(
    root: Path,
    *,
    caches: Path,
    run: Callable[[Sequence[str]], int],
    output: Callable[[Sequence[str]], str | None],
    which: Callable[[str], str | None],
    spawn: Spawn,
) -> str:
    """Bring worktree ``root``'s overlay current, starting a backbone if none exists.

    Args:
      root: The worktree.
      caches: The CodeGraph cache directory.
      run: Runs one argv; returns its exit code.
      output: Runs one argv; returns its stdout, or None when it failed.
      which: Finds an executable on PATH.
      spawn: Starts the backbone build detached.

    Returns:
      problem: Why the gates must select without the graph; empty otherwise.

    """
    if which("codegraph") is None:
        run(["npm", "install", "-g", "@colbymchenry/codegraph"])
        if which("codegraph") is None:
            return (
                "The codegraph CLI is missing, and `npm install -g "
                "@colbymchenry/codegraph` did not install it."
            )
    if not backbone.backbone_database(caches).exists():
        log = backbone.overlay_database(caches, root=root).with_name("refresh.log")
        spawn(
            [
                sys.executable,
                "-m",
                "codedoctor.affected.codegraph_sync",
                "--caches",
                str(caches),
                "--refresh",
            ],
            cwd=root,
            log=log,
        )
        return (
            f"No CodeGraph backbone yet; building one in the background (log: {log}). "
            "Gates select without the graph until it lands."
        )
    return backbone.build_overlay(
        root,
        caches=caches,
        output=output,
        index=partial(_index, run=run),
    )


def refresh(
    repo: Path,
    *,
    caches: Path,
    fetch: bool,
    run: Callable[[Sequence[str]], int],
    output: Callable[[Sequence[str]], str | None],
) -> str:
    """Republish the backbone at ``repo``'s ``origin/main``, if it moved.

    ``origin/main`` is checked out into a snapshot tree beside the backbone
    through a private index, so ``repo``'s own index, HEAD, and working tree
    are never touched; the snapshot is then synced, copied, stamped, and
    renamed over the published backbone. Readers holding the old file keep
    reading it.

    Args:
      repo: Any clone holding ``origin/main``.
      caches: The CodeGraph cache directory.
      fetch: Fetch ``origin`` first, as the timer does. The hook's bootstrap
        does not: a commit must not wait on the network.
      run: Runs one argv; returns its exit code.
      output: Runs one argv; returns its stdout, or None when it failed.

    Returns:
      problem: Why the backbone was not brought current; empty when it was,
        or when another refresh holds the lock.

    """
    home = backbone.backbone_database(caches).parent
    if not home.is_dir():
        home.mkdir(parents=True)
        # Setgid keeps every file in the group; the readers are the whole group.
        home.chmod(0o2775)
    with (home / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return ""
        return _refresh_locked(repo, home=home, fetch=fetch, run=run, output=output)


def spawn(argv: Sequence[str], *, cwd: Path, log: Path) -> subprocess.Popen[bytes]:
    """Start ``argv`` in its own session, so it outlives the hook that started it.

    Args:
      argv: The command.
      cwd: Its working directory.
      log: Where its output is appended.

    Returns:
      child: The started process.

    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as sink:
        return subprocess.Popen(  # noqa: S603 -- argv is this module's own refresh command.
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


class _Flags(Protocol):
    caches: Path
    refresh: bool
    fetch: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--caches",
        type=Path,
        required=True,
        help="CodeGraph cache directory: the backbone, and one overlay per worktree.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Republish the backbone at origin/main of the clone in the cwd.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="With --refresh, fetch origin first.",
    )


def _index(tree: Path, *, run: Callable[[Sequence[str]], int]) -> int:
    """Index ``tree`` from scratch into its ``.codegraph/codegraph.db``."""
    return run(["codegraph", "init", "--yes", str(tree)])


def _refresh_locked(
    repo: Path,
    *,
    home: Path,
    fetch: bool,
    run: Callable[[Sequence[str]], int],
    output: Callable[[Sequence[str]], str | None],
) -> str:
    """Do ``refresh``'s work; the caller holds the lock."""
    if fetch and (
        status := run(["git", "-C", str(repo), "fetch", "--quiet", "origin"])
    ):
        return f"`git fetch origin` failed (exit {status})."
    sha = (
        output(
            ["git", "-C", str(repo), "rev-parse", "--verify", "origin/main^{commit}"],
        )
        or ""
    ).strip()
    if not sha:
        return f"origin/main does not resolve in {repo}."
    database = home / "codegraph.db"
    published = backbone.stamped(database)
    if published is None or published.sha != sha:
        tree = home / "tree"
        tree.mkdir(exist_ok=True)
        if status := run(
            [
                "env",
                f"GIT_INDEX_FILE={home / 'index'}",
                "git",
                "-C",
                str(repo),
                f"--work-tree={tree}",
                "read-tree",
                "--reset",
                "-u",
                sha,
            ],
        ):
            return f"`git read-tree` of {sha[:12]} failed (exit {status})."
        working = tree / ".codegraph" / "codegraph.db"
        if status := run(
            ["codegraph", "sync", "--quiet", str(tree)]
            if working.is_file()
            else ["codegraph", "init", "--yes", str(tree)],
        ):
            return f"Indexing {sha[:12]} failed (codegraph exit {status})."
        _publish(working, database, sha=sha)
    if ensure_derived(database) is None:
        return f"Building the derived tables of {database} failed."
    return ""


def _publish(working: Path, database: Path, *, sha: str) -> None:
    """Copy ``working`` beside ``database``, stamp it ``sha``, and rename it over."""
    scratch = database.with_name(f"{database.name}.tmp")
    scratch.unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{pathname2url(str(working))}?mode=ro", uri=True)
    try:
        source.execute("""VACUUM INTO ?""", (str(scratch),))
    finally:
        source.close()
    backbone.stamp(scratch, sha=sha)
    scratch.chmod(0o664)
    scratch.replace(database)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
