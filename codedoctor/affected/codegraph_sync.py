#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Keep this worktree's CodeGraph index current.

Hook mode runs when an enabled hook's resolved PRECOMMIT_SCOPE needs the graph.
It installs the CLI if missing and syncs the current filesystem state into the
worktree's index. It never resolves or fetches a commit. It never fails a commit:
a problem is printed, and `affected` retries before selecting dependents.

--refresh republishes the backbone at origin/main; a timer runs it with --fetch.

--cli links this worktree a `.codegraph` seeded from the backbone, so the codegraph
CLI and MCP server can sync and query it. Run it once per worktree before them.

Examples:
  python -m codedoctor.affected.codegraph_sync --caches /opt/scratch/caches/codegraph
  python -m codedoctor.affected.codegraph_sync --caches /opt/scratch/caches/codegraph --refresh --fetch
  python -m codedoctor.affected.codegraph_sync --caches /opt/scratch/caches/codegraph --cli

'''
# fmt: on

from __future__ import annotations

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
from codedoctor.gate.gate import output_argv, run_argv, scope, skipped


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def main() -> int:
    """Refresh the backbone, link a CLI index, or sync this worktree's overlay.

    Returns:
      exit_code: 0, except a failed ``--refresh`` or ``--cli``, which their
        callers must see.

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
    if flags.cli:
        if problem := link_cli_index(Path.cwd(), caches=flags.caches):
            sys.stderr.write(f"codegraph-sync: {problem}\n")
        return int(bool(problem))
    if opted_in(os.environ) and (
        problem := sync(
            Path.cwd(),
            caches=flags.caches,
            run=run_argv,
            which=shutil.which,
        )
    ):
        sys.stderr.write(f"codegraph-sync: {problem}\n")
    return 0


def opted_in(environ: Mapping[str, str]) -> bool:
    """Report whether an enabled hook's resolved scope needs the graph.

    Args:
      environ: Scope overrides and pre-commit's SKIP list.

    Returns:
      needed: Whether any unskipped hook has a positive dependency depth.

    """
    depths = {
        name: scope(name, environ=environ)
        for name in ("ty-check", "pyrefly-check", "basedpyright", "pytest", "mutation")
    }
    return not skipped(environ, "harvest-affected") and any(
        depth > 0 and not skipped(environ, name) for name, depth in depths.items()
    )


def sync(
    root: Path,
    *,
    caches: Path,
    run: Callable[[Sequence[str]], int],
    which: Callable[[str], str | None],
) -> str:
    """Sync the current worktree's files into its CodeGraph index.

    Args:
      root: The worktree.
      caches: The CodeGraph cache directory.
      run: Runs one argv; returns its exit code.
      which: Finds an executable on PATH.

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
    return backbone.sync_worktree(root, caches=caches, run=run)


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


# Neither index the gates read is this worktree's whole tree -- the backbone is
# origin/main, the overlay only its changes -- so the CLI gets its own, seeded from the
# backbone: a ``codegraph sync`` then indexes only what differs, 15 s rather than a full
# ``init``. A symlink, because ``CODEGRAPH_DIR`` resolves relative to the project and
# the index is too large for the checkout. An existing ``.codegraph`` is the owner's.
# Only on request: the gates once linked one into every worktree they ran in, each a
# full copy of the backbone (825 MB), whether or not anyone used the CLI there.
def link_cli_index(root: Path, *, caches: Path) -> str:
    """Give worktree ``root`` a ``.codegraph`` the CLI and MCP server can sync.

    Args:
      root: The worktree.
      caches: The CodeGraph cache directory.

    Returns:
      problem: Why ``root`` has no CLI index; empty when it has one.

    """
    index = root / ".codegraph"
    if index.exists() or index.is_symlink():
        return ""
    database = backbone.backbone_database(caches)
    home = backbone.overlay_database(caches, root=root).parent / "cli"
    seeded = home / "codegraph.db"
    if not seeded.is_file() and not database.is_file():
        return (
            "No CodeGraph backbone to seed from; run --refresh, or run hooks "
            "with PRECOMMIT_SCOPE=affected:1."
        )
    try:
        home.mkdir(parents=True, exist_ok=True)
        if not seeded.exists():
            shutil.copyfile(database, seeded)
        index.symlink_to(home)
    except OSError as error:
        return f"Linking {index} to {home} failed: {error}"
    return ""


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
    cli: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--caches",
        type=Path,
        required=True,
        help="CodeGraph cache directory: the backbone, and one overlay per worktree.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--refresh",
        action="store_true",
        help="Republish the backbone at origin/main of the clone in the cwd.",
    )
    mode.add_argument(
        "--cli",
        action="store_true",
        help="Link the cwd's worktree a .codegraph for the codegraph CLI and MCP.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="With --refresh, fetch origin first.",
    )


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
    sha = _origin_main(repo, output=output)
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


def _origin_main(repo: Path, *, output: Callable[[Sequence[str]], str | None]) -> str:
    """Return the commit ``origin/main`` names in ``repo``; empty when it names none."""
    argv = ["git", "-C", str(repo), "rev-parse", "--verify", "origin/main^{commit}"]
    return (output(argv) or "").strip()


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
