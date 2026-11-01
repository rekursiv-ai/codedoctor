"""Lay out one shared CodeGraph backbone and a small overlay per worktree.

The backbone indexes one ``origin/main`` commit, once, for every user of a
machine. A worktree's overlay indexes only the files that differ from that
commit; ``affected`` reads the backbone with the overlay laid over it. Both
carry a stamp naming what they describe, so a reader can prove an overlay
current before trusting it.

  <caches>/backbone/codegraph.db          stamped with the commit it indexes
  <caches>/overlays/<name>-<hash>/codegraph.db
                                          stamped with that commit, the
                                          fingerprint of the changed files,
                                          and their paths
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from urllib.request import pathname2url

import hashlib
import shutil
import sqlite3
import tempfile


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# Every column the walk in `affected` reads. CodeGraph is an unpinned npm package, so a
# release that renames one must degrade the gates to their floor rather than let a
# query fail -- or worse, succeed against a different meaning.
REQUIRED_COLUMNS: Final = {
    "nodes": (
        "id",
        "kind",
        "name",
        "qualified_name",
        "file_path",
        "start_line",
        "end_line",
    ),
    "edges": ("source", "target", "kind"),
    "unresolved_refs": ("from_node_id", "reference_kind", "file_path", "name_tail"),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Stamp:
    """What an index describes.

    Attributes:
      sha: The ``origin/main`` commit the backbone indexes.
      fingerprint: An overlay's changed files and their contents; empty for
        the backbone.
      paths: An overlay's changed files, sorted; deleted ones included.

    """

    sha: str
    fingerprint: str = ""
    paths: tuple[str, ...] = ()


def backbone_database(caches: Path) -> Path:
    """Return the shared backbone index under ``caches``."""
    return caches / "backbone" / "codegraph.db"


def overlay_database(caches: Path, *, root: Path) -> Path:
    """Return worktree ``root``'s overlay index under ``caches``.

    The directory names the worktree and hashes its absolute path: keying on the
    basename alone made every checkout named ``loop`` share one index.

    Args:
      caches: The CodeGraph cache directory.
      root: The worktree.

    Returns:
      overlay: The overlay's index path.

    """
    resolved = root.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
    return caches / "overlays" / f"{resolved.name}-{digest}" / "codegraph.db"


def stamp(
    database: Path,
    *,
    sha: str,
    fingerprint: str = "",
    paths: Sequence[str] = (),
) -> None:
    """Record what ``database`` describes, inside it, and make it self-contained.

    Rollback journaling, not CodeGraph's WAL: a published index is one file a
    reader opens read-only, with no ``-wal`` or ``-shm`` beside it to own.

    Args:
      database: The index to stamp.
      sha: The backbone commit.
      fingerprint: An overlay's ``fingerprint``.
      paths: An overlay's changed files.

    """
    connection = sqlite3.connect(database)
    try:
        connection.execute("""PRAGMA journal_mode=DELETE""")
        connection.executescript(
            """
            DROP TABLE IF EXISTS codedoctor_stamp;
            DROP TABLE IF EXISTS codedoctor_changed;
            CREATE TABLE codedoctor_stamp (sha TEXT NOT NULL, fingerprint TEXT NOT NULL);
            CREATE TABLE codedoctor_changed (path TEXT PRIMARY KEY);
            """,
        )
        connection.execute(
            """INSERT INTO codedoctor_stamp VALUES (?, ?)""",
            (sha, fingerprint),
        )
        connection.executemany(
            """INSERT INTO codedoctor_changed VALUES (?)""",
            ((path,) for path in paths),
        )
        connection.commit()
    finally:
        connection.close()


def stamped(database: Path) -> Stamp | None:
    """Return what ``database`` describes.

    Args:
      database: A backbone or overlay index.

    Returns:
      stamp: Its stamp; None when it has none or cannot be read.

    """
    if not database.is_file():
        return None
    try:
        connection = sqlite3.connect(
            f"file:{pathname2url(str(database))}?mode=ro",
            uri=True,
        )
    except sqlite3.Error:
        return None
    try:
        row = cast(
            tuple[str, str] | None,
            connection.execute(
                """SELECT sha, fingerprint FROM codedoctor_stamp""",
            ).fetchone(),
        )
        paths = cast(
            list[tuple[str]],
            connection.execute(
                """SELECT path FROM codedoctor_changed ORDER BY path""",
            ).fetchall(),
        )
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    if row is None:
        return None
    return Stamp(sha=row[0], fingerprint=row[1], paths=tuple(path for (path,) in paths))


def changed(
    root: Path,
    *,
    sha: str,
    output: Callable[[Sequence[str]], str | None],
) -> list[str] | None:
    """Return the files of worktree ``root`` that differ from commit ``sha``.

    The working tree, not the index: under the stashing ``git commit`` hook the
    two are the same tree, and under ``--files`` the working tree is what is
    gated. Deleted files are included -- their backbone rows must go -- and so
    are untracked ones, which the backbone never saw.

    Args:
      root: The worktree.
      sha: The backbone commit.
      output: Runs one argv; returns its stdout, or None when it failed.

    Returns:
      paths: Sorted repository-relative paths; None when git could not answer,
        as for a commit this clone has not fetched.

    """
    diff = output(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            sha,
            "--",
        ],
    )
    others = output(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
    )
    if diff is None or others is None:
        return None
    return sorted({path for path in f"{diff}\0{others}".split("\0") if path})


def fingerprint(root: Path, *, sha: str, paths: Sequence[str]) -> str:
    """Return a digest of commit ``sha`` and the content of each of ``paths``.

    Args:
      root: The worktree.
      sha: The backbone commit.
      paths: The changed files; a missing one counts as deleted.

    Returns:
      fingerprint: A hex digest that moves with any of them.

    """
    digest = hashlib.sha256(sha.encode())
    for path in paths:
        file = root / path
        content = (
            hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else "-"
        )
        digest.update(f"\0{path}\0{content}".encode())
    return digest.hexdigest()


def build_overlay(
    root: Path,
    *,
    caches: Path,
    output: Callable[[Sequence[str]], str | None],
    index: Callable[[Path], int],
) -> str:
    """Index worktree ``root``'s changes from the backbone into its overlay.

    The changed files are mirrored at their repository-relative paths into a
    private directory beside the overlay and indexed there, so the overlay's
    ``file_path`` values name the same files the backbone's do. The result is
    stamped and renamed into place only once complete; a failure publishes
    nothing, and the old overlay's stale stamp then keeps readers off it.

    Args:
      root: The worktree.
      caches: The CodeGraph cache directory.
      output: Runs one argv; returns its stdout, or None when it failed.
      index: Indexes one directory as CodeGraph does, into its
        ``.codegraph/codegraph.db``; returns the exit code.

    Returns:
      problem: Why no current overlay exists; empty when one does.

    """
    base = stamped(backbone_database(caches))
    if base is None:
        return _absent(backbone_database(caches))
    paths = changed(root, sha=base.sha, output=output)
    if paths is None:
        return _unfetched(base.sha)
    overlay = overlay_database(caches, root=root)
    key = fingerprint(root, sha=base.sha, paths=paths)
    current = stamped(overlay)
    if current is not None and (current.sha, current.fingerprint) == (base.sha, key):
        return ""
    overlay.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="backbone-", dir=overlay.parent) as scratch:
        tree = Path(scratch) / "tree"
        present = [path for path in paths if (root / path).is_file()]
        for path in present:
            (tree / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / path, tree / path)
        built = tree / ".codegraph" / "codegraph.db"
        if present and (status := index(tree)):
            return f"Indexing the overlay failed (codegraph exit {status})."
        if not present:
            built = Path(scratch) / "codegraph.db"
            sqlite3.connect(built).close()
        stamp(built, sha=base.sha, fingerprint=key, paths=paths)
        built.chmod(0o664)
        built.replace(overlay)
    return ""


def overlay_problem(
    caches: Path,
    *,
    root: Path,
    output: Callable[[Sequence[str]], str | None],
) -> tuple[str, Path | None]:
    """Return why worktree ``root`` cannot read the graph, else its overlay.

    Recomputes the fingerprint rather than trusting the last sync: under
    ``SKIP=codegraph-sync``, or after an edit since, the overlay describes a
    tree that no longer exists, and walking it selects tests that do not.

    Args:
      caches: The CodeGraph cache directory.
      root: The worktree.
      output: Runs one argv; returns its stdout, or None when it failed.

    Returns:
      problem: Why the graph must not be read; empty when it may.
      overlay: The overlay to lay over the backbone; None when the worktree
        matches the backbone or there is a problem.

    """
    database = backbone_database(caches)
    base = stamped(database)
    if base is None:
        return (_absent(database), None)
    if missing := _missing_columns(database):
        return (_renamed(database, missing), None)
    overlay = overlay_database(caches, root=root)
    current = stamped(overlay)
    if current is None:
        return (
            (
                f"No overlay for {root.resolve()} at {overlay}; the codegraph-sync hook "
                "builds it."
            ),
            None,
        )
    if current.sha != base.sha:
        return (
            (
                f"The overlay describes backbone commit {current.sha[:12]}, but the "
                f"backbone is now {base.sha[:12]}; the codegraph-sync hook rebuilds it."
            ),
            None,
        )
    paths = changed(root, sha=base.sha, output=output)
    if paths is None:
        return (_unfetched(base.sha), None)
    if fingerprint(root, sha=base.sha, paths=paths) != current.fingerprint:
        return (
            (
                "The overlay is stale: the worktree changed since the codegraph-sync "
                "hook built it."
            ),
            None,
        )
    if not current.paths:
        return ("", None)
    if missing := _missing_columns(overlay):
        return (_renamed(overlay, missing), None)
    return ("", overlay)


def _absent(database: Path) -> str:
    if database.exists():
        return (
            f"The CodeGraph backbone at {database} is unreadable or unstamped; "
            "codegraph-sync --refresh republishes it."
        )
    return (
        f"No CodeGraph backbone at {database}; the codegraph-sync hook builds one, "
        "and its --refresh mode keeps it current."
    )


def _unfetched(sha: str) -> str:
    return f"Backbone commit {sha[:12]} is not in this clone; run `git fetch origin`."


def _renamed(database: Path, missing: Sequence[str]) -> str:
    return (
        f"{database} lacks {', '.join(missing)}: CodeGraph's schema changed, and "
        "the walk in `affected.py` must follow it."
    )


def _missing_columns(database: Path) -> list[str]:
    """Return each required ``table.column`` ``database`` lacks; all when unreadable."""
    required = [
        f"{table}.{column}"
        for table, columns in REQUIRED_COLUMNS.items()
        for column in columns
    ]
    try:
        connection = sqlite3.connect(
            f"file:{pathname2url(str(database))}?mode=ro",
            uri=True,
        )
        try:
            present = {
                f"{table}.{column}"
                for table in REQUIRED_COLUMNS
                for (column,) in cast(
                    list[tuple[str]],
                    connection.execute(
                        """SELECT name FROM pragma_table_info(?)""",
                        (table,),
                    ).fetchall(),
                )
            }
        finally:
            connection.close()
    except sqlite3.Error:
        return ["(unreadable)", *required]
    return [column for column in required if column not in present]
