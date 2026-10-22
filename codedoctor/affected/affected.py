"""Print the reverse-dependency closure of the changed files, per CodeGraph.

Reads `.codegraph/codegraph.db` read-only; `codegraph sync` is the writer.
A file's symbols are all seeds unless --diff names its changed lines.

Examples:
  python -m codedoctor.affected lib.py
  git diff -U0 -- lib.py | python -m codedoctor.affected --diff /dev/stdin lib.py

"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypedDict, cast
from urllib.request import pathname2url

import argparse
import configparser
import hashlib
import os
import re
import sqlite3
import tempfile
import threading
import tomllib

from treekle import ReadError, from_plain

from codedoctor.lib.files import walk
from codedoctor.lib.worker_count import worker_count


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class _Flags(Protocol):
    skip_token: str
    depth: int
    database: str
    files: list[str]
    diff: str | None
    emit: str
    workers: int
    no_universe: bool


def main() -> int:
    """Print one nodeid or file path per line for the changed files in argv.

    Returns:
      result: Process exit code (0 on success).

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    flags = _parse_args(parser)
    depth = 0 if _skipped(flags.skip_token) else flags.depth
    lines = harvest(
        Path(flags.database),
        changed=flags.files,
        depth=depth,
        emit=flags.emit,
        workers=flags.workers,
        # Only the ``files`` projection reads it; see ``harvest``.
        saturated=(
            flags.emit == "files"
            and not flags.no_universe
            and _covers_universe(flags.files)
        ),
        hunks={} if flags.diff is None else _parse_diff(Path(flags.diff)),
    )
    for line in lines:
        print(line)  # noqa: T201 -- CLI output is this function's product.
    return 0


def harvest(
    database: Path,
    *,
    changed: list[str],
    depth: int,
    emit: str,
    workers: int = 1,
    saturated: bool = False,
    hunks: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> list[str]:
    """Return the reverse-dependency closure of ``changed``, projected by ``emit``.

    Args:
      database: CodeGraph SQLite index.
      changed: Repository-relative paths of the changed files.
      depth: Maximum reverse-dependency hops.
      emit: ``"nodeids"`` for pytest selectors, ``"files"`` for file paths.
      workers: Seeds walked concurrently. The walk is one independent recursive
        query per seed and dominates the run: over 20 changed files at depth 1,
        every other phase combined costs 0.02s. Sized by the caller's core
        budget, clamped to the seed count. Median of 3 interleaved repeats on
        a 16-core workstation, that changeset ran 84.8s serial against 27.7s at
        5 workers (3.1x); single runs of either vary 4-8%, so one A/B pair
        cannot rank them.
      saturated: ``changed`` already covers every Python file in the tree.
        A closure cannot name a file outside the tree it is drawn from, so
        the ``files`` answer is provably ``changed`` itself and the walk is
        pure cost. `--all-files` is exactly this case: 4220 seeds, 26.6s, to
        return the 4220 files handed in.

        A SEED COUNT is not a substitute and was the first attempt. Cutting off
        above 500 seeds silently dropped real dependents -- at 600 seeds the
        walk reaches 1443 files beyond its input (240%), every one a file
        basedpyright would then not check. Coverage must be given up only when
        it is provably empty, never when it merely got expensive.
      hunks: Changed ``(first, last)`` line ranges per path. A path without
        ranges seeds every symbol in it, which is right for a new file and
        the safe superset otherwise.

    Returns:
      lines: Sorted selectors. ``"files"`` always unions ``changed`` back in, so
        a type checker never sees less than it was handed. ``"nodeids"`` is the
        graph's answer PER CHANGED FILE, each falling back to that file's paired
        sidecar when the graph names nothing for it -- an unusable index, a file
        it has not indexed, or a walk whose every hit was uncollectable.

    """
    # `.pyi` seeds too: `basedpyright` is handed stubs, and a stub's dependents
    # are exactly what a signature change breaks.
    seeded = [path for path in changed if path.endswith((".py", ".pyi"))]
    # A saturated universe makes the walk's answer knowable without walking:
    # every dependent is already in `changed`. Skipped before any handle opens, the same shape as the depth-0 guard below --
    # "do not consult the index", not "discard what it said".
    #
    # `"nodeids"` is deliberately NOT short-circuited here. Saturation proves
    # the FILE set cannot grow; it proves nothing about which of those files'
    # individual tests to select, and returning empty would hand pytest the
    # whole suite. That walk is 48.3s at `--all-files`, against a suite that
    # takes minutes.
    if emit == "files" and saturated:
        return sorted(set(changed))
    patterns = _collection_patterns(Path())
    # A non-positive depth opens nothing rather than passing the walk through:
    # the CTE emits its own seeds at depth 0, and for a changed TEST file those
    # seeds ARE tests, so returning them would dress a stale index up as a
    # selection. `depth > 0`, not truthiness -- only `0` is falsy, so a
    # NEGATIVE depth walked the index and `back.depth < :depth` then failed at
    # every hop, returning those same seed rows.
    # `database.exists()` rather than opening one. Not `connections.handle()`: that would open a handle HERE, on the
    # main thread, which no worker can reuse -- one wasted open per run.
    connections = (
        _Connections(database=database, derived=_ensure_derived(database))
        if depth > 0 and seeded and database.exists()
        else None
    )
    try:
        selected: set[str] = set()
        if seeded:
            with ThreadPoolExecutor(
                max_workers=max(1, min(workers, len(seeded))),
            ) as pool:
                for found in pool.map(
                    partial(
                        _harvest_seed,
                        connections=connections,
                        depth=depth,
                        hunks=hunks or {},
                        emit=emit,
                        patterns=patterns,
                    ),
                    seeded,
                ):
                    selected |= found
        return sorted(selected | set(changed) if emit == "files" else selected)
    finally:
        if connections:
            connections.close()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Connections:
    """Read-only handles on the index, one per worker thread.

    Never one handle shared across the pool: `_touched_table` DROPs and
    recreates a temp table for each seed, so two seeds on one connection would
    read each other's line ranges. That is the whole reason for the split, and
    it is a CORRECTNESS one. The derived tables are shared: each handle attaches
    the one cache `_ensure_derived` built.
    """

    database: Path
    # Attached read-only to every handle; ``None`` when the build failed, which
    # degrades every walk to the sidecar fallback.
    derived: Path | None
    _local: threading.local = field(default_factory=threading.local)
    _opened: list[sqlite3.Connection] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def handle(self) -> sqlite3.Connection | None:
        """Return this thread's connection, opening it on first use.

        Returns:
          result: The sqlite3.Connection | None.

        """
        # `hasattr`, not a truthiness check: `None` is a cached ANSWER here --
        # `_connect` returns it for an unusable index -- and retrying the open
        # per seed would pay the failure N times over.
        if not hasattr(self._local, "connection"):
            connection = _connect(self.database)
            self._local.connection = connection
            if connection:
                with self._lock:
                    self._opened.append(connection)
        return cast(sqlite3.Connection | None, self._local.connection)

    def attach(self, connection: sqlite3.Connection) -> None:
        """Attach the derived tables to ``connection`` once.

        Raises ``sqlite3.Error`` on a missing or unreadable cache, which
        ``_reachable`` turns into the sidecar fallback like any read failure.

        Args:
          connection: This thread's index handle.

        """
        if getattr(self._local, "attached", False):
            return
        if self.derived is None:
            raise sqlite3.OperationalError("derived tables unavailable")
        connection.execute(
            """ATTACH DATABASE ? AS derived""",
            (f"file:{pathname2url(str(self.derived))}?mode=ro",),
        )
        self._local.attached = True

    def close(self) -> None:
        """Close every handle this pool opened."""
        with self._lock:
            for connection in self._opened:
                connection.close()
            self._opened.clear()


# Only failures SQLite raises at open time land here -- a path that is a directory, a
# permission denial. A corrupt file is not one of them: SQLite opens lazily, so garbage
# bytes yield a live handle that raises `DatabaseError` on the first read, which
# `_reachable` and `_selectors` catch. Both routes matter, because the index is an
# unpinned third-party artifact and raising would abort a caller that reads an empty
# selection as "nothing to run" and exits green.
def _connect(database: Path) -> sqlite3.Connection | None:
    """Open ``database`` read-only, or return ``None`` when the OPEN fails."""
    if not database.exists():
        return None
    try:
        # `check_same_thread=False` because the OPENING thread is a pool worker
        # while the CLOSING one is `harvest`'s caller, and sqlite3's default
        # guard rejects that on `close()` alone. Not a claim of concurrent use:
        # `_Connections` hands each worker its own handle, and the pool is
        # joined before anything is closed.
        # `pathname2url`, not an f-string of the path: `?` and `#` are URI
        # delimiters, so a checkout under a directory holding one truncated the
        # name there and took `mode=ro` with it. SQLite does not fail on that --
        # it CREATES the truncated name, writable and empty, so the walk read a
        # file that is not the index and left a stray one in the tree.
        # pragma: no mutate start -- check_same_thread=None is falsy too.
        return sqlite3.connect(
            f"file:{pathname2url(str(database))}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        # pragma: no mutate end
    except sqlite3.Error:
        return None


# ONE seed per call, which is also what makes the pool safe: the walk and the selector
# phase both run on the handle this thread owns, and `_selectors` needs the same handle
# the walk used to resolve owning classes.
def _harvest_seed(
    seed: str,
    *,
    connections: _Connections | None,
    depth: int,
    hunks: Mapping[str, Sequence[tuple[int, int]]],
    emit: str,
    patterns: tuple[list[str], list[str], list[str]],
) -> set[str]:
    """Return one seed's projected closure, walked on this worker's connection."""
    connection = connections.handle() if connections else None
    if connection and connections:
        try:
            connections.attach(connection)
        except sqlite3.Error:
            connection = None
    rows = _reachable(connection, seed=seed, depth=depth, lines=hunks.get(seed, ()))
    if emit == "files":
        return {row[3] for row in rows if row[3].endswith((".py", ".pyi"))}
    # The sidecar is a per-file fallback, not a union: when the graph names
    # individual tests, adding the whole paired file would run every test in it.
    # Measured over 60 commits, the graph alone reaches 100% median recall at
    # depth 1.
    return _selectors(connection, rows=rows, patterns=patterns) or {
        str(pair) for pair in _sidecars([seed])
    }


def _selectors(
    connection: sqlite3.Connection | None,
    *,
    rows: list[tuple[str, str, str, str]],
    patterns: tuple[list[str], list[str], list[str]],
) -> set[str]:
    """Return the pytest selectors among ``rows``, dropping uncollectable ones."""
    if not connection:
        return set()
    functions, classes, files = patterns
    try:
        # A set: the walk yields one row per depth at which a node was reached.
        return {
            selector
            for node_id, kind, name, file_path in rows
            if kind in {"function", "method"}
            and _collected(name, functions)
            # `python_files` too: a `test_*` helper in a non-test module yields
            # a selector pytest collects nothing from, and one such selector
            # takes the whole invocation down with it.
            and any(fnmatch(Path(file_path).name, pattern) for pattern in files)
            and (
                selector := _selector(
                    connection,
                    node_id,
                    kind=kind,
                    name=name,
                    file_path=file_path,
                    classes=classes,
                )
            )
        }
    except sqlite3.Error:
        return set()


# Any unified diff: `git diff`, `diff -u`, or several concatenated, which union. Only
# the new side matters, since the index describes the tree as it is now. `diff -u`
# follows the name with a tab and a timestamp.
def _parse_diff(diff: Path) -> dict[str, list[tuple[int, int]]]:
    """Return ``path -> [(first, last)]`` new-side line ranges in ``diff``."""
    spans: dict[str, list[tuple[int, int]]] = {}
    path: str | None = None
    for line in diff.read_bytes().decode().splitlines():
        match = re.match(
            r"\+\+\+ (?:b/)?([^\t]*)|@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@",
            line,
        )
        if match is None:
            continue
        name, start, length = cast(tuple[str | None, str, str | None], match.groups())
        if name is not None:
            path = None if name == "/dev/null" else name
        elif path is not None:
            first = int(start)
            # A pure deletion reports length 0 at the line it was cut from;
            # the symbol that lost those lines still changed.
            spans.setdefault(path, []).append(
                (first, first + max(int(length or 1), 1) - 1),
            )
    return spans


# A hook cannot ask pre-commit whether a sibling ran, but `SKIP` reaches every hook's
# environment. Reading it is what lets one token disable the index for the whole tier:
# skip the writer and the readers stop reading, rather than silently serving a stale
# graph.
def _skipped(token: str) -> bool:
    """Report whether pre-commit's ``SKIP`` names ``token``."""
    skip = os.environ.get("SKIP")
    return skip is not None and token in {name.strip() for name in skip.split(",")}


# The `tests/` candidate is a SIBLING of the source directory, not a root-level mirror
# of its full path: a `tools/` package may keep its tests in `tools/tests/`, the one
# exception the house style allows to the adjacent-sidecar rule.
def _sidecars(changed: list[str]) -> list[Path]:
    """Return the test files paired by name with ``changed``, where they exist."""
    paired: list[Path] = []
    for path in changed:
        source = Path(path)
        stem = source.stem.removesuffix("_test").removeprefix("test_")
        paired.extend(
            candidate
            for candidate in (
                source.with_name(f"{stem}_test.py"),
                source.with_name(f"test_{stem}.py"),
                source.parent / "tests" / f"{stem}_test.py",
                source.parent / "tests" / f"test_{stem}.py",
            )
            if candidate.is_file()
        )
    return paired


# ONE seed, never a list: `harvest` must know which changed file produced a hit so an
# unindexed file still gets its own sidecar, and a batched query returns an
# undifferentiated row set. Accepting a list is what let one indexed file suppress every
# other file's fallback.
#
# ``lines`` rolls each changed hunk to the symbol whose body contains it, so depth means
# "callers of what I changed" rather than "callers of anything in that file" -- editing
# one function in a 71-symbol module otherwise seeded all 71 and returned their
# dependents too. Empty ``lines`` seeds the whole file, which is what a new or wholly-
# rewritten file needs.
#
# Swallows every ``sqlite3.Error`` for the reason ``_connect`` documents.
def _reachable(
    connection: sqlite3.Connection | None,
    *,
    seed: str,
    depth: int,
    lines: Sequence[tuple[int, int]] = (),
    # `UNION`, not `UNION ALL`: the graph has cycles and only the deduplicating
    # form terminates.
    #
    # `kind != 'file'` on the walk. A file node carries an `imports` edge per
    # importer, so one hop through it reaches every importer of the whole file
    # rather than of the changed symbol -- and two hops reach
    # the shared test runner module, which 1454 files import.
    #
    # Two arms feed `back`: resolved `edges`, and the `unresolved_refs` rows
    # CodeGraph recorded but could not bind. Each carries its own admission
    # test, documented at the arm.
    sql: str = """
WITH RECURSIVE seed(id) AS (
  SELECT nodes.id FROM nodes
  LEFT JOIN touched ON touched.first <= nodes.end_line
                   AND touched.last >= nodes.start_line
  WHERE nodes.file_path = :seed AND nodes.kind != 'file'
    AND (touched.first IS NOT NULL OR NOT :scoped)
),
back(id, depth) AS (
  SELECT id, 0 FROM seed
  UNION
  SELECT edges.source, back.depth + 1
  FROM edges
  JOIN back ON edges.target = back.id
  JOIN nodes source ON source.id = edges.source
  JOIN nodes target ON target.id = edges.target
  WHERE back.depth < :depth AND source.kind != 'file'
    -- Traverse an edge to a MULTIPLY-DEFINED name only when the source file
    -- imports the module that definition lives in. `codegraph node <name>`
    -- answers such a name with every definition and "pick the one you need";
    -- the edge carries one of them, and taking it unexamined put
    -- `sift/pruning_test.py` in a `sessions/types.py` changeset though it
    -- imports its `SystemMessage` from `trackinizer`. A name with ONE
    -- definition is left alone: CodeGraph also resolves on constructor shape,
    -- which is how an aliased import legitimately reaches another module.
    -- Unless it is PRIVATE: CodeGraph bound every `self._clock()` in the tree
    -- to the sole top-level `_clock`, and a `_name` is never aliased or
    -- re-exported, so only an importer of its module can reach it.
    --
    -- `modules` is a Python-built derived table mapping `file_path -> dotted
    -- module`, populated rather than derived here: SQL `replace(path, '.py',
    -- '')` strips that substring ANYWHERE, turning `api.pyi` into `apii`.
    --
    -- `anycount` and `imports` are the same construct for the same reason:
    -- correlated work the CTE would otherwise redo per candidate row. See
    -- `_build_derived`.
    --
    -- The suffix match is anchored on a COMPONENT BOUNDARY, via `_boundary`:
    -- a dotted module path is a sequence of components, not a string, and a
    -- bare `LIKE '%' || name` matched any tail -- importing stdlib `types`
    -- claimed `trackinizer/trax/run/custom_types.py`, `time` claimed
    -- `sagent/types/runtime.py`. Same defect as the `replace(path, '.py', '')`
    -- above, one operand over.
    AND (
      source.file_path = target.file_path
      OR ((SELECT n FROM anycount WHERE name = target.name) < 2
          AND (substr(target.name, 1, 1) != '_'
               -- `from pkg import mod` records only `pkg`.
               OR EXISTS (
                 SELECT 1 FROM imports imported
                 JOIN modules ON modules.file_path = target.file_path
                 WHERE imported.file_path = source.file_path
                   AND (modules.package = imported.module
                        OR modules.package LIKE imported.boundary ESCAPE '\\'))))
      OR EXISTS (
        SELECT 1 FROM imports imported
        JOIN modules ON modules.file_path = target.file_path
        WHERE imported.file_path = source.file_path
          AND (modules.dotted = imported.module
               OR modules.dotted LIKE imported.boundary ESCAPE '\\')
      )
    )
  UNION
  -- References CodeGraph SAW but could not bind: a call through a
  -- dynamically imported module (`importlib.import_module("pkg.mod")`) lands
  -- here rather than in `edges`.
  --
  -- The caller must IMPORT the defining module. Counting first-party
  -- definitions is not enough: a stdlib or third-party callee has ZERO of
  -- them, so any repo function sharing its name captures every such call --
  -- adding `chain` to `sessions/fuse.py` pulled ten `itertools.chain` call
  -- sites across craftax, priml, and sagent into that file's changeset.
  --
  -- Reads `refs`/`defcount`/`imports`, not `unresolved_refs`/`nodes` directly.
  -- This arm was ~99% of the walk's runtime against the base tables; see
  -- `_build_derived` for what each replaces and the measurement.
  SELECT unresolved_refs.from_node_id, back.depth + 1
  FROM refs AS unresolved_refs
  JOIN nodes target ON target.id = back.id
  JOIN back ON target.name = unresolved_refs.name_tail
  JOIN modules ON modules.file_path = target.file_path
  WHERE back.depth < :depth
    AND target.kind IN ('function', 'method', 'class')
    AND 1 = (SELECT n FROM defcount WHERE name = unresolved_refs.name_tail)
    AND EXISTS (
      SELECT 1 FROM imports imported
      WHERE imported.file_path = unresolved_refs.file_path
        AND (modules.dotted = imported.module
             OR modules.dotted LIKE imported.boundary ESCAPE '\\')
    )
)
SELECT nodes.id, nodes.kind, nodes.name, nodes.file_path
FROM back JOIN nodes ON nodes.id = back.id
""",
) -> list[tuple[str, str, str, str]]:
    """Return ``(id, kind, name, file_path)`` rows reachable from one seed file."""
    if not connection:
        return []
    try:
        _touched_table(connection, lines)
        return cast(
            list[tuple[str, str, str, str]],
            connection.execute(
                sql,
                {"seed": seed, "depth": depth, "scoped": bool(lines)},
            ).fetchall(),
        )
    except sqlite3.Error:
        return []


def _touched_table(
    connection: sqlite3.Connection,
    lines: Sequence[tuple[int, int]],
) -> None:
    """Replace the ``touched`` line-range table with this seed's hunks."""
    connection.executescript(
        """
        DROP TABLE IF EXISTS temp.touched;
        CREATE TEMP TABLE touched (first INTEGER, last INTEGER);
        """,
    )
    connection.executemany("""INSERT INTO touched VALUES (?, ?)""", lines)


# Built ONCE per index version into a file beside the index, then attached read-only by
# every worker of every run. Building per connection cost 1.2s per worker, so a pool of
# 8 took 9.3s against 1.4s serial -- workers made the harvest slower.
#
# The key covers the index file and its WAL (CodeGraph writes land in the WAL until a
# checkpoint) plus this module, so a sync or a change to the build invalidates it. The
# build goes to a private temp file and is renamed into place only once complete: a
# build that dies partway publishes nothing, so no run can trust half the tables.
def _ensure_derived(database: Path) -> Path | None:
    """Return the derived-table cache for ``database``, building it if stale."""
    stats = [
        (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (database, Path(f"{database}-wal"), _THIS)
        if path.exists()
    ]
    key = hashlib.sha256(repr(stats).encode()).hexdigest()[:16]
    prefix = f"{database.name}.affected-"
    cache = database.with_name(f"{prefix}{key}")
    if cache.exists():
        return cache
    handle, scratch = tempfile.mkstemp(dir=database.parent, prefix=f"{prefix}tmp-")
    os.close(handle)
    try:
        # `uri=True`: ATTACH reads a `file:` URI only on a URI connection, and
        # the URI is what carries `mode=ro` for the index.
        connection = sqlite3.connect(f"file:{pathname2url(scratch)}", uri=True)
        try:
            connection.execute(
                """ATTACH DATABASE ? AS source""",
                (f"file:{pathname2url(str(database))}?mode=ro",),
            )
            _build_derived(connection)
            connection.commit()
        finally:
            connection.close()
        Path(scratch).replace(cache)
    except sqlite3.Error:
        Path(scratch).unlink(missing_ok=True)
        return None
    for stale in database.parent.glob(f"{prefix}*"):
        if stale != cache and not stale.name.startswith(f"{prefix}tmp-"):
            stale.unlink(missing_ok=True)
    return cache


# `modules` is built in Python because the path arithmetic has to strip a SUFFIX: SQL's
# `replace(path, '.py', '')` removes that substring anywhere it appears, so `api.pyi`
# became `apii` and matched no import.
#
# The other four exist purely so the recursive CTE stops re-deriving them. Each
# replaces a construct SQLite cannot index from inside the walk, and together they are
# the difference between a walk that finishes and one that does not -- 25 seeds at depth
# 1 went 32.3s to 0.34s, byte-identical rows:
#
# `refs` the `reference_kind = 'calls'` filter. `unresolved_refs` has no index on it, so
# every hop built an AUTOMATIC PARTIAL COVERING INDEX over 276k rows and threw it away.
# `defcount` the `COUNT(DISTINCT file_path)` guard on the unresolved arm, `anycount` and
# its counterpart on the edges arm: correlated aggregates re-run per candidate row (`USE
# TEMP B-TREE FOR count`). `imports` the `kind = 'import'` probes, which read a whole
# file's nodes by `file_path` and discarded ~95% of them.
def _build_derived(connection: sqlite3.Connection) -> None:
    """Materialize the walk's correlated subqueries as indexed tables."""
    connection.executescript(
        """
        CREATE TABLE modules (file_path TEXT, dotted TEXT, package TEXT);
        CREATE INDEX ix_modules ON modules(file_path);
        """,
    )
    connection.executemany(
        """INSERT INTO modules VALUES (?, ?, ?)""",
        (
            (path, _dotted(path), _dotted(path).rpartition(".")[0])
            for (path,) in cast(
                list[tuple[str]],
                connection.execute(
                    """SELECT DISTINCT file_path FROM source.nodes
                    WHERE file_path LIKE '%.py' OR file_path LIKE '%.pyi'""",
                ).fetchall(),
            )
        ),
    )
    connection.executescript(
        """
        CREATE TABLE refs AS
          SELECT from_node_id, name_tail, file_path FROM source.unresolved_refs
          WHERE reference_kind = 'calls';
        CREATE INDEX ix_refs ON refs(name_tail);
        CREATE TABLE defcount AS
          SELECT name, COUNT(DISTINCT file_path) AS n FROM source.nodes
          WHERE kind IN ('function', 'method', 'class') GROUP BY name;
        CREATE INDEX ix_defcount ON defcount(name);
        CREATE TABLE anycount AS
          SELECT name, COUNT(DISTINCT file_path) AS n FROM source.nodes
          WHERE kind != 'import' GROUP BY name;
        CREATE INDEX ix_anycount ON anycount(name);
        CREATE TABLE imports (file_path TEXT, module TEXT, boundary TEXT);
        CREATE INDEX ix_imports ON imports(file_path);
        """,
    )
    connection.executemany(
        """INSERT INTO imports VALUES (?, ?, ?)""",
        (
            # `_module(...)`, not the raw `qualified_name`: the walk's equality
            # arm and its LIKE arm must test the SAME name. Storing the raw one
            # left `impl = '.impl'` false for a relative import of a TOP-LEVEL
            # module, which is the one case no tail pattern can cover either.
            (path, _module(qualified_name), _boundary(qualified_name))
            for path, qualified_name in cast(
                list[tuple[str, str]],
                connection.execute(
                    """SELECT file_path, qualified_name FROM source.nodes
                    WHERE kind = 'import'""",
                ).fetchall(),
            )
        ),
    )


def _dotted(path: str) -> str:
    """Return the dotted module a ``.py`` or ``.pyi`` path defines."""
    return (
        Path(path)
        .with_suffix("")
        .as_posix()
        .removesuffix("/__init__")
        .replace("/", ".")
    )


# The companion to `_boundary`, and the reason both exist: the walk tests an import two
# ways -- equality against a top-level module, and a boundary pattern against a dotted
# tail -- and BOTH must see the same name. A top-level module has no dot for the pattern
# to anchor on, so equality is its only route, and a relative import that kept its
# leading dots failed that comparison for a module the pattern could not reach either.
def _module(qualified_name: str) -> str:
    """Return the module ``qualified_name`` names, with relative dots removed."""
    if qualified_name.startswith(("./", "../")):
        return ""
    return qualified_name.lstrip(".")


# A dotted module path is a sequence of components, so `types` must match `a.b.types`
# and not `a.b.custom_types`. Precomputed per import row rather than assembled in SQL,
# because the three cases do not share one expression:
#
# - ABSOLUTE (`pkg.lib.userdirs`): the module itself, or any dotted tail of one --
# `'%.' || name` plus the equality, since a top-level module (`twin` -> `twin`) contains
# no dot to anchor on. - RELATIVE (`.main`, `..pkg.mod`): the leading dots say how far
# UP the package tree to walk, so they are a RESOLUTION instruction and never part of
# the name matched. They are STRIPPED, and the remainder is anchored like any other
# tail. Keeping them literal builds `%..pkg`, which needs two consecutive dots -- a
# string no module name contains -- so every multi-dot import silently matched nothing.
# 15 Python imports in this repository are relative; an anchor that ignored them
# entirely made `credit_assignment/compare.py` stop reaching its own `main.py`. -
# Vendored JS (`./api`, `../lib/api`): a slash path, not a dotted one. Nothing under
# `modules.dotted` can match it, and its `.` and `..` must not be read as Python
# relative syntax. - NAMELESS (`.`, `..`, or empty): `from . import x` names a PACKAGE,
# so nothing remains to anchor. Returns the empty pattern, which matches only the empty
# string and so admits no module. The tempting `f"%.{''}"` is `"%."`, which every dotted
# module matches -- broader than the unanchored `LIKE` this anchor replaced, and failing
# OPEN rather than closed.
#
# Covers only the TAIL match; the caller ORs in an equality test for the top-level
# module (`twin` -> `twin`), which has no dot to anchor on and which no tail pattern can
# express.
#
# `_` and `%` are escaped: `_` is LIKE's single-character wildcard and it is legal in a
# module name, so `custom_json` would otherwise also match `customXjson`.
def _boundary(qualified_name: str) -> str:
    r"""Return the LIKE pattern matching ``qualified_name`` at a component boundary."""
    if qualified_name.startswith(("./", "../")):
        # A path, not a module. Match nothing rather than mis-parsing the dots.
        return ""
    tail = qualified_name.lstrip(".")
    if not tail:
        return ""
    escaped = tail.replace("\\", r"\\").replace("_", r"\_").replace("%", r"\%")
    return f"%.{escaped}"


# All three are read from configuration rather than hardcoded, because all three are
# settable and none is a name prefix. Each was a bug once: a ``startswith("test_")``
# function filter dropped 25 collectable tests in this repository, treating every class
# as collectable emitted selectors pytest rejects, and ignoring ``python_files`` emitted
# a nodeid for a ``test_*`` helper in a production module. One bad selector fails the
# whole invocation.
class _Pytest(TypedDict, total=False):
    ini_options: dict[str, object]


class _Tools(TypedDict, total=False):
    pytest: _Pytest


class _Pyproject(TypedDict, total=False):
    tool: _Tools


def _collection_patterns(root: Path) -> tuple[list[str], list[str], list[str]]:
    """Return pytest's ``(python_functions, python_classes, python_files)`` globs."""
    settings: dict[str, object] = {}
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as handle:
            document = tomllib.load(handle)
        # A malformed table is pytest's to reject; selection keeps the defaults.
        try:
            tools = from_plain(document, _Pyproject).get("tool", _Tools())
            settings = tools.get("pytest", _Pytest()).get("ini_options", {})
        except ReadError:
            settings = {}
    ini = root / "pytest.ini"
    if ini.is_file():
        parser = configparser.ConfigParser(default_section="pytest")
        parser.read(ini)
        settings = dict(parser.defaults()) or settings
    return (
        _globs(settings.get("python_functions"), default=["test*"]),
        _globs(settings.get("python_classes"), default=["Test*"]),
        # Both spellings, mirroring `_pytest/python.py:91`. Defaulting to one
        # made every `*_test.py` selector uncollectable wherever a project
        # relies on the default rather than restating it.
        _globs(settings.get("python_files"), default=["test_*.py", "*_test.py"]),
    )


# An empty result falls back to ``default`` rather than being returned: no pattern
# matches NO name, so a blank `python_files =` would select zero tests instead of
# meaning "unset". Same shape as an empty pathspec meaning "every path" to git -- a
# degenerate input must never silently invert.
def _globs(value: object, *, default: list[str]) -> list[str]:
    """Normalize an ini value pytest accepts as either a list or a word list."""
    if isinstance(value, str):
        return value.split() or default
    # ``convert``, not ``isinstance(value, list)``: the narrowing would
    # hand it a ``list[Unknown]`` and lose the element type the conversion
    # establishes.
    try:
        words = from_plain(value, list[object], default=[])
    except ReadError:
        return default
    return [str(word) for word in words] or default


# Mirrors ``_pytest/python.py`` ``_matches_prefix_or_glob_option``: every
# pattern is also a name prefix, so ``python_functions = check`` collects
# ``check_helper``. Glob-only matching silently drops those tests.
def _collected(name: str, patterns: Sequence[str]) -> bool:
    """Report whether pytest collects ``name`` under ``patterns``."""
    return any(name.startswith(p) or fnmatch(name, p) for p in patterns)


# A method inside a class pytest will not collect has no nodeid at all -- emitting
# ``file::Class::name`` for it makes pytest exit 4 and takes every other selector in the
# same invocation down with it.
def _selector(
    connection: sqlite3.Connection,
    node_id: str,
    *,
    kind: str,
    name: str,
    file_path: str,
    classes: list[str],
) -> str:
    """Return the pytest nodeid for this symbol, or empty when uncollectable."""
    if kind != "method":
        return f"{file_path}::{name}"
    row = cast(
        tuple[str] | None,
        connection.execute(
            """SELECT nodes.name FROM edges JOIN nodes ON nodes.id = edges.source
            WHERE edges.target = ? AND edges.kind = 'contains' AND nodes.kind = 'class'""",
            (node_id,),
        ).fetchone(),
    )
    if not row:
        return f"{file_path}::{name}"
    if not _collected(row[0], classes):
        return ""
    return f"{file_path}::{row[0]}::{name}"


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> _Flags:
    """Register this script's flags and parse ``argv``."""
    parser.add_argument("files", nargs="*", help="Changed repository-relative paths.")
    parser.add_argument(
        "--diff",
        metavar="PATH",
        help="Unified diff naming the changed lines, e.g. /dev/stdin.",
    )
    parser.add_argument(
        "--emit",
        choices=("nodeids", "files"),
        default="nodeids",
        help="Projection of the closure.",
    )
    parser.add_argument(
        "--database",
        default=".codegraph/codegraph.db",
        help="CodeGraph SQLite index.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Reverse-dependency hops to traverse.",
    )
    parser.add_argument(
        "--skip-token",
        default="harvest-affected",
        help="Ignore the index when pre-commit's SKIP names this hook.",
    )
    parser.add_argument(
        "--workers",
        type=worker_count,
        # Serial by default, and PASSED IN: the gate that walks sizes the
        # count, so this walk shares the budget governing its checker.
        default=1,
        help="Seeds walked concurrently. Empty or non-positive means serial.",
    )
    parser.add_argument(
        "--no-universe",
        action="store_true",
        help="Always walk, even when the inputs cover every Python file in the tree.",
    )
    return cast(_Flags, parser.parse_args(argv))


# A tree with no Python is NOT covered: the empty set is trivially covered by any
# input, and asserting saturation there would skip every walk -- the same shape as the
# blank glob in `_globs`.
def _covers_universe(changed: Sequence[str]) -> bool:
    """Report whether ``changed`` names every Python file under the cwd."""
    universe = walk.expand([Path()], suffixes={".py", ".pyi"})
    return bool(universe) and {p.as_posix() for p in universe} <= {
        Path(path).as_posix() for path in changed
    }
