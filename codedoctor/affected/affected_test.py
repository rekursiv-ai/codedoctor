"""Tests for ``affected``: closure depth, both projections, and degraded modes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import Mock

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import threading

import pytest

from codedoctor.affected import affected, backbone
from codedoctor.affected.affected import _collection_patterns, _skipped, harvest


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_CWD: Final = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test from ``tmp_path``, never the repository root.

    `harvest` resolves `python_files`/`python_functions`/`python_classes` from
    the CWD, and `_sidecars` probes the CWD for paired files. A test that omits
    the chdir reads THIS repository's pytest config instead, so it passes or
    fails on a setting it never states -- non-hermetic per STYLE.md Testing.
    Autouse rather than per-test because omitting it is silent: the test still
    passes here, and only diverges once the two configs do.
    """
    monkeypatch.chdir(tmp_path)


# The harvester's only contract is that pytest accepts what it prints, and only pytest
# knows its own collection rules -- `python_functions`, `python_classes`, and
# `python_files` are all configurable and none is expressible as a name prefix.
# Asserting against a hand-written prefix instead is what let a fixture and the code
# agree on `Suite::test_method`, which pytest rejects.
def _collects(tmp_path: Path, nodeids: list[str]) -> subprocess.CompletedProcess[str]:
    """Ask a real pytest to collect ``nodeids``."""
    return subprocess.run(  # noqa: S603 -- The test invokes the fixed Python pytest harness with fixture-controlled arguments.
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            "--rootdir",
            str(tmp_path),
            "-c",
            str(tmp_path / "pytest.ini"),
            *nodeids,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
        # Neither the outer run's SKIP nor its addopts may reach this pytest:
        # `-p no:randomly` and marker deselection would change the collected
        # count this asserts on.
        env={**os.environ, "SKIP": "", "PYTEST_ADDOPTS": ""},
    )


# Always sets `SKIP`, never inherits it: the harvester reads that variable to decide
# whether to consult the index, so an ambient value from the operator's shell or an
# outer pre-commit run would silently change what these tests measure.
def _run_cli(*argv: str, skip: str = "") -> subprocess.CompletedProcess[str]:
    """Run the script as a subprocess with ``SKIP`` set to exactly ``skip``."""
    return subprocess.run(  # noqa: S603 -- The test invokes the repository's fixed affected.py entry point without a shell.
        [sys.executable, "-m", "codedoctor.affected", *argv],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SKIP": skip},
    )


# Records before delegating, never after: `sqlite3.connect` raises on a corrupt file, so
# a counter on the return path counts zero for exactly the case a caching test needs to
# observe -- the repeated FAILING open.
#
# States `_connect`'s keyword arguments rather than forwarding `**kwargs`: a splat of
# `object` matches none of `connect`'s overloads.
def _connect_double(
    attempts: list[str],
    observe: Callable[[sqlite3.Connection], None] = lambda connection: None,  # noqa: ARG005 -- The default callback preserves the injectable observation seam without using its connection.
) -> Callable[..., sqlite3.Connection]:
    """Return a `sqlite3.connect` double recording every ATTEMPT in ``attempts``."""
    real = sqlite3.connect

    def connect(
        database: str,
        *,
        uri: bool = False,
        check_same_thread: bool = True,
    ) -> sqlite3.Connection:
        attempts.append(database)
        connection = real(database, uri=uri, check_same_thread=check_same_thread)
        observe(connection)
        return connection

    return connect


# The recursive walk consumes its own ``edges`` reads before the selector phase runs
# class-owner lookups, so the threshold must clear the walk and land inside the phase
# under test -- a blanket denial trips `_reachable`'s guard instead and proves nothing.
# Injected through an authorizer because `sqlite3.Connection` is immutable:
# `monkeypatch.setattr` on `execute` raises `TypeError`.
def _deny_edges_after(permitted: int) -> Callable[..., int]:
    """Return an authorizer allowing ``permitted`` ``edges`` reads, then denying."""
    seen: list[int] = []

    def authorizer(action: int, first: object, *args: object) -> int:
        del action, args
        if first == "edges":
            seen.append(1)
            if len(seen) > permitted:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    return authorizer


# Every node is a case the harvester must decide, and each was once a bug:
#
# - ``lib_test.py`` one hop out, ``far_test.py`` two, so depth separates them. -
# ``TestSuite.test_owned`` becomes ``Class::name``; ``Suite.test_method`` must be
# dropped, since pytest collects only ``python_classes`` matches. - ``testlegacy``
# carries no underscore, which a ``test_`` filter missed. - ``test_public`` sits in
# ``lib.py``, which ``python_files`` excludes. - ``hub_test.py`` reaches ``lib.py`` only
# through a file node, which the walk must refuse to traverse. - ``n:other`` is an
# untouched symbol inside a changed file. - ``dyn_test.py`` reaches ``helper`` only
# through `unresolved_refs`. - ``other_test.py`` holds an edge to the WRONG ``Widget``
# definition.
#
# Line ranges are ten apart and never adjacent, so a hunk can name exactly one symbol.
def _index(tmp_path: Path) -> Path:
    """Build a CodeGraph-shaped index: ``lib.py`` <- ``mid.py`` <- ``far_test.py``."""
    database = tmp_path / "codegraph.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT,"
        " file_path TEXT, qualified_name TEXT, start_line INTEGER, end_line INTEGER);"
        "CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);"
        "CREATE TABLE unresolved_refs (from_node_id TEXT, reference_name TEXT,"
        " reference_kind TEXT, file_path TEXT, name_tail TEXT);",
    )
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:lib", "function", "helper", "lib.py", "helper", 1, 9),
            ("n:other", "function", "untouched", "lib.py", "untouched", 11, 19),
            ("n:mid", "function", "middle", "mid.py", "middle", 21, 29),
            (
                "n:libtest",
                "function",
                "test_helper",
                "lib_test.py",
                "test_helper",
                31,
                39,
            ),
            ("n:suite", "class", "Suite", "lib_test.py", "Suite", 41, 49),
            ("n:method", "method", "test_method", "lib_test.py", "test_method", 51, 59),
            ("n:owner", "class", "TestSuite", "lib_test.py", "TestSuite", 61, 69),
            ("n:owned", "method", "test_owned", "lib_test.py", "test_owned", 71, 79),
            ("n:legacy", "function", "testlegacy", "lib_test.py", "testlegacy", 81, 89),
            ("n:prodtest", "function", "test_public", "lib.py", "test_public", 91, 99),
            ("n:fartest", "function", "test_far", "far_test.py", "test_far", 101, 109),
            ("n:notatest", "function", "fixture_helper", "lib_test.py", "x", 200, 209),
            ("file:lib", "file", "lib.py", "lib.py", "lib.py", 111, 119),
            ("n:hubtest", "function", "test_hub", "hub_test.py", "test_hub", 121, 129),
            # `Widget` is defined twice. `other_test.py` imports the `twin.py`
            # one, but the edge points at `lib.py`'s -- the ambiguity CodeGraph
            # surfaces (`codegraph node` returns BOTH and says "pick the one you
            # need") and a caller must resolve.
            ("n:widget", "class", "Widget", "lib.py", "Widget", 131, 139),
            ("n:twin", "class", "Widget", "twin.py", "Widget", 141, 149),
            (
                "n:othertest",
                "function",
                "test_widget",
                "other_test.py",
                "test_widget",
                151,
                159,
            ),
            ("imp:other", "import", "twin", "other_test.py", "twin", 161, 169),
            # Reaches `helper` only through a dynamically imported module, so
            # CodeGraph records the call in `unresolved_refs`, never `edges`.
            # It still IMPORTS `lib`; only the attribute binds at runtime.
            (
                "n:dyntest",
                "function",
                "test_dynamic",
                "dyn_test.py",
                "test_dynamic",
                171,
                179,
            ),
            ("imp:dyn", "import", "lib", "dyn_test.py", "lib", 181, 189),
        ],
    )
    connection.execute(
        "INSERT INTO unresolved_refs VALUES (?, ?, ?, ?, ?)",
        ("n:dyntest", "loaded.helper", "calls", "dyn_test.py", "helper"),
    )
    connection.executemany(
        "INSERT INTO edges VALUES (?, ?, ?)",
        [
            ("n:libtest", "n:lib", "calls"),
            ("n:method", "n:lib", "calls"),
            ("n:owned", "n:lib", "calls"),
            ("n:legacy", "n:lib", "calls"),
            ("n:prodtest", "n:lib", "calls"),
            ("n:notatest", "n:lib", "calls"),
            ("n:mid", "n:lib", "calls"),
            ("n:fartest", "n:mid", "calls"),
            ("n:suite", "n:method", "contains"),
            ("n:owner", "n:owned", "contains"),
            ("file:lib", "n:lib", "contains"),
            ("n:hubtest", "file:lib", "imports"),
            ("n:othertest", "n:widget", "instantiates"),
        ],
    )
    connection.commit()
    connection.close()
    return database


# The `pytest.ini` is what `harvest` reads for collection patterns, so it must state
# them rather than inherit: a test that relies on the repository's own config asserts
# against a setting it never names.
def _tree(tmp_path: Path) -> Path:
    """Write the source tree `_index` describes, so a real pytest can collect it."""
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\npython_files = *_test.py test_*.py\n",
        encoding="utf-8",
    )
    (tmp_path / "lib.py").write_text(
        "def helper(): pass\n\n\ndef test_public(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "lib_test.py").write_text(
        "def test_helper(): pass\n\n\n"
        "class Suite:\n    def test_method(self): pass\n\n\n"
        "class TestSuite:\n    def test_owned(self): pass\n\n\n"
        "def testlegacy(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "dyn_test.py").write_text(
        "def test_dynamic(): pass\n",
        encoding="utf-8",
    )
    return tmp_path


def test_only_the_symbol_containing_a_changed_line_is_seeded(tmp_path: Path) -> None:
    """Depth means "callers of what I changed", not "of anything in that file".

    `lib.py` holds `helper` at lines 1-9 and `untouched` at 11-19, and nothing
    calls `untouched`. A hunk inside it therefore yields an empty closure, and
    the selection falls to the sidecar rather than to `helper`'s dependents.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    assert harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
        hunks={"lib.py": [(12, 13)]},
    ) == ["lib_test.py"]


def test_a_hunk_inside_a_symbol_selects_its_dependents(tmp_path: Path) -> None:
    """The other half of line seeding: a hunk in `helper` does select them."""
    database = _index(tmp_path)
    _tree(tmp_path)
    assert "lib_test.py::test_helper" in harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
        hunks={"lib.py": [(2, 3)]},
    )


def test_a_file_without_hunks_seeds_every_symbol(tmp_path: Path) -> None:
    """No diff for a file means the whole file changed, as for a new file."""
    database = _index(tmp_path)
    _tree(tmp_path)
    assert "lib_test.py::test_helper" in harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
        hunks={"mid.py": [(21, 29)]},
    )


def test_an_unresolved_call_reaches_its_one_candidate(tmp_path: Path) -> None:
    """A dynamically imported callee is evidence CodeGraph records, not discards.

    `importlib.import_module("pkg.mod")` binds at runtime, so the call lands in
    `unresolved_refs` with a `name_tail` rather than in `edges`, and a walk over
    `edges` alone never sees it. Admitted only when the tail names exactly ONE
    first-party definition AND the caller imports that module -- the companion
    test covers the second half.
    """
    _tree(tmp_path)
    assert "dyn_test.py" in harvest(
        _index(tmp_path),
        changed=["lib.py"],
        depth=1,
        emit="files",
    )


def test_an_unresolved_call_needs_the_caller_to_import_the_definition(
    tmp_path: Path,
) -> None:
    """A stdlib callee has no first-party definition to be ambiguous WITH.

    `stray_test.py` calls something named `helper` without importing `lib`, so
    the name matches by coincidence -- an `itertools.chain` call site, say.
    Counting definitions alone let adding `chain` to `sessions/fuse.py` pull
    ten such call sites across craftax, priml, and sagent into its changeset.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("n:stray", "function", "test_stray", "stray_test.py", "test_stray", 1, 9),
    )
    connection.execute(
        "INSERT INTO unresolved_refs VALUES (?, ?, ?, ?, ?)",
        ("n:stray", "itertools.helper", "calls", "stray_test.py", "helper"),
    )
    connection.commit()
    connection.close()
    _tree(tmp_path)
    assert "stray_test.py" not in harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="files",
    )


def test_an_import_matches_a_whole_module_component_not_a_bare_suffix(
    tmp_path: Path,
) -> None:
    """A dotted path is a sequence of components, and `LIKE '%x'` ignores that.

    `stdlib_test.py` imports `types` -- the standard library module -- and
    `custom_types.py` ends in those same five characters, so an unanchored
    suffix match reported the import as satisfied and admitted the edge. Real
    instance: importing `types` claimed
    `endpoints/claude/custom_types.py`, and importing `time` claimed
    `sagent/types/runtime.py`. 13 of 133 admitted rows on this repository.

    Uses the AMBIGUOUS `Widget`, since that is the only path where the import
    test runs -- a name with one definition never consults it.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "n:stdlibtest",
                "function",
                "test_stdlib",
                "stdlib_test.py",
                "test_stdlib",
                191,
                199,
            ),
            # Imports stdlib `types`, NOT `custom_types`.
            ("imp:stdlib", "import", "types", "stdlib_test.py", "types", 201, 209),
            ("n:custom", "class", "Widget", "custom_types.py", "Widget", 211, 219),
        ],
    )
    connection.execute(
        "INSERT INTO edges VALUES ('n:stdlibtest', 'n:custom', 'instantiates')",
    )
    connection.commit()
    connection.close()
    assert "stdlib_test.py::test_stdlib" not in harvest(
        database,
        changed=["custom_types.py"],
        depth=1,
        emit="nodeids",
    )


def test_an_unresolved_call_matches_a_module_component_not_a_bare_suffix(
    tmp_path: Path,
) -> None:
    """The unresolved arm carries the same import test, so it carried the bug.

    Fixing one arm and not the other is the failure `AGENTS.md` names: an
    incident has a CLASS, and every sibling of the same shape needs the audit.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:only", "function", "solitary", "custom_lib.py", "solitary", 191, 199),
            (
                "n:suffixtest",
                "function",
                "test_suffix",
                "suffix_test.py",
                "test_suffix",
                201,
                209,
            ),
            # Imports `lib`, which is a SUFFIX of `custom_lib` but not a
            # component of it.
            ("imp:suffix", "import", "lib", "suffix_test.py", "lib", 211, 219),
        ],
    )
    connection.execute(
        "INSERT INTO unresolved_refs VALUES (?, ?, ?, ?, ?)",
        ("n:suffixtest", "loaded.solitary", "calls", "suffix_test.py", "solitary"),
    )
    connection.commit()
    connection.close()
    _tree(tmp_path)
    assert "suffix_test.py" not in harvest(
        database,
        changed=["custom_lib.py"],
        depth=1,
        emit="files",
    )


def test_an_exact_module_name_still_satisfies_the_import_test(tmp_path: Path) -> None:
    """The anchor must not reject the TOP-LEVEL module, which has no dot.

    `'%.' || name` alone matches only a dotted path, so a root-level module
    (`twin.py` -> `twin`) would stop satisfying its own import and the walk
    would drop every ambiguous edge into it. That is the same
    degenerate-inversion this file guards against elsewhere -- a fix for one
    end of a pattern silently breaking the other.

    Adds the edge the base fixture deliberately withholds: there `other_test`
    instantiates `lib.py`'s `Widget`, which is the WRONG-definition case. Here
    it reaches `twin.py`'s, which its import genuinely backs.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO edges VALUES ('n:othertest', 'n:twin', 'instantiates')",
    )
    connection.commit()
    connection.close()
    assert "other_test.py::test_widget" in harvest(
        database,
        changed=["twin.py"],
        depth=1,
        emit="nodeids",
    )


def test_a_relative_import_still_backs_its_own_package(tmp_path: Path) -> None:
    """`from .main import X` records `.main`, which already carries the boundary.

    Prefixing another dot yields `..main`, which matches no module, so an
    anchor written only for absolute imports silently drops every relative one
    -- 15 in this repository, including `credit_assignment/compare.py`, which
    stopped reaching its own `main.py`. The audit that caught it is the one
    `AGENTS.md` requires: the anchor is a CLASS of change, so every import
    spelling had to be checked, not just the one that motivated it.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:pkgw", "class", "Widget", "pkg/impl.py", "Widget", 191, 199),
            (
                "n:reltest",
                "function",
                "test_rel",
                "pkg/rel_test.py",
                "test_rel",
                201,
                209,
            ),
            ("imp:rel", "import", "impl", "pkg/rel_test.py", ".impl", 211, 219),
        ],
    )
    connection.execute(
        "INSERT INTO edges VALUES ('n:reltest', 'n:pkgw', 'instantiates')",
    )
    connection.commit()
    connection.close()
    assert "pkg/rel_test.py::test_rel" in harvest(
        database,
        changed=["pkg/impl.py"],
        depth=1,
        emit="nodeids",
    )


def test_a_multi_dot_relative_import_still_backs_its_package(tmp_path: Path) -> None:
    """`from ..sibling import X` records `..sibling`, and no module holds `..`.

    Treating the leading dots as literal text builds `%..sibling`, which needs
    two CONSECUTIVE dots in a dotted path -- a string no module name can
    contain -- so every such import silently satisfies nothing. The single-dot
    fix landed first and stopped there, which is the sibling-audit failure
    `AGENTS.md` names: the CLASS is "relative import", not "one leading dot".

    The dots encode how far UP the package tree to walk, so they are a
    RESOLUTION instruction, never part of the name being matched.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:sib", "class", "Widget", "proj/sibling.py", "Widget", 191, 199),
            (
                "n:deeptest",
                "function",
                "test_deep",
                "proj/inner/deep_test.py",
                "test_deep",
                201,
                209,
            ),
            (
                "imp:deep",
                "import",
                "sibling",
                "proj/inner/deep_test.py",
                "..sibling",
                211,
                219,
            ),
        ],
    )
    connection.execute(
        "INSERT INTO edges VALUES ('n:deeptest', 'n:sib', 'instantiates')",
    )
    connection.commit()
    connection.close()
    assert "proj/inner/deep_test.py::test_deep" in harvest(
        database,
        changed=["proj/sibling.py"],
        depth=1,
        emit="nodeids",
    )


@pytest.mark.parametrize("qualified_name", ["", ".", "..", "..."])
def test_a_nameless_import_backs_no_module_at_all(qualified_name: str) -> None:
    """`from . import x` names a PACKAGE, not a module, so it anchors nothing.

    Stripping the dots off these leaves an empty tail, and `'%.' || ''` is
    `'%.'` -- a pattern every dotted module in the repository matches. That
    inverts the guard completely: the import test would be satisfied by any
    target, which is broader than the unanchored `LIKE` this whole anchor was
    written to replace.

    Asserted on `_boundary` directly because the walk cannot witness it: the
    index carries no such row today, and a guard that fails open only shows up
    once one appears. Same shape as the empty pathspec in `_hunks` and the
    blank glob in `_globs` -- a degenerate input must never mean "everything".
    """
    assert affected._boundary(qualified_name) == ""


def test_an_empty_boundary_matches_no_module(tmp_path: Path) -> None:
    """The sentinel `_boundary` returns for an unusable name must match NOTHING.

    Companion to the parametrized test above, which pins the sentinel's VALUE;
    this pins its MEANING against real SQL. An empty LIKE pattern matches only
    the empty string, and `modules.dotted` is never empty -- but a sentinel
    changed to `"%"` would pass the value test's negation and admit every edge.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:pkgonly", "class", "Widget", "proj/thing.py", "Widget", 191, 199),
            (
                "n:bare",
                "function",
                "test_bare",
                "proj/bare_test.py",
                "test_bare",
                201,
                209,
            ),
            ("imp:bare", "import", "x", "proj/bare_test.py", ".", 211, 219),
        ],
    )
    connection.execute(
        "INSERT INTO edges VALUES ('n:bare', 'n:pkgonly', 'instantiates')",
    )
    connection.commit()
    connection.close()
    assert "proj/bare_test.py::test_bare" not in harvest(
        database,
        changed=["proj/thing.py"],
        depth=1,
        emit="nodeids",
    )


def test_a_relative_import_of_a_top_level_module_still_backs_it(
    tmp_path: Path,
) -> None:
    """`from .impl import X` beside a ROOT-level `impl.py` matches on neither arm.

    The tail pattern is `%.impl`, and a top-level module's dotted name is bare
    `impl` with no dot to anchor -- that is exactly why the equality arm exists.
    But the equality compares against the RAW `qualified_name`, which still
    carries the leading dot, so `impl = .impl` is false and the import backs
    nothing.

    The same hole the absolute case documents, reached from the other side: the
    equality arm was written for absolute names and never revisited when
    relative ones started being stripped. Both arms must see the same name.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:rootw", "class", "Widget", "impl.py", "Widget", 191, 199),
            (
                "n:roottest",
                "function",
                "test_root",
                "root_test.py",
                "test_root",
                201,
                209,
            ),
            ("imp:root", "import", "impl", "root_test.py", ".impl", 211, 219),
        ],
    )
    connection.execute(
        "INSERT INTO edges VALUES ('n:roottest', 'n:rootw', 'instantiates')",
    )
    connection.commit()
    connection.close()
    assert "root_test.py::test_root" in harvest(
        database,
        changed=["impl.py"],
        depth=1,
        emit="nodeids",
    )


def test_a_torn_derived_build_is_retried_not_cached_as_done(tmp_path: Path) -> None:
    """A half-built derived set must never be published as the cache.

    An index whose schema lacks `unresolved_refs` -- an older CodeGraph, a sync
    interrupted mid-migration -- fails partway through the build. Publishing
    what was built so far would make every later run trust it, the walk would
    query tables that do not exist, and `_reachable` would swallow the error
    into an empty selection: green, having run no tests.
    """
    database = tmp_path / "partial.db"
    connection = sqlite3.connect(database)
    # No `unresolved_refs`: the second CREATE has nothing to select from.
    connection.executescript(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT,"
        " file_path TEXT, qualified_name TEXT, start_line INTEGER, end_line INTEGER);"
        "CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);",
    )
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:src", "function", "helper", "lib.py", "helper", 1, 9),
            ("n:dep", "function", "test_dep", "dep_test.py", "test_dep", 11, 19),
        ],
    )
    connection.execute("INSERT INTO edges VALUES ('n:dep', 'n:src', 'calls')")
    connection.commit()
    connection.close()

    assert affected.ensure_derived(database) is None
    # The retry must REBUILD rather than trust the first attempt's debris.
    assert affected.ensure_derived(database) is None
    assert sorted(path.name for path in tmp_path.iterdir()) == ["partial.db"]


def test_the_derived_tables_are_built_once_per_index_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every worker of every run shares ONE build until the index changes.

    Building per connection cost 1.2s per worker, so a pool of 8 ran 9.3s
    against 1.4s serial: adding workers made the harvest slower.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    builds: list[Path] = []
    real = affected._build_derived

    def counting(connection: sqlite3.Connection) -> None:
        builds.append(database)
        real(connection)

    monkeypatch.setattr(affected, "_build_derived", counting)
    changed = ["lib.py", "mid.py", "twin.py"]
    first = harvest(database, changed=changed, depth=1, emit="nodeids", workers=3)
    second = harvest(database, changed=changed, depth=1, emit="nodeids", workers=3)
    assert len(builds) == 1
    assert first == second

    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO edges VALUES ('n:fartest', 'n:lib', 'calls')")
    connection.commit()
    connection.close()
    assert "far_test.py::test_far" in harvest(
        database,
        changed=changed,
        depth=1,
        emit="nodeids",
        workers=3,
    )
    assert len(builds) == 2


def test_an_underscore_in_a_module_name_is_not_a_like_wildcard(
    tmp_path: Path,
) -> None:
    """`_` matches any single character in LIKE, and module names are full of it.

    Without escaping, importing `custom_types` would also satisfy a module
    named `customXtypes` -- the same accidental-match class the dot anchor
    exists to close, one metacharacter over.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "n:wild",
                "class",
                "Widget",
                "top/pkg/customXtypes.py",
                "Widget",
                191,
                199,
            ),
            (
                "n:wildtest",
                "function",
                "test_wild",
                "wild_test.py",
                "test_wild",
                201,
                209,
            ),
            (
                "imp:wild",
                "import",
                "custom_types",
                "wild_test.py",
                "pkg.custom_types",
                211,
                219,
            ),
        ],
    )
    connection.execute(
        "INSERT INTO edges VALUES ('n:wildtest', 'n:wild', 'instantiates')",
    )
    connection.commit()
    connection.close()
    assert "wild_test.py::test_wild" not in harvest(
        database,
        changed=["top/pkg/customXtypes.py"],
        depth=1,
        emit="nodeids",
    )


def test_an_ambiguous_name_resolves_through_the_source_file_imports(
    tmp_path: Path,
) -> None:
    """A name with two definitions must not attribute the edge to the wrong one.

    `codegraph node Widget` returns BOTH definitions and tells the caller to
    pick; the edge alone does not carry that choice. `other_test.py` imports
    `twin`, so a change to `lib.py`'s `Widget` does not reach it.
    """
    database = _index(tmp_path)
    assert "other_test.py::test_widget" not in harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
    )


def test_an_unambiguous_name_needs_no_import_to_back_it(tmp_path: Path) -> None:
    """Only AMBIGUOUS names consult imports; a sole definition stays trusted.

    `helper` is defined once, so its edges are unambiguous however the caller
    reached it -- an alias, a re-export, or a structural match CodeGraph
    resolved on constructor shape rather than on name.
    """
    database = _index(tmp_path)
    assert "lib_test.py::test_helper" in harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
    )


def test_a_unique_private_name_still_needs_an_import(tmp_path: Path) -> None:
    """A sole ``_clock`` must not capture every ``self._clock()`` in the tree.

    CodeGraph bound unrelated attribute calls to the only top-level ``_clock``,
    so a benchmark script pulled trackinizer and knowbuddy into its closure.
    A private name is never aliased or re-exported, so only an importer of
    its module can reach it.
    """
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:clock", "function", "_clock", "pkg/mod.py", "_clock", 1, 9),
            ("n:stranger", "function", "test_s", "stranger_test.py", "test_s", 1, 9),
            ("n:direct", "function", "test_d", "direct_test.py", "test_d", 1, 9),
            ("imp:direct", "import", "pkg.mod", "direct_test.py", "pkg.mod", 11, 19),
            # `from pkg import mod` records the PACKAGE, never the module.
            ("n:package", "function", "test_p", "package_test.py", "test_p", 1, 9),
            ("imp:package", "import", "pkg", "package_test.py", "pkg", 11, 19),
        ],
    )
    connection.executemany(
        "INSERT INTO edges VALUES (?, ?, ?)",
        [
            ("n:stranger", "n:clock", "calls"),
            ("n:direct", "n:clock", "calls"),
            ("n:package", "n:clock", "calls"),
        ],
    )
    connection.commit()
    connection.close()

    selected = harvest(database, changed=["pkg/mod.py"], depth=1, emit="nodeids")

    assert "stranger_test.py::test_s" not in selected
    assert "direct_test.py::test_d" in selected
    assert "package_test.py::test_p" in selected


@pytest.mark.cli_python_subprocess
def test_every_emitted_nodeid_is_collectable_by_pytest(tmp_path: Path) -> None:
    """The harvester's output is pytest's input; only pytest defines validity.

    Marked rather than shrunk: the spawned pytest IS the assertion. Only pytest
    knows its own collection rules, so asserting against ``harvest``'s return
    value instead would compare the harvester to a hand-written restatement of
    those rules -- which is what let a fixture and the code agree on
    ``Suite::test_method``, a nodeid pytest rejects. Measured 1.29s on a 64-core host,
    all of it the child interpreter's own startup and collection.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    nodeids = harvest(database, changed=["lib.py"], depth=1, emit="nodeids")
    completed = _collects(tmp_path, nodeids)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    # Exit 0 alone is not acceptance: pytest tolerates an explicit path outside
    # `python_files` by collecting nothing from it. Every selector must yield a
    # test, or the count and the request disagree.
    assert f"{len(nodeids)} tests collected" in completed.stdout, completed.stdout


def test_a_collectable_test_without_an_underscore_is_emitted(tmp_path: Path) -> None:
    """`python_functions` defaults to `test*`; 25 such tests exist in this repo."""
    database = _index(tmp_path)
    _tree(tmp_path)
    assert "lib_test.py::testlegacy" in harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
    )


def test_the_fallback_is_per_file_not_global(tmp_path: Path) -> None:
    """CG-FALLBACK-001: one indexed file suppressed every other file's sidecar."""
    database = _index(tmp_path)
    _tree(tmp_path)
    (tmp_path / "new.py").write_text("def brand(): pass\n", encoding="utf-8")
    (tmp_path / "new_test.py").write_text("def test_brand(): pass\n", encoding="utf-8")
    assert "new_test.py" in harvest(
        database,
        changed=["lib.py", "new.py"],
        depth=1,
        emit="nodeids",
    )


def test_a_query_failure_mid_read_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CG-READ-RACE-003: a failure AFTER the walk aborted instead of degrading.

    Fails the selector query rather than the connect: one handle serves the walk
    and the class lookups, so the index can be replaced or locked between two
    reads on it -- the window a connect-time-only guard leaves open.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    # Measured: `_reachable` issues 3 `edges` reads for this fixture, then
    # `_selector` issues its class-owner lookups. Denying at 3 lands in the
    # selector phase; denying earlier trips the walk's own guard instead.
    monkeypatch.setattr(
        sqlite3,
        "connect",
        _connect_double(
            [],
            lambda connection: connection.set_authorizer(_deny_edges_after(3)),
        ),
    )
    assert harvest(database, changed=["lib.py"], depth=1, emit="nodeids") == [
        "lib_test.py",
    ]


def test_nodeids_name_the_class_owning_a_test_method(tmp_path: Path) -> None:
    """`Suite` is absent: pytest collects only classes matching `python_classes`."""
    _tree(tmp_path)
    assert harvest(
        _index(tmp_path),
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
    ) == [
        "dyn_test.py::test_dynamic",
        "lib_test.py::TestSuite::test_owned",
        "lib_test.py::test_helper",
        "lib_test.py::testlegacy",
    ]


def test_depth_bounds_the_walk(tmp_path: Path) -> None:
    database = _index(tmp_path)
    assert "far_test.py::test_far" not in harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
    )
    assert "far_test.py::test_far" in harvest(
        database,
        changed=["lib.py"],
        depth=2,
        emit="nodeids",
    )


def test_the_walk_refuses_to_route_through_a_file_node(tmp_path: Path) -> None:
    """A file node's `imports` fan-in reaches every importer of the whole file."""
    assert "hub_test.py::test_hub" not in harvest(
        _index(tmp_path),
        changed=["lib.py"],
        depth=3,
        emit="nodeids",
    )


def test_a_node_reached_at_two_depths_is_emitted_once(tmp_path: Path) -> None:
    """The walk yields one row per depth at which a node was reached."""
    database = _index(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO edges VALUES ('n:fartest', 'n:lib', 'calls')")
    connection.commit()
    connection.close()
    nodeids = harvest(database, changed=["lib.py"], depth=2, emit="nodeids")
    assert nodeids.count("far_test.py::test_far") == 1


def test_files_projection_includes_the_inputs(tmp_path: Path) -> None:
    assert harvest(
        _index(tmp_path),
        changed=["lib.py"],
        depth=1,
        emit="files",
    ) == ["dyn_test.py", "lib.py", "lib_test.py", "mid.py"]


def test_a_missing_index_still_names_the_changed_files(tmp_path: Path) -> None:
    """The type-check consumer must never see fewer files than it was handed."""
    assert harvest(
        tmp_path / "absent.db",
        changed=["lib.py"],
        depth=1,
        emit="files",
    ) == ["lib.py"]


def test_a_missing_index_falls_back_to_the_sidecar(tmp_path: Path) -> None:
    """The paired test is the floor: a stale index must not select nothing."""
    (tmp_path / "lib.py").touch()
    (tmp_path / "lib_test.py").touch()
    assert harvest(
        tmp_path / "absent.db",
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
    ) == ["lib_test.py"]


def test_a_missing_index_and_no_sidecar_selects_nothing(tmp_path: Path) -> None:
    assert not harvest(
        tmp_path / "absent.db",
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
    )


def test_a_populated_closure_supersedes_the_sidecar(tmp_path: Path) -> None:
    """The graph names individual tests; the sidecar would run the whole file."""
    database = _index(tmp_path)
    (tmp_path / "lib.py").touch()
    (tmp_path / "lib_test.py").touch()
    nodeids = harvest(database, changed=["lib.py"], depth=1, emit="nodeids")
    assert "lib_test.py" not in nodeids
    assert "lib_test.py::test_helper" in nodeids


def test_an_unindexed_file_is_still_named_for_type_checking(tmp_path: Path) -> None:
    """A brand-new file has no ``nodes`` row, so it seeds an empty closure."""
    assert harvest(
        _index(tmp_path),
        changed=["brand_new.py"],
        depth=1,
        emit="files",
    ) == ["brand_new.py"]


def test_non_python_inputs_seed_nothing_but_are_not_dropped(tmp_path: Path) -> None:
    assert harvest(
        _index(tmp_path),
        changed=["assets/index.html"],
        depth=1,
        emit="files",
    ) == ["assets/index.html"]


def test_the_sidecar_probe_finds_a_sibling_tests_directory(tmp_path: Path) -> None:
    """A `tools/` package may keep its tests in `tools/tests/` (house style)."""
    (tmp_path / "pkg" / "tests").mkdir(parents=True)
    (tmp_path / "pkg" / "thing.py").touch()
    (tmp_path / "pkg" / "tests" / "thing_test.py").touch()
    assert harvest(
        tmp_path / "absent.db",
        changed=["pkg/thing.py"],
        depth=1,
        emit="nodeids",
    ) == ["pkg/tests/thing_test.py"]


def test_workers_do_not_share_one_seeds_line_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each worker needs its OWN handle: `touched` is a per-seed temp table.

    `_touched_table` DROPs and recreates it for every seed, so two seeds sharing
    one connection both read whichever hunks landed last. The barrier forces
    that overlap rather than waiting for a race to show up -- both workers write
    their ranges before either queries -- and the two seeds are chosen so BOTH
    orderings are visibly wrong, not just the unlucky one:

    - `lib.py` at 1-9 selects `helper`'s dependents; at `mid.py`'s 21-29 it
      matches no symbol and falls to the sidecar.
    - `mid.py` at 21-29 selects `far_test.py`; at `lib.py`'s 1-9 it matches
      nothing and, having no sidecar on disk, selects nothing at all.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    barrier = threading.Barrier(2, timeout=10)
    real = affected._touched_table

    def synchronized(
        connection: sqlite3.Connection,
        lines: Sequence[tuple[int, int]],
    ) -> None:
        real(connection, lines)
        barrier.wait()

    monkeypatch.setattr(affected, "_touched_table", synchronized)
    assert harvest(
        database,
        changed=["lib.py", "mid.py"],
        depth=1,
        emit="nodeids",
        workers=2,
        hunks={"lib.py": [(1, 9)], "mid.py": [(21, 29)]},
    ) == [
        "dyn_test.py::test_dynamic",
        "far_test.py::test_far",
        "lib_test.py::TestSuite::test_owned",
        "lib_test.py::test_helper",
        "lib_test.py::testlegacy",
    ]


def test_the_command_line_never_runs_a_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Changed lines and the file universe come from the caller and the tree, not git."""
    database = _index(tmp_path)
    _tree(tmp_path)
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=AssertionError("spawned")))
    monkeypatch.setattr(
        subprocess,
        "Popen",
        Mock(side_effect=AssertionError("spawned")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["affected.py", "--database", str(database), "--", "lib.py"],
    )
    assert affected.main() == 0
    assert "lib_test.py::test_helper" in capfd.readouterr().out.split()


def test_a_worker_exception_still_closes_every_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising worker must not leak the handles its siblings opened.

    `pool.map` re-raises on iteration, so the `for` loop in `harvest` aborts
    mid-walk. Without the `finally`, every connection opened before the raise
    stays open until the interpreter collects it -- and a pre-commit hook that
    leaks a read handle per run keeps the index file alive against the sync
    that wants to replace it.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    handles: list[sqlite3.Connection] = []
    monkeypatch.setattr(sqlite3, "connect", _connect_double([], handles.append))
    real = affected._reachable
    seen: list[str] = []

    def exploding(
        connection: sqlite3.Connection | None,
        *,
        seed: str,
        depth: int,
        lines: Sequence[tuple[int, int]] = (),
    ) -> list[tuple[str, str, str, str]]:
        seen.append(seed)
        if len(seen) == 2:
            message = "worker failed mid-walk"
            raise RuntimeError(message)
        return real(connection, seed=seed, depth=depth, lines=lines)

    monkeypatch.setattr(affected, "_reachable", exploding)
    with pytest.raises(RuntimeError, match="worker failed mid-walk"):
        harvest(database, changed=["lib.py", "mid.py"], depth=1, emit="nodeids")
    assert handles, "the walk never opened a handle, so nothing was under test"
    # A CLOSED connection raises on any use; a leaked one answers. That is the
    # only way to observe closure -- `sqlite3` exposes no `.closed`.
    for handle in handles:
        with pytest.raises(sqlite3.ProgrammingError):
            handle.execute("SELECT 1")


def test_the_pool_opens_one_connection_per_thread_not_per_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-seed opens cost N+1 on an N-file changeset, and rebuild `modules` N times.

    Serial must open exactly one handle however many seeds it walks; that is
    what `_Connections` caches on the thread rather than on the call.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    opened: list[str] = []
    monkeypatch.setattr(sqlite3, "connect", _connect_double(opened))
    harvest(database, changed=["lib.py", "mid.py", "twin.py"], depth=1, emit="nodeids")
    # The derived-table build opens its own writable file; count index opens.
    assert sum(entry.endswith("codegraph.db?mode=ro") for entry in opened) == 1


def test_an_unusable_index_is_not_reopened_once_per_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_connect` returning None is a cached ANSWER, not a miss to retry.

    A truthiness check on the thread-local would re-run the failing open for
    every seed, paying a locked or schema-changed index N times before falling
    back N times.

    The index must EXIST for this to be reachable -- `harvest` skips the pool
    entirely when it does not -- so the failure is injected at `_connect`,
    which is also where the real ones (a held write lock, a schema CodeGraph
    changed under us) surface.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    attempts: list[Path] = []

    def unusable(database: Path) -> sqlite3.Connection | None:
        attempts.append(database)
        return None

    monkeypatch.setattr(affected, "_connect", unusable)
    assert harvest(
        database,
        changed=["lib.py", "mid.py", "twin.py"],
        depth=1,
        emit="nodeids",
    ) == ["lib_test.py"]
    assert len(attempts) == 1


def test_more_workers_than_seeds_is_clamped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-file commit is the common case; it must not size a pool of 64.

    Asserts the POOL SIZE, not the output: the clamp cannot change which
    selectors come back, only how many threads are built to find them, so an
    equality check on `harvest`'s return value passes with the clamp deleted.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    sizes: list[int | None] = []

    # `concurrent.futures.ThreadPoolExecutor`, not `affected.ThreadPoolExecutor`:
    # the module re-exports nothing, so reading the name off it is a private
    # import. The monkeypatch still targets `affected`, which is where the
    # binding the code under test resolves lives.
    def recorded(max_workers: int | None = None) -> ThreadPoolExecutor:
        sizes.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(affected, "ThreadPoolExecutor", recorded)
    harvest(database, changed=["lib.py"], depth=1, emit="nodeids", workers=64)
    assert sizes == [1]


def test_the_worker_count_reaches_harvest_from_the_command_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--workers` is how `$WORKERS` reaches the pool; the hooks pass it through.

    Asserts the ARGUMENT, not the output: worker count cannot change which
    selectors come back, so an end-to-end run through the CLI stays green with
    `main` dropping the flag entirely. Only the call site can witness it.
    """
    received: list[int] = []

    def recorded(database: Path, **kwargs: object) -> list[str]:
        del database
        # `.get`, not `[...]`: a `main` that drops the flag must fail the
        # assertion below naming the count, not raise KeyError from the double.
        received.append(cast(int, kwargs.get("workers", 0)))
        return []

    monkeypatch.setattr(affected, "harvest", recorded)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "affected.py",
            "--workers",
            "4",
            "--database",
            str(tmp_path / "absent.db"),
            "--",
            "lib.py",
        ],
    )
    assert affected.main() == 0
    assert received == [4]


@pytest.mark.cli_python_subprocess
def test_an_empty_worker_count_is_serial_not_an_argparse_error(
    tmp_path: Path,
) -> None:
    """An empty ``--workers`` must degrade to serial, never take the harvester down.

    A shell interpolating an unset count passes `--workers ''`. `type=int`
    rejects that with exit 2, and the gate reads a nonzero exit as "no
    selection" and silently drops to the sidecar probe, which is the downgrade
    the graph exists to avoid.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    completed = _run_cli(
        "--emit",
        "nodeids",
        "--workers",
        "",
        "--database",
        str(database),
        "--",
        "lib.py",
    )
    assert completed.returncode == 0, completed.stderr
    assert "lib_test.py::test_helper" in completed.stdout.split()


@pytest.mark.cli_python_subprocess
def test_an_unparsable_worker_count_is_serial_not_an_argparse_error(
    tmp_path: Path,
) -> None:
    """The whole point of `worker_count` is that exit 2 is unreachable.

    Companion to the empty-string test above, at the same CLI boundary and for
    the same reason -- that one only proves the ONE unusable spelling the
    `str.isdigit` guard already handled. A value the guard admits and `int()`
    rejects raises inside the type callable, which argparse reports as exit 2,
    which the hook shell reads as "no selection". Asserted through the real
    argv because the exit code is the observable the hook acts on.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    completed = _run_cli(
        "--emit",
        "nodeids",
        "--workers=\N{SUPERSCRIPT TWO}",
        "--database",
        str(database),
        "--",
        "lib.py",
    )
    assert completed.returncode == 0, completed.stderr
    assert "lib_test.py::test_helper" in completed.stdout.split()


@pytest.mark.cli_python_subprocess
def test_a_real_walk_runs_through_the_command_line(tmp_path: Path) -> None:
    """End-to-end smoke: the pool path survives a real index under a real argv.

    Points at a POPULATED index so the executor is actually constructed. An
    earlier version used an absent database, where `harvest` short-circuits
    before building the pool, so the subprocess never exercised it.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    completed = _run_cli(
        "--emit",
        "nodeids",
        "--workers",
        "4",
        "--database",
        str(database),
        "--",
        "lib.py",
        "mid.py",
    )
    assert completed.returncode == 0, completed.stderr
    assert "lib_test.py::test_helper" in completed.stdout.split()


def test_a_saturated_changeset_returns_the_inputs_without_walking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--all-files` hands in every file, so the closure is already the input.

    A reverse-dependency closure cannot name a file outside the tree it is
    drawn from. When `changed` covers that tree the answer is knowable without
    the walk, which cost 26.6s on this repository to return the 4220 paths it
    was given.

    Asserts the OPEN COUNT beside the value: a `harvest` that walks the whole
    graph and then unions `changed` back in returns the identical list, having
    paid the full cost, so the output alone cannot witness the short circuit.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    opened: list[str] = []
    monkeypatch.setattr(sqlite3, "connect", _connect_double(opened))
    assert harvest(
        database,
        changed=["lib.py", "mid.py"],
        depth=1,
        emit="files",
        saturated=True,
    ) == ["lib.py", "mid.py"]
    assert opened == []


def test_an_unsaturated_changeset_still_walks(tmp_path: Path) -> None:
    """The guard is COVERAGE, not size -- a big changeset still gets its closure.

    This is the case a seed-count ceiling got wrong: cutting off above 500 seeds
    dropped every dependent outside the input, and at 600 seeds the walk reaches
    1443 files beyond it (240%). Each is a file basedpyright would then skip, so
    the cutoff traded coverage for time silently.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    assert "mid.py" in harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="files",
        saturated=False,
    )


def test_a_saturated_changeset_still_selects_nodeids(tmp_path: Path) -> None:
    """Saturation bounds the FILE set; it says nothing about which tests to run.

    Returning empty here would hand the hook shell no selection, and its
    fallback names whole test files -- the scope nodeid selection exists to
    beat. The walk is worth its 48.3s against a suite that takes minutes.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    assert "lib_test.py::test_helper" in harvest(
        database,
        changed=["lib.py", "mid.py"],
        depth=1,
        emit="nodeids",
        saturated=True,
    )


def test_an_empty_tree_is_never_covered(tmp_path: Path) -> None:
    """No Python files must not assert coverage: every input covers the empty set."""
    (tmp_path / "notes.txt").touch()
    assert not affected._covers_universe([])


def test_coverage_counts_every_python_file_in_the_tree(tmp_path: Path) -> None:
    """The cwd's `.py` and `.pyi` files, minus hidden and `.gitignore`d trees."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("build/\n")
    for name in ("a.py", "pkg/b.pyi", "pkg/c.txt", "build/d.py", ".venv/e.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    assert affected._covers_universe(["a.py", "pkg/b.pyi"])
    assert affected._covers_universe(["./a.py", "pkg/b.pyi", "extra.py"])
    assert not affected._covers_universe(["a.py"])


@pytest.mark.parametrize(
    ("argv", "covered", "want"),
    [
        (["--emit", "files"], True, True),
        (["--emit", "files"], False, False),
        (["--emit", "files", "--no-universe"], True, False),
        (["--emit", "nodeids"], True, False),
    ],
)
def test_only_the_files_projection_checks_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    covered: bool,
    want: bool,
) -> None:
    """Asserts the ARGUMENT: no fixture hands in every tracked file.

    `nodeids` never reads it, and `--no-universe` is the escape hatch, so
    neither may pay for the walk.
    """
    received: list[object] = []
    checked: list[list[str]] = []

    def recorded(database: Path, **kwargs: object) -> list[str]:
        del database
        received.append(kwargs["saturated"])
        return []

    def covers(changed: list[str]) -> bool:
        checked.append(changed)
        return covered

    monkeypatch.setattr(affected, "harvest", recorded)
    monkeypatch.setattr(affected, "_covers_universe", covers)
    monkeypatch.setattr(
        sys,
        "argv",
        ["affected.py", *argv, "--database", str(tmp_path / "absent.db"), "--", "a.py"],
    )
    assert affected.main() == 0
    assert received == [want]
    assert checked == ([["a.py"]] if argv == ["--emit", "files"] else [])


def test_skip_reads_one_name_out_of_precommits_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKIP", "ruff-check, harvest-affected ,pytest")
    assert _skipped("harvest-affected")
    assert not _skipped("codegraph-sync")


def test_skip_is_absent_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKIP", raising=False)
    assert not _skipped("harvest-affected")


def test_depth_zero_ignores_the_index(tmp_path: Path) -> None:
    """What the `harvest-affected` skip token resolves to: seeds and floors only."""
    database = _index(tmp_path)
    (tmp_path / "lib.py").touch()
    (tmp_path / "lib_test.py").touch()
    assert harvest(database, changed=["lib.py"], depth=0, emit="nodeids") == [
        "lib_test.py",
    ]
    assert harvest(database, changed=["lib.py"], depth=0, emit="files") == ["lib.py"]


def test_depth_zero_on_a_changed_test_file_still_falls_to_the_sidecar(
    tmp_path: Path,
) -> None:
    """CG-SKIP-001: the seed rows are themselves tests, so depth 0 leaked them."""
    database = _index(tmp_path)
    _tree(tmp_path)
    assert harvest(database, changed=["lib_test.py"], depth=0, emit="nodeids") == [
        "lib_test.py",
    ]


@pytest.mark.parametrize("depth", [0, -1, -3])
def test_a_non_positive_depth_reaches_no_edge(tmp_path: Path, depth: int) -> None:
    """CG-SKIP-001 again, one sign over: `depth and seeded` is truthiness.

    Only `0` is falsy, so a NEGATIVE depth walked the graph -- and `back.depth <
    :depth` is then false at every hop, so the CTE returned exactly its own
    seed rows. For a changed TEST file those seeds ARE tests, which is the
    stale-index-dressed-as-a-selection the depth-0 guard was written to stop.
    Same shape as the empty pathspec in `_hunks` and the blank glob in
    `_globs`: a degenerate input silently inverting the intent.

    Asserted on the SIDECAR rather than on emptiness -- the floor is what the
    guard promises, and an empty list would also pass a `harvest` that walked
    and found nothing.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    assert harvest(database, changed=["lib_test.py"], depth=depth, emit="nodeids") == [
        "lib_test.py",
    ]


def test_a_negative_depth_opens_no_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is "do not consult the index", not "discard what it said".

    Companion to the parametrized test above, which the selection alone cannot
    distinguish: a `harvest` that opens the index, walks it, and happens to
    return the seed rows would satisfy it on some fixtures. What depth 0 buys
    is that `_hunks` never shells out to `git diff` and no handle is opened,
    and only the open count witnesses that.
    """
    database = _index(tmp_path)
    _tree(tmp_path)
    opened: list[str] = []
    monkeypatch.setattr(sqlite3, "connect", _connect_double(opened))
    harvest(database, changed=["lib.py"], depth=-1, emit="nodeids")
    assert opened == []


@pytest.mark.parametrize("directory", ["has?query", "has#frag", "has space", "has%pct"])
def test_a_punctuated_path_still_opens_the_index_read_only(
    tmp_path: Path,
    directory: str,
) -> None:
    """`f"file:{database}?mode=ro"` interpolates a PATH into a URI unescaped.

    `?` and `#` are URI delimiters, so a checkout under one truncates the
    filename there and drops `mode=ro` with it. SQLite then CREATES the
    truncated name -- writable, empty, and not the index -- so the walk finds
    nothing, every seed falls to its sidecar, and a stray file is left in the
    tree. The read-only promise `_connect` documents is the one the writer
    (`codegraph sync`) relies on.

    Asserted on WRITABILITY, not on the selection: a corrupt-index fallback
    already returns the sidecar, so output alone cannot tell "opened read-only
    and found nothing" from "opened the wrong file entirely".
    """
    root = tmp_path / directory
    root.mkdir()
    database = _index(root)
    connection = affected._connect(database)
    assert connection is not None
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE wrote_to_the_index (x)")
    finally:
        connection.close()
    # A truncated URI does not fail -- it opens a DIFFERENT file and creates it.
    assert sorted(path.name for path in tmp_path.iterdir()) == [directory]


def test_an_unreadable_index_falls_back_instead_of_raising(tmp_path: Path) -> None:
    """CG-FAILOPEN-002: a raised SQLite error left the hook selecting nothing."""
    database = tmp_path / "corrupt.db"
    database.write_text("not a database", encoding="utf-8")
    _tree(tmp_path)
    assert harvest(database, changed=["lib.py"], depth=1, emit="nodeids") == [
        "lib_test.py",
    ]
    assert harvest(database, changed=["lib.py"], depth=1, emit="files") == ["lib.py"]


def test_a_stub_seeds_its_dependents(tmp_path: Path) -> None:
    """CG-PYI-009: `types_or` accepts `.pyi`, so the closure must too."""
    database = tmp_path / "codegraph.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT,"
        " file_path TEXT, qualified_name TEXT, start_line INTEGER, end_line INTEGER);"
        "CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);"
        "CREATE TABLE unresolved_refs (from_node_id TEXT, reference_name TEXT,"
        " reference_kind TEXT, file_path TEXT, name_tail TEXT);",
    )
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:api", "function", "declared", "api.pyi", "declared", 171, 179),
            ("n:caller", "function", "uses", "caller.py", "uses", 181, 189),
        ],
    )
    connection.execute("INSERT INTO edges VALUES ('n:caller', 'n:api', 'calls')")
    connection.commit()
    connection.close()
    assert harvest(database, changed=["api.pyi"], depth=1, emit="files") == [
        "api.pyi",
        "caller.py",
    ]


def test_a_blank_collection_setting_falls_back_to_the_default(tmp_path: Path) -> None:
    """An empty pattern list matches NOTHING, so it cannot mean "unset".

    Same shape as the empty-pathspec bug in `_hunks`: a degenerate input yields
    a degenerate collection that silently means the opposite of the intent.
    `any(fnmatch(name, p) for p in [])` is False for every name, so a config
    stating `python_files =` with no value selected zero tests rather than
    falling back to pytest's own default.
    """
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\npython_files =\npython_functions =   \n",
        encoding="utf-8",
    )
    functions, _, files = _collection_patterns(tmp_path)
    assert files == ["test_*.py", "*_test.py"]
    assert functions == ["test*"]


@pytest.mark.cli_python_subprocess
def test_an_option_shaped_path_is_a_path_not_a_flag(tmp_path: Path) -> None:
    """CG-ARGV-012: argparse consumed a leading-dash filename as an option."""
    completed = _run_cli(
        "--emit",
        "files",
        "--database",
        str(tmp_path / "absent.db"),
        "--",
        "-broken.py",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["-broken.py"]


def test_main_prints_only_the_selection(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Stdout carries the selection; stderr stays silent."""

    def harvest_stub(*args: object, **kwargs: object) -> list[str]:
        del args, kwargs
        return ["x.py"]

    _recorded_harvest(monkeypatch)
    monkeypatch.setattr(affected, "harvest", harvest_stub)
    monkeypatch.setattr(sys, "argv", ["affected", "--", "x.py"])
    assert affected.main() == 0
    assert capfd.readouterr() == ("x.py\n", "")


def test_connect_returns_none_when_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An SQLite open error degrades to the sidecar path."""
    database = tmp_path / "index.db"
    database.touch()

    def fail(*args: object, **kwargs: object) -> sqlite3.Connection:
        del args, kwargs
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(sqlite3, "connect", fail)
    assert affected._connect(database) is None


def test_selectors_returns_empty_on_sql_error() -> None:
    """A selector query failure does not abort the hook."""
    connection = sqlite3.connect(":memory:")
    try:
        assert (
            affected._selectors(
                connection,
                rows=[("id", "method", "test", "x_test.py")],
                patterns=(["test*"], ["Test*"], ["*_test.py"]),
            )
            == set()
        )
    finally:
        connection.close()


def test_a_unified_diff_parses_to_new_side_ranges(tmp_path: Path) -> None:
    """Concatenated diffs union; a pure deletion marks the line it was cut from."""
    diff = tmp_path / "change.diff"
    diff.write_text(
        "@@ -1 +1 @@ before any file\n"
        "+++ b/a.py\n@@ -1 +2,3 @@\n"
        "+++ b/a.py\n@@ -5,2 +8,0 @@\n"
        "+++ sp ace.py\t2026-10-01 00:00:00\n@@ -1 +1 @@ ctx\n"
        "+++ /dev/null\n@@ -1,4 +0,0 @@\n",
    )
    assert affected._parse_diff(diff) == {
        "a.py": [(2, 4), (8, 8)],
        "sp ace.py": [(1, 1)],
    }


def test_an_empty_diff_has_no_hunks(tmp_path: Path) -> None:
    (tmp_path / "empty.diff").touch()

    assert affected._parse_diff(tmp_path / "empty.diff") == {}


def test_the_diff_reaches_harvest_from_the_command_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--diff PATH` is the only source of changed lines; `/dev/stdin` pipes one."""
    received: list[object] = []

    def recorded(database: Path, **kwargs: object) -> list[str]:
        del database
        received.append(kwargs.get("hunks", "absent"))
        return []

    (tmp_path / "change.diff").write_text("+++ b/lib.py\n@@ -1 +2,3 @@\n")
    monkeypatch.setattr(affected, "harvest", recorded)
    monkeypatch.setattr(
        sys,
        "argv",
        ["affected.py", "--diff", str(tmp_path / "change.diff"), "--", "lib.py"],
    )
    assert affected.main() == 0
    assert received == [{"lib.py": [(2, 4)]}]


def test_without_a_diff_every_file_is_seeded_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[object] = []

    def recorded(database: Path, **kwargs: object) -> list[str]:
        del database
        received.append(kwargs.get("hunks", "absent"))
        return []

    monkeypatch.setattr(affected, "harvest", recorded)
    monkeypatch.setattr(sys, "argv", ["affected.py", "--", "lib.py"])
    assert affected.main() == 0
    assert received == [{}]


def test_module_and_boundary_handle_relative_and_path_imports() -> None:
    """Relative Python names strip dots; slash imports match nothing."""
    assert affected._module("..pkg") == "pkg"
    assert affected._module("./pkg") == ""
    assert affected._boundary("../pkg") == ""


def test_connect_returns_none_for_missing_database(tmp_path: Path) -> None:
    """Missing indexes are optional inputs."""
    assert affected._connect(tmp_path / "missing.db") is None


def test_collection_patterns_read_pyproject_lists(tmp_path: Path) -> None:
    """Pyproject list settings override pytest defaults."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npython_functions = ["check*"]\n'
        'python_classes = ["Suite"]\npython_files = ["*_spec.py"]\n',
    )
    assert _collection_patterns(tmp_path) == (["check*"], ["Suite"], ["*_spec.py"])


@pytest.mark.parametrize(
    "body",
    ["tool = 1\n", "[tool]\npytest = 1\n", "[tool.pytest]\nini_options = 1\n"],
)
def test_a_non_table_pytest_setting_keeps_the_defaults(
    tmp_path: Path,
    body: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text(body)
    assert _collection_patterns(tmp_path) == (
        ["test*"],
        ["Test*"],
        ["test_*.py", "*_test.py"],
    )


def _prefix_tree(tmp_path: Path) -> list[str]:
    """Index a tree whose ini collects by word prefix; return the harvested ids."""
    database = _index(tmp_path)
    _tree(tmp_path)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\npython_files = *_test.py\n"
        "python_functions = test fixture\npython_classes = Sui\n",
    )
    with (tmp_path / "lib_test.py").open("a") as handle:
        handle.write("\n\ndef fixture_helper(): pass\n")
    return harvest(database, changed=["lib.py"], depth=1, emit="nodeids")


def test_a_plain_word_pattern_matches_as_a_prefix_like_pytest(tmp_path: Path) -> None:
    """Pytest matches each plain word as a prefix as well as a glob.

    ``_pytest/python.py`` ``_matches_prefix_or_glob_option``: ``fixture``
    collects ``fixture_helper`` and ``Sui`` collects ``Suite``.
    """
    assert {"lib_test.py::fixture_helper", "lib_test.py::Suite::test_method"} <= set(
        _prefix_tree(tmp_path),
    )


@pytest.mark.cli_python_subprocess
def test_pytest_collects_every_prefix_matched_nodeid(tmp_path: Path) -> None:
    """Only pytest defines its prefix rule, so a real one must collect each id."""
    nodeids = _prefix_tree(tmp_path)
    completed = _collects(tmp_path, nodeids)
    assert f"{len(nodeids)} tests collected" in completed.stdout, completed.stdout


def test_globs_accepts_lists_and_falls_back_for_other_values() -> None:
    """The normalizer supports TOML lists and safe defaults."""
    assert affected._globs(["a", 3], default=["x"]) == ["a", "3"]
    assert affected._globs(None, default=["x"]) == ["x"]


def test_selector_method_without_class_owner_uses_file_selector(
    tmp_path: Path,
) -> None:
    """A method without a class edge remains directly selectable."""
    database = _graph(tmp_path / "index.db", nodes=[])
    derived = affected.ensure_derived(database)
    assert derived is not None
    connection = sqlite3.connect(database)
    connection.execute("ATTACH DATABASE ? AS derived", (str(derived),))
    affected._lay_over(connection, None)
    try:
        assert (
            affected._selector(
                connection,
                "m",
                kind="method",
                name="test_m",
                file_path="x.py",
                classes=["Test*"],
            )
            == "x.py::test_m"
        )
    finally:
        connection.close()


def test_skip_splits_on_commas_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKIP", "ruff-check,harvest-affected")

    assert _skipped("harvest-affected")
    assert not _skipped("ruff-check,harvest-affected")


@pytest.mark.parametrize("token", ["", "harvest-affected"])
def test_an_absent_skip_names_no_token(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    monkeypatch.delenv("SKIP", raising=False)

    assert not _skipped(token)


def _recorded_harvest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    overlay: Path | None = None,
) -> list[dict[str, object]]:
    """Replace ``harvest`` with a recorder; the graph reads as current, with ``overlay``."""
    calls: list[dict[str, object]] = []

    def recorded(database: Path, **kwargs: object) -> list[str]:
        calls.append({"database": database, **kwargs})
        return []

    def current(caches: Path, **_: object) -> tuple[str, Path | None]:
        del caches
        return ("", overlay)

    monkeypatch.setattr(affected, "harvest", recorded)
    monkeypatch.setattr(backbone, "overlay_problem", current)
    return calls


def test_the_skip_token_forces_depth_zero_and_keeps_the_other_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _recorded_harvest(monkeypatch)
    monkeypatch.setenv("SKIP", "mine")
    monkeypatch.setattr(
        sys,
        "argv",
        ["affected.py", "--skip-token", "mine", "--depth", "3", "--emit", "files"],
    )

    assert affected.main() == 0
    assert (calls[0]["depth"], calls[0]["emit"]) == (0, "files")


def test_an_unskipped_run_walks_the_requested_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _recorded_harvest(monkeypatch)
    monkeypatch.setenv("SKIP", "harvest-affected")
    monkeypatch.setattr(
        sys,
        "argv",
        ["affected.py", "--skip-token", "x", "--depth", "3"],
    )

    assert affected.main() == 0
    assert calls[0]["depth"] == 3


def test_flag_defaults_and_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hooks pass every flag; a bare run uses these."""
    monkeypatch.setattr(sys, "argv", ["affected.py", "--depth", "9", "ignored.py"])
    flags = affected._parse_args(argparse.ArgumentParser(), argv=[])

    assert vars(flags) == {
        "files": [],
        "diff": None,
        "emit": "nodeids",
        "caches": "/opt/scratch/caches/codegraph",
        "database": "",
        "depth": 1,
        "skip_token": "harvest-affected",
        "workers": 1,
        "no_universe": False,
    }
    assert (
        affected._parse_args(argparse.ArgumentParser(), argv=["--depth", "2"]).depth
        == 2
    )


@pytest.mark.parametrize("emit", ["nodeids", "files"])
def test_emit_takes_either_projection(emit: str) -> None:
    flags = affected._parse_args(argparse.ArgumentParser(), argv=["--emit", emit])

    assert flags.emit == emit


def test_emit_rejects_anything_else() -> None:
    with pytest.raises(SystemExit):
        affected._parse_args(argparse.ArgumentParser(), argv=["--emit", "tests"])


def test_help_documents_every_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("COLUMNS", "10000")
    monkeypatch.setattr(sys, "argv", ["affected.py", "--help"])
    with pytest.raises(SystemExit):
        affected.main()
    text = " ".join(capsys.readouterr().out.split())

    for expected in (
        "Reads the shared backbone with this worktree's overlay laid over it,",
        "python -m codedoctor.affected --diff /dev/stdin lib.py",
        "files Changed repository-relative paths.",
        "--diff PATH Unified diff naming the changed lines, e.g. /dev/stdin.",
        "--emit {nodeids,files} Projection of the closure.",
    ):
        assert expected in text
    assert "exec uv" not in text


@pytest.mark.parametrize(
    ("path", "dotted"),
    [("a/b.py", "a.b"), ("a/b.pyi", "a.b"), ("a/b/__init__.py", "a.b"), ("c.py", "c")],
)
def test_a_path_names_its_dotted_module(path: str, dotted: str) -> None:
    assert affected._dotted(path) == dotted


@pytest.mark.parametrize(
    ("name", "module", "boundary"),
    [
        ("./x", "", ""),
        ("../x", "", ""),
        (".Xa", "Xa", "%.Xa"),
        ("a\\b_c%d", "a\\b_c%d", "%.a\\\\b\\_c\\%d"),
    ],
)
def test_module_and_boundary_of_an_import(
    name: str,
    module: str,
    boundary: str,
) -> None:
    assert (affected._module(name), affected._boundary(name)) == (module, boundary)


def test_pytest_ini_settings_win_over_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npython_functions = ["from_toml*"]\n',
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\npython_functions = check* verify*\n",
    )

    assert _collection_patterns(tmp_path)[0] == ["check*", "verify*"]


def test_a_pytest_ini_without_the_section_defers_to_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npython_functions = ["from_toml*"]\n',
    )
    (tmp_path / "pytest.ini").write_text("[other]\npython_functions = check*\n")

    assert _collection_patterns(tmp_path)[0] == ["from_toml*"]


def test_a_pyproject_without_pytest_settings_keeps_the_defaults(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    assert _collection_patterns(tmp_path) == (
        ["test*"],
        ["Test*"],
        ["test_*.py", "*_test.py"],
    )


def test_globs_split_a_word_list_and_ignore_other_values() -> None:
    assert affected._globs(" a  b ", default=["x"]) == ["a", "b"]
    assert affected._globs(3, default=["x"]) == ["x"]


def test_a_module_row_names_its_package_not_its_root() -> None:
    assert affected._module_rows([("a/b/c.py",), ("top.py",)]) == [
        ("a/b/c.py", "a.b.c", "a.b"),
        ("top.py", "top", ""),
    ]


def test_one_worker_walks_every_seed_on_one_attached_handle(tmp_path: Path) -> None:
    """Attaching twice raises `database derived is already in use`.

    Swallowed as a read failure, that dropped every seed after a worker's first
    to its sidecar.
    """
    database = _graph(
        tmp_path / "backbone.db",
        nodes=[
            ("n:a", "function", "a", "a.py", 1),
            ("n:b", "function", "b", "b.py", 1),
            ("n:ta", "function", "test_a", "t_test.py", 1),
            ("n:tb", "function", "test_b", "t_test.py", 11),
        ],
        edges=[("n:ta", "n:a", "calls"), ("n:tb", "n:b", "calls")],
    )

    assert harvest(
        database,
        changed=["a.py", "b.py"],
        depth=1,
        emit="nodeids",
        workers=1,
    ) == ["t_test.py::test_a", "t_test.py::test_b"]


def test_a_stub_dependent_is_a_file_to_check(tmp_path: Path) -> None:
    """A dependent ``.pyi`` is type-checked like a ``.py``; other files are not."""
    database = tmp_path / "codegraph.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT,"
        " file_path TEXT, qualified_name TEXT, start_line INTEGER, end_line INTEGER);"
        "CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);"
        "CREATE TABLE unresolved_refs (from_node_id TEXT, reference_name TEXT,"
        " reference_kind TEXT, file_path TEXT, name_tail TEXT);",
    )
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("n:api", "function", "declared", "api.py", "declared", 1, 9),
            ("n:stub", "function", "uses", "dep.pyi", "uses", 11, 19),
            ("n:js", "function", "web", "app.js", "web", 21, 29),
        ],
    )
    connection.executemany(
        "INSERT INTO edges VALUES (?, ?, ?)",
        [("n:stub", "n:api", "calls"), ("n:js", "n:api", "calls")],
    )
    connection.commit()
    connection.close()

    assert harvest(database, changed=["api.py"], depth=1, emit="files") == [
        "api.py",
        "dep.pyi",
    ]


@pytest.mark.parametrize(
    ("changed", "sidecar"),
    [
        ("pkg/test_thing.py", "pkg/thing_test.py"),
        ("pkg/thing.py", "pkg/tests/test_thing.py"),
        ("pkg/thing_test.py", "pkg/test_thing.py"),
    ],
)
def test_every_sidecar_spelling_pairs_with_its_source(
    tmp_path: Path,
    changed: str,
    sidecar: str,
) -> None:
    (tmp_path / sidecar).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / sidecar).touch()

    assert affected._sidecars([changed]) == [Path(sidecar)]


# `(id, kind, name, file_path, start_line)`; `qualified_name` is the name, and every
# symbol spans five lines. Ids differ between the two indexes on purpose: CodeGraph
# hashes the LINE into each id, so an edited file's symbols come back renamed.
type _Node = tuple[str, str, str, str, int]


def _graph(
    database: Path,
    *,
    nodes: Sequence[_Node],
    edges: Sequence[tuple[str, str, str]] = (),
    refs: Sequence[tuple[str, str, str]] = (),
) -> Path:
    """Write a CodeGraph-shaped index; ``refs`` are ``(caller, name_tail, file)`` calls."""
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT,"
        " file_path TEXT, qualified_name TEXT, start_line INTEGER, end_line INTEGER);"
        "CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);"
        "CREATE TABLE unresolved_refs (from_node_id TEXT, reference_name TEXT,"
        " reference_kind TEXT, file_path TEXT, name_tail TEXT);",
    )
    connection.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (i, kind, name, path, name, line, line + 4)
            for i, kind, name, path, line in nodes
        ],
    )
    connection.executemany("INSERT INTO edges VALUES (?, ?, ?)", edges)
    connection.executemany(
        "INSERT INTO unresolved_refs VALUES (?, ?, 'calls', ?, ?)",
        [(caller, tail, path, tail) for caller, tail, path in refs],
    )
    connection.commit()
    connection.close()
    return database


def _overlay(
    tmp_path: Path,
    *,
    changed: Sequence[str],
    nodes: Sequence[_Node] = (),
    edges: Sequence[tuple[str, str, str]] = (),
    refs: Sequence[tuple[str, str, str]] = (),
) -> Path:
    """Write an overlay as `codegraph_sync` publishes one: indexed, then stamped."""
    overlay = _graph(tmp_path / "overlay.db", nodes=nodes, edges=edges, refs=refs)
    backbone.stamp(overlay, sha="a" * 40, fingerprint="f", paths=changed)
    return overlay


def test_a_deleted_test_in_a_changed_file_is_never_selected(tmp_path: Path) -> None:
    """The 2026-11-01 failure: pytest exited 4 on two tests the commit deleted.

    The index still held `reread_test.py` as it was before the commit, so the
    changed file's own seed rows named tests that no longer existed. The overlay
    replaces every row of a changed file, deleted symbols included.
    """
    database = _graph(
        tmp_path / "backbone.db",
        nodes=[
            ("old:helper", "function", "helper", "lib.py", 1),
            ("old:gone", "function", "test_gone", "lib_test.py", 1),
        ],
        edges=[("old:gone", "old:helper", "calls")],
    )
    overlay = _overlay(
        tmp_path,
        changed=["lib.py", "lib_test.py"],
        nodes=[
            ("new:helper", "function", "helper", "lib.py", 3),
            ("new:kept", "function", "test_kept", "lib_test.py", 7),
        ],
        edges=[("new:kept", "new:helper", "calls")],
    )

    for changed in (["lib.py"], ["lib_test.py"]):
        assert harvest(
            database,
            changed=changed,
            depth=1,
            emit="nodeids",
            overlay=overlay,
        ) == ["lib_test.py::test_kept"]
    assert harvest(database, changed=["lib_test.py"], depth=1, emit="nodeids") == [
        "lib_test.py::test_gone",
    ]


def test_a_deleted_file_reaches_nothing(tmp_path: Path) -> None:
    database = _graph(
        tmp_path / "backbone.db",
        nodes=[
            ("old:helper", "function", "helper", "lib.py", 1),
            ("old:gone", "function", "test_gone", "gone_test.py", 1),
            ("old:kept", "function", "test_kept", "kept_test.py", 1),
        ],
        edges=[
            ("old:gone", "old:helper", "calls"),
            ("old:kept", "old:helper", "calls"),
        ],
    )
    overlay = _overlay(
        tmp_path,
        changed=["lib.py", "gone_test.py"],
        nodes=[("new:helper", "function", "helper", "lib.py", 9)],
    )

    assert harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
        overlay=overlay,
    ) == ["kept_test.py::test_kept"]


def test_an_unchanged_caller_still_reaches_an_edited_symbol(tmp_path: Path) -> None:
    """The backbone's edge names the symbol's OLD id; the walk must follow it anyway.

    `b.f` calls `a.g`. Editing `a.py` moves `g`, which re-hashes its id, so the
    backbone edge points at an id the overlay no longer has. Two hops out, the
    changed `c_test.py` calls `b.f` -- a call the overlay, indexed without `b.py`,
    could only record unresolved.
    """
    database = _graph(
        tmp_path / "backbone.db",
        nodes=[
            ("old:g", "function", "g", "a.py", 1),
            ("old:f", "function", "f", "b.py", 1),
            ("old:c", "function", "test_c", "c_test.py", 1),
        ],
        edges=[("old:f", "old:g", "calls")],
    )
    overlay = _overlay(
        tmp_path,
        changed=["a.py", "c_test.py"],
        nodes=[
            ("new:g", "function", "g", "a.py", 21),
            ("new:c", "function", "test_c", "c_test.py", 1),
            ("new:import", "import", "b", "c_test.py", 9),
        ],
        refs=[("new:c", "f", "c_test.py")],
    )

    assert harvest(
        database,
        changed=["a.py"],
        depth=1,
        emit="files",
        overlay=overlay,
    ) == [
        "a.py",
        "b.py",
    ]
    assert "c_test.py::test_c" in harvest(
        database,
        changed=["a.py"],
        depth=2,
        emit="nodeids",
        overlay=overlay,
    )


def test_a_depth_three_walk_crosses_backbone_and_overlay_both_ways(
    tmp_path: Path,
) -> None:
    """The type checkers walk 3 hops: overlay -> backbone -> overlay -> overlay.

    Changed `a.py` defines `g`; unchanged `b.f` calls it (a backbone edge onto
    `g`'s old id). Changed `c.h` calls `b.f`, which its overlay, indexed without
    `b.py`, recorded only as unresolved. Changed `d_test.py` calls `c.h`.
    """
    database = _graph(
        tmp_path / "backbone.db",
        nodes=[
            ("old:g", "function", "g", "a.py", 1),
            ("old:f", "function", "f", "b.py", 1),
            ("old:h", "function", "h", "c.py", 1),
            ("old:d", "function", "test_d", "d_test.py", 1),
        ],
        edges=[("old:f", "old:g", "calls")],
    )
    overlay = _overlay(
        tmp_path,
        changed=["a.py", "c.py", "d_test.py"],
        nodes=[
            ("new:g", "function", "g", "a.py", 31),
            ("new:h", "function", "h", "c.py", 41),
            ("new:import", "import", "b", "c.py", 51),
            ("new:d", "function", "test_d", "d_test.py", 61),
        ],
        edges=[("new:d", "new:h", "calls")],
        refs=[("new:h", "f", "c.py")],
    )

    walked = {
        depth: harvest(
            database,
            changed=["a.py"],
            depth=depth,
            emit="files",
            overlay=overlay,
        )
        for depth in (1, 2, 3)
    }

    assert walked == {
        1: ["a.py", "b.py"],
        2: ["a.py", "b.py", "c.py"],
        3: ["a.py", "b.py", "c.py", "d_test.py"],
    }


def test_an_unchanged_caller_reaches_a_new_symbol_it_could_not_bind(
    tmp_path: Path,
) -> None:
    """A call into a symbol the backbone lacks sits in `unresolved_refs` until it lands."""
    database = _graph(
        tmp_path / "backbone.db",
        nodes=[
            ("old:g", "function", "g", "a.py", 1),
            ("old:d", "function", "test_d", "d_test.py", 1),
            ("old:import", "import", "a", "d_test.py", 9),
        ],
        refs=[("old:d", "fresh", "d_test.py")],
    )
    overlay = _overlay(
        tmp_path,
        changed=["a.py"],
        nodes=[
            ("old:g", "function", "g", "a.py", 1),
            ("new:fresh", "function", "fresh", "a.py", 11),
        ],
    )

    assert harvest(database, changed=["a.py"], depth=1, emit="nodeids") == []
    assert harvest(
        database,
        changed=["a.py"],
        depth=1,
        emit="nodeids",
        overlay=overlay,
    ) == ["d_test.py::test_d"]


def test_a_deleted_definition_makes_a_name_unique_again(tmp_path: Path) -> None:
    """The unresolved arm admits only a name with ONE definition, counted after the overlay.

    `helper` is defined in `lib.py` and `dup.py`, so the backbone refuses to bind
    `e_test.py`'s unresolved call. Deleting `dup.py` leaves one definition.
    """
    database = _graph(
        tmp_path / "backbone.db",
        nodes=[
            ("old:helper", "function", "helper", "lib.py", 1),
            ("old:dup", "function", "helper", "dup.py", 1),
            ("old:e", "function", "test_e", "e_test.py", 1),
            ("old:import", "import", "lib", "e_test.py", 9),
        ],
        refs=[("old:e", "helper", "e_test.py")],
    )
    overlay = _overlay(tmp_path, changed=["dup.py"])

    assert harvest(database, changed=["lib.py"], depth=1, emit="nodeids") == []
    assert harvest(
        database,
        changed=["lib.py"],
        depth=1,
        emit="nodeids",
        overlay=overlay,
    ) == ["e_test.py::test_e"]


def test_an_overlay_method_is_selected_through_its_overlay_class(
    tmp_path: Path,
) -> None:
    database = _graph(
        tmp_path / "backbone.db",
        nodes=[("old:helper", "function", "helper", "lib.py", 1)],
    )
    overlay = _overlay(
        tmp_path,
        changed=["x_test.py"],
        nodes=[
            ("new:class", "class", "TestX", "x_test.py", 1),
            ("new:method", "method", "test_m", "x_test.py", 2),
        ],
        edges=[
            ("new:class", "new:method", "contains"),
            ("new:method", "old:helper", "calls"),
        ],
    )

    assert harvest(
        database,
        changed=["x_test.py"],
        depth=1,
        emit="nodeids",
        overlay=overlay,
    ) == ["x_test.py::TestX::test_m"]


def _arm(sql: str, name: str) -> str:
    """Return arm ``name`` of the walk, comments dropped and whitespace collapsed."""
    body = sql.split(f"-- ARM {name}\n", 1)[1]
    body = re.split(r"\n\s*UNION\n|\n\)\nSELECT", body, maxsplit=1)[0]
    return " ".join(
        " ".join(line.split("--", 1)[0] for line in body.splitlines()).split(),
    )


@pytest.mark.parametrize(
    ("kind", "tables", "only"),
    [
        (
            "edges",
            [("main.edges", "temp.oedges"), ("main.nodes", "temp.onodes")],
            "AND source.file_path NOT IN (SELECT path FROM temp.changed) ",
        ),
        (
            "refs",
            [("derived.refs", "temp.orefs"), ("main.nodes", "temp.onodes")],
            "AND unresolved_refs.file_path NOT IN (SELECT path FROM temp.changed) ",
        ),
    ],
)
def test_each_arm_pair_shares_one_admission_rule(
    kind: str,
    tables: list[tuple[str, str]],
    only: str,
) -> None:
    """The backbone and overlay twins differ in their tables and one filter, nothing else.

    They are written out rather than formatted from one template, so a fix to one
    admission test that missed its twin would otherwise pass every walk test that
    never crosses that arm.
    """
    sql = affected._walk()
    backbone_arm = _arm(sql, f"{kind}/backbone")
    for main_table, temp_table in tables:
        backbone_arm = backbone_arm.replace(main_table, temp_table)

    assert only in backbone_arm
    assert backbone_arm.replace(only, "") == _arm(sql, f"{kind}/overlay")


def test_a_graph_problem_is_reported_and_the_walk_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A gate must never walk an index it cannot trust, and never do so silently."""
    (tmp_path / "x_test.py").touch()
    monkeypatch.delenv("SKIP", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["affected", "--caches", str(tmp_path / "caches"), "--", "x.py"],
    )

    assert affected.main() == 0
    out, err = capfd.readouterr()
    assert out == "x_test.py\n"
    assert "No CodeGraph backbone" in err
    assert "without the graph" in err


def _stale_backbone(caches: Path) -> None:
    """Publish a backbone in which ``x_test.py::test_x`` calls ``x.f``, and no overlay."""
    database = backbone.backbone_database(caches)
    database.parent.mkdir(parents=True)
    _graph(
        database,
        nodes=[
            ("n:f", "function", "f", "x.py", 1),
            ("n:t", "function", "test_x", "x_test.py", 1),
        ],
        edges=[("n:t", "n:f", "calls")],
    )
    backbone.stamp(database, sha="a" * 40)


def test_a_problem_with_a_readable_backbone_still_never_walks_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Without a current overlay the backbone may name deleted tests; it must not be walked."""
    _stale_backbone(tmp_path / "caches")
    (tmp_path / "x_test.py").touch()
    monkeypatch.delenv("SKIP", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["affected", "--caches", str(tmp_path / "caches"), "--", "x.py"],
    )

    assert affected.main() == 0
    out, err = capfd.readouterr()
    assert out == "x_test.py\n"
    assert "No overlay" in err


@pytest.mark.cli_git
def test_main_checks_the_overlay_against_the_real_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Freshness is decided by asking git, not by trusting the stamp."""
    caches = tmp_path / "caches"
    _stale_backbone(caches)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)  # noqa: S607 -- git from PATH.
    overlay = backbone.overlay_database(caches, root=tmp_path)
    overlay.parent.mkdir(parents=True)
    _graph(overlay, nodes=[])
    backbone.stamp(overlay, sha="a" * 40, fingerprint="stale", paths=["x.py"])
    monkeypatch.delenv("SKIP", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["affected", "--caches", str(caches), "--", "x.py"],
    )

    assert affected.main() == 0
    assert "git fetch origin" in capfd.readouterr().err


def test_a_skipped_graph_reads_and_reports_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """`SKIP` already chose the floor; checking the overlay would only add noise."""
    monkeypatch.setenv("SKIP", "harvest-affected")
    monkeypatch.setattr(
        sys,
        "argv",
        ["affected", "--caches", str(tmp_path / "caches"), "--", "x.py"],
    )

    assert affected.main() == 0
    assert capfd.readouterr().err == ""


def test_the_derived_cache_is_built_beside_the_index_under_its_tmp_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rename onto the cache must not cross devices; /tmp is often its own.

    The `tmp-` prefix is also what spares a concurrent builder's scratch file
    from the stale-cache sweep.
    """
    database = _index(tmp_path)
    seen: list[list[str]] = []
    real = affected._build_derived

    def spying(connection: sqlite3.Connection) -> None:
        seen.append(sorted(path.name for path in tmp_path.iterdir()))
        real(connection)

    monkeypatch.setattr(affected, "_build_derived", spying)
    builder = database.with_name(f"{database.name}.affected-tmp-other")
    builder.touch()

    cache = affected.ensure_derived(database)

    (during,) = seen
    (scratch,) = [
        name
        for name in during
        if name.startswith(f"{database.name}.affected-tmp-") and name != builder.name
    ]
    assert scratch not in {path.name for path in tmp_path.iterdir()}
    assert cache is not None
    assert builder.exists()


def test_the_derived_cache_is_group_readable_and_keyed_on_this_code(
    tmp_path: Path,
) -> None:
    """Every user reads one shared cache; checkouts of other code keep their own."""
    database = _index(tmp_path)
    first = affected.ensure_derived(database)
    assert first is not None
    assert first.stat().st_mode & 0o777 == 0o664
    index, code = first.name.removeprefix(f"{database.name}.affected-").split("-")
    assert len(code) == 16
    # Same index, other code: another checkout's cache, which this build must spare.
    first.unlink()
    sibling = database.with_name(f"{database.name}.affected-{index}-0000")
    other = database.with_name(f"{database.name}.affected-ffff-0000")
    sibling.touch()
    other.touch()

    assert affected.ensure_derived(database) == first
    assert sibling.exists()
    assert not other.exists()
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO edges VALUES ('n:a', 'n:b', 'calls')")
    second = affected.ensure_derived(database)
    assert second is not None
    assert second.name.endswith(f"-{code}")
    assert not first.exists()
    assert not sibling.exists()


def test_a_fresh_overlay_reaches_harvest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = tmp_path / "overlay.db"
    calls = _recorded_harvest(monkeypatch, overlay=found)
    monkeypatch.delenv("SKIP", raising=False)
    monkeypatch.setattr(sys, "argv", ["affected", "--caches", "/c", "--", "x.py"])

    assert affected.main() == 0
    assert calls[0]["database"] == Path("/c/backbone/codegraph.db")
    assert calls[0]["overlay"] == found
    assert calls[0]["depth"] == 1


def test_an_explicit_database_is_read_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--database` names an index to walk as it stands, with no overlay."""
    calls = _recorded_harvest(monkeypatch)
    monkeypatch.delenv("SKIP", raising=False)
    monkeypatch.setattr(sys, "argv", ["affected", "--database", "x.db", "--", "x.py"])

    assert affected.main() == 0
    assert (calls[0]["database"], calls[0]["overlay"]) == (Path("x.db"), None)


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
