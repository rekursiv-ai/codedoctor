"""Tests for the backbone refresh and the per-commit overlay sync."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import fcntl
import os
import shutil
import sqlite3
import subprocess
import sys

import pytest

from codedoctor.affected import backbone, codegraph_sync
from codedoctor.gate.gate import output_argv, run_argv


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_OLD, _NEW = "a" * 40, "b" * 40


class _Tools:
    """Stands in for git, npm, and the codegraph CLI; records every argv."""

    def __init__(
        self,
        *,
        sha: str = _NEW,
        codegraph: int = 0,
        diff: str = "",
        installed: bool = True,
        installable: bool = False,
    ) -> None:
        self.sha = sha
        self.codegraph = codegraph
        self.diff = diff
        self.installed = installed
        self.installable = installable
        self.calls: list[list[str]] = []
        self.spawned: list[tuple[list[str], Path, Path]] = []

    def run(self, argv: Sequence[str]) -> int:
        """Run one argv: codegraph writes an index into the tree it names."""
        self.calls.append(list(argv))
        if argv[0] == "codegraph" and not self.codegraph:
            _index(Path(argv[-1]) / ".codegraph" / "codegraph.db")
        return self.codegraph if argv[0] == "codegraph" else 0

    def output(self, argv: Sequence[str]) -> str | None:
        """Answer ``rev-parse`` with ``sha`` (None when empty), ``git diff`` with ``diff``."""
        self.calls.append(list(argv))
        if "rev-parse" in argv:
            return f"{self.sha}\n" if self.sha else None
        return self.diff if "diff" in argv else ""

    def spawn(self, argv: Sequence[str], *, cwd: Path, log: Path) -> object:
        """Record a detached process in place of starting it."""
        self.spawned.append((list(argv), cwd, log))
        return None

    def which(self, name: str) -> str | None:
        """Find only the codegraph CLI: installed, or installable and installed by npm."""
        npm = ["npm", "install", "-g", "@colbymchenry/codegraph"]
        found = name == "codegraph" and (
            self.installed or (self.installable and npm in self.calls)
        )
        return f"/bin/{name}" if found else None

    def ran(self, tool: str) -> list[list[str]]:
        """Return the recorded argvs holding the word ``tool``."""
        return [argv for argv in self.calls if tool in argv]


def _index(database: Path) -> None:
    """Write a CodeGraph-shaped index, in CodeGraph's WAL mode."""
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        "CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT,"
        " qualified_name TEXT, file_path TEXT, start_line INTEGER, end_line INTEGER);"
        "CREATE TABLE IF NOT EXISTS edges (source TEXT, target TEXT, kind TEXT);"
        "CREATE TABLE IF NOT EXISTS unresolved_refs (from_node_id TEXT,"
        " reference_name TEXT, reference_kind TEXT, file_path TEXT, name_tail TEXT);",
    )
    connection.close()


def _refresh(tmp_path: Path, tools: _Tools, *, fetch: bool = False) -> str:
    return codegraph_sync.refresh(
        tmp_path / "repo",
        caches=tmp_path / "caches",
        fetch=fetch,
        run=tools.run,
        output=tools.output,
    )


def _sync(tmp_path: Path, tools: _Tools) -> str:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    return codegraph_sync.sync(
        root,
        caches=tmp_path / "caches",
        run=tools.run,
        output=tools.output,
        which=tools.which,
        spawn=tools.spawn,
    )


def test_a_first_refresh_snapshots_indexes_and_publishes(tmp_path: Path) -> None:
    tools = _Tools()

    assert _refresh(tmp_path, tools) == ""

    home = tmp_path / "caches" / "backbone"
    database = backbone.backbone_database(tmp_path / "caches")
    assert tools.ran("read-tree") == [
        [
            "env",
            f"GIT_INDEX_FILE={home / 'index'}",
            "git",
            "-C",
            str(tmp_path / "repo"),
            f"--work-tree={home / 'tree'}",
            "read-tree",
            "--reset",
            "-u",
            _NEW,
        ],
    ]
    assert tools.ran("codegraph") == [
        ["codegraph", "init", "--yes", str(home / "tree")],
    ]
    assert backbone.stamped(database) == backbone.Stamp(sha=_NEW)
    assert database.stat().st_mode & 0o777 == 0o664
    assert home.stat().st_mode & 0o7777 == 0o2775


def test_the_published_backbone_holds_what_codegraph_indexed(tmp_path: Path) -> None:
    tools = _Tools()

    def indexing(argv: Sequence[str]) -> int:
        status = tools.run(argv)
        if argv[0] == "codegraph":
            index = Path(argv[-1]) / ".codegraph" / "codegraph.db"
            with closing(sqlite3.connect(index)) as connection, connection:
                connection.execute("INSERT INTO nodes (id) VALUES ('indexed')")
        return status

    codegraph_sync.refresh(
        tmp_path / "repo",
        caches=tmp_path / "caches",
        fetch=False,
        run=indexing,
        output=tools.output,
    )

    database = backbone.backbone_database(tmp_path / "caches")
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT id FROM nodes").fetchall() == [("indexed",)]


def test_the_published_backbone_is_one_self_contained_file(tmp_path: Path) -> None:
    """Readers open it read-only; a WAL beside it would need their write access."""
    _refresh(tmp_path, _Tools())
    database = backbone.backbone_database(tmp_path / "caches")

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}.tmp").exists()


def test_the_refresh_prebuilds_the_derived_tables_for_every_reader(
    tmp_path: Path,
) -> None:
    _refresh(tmp_path, _Tools())
    home = tmp_path / "caches" / "backbone"

    (cache,) = home.glob("codegraph.db.affected-*")
    assert cache.stat().st_mode & 0o777 == 0o664


def test_a_later_refresh_syncs_the_index_it_kept(tmp_path: Path) -> None:
    _refresh(tmp_path, _Tools(sha=_OLD))
    tools = _Tools()

    assert _refresh(tmp_path, tools) == ""
    assert tools.ran("codegraph") == [
        [
            "codegraph",
            "sync",
            "--quiet",
            str(tmp_path / "caches" / "backbone" / "tree"),
        ],
    ]
    assert backbone.stamped(backbone.backbone_database(tmp_path / "caches")) == (
        backbone.Stamp(sha=_NEW)
    )


def test_a_current_backbone_is_left_alone(tmp_path: Path) -> None:
    _refresh(tmp_path, _Tools())
    tools = _Tools()

    assert _refresh(tmp_path, tools) == ""
    assert tools.ran("codegraph") == []
    assert tools.ran("read-tree") == []


def test_a_failed_index_keeps_the_published_backbone(tmp_path: Path) -> None:
    """`codegraph sync` exits 1 on a read-only index; the old hook ignored that."""
    _refresh(tmp_path, _Tools(sha=_OLD))

    problem = _refresh(tmp_path, _Tools(codegraph=1))

    assert "exit 1" in problem
    assert backbone.stamped(backbone.backbone_database(tmp_path / "caches")) == (
        backbone.Stamp(sha=_OLD)
    )


def test_an_index_missing_a_table_fails_the_derived_build(tmp_path: Path) -> None:
    """Readers would rebuild the derived tables and fail too, so the refresh fails."""
    tools = _Tools()

    def tableless(argv: Sequence[str]) -> int:
        tools.calls.append(list(argv))
        if argv[0] == "codegraph":
            index = Path(argv[-1]) / ".codegraph" / "codegraph.db"
            index.parent.mkdir(parents=True)
            with closing(sqlite3.connect(index)) as connection, connection:
                connection.execute("CREATE TABLE nodes (id TEXT)")
        return 0

    problem = codegraph_sync.refresh(
        tmp_path / "repo",
        caches=tmp_path / "caches",
        fetch=False,
        run=tableless,
        output=tools.output,
    )

    database = backbone.backbone_database(tmp_path / "caches")
    assert problem == f"Building the derived tables of {database} failed."
    assert not list(database.parent.glob("codegraph.db.affected-*"))


def test_a_held_lock_skips_the_refresh(tmp_path: Path) -> None:
    """Two refreshes at once would both write the one snapshot tree."""
    home = tmp_path / "caches" / "backbone"
    home.mkdir(parents=True)
    tools = _Tools()
    with (home / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert _refresh(tmp_path, tools) == ""
    assert tools.calls == []


def test_fetch_runs_before_resolving_origin_main(tmp_path: Path) -> None:
    tools = _Tools()

    _refresh(tmp_path, tools, fetch=True)

    assert tools.calls[0] == [
        "git",
        "-C",
        str(tmp_path / "repo"),
        "fetch",
        "--quiet",
        "origin",
    ]
    assert tools.calls[1] == [
        "git",
        "-C",
        str(tmp_path / "repo"),
        "rev-parse",
        "--verify",
        "origin/main^{commit}",
    ]


def test_a_failed_fetch_or_missing_origin_main_is_reported(tmp_path: Path) -> None:
    def failing(argv: Sequence[str]) -> int:
        return 128 if "fetch" in argv else 0

    assert "fetch" in codegraph_sync.refresh(
        tmp_path,
        caches=tmp_path / "caches",
        fetch=True,
        run=failing,
        output=_Tools().output,
    )
    tools = _Tools(sha="")
    assert "origin/main" in _refresh(tmp_path, tools)
    assert tools.ran("read-tree") == []


def test_a_failed_snapshot_is_reported(tmp_path: Path) -> None:
    def failing(argv: Sequence[str]) -> int:
        return 128 if "read-tree" in argv else 0

    problem = codegraph_sync.refresh(
        tmp_path,
        caches=tmp_path / "caches",
        fetch=False,
        run=failing,
        output=_Tools().output,
    )

    assert "read-tree" in problem


# Distinct past its 12th character, so a message shows exactly the prefix it names.
_LONG = "0123456789abcdef" * 2 + "01234567"


def test_a_failed_snapshot_names_the_commit_and_status(tmp_path: Path) -> None:
    def failing(argv: Sequence[str]) -> int:
        return 128 if "read-tree" in argv else 0

    problem = codegraph_sync.refresh(
        tmp_path,
        caches=tmp_path / "caches",
        fetch=False,
        run=failing,
        output=_Tools(sha=_LONG).output,
    )

    assert problem == "`git read-tree` of 0123456789ab failed (exit 128)."


def test_a_failed_index_names_the_commit_and_status(tmp_path: Path) -> None:
    problem = _refresh(tmp_path, _Tools(sha=_LONG, codegraph=3))

    assert problem == "Indexing 0123456789ab failed (codegraph exit 3)."


def test_a_missing_backbone_starts_one_in_the_background(tmp_path: Path) -> None:
    """The first commit pays nothing; its gates select without the graph."""
    tools = _Tools()

    problem = _sync(tmp_path, tools)

    log = backbone.overlay_database(tmp_path / "caches", root=tmp_path / "root")
    assert problem == (
        "No CodeGraph backbone yet; building one in the background "
        f"(log: {log.with_name('refresh.log')}). "
        "Gates select without the graph until it lands."
    )
    assert tools.spawned == [
        (
            [
                sys.executable,
                "-m",
                "codedoctor.affected.codegraph_sync",
                "--caches",
                str(tmp_path / "caches"),
                "--refresh",
            ],
            tmp_path / "root",
            backbone.overlay_database(
                tmp_path / "caches",
                root=tmp_path / "root",
            ).with_name(
                "refresh.log",
            ),
        ),
    ]
    assert str(tools.spawned[0][2]) in problem


def test_a_published_backbone_gets_this_worktrees_overlay(tmp_path: Path) -> None:
    _refresh(tmp_path, _Tools())
    tools = _Tools()

    assert _sync(tmp_path, tools) == ""
    assert tools.spawned == []
    overlay = backbone.overlay_database(tmp_path / "caches", root=tmp_path / "root")
    assert backbone.stamped(overlay) is not None


def test_a_backbone_behind_origin_main_is_refreshed_in_the_background(
    tmp_path: Path,
) -> None:
    """Nothing else republished it: 29 commits behind, the overlay held 1,019 files."""
    _refresh(tmp_path, _Tools(sha=_OLD))
    tools = _Tools(sha=_NEW)

    assert _sync(tmp_path, tools) == ""
    assert tools.spawned == [
        (
            [
                sys.executable,
                "-m",
                "codedoctor.affected.codegraph_sync",
                "--caches",
                str(tmp_path / "caches"),
                "--refresh",
                "--fetch",
            ],
            tmp_path / "root",
            backbone.overlay_database(
                tmp_path / "caches",
                root=tmp_path / "root",
            ).with_name("refresh.log"),
        ),
    ]
    overlay = backbone.overlay_database(tmp_path / "caches", root=tmp_path / "root")
    assert backbone.stamped(overlay) is not None
    assert tools.ran("rev-parse") == [
        [
            "git",
            "-C",
            str(tmp_path / "root"),
            "rev-parse",
            "--verify",
            "origin/main^{commit}",
        ],
    ]


def test_the_gate_gives_no_worktree_a_cli_index(tmp_path: Path) -> None:
    """Each gate-made CLI index copied the whole backbone: 825 MB per worktree."""
    _refresh(tmp_path, _Tools())

    _sync(tmp_path, _Tools())

    assert not (tmp_path / "root" / ".codegraph").exists()
    overlay = backbone.overlay_database(tmp_path / "caches", root=tmp_path / "root")
    assert not (overlay.parent / "cli").exists()


def test_the_gate_restores_a_removed_cli_link_without_copying(tmp_path: Path) -> None:
    _refresh(tmp_path, _Tools())
    root = tmp_path / "root"
    root.mkdir()
    caches = tmp_path / "caches"
    assert codegraph_sync.link_cli_index(root, caches=caches) == ""
    link = root / ".codegraph"
    database = link.resolve() / "codegraph.db"
    before = database.read_bytes(), database.stat().st_ino
    link.unlink()

    assert _sync(tmp_path, _Tools()) == ""

    assert link.is_symlink()
    assert link.resolve() == database.parent
    assert (database.read_bytes(), database.stat().st_ino) == before


def test_the_gate_restores_a_cli_link_without_a_backbone(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    caches = tmp_path / "caches"
    database = backbone.overlay_database(caches, root=root).parent / "cli/codegraph.db"
    _index(database)

    assert "building one in the background" in _sync(tmp_path, _Tools())

    assert (root / ".codegraph").is_symlink()
    assert (root / ".codegraph").resolve() == database.parent


def test_the_gate_reports_a_cli_link_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    caches = tmp_path / "caches"
    database = backbone.overlay_database(caches, root=root).parent / "cli/codegraph.db"
    _index(database)

    def denied(self: Path, target: Path) -> None:
        assert self == root / ".codegraph"
        assert target == database.parent
        raise PermissionError("link denied")

    monkeypatch.setattr(Path, "symlink_to", denied)

    assert "link denied" in _sync(tmp_path, _Tools())
    assert not (root / ".codegraph").is_symlink()


def test_the_gate_does_not_link_an_unseeded_cli_directory(tmp_path: Path) -> None:
    _refresh(tmp_path, _Tools())
    root = tmp_path / "root"
    root.mkdir()
    home = backbone.overlay_database(tmp_path / "caches", root=root).parent / "cli"
    home.mkdir(parents=True)

    assert _sync(tmp_path, _Tools()) == ""

    assert not (root / ".codegraph").is_symlink()
    assert list(home.iterdir()) == []


@pytest.mark.parametrize("symlink", [False, True])
def test_the_gate_preserves_an_existing_cli_index(
    tmp_path: Path,
    symlink: bool,
) -> None:
    _refresh(tmp_path, _Tools())
    root = tmp_path / "root"
    root.mkdir()
    caches = tmp_path / "caches"
    assert codegraph_sync.link_cli_index(root, caches=caches) == ""
    link = root / ".codegraph"
    link.unlink()
    own = tmp_path / "own"
    if symlink:
        link.symlink_to(own)
    else:
        link.mkdir()

    assert _sync(tmp_path, _Tools()) == ""

    assert link.is_symlink() is symlink
    if symlink:
        assert link.readlink() == own
    else:
        assert list(link.iterdir()) == []


def test_a_worktree_gets_a_cli_index_seeded_from_the_backbone(tmp_path: Path) -> None:
    """Without ``.codegraph`` in the worktree, ``codegraph sync`` there refused to run."""
    _refresh(tmp_path, _Tools())
    root = tmp_path / "root"
    root.mkdir()

    assert codegraph_sync.link_cli_index(root, caches=tmp_path / "caches") == ""

    link = root / ".codegraph"
    overlay = backbone.overlay_database(tmp_path / "caches", root=root)
    assert link.is_symlink()
    assert link.resolve() == overlay.parent / "cli"
    seeded = backbone.stamped(link / "codegraph.db")
    published = backbone.stamped(backbone.backbone_database(tmp_path / "caches"))
    assert seeded is not None
    assert seeded == published


def test_a_removed_link_is_relinked_to_the_index_it_had(tmp_path: Path) -> None:
    _refresh(tmp_path, _Tools())
    root = tmp_path / "root"
    root.mkdir()
    caches = tmp_path / "caches"
    assert codegraph_sync.link_cli_index(root, caches=caches) == ""
    (root / ".codegraph").unlink()

    assert codegraph_sync.link_cli_index(root, caches=caches) == ""

    overlay = backbone.overlay_database(caches, root=root)
    assert (root / ".codegraph").resolve() == overlay.parent / "cli"


def test_an_existing_cli_index_is_left_alone(tmp_path: Path) -> None:
    _refresh(tmp_path, _Tools())
    own = tmp_path / "root" / ".codegraph"
    own.mkdir(parents=True)

    assert codegraph_sync.link_cli_index(own.parent, caches=tmp_path / "caches") == ""

    assert not own.is_symlink()
    assert list(own.iterdir()) == []


def test_a_cli_index_without_a_backbone_is_refused(tmp_path: Path) -> None:
    """An empty ``cli`` directory once stood in for an index the copy never made."""
    root = tmp_path / "root"
    root.mkdir()

    problem = codegraph_sync.link_cli_index(root, caches=tmp_path / "caches")

    assert problem == (
        "No CodeGraph backbone to seed from; run --refresh, or run hooks "
        "with PRECOMMIT_SCOPE=affected:1."
    )
    assert not (root / ".codegraph").exists()
    overlay = backbone.overlay_database(tmp_path / "caches", root=root)
    assert not (overlay.parent / "cli").exists()


def test_an_overlay_is_indexed_by_the_codegraph_cli(tmp_path: Path) -> None:
    _refresh(tmp_path, _Tools())
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "a.py").write_text("def f(): pass\n")
    tools = _Tools(diff="a.py\0")

    assert _sync(tmp_path, tools) == ""
    ((codegraph, init, yes, tree),) = tools.ran("codegraph")
    assert (codegraph, init, yes) == ("codegraph", "init", "--yes")
    assert Path(tree).name == "tree"


def test_a_missing_cli_that_npm_cannot_install_is_reported(tmp_path: Path) -> None:
    tools = _Tools(installed=False)

    problem = _sync(tmp_path, tools)

    assert tools.calls == [["npm", "install", "-g", "@colbymchenry/codegraph"]]
    assert problem == (
        "The codegraph CLI is missing, and `npm install -g "
        "@colbymchenry/codegraph` did not install it."
    )


def test_a_missing_cli_is_installed_then_used(tmp_path: Path) -> None:
    tools = _Tools(installed=False, installable=True)

    problem = _sync(tmp_path, tools)

    assert tools.calls == [["npm", "install", "-g", "@colbymchenry/codegraph"]]
    assert "background" in problem


def test_an_installed_cli_is_not_reinstalled(tmp_path: Path) -> None:
    tools = _Tools()

    _sync(tmp_path, tools)

    assert tools.ran("npm") == []


@pytest.mark.parametrize("value", ["exact", "sidecar"])
def test_narrow_global_scopes_need_no_sync(value: str) -> None:
    assert not codegraph_sync.opted_in({"PRECOMMIT_SCOPE": value})


def test_default_type_scopes_need_sync() -> None:
    assert codegraph_sync.opted_in({})


@pytest.mark.parametrize(
    "name",
    ["TY_CHECK", "PYREFLY_CHECK", "BASEDPYRIGHT", "PYTEST", "MUTATION"],
)
def test_each_affected_hook_needs_sync(name: str) -> None:
    environ = {
        f"PRECOMMIT_SCOPE_{hook}": "sidecar"
        for hook in (
            "TY_CHECK",
            "PYREFLY_CHECK",
            "BASEDPYRIGHT",
            "PYTEST",
            "MUTATION",
        )
    }
    environ[f"PRECOMMIT_SCOPE_{name}"] = "affected:2"
    assert codegraph_sync.opted_in(environ)


@pytest.mark.parametrize(
    "skip",
    ["harvest-affected", "ty-check,pyrefly-check,basedpyright"],
)
def test_skipped_graph_consumers_need_no_sync(skip: str) -> None:
    assert not codegraph_sync.opted_in({"SKIP": skip})


def _main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *argv: str,
    opt_in: str = "affected:1",
) -> list[str]:
    """Run ``main`` in ``tmp_path``; return each entry point's call, as it saw it."""
    called: list[str] = []
    caches = tmp_path / "caches"

    def sync(root: Path, **kwargs: object) -> str:
        assert kwargs == {
            "caches": caches,
            "run": run_argv,
            "output": output_argv,
            "which": shutil.which,
            "spawn": codegraph_sync.spawn,
        }
        called.append(f"sync {root}")
        return "overlay problem"

    def refresh(repo: Path, *, fetch: bool, **kwargs: object) -> str:
        assert kwargs == {"caches": caches, "run": run_argv, "output": output_argv}
        called.append(f"refresh {repo} fetch={fetch} umasks={masks}")
        return "refresh problem"

    def link_cli_index(root: Path, **kwargs: object) -> str:
        assert kwargs == {"caches": caches}
        called.append(f"cli {root}")
        return "cli problem"

    # Recorded, never applied: a real `os.umask` here would outlive the test.
    masks: list[int] = []

    def umask(mask: int) -> int:
        masks.append(mask)
        return 0o022

    monkeypatch.setattr(os, "umask", umask)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PRECOMMIT_SCOPE", opt_in)
    monkeypatch.delenv("SKIP", raising=False)
    monkeypatch.setattr(codegraph_sync, "sync", sync)
    monkeypatch.setattr(codegraph_sync, "refresh", refresh)
    monkeypatch.setattr(codegraph_sync, "link_cli_index", link_cli_index)
    monkeypatch.setattr(
        sys,
        "argv",
        ["codegraph_sync", "--caches", str(tmp_path / "caches"), *argv],
    )
    return called


@pytest.mark.parametrize(
    ("opt_in", "synced"),
    [("affected:1", True), ("sidecar", False)],
)
def test_the_hook_reports_a_problem_and_still_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    opt_in: str,
    synced: bool,
) -> None:
    """A commit never fails on the index; `affected` refuses a stale overlay itself."""
    called = _main(monkeypatch, tmp_path, opt_in=opt_in)

    assert codegraph_sync.main() == 0
    assert called == ([f"sync {tmp_path}"] if synced else [])
    assert ("codegraph-sync: overlay problem" in capsys.readouterr().err) is synced


def test_a_failed_refresh_fails_its_timer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    called = _main(monkeypatch, tmp_path, "--refresh", "--fetch", opt_in="0")

    assert codegraph_sync.main() == 1
    assert called == [f"refresh {tmp_path} fetch=True umasks=[2]"]
    assert "refresh problem" in capsys.readouterr().err


@pytest.mark.parametrize("opt_in", ["1", "0"])
def test_cli_mode_links_the_index_whatever_the_gates_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    opt_in: str,
) -> None:
    """The CLI's index is asked for by its user, not by the gates' opt-in."""
    called = _main(monkeypatch, tmp_path, "--cli", opt_in=opt_in)

    assert codegraph_sync.main() == 1
    assert called == [f"cli {tmp_path}"]
    assert "codegraph-sync: cli problem" in capsys.readouterr().err


def test_the_help_describes_both_modes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["codegraph_sync", "--help"])
    with pytest.raises(SystemExit):
        codegraph_sync.main()
    text = capsys.readouterr().out

    assert "Keep the shared CodeGraph backbone" in text
    assert "--refresh republishes the backbone" in text
    assert "--cli links this worktree" in text
    assert "exec uv" not in text


def test_the_help_has_no_description_when_docstrings_are_stripped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As ``python -OO`` runs it.
    monkeypatch.setattr(codegraph_sync, "__doc__", None)
    monkeypatch.setattr(sys, "argv", ["codegraph_sync", "--help"])

    with pytest.raises(SystemExit):
        codegraph_sync.main()

    assert capsys.readouterr().out.split("\n\n")[1].startswith("options:")


def test_the_caches_must_be_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """No default: the hook's config names the one shared directory."""
    monkeypatch.setattr(sys, "argv", ["codegraph_sync", "--refresh"])

    with pytest.raises(SystemExit) as raised:
        codegraph_sync.main()

    assert raised.value.code == 2


def test_a_spawned_refresh_outlives_the_hook(tmp_path: Path) -> None:
    """Its own session, no stdin, one log: the hook's exit must not take it down."""
    log = tmp_path / "logs" / "nested" / "refresh.log"
    script = (
        "import os, sys; print(os.getsid(0) == os.getpid(), repr(sys.stdin.read()),"
        " os.getcwd(), flush=True); sys.stderr.write('err\\n')"
    )

    for _ in range(2):
        child = codegraph_sync.spawn(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            log=log,
        )
        assert child.wait(timeout=30) == 0

    assert log.read_text() == f"True '' {tmp_path}\nerr\n" * 2


def test_a_spawned_refresh_never_reads_the_hooks_stdin(tmp_path: Path) -> None:
    """A detached child holding the hook's stdin would steal a terminal's input."""
    log = tmp_path / "refresh.log"
    read, write = os.pipe()
    saved = os.dup(0)
    try:
        os.dup2(read, 0)
        child = codegraph_sync.spawn(
            [sys.executable, "-c", "import os; print(os.readlink('/proc/self/fd/0'))"],
            cwd=tmp_path,
            log=log,
        )
        assert child.wait(timeout=30) == 0
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read, write):
            os.close(fd)

    assert log.read_text() == f"{os.devnull}\n"


@pytest.mark.cli_git
def test_a_snapshot_follows_origin_main_deletions_included(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot is updated in place; a file deleted upstream must leave it."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kept.py").write_text("a")
    (repo / "gone.py").write_text("a")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    calls: list[list[str]] = []
    real = _real_git_fake_codegraph(calls)

    assert (
        codegraph_sync.refresh(
            repo,
            caches=tmp_path / "caches",
            fetch=False,
            run=real,
            output=output_argv,
        )
        == ""
    )
    tree = tmp_path / "caches" / "backbone" / "tree"
    assert sorted(path.name for path in tree.iterdir()) == [
        ".codegraph",
        "gone.py",
        "kept.py",
    ]
    _git(repo, "rm", "-q", "gone.py")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "two")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")

    assert (
        codegraph_sync.refresh(
            repo,
            caches=tmp_path / "caches",
            fetch=False,
            run=real,
            output=output_argv,
        )
        == ""
    )
    assert sorted(path.name for path in tree.iterdir()) == [".codegraph", "kept.py"]
    assert _git(repo, "status", "--porcelain") == ""


def _real_git_fake_codegraph(calls: list[list[str]]) -> Callable[[Sequence[str]], int]:
    def run(argv: Sequence[str]) -> int:
        calls.append(list(argv))
        if argv[0] == "codegraph":
            _index(Path(argv[-1]) / ".codegraph" / "codegraph.db")
            return 0
        return run_argv(argv)

    return run


def _git(root: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 -- git from PATH; arguments are test literals.
        ["git", *args],  # noqa: S607 -- git from PATH, as every git test here runs it.
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
