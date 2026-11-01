"""Tests for the shared backbone's layout and the per-worktree overlay."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import os
import sqlite3
import subprocess

import pytest

from codedoctor.affected import backbone
from codedoctor.gate.gate import output_argv


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_SHA = "a" * 40


def _codegraph(database: Path, *, files: Sequence[str] = ()) -> Path:
    """Write a CodeGraph-shaped index holding one function per file in ``files``."""
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT,"
        " qualified_name TEXT, file_path TEXT, start_line INTEGER, end_line INTEGER);"
        "CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);"
        "CREATE TABLE unresolved_refs (from_node_id TEXT, reference_name TEXT,"
        " reference_kind TEXT, file_path TEXT, name_tail TEXT);",
    )
    connection.executemany(
        "INSERT INTO nodes VALUES (?, 'function', 'f', 'f', ?, 1, 2)",
        [(f"n:{path}", path) for path in files],
    )
    connection.commit()
    connection.close()
    return database


def _published(caches: Path, *, sha: str = _SHA) -> Path:
    """Publish a stamped backbone under ``caches``."""
    database = _codegraph(backbone.backbone_database(caches))
    backbone.stamp(database, sha=sha)
    return database


def _git(
    *,
    diff: str | None = "",
    others: str | None = "",
    seen: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], str | None]:
    """Return a git double answering ``diff`` and ``ls-files --others``, into ``seen``."""

    def output(argv: Sequence[str]) -> str | None:
        if seen is not None:
            seen.append(list(argv))
        return others if "ls-files" in argv else diff

    return output


def test_a_backbone_stamp_carries_only_its_commit(tmp_path: Path) -> None:
    database = _codegraph(tmp_path / "x.db")
    backbone.stamp(database, sha=_SHA)

    assert backbone.stamped(database) == backbone.Stamp(
        sha=_SHA,
        fingerprint="",
        paths=(),
    )


def _indexer(calls: list[Path]) -> Callable[[Path], int]:
    """Return an indexer double: records the tree, writes its index like CodeGraph."""

    def index(tree: Path) -> int:
        calls.append(tree)
        files = sorted(path.relative_to(tree).as_posix() for path in tree.rglob("*.py"))
        _codegraph(tree / ".codegraph" / "codegraph.db", files=files)
        return 0

    return index


def test_every_worktree_reads_the_one_backbone(tmp_path: Path) -> None:
    assert backbone.backbone_database(tmp_path) == (
        tmp_path / "backbone" / "codegraph.db"
    )


def test_each_worktree_gets_its_own_overlay(tmp_path: Path) -> None:
    """Two checkouts with one basename once shared a whole 1 GB index."""
    first = backbone.overlay_database(tmp_path, root=tmp_path / "a" / "loop")
    second = backbone.overlay_database(tmp_path, root=tmp_path / "b" / "loop")

    assert first != second
    assert first.parent.parent == tmp_path / "overlays"
    assert first.parent.name.startswith("loop-")
    assert first == backbone.overlay_database(tmp_path, root=tmp_path / "a" / "loop")


def test_a_stamp_round_trips(tmp_path: Path) -> None:
    database = _codegraph(tmp_path / "x.db")
    backbone.stamp(database, sha=_SHA, fingerprint="f", paths=["b.py", "a.py"])

    assert backbone.stamped(database) == backbone.Stamp(
        sha=_SHA,
        fingerprint="f",
        paths=("a.py", "b.py"),
    )


@pytest.mark.parametrize("content", [b"", b"not sqlite at all, just bytes " * 40])
def test_an_unstamped_database_has_no_stamp(tmp_path: Path, content: bytes) -> None:
    (tmp_path / "x.db").write_bytes(content)

    assert backbone.stamped(tmp_path / "x.db") is None
    assert backbone.stamped(tmp_path / "absent.db") is None


def test_changed_unions_the_diff_and_untracked_files() -> None:
    assert backbone.changed(
        Path("/repo"),
        sha=_SHA,
        output=_git(diff="b.py\0a.py\0", others="new.py\0a.py\0"),
    ) == ["a.py", "b.py", "new.py"]


@pytest.mark.parametrize(("diff", "others"), [(None, ""), ("", None)])
def test_changed_is_none_when_git_cannot_answer(
    diff: str | None,
    others: str | None,
) -> None:
    """A backbone commit this clone never fetched makes `git diff` fail."""
    assert (
        backbone.changed(Path("/repo"), sha=_SHA, output=_git(diff=diff, others=others))
        is None
    )


def test_changed_diffs_the_worktree_against_the_backbone_commit() -> None:
    seen: list[list[str]] = []

    def output(argv: Sequence[str]) -> str:
        seen.append(list(argv))
        return ""

    backbone.changed(Path("/repo"), sha=_SHA, output=output)

    assert seen == [
        ["git", "-C", "/repo", "diff", "--name-only", "--no-renames", "-z", _SHA, "--"],
        ["git", "-C", "/repo", "ls-files", "--others", "--exclude-standard", "-z"],
    ]


def test_the_fingerprint_moves_with_content_deletion_and_commit(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("one")
    first = backbone.fingerprint(tmp_path, sha=_SHA, paths=["a.py"])

    assert first == backbone.fingerprint(tmp_path, sha=_SHA, paths=["a.py"])
    assert first != backbone.fingerprint(tmp_path, sha="b" * 40, paths=["a.py"])
    (tmp_path / "a.py").write_text("two")
    assert first != backbone.fingerprint(tmp_path, sha=_SHA, paths=["a.py"])
    second = backbone.fingerprint(tmp_path, sha=_SHA, paths=["a.py"])
    (tmp_path / "a.py").unlink()
    assert second != backbone.fingerprint(tmp_path, sha=_SHA, paths=["a.py"])


def test_the_fingerprint_separates_a_path_from_its_content(tmp_path: Path) -> None:
    """Concatenating name and bytes would make `a` + `b.py` equal `ab` + `.py`."""
    (tmp_path / "ab").write_text(".py")
    (tmp_path / "a").write_text("b.py")

    assert backbone.fingerprint(tmp_path, sha=_SHA, paths=["ab"]) != (
        backbone.fingerprint(tmp_path, sha=_SHA, paths=["a"])
    )


def test_an_overlay_indexes_only_the_changed_files(tmp_path: Path) -> None:
    caches, root = tmp_path / "caches", tmp_path / "root"
    _published(caches)
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("def f(): pass\n")
    (root / "pkg" / "b.py").write_text("def g(): pass\n")
    (root / "pkg" / "c.py").write_text("def h(): pass\n")
    calls: list[Path] = []
    seen: list[list[str]] = []

    problem = backbone.build_overlay(
        root,
        caches=caches,
        output=_git(diff="pkg/a.py\0pkg/c.py\0gone.py\0", seen=seen),
        index=_indexer(calls),
    )

    overlay = backbone.overlay_database(caches, root=root)
    changed = ["gone.py", "pkg/a.py", "pkg/c.py"]
    assert problem == ""
    assert len(calls) == 1
    # Indexed beside the overlay, so the finished index renames in place: a
    # rename across filesystems fails, and /tmp is often one.
    assert calls[0].parent.parent == overlay.parent
    assert calls[0].parent.name.startswith("backbone-")
    assert calls[0].name == "tree"
    assert seen[0][:3] == ["git", "-C", str(root)]
    assert _SHA in seen[0]
    assert backbone.stamped(overlay) == backbone.Stamp(
        sha=_SHA,
        fingerprint=backbone.fingerprint(root, sha=_SHA, paths=changed),
        paths=tuple(changed),
    )
    with sqlite3.connect(overlay) as connection:
        assert connection.execute(
            "SELECT file_path FROM nodes ORDER BY file_path",
        ).fetchall() == [("pkg/a.py",), ("pkg/c.py",)]
    assert overlay.stat().st_mode & 0o777 == 0o664
    assert [path.name for path in overlay.parent.iterdir()] == ["codegraph.db"]


def test_an_unchanged_worktree_reuses_its_overlay(tmp_path: Path) -> None:
    """Every commit re-indexing the same files would cost ~1s for nothing."""
    caches, root = tmp_path / "caches", tmp_path / "root"
    _published(caches)
    root.mkdir()
    (root / "a.py").write_text("x")
    calls: list[Path] = []
    git = _git(diff="a.py\0")

    counts = []
    for content in ("x", "x", "y", "y"):
        (root / "a.py").write_text(content)
        assert (
            backbone.build_overlay(
                root,
                caches=caches,
                output=git,
                index=_indexer(calls),
            )
            == ""
        )
        counts.append(len(calls))

    assert counts == [1, 1, 2, 2]


def test_a_worktree_matching_the_backbone_needs_no_index(tmp_path: Path) -> None:
    caches, root = tmp_path / "caches", tmp_path / "root"
    _published(caches)
    root.mkdir()
    calls: list[Path] = []

    assert (
        backbone.build_overlay(
            root,
            caches=caches,
            output=_git(),
            index=_indexer(calls),
        )
        == ""
    )
    assert calls == []
    assert backbone.stamped(backbone.overlay_database(caches, root=root)) == (
        backbone.Stamp(
            sha=_SHA,
            fingerprint=backbone.fingerprint(root, sha=_SHA, paths=[]),
            paths=(),
        )
    )


def test_an_overlay_needs_a_backbone(tmp_path: Path) -> None:
    problem = backbone.build_overlay(
        tmp_path,
        caches=tmp_path / "caches",
        output=_git(),
        index=_indexer([]),
    )

    assert "No CodeGraph backbone" in problem


def test_an_unfetched_backbone_commit_is_reported(tmp_path: Path) -> None:
    caches = tmp_path / "caches"
    _published(caches)

    problem = backbone.build_overlay(
        tmp_path,
        caches=caches,
        output=_git(diff=None),
        index=_indexer([]),
    )

    assert "git fetch origin" in problem
    assert _SHA[:12] in problem


def test_a_failed_index_publishes_nothing(tmp_path: Path) -> None:
    caches, root = tmp_path / "caches", tmp_path / "root"
    _published(caches)
    root.mkdir()
    (root / "a.py").write_text("x")

    problem = backbone.build_overlay(
        root,
        caches=caches,
        output=_git(diff="a.py\0"),
        index=_failing,
    )

    assert "exit 3" in problem
    assert not backbone.overlay_database(caches, root=root).exists()
    assert list(backbone.overlay_database(caches, root=root).parent.iterdir()) == []


def _fresh(tmp_path: Path, *, diff: str = "a.py\0") -> tuple[Path, Path, Path]:
    caches, root = tmp_path / "caches", tmp_path / "root"
    _published(caches)
    root.mkdir(exist_ok=True)
    (root / "a.py").write_text("x")
    assert (
        backbone.build_overlay(
            root,
            caches=caches,
            output=_git(diff=diff),
            index=_indexer([]),
        )
        == ""
    )
    return caches, root, backbone.overlay_database(caches, root=root)


def test_a_fresh_overlay_is_used(tmp_path: Path) -> None:
    caches, root, overlay = _fresh(tmp_path)
    seen: list[list[str]] = []

    assert backbone.overlay_problem(
        caches,
        root=root,
        output=_git(diff="a.py\0", seen=seen),
    ) == ("", overlay)
    assert seen[0][:3] == ["git", "-C", str(root)]
    assert _SHA in seen[0]


def test_an_empty_overlay_leaves_the_backbone_alone(tmp_path: Path) -> None:
    caches, root, _ = _fresh(tmp_path, diff="")

    assert backbone.overlay_problem(caches, root=root, output=_git()) == ("", None)


def test_an_edited_file_makes_the_overlay_stale(tmp_path: Path) -> None:
    """`SKIP=codegraph-sync` must not let a gate trust last commit's overlay."""
    caches, root, _ = _fresh(tmp_path)
    (root / "a.py").write_text("y")

    problem, overlay = backbone.overlay_problem(
        caches,
        root=root,
        output=_git(diff="a.py\0"),
    )

    assert "stale" in problem
    assert overlay is None


def test_a_newly_changed_file_makes_the_overlay_stale(tmp_path: Path) -> None:
    caches, root, _ = _fresh(tmp_path)

    problem, _ = backbone.overlay_problem(
        caches,
        root=root,
        output=_git(diff="a.py\0b.py\0"),
    )

    assert "stale" in problem


def test_a_republished_backbone_makes_the_overlay_stale(tmp_path: Path) -> None:
    caches, root, _ = _fresh(tmp_path)
    backbone.backbone_database(caches).unlink()
    _published(caches, sha="b" * 40)

    problem, overlay = backbone.overlay_problem(
        caches,
        root=root,
        output=_git(diff="a.py\0"),
    )

    assert "bbbbbbbbbbbb" in problem
    assert overlay is None


def test_a_missing_backbone_or_overlay_is_a_problem(tmp_path: Path) -> None:
    caches, root = tmp_path / "caches", tmp_path / "root"

    assert (
        "No CodeGraph backbone"
        in backbone.overlay_problem(caches, root=root, output=_git())[0]
    )
    _published(caches)
    assert "No overlay" in backbone.overlay_problem(caches, root=root, output=_git())[0]


def test_an_unfetched_commit_is_a_problem_for_readers_too(tmp_path: Path) -> None:
    caches, root, _ = _fresh(tmp_path)

    assert (
        "git fetch origin"
        in backbone.overlay_problem(
            caches,
            root=root,
            output=_git(diff=None),
        )[0]
    )


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("nodes", "qualified_name"),
        ("edges", "target"),
        ("unresolved_refs", "name_tail"),
    ],
)
def test_a_renamed_column_is_reported_not_queried(
    tmp_path: Path,
    table: str,
    column: str,
) -> None:
    """An unpinned CodeGraph upgrade must degrade the gate, not mis-select."""
    caches = tmp_path / "caches"
    database = _published(caches)
    with sqlite3.connect(database) as connection:
        connection.execute(f"ALTER TABLE {table} RENAME COLUMN {column} TO renamed")

    problem, overlay = backbone.overlay_problem(caches, root=tmp_path, output=_git())

    assert f"{table}.{column}" in problem
    assert str(database) in problem
    assert overlay is None


def test_a_renamed_overlay_column_is_reported(tmp_path: Path) -> None:
    caches, root, overlay = _fresh(tmp_path)
    with sqlite3.connect(overlay) as connection:
        connection.execute("ALTER TABLE edges RENAME COLUMN source TO renamed")

    problem, _ = backbone.overlay_problem(caches, root=root, output=_git(diff="a.py\0"))

    assert "edges.source" in problem
    assert str(overlay) in problem


def test_an_index_that_cannot_be_read_lacks_every_column(tmp_path: Path) -> None:
    (tmp_path / "x.db").write_bytes(b"garbage " * 100)

    missing = backbone._missing_columns(tmp_path / "x.db")

    assert missing[0] == "(unreadable)"
    assert "nodes.id" in missing


def test_an_unreadable_backbone_is_a_problem(tmp_path: Path) -> None:
    database = backbone.backbone_database(tmp_path)
    database.parent.mkdir(parents=True)
    database.write_bytes(b"garbage " * 100)

    assert (
        "unreadable"
        in backbone.overlay_problem(tmp_path, root=tmp_path, output=_git())[0]
    )


@pytest.mark.cli_git
def test_changed_reads_a_real_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staged, unstaged, deleted, and untracked differ; ignored files do not."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    for path, text in [
        ("kept.py", "a"),
        ("edited.py", "a"),
        ("staged.py", "a"),
        ("deleted.py", "a"),
    ]:
        (tmp_path / path).write_text(text)
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    _run_git(tmp_path, "init", "-q")
    _run_git(tmp_path, "add", ".")
    _run_git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-qm",
        "base",
    )
    sha = _run_git(tmp_path, "rev-parse", "HEAD").strip()
    (tmp_path / "edited.py").write_text("b")
    (tmp_path / "staged.py").write_text("b")
    _run_git(tmp_path, "add", "staged.py")
    (tmp_path / "deleted.py").unlink()
    (tmp_path / "new.py").write_text("b")
    (tmp_path / "ignored.py").write_text("b")

    assert backbone.changed(tmp_path, sha=sha, output=output_argv) == [
        "deleted.py",
        "edited.py",
        "new.py",
        "staged.py",
    ]


def _failing(tree: Path) -> int:
    del tree
    return 3


def _run_git(root: Path, *args: str) -> str:
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
