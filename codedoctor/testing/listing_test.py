"""The selection file must carry node ids and nothing else.

The consuming job passes every line to pytest as an argument, and one
unmatchable argument collects zero tests rather than the rest -- so a stray
banner or summary line is not a cosmetic flaw, it is a silent empty run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import textwrap

import pytest

from codedoctor.testing import listing


if TYPE_CHECKING:
    from collections.abc import Sequence


pytest_plugins = ("pytester",)

_PLUGIN: Final = "codedoctor.testing.listing"


@pytest.fixture
def isolated(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> pytest.Pytester:
    """Return a pytester whose runs load only the plugins they name.

    The installed ones (asyncio, randomly, xdist) print and warn on their own,
    and none is under test. The outer run's own destinations stay out of these.
    """
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    for name in (
        "PYTEST_COLLECT_SELECTED",
        "PYTEST_COLLECT_SELECTED_DIR",
        "PYTEST_COLLECT_TIERS",
    ):
        monkeypatch.delenv(name, raising=False)
    return pytester


def _collect(
    isolated: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    destination: Path,
    *,
    noisy: bool,
) -> pytest.RunResult:
    """Collect the tree with the plugin enabled, optionally beside a noisy one."""
    isolated.makepyfile(
        probe_test="""
        def test_one():
            pass


        def test_two():
            pass
        """,
    )
    if noisy:
        # Stands in for py-pglite, which prints a tips banner at session finish.
        isolated.makeconftest(
            """
            def pytest_collection_finish(session):
                del session
                print("===== Framework Isolation Tips =====")
                print("  pytest -m sqlalchemy -p no:django")
            """,
        )
    monkeypatch.setenv("PYTEST_COLLECT_SELECTED", str(destination))
    return isolated.runpytest_inprocess("--collect-only", "-q", "-p", _PLUGIN)


def test_every_written_line_is_a_node_id(
    isolated: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin printing banners cannot contaminate the selection file."""
    destination = isolated.path / "selected" / "pytest-global.txt"
    result = _collect(isolated, monkeypatch, destination, noisy=True)

    assert result.ret == 0, result.outlines
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines, "the plugin wrote no node ids"
    assert all("::" in line for line in lines), lines
    # The banner reached the console but not the file -- the whole point.
    assert "===== Framework Isolation Tips =====" in result.outlines
    assert not any("Isolation Tips" in line for line in lines)


def test_a_deselecting_plugin_narrows_the_file(
    isolated: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What a deselector left is what lands in the file.

    testmon deselects from `pytest_collection_modifyitems` registered
    ``trylast``, and pytest runs that hook LIFO -- so an ordinary or even a
    second ``trylast`` implementation sees the list BEFORE the deselection and
    records tests testmon excluded. Reading `session.items` at collection
    finish is ordering-free: that hook runs once every `modifyitems` has.

    The consuming job runs exactly this file, so over-recording silently
    forfeits the selection the job exists to compute.
    """
    isolated.makeconftest(
        """
        import pytest


        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(config, items):
            config.hook.pytest_deselected(items=items[1:])
            items[:] = items[:1]
        """,
    )
    destination = isolated.path / "selected.txt"
    result = _collect(isolated, monkeypatch, destination, noisy=False)

    assert result.ret == 0, result.outlines
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, lines


def test_the_summary_line_is_not_written(
    isolated: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``N/M tests collected`` is console chrome, not a node id.

    This is the exact line that, passed back as an argument, collected zero
    tests and failed a required check.
    """
    destination = isolated.path / "selected.txt"
    result = _collect(isolated, monkeypatch, destination, noisy=False)

    assert result.ret == 0, result.outlines
    assert any("tests collected" in line for line in result.outlines)
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert not any("tests collected" in line for line in lines), lines
    assert len(lines) == 2, lines


def test_one_collection_writes_each_pr_pytest_tier(
    isolated: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One selector job writes every artifact the PR pytest matrix consumes."""
    isolated.makepyfile(
        probe_test="""
        import pytest


        def test_global():
            pass


        @getattr(pytest.mark, "slow")
        def test_slow():
            pass


        @getattr(pytest.mark, "ci_smoke")
        def test_smoke():
            pass


        @getattr(pytest.mark, "integration")
        def test_integration():
            pass
        """,
    )
    # Registered, as the repository registers them.
    isolated.makeini("[pytest]\nmarkers =\n    slow\n    ci_smoke\n    integration\n")
    destination = isolated.path / "selected"
    monkeypatch.setenv("PYTEST_COLLECT_SELECTED_DIR", str(destination))

    result = isolated.runpytest_inprocess(
        "--collect-only",
        "-q",
        "-p",
        _PLUGIN,
        "-m",
        "not cluster and not cuda and not performance",
    )

    assert result.ret == 0, result.outlines
    assert (destination / "pytest-global.txt").read_text(encoding="utf-8") == (
        "probe_test.py::test_global\n"
    )
    assert (destination / "pytest-slow-global.txt").read_text(encoding="utf-8") == (
        "probe_test.py::test_slow\n"
    )
    assert (destination / "pytest-ci-smoke-global.txt").read_text(encoding="utf-8") == (
        "probe_test.py::test_smoke\n"
    )
    assert (destination / "pytest-integration-global.txt").read_text(
        encoding="utf-8",
    ) == "probe_test.py::test_integration\n"
    assert (destination / "selected-counts.json").read_text() == (
        '{"pytest-global": 1, "pytest-slow-global": 1, '
        '"pytest-ci-smoke-global": 1, "pytest-integration-global": 1}\n'
    )


def test_a_real_collection_is_narrowed_to_the_listing(
    isolated: pytest.Pytester,
) -> None:
    for relative in ("keep/kept_test.py", "keep/dropped_test.py", "skip/other_test.py"):
        (isolated.path / relative).parent.mkdir(parents=True, exist_ok=True)
        (isolated.path / relative).write_text(
            "def test_it():\n    pass\n",
            encoding="utf-8",
        )
    # A conftest that cannot import proves its directory is never entered.
    (isolated.path / "skip" / "conftest.py").write_text(
        "import missing_module\n",
        encoding="utf-8",
    )
    (isolated.path / "files.txt").write_text("keep/kept_test.py\n", encoding="utf-8")

    ran = _narrowed(isolated, listing_file="files.txt")

    assert ran.ret == 0, ran.outlines
    assert ran.passed == ["keep/kept_test.py::test_it"]


def test_the_shards_of_a_split_file_run_each_of_its_tests_once(
    isolated: pytest.Pytester,
) -> None:
    isolated.makepyfile(
        split_test="".join(
            f"def test_{name}():\n    pass\n" for name in ("a", "b", "c", "new")
        ),
    )
    ran: list[set[str]] = []
    # One listing per shard, as each CI shard is handed its own.
    for shard, lines in enumerate(
        (
            "split_test.py\n!split_test.py::test_b\n!split_test.py::test_c\n",
            "split_test.py::test_b\n",
            "split_test.py::test_c\nsplit_test.py::test_gone\n",
        ),
    ):
        (isolated.path / f"shard{shard}.txt").write_text(lines, encoding="utf-8")
        shard_run = _narrowed(isolated, listing_file=f"shard{shard}.txt")
        assert shard_run.ret == 0, shard_run.outlines
        ran.append({nodeid.partition("::")[2] for nodeid in shard_run.passed})

    # The whole-file shard keeps the test no listing names; renamed ones vanish.
    assert ran == [{"test_a", "test_new"}, {"test_b"}, {"test_c"}]


def test_a_pytest_a_narrowed_test_starts_collects_its_own_tests(
    isolated: pytest.Pytester,
) -> None:
    """The listing narrows one pytest, not every pytest its tests start.

    Tests that run pytest over a tree of their own inherit the environment, so
    a listing handed over as a variable narrowed their run too: it collected
    nothing and exited 5. The inner run is a real child, which is what inherits.
    """
    inner = isolated.mkdir("inner")
    (inner / "inner_test.py").write_text("def test_it():\n    pass\n", encoding="utf-8")
    outer = isolated.mkdir("outer")
    (outer / "outer_test.py").write_text(
        textwrap.dedent(
            f"""
            import subprocess
            import sys

            def test_inner_pytest_runs_its_test():
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                    capture_output=True,
                    check=False,
                    cwd={str(inner)!r},
                    text=True,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                assert "1 passed" in result.stdout
            """,
        ),
        encoding="utf-8",
    )
    (outer / "files.txt").write_text("outer/outer_test.py\n", encoding="utf-8")

    ran = _narrowed(isolated, listing_file="outer/files.txt")

    assert ran.ret == 0, ran.outlines
    assert ran.passed == ["outer/outer_test.py::test_inner_pytest_runs_its_test"]


def _narrowed(isolated: pytest.Pytester, *, listing_file: str) -> _Narrowed:
    """Run pytest over the tree narrowed to ``listing_file``, as a CI job does."""
    result = isolated.runpytest_inprocess(
        "-v",
        "-p",
        "no:cacheprovider",
        "-p",
        _PLUGIN,
        f"--test-files={listing_file}",
    )
    return _Narrowed(
        ret=int(result.ret),
        outlines=result.outlines,
        passed=[
            line.partition(" ")[0]
            for line in result.outlines
            if line.endswith("PASSED") or " PASSED " in line
        ],
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Narrowed:
    """What a narrowed run did: its status, its console, and the tests it passed."""

    ret: int
    outlines: list[str]
    passed: list[str]


class _Mark:
    def __init__(self, name: str) -> None:
        self.name = name


class _Item:
    """The two attributes the writer reads from a ``pytest.Item``."""

    def __init__(self, nodeid: str, *markers: str) -> None:
        self.nodeid = nodeid
        self._markers = [_Mark(name) for name in markers]

    def iter_markers(self) -> list[_Mark]:
        return self._markers


class _Session:
    def __init__(self, items: list[_Item]) -> None:
        self.items = items


def test_each_tier_receives_exactly_its_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every exclusion is exercised alone, so dropping any one name from either
    # tier's exclusion set admits exactly the test that names it.
    items = [
        _Item("t.py::plain"),
        _Item("t.py::cuda", "cuda"),
        _Item("t.py::cluster", "cluster"),
        _Item("t.py::perf", "performance"),
        _Item("t.py::integ", "integration"),
        _Item("t.py::smoke", "ci_smoke"),
        _Item("t.py::slow", "slow"),
        _Item("t.py::slow_cuda", "slow", "cuda"),
        _Item("t.py::slow_cluster", "slow", "cluster"),
        _Item("t.py::slow_perf", "slow", "performance"),
        _Item("t.py::slow_integ", "slow", "integration"),
        _Item("t.py::smoke_slow", "ci_smoke", "slow"),
    ]
    out = tmp_path / "not" / "yet" / "made"
    monkeypatch.delenv("PYTEST_COLLECT_SELECTED", raising=False)
    monkeypatch.setenv("PYTEST_COLLECT_SELECTED_DIR", str(out))

    listing.pytest_collection_finish(
        _Session(items),
    )

    def tier(name: str) -> list[str]:
        return (out / f"{name}.txt").read_bytes().decode("utf-8").splitlines()

    assert tier("pytest-global") == ["t.py::plain"]
    assert tier("pytest-slow-global") == ["t.py::slow", "t.py::smoke_slow"]
    assert tier("pytest-ci-smoke-global") == ["t.py::smoke", "t.py::smoke_slow"]
    assert tier("pytest-integration-global") == [
        "t.py::integ",
        "t.py::slow_integ",
    ]
    assert (out / "selected-counts.json").read_bytes().decode("utf-8") == (
        '{"pytest-global": 1, "pytest-slow-global": 2, '
        '"pytest-ci-smoke-global": 2, "pytest-integration-global": 2}\n'
    )


def test_one_file_receives_every_selected_node_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "two" / "levels" / "selected.txt"
    monkeypatch.setenv("PYTEST_COLLECT_SELECTED", str(destination))
    monkeypatch.delenv("PYTEST_COLLECT_SELECTED_DIR", raising=False)
    # Non-ASCII so a locale-default encoding would write different bytes.
    nodeids = ["a.py::x[ü]", "b.py::y"]

    listing.pytest_collection_finish(
        _Session([_Item(nodeids[0], "slow"), _Item(nodeids[1])]),
    )

    assert destination.read_bytes() == "a.py::x[ü]\nb.py::y\n".encode()


def test_no_variable_writes_nothing_in_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTEST_COLLECT_SELECTED", raising=False)
    monkeypatch.delenv("PYTEST_COLLECT_SELECTED_DIR", raising=False)
    monkeypatch.delenv("PYTEST_COLLECT_TIERS", raising=False)
    monkeypatch.chdir(tmp_path)

    listing.pytest_collection_finish(
        _Session([_Item("a::b")]),
    )

    assert list(tmp_path.iterdir()) == []


def test_the_tier_file_names_each_file_once_per_tier_it_holds_tests_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "made" / "tiers.tsv"
    monkeypatch.setenv("PYTEST_COLLECT_TIERS", str(destination))
    monkeypatch.delenv("PYTEST_COLLECT_SELECTED", raising=False)
    monkeypatch.delenv("PYTEST_COLLECT_SELECTED_DIR", raising=False)
    items = [
        _Item("b/two_test.py::fast"),
        _Item("b/two_test.py::fast[ü]"),
        _Item("a/one_test.py::slow", "slow"),
        _Item("a/one_test.py::Case::golden", "golden"),
        _Item("a/one_test.py::slow_golden", "slow", "golden"),
        _Item("c/three_test.py::gpu", "cuda"),
    ]

    listing.pytest_collection_finish(_Session(items))

    # A golden is no tier of its own: it runs in whichever tier holds it.
    assert destination.read_bytes().decode("utf-8").splitlines() == [
        "pytest-global\ta/one_test.py",
        "pytest-global\tb/two_test.py",
        "pytest-slow-global\ta/one_test.py",
    ]


def test_a_listing_keeps_only_its_files_and_their_directories(tmp_path: Path) -> None:
    files = tmp_path / "files.txt"
    files.write_text("pkg/kept_test.py\n", encoding="utf-8")
    config = _Config(tmp_path / "root", test_files=files)
    root = config.rootpath
    (root / "pkg").mkdir(parents=True)
    (root / "other").mkdir()

    assert _ignored(root / "pkg", config=config) is None
    assert _ignored(root / "pkg/kept_test.py", config=config) is None
    assert _ignored(root / "pkg/test_dropped.py", config=config) is True
    assert _ignored(root / "pkg/dropped_test.py", config=config) is True
    assert _ignored(root / "pkg/conftest.py", config=config) is None
    assert _ignored(root / "other", config=config) is True


def test_the_listing_is_a_path_option() -> None:
    parser = _Parser()
    listing.pytest_addoption(parser)
    assert parser.types == {"--test-files": Path}
    assert parser.helps == {
        "--test-files": "Collect only the test files this file lists, one per line.",
    }


def test_no_listing_ignores_nothing(tmp_path: Path) -> None:
    config = _Config(tmp_path, test_files=None)
    assert (
        listing.pytest_ignore_collect(
            collection_path=tmp_path / "a_test.py",
            config=config,
        )
        is None
    )


def test_a_split_file_is_collected_wherever_a_listing_names_its_tests(
    tmp_path: Path,
) -> None:
    files = tmp_path / "files.txt"
    files.write_text("pkg/split_test.py::test_b\n", encoding="utf-8")
    config = _Config(tmp_path / "root", test_files=files)
    root = config.rootpath
    (root / "pkg").mkdir(parents=True)

    assert _ignored(root / "pkg", config=config) is None
    assert _ignored(root / "pkg/split_test.py", config=config) is None
    assert _ignored(root / "pkg/other_test.py", config=config) is True


def test_a_split_listing_deselects_the_tests_it_gives_away(tmp_path: Path) -> None:
    items = [_Item(f"s_test.py::{name}") for name in ("a", "b", "new")]
    owner = tmp_path / "owner.txt"
    owner.write_text("s_test.py\n!s_test.py::b\n", encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text("s_test.py::b\ns_test.py::gone\n", encoding="utf-8")

    kept: list[list[str]] = []
    for files in (owner, other):
        config = _ItemConfig(tmp_path, test_files=files)
        narrowed = list(items)
        listing.pytest_collection_modifyitems(config, narrowed)
        kept.append([item.nodeid for item in narrowed])
        assert [item.nodeid for item in config.hook.deselected] == [
            item.nodeid for item in items if item not in narrowed
        ]

    # The shard holding the file whole keeps the test no listing names.
    assert kept == [["s_test.py::a", "s_test.py::new"], ["s_test.py::b"]]


def test_a_class_test_is_matched_by_its_file(tmp_path: Path) -> None:
    """``s_test.py::Case::a`` lives in ``s_test.py``, not in ``s_test.py::Case``."""
    files = tmp_path / "files.txt"
    files.write_text("s_test.py::Case::a\n", encoding="utf-8")
    config = _ItemConfig(tmp_path, test_files=files)
    items = [_Item("s_test.py::Case::a"), _Item("s_test.py::Case::b")]

    listing.pytest_collection_modifyitems(config, items)

    assert [item.nodeid for item in items] == ["s_test.py::Case::a"]


def test_a_test_named_without_parameters_keeps_every_parametrization(
    tmp_path: Path,
) -> None:
    """The dependency graph names a test as defined, never as each case collects."""
    files = tmp_path / "files.txt"
    files.write_text(
        "s_test.py::p\ns_test.py::q[1]\nt_test.py\n!t_test.py::r\n",
        encoding="utf-8",
    )
    config = _ItemConfig(tmp_path, test_files=files)
    items = [
        _Item(nodeid)
        for nodeid in (
            "s_test.py::p[1]",
            "s_test.py::p[2]",
            "s_test.py::p[a[1]]",
            "s_test.py::pp",
            "s_test.py::q[1]",
            "s_test.py::q[2]",
            "t_test.py::r[1]",
            "t_test.py::rest",
        )
    ]

    listing.pytest_collection_modifyitems(config, items)

    assert [item.nodeid for item in items] == [
        "s_test.py::p[1]",
        "s_test.py::p[2]",
        "s_test.py::p[a[1]]",
        "s_test.py::q[1]",
        "t_test.py::rest",
    ]


def test_a_files_only_listing_deselects_nothing(tmp_path: Path) -> None:
    files = tmp_path / "files.txt"
    files.write_text("a_test.py\n", encoding="utf-8")
    config = _ItemConfig(tmp_path, test_files=files)
    items = [_Item("a_test.py::one"), _Item("a_test.py::two")]

    listing.pytest_collection_modifyitems(config, items)

    assert len(items) == 2
    assert config.hook.deselected == []


def test_no_listing_keeps_every_test_of_a_split_file(tmp_path: Path) -> None:
    """Without ``--test-files`` the filter is inert, whatever the nodeids say."""
    config = _ItemConfig(tmp_path, test_files=None)
    items = [_Item("s_test.py::a"), _Item("s_test.py::b[1]")]

    listing.pytest_collection_modifyitems(config, items)

    assert [item.nodeid for item in items] == ["s_test.py::a", "s_test.py::b[1]"]
    assert config.hook.deselected == []


def _ignored(path: Path, *, config: _Config) -> bool | None:
    return listing.pytest_ignore_collect(
        collection_path=path,
        config=config,
    )


class _Parser:
    """Records the type and help of each option registered with ``addoption``."""

    def __init__(self) -> None:
        self.types: dict[str, object] = {}
        self.helps: dict[str, object] = {}

    def addoption(self, *names: str, **attrs: object) -> None:
        self.types.update(dict.fromkeys(names, attrs.get("type")))
        self.helps.update(dict.fromkeys(names, attrs.get("help")))


class _Config:
    """The parts of ``pytest.Config`` the collection filter reads."""

    def __init__(self, rootpath: Path, *, test_files: Path | None) -> None:
        self.rootpath = rootpath
        self._test_files = test_files

    def getini(self, name: str) -> list[str]:
        assert name == "python_files", name
        return ["*_test.py", "test_*.py"]

    def getoption(self, name: str) -> Path | None:
        assert name == "test_files", name
        return self._test_files


class _Named(Protocol):
    @property
    def nodeid(self) -> str: ...


class _Hooks:
    """Records what ``pytest_deselected`` was handed."""

    def __init__(self) -> None:
        self.deselected: list[_Named] = []

    def pytest_deselected(self, *, items: Sequence[_Named]) -> None:
        self.deselected.extend(items)


class _ItemConfig(_Config):
    """The parts of ``pytest.Config`` the item filter reads."""

    def __init__(self, rootpath: Path, *, test_files: Path | None) -> None:
        super().__init__(rootpath, test_files=test_files)
        self.hook = _Hooks()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
