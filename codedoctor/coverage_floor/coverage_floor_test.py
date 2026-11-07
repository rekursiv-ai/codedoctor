"""Tests for the per-file and per-directory coverage floor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import argparse
import sys

from treekle import from_plain

import coverage
import pytest
import yaml

from codedoctor.coverage_floor import coverage_floor
from codedoctor.lib.testing.checks import assert_help_without_docstring


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_THIS: Final = Path(__file__).resolve()

_SOURCE: Final = "def f(x):\n    if x:\n        return 1\n    return 2\n"

# ``f(1)`` runs every line of ``_SOURCE`` but its last.
_CALLED: Final = [1, 2, 3]
_ALL_TWENTY: Final = list(range(1, 21))


def _tree(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


# Recorded, not traced: a second tracer started in this process replaces the suite's
# own collector, which then misses every line the remaining tests run.
# The empty context is import; a test's is its node id and phase, as pytest-cov names
# it.
def _measured(
    tmp_path: Path,
    runs: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    data: Path,
) -> None:
    """Write ``data`` as the ``pytest`` gate's run: per context, each module's lines."""
    recorded = coverage.CoverageData(basename=str(data))
    for context, lines in runs.items():
        recorded.set_context(context)
        recorded.add_lines(
            {
                str((tmp_path / path).resolve()): list(run)
                for path, run in lines.items()
            },
        )
    recorded.write()


def _main(
    monkeypatch: pytest.MonkeyPatch,
    *args: str,
    data: str = ".coverage",
) -> int:
    monkeypatch.setattr(sys, "argv", ["coverage_floor", "--data", data, *args])
    return coverage_floor.main()


def _owners(sources: Sequence[str]) -> dict[str, list[str]]:
    """Owners whose tests ran nothing: only import-time lines count."""
    return {source: [] for source in sources}


def test_sidecars_pair_floored_sources_with_every_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tree(
        tmp_path,
        dict.fromkeys(
            [
                "app/a.py",
                "app/a_test.py",
                "pkg/b.py",
                "pkg/test_b.py",
                "pkg/c.py",
                "pkg/tests/c_test.py",
                "cfg/untested.py",
                "scratch/d.py",
                "scratch/d_test.py",
            ],
            "",
        ),
    )
    monkeypatch.chdir(tmp_path)

    assert coverage_floor.sidecars(
        [
            "app/a.py",
            "app/a_test.py",
            "pkg/b.py",
            "pkg/c.py",
            "cfg/untested.py",
            "scratch/d.py",
            "pkg/conftest.py",
        ],
        floored=("cfg/", "pkg/", "app/"),
    ) == (["app/a.py", "pkg/b.py", "pkg/c.py"], ["cfg/untested.py"])


def test_a_path_coverage_omits_is_never_floored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tree(
        tmp_path,
        {
            "pyproject.toml": "[tool.coverage.run]\nomit = ['*/stubs/*']\n",
            "stubs/lib.py": "",
            "pkg/m.py": "",
        },
    )
    monkeypatch.chdir(tmp_path)

    assert coverage_floor.sidecars(["stubs/lib.py", "pkg/m.py"], floored=("",)) == (
        [],
        ["pkg/m.py"],
    )


def test_a_conftest_or_test_named_module_is_never_a_floored_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tree(tmp_path, dict.fromkeys(["pkg/conftest.py", "pkg/test_x.py"], ""))
    monkeypatch.chdir(tmp_path)

    assert coverage_floor.sidecars(
        ["pkg/conftest.py", "pkg/test_x.py"],
        floored=("pkg/",),
    ) == ([], [])


@pytest.mark.parametrize(
    "case",
    [
        (["pkg/m.py"], 1, "coverage-floor: pkg/m.py has no sidecar test\n"),
        (
            [
                "pkg/__init__.py",
                "pkg/__main__.py",
                "pkg/gone.py",
                "pkg/third_party/vendored.py",
                "pkg/testdata/fixture.py",
            ],
            0,
            "",
        ),
    ],
    ids=["untested", "exempt"],
)
def test_a_floored_source_without_a_sidecar_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: tuple[list[str], int, str],
) -> None:
    paths, status, shown = case
    _tree(
        tmp_path,
        {
            # Coverage omits it, as the repository's own configuration does.
            ".coveragerc": "[run]\nomit = */testdata/*\n",
            "pkg/__init__.py": "X = 1\n",
            "pkg/__main__.py": "X = 1\n",
            "pkg/m.py": _SOURCE,
            "pkg/third_party/vendored.py": _SOURCE,
            "pkg/testdata/fixture.py": _SOURCE,
        },
    )
    monkeypatch.chdir(tmp_path)

    assert _main(monkeypatch, "--floor", "pkg/=100", "--", *paths) == status
    assert capsys.readouterr().out == shown


def test_a_run_floors_each_source_by_its_own_tests_lines_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In ``pytest``'s run another file's test may reach every line; it never counts."""
    _tree(tmp_path, {"pkg/m.py": _SOURCE, "pkg/m_test.py": "", "pkg/o_test.py": ""})
    monkeypatch.chdir(tmp_path)
    _measured(
        tmp_path,
        {
            "": {"pkg/m.py": [1]},
            "pkg/m_test.py::test_f|run": {"pkg/m.py": [2, 3]},
            "pkg/o_test.py::test_g|run": {"pkg/m.py": [1, 2, 3, 4]},
        },
        data=tmp_path / ".coverage",
    )

    assert _main(monkeypatch, "--floor", "pkg/=100", "--", "pkg/m.py") == 1
    assert capsys.readouterr().out == (
        "coverage-floor: pkg/m.py at 75% is below 100%; missed 4\n"
    )


def test_a_split_test_counts_as_the_modules_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``m_pglite_test.py`` is ``m``'s, so the database tier's lines count."""
    _tree(
        tmp_path,
        {"pkg/m.py": _SOURCE, "pkg/m_test.py": "", "pkg/m_pglite_test.py": ""},
    )
    monkeypatch.chdir(tmp_path)
    _measured(
        tmp_path,
        {
            "pkg/m_test.py::test_f|run": {"pkg/m.py": _CALLED},
            "pkg/m_pglite_test.py::test_db|run": {"pkg/m.py": [4]},
        },
        data=tmp_path / ".coverage",
    )

    assert _main(monkeypatch, "--floor", "pkg/=100", "--", "pkg/m.py") == 0


@pytest.mark.parametrize("prefix", ["pkg/", ""], ids=["prefix", "everything"])
def test_floors_come_from_the_pyproject_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prefix: str,
) -> None:
    """A bare run floors what ``[tool.coverage-floor]`` names, as the hook runs it."""
    _tree(
        tmp_path,
        {
            "pyproject.toml": f"[tool.coverage-floor]\nfloor = ['{prefix}=100']\n",
            "pkg/m.py": _SOURCE,
            "pkg/m_test.py": "",
        },
    )
    monkeypatch.chdir(tmp_path)
    _measured(
        tmp_path,
        {"pkg/m_test.py::test_f|run": {"pkg/m.py": _CALLED}},
        data=tmp_path / ".coverage",
    )

    assert _main(monkeypatch, "pkg/m.py") == 1
    assert capsys.readouterr().out == (
        "coverage-floor: pkg/m.py at 75% is below 100%; missed 4\n"
    )


def test_one_run_reports_both_untested_and_under_covered_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An untested source must not hide another source's shortfall."""
    _tree(
        tmp_path,
        {"pkg/bare.py": _SOURCE, "pkg/m.py": _SOURCE, "pkg/m_test.py": ""},
    )
    monkeypatch.chdir(tmp_path)
    _measured(
        tmp_path,
        {"pkg/m_test.py::test_f|run": {"pkg/m.py": _CALLED}},
        data=tmp_path / ".coverage",
    )

    assert (
        _main(monkeypatch, "--floor", "pkg/=100", "--", "pkg/bare.py", "pkg/m.py") == 1
    )
    assert capsys.readouterr().out == (
        "coverage-floor: pkg/bare.py has no sidecar test\n"
        "coverage-floor: pkg/m.py at 75% is below 100%; missed 4\n"
    )


def test_main_floors_by_directory_when_asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pooled with ``n.py``'s 20 lines, ``m.py``'s one miss leaves ``pkg`` at 96%."""
    _tree(
        tmp_path,
        {
            "pkg/m.py": _SOURCE,
            "pkg/m_test.py": "",
            "pkg/n.py": "".join(f"X{i} = {i}\n" for i in range(20)),
            "pkg/n_test.py": "",
        },
    )
    monkeypatch.chdir(tmp_path)
    _measured(
        tmp_path,
        {
            "pkg/m_test.py::t|run": {"pkg/m.py": _CALLED},
            "": {"pkg/n.py": _ALL_TWENTY},
        },
        data=tmp_path / ".coverage",
    )
    args = ("--scope", "directory", "--", "pkg/m.py", "pkg/n.py")

    assert _main(monkeypatch, "--floor", "pkg/=95", *args) == 0
    assert _main(monkeypatch, "--floor", "pkg/=97", *args) == 1
    assert capsys.readouterr().out == (
        "coverage-floor: pkg at 96% is below 97%; missed pkg/m.py: 4\n"
    )


def test_main_reads_its_paths_from_the_gates_argfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--all-files`` as argv overflowed macOS's ``ARG_MAX``; the gate hands an @argfile."""
    _tree(tmp_path, {"pkg/m.py": _SOURCE, "coverage.args": "pkg/m.py\n"})
    monkeypatch.chdir(tmp_path)

    assert _main(monkeypatch, "--floor", "pkg/=100", "--", "@coverage.args") == 1
    assert capsys.readouterr().out == "coverage-floor: pkg/m.py has no sidecar test\n"


@pytest.mark.parametrize("skip", ["", "pytest"])
def test_with_nothing_floored_changed_the_floor_passes_with_or_without_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    skip: str,
) -> None:
    """No floored source changed: no data is needed, and none is missing."""
    _tree(tmp_path, {"docs/x.py": _SOURCE})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SKIP", skip)

    assert _main(monkeypatch, "--floor", "pkg/=100", "--", "docs/x.py") == 0


@pytest.mark.parametrize(
    "case",
    [
        (
            "pytest",
            {"pkg/m_test.py": ""},
            0,
            "coverage-floor: skipped with the pytest hook; nothing floored\n",
        ),
        (
            "ruff-check,superseded-at-pre-push",
            {"pkg/m_test.py": ""},
            0,
            "coverage-floor: skipped with the pytest hook; nothing floored\n",
        ),
        ("pytest", {}, 1, "coverage-floor: pkg/m.py has no sidecar test\n"),
        (
            "",
            {"pkg/m_test.py": ""},
            1,
            "coverage-floor: the pytest hook did not run; run it with this one\n",
        ),
    ],
    ids=["skipped", "skipped-by-alias", "untested", "pytest-never-ran"],
)
def test_without_pytests_data_the_floor_passes_only_where_pytest_was_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: tuple[str, dict[str, str], int, str],
) -> None:
    """An untested source still fails: that needs no run.

    With no ``SKIP`` the floor ran alone, measuring nothing, and fails rather than
    pass on nothing.
    """
    skip, files, status, shown = case
    _tree(tmp_path, {"pkg/m.py": _SOURCE, **files})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SKIP", skip)

    assert _main(monkeypatch, "--floor", "pkg/=100", "--", "pkg/m.py") == status
    assert capsys.readouterr().out == shown


@pytest.mark.parametrize(
    ("hand_off", "reason"),
    [("pytest exited 2\n", "pytest exited 2"), ("", "an empty hand-off")],
)
def test_a_pytest_run_that_measured_nothing_fails_the_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hand_off: str,
    reason: str,
) -> None:
    """``pytest`` failed: its hand-off says so, and the floor must not pass."""
    _tree(tmp_path, {"pkg/m.py": _SOURCE, "pkg/m_test.py": "", "data": hand_off})
    monkeypatch.chdir(tmp_path)

    assert _main(monkeypatch, "--floor", "pkg/=100", "--", "pkg/m.py", data="data") == 1
    assert capsys.readouterr().out == (
        f"coverage-floor: {reason}; nothing to floor 1 file(s) by\n"
    )


def test_the_skip_tokens_name_the_configs_pytest_hook() -> None:
    """The alias a ``SKIP`` names ``pytest`` by is the config's, kept in step."""
    config = next(
        directory / ".pre-commit-config.yaml"
        for directory in _THIS.parents
        if (directory / ".pre-commit-config.yaml").is_file()
    )
    hooks = [
        hook
        for repo in from_plain(
            from_plain(yaml.safe_load(config.read_text()), dict[str, object])["repos"],
            list[dict[str, object]],
        )
        for hook in from_plain(repo.get("hooks", []), list[dict[str, object]])
    ]
    (pytest_hook,) = [hook for hook in hooks if hook["id"] == "pytest"]

    assert ("pytest", pytest_hook["alias"]) == coverage_floor.PYTEST_SKIP_TOKENS


def _units(tmp_path: Path) -> None:
    """Lay out ``pkg/m.py`` (one of four lines missed) and two fully run siblings."""
    _tree(
        tmp_path,
        {
            "pkg/m.py": _SOURCE,
            "pkg/n.py": "".join(f"X{i} = {i}\n" for i in range(20)),
            "pkg/deep/o.py": "Y = 1\n",
        },
    )


@pytest.mark.parametrize(
    "case",
    [
        ("file", {"pkg/": 100.0}, "pkg/m.py at 75% is below 100%; missed 4"),
        ("file", {"pkg/": 75.0}, ""),
        # The longer prefix wins, whichever is listed first.
        ("file", {"pkg/m": 75.0, "pkg/": 100.0}, ""),
        (
            "file",
            {"pkg/": 75.0, "pkg/m": 100.0},
            "pkg/m.py at 75% is below 100%; missed 4",
        ),
        ("directory", {"pkg/": 96.0}, ""),
        ("directory", {"pkg/": 97.0}, "pkg at 96% is below 97%; missed pkg/m.py: 4"),
    ],
    ids=[
        "file-below",
        "file-at",
        "longer-prefix-first",
        "longer-prefix-last",
        "directory-at",
        "directory-below",
    ],
)
def test_report_floors_each_measured_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: tuple[Literal["file", "directory"], dict[str, float], str],
) -> None:
    scope, floors, shown = case
    _units(tmp_path)
    monkeypatch.chdir(tmp_path)
    data = tmp_path / ".coverage"
    _measured(tmp_path, {"": {"pkg/m.py": _CALLED, "pkg/n.py": _ALL_TWENTY}}, data=data)
    sources = ["pkg/m.py", "pkg/n.py"]

    below = coverage_floor.report(
        data,
        sources=sources,
        floors=floors,
        scope=scope,
        owners=_owners(sources),
    )

    assert below is bool(shown)
    assert capsys.readouterr().out == (f"coverage-floor: {shown}\n" if shown else "")


def test_report_skips_a_source_the_data_never_saw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _units(tmp_path)
    monkeypatch.chdir(tmp_path)
    data = tmp_path / ".coverage"
    _measured(tmp_path, {"": {"pkg/n.py": _ALL_TWENTY}}, data=data)
    sources = ["pkg/m.py", "pkg/n.py"]

    assert not coverage_floor.report(
        data,
        sources=sources,
        floors={"pkg/": 100.0},
        scope="file",
        owners=_owners(sources),
    )
    assert capsys.readouterr().out == ""


def test_report_pools_only_one_directory_s_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``pkg/deep``, fully run, does not lift ``pkg``'s pool, nor join it."""
    _units(tmp_path)
    monkeypatch.chdir(tmp_path)
    data = tmp_path / ".coverage"
    _measured(tmp_path, {"": {"pkg/m.py": _CALLED, "pkg/deep/o.py": [1]}}, data=data)
    sources = ["pkg/m.py", "pkg/deep/o.py"]

    assert coverage_floor.report(
        data,
        sources=sources,
        floors={"pkg/": 80.0},
        scope="directory",
        owners=_owners(sources),
    )
    assert capsys.readouterr().out == (
        "coverage-floor: pkg at 75% is below 80%; missed pkg/m.py: 4\n"
    )


def test_report_of_an_empty_module_is_fully_covered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tree(tmp_path, {"pkg/empty.py": "", "pkg/n.py": "X = 1\n"})
    monkeypatch.chdir(tmp_path)
    data = tmp_path / ".coverage"
    _measured(tmp_path, {"": {"pkg/empty.py": [], "pkg/n.py": [1]}}, data=data)
    sources = ["pkg/empty.py", "pkg/n.py"]

    assert not coverage_floor.report(
        data,
        sources=sources,
        floors={"pkg/": 100.0},
        scope="file",
        owners=_owners(sources),
    )


def test_report_pools_every_source_s_missed_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tree(tmp_path, {"pkg/m.py": _SOURCE, "pkg/p.py": _SOURCE})
    monkeypatch.chdir(tmp_path)
    data = tmp_path / ".coverage"
    _measured(tmp_path, {"": {"pkg/m.py": _CALLED, "pkg/p.py": _CALLED}}, data=data)
    sources = ["pkg/m.py", "pkg/p.py"]

    assert coverage_floor.report(
        data,
        sources=sources,
        floors={"pkg/": 100.0},
        scope="directory",
        owners=_owners(sources),
    )
    assert capsys.readouterr().out == (
        "coverage-floor: pkg at 75% is below 100%; missed pkg/m.py: 4; pkg/p.py: 4\n"
    )


def test_report_rounds_to_the_configured_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two of three lines run is 66.67% at precision 2, short of 66.8."""
    _tree(
        tmp_path,
        {
            "pkg/m.py": "def f():\n    return 1\nX = 1\n",
            ".coveragerc": "[report]\nprecision = 2\n",
        },
    )
    monkeypatch.chdir(tmp_path)
    data = tmp_path / ".coverage"
    _measured(tmp_path, {"": {"pkg/m.py": [1, 3]}}, data=data)

    assert coverage_floor.report(
        data,
        sources=["pkg/m.py"],
        floors={"pkg/": 66.8},
        scope="file",
        owners=_owners(["pkg/m.py"]),
    )
    assert capsys.readouterr().out == (
        "coverage-floor: pkg/m.py at 66.67% is below 66.8%; missed 2\n"
    )


@pytest.mark.parametrize("text", ["pkg=95", "95", "pkg/=x"])
def test_a_malformed_floor_is_a_usage_error_saying_what_a_floor_is(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    text: str,
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _main(monkeypatch, "--floor", text)

    assert f"Floor {text!r} is not PREFIX/=PERCENT." in capsys.readouterr().err


def test_the_data_is_required(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """It runs no tests, so without ``pytest``'s run there is nothing to floor by."""
    monkeypatch.setattr(sys, "argv", ["coverage_floor", "pkg/m.py"])

    with pytest.raises(SystemExit, match="2"):
        coverage_floor.main()

    assert "--data" in capsys.readouterr().err


def test_flags_parse_to_their_types(capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()
    coverage_floor._add_arguments(parser)

    defaults = cast(coverage_floor.Flags, parser.parse_args(["--data", "d"]))
    flags = cast(
        coverage_floor.Flags,
        parser.parse_args(
            ["--data", "d", "--scope", "directory", "--floor", "a=b/=90", "x.py"],
        ),
    )

    assert (defaults.paths, defaults.scope, defaults.floor) == ([], "file", [])
    assert (flags.data, flags.paths, flags.scope) == (Path("d"), ["x.py"], "directory")
    assert flags.floor == ["a=b/=90"]
    assert coverage_floor._floor(flags.floor[-1]) == ("a=b/", 90.0)
    with pytest.raises(SystemExit):
        parser.parse_args(["--data", "d", "--scope", "repo"])
    assert "invalid choice: 'repo'" in capsys.readouterr().err
    parser.print_help()
    shown = f" {' '.join(capsys.readouterr().out.split())} "
    for phrase in (
        "--floor PREFIX/=PERCENT",
        "Changed repository-relative paths.",
        "Floor each changed file, or each directory's changed files pooled",
    ):
        assert f" {phrase} " in shown


def test_the_help_opens_with_the_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["coverage_floor", "--help"])
    with pytest.raises(SystemExit):
        coverage_floor.main()
    assert (
        capsys.readouterr().out.split("\n\n")[1]
        == (coverage_floor.__doc__ or "").split("\n\n")[0]
    )


def test_without_a_docstring_the_help_has_no_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert_help_without_docstring(monkeypatch, coverage_floor, coverage_floor.main)
    assert capsys.readouterr().out.split("\n\n")[1].startswith("positional arguments")


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
