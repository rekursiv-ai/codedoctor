"""Tests for the per-file and per-directory coverage floor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from unittest.mock import Mock

import argparse
import subprocess
import sys
import threading

import coverage
import pytest

from codedoctor.coverage_floor import coverage_floor
from codedoctor.lib.testing.checks import assert_help_without_docstring


if TYPE_CHECKING:
    from collections.abc import Sequence


def _tree(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


_SOURCE = "def f(x):\n    if x:\n        return 1\n    return 2\n"


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
                "pkg/tests/test_c.py",
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
    ) == (
        ["app/a.py", "pkg/b.py", "pkg/c.py"],
        [
            "app/a_test.py",
            "pkg/test_b.py",
            "pkg/tests/c_test.py",
            "pkg/tests/test_c.py",
        ],
        ["cfg/untested.py"],
    )


def test_groups_balance_by_size_and_never_exceed_the_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = {"big": 90, "mid": 50, "s1": 40, "s2": 10}
    _tree(tmp_path, {f"{name}.py": "x" * size for name, size in sizes.items()})
    monkeypatch.chdir(tmp_path)
    tests = [f"{name}.py" for name in sizes]

    assert coverage_floor.groups(tests, count=2) == [
        ["big.py", "s2.py"],
        ["mid.py", "s1.py"],
    ]
    assert coverage_floor.groups(tests, count=9) == [[t] for t in tests]
    assert coverage_floor.groups(tests, count=0) == [tests]


def test_groups_keep_each_directory_contiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tests = ["pkg/a_test.py", "other_test.py", "pkg/b_test.py"]
    _tree(tmp_path, dict(zip(tests, ["x", "x" * 50, "x" * 90], strict=True)))
    monkeypatch.chdir(tmp_path)

    assert coverage_floor.groups(tests, count=1) == [
        ["other_test.py", "pkg/a_test.py", "pkg/b_test.py"],
    ]


@pytest.mark.cli_python_subprocess
@pytest.mark.parametrize(
    "case",
    [
        ("def test_f():\n    assert f(1) == 1\n    assert f(0) == 2\n", 0, ""),
        (
            "def test_f():\n    assert f(1) == 1\n",
            1,
            "coverage-floor: pkg/m.py at 75% is below 100%; missed 4",
        ),
        ("def test_f():\n    assert f(1) == 2\n", 1, "assert 1 == 2"),
    ],
    ids=["covered", "below-floor", "failing-test"],
)
def test_main_floors_each_changed_source_by_its_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: tuple[str, int, str],
) -> None:
    test_body, status, shown = case
    _tree(
        tmp_path,
        {
            "pyproject.toml": "[tool.pytest.ini_options]\npythonpath = ['.']\n",
            "pkg/__init__.py": "",
            "pkg/m.py": _SOURCE,
            "pkg/m_test.py": f"from pkg.m import f\n\n\n{test_body}",
            "pkg/n.py": "X = 1\n",
            "pkg/n_test.py": "from pkg.n import X\n\n\ndef test_x():\n    assert X\n",
            "pkg/sibling.py": _SOURCE,
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage_floor",
            "--workers",
            "2",
            "--floor",
            "pkg/=100",
            "--",
            "pkg/m.py",
            "pkg/n.py",
        ],
    )

    assert coverage_floor.main() == status

    out = capsys.readouterr().out
    assert shown in out
    # Measured with its directory, but not changed, so never floored.
    assert "sibling.py" not in out


def test_main_reads_its_paths_from_the_gates_argfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--all-files`` as argv overflowed macOS's ``ARG_MAX``; the gate hands an @argfile."""
    _tree(tmp_path, {"pkg/m.py": _SOURCE, "coverage.args": "pkg/m.py\n"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["coverage_floor", "--floor", "pkg/=100", "--", "@coverage.args"],
    )

    assert coverage_floor.main() == 1
    assert capsys.readouterr().out == "coverage-floor: pkg/m.py has no sidecar test\n"


def _padded(count: int) -> str:
    """Return ``_SOURCE`` plus ``count`` always-run statements."""
    return _SOURCE + "".join(f"X{i} = {i}\n" for i in range(count))


@pytest.mark.cli_python_subprocess
@pytest.mark.parametrize(
    "case",
    [
        (
            "file",
            95,
            1,
            "coverage-floor: pkg/m.py at 75% is below 95%; missed 4\n",
        ),
        ("directory", 95, 0, ""),
        (
            "directory",
            97,
            1,
            "coverage-floor: pkg at 96% is below 97%; missed pkg/m.py: 4\n",
        ),
    ],
    ids=["file", "directory", "directory-below"],
)
def test_scope_floors_each_file_or_pools_each_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: tuple[str, int, int, str],
) -> None:
    scope, floor, status, shown = case
    _tree(
        tmp_path,
        {
            "pyproject.toml": "[tool.pytest.ini_options]\npythonpath = ['.']\n",
            "pkg/__init__.py": "",
            # 1 of 4 statements missed: 75% alone.
            "pkg/m.py": _SOURCE,
            "pkg/m_test.py": "from pkg.m import f\n\n\ndef test_f():\n    assert f(1) == 1\n",
            # 20 statements, all run: pooled with m.py, 23 of 24 is 96%.
            "pkg/n.py": "".join(f"X{i} = {i}\n" for i in range(20)),
            "pkg/n_test.py": "from pkg.n import X0\n\n\ndef test_x():\n    assert X0 == 0\n",
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage_floor",
            "--scope",
            scope,
            "--floor",
            f"pkg/={floor}",
            "--",
            "pkg/m.py",
            "pkg/n.py",
        ],
    )

    assert coverage_floor.main() == status
    # Only a unit below its floor prints: no table, and n.py, fully run, never.
    assert capsys.readouterr().out == shown


@pytest.mark.parametrize("text", ["pkg=95", "95", "pkg/=x"])
def test_a_malformed_floor_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["coverage_floor", "--floor", text])

    with pytest.raises(SystemExit, match="2"):
        coverage_floor.main()


def test_an_unparsable_sidecar_is_one_line_and_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tree(
        tmp_path,
        {"pkg/m.py": _SOURCE, "pkg/m_test.py": "from m import a, b | c\n"},
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["coverage_floor", "--floor", "pkg/=100", "--", "pkg/m.py"],
    )
    run = Mock()
    monkeypatch.setattr(coverage_floor, "run_groups", run)

    assert coverage_floor.main() == 1
    assert capsys.readouterr().out == (
        "coverage-floor: pkg/m_test.py:1: invalid syntax\n"
    )
    run.assert_not_called()


def test_unparsable_skips_a_missing_file_and_checks_the_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tree(tmp_path, {"bad_test.py": "def f(:\n"})
    monkeypatch.chdir(tmp_path)

    assert coverage_floor._unparsable(["gone_test.py", "bad_test.py"]) == [
        "bad_test.py:1: invalid syntax",
    ]


@pytest.mark.parametrize(
    ("operator", "expected"),
    [({}, "Haswell"), ({"OPENBLAS_CORETYPE": "Zen"}, "Zen")],
    ids=["pinned", "operator-wins"],
)
def test_each_group_pins_openblas_before_numpy_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operator: dict[str, str],
    expected: str,
) -> None:
    monkeypatch.delenv("OPENBLAS_CORETYPE", raising=False)
    for name, value in operator.items():
        monkeypatch.setenv(name, value)
    envs: list[dict[str, str]] = []

    def run(args: list[str], *, env: dict[str, str], **_: object) -> None:
        del args
        envs.append(env)

    monkeypatch.setattr(subprocess, "run", run)

    coverage_floor._pytest(0, ["a_test.py"], cov=[], data=tmp_path / ".coverage")

    [env] = envs
    assert (env["OPENBLAS_CORETYPE"], env["OPENBLAS_NUM_THREADS"]) == (expected, "1")


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
def test_a_floored_source_without_a_sidecar_fails_and_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: tuple[list[str], int, str],
) -> None:
    paths, status, shown = case
    _tree(
        tmp_path,
        {
            "pkg/__init__.py": "X = 1\n",
            "pkg/__main__.py": "X = 1\n",
            "pkg/m.py": _SOURCE,
            "pkg/third_party/vendored.py": _SOURCE,
            "pkg/testdata/fixture.py": _SOURCE,
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["coverage_floor", "--floor", "pkg/=100", "--", *paths],
    )
    run = Mock()
    monkeypatch.setattr(coverage_floor, "run_groups", run)

    assert coverage_floor.main() == status
    assert capsys.readouterr().out == shown
    run.assert_not_called()


@pytest.mark.cli_python_subprocess
def test_a_source_its_sidecar_never_runs_is_below_the_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tree(
        tmp_path,
        {
            "pyproject.toml": "[tool.pytest.ini_options]\npythonpath = ['.']\n",
            "pkg/__init__.py": "",
            "pkg/m.py": _SOURCE,
            "pkg/m_test.py": "def test_nothing():\n    assert True\n",
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["coverage_floor", "--floor", "pkg/=100", "--", "pkg/m.py"],
    )

    assert coverage_floor.main() == 1
    assert capsys.readouterr().out == (
        "coverage-floor: pkg/m.py at 0% is below 100%; missed 1-4\n"
    )


def test_main_without_a_floored_sidecar_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["coverage_floor", "pkg/m.py", "ops/x.py"])

    assert coverage_floor.main() == 0
    assert capsys.readouterr().out == ""


def _measured(tmp_path: Path, runs: dict[str, str], *, data: Path) -> None:
    """Write ``runs`` (module path to calling code) as sources, run each measured."""
    measurement = coverage.Coverage(data_file=str(data))
    for path, call in runs.items():
        source = tmp_path / path
        code = compile(source.read_text(), str(source.resolve()), "exec")
        measurement.start()
        try:
            namespace: dict[str, object] = {}
            exec(code, namespace)  # noqa: S102 -- the test's own fixture.
            exec(call, namespace)  # noqa: S102 -- the test's own fixture.
        finally:
            measurement.stop()
    measurement.save()


# A group runs on a pool thread, where starting a tracer would interleave with this
# suite's own coverage collector.
def _recorded(data: Path, source: Path) -> None:
    """Write ``data`` as a run that saw line 1 of ``source``, tracing nothing."""
    recorded = coverage.CoverageData(basename=str(data))
    recorded.add_lines({str(source.resolve()): [1]})
    recorded.write()


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
    _measured(tmp_path, {"pkg/m.py": "f(1)", "pkg/n.py": ""}, data=data)

    below = coverage_floor.report(
        data,
        sources=["pkg/m.py", "pkg/n.py"],
        floors=floors,
        scope=scope,
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
    _measured(tmp_path, {"pkg/n.py": ""}, data=data)

    assert not coverage_floor.report(
        data,
        sources=["pkg/m.py", "pkg/n.py"],
        floors={"pkg/": 100.0},
        scope="file",
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
    _measured(tmp_path, {"pkg/m.py": "f(1)", "pkg/deep/o.py": ""}, data=data)

    assert coverage_floor.report(
        data,
        sources=["pkg/m.py", "pkg/deep/o.py"],
        floors={"pkg/": 80.0},
        scope="directory",
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
    _measured(tmp_path, {"pkg/empty.py": "", "pkg/n.py": ""}, data=data)

    assert not coverage_floor.report(
        data,
        sources=["pkg/empty.py", "pkg/n.py"],
        floors={"pkg/": 100.0},
        scope="file",
    )


def test_report_pools_every_source_s_missed_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tree(tmp_path, {"pkg/m.py": _SOURCE, "pkg/p.py": _SOURCE})
    monkeypatch.chdir(tmp_path)
    data = tmp_path / ".coverage"
    _measured(tmp_path, {"pkg/m.py": "f(1)", "pkg/p.py": "f(1)"}, data=data)

    assert coverage_floor.report(
        data,
        sources=["pkg/m.py", "pkg/p.py"],
        floors={"pkg/": 100.0},
        scope="directory",
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
    _measured(tmp_path, {"pkg/m.py": ""}, data=data)

    assert coverage_floor.report(
        data,
        sources=["pkg/m.py"],
        floors={"pkg/": 66.8},
        scope="file",
    )
    assert capsys.readouterr().out == (
        "coverage-floor: pkg/m.py at 66.67% is below 66.8%; missed 2\n"
    )


def test_every_group_runs_at_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More groups than a default pool's 32 threads still all start together."""
    count = 33
    started = threading.Barrier(count, timeout=10)

    def waiting(
        index: int,
        tests: Sequence[str],
        *,
        cov: Sequence[str],
        data: Path,
    ) -> subprocess.CompletedProcess[str]:
        del tests, cov
        started.wait()
        _recorded(data.with_name(f"{data.name}.{index}"), tmp_path / f"t{index}.py")
        return subprocess.CompletedProcess([], 0, stdout="")

    monkeypatch.setattr(coverage_floor, "_pytest", waiting)

    assert not coverage_floor.run_groups(
        [[f"t{i}_test.py"] for i in range(count)],
        cov=[],
        data=tmp_path / ".coverage",
    )


def test_each_group_runs_as_one_pytest_into_its_own_data_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _units(tmp_path)
    monkeypatch.chdir(tmp_path)
    data = tmp_path / ".coverage"
    runs: list[tuple[int, list[str], list[str], Path]] = []

    def pytest_group(
        index: int,
        tests: Sequence[str],
        *,
        cov: Sequence[str],
        data: Path,
    ) -> subprocess.CompletedProcess[str]:
        runs.append((index, list(tests), list(cov), data))
        source = tmp_path / ("pkg/m.py" if index == 0 else "pkg/n.py")
        _recorded(data.with_name(f"{data.name}.{index}"), source)
        return subprocess.CompletedProcess([], index, stdout=f"group {index}\n")

    monkeypatch.setattr(coverage_floor, "_pytest", pytest_group)

    failed = coverage_floor.run_groups(
        [["a_test.py"], ["b_test.py", "c_test.py"]],
        cov=["--cov=pkg"],
        data=data,
    )

    assert failed
    assert sorted(runs) == [
        (0, ["a_test.py"], ["--cov=pkg"], data),
        (1, ["b_test.py", "c_test.py"], ["--cov=pkg"], data),
    ]
    assert capsys.readouterr().out == "group 1\n"
    combined = coverage.Coverage(data_file=str(data))
    combined.load()
    assert {Path(name).name for name in combined.get_data().measured_files()} == {
        "m.py",
        "n.py",
    }


def test_groups_that_all_pass_did_not_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def passing(
        index: int,
        tests: Sequence[str],
        *,
        cov: Sequence[str],
        data: Path,
    ) -> subprocess.CompletedProcess[str]:
        del index, tests, cov, data
        return subprocess.CompletedProcess([], 0, stdout="")

    monkeypatch.setattr(coverage_floor, "_pytest", passing)

    assert not coverage_floor.run_groups(
        [["a_test.py"]],
        cov=[],
        data=tmp_path / ".coverage",
    )


def test_one_group_is_a_serial_quiet_pytest_measuring_its_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "8")
    data = tmp_path / ".coverage"

    coverage_floor._pytest(3, ["a_test.py", "b_test.py"], cov=["--cov=pkg"], data=data)

    [(argv, kwargs)] = calls
    assert argv == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:randomly",
        "--tb=line",
        "--cov-report=",
        "--cov=pkg",
        "--",
        "a_test.py",
        "b_test.py",
    ]
    env = kwargs.pop("env")
    assert kwargs == {
        "check": False,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    assert isinstance(env, dict)
    assert {
        name: env[name]
        for name in (
            "COVERAGE_FILE",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    } == {
        "COVERAGE_FILE": f"{data}.3",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


@pytest.mark.parametrize(
    ("failed", "below", "status"),
    [(False, False, 0), (True, False, 1), (False, True, 1), (True, True, 1)],
)
def test_main_runs_the_groups_then_floors_their_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed: bool,
    below: bool,
    status: int,
) -> None:
    _tree(
        tmp_path,
        {
            "pkg/m.py": _SOURCE,
            "pkg/m_test.py": "x" * 9,
            "pkg/sub/n.py": _SOURCE,
            "pkg/sub/n_test.py": "x",
        },
    )
    monkeypatch.chdir(tmp_path)
    seen: dict[str, object] = {}

    def run_groups(
        groups: list[list[str]],
        *,
        cov: list[str],
        data: Path,
    ) -> bool:
        seen.update(groups=groups, cov=cov, data=data)
        assert data.parent.is_dir()
        return failed

    def report(
        data: Path,
        *,
        sources: list[str],
        floors: dict[str, float],
        scope: str,
    ) -> bool:
        seen.update(report=(data, sources, floors, scope))
        return below

    monkeypatch.setattr(coverage_floor, "run_groups", run_groups)
    monkeypatch.setattr(coverage_floor, "report", report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage_floor",
            "--workers",
            "2",
            "--scope",
            "directory",
            "--floor",
            "pkg/=90",
            "--",
            "pkg/m.py",
            "pkg/sub/n.py",
        ],
    )

    assert coverage_floor.main() == status
    data = seen["data"]
    assert isinstance(data, Path)
    assert data.name == ".coverage"
    assert data.parent.name.startswith("coverage-floor-")
    assert seen["groups"] == [["pkg/m_test.py"], ["pkg/sub/n_test.py"]]
    assert seen["cov"] == ["--cov=pkg", "--cov=pkg/sub"]
    assert seen["report"] == (
        data,
        ["pkg/m.py", "pkg/sub/n.py"],
        {**dict(_default_floors()), "pkg/": 90.0},
        "directory",
    )


def _never_below(data: Path, **_: object) -> bool:
    del data
    return False


def test_main_sizes_the_groups_to_the_machine_without_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tree(tmp_path, {f"pkg/m{i}.py": "" for i in range(3)})
    _tree(tmp_path, {f"pkg/m{i}_test.py": "x" for i in range(3)})
    monkeypatch.chdir(tmp_path)
    counts: list[list[list[str]]] = []

    def run_groups(groups: list[list[str]], **_: object) -> bool:
        counts.append(groups)
        return False

    monkeypatch.setattr(coverage_floor, "run_groups", run_groups)
    monkeypatch.setattr(coverage_floor, "report", _never_below)
    monkeypatch.setattr(coverage_floor, "default_workers", lambda: 2)
    monkeypatch.setattr(
        sys,
        "argv",
        ["coverage_floor", "--floor", "pkg/=100", "--", "pkg/m0.py", "pkg/m1.py"],
    )

    assert coverage_floor.main() == 0
    assert [len(groups) for groups in counts] == [2]


def test_the_defaults_floor_files_and_size_to_the_machine() -> None:
    parser = argparse.ArgumentParser()
    coverage_floor._add_arguments(parser)

    flags = cast(coverage_floor.Flags, parser.parse_args([]))

    assert (flags.paths, flags.workers, flags.scope) == ([], None, "file")
    assert flags.floor == _default_floors()


def _default_floors() -> list[tuple[str, float]]:
    """Return the floors ``--floor`` adds to when none is given."""
    return []


def test_flags_parse_to_their_types(capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()
    coverage_floor._add_arguments(parser)

    flags = cast(
        coverage_floor.Flags,
        parser.parse_args(
            ["--workers", "3", "--scope", "directory", "--floor", "a=b/=90", "x.py"],
        ),
    )

    assert (flags.paths, flags.workers, flags.scope) == (["x.py"], 3, "directory")
    assert flags.floor[-1] == ("a=b/", 90.0)
    assert cast(coverage_floor.Flags, parser.parse_args(["--scope", "file"])).scope == (
        "file"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--scope", "repo"])
    assert "invalid choice: 'repo'" in capsys.readouterr().err
    parser.print_help()
    shown = f" {' '.join(capsys.readouterr().out.split())} "
    for phrase in (
        "--floor PREFIX/=PERCENT",
        "Changed repository-relative paths.",
        "Floor each changed file, or each directory's changed files pooled",
    ):
        assert f" {phrase} " in shown


def test_a_malformed_floor_says_what_a_floor_is(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["coverage_floor", "--floor", "pkg=95"])

    with pytest.raises(SystemExit):
        coverage_floor.main()

    assert "Floor 'pkg=95' is not PREFIX/=PERCENT." in capsys.readouterr().err


def test_a_conftest_or_test_named_module_is_never_a_floored_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tree(tmp_path, dict.fromkeys(["pkg/conftest.py", "pkg/test_x.py"], ""))
    monkeypatch.chdir(tmp_path)

    assert coverage_floor.sidecars(
        ["pkg/conftest.py", "pkg/test_x.py"],
        floored=("pkg/",),
    ) == ([], [], [])


def test_groups_place_the_largest_test_first_whatever_its_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = {"a": 10, "b": 40, "c": 50, "d": 90}
    _tree(tmp_path, {f"{name}.py": "x" * size for name, size in sizes.items()})
    monkeypatch.chdir(tmp_path)

    assert coverage_floor.groups([f"{name}.py" for name in sizes], count=2) == [
        ["a.py", "d.py"],
        ["b.py", "c.py"],
    ]


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


def test_parsing_never_sizes_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every gate passes ``--workers``; probing the machine for a default is waste."""
    sized: list[int] = []
    monkeypatch.setattr(coverage_floor, "default_workers", lambda: sized.append(1) or 1)
    parser = argparse.ArgumentParser()
    coverage_floor._add_arguments(parser)

    parser.parse_args(["--workers", "2"])

    assert sized == []


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
