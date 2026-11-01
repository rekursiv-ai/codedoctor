"""Tests for the per-file and per-directory coverage floor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import argparse
import subprocess
import sys

import pytest

from codedoctor.coverage_floor import coverage_floor


if TYPE_CHECKING:
    from pathlib import Path


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
