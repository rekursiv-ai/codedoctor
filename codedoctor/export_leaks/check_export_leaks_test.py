"""Tests for the pre-commit export leak gate.

Hermetic: each test builds the checkout it needs under ``tmp_path``. The checks
of the monorepo's own package configs live with copybarista, in
``package_config_reverse_test.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import subprocess
import sys
import threading

import pytest

from codedoctor.export_leaks import check_export_leaks
from codedoctor.export_leaks.check_export_leaks import (
    package_configs,
    python_package_configs,
    touched_configs,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def test_package_configs_find_each_synced_package(tmp_path: Path) -> None:
    """One to three levels under ``loop/`` and one under ``website/``."""
    expected = [
        _package(tmp_path, "loop/a"),
        _package(tmp_path, "loop/b/c"),
        _package(tmp_path, "loop/d/e/f"),
        _package(tmp_path, "website/site"),
    ]

    assert sorted(package_configs(root=tmp_path)) == sorted(expected)


def test_package_configs_skip_an_unsynced_or_too_deep_config(tmp_path: Path) -> None:
    """A fixture config has no sync config, or sits deeper than any package."""
    (tmp_path / "loop/fixture").mkdir(parents=True)
    (tmp_path / "loop/fixture/copy.barista.toml").write_text("", encoding="utf-8")
    _package(tmp_path, "loop/a/b/c/d")
    (tmp_path / "loop/orphan").mkdir(parents=True)
    (tmp_path / "loop/orphan/copybarista.sync.toml").write_text(
        "[sync]\n",
        encoding="utf-8",
    )

    assert package_configs(root=tmp_path) == []


def test_package_configs_in_a_checkout_that_exports_nothing(tmp_path: Path) -> None:
    """An ordinary repository has no package, so the gate has nothing to do."""
    assert package_configs(root=tmp_path) == []
    assert touched_configs(["a.py"], root=tmp_path) == []


def test_python_package_configs_leave_out_a_non_python_package(tmp_path: Path) -> None:
    library = _package(tmp_path, "loop/core")
    _package(tmp_path, "website/site", sync="[sync]\npython_package = false\n")

    assert python_package_configs(root=tmp_path) == [library]


def test_touched_configs_selects_packages_that_copy_a_shared_file(
    tmp_path: Path,
) -> None:
    """A package's inputs include files outside its own directory."""
    copier = _package(
        tmp_path,
        "loop/a",
        config='[[files.copy]]\nsource = "shared/x.md"\n',
    )
    _package(tmp_path, "loop/b")

    assert touched_configs(["shared/x.md"], root=tmp_path) == [copier]


def test_touched_configs_selects_by_package_containment(tmp_path: Path) -> None:
    owner = _package(tmp_path, "loop/a")
    _package(tmp_path, "loop/b")

    assert touched_configs(["loop/a/git.py"], root=tmp_path) == [owner]


def test_touched_configs_ignores_a_file_no_package_exports(tmp_path: Path) -> None:
    _package(tmp_path, "loop/a")

    assert touched_configs(["docs/vision.md"], root=tmp_path) == []


def test_touched_configs_without_paths_checks_everything(tmp_path: Path) -> None:
    """An empty path list means a manual or --all-files run."""
    _package(tmp_path, "loop/a")
    _package(tmp_path, "loop/b")

    assert touched_configs([], root=tmp_path) == package_configs(root=tmp_path)


def test_touched_configs_selects_everything_for_conftest(tmp_path: Path) -> None:
    """The root conftest is an implicit input to every package."""
    _package(tmp_path, "loop/a")
    _package(tmp_path, "loop/b")

    assert touched_configs(["conftest.py"], root=tmp_path) == package_configs(
        root=tmp_path,
    )


def test_main_exports_packages_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--workers N`` exports N packages at once.

    A barrier is what separates real concurrency from a serial loop: run
    sequentially, the first `check` waits on peers that never start and the
    barrier's own timeout fails the test, so this cannot pass without the pool.
    """
    configs = _configs(3)
    barrier = threading.Barrier(len(configs), timeout=10.0)

    def rendezvous(config: Path) -> str:
        del config
        barrier.wait()
        return ""

    monkeypatch.setattr(check_export_leaks, "touched_configs", _select(configs))
    monkeypatch.setattr(check_export_leaks, "check", rendezvous)
    monkeypatch.setattr(sys, "argv", ["check_export_leaks.py", "--workers", "3"])

    assert check_export_leaks.main() == 0


def test_main_reports_failures_in_config_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Completion order under the pool is nondeterministic; the report is not."""
    configs = _configs(3)

    def fail(config: Path) -> str:
        return f"boom {config.parent.name}"

    monkeypatch.setattr(check_export_leaks, "touched_configs", _select(configs))
    monkeypatch.setattr(check_export_leaks, "check", fail)
    monkeypatch.setattr(sys, "argv", ["check_export_leaks.py"])

    assert check_export_leaks.main() == 1
    assert capsys.readouterr().err.splitlines() == [
        line
        for index in range(3)
        for line in (f"loop/p{index}/copy.barista.toml:", f"boom p{index}")
    ]


def test_main_reads_its_paths_from_the_gates_argfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--all-files`` as argv overflowed macOS's ``ARG_MAX``; the gate hands an @argfile."""
    seen: list[str] = []

    def touched(paths: Sequence[str]) -> list[Path]:
        seen.extend(paths)
        return []

    argfile = tmp_path / "leaks.args"
    argfile.write_text("a.py\n-b.py\n")
    monkeypatch.setattr(check_export_leaks, "touched_configs", touched)
    monkeypatch.setattr(sys, "argv", ["check_export_leaks.py", "--", f"@{argfile}"])

    assert check_export_leaks.main() == 0
    assert seen == ["a.py", "-b.py"]


def test_main_checks_only_the_packages_its_paths_touch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _package(tmp_path, "loop/a")
    _package(tmp_path, "loop/b")
    checked: list[Path] = []

    def record(config: Path) -> str:
        checked.append(config)
        return ""

    monkeypatch.setattr(
        check_export_leaks,
        "touched_configs",
        partial(touched_configs, root=tmp_path),
    )
    monkeypatch.setattr(check_export_leaks, "check", record)
    monkeypatch.setattr(sys, "argv", ["check_export_leaks.py", "loop/a/git.py"])

    assert check_export_leaks.main() == 0
    assert checked == [owner]


def test_main_returns_zero_when_no_packages_are_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path with no export owner is a successful no-op."""
    monkeypatch.setattr(check_export_leaks, "touched_configs", Mock(return_value=[]))
    monkeypatch.setattr(sys, "argv", ["check_export_leaks", "docs/none.md"])
    assert check_export_leaks.main() == 0


@pytest.mark.parametrize(("workers", "size"), [("2", 2), ("8", 3)])
def test_main_sizes_its_pool_by_the_flag_and_the_package_count(
    monkeypatch: pytest.MonkeyPatch,
    workers: str,
    size: int,
) -> None:
    """``--workers`` bounds the pool, and three packages need no more than three."""
    sizes = _record_pool_sizes(monkeypatch)
    monkeypatch.setattr(check_export_leaks, "touched_configs", _select(_configs(3)))
    monkeypatch.setattr(check_export_leaks, "check", _passing)
    monkeypatch.setattr(sys, "argv", ["check_export_leaks.py", "--workers", workers])

    assert check_export_leaks.main() == 0
    assert sizes == [size]


def test_workers_default_to_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = _record_pool_sizes(monkeypatch)
    monkeypatch.setattr(check_export_leaks, "touched_configs", _select(_configs(3)))
    monkeypatch.setattr(check_export_leaks, "check", _passing)
    monkeypatch.setattr(sys, "argv", ["check_export_leaks.py"])

    assert check_export_leaks.main() == 0
    assert sizes == [1]


@pytest.mark.parametrize(
    ("docstring", "after_usage"),
    [
        (check_export_leaks.__doc__, "Run every exported package's"),
        (None, "positional arguments:"),
    ],
    ids=["docstring", "optimized away"],
)
def test_main_help_describes_the_gate_from_its_docstring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    docstring: str | None,
    after_usage: str,
) -> None:
    """``python -OO`` strips the docstring; the help then has no description."""
    monkeypatch.setattr(check_export_leaks, "__doc__", docstring)
    monkeypatch.setattr(sys, "argv", ["check_export_leaks.py", "--help"])

    with pytest.raises(SystemExit) as exited:
        check_export_leaks.main()

    assert exited.value.code == 0
    assert capsys.readouterr().out.split("\n\n")[1].startswith(after_usage)


def test_check_returns_export_stderr_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export failures are returned without attempting formatting."""
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 2, "", "bad export\n")),
    )
    assert check_export_leaks.check(_package(tmp_path, "loop/a"), root=tmp_path) == (
        "bad export"
    )


def test_check_exports_into_scratch_then_format_checks_that_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The export command, and the tree the format check then reads, exactly."""
    config = _package(tmp_path, "loop/a")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(subprocess, "run", run)

    assert check_export_leaks.check(config, root=tmp_path) == ""

    export, fmt = run.call_args_list
    scratch = Path(cast("list[str]", export.args[0])[7])
    assert scratch.name.startswith("leak-check-")
    assert export.args == (
        [
            sys.executable,
            "-m",
            "copybarista",
            "export",
            "loop/a/copy.barista.toml",
            ".",
            "--folder-dir",
            str(scratch),
            "--force",
        ],
    )
    assert export.kwargs == {
        "cwd": tmp_path,
        "capture_output": True,
        "text": True,
        "check": False,
    }
    assert fmt.args == ([sys.executable, "-m", "ruff", "format", "--check", "."],)
    assert fmt.kwargs == {
        "cwd": scratch,
        "capture_output": True,
        "text": True,
        "check": False,
    }


@pytest.mark.parametrize("stderr", ["", "  \n"])
def test_export_exit_is_failure_even_without_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 1, "", stderr)),
    )
    assert (
        check_export_leaks.check(_package(tmp_path, "loop/a"), root=tmp_path)
        == "export exited 1"
    )


def test_format_check_reports_ruff_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Formatting failures include the exported-tree explanation and output."""
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 1, "bad.py\n", "")),
    )
    assert check_export_leaks._format_check(tmp_path) == (
        "exported tree is not format-clean (the export rewrites line lengths; "
        "run `ruff format` on the exported tree to see the wrapping it wants):\n"
        "bad.py"
    )


def test_format_check_accepts_clean_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean exported tree returns no diagnostic."""
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, "", "")),
    )
    assert check_export_leaks._format_check(tmp_path) == ""


def test_copy_sources_rejects_invalid_toml_shapes(tmp_path: Path) -> None:
    """Malformed and non-list copy declarations are safely empty."""
    config = tmp_path / "copy.barista.toml"
    config.write_text("not = [", encoding="utf-8")
    assert check_export_leaks._copy_sources(config) == ()
    config.write_text('files = { copy = "no" }', encoding="utf-8")
    assert check_export_leaks._copy_sources(config) == ()


def test_copy_sources_keeps_only_string_sources(tmp_path: Path) -> None:
    """Copy entries with missing or non-string sources are ignored."""
    config = tmp_path / "copy.barista.toml"
    config.write_text(
        '[[files.copy]]\nsource = "a.py"\n[[files.copy]]\nsource = 3\n[[files.copy]]\n',
        encoding="utf-8",
    )
    assert check_export_leaks._copy_sources(config) == ("a.py",)


def test_copy_sources_handles_non_mapping_entries(tmp_path: Path) -> None:
    """Only mapping entries can provide copy sources."""
    config = tmp_path / "copy.barista.toml"
    config.write_text('files = { copy = ["bad"] }', encoding="utf-8")
    assert check_export_leaks._copy_sources(config) == ()
    config.write_text(
        'files = { copy = ["bad", { source = "kept" }] }',
        encoding="utf-8",
    )
    assert check_export_leaks._copy_sources(config) == ("kept",)
    config.write_text('files = "bad"', encoding="utf-8")
    assert check_export_leaks._copy_sources(config) == ()


def _package(
    root: Path,
    directory: str,
    *,
    config: str = "",
    sync: str = "[sync]\n",
) -> Path:
    """Write a synced package under ``root``; return its ``copy.barista.toml``."""
    (root / directory).mkdir(parents=True)
    (root / directory / "copybarista.sync.toml").write_text(sync, encoding="utf-8")
    path = root / directory / "copy.barista.toml"
    path.write_text(config, encoding="utf-8")
    return path


def _configs(count: int) -> list[Path]:
    """Return ``count`` config paths under the gate's own checkout; none is read."""
    return [
        check_export_leaks._ROOT / f"loop/p{index}/copy.barista.toml"
        for index in range(count)
    ]


def _select(configs: list[Path]) -> Callable[[Sequence[str]], list[Path]]:
    """Return a `touched_configs` replacement yielding `configs` for any paths."""

    def selected(paths: Sequence[str]) -> list[Path]:
        del paths
        return configs

    return selected


def _record_pool_sizes(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    """Replace the gate's pool with one recording each ``max_workers`` it gets."""
    sizes: list[int | None] = []

    def sized(*, max_workers: int | None = None) -> ThreadPoolExecutor:
        sizes.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(check_export_leaks, "ThreadPoolExecutor", sized)
    return sizes


def _passing(config: Path) -> str:
    """Stand in for `check` on a package whose export is clean."""
    del config
    return ""


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
