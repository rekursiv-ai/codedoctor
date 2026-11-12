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

import ast
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


def test_check_reports_every_failure_of_the_exported_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each check reads the exported tree; one failing does not hide another.

    The stubs are the checkout's own stub trees; a file there is no stubbed
    package.
    """
    stubs = tmp_path / "loop" / "typings" / "stubs"
    (stubs / "torch").mkdir(parents=True)
    (stubs / "numpy").mkdir()
    (stubs / "pkg.pyi").write_text("")

    def run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "export" in argv:
            tree = Path(argv[7])
            (tree / "pkg").mkdir()
            (tree / "pkg" / "__init__.py").write_text(
                "from pkg.gone import x\nimport pkg.lost\nimport torch, numpy\n",
            )
            (tree / "pyproject.toml").write_text('[project]\nname = "pkg"\n')
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 1, "pkg/__init__.py\n", "")

    monkeypatch.setattr(subprocess, "run", run)

    assert check_export_leaks.check(_package(tmp_path, "loop/a"), root=tmp_path) == (
        "exported tree is not format-clean (the export rewrites line lengths; "
        "run `ruff format` on the exported tree to see the wrapping it wants):\n"
        "pkg/__init__.py\n"
        "exported imports of unshipped code:\n"
        "pkg/__init__.py: pkg.gone\n"
        "pkg/__init__.py: pkg.lost\n"
        "exported tree imports stubbed packages without depending on "
        "rekursiv-ai-typings: numpy, torch"
    )


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


def test_an_import_of_a_missing_module_or_name_is_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name counts only when the shipped ``.py`` binds it in module scope."""
    package = tmp_path / "pkg" / "data"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "shipped.py").write_text(
        "import os.path\nfrom typing import Final\nx: Final = 1\n"
        "if condition:\n    conditional = 1\n"
        "def f(): ...\nclass C: ...\n"
        "def helper():\n"
        "    local_store = 1\n"
        "    from math import sqrt as local_import\n"
        "class Namespace:\n"
        "    class_store = 1\n"
        "comprehension = [comprehension_store for comprehension_store in ()]\n"
        "callback = lambda: (lambda_store := 1)\n",
        encoding="utf-8",
    )
    (package / "shipped.pyi").write_text(
        "class stripped: ...\nclass StubOnly: ...\n",
        encoding="utf-8",
    )
    (package / "stubs.pyi").write_text("class Public: ...\n", encoding="utf-8")
    (package / "user.py").write_text(
        "from pkg.data.shipped import C as ClassAlias, f, os, x, conditional\n"
        "from pkg.data.shipped import local_store, local_import\n"
        "from pkg.data.shipped import class_store, comprehension_store\n"
        "from pkg.data.shipped import lambda_store\n"
        "from pkg.data.shipped import *\n"
        "from pkg.data.shipped import stripped\n"
        "from pkg.data.shipped import StubOnly\n"
        "from pkg.data.stubs import Public\n"
        "from pkg.data import shipped\n"
        "from pkg.data.excluded import y\n"
        "import pkg.data.gone\n",
        encoding="utf-8",
    )
    namespace = tmp_path / "pkg" / "namespace"
    namespace.mkdir()
    (namespace / "leaf.py").write_text("", encoding="utf-8")
    (namespace / "leaf_test.py").write_text(
        "import pkg.namespace\nfrom pkg.namespace import leaf\n",
        encoding="utf-8",
    )
    parsed: dict[str, int] = {}
    parse = ast.parse

    def count_parse(source: str, filename: str = "<unknown>") -> ast.Module:
        parsed[filename] = parsed.get(filename, 0) + 1
        return parse(source, filename=filename)

    with monkeypatch.context() as patch:
        patch.setattr(ast, "parse", count_parse)
        unresolved = check_export_leaks._unresolved_imports(tmp_path)

    assert unresolved == [
        "pkg/data/user.py: pkg.data.shipped.local_store",
        "pkg/data/user.py: pkg.data.shipped.local_import",
        "pkg/data/user.py: pkg.data.shipped.class_store",
        "pkg/data/user.py: pkg.data.shipped.comprehension_store",
        "pkg/data/user.py: pkg.data.shipped.lambda_store",
        "pkg/data/user.py: pkg.data.shipped.stripped",
        "pkg/data/user.py: pkg.data.shipped.StubOnly",
        "pkg/data/user.py: pkg.data.stubs",
        "pkg/data/user.py: pkg.data.excluded",
        "pkg/data/user.py: pkg.data.gone",
    ]
    assert parsed == {str(path): 1 for path in sorted(tmp_path.glob("pkg/**/*.py"))}


def test_an_import_nested_in_a_function_or_handler_is_checked(tmp_path: Path) -> None:
    """Imports hide under ``TYPE_CHECKING``, in functions, and in handlers."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text(
        "def lazy():\n"
        "    try:\n"
        "        pass\n"
        "    except ImportError:\n"
        "        from pkg import in_handler\n"
        "    match 1:\n"
        "        case 1:\n"
        "            import pkg.in_case\n",
        encoding="utf-8",
    )

    assert check_export_leaks._unresolved_imports(tmp_path) == [
        "pkg/__init__.py: pkg.in_handler",
        "pkg/__init__.py: pkg.in_case",
    ]


def test_a_dependency_or_relative_import_is_not_the_trees_own(tmp_path: Path) -> None:
    """Only a name the tree ships at its top level is checked."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text(
        "import numpy.missing\nfrom torch import absent\nfrom . import sibling\n",
        encoding="utf-8",
    )
    (tmp_path / "setup.py").write_text("import pkg\nfrom pkg import nothing\n")

    assert check_export_leaks._unresolved_imports(tmp_path) == ["setup.py: pkg.nothing"]


def test_a_hidden_directory_is_not_scanned(tmp_path: Path) -> None:
    """``.github`` scripts and caches ship no importable module."""
    hidden = tmp_path / ".github" / "scripts"
    hidden.mkdir(parents=True)
    (hidden / "tool.py").write_text("from pkg import hidden\n", encoding="utf-8")
    (tmp_path / "pkg.py").write_text("", encoding="utf-8")
    (tmp_path / "user.py").write_text("from pkg import absent\n", encoding="utf-8")

    assert check_export_leaks._unresolved_imports(tmp_path) == ["user.py: pkg.absent"]


_STUBBED = frozenset({"torch", "datasets", "cv2"})


def _tree(root: Path, pyproject: str | None) -> Path:
    """Write a tree importing two stubbed packages; return its root."""
    (root / "pkg").mkdir()
    (root / "pkg" / "model.py").write_text("import torch.nn as nn, os\n")
    (root / "pkg" / "data.pyi").write_text("from datasets import Dataset\n")
    (root / "typings" / "cv2").mkdir(parents=True)
    (root / "typings" / "cv2" / "__init__.pyi").write_text("import cv2.dnn\n")
    if pyproject is not None:
        (root / "pyproject.toml").write_text(pyproject)
    return root


@pytest.mark.parametrize(
    "pyproject",
    [
        '[project]\nname = "pkg"\ndependencies = ["rekursiv-ai-typings>=0.1"]\n',
        (
            '[project]\nname = "pkg"\n'
            '[project.optional-dependencies]\nlint = ["Rekursiv_AI-Typings"]\n'
        ),
        (
            '[project]\nname = "pkg"\n[dependency-groups]\n'
            'dev = [{include-group = "lint"}, "rekursiv-ai-typings ; python_version>\'3\'"]\n'
        ),
        '[project]\nname = "rekursiv-ai-typings"\n',
        '[dependency-groups]\ndev = ["rekursiv-ai-typings"]\n',
    ],
    ids=["dependency", "extra", "group", "itself", "no-project-table"],
)
def test_a_declared_stubs_dependency_satisfies_the_rule(
    tmp_path: Path,
    pyproject: str,
) -> None:
    assert (
        check_export_leaks._undeclared_stubs(_tree(tmp_path, pyproject), _STUBBED) == []
    )


def test_an_undeclared_one_names_each_stubbed_import(tmp_path: Path) -> None:
    """``.py`` and ``.pyi`` imports count; the tree's own ``typings/`` does not."""
    tree = _tree(
        tmp_path,
        '[project]\nname = "pkg"\ndependencies = ["rekursiv-ai-typings-extra"]\n'
        '[dependency-groups]\ndev = [{include-group = "x"}]\n',
    )

    assert check_export_leaks._undeclared_stubs(tree, _STUBBED) == ["datasets", "torch"]


def test_a_tree_with_no_pyproject_is_no_python_package(tmp_path: Path) -> None:
    assert check_export_leaks._undeclared_stubs(_tree(tmp_path, None), _STUBBED) == []


def test_import_lines_yield_each_top_level_name(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "import a.b as c, d\n  from e.f import g\nx = 'import h'\n# import i\n"
        "import j\tas k,  l as m\n",
    )

    assert check_export_leaks._imported_top_level_packages(tmp_path) == {
        "a",
        "d",
        "e",
        "j",
        "l",
    }


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
