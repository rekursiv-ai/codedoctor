"""Test mutation selection, sampling, snapshot layout, and result reporting."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.context import ForkContext
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast
from unittest.mock import Mock
from urllib import request

import argparse
import ast
import fnmatch
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib

from mutmut import configuration
from mutmut.mutation.file_mutation import create_mutations
from mutmut.mutation.pragma_handling import PragmaParseError
from mutmut.mutation.trampoline_templates import CLASS_NAME_SEPARATOR
from mutmut.state import state
from treekle import from_plain

import pytest

from codedoctor.mutation import mutated


if TYPE_CHECKING:
    from collections.abc import Iterator

from codedoctor.mutation.mutated import (
    Shard,
    ShardReport,
    _functions_in,
    _only_unobservable,
    copy_snapshot,
    fork_safe_env,
    function_filters,
    mutable_sources,
    mutant_outcomes,
    plan_shards,
    run_shard,
    sample,
    sample_size,
    single_threaded_env,
)


_CWD: Final = Path(__file__).resolve().parent


def _repo(tmp_path: Path, *, body: str) -> Path:
    """Return a git root holding ``pkg/mod.py`` with ``body``."""
    package = tmp_path / "pkg"
    package.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    _ = (package / "mod.py").write_text(body, encoding="utf-8")
    _ = (tmp_path / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["pkg"]\n',
        encoding="utf-8",
    )
    return tmp_path


def _runnable(root: Path, *, source_paths: str = '["pkg"]') -> None:
    """Configure ``root`` so mutmut can run its pytest suite in isolation."""
    (root / "pyproject.toml").write_text(
        f"[tool.mutmut]\nsource_paths = {source_paths}\n"
        "mutate_only_covered_lines = true\n"
        'pytest_add_cli_args = ["-o", "addopts="]\n'
        '[tool.pytest.ini_options]\npythonpath = ["."]\n',
    )
    (root / ".gitignore").write_text("runtime/\n")


def _report_for(root: Path, *, shard: Shard, **_: object) -> ShardReport:
    """Stand in for ``run_shard``: an empty report for ``shard``'s file."""
    del root
    return ShardReport(source=shard.source)


def _git(root: Path, *args: str) -> None:
    """Run git in ``root``."""
    subprocess.run(["git", *args], cwd=root, check=True)  # noqa: S603, S607 -- git from PATH; arguments are test literals.


def _template(root: Path, tmp_path: Path) -> Path:
    """Lay out ``root`` whole as the template shards clone, without git."""
    template = tmp_path / "template"
    shutil.copytree(root, template, symlinks=True)
    return template


_TWO_FUNCTIONS = """\
def first(value):
    return value + 1


def second(value):
    return value * 2
"""


@pytest.fixture(autouse=True)
def isolate_mutation_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """Keep the outer seed, git config, and project out, and snapshots inside.

    Mutmut's configuration is a process global read from the cwd, so each
    test starts in an empty project rather than wherever pytest was launched.

    Yields:
      temporary: Directory ``TemporaryDirectory`` snapshots are created in.

    """
    monkeypatch.delenv("PRECOMMIT_MUTATION_SEED", raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    temporary = tmp_path_factory.mktemp("snapshots")
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    project = tmp_path_factory.mktemp("project")
    (project / "pyproject.toml").write_text('[tool.mutmut]\nsource_paths = ["."]\n')
    monkeypatch.chdir(project)
    configuration.reset_config()
    # Under mutmut's stats phase every trampoline hit reads this; loaded lazily,
    # it would come from whichever directory a test had moved into.
    configuration.config()
    yield temporary
    configuration.reset_config()


def test_every_function_in_a_named_file_is_selected(tmp_path: Path) -> None:
    """Files in, functions out: no diff decides which functions count."""
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)

    assert function_filters(root, files=["pkg/mod.py"], sources=("pkg",)) == (
        "pkg.mod.x_first__mutmut_*",
        "pkg.mod.x_second__mutmut_*",
    )


def test_selection_restores_the_configuration_it_found(tmp_path: Path) -> None:
    """A mutmut session's stats phase keeps reading the configuration it loaded."""
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    before = configuration.config()

    assert function_filters(root, files=["pkg/mod.py"], sources=("pkg",))
    assert configuration.config() is before


def test_parallel_selection_matches_serial(tmp_path: Path) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    files = ["pkg/mod.py", "pkg/mod.py"]
    for name in ("a", "b", "c"):
        (root / f"pkg/{name}.py").write_text(
            f"def {name}(value):\n    return value + 1\n",
        )
        files.append(f"pkg/{name}.py")

    serial = function_filters(root, files=files, sources=("pkg",))

    assert function_filters(root, files=files, sources=("pkg",), workers=3) == serial
    assert serial == (
        "pkg.a.x_a__mutmut_*",
        "pkg.b.x_b__mutmut_*",
        "pkg.c.x_c__mutmut_*",
        "pkg.mod.x_first__mutmut_*",
        "pkg.mod.x_second__mutmut_*",
    )


def test_selection_never_runs_a_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"ran a subprocess: {args} {kwargs}")

    monkeypatch.setattr(subprocess, "run", refuse)

    assert function_filters(root, files=["pkg/mod.py"], sources=("pkg",))


def test_a_file_outside_source_paths_is_skipped(tmp_path: Path) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    _ = (root / "other.py").write_text(_TWO_FUNCTIONS, encoding="utf-8")

    assert function_filters(root, files=["other.py"], sources=("pkg",)) == ()


@pytest.mark.parametrize("name", ["mod_test.py", "test_mod.py", "conftest.py"])
def test_test_files_are_never_mutated(tmp_path: Path, name: str) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    _ = (root / "pkg" / name).write_text(_TWO_FUNCTIONS, encoding="utf-8")

    assert function_filters(root, files=[f"pkg/{name}"], sources=("pkg",)) == ()


def test_a_method_is_named_with_its_class(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        body="class Holder:\n    def method(self, value):\n        return value + 1\n",
    )

    assert function_filters(root, files=["pkg/mod.py"], sources=("pkg",)) == (
        "pkg.mod.xǁHolderǁmethod__mutmut_*",
    )


def test_the_class_separator_default_matches_mutmuts_own(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("class Holder:\n    def method(self):\n        return 1\n")
    assert _functions_in(path) == (
        f"x{CLASS_NAME_SEPARATOR}Holder{CLASS_NAME_SEPARATOR}method",
    )


@pytest.mark.parametrize("identifier", ["K", "ｆ", "e\u0301", "fǁ"])
@pytest.mark.parametrize(
    "declaration",
    [
        "def {identifier}():\n    return 1\n",
        "class {identifier}:\n    def method(self):\n        return 1\n",
        "class Holder:\n    def {identifier}(self):\n        return 1\n",
    ],
)
def test_unsupported_identifiers_report_the_source(
    tmp_path: Path,
    identifier: str,
    declaration: str,
) -> None:
    path = tmp_path / "mod.py"
    path.write_bytes(
        (
            declaration.format(identifier=identifier)
            + "def ordinary():\n    return 2\n"
        ).encode(),
    )

    with pytest.raises(ValueError, match=re.escape(str(path))) as raised:
        _functions_in(path)
    assert str(raised.value).startswith(
        f"{path}: mutmut does not support identifier ",
    )
    assert identifier in str(raised.value)


@pytest.mark.parametrize(
    ("patterns", "expected"),
    [("[]", ("pkg.mod.x_first__mutmut_*",)), ('["return"]', ())],
)
def test_mutation_patterns_come_from_the_supplied_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patterns: str,
    expected: tuple[str, ...],
) -> None:
    root = tmp_path / "root"
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "mod.py").write_text("def first():\n    return 1\n")
    (root / "pyproject.toml").write_text(
        f'[tool.mutmut]\nsource_paths = ["pkg"]\ndo_not_mutate_patterns = {patterns}\n',
    )
    child = root / "other"
    child.mkdir()
    (child / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["."]\ndo_not_mutate_patterns = ["return"]\n',
    )
    monkeypatch.chdir(child)
    configuration.reset_config()
    try:
        configuration.config()
        assert (
            function_filters(root, files=["pkg/mod.py"], sources=("pkg",)) == expected
        )
        assert Path.cwd() == child
        assert configuration.config().do_not_mutate_patterns == ["return"]
    finally:
        configuration.reset_config()


@pytest.mark.parametrize(
    "setting",
    ['do_not_mutate = ["blocked.py"]', 'only_mutate = ["allowed.py"]'],
)
def test_file_exclusions_do_not_consume_the_function_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.mutmut]\nsource_paths = ["."]\n{setting}\n',
    )
    for name in ("blocked.py", "allowed.py"):
        (tmp_path / name).write_text("def first():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    configuration.reset_config()
    try:
        assert function_filters(
            tmp_path,
            files=["blocked.py", "allowed.py"],
            sources=(".",),
        ) == ("allowed.x_first__mutmut_*",)
    finally:
        configuration.reset_config()


def test_a_nested_function_selects_its_enclosing_method(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        body=(
            "class Holder:\n"
            "    def method(self, value):\n"
            "        def helper(inner):\n"
            "            return inner + 1\n"
            "        return helper(value)\n"
        ),
    )

    assert function_filters(root, files=["pkg/mod.py"], sources=("pkg",)) == (
        "pkg.mod.xǁHolderǁmethod__mutmut_*",
    )


def test_a_nested_classes_method_selects_nothing(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        body=(
            "class Outer:\n"
            "    class Inner:\n"
            "        def deep(self, value):\n"
            "            return value + 1\n"
        ),
    )

    assert function_filters(root, files=["pkg/mod.py"], sources=("pkg",)) == ()


@pytest.mark.parametrize("source", ["pkg", "pkg/", "./pkg"])
def test_source_path_spellings_select_the_same_function(
    tmp_path: Path,
    source: str,
) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    (root / "pkg/fresh.py").write_text(_TWO_FUNCTIONS)
    assert function_filters(root, files=["./pkg/fresh.py"], sources=(source,)) == (
        "pkg.fresh.x_first__mutmut_*",
        "pkg.fresh.x_second__mutmut_*",
    )


@pytest.mark.parametrize("decorator", ["classmethod", "staticmethod"])
def test_a_class_or_static_method_is_selected(
    tmp_path: Path,
    decorator: str,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        f"class Holder:\n    @{decorator}\n    def first(cls):\n        return 1\n",
    )
    assert _functions_in(path) == ("xǁHolderǁfirst",)


def test_function_filter_excludes_a_longer_function_name(tmp_path: Path) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    (root / "pkg/fresh.py").write_text("def foo():\n    return 1\n")
    filters = function_filters(root, files=["pkg/fresh.py"], sources=("pkg",))
    assert any(fnmatch.fnmatch("pkg.fresh.x_foo__mutmut_1", p) for p in filters)
    assert not any(fnmatch.fnmatch("pkg.fresh.x_foobar__mutmut_1", p) for p in filters)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_utf8_names_are_selected_with_every_source_newline(
    tmp_path: Path,
    newline: str,
) -> None:
    path = tmp_path / "mod.py"
    source = (
        "def café():\n    return 1\n\n"
        "class Holder:\n    def méthod(self):\n        return 2\n"
    )
    path.write_bytes(source.replace("\n", newline).encode())

    assert _functions_in(path) == (
        "x_café",
        "xǁHolderǁméthod",
    )


def test_invalid_mutation_directives_name_the_source_file(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("# pragma: no mutate end\ndef first():\n    return 1\n")

    with pytest.raises(PragmaParseError, match=re.escape(str(path))):
        _functions_in(path)


@pytest.mark.parametrize("prefix", ["def first(value):", "async def first(value):"])
@pytest.mark.parametrize("docstring", ["", '    """Contract."""\n'])
@pytest.mark.parametrize("body", ["...", "pass", "return value", "return ..."])
def test_declarations_do_not_consume_the_function_budget(
    tmp_path: Path,
    prefix: str,
    docstring: str,
    body: str,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        f"{prefix}\n{docstring}    {body}\n\n"
        "class Contract:\n    def method(self):\n        ...\n\n"
        "def implemented():\n    return 1\n",
    )

    assert _functions_in(path) == ("x_implemented",)


@pytest.mark.parametrize("body", ["...", "pass", '"""Contract."""\n        ...'])
def test_a_stub_with_mutable_defaults_is_not_selected(
    tmp_path: Path,
    body: str,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        "class Contract:\n"
        f"    def stub(self, flag=False, prefix=''):\n        {body}\n\n"
        "    def real(self, flag=False):\n        return flag\n",
    )

    assert _functions_in(path) == ("xǁContractǁreal",)


@pytest.mark.parametrize(
    "annotation",
    ["language.tensor", "'tuple[language.tensor, ...]'", "language.tuple"],
)
def test_triton_kernel_bodies_are_not_selected(
    tmp_path: Path,
    annotation: str,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        f"def _body(ptr: {annotation}, n: int):\n    return ptr + n\n\n"
        "def build(language_tensor: int):\n    return language_tensor + 1\n",
    )

    assert _functions_in(path) == ("x_build",)


@pytest.mark.parametrize("body", ["...\n    return 1", "return ..."])
def test_executable_bodies_containing_ellipsis_are_selected(
    tmp_path: Path,
    body: str,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(f'def implemented(value=1):\n    """Contract."""\n    {body}\n')

    assert _functions_in(path) == ("x_implemented",)


@pytest.mark.parametrize("decoration", ["@cache", "@property", "@other\n@classmethod"])
def test_unsupported_decorators_do_not_consume_the_function_budget(
    tmp_path: Path,
    decoration: str,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        f"{decoration}\ndef first(value):\n    return value + 1\n\n"
        "def second(value):\n    return value * 2\n",
    )
    assert _functions_in(path) == ("x_second",)


@pytest.mark.parametrize("name", ["__new__", "__setattr__", "__getattribute__"])
def test_unsupported_special_methods_are_not_selected(
    tmp_path: Path,
    name: str,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        f"class Holder:\n    def {name}(self, value):\n        return value + 1\n",
    )
    assert _functions_in(path) == ()


def test_a_file_that_does_not_parse_selects_nothing(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("def broken(:\n")

    assert _functions_in(path) == ()


def test_a_directory_without_a_config_has_no_mutable_sources(tmp_path: Path) -> None:
    assert mutable_sources(tmp_path) == ()


@pytest.mark.parametrize("config", ['[project]\nname = "example"\n', "[tool]\n"])
def test_missing_mutmut_tables_select_no_sources(tmp_path: Path, config: str) -> None:
    (tmp_path / "pyproject.toml").write_text(config)

    assert mutable_sources(tmp_path) == ()


@pytest.mark.parametrize(
    "config",
    [
        "tool = 3\n",
        "[tool]\nmutmut = 3\n",
        '[tool.mutmut]\nsource_paths = "pkg"\n',
        '[tool.mutmut]\nsource_paths = ["pkg", 1, "lib"]\n',
    ],
)
def test_a_malformed_mutmut_table_is_an_error(tmp_path: Path, config: str) -> None:
    """A wrong shape is the author's typo, never silently fewer sources."""
    (tmp_path / "pyproject.toml").write_text(config)

    with pytest.raises(TypeError):
        mutable_sources(tmp_path)


def test_a_non_string_source_path_names_the_table(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["pkg", 1]\n',
    )

    with pytest.raises(TypeError) as raised:
        mutable_sources(tmp_path)
    assert str(raised.value) == (
        "[tool.mutmut] source_paths must be strings: ['pkg', 1]"
    )


class TestSample:
    """Sample functions evenly and stably."""

    filters: Final = tuple(
        f"{module}.x_{function}__mutmut_*"
        for module in ("a", "b", "c", "d", "e")
        for function in ("f", "g", "h", "i")
    )

    def test_every_function_can_be_sampled_whatever_its_module(self) -> None:
        """Taking the alphabetically first names would only ever pick module a."""
        picked = {
            name
            for seed in range(100)
            for name in sample(self.filters, seed=str(seed), size=4)
        }

        assert picked == set(self.filters)

    def test_a_seed_always_picks_the_same_functions(self) -> None:
        picks = sample(self.filters, seed="510", size=4)

        assert len(picks) == 4
        assert picks == tuple(sorted(picks))
        assert sample(tuple(reversed(self.filters)), seed="510", size=4) == picks

    def test_a_new_function_displaces_at_most_one_pick(self) -> None:
        """Redrawing with ``random.sample`` would replace most of the picks."""
        for seed in map(str, range(50)):
            before = set(sample(self.filters, seed=seed, size=4))
            after = set(
                sample((*self.filters, "z.x_new__mutmut_*"), seed=seed, size=4),
            )

            assert len(before - after) <= 1

    def test_a_small_change_keeps_every_function(self) -> None:
        assert sample(self.filters[:3], seed="", size=4) == self.filters[:3]

    @pytest.mark.parametrize("size", [1, 7])
    def test_the_size_is_the_sample_size(self, size: int) -> None:
        assert len(sample(self.filters, seed="", size=size)) == size

    def test_a_larger_size_extends_a_smaller_ones_picks(self) -> None:
        """Bottom-k: raising the count adds functions; it never swaps them."""
        small = set(sample(self.filters, seed="510", size=2))

        assert small < set(sample(self.filters, seed="510", size=5))


@pytest.mark.parametrize(
    ("candidates", "rate", "most", "size"),
    [
        (30, 0.1, None, 3),
        (59, 0.05, None, 3),
        (459, 0.05, 4, 4),
        (459, 0.05, None, 23),
        (1, 0.05, 4, 1),
        (0, 0.05, 4, 0),
        (6, 1.0, None, 6),
        (6, 1.0, 2, 2),
    ],
)
def test_the_sample_size_is_the_rounded_up_rate_under_the_cap(
    candidates: int,
    rate: float,
    most: int | None,
    size: int,
) -> None:
    assert sample_size(candidates, rate=rate, most=most) == size


class TestMutantOutcomes:
    """Classify mutmut's recorded exit codes for the selected functions."""

    def test_killed_and_timed_out_mutants_need_no_attention(self) -> None:
        outcomes = mutant_outcomes(
            {"m.x_f__mutmut_1": 1, "m.x_f__mutmut_2": 36, "m.x_f__mutmut_3": 37},
            filters=("m.x_f__mutmut_*",),
        )

        assert not any(outcomes.values())

    def test_each_failure_mode_is_grouped(self) -> None:
        outcomes = mutant_outcomes(
            {
                "m.x_f__mutmut_1": 0,
                "m.x_f__mutmut_2": 33,
                "m.x_f__mutmut_3": None,
                "m.x_f__mutmut_4": -11,
            },
            filters=("m.x_f__mutmut_*",),
        )

        assert outcomes == {
            "survived": ("m.x_f__mutmut_1",),
            "no tests": ("m.x_f__mutmut_2",),
            "inconclusive": (
                "m.x_f__mutmut_3 (not checked, exit None)",
                "m.x_f__mutmut_4 (segfault, exit -11)",
            ),
            "unchecked": (),
        }

    def test_unselected_functions_sharing_a_prefix_are_ignored(self) -> None:
        outcomes = mutant_outcomes(
            {"m.x_f__mutmut_1": 1, "m.x_foo__mutmut_1": 0, "m.x_foo__mutmut_2": None},
            filters=("m.x_f__mutmut_*",),
        )

        assert not any(outcomes.values())

    def test_a_killed_mutant_does_not_hide_the_ones_after_it(self) -> None:
        outcomes = mutant_outcomes(
            {"m.x_f__mutmut_1": 1, "m.x_f__mutmut_2": 0},
            filters=("m.x_f__mutmut_*",),
        )

        assert outcomes["survived"] == ("m.x_f__mutmut_2",)

    def test_a_function_without_a_run_mutant_is_unchecked(self) -> None:
        outcomes = mutant_outcomes(
            {"m.x_f__mutmut_1": 1},
            filters=("m.x_f__mutmut_*", "m.x_g__mutmut_*"),
        )

        assert outcomes["unchecked"] == ("m.x_g__mutmut_*",)


class _MutmutCli(Protocol):
    def tests_for_mutant_names(self, mutant_names: list[str], /) -> set[str]: ...


def test_selectors_run_clean_tests_for_only_the_exact_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(return_value=subprocess.CompletedProcess([], 1, "", ""))
    with monkeypatch.context() as patch:
        patch.setattr(mutated, "_run_mutmut", run)
        patch.setattr(mutated, "copy_snapshot", Mock())
        patch.setattr(mutated, "_paired_tests", Mock(return_value=("t.py",)))
        run_shard(
            tmp_path,
            shard=Shard(source="pkg/mod.py", filters=("pkg.mod.x_foo__mutmut_*",)),
            snapshot=tmp_path / "snapshot",
            template=tmp_path,
            workers=1,
        )
    argv = cast("list[str]", run.call_args.args[0])
    # Importing mutmut's CLI reads its config from the cwd, so it waits for the
    # fixture's project rather than running wherever pytest collects.
    cli = cast(_MutmutCli, importlib.import_module("mutmut.__main__"))
    # Restore before teardown: under mutmut this state is live, and its stats
    # hook writes to it when the test ends.
    with monkeypatch.context() as patch:
        patch.setattr(
            state(),
            "tests_by_mangled_function_name",
            {"pkg.mod.x_foo": {"test_foo"}, "pkg.mod.x_foobar": {"test_foobar"}},
        )
        tests = cli.tests_for_mutant_names(
            argv[argv.index("--max-children") + 2 :],
        )

    assert tests == {"test_foo"}


@pytest.mark.cli_git
def test_the_template_holds_every_file_git_lists(tmp_path: Path) -> None:
    """Tests read tracked dotfiles and force-added files; ignored trees stay out."""
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)
    (root / ".gitignore").write_text("runtime/\n.venv/\n")
    (root / ".pre-commit-config.yaml").write_text("repos: []\n")
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/ci.yml").write_text("on: push\n")
    (root / "runtime").mkdir()
    (root / "runtime/data").write_text("ignored")
    (root / "runtime/kept.py").write_text("force-added")
    (root / ".venv").mkdir()
    (root / ".venv/site.py").write_text("ignored")
    (root / "pkg/gone.py").write_text("deleted, still in the index")
    (root / "pkg/link.py").symlink_to("mod.py")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "add", "-f", "runtime/kept.py")
    (root / "pkg/gone.py").unlink()
    (root / "support.txt").write_text("untracked")
    template = tmp_path / "template"

    mutated.copy_template(root, template=template)

    assert {
        path.relative_to(template).as_posix()
        for path in template.rglob("*")
        if path.is_symlink() or path.is_file()
    } == {
        ".github/workflows/ci.yml",
        ".gitignore",
        ".pre-commit-config.yaml",
        "pkg/link.py",
        "pkg/mod.py",
        "pyproject.toml",
        "runtime/kept.py",
        "support.txt",
    }
    assert (template / "pkg/link.py").is_symlink()


def test_the_snapshot_mutates_sources_and_copies_the_rest_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)
    _runnable(root)
    (root / "support.txt").write_text("support")
    (root / "pkg/link.py").symlink_to("mod.py")
    template = _template(root, tmp_path)
    snapshot = tmp_path / "snapshot"
    run = Mock(wraps=subprocess.run)
    monkeypatch.setattr(subprocess, "run", run)

    copy_snapshot(
        root,
        snapshot=snapshot,
        sources=("pkg/mod.py",),
        tests=("pkg/mod_test.py",),
        template=template,
    )

    (copy,) = run.call_args_list
    assert cast("list[str]", copy.args[0])[0] == shutil.which("cp")
    assert (snapshot / "pkg/mod.py").is_file()
    assert not (snapshot / "mutants/pkg/mod.py").exists()
    assert (snapshot / "mutants/support.txt").read_text() == "support"
    assert (snapshot / "mutants/pkg/link.py").is_symlink()
    assert not (snapshot / "support.txt").exists()
    for config in (snapshot, snapshot / "mutants"):
        assert mutable_sources(config) == ("pkg/mod.py",)
    assert not (root / "mutants").exists()


@pytest.mark.parametrize("absolute", [False, True])
def test_selected_source_symlinks_are_isolated_readable_files(
    tmp_path: Path,
    absolute: bool,
) -> None:
    root = tmp_path / "repo"
    package = root / "pkg"
    package.mkdir(parents=True)
    target = package / "mod.py"
    body = b"def first():\n    return 1\n"
    target.write_bytes(body)
    (package / "link.py").symlink_to(target if absolute else "mod.py")
    (root / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["pkg"]\n',
    )
    snapshot = tmp_path / "snapshot"

    copy_snapshot(
        root,
        snapshot=snapshot,
        sources=("pkg/link.py",),
        tests=("pkg/link_test.py",),
        template=_template(root, tmp_path),
    )

    copied = snapshot / "pkg/link.py"
    assert copied.read_bytes() == body
    assert not copied.is_symlink()
    copied.write_bytes(b"isolated")
    assert target.read_bytes() == body


@pytest.mark.cli_git
@pytest.mark.cli_python_subprocess
def test_a_selected_symlink_runs_its_paired_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path, body="def first(value):\n    return value + 1\n")
    _runnable(root)
    _git(root, "init", "-q")
    (root / "pkg/link.py").symlink_to("mod.py")
    (root / "pkg/link_test.py").write_text(
        "from pkg.link import first\ndef test_first():\n    assert first(2) == 3\n",
    )
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mutated", "pkg/link.py"])

    assert mutated.main() == 0
    assert not (root / "mutants").exists()


def test_emitting_filters_samples_without_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    (root / "pyproject.toml").write_text('[tool.mutmut]\nsource_paths = ["."]\n')
    paths = ["other.py", "pkg/fresh.py", "script.py"]
    for path in paths:
        (root / path).write_text(_TWO_FUNCTIONS)
    monkeypatch.chdir(root)
    run = Mock()
    monkeypatch.setattr(mutated, "run_shard", run)
    monkeypatch.setattr(
        sys,
        "argv",
        # One worker: the sample is what is measured, and forking cost 1s of it.
        [
            "mutated",
            "--emit",
            "filters",
            "--max-functions-sampled",
            "4",
            "--workers",
            "1",
            *paths,
        ],
    )

    assert mutated.main() == 0

    output = capsys.readouterr()
    candidates = function_filters(root, files=paths, sources=(".",))
    assert tuple(output.out.splitlines()) == sample(candidates, seed="", size=4)
    assert "4 selected; 2 skipped" in output.err
    run.assert_not_called()
    assert not (root / "mutants").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        ["mutated", "--emit", "filters", "--function-sample-rate", "0.1", *paths],
    )

    assert mutated.main() == 0
    output = capsys.readouterr()
    assert len(output.out.splitlines()) == 1
    assert "[mutated] 1 selected; 5 skipped (seed '')\n" in output.err

    monkeypatch.setattr(
        sys,
        "argv",
        ["mutated", "--emit", "filters", "--seed", "7", *paths],
    )

    assert mutated.main() == 0
    output = capsys.readouterr()
    assert len(output.out.splitlines()) == 6
    assert "skipped" not in output.err


def test_the_root_is_the_directory_of_the_nearest_pyproject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run from a subdirectory: paths are cwd-relative, filters project-relative."""
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    monkeypatch.chdir(root / "pkg")
    monkeypatch.setattr(sys, "argv", ["mutated", "--emit", "filters", "mod.py"])

    assert mutated.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "pkg.mod.x_first__mutmut_*",
        "pkg.mod.x_second__mutmut_*",
    ]


@pytest.mark.parametrize("argv", [[], ["."], ["pkg"], ["pkg/"]])
def test_a_directory_mutates_the_python_files_under_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    (root / "pkg/mod_test.py").write_text("def test_f():\n    pass\n")
    (root / "pkg/notes.txt").write_text("def first(): ...\n")
    (root / ".gitignore").write_text("pkg/ignored.py\n")
    (root / "pkg/ignored.py").write_text(_TWO_FUNCTIONS)
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mutated", "--emit", "filters", *argv])

    assert mutated.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "pkg.mod.x_first__mutmut_*",
        "pkg.mod.x_second__mutmut_*",
    ]


def test_an_empty_worker_count_is_serial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    flags, _ = mutated._parse_args(argparse.ArgumentParser(), argv=["--workers", ""])

    assert flags.workers == 1


def test_a_file_outside_the_project_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)
    (tmp_path / "other.py").write_text("def f():\n    return 1\n")
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mutated", "--emit", "filters", "../other.py"])

    assert mutated.main() == 1
    assert "cannot select functions" in capsys.readouterr().err


def test_flag_defaults_come_from_the_pyproject_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    with (root / "pyproject.toml").open("a") as handle:
        handle.write(
            "[tool.mutated]\nfunction-sample-rate = 0.5\nmax-functions-sampled = 1\n"
            'seed = "fixed"\n',
        )
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mutated", "--emit", "filters", "pkg/mod.py"])

    assert mutated.main() == 0
    picked = capsys.readouterr().out.splitlines()
    assert picked == list(
        sample(
            ("pkg.mod.x_first__mutmut_*", "pkg.mod.x_second__mutmut_*"),
            seed="fixed",
            size=1,
        ),
    )


def test_there_is_no_diff_base_flag() -> None:
    parser = argparse.ArgumentParser()
    mutated._add_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--base", "HEAD"])


def test_invalid_emit_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mutated", "--emit", "typo"])

    with pytest.raises(SystemExit) as error:
        mutated.main()

    assert error.value.code == 2
    output = capfd.readouterr()
    assert output.out == ""
    assert "invalid choice" in output.err
    assert not (root / "mutants").exists()


@pytest.mark.cli_git
@pytest.mark.cli_python_subprocess
def test_missing_paired_tests_fails_without_running_a_repository_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    _git(root, "init", "-q")
    (root / "pkg/fresh.py").write_text(_TWO_FUNCTIONS)
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mutated", "--blocking", "pkg/fresh.py"])

    assert mutated.main() == 1
    captured = capsys.readouterr()
    assert "mutation: pkg/fresh.py: no paired tests found\n" in captured.err
    assert "these files recorded no results" in captured.out
    assert "  pkg/fresh.py\n" in captured.out
    assert not (root / "mutants").exists()


@pytest.mark.cli_git
@pytest.mark.cli_python_subprocess
def test_a_surviving_mutant_is_reported_with_its_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    isolate_mutation_environment: Path,
) -> None:
    """Run real mutmut so a change to its result format fails here, not in CI."""
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    _runnable(root)
    _git(root, "init", "-q")
    (root / "pkg/__init__.py").touch()
    (root / "pkg/mod.py").write_text(_TWO_FUNCTIONS.replace("+ 1", "+ 0"))
    # Asserting only the type lets every arithmetic mutant survive.
    (root / "pkg/mod_test.py").write_text(
        "from pkg.mod import first\n"
        "def test_first():\n    assert isinstance(first(2), int)\n",
    )
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mutated", "--blocking", "pkg/mod.py"])

    assert mutated.main() == 1

    out = capsys.readouterr().out
    assert "survived the selected tests" in out
    assert "  pkg.mod.x_first__mutmut_1\n" in out
    assert (
        "# pkg.mod.x_first__mutmut_1: survived\n--- pkg/mod.py\n+++ pkg/mod.py\n" in out
    )
    assert "-    return value + 0" in out
    assert not (root / "mutants").exists()
    assert not any(isolate_mutation_environment.iterdir())


@pytest.mark.cli_git
@pytest.mark.cli_python_subprocess
def test_every_selected_function_must_be_exercised(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)
    _runnable(root)
    (root / "pkg/__init__.py").touch()
    (root / "pkg/mod_test.py").write_text(
        "from pkg.mod import first\ndef test_first():\n    assert first(2) == 3\n",
    )

    report = run_shard(
        root,
        shard=Shard(
            source="pkg/mod.py",
            filters=("pkg.mod.x_first__mutmut_*", "pkg.mod.x_second__mutmut_*"),
        ),
        snapshot=tmp_path / "snapshot",
        template=_template(root, tmp_path),
        workers=1,
    )

    assert report.failure == ""
    assert report.outcomes["unchecked"] == ()
    assert report.outcomes["no tests"]
    assert all(
        name.startswith("pkg.mod.x_second__mutmut_")
        for name in report.outcomes["no tests"]
    )
    assert not any(
        name.startswith("pkg.mod.x_first__mutmut_")
        for names in report.outcomes.values()
        for name in names
    )

    mutated._progress(report, done=1, total=1)
    summary = capsys.readouterr().err
    assert "no tests" in summary
    assert "unchecked" not in summary


@pytest.mark.cli_git
@pytest.mark.cli_python_subprocess
def test_a_function_whose_mutants_are_all_killed_passes(tmp_path: Path) -> None:
    root = _repo(
        tmp_path / "repo",
        body="def foo(value):\n    return value + 1\n\n"
        "def foobar(value):\n    return value + 2\n",
    )
    _runnable(root)
    (root / "pkg/__init__.py").touch()
    (root / "pkg/mod_test.py").write_text(
        "from pkg.mod import foo\ndef test_foo():\n    assert foo(2) == 3\n",
    )

    report = run_shard(
        root,
        shard=Shard(source="pkg/mod.py", filters=("pkg.mod.x_foo__mutmut_*",)),
        snapshot=tmp_path / "snapshot",
        template=_template(root, tmp_path),
        workers=1,
    )

    assert report == ShardReport(
        source="pkg/mod.py",
        outcomes=dict.fromkeys(
            ("no tests", "inconclusive", "survived", "unchecked"),
            (),
        ),
        checked=report.checked,
    )
    assert report.checked > 0
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.cli_git
@pytest.mark.cli_python_subprocess
def test_a_mutant_confined_to_a_cast_type_is_not_a_survivor(tmp_path: Path) -> None:
    root = _repo(
        tmp_path / "repo",
        body="from typing import cast\n\n"
        "def foo(value):\n    return cast(\n        int,\n        value,\n    ) + 1\n",
    )
    _runnable(root)
    (root / "pkg/__init__.py").touch()
    (root / "pkg/mod_test.py").write_text(
        "from pkg.mod import foo\ndef test_foo():\n    assert foo(2) == 3\n",
    )

    report = run_shard(
        root,
        shard=Shard(source="pkg/mod.py", filters=("pkg.mod.x_foo__mutmut_*",)),
        snapshot=tmp_path / "snapshot",
        template=_template(root, tmp_path),
        workers=1,
    )

    assert report.failure == ""
    assert report.outcomes["survived"] == ()
    assert report.checked > 1


@pytest.mark.cli_git
@pytest.mark.cli_python_subprocess
def test_an_unreachable_docstring_fallback_is_not_a_survivor(tmp_path: Path) -> None:
    root = _repo(
        tmp_path / "repo",
        body='"""Title.\n\nBody.\n"""\n\n'
        'def describe():\n    return (__doc__ or "").split("\\n", 2)[2]\n',
    )
    _runnable(root)
    (root / "pkg/__init__.py").touch()
    (root / "pkg/mod_test.py").write_text(
        "from pkg.mod import describe\n"
        "def test_describe():\n    assert describe() == 'Body.\\n'\n",
    )

    report = run_shard(
        root,
        shard=Shard(source="pkg/mod.py", filters=("pkg.mod.x_describe__mutmut_*",)),
        snapshot=tmp_path / "snapshot",
        template=_template(root, tmp_path),
        workers=1,
    )

    assert report.failure == ""
    assert report.outcomes["survived"] == ()
    assert report.checked > 1


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("x = cast(Foo.Bar, y)", "x = cast(None, y)", True),
        (
            "x = cast(\n    Foo,\n    load(y),\n)",
            "x = cast(\n    None,\n    load(y),\n)",
            True,
        ),
        ("x = typing.cast(Foo, y)", "x = typing.cast(None, y)", True),
        ("x = cast(Foo, y)", "x = cast(Foo, y)", False),
        ("x = cast(dict[str, int], y)", "x = cast(None, y)", True),
        ('x = cast("list[int]", y)', "x = cast(None, y)", True),
        ("x = cast(Foo, y)", "x = cast(Foo, None)", False),
        ("x = cast(Foo, y) + 1", "x = cast(Foo, y) - 1", False),
        ("x = f(Foo, y)", "x = f(None, y)", False),
        (
            'p.add("--n", type=int, help="Rows.")',
            'p.add("--n", type=int, help="XXRows.XX")',
            True,
        ),
        ("p.add('--n', help='Rows.')", "p.add('--n', help=None)", True),
        ("p.add('--n', help='Rows.')", "p.add('--n', help=SUPPRESS)", False),
        (
            'p.add("--n", default=8, help="Rows.")',
            'p.add("--n", default=9, help="Rows.")',
            False,
        ),
        ('p.add("--n", help="a" "b")', 'p.add("--n", help="a" "XXbXX")', True),
        ('p.add("--n", help="Rows.")', 'p.add("--N", help="Rows.")', False),
        # ``usedforsecurity`` only marks a digest's purpose; the digest is the same.
        (
            "h = hashlib.sha1(b, usedforsecurity=False)",
            "h = hashlib.sha1(b, usedforsecurity=True)",
            True,
        ),
        ("h = hashlib.sha1(b, usedforsecurity=False)", "h = hashlib.sha1(b, )", True),
        (
            "h = hashlib.sha1(b, usedforsecurity=False)",
            "h = hashlib.sha1(None, usedforsecurity=False)",
            False,
        ),
        # A falsy docstring raises IndexError with either fallback.
        (
            'd = (__doc__ or "").split("\\n", 2)[2]',
            'd = (__doc__ or "XXXX").split("\\n", 2)[2]',
            True,
        ),
        (
            'd = (__doc__ or "a\\nb\\nc").split("\\n", 2)[2]',
            'd = (__doc__ or "XXa\\nb\\ncXX").split("\\n", 2)[2]',
            False,
        ),
        (
            'd = (__doc__ or "").split("\\n", 2)[2]',
            'd = (__doc__ or "a\\nb\\nIndexError").split("\\n", 2)[2]',
            False,
        ),
        (
            'd = (__doc__ or "").split("\\n")[0]',
            'd = (__doc__ or "XXXX").split("\\n")[0]',
            False,
        ),
        ('d = __doc__ or ""', 'd = __doc__ or "XXXX"', False),
        (
            'd = (__doc__ or "").split("\\n", 2)[2]',
            'd = (__doc__ or "").rsplit("\\n", 2)[2]',
            False,
        ),
        (
            'd = (__doc__ or "").split("\\n", 2)[2]',
            'd = (__doc__ and "").split("\\n", 2)[2]',
            False,
        ),
        # An empty separator raises ValueError, which the gate must not hit.
        ('d = (x or "").split("")[2]', 'd = (x or "XXXX").split("")[2]', False),
        (
            'd = (x or "").split("\\n", maxsplit=2)[2]',
            'd = (x or "XXXX").split("\\n", maxsplit=2)[2]',
            False,
        ),
    ],
)
def test_only_unobservable_mutants_are_dropped(
    before: str,
    after: str,
    expected: bool,
) -> None:
    assert _only_unobservable(before, after) is expected


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("typing.cast(int, x)", "cast"),
        ("cast(int, x)", "cast"),
        ("casts[0](int, x)", ""),
        ("make()(int, x)", ""),
    ],
)
def test_the_callee_is_the_bare_name_or_empty(source: str, name: str) -> None:
    """A callee that is neither a name nor an attribute names nothing."""
    call = ast.parse(source, mode="eval").body
    assert isinstance(call, ast.Call)
    assert mutated._callee(call) == name


@pytest.mark.cli_git
@pytest.mark.cli_python_subprocess
def test_broad_run_uses_paired_tests_and_unmutated_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(
        tmp_path,
        body=(
            "from functools import cache\nfrom helper import increment\n"
            "@cache\ndef earlier(x):\n    return x + 1\n"
            "def foo(x):\n    return increment(x) + 1\n"
        ),
    )
    _runnable(root, source_paths='["."]')
    _git(root, "init", "-q")
    with (root / "pyproject.toml").open("a") as handle:
        handle.write('[tool.coverage.run]\nomit = ["*/pkg/*"]\n')
    (root / "pkg/__init__.py").touch()
    (root / "pkg/link.py").symlink_to("mod.py")
    (root / "package_link").symlink_to("pkg", target_is_directory=True)
    (root / "helper.py").write_text(
        "from pathlib import Path\ndef increment(x):\n"
        "    return x + int(Path('support/offset.txt').read_text())\n",
    )
    (root / "support").mkdir()
    (root / "support/offset.txt").write_text("1")
    (root / "conftest.py").write_text(
        "import pytest\n@pytest.fixture\ndef number():\n    return 2\n",
    )
    (root / "pkg/mod_test.py").write_text(
        "from pkg.mod import foo\nfrom pathlib import Path\ndef test_foo(number):\n"
        "    assert foo(number) == 5\n"
        "    assert Path('pkg/link.py').is_symlink()\n"
        "    assert Path('package_link').is_symlink()\n",
    )
    (root / "unrelated_test.py").write_text("raise RuntimeError('must not collect')\n")
    (root / "pkg/mod.py").write_text(
        (root / "pkg/mod.py").read_text().replace("+ 1", "+ 2"),
    )
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mutated", "pkg/mod.py"])

    assert mutated.main() == 0
    assert not (root / "mutants").exists()


@pytest.mark.cli_git
@pytest.mark.cli_python_subprocess
@pytest.mark.parametrize("module", ["pkg/__init__.py", "docs.example/mod.py"])
def test_source_paths_survive_mutant_name_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: str,
) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    _runnable(root, source_paths='["."]')
    _git(root, "init", "-q")
    source = root / module
    source.parent.mkdir(exist_ok=True)
    source.write_text("def foo(x):\n    return x + 1\n")
    import_name = module.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
    source.with_name(f"{source.stem}_test.py").write_text(
        "import importlib.util\nfrom pathlib import Path\n"
        f"spec = importlib.util.spec_from_file_location('{import_name}', Path(__file__).with_name('{source.name}'))\n"
        "sample = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(sample)\n"
        "def test_foo():\n    assert sample.foo(2) == 3\n",
    )
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mutated", module])

    # Zero also proves a mutant of foo ran: an unchecked function fails.
    assert mutated.main() == 0


def test_main_emits_selected_filters_without_running_mutmut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The explain and emit branches report selected candidates."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mutated, "mutable_sources", Mock(return_value=("pkg",)))
    monkeypatch.setattr(
        mutated,
        "function_filters",
        Mock(return_value=("pkg.mod.x_f__mutmut_*",)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mutated", "--emit", "filters", "--explain", "pkg/mod.py"],
    )
    assert mutated.main() == 0
    output = capfd.readouterr()
    assert output.out == "pkg.mod.x_f__mutmut_*\n"
    assert "1 file(s) -> 1 function(s)" in output.err


@pytest.mark.parametrize(
    ("most", "message"),
    [("1", "[mutated] 1 selected; 2 skipped (seed 'S')\n"), ("3", "")],
)
def test_main_reports_how_many_functions_the_sample_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    most: str,
    message: str,
) -> None:
    root = _repo(
        tmp_path,
        body="".join(f"def {n}(x):\n    return x + 1\n" for n in "abc"),
    )
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mutated",
            *("--emit", "filters", "--seed", "S"),
            *("--max-functions-sampled", most, "pkg/mod.py"),
        ],
    )

    assert mutated.main() == 0
    assert capsys.readouterr().err == message


@pytest.mark.parametrize(
    "budget",
    [(1, 1, 1), (3, 1, 2), (16, 8, 2), (20, 8, 2), (64, 8, 2)],
)
def test_main_runs_each_file_alone_within_the_worker_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    budget: tuple[int, int, int],
) -> None:
    """Every file is its own shard; the budget splits across shards, at most 8 each."""
    workers, children, parallel = budget
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mutated, "mutable_sources", Mock(return_value=("pkg",)))
    filters = ("pkg.a.x_f__mutmut_*", "pkg.b.x_f__mutmut_*", "pkg.b.x_g__mutmut_*")
    select = Mock(return_value=filters)
    monkeypatch.setattr(mutated, "function_filters", select)
    run = Mock(side_effect=_report_for)
    monkeypatch.setattr(mutated, "run_shard", run)
    monkeypatch.setattr(mutated, "copy_template", Mock())
    pool = Mock(wraps=ThreadPoolExecutor)
    monkeypatch.setattr(mutated, "ThreadPoolExecutor", pool)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mutated",
            "--workers",
            str(workers),
            "--explain",
            "pkg/a.py",
            "pkg/b.py",
        ],
    )

    assert mutated.main() == 0

    assert (
        f"[mutated] 2 file(s), {parallel} at a time x {children} worker(s)\n"
        in capsys.readouterr().err
    )
    assert all(c.args == (tmp_path,) for c in run.call_args_list)
    select.assert_called_once_with(
        tmp_path,
        files=["pkg/a.py", "pkg/b.py"],
        sources=("pkg",),
        workers=workers,
    )
    shards = [cast(Shard, c.kwargs["shard"]) for c in run.call_args_list]
    assert sorted(shards, key=lambda shard: shard.source) == [
        Shard(source="pkg/a.py", filters=("pkg.a.x_f__mutmut_*",)),
        Shard(
            source="pkg/b.py",
            filters=("pkg.b.x_f__mutmut_*", "pkg.b.x_g__mutmut_*"),
        ),
    ]
    assert {call.kwargs["workers"] for call in run.call_args_list} == {children}
    pool.assert_called_once_with(max_workers=parallel)
    snapshots = {cast(Path, c.kwargs["snapshot"]) for c in run.call_args_list}
    assert len(snapshots) == 2


def test_paired_tests_are_the_sidecars_under_the_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved against ``root``, not the cwd, and without a subprocess."""
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)
    (root / "pkg/mod_test.py").touch()
    (root / "pkg/test_mod.py").touch()
    monkeypatch.chdir(tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"ran a subprocess: {args} {kwargs}")

    monkeypatch.setattr(subprocess, "run", refuse)

    assert mutated._paired_tests(root, sources=("pkg/mod.py",)) == (
        "pkg/mod_test.py",
        "pkg/test_mod.py",
    )
    assert Path.cwd() == tmp_path


def test_paired_tests_ask_the_index_for_depth_zero_nodeids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an index any arguments fall back to sidecars, so pin the call."""
    seen: list[tuple[object, ...]] = []

    def harvest(db: Path, *, changed: list[str], depth: int, emit: str) -> list[str]:
        seen.append((db, changed, depth, emit, Path.cwd()))
        return ["t"]

    monkeypatch.setattr(mutated, "harvest", harvest)

    assert mutated._paired_tests(tmp_path, sources=("a.py",)) == ("t",)
    assert seen == [
        (Path(".codegraph/codegraph.db"), ["a.py"], 0, "nodeids", tmp_path),
    ]


def test_filters_skip_tests_and_paths_outside_the_sources(tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "mod.py"
    source.parent.mkdir()
    source.write_text("def f():\n    return 1\n")
    (tmp_path / "pyproject.toml").write_text('[tool.mutmut]\nsource_paths = ["pkg"]\n')
    assert function_filters(
        tmp_path,
        files=["pkg/mod.py", "pkg/mod_test.py", "other.py"],
        sources=("pkg",),
    ) == ("pkg.mod.x_f__mutmut_*",)


@pytest.mark.parametrize("vendored", ["third_party", "typings", ".export"])
def test_filters_skip_vendored_trees(tmp_path: Path, vendored: str) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    (root / "pkg" / vendored).mkdir()
    (root / "pkg" / vendored / "lib.py").write_text(_TWO_FUNCTIONS)

    assert (
        function_filters(
            root,
            files=[f"pkg/{vendored}/lib.py"],
            sources=("pkg",),
        )
        == ()
    )


_SAMPLED = """\
def first(value):
    return value + 1


def stub(value):
    ...


class Holder:
    def method(self, value):
        return value - 1

    def declared(self):
        pass


def second(value):
    return value * 2
"""


@pytest.mark.parametrize("seed", ["", "1", "2", "3", "seven"])
@pytest.mark.parametrize(
    ("rate", "most"),
    [(1.0, 1), (1.0, 2), (0.5, 2), (0.3, 1), (1.0, 9)],
)
def test_a_capped_sample_checks_lazily_and_matches_the_full_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: str,
    rate: float,
    most: int,
) -> None:
    """Parse-ranked selection returns the sample of every mutable function.

    Stubs parse as functions but are never mutable, so the lazy path must skip
    them and keep walking, and never pick fewer than the full path would.
    """
    root = _repo(tmp_path, body=_SAMPLED)
    for name in "abcdef":
        (root / f"pkg/{name}.py").write_text(_SAMPLED.replace("Holder", name.upper()))
    files = ["pkg/mod.py", *(f"pkg/{n}.py" for n in "abcdef")]
    full = function_filters(root, files=files, sources=("pkg",))
    expected = sample(
        full,
        seed=seed,
        size=sample_size(len(full), rate=rate, most=most),
    )
    checked: list[list[str]] = []
    real = mutated.function_filters

    def spy(
        project: Path,
        *,
        files: list[str],
        sources: tuple[str, ...],
        workers: int = 1,
    ) -> tuple[str, ...]:
        checked.append(files)
        return real(project, files=files, sources=sources, workers=workers)

    monkeypatch.setattr(mutated, "function_filters", spy)

    filters, _ = mutated.select_functions(
        root,
        files=files,
        sources=("pkg",),
        seed=seed,
        rate=rate,
        most=most,
        workers=1,
    )

    assert filters == expected
    if most == 1:
        assert sum(map(len, checked)) < len(files)


def test_a_cap_above_the_mutable_count_keeps_every_mutable_function(
    tmp_path: Path,
) -> None:
    """The ranking runs out before the cap: the sample is the rate of what exists."""
    root = _repo(tmp_path, body=_SAMPLED)
    (root / "pkg/broken.py").write_text("def broken(:\n")
    files = ["pkg/mod.py", "pkg/broken.py"]
    full = function_filters(root, files=files, sources=("pkg",))

    assert mutated.select_functions(
        root,
        files=files,
        sources=("pkg",),
        seed="",
        rate=0.5,
        most=9,
    ) == (sample(full, seed="", size=2), 3)


def test_an_uncapped_sample_checks_every_file(tmp_path: Path) -> None:
    root = _repo(tmp_path, body=_SAMPLED)
    full = function_filters(root, files=["pkg/mod.py"], sources=("pkg",))
    assert mutated.select_functions(
        root,
        files=["pkg/mod.py"],
        sources=("pkg",),
        seed="",
        rate=0.5,
        most=None,
    ) == (sample(full, seed="", size=2), 3)


_MANY = "".join(f"def f{n}(value):\n    return value + {n}\n\n\n" for n in range(12))


def _changed_repo(tmp_path: Path) -> Path:
    """Return a git root whose ``pkg/mod.py`` changed ``f7`` since ``HEAD``."""
    root = _repo(tmp_path, body=_MANY)
    shutil.rmtree(root / ".git")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    source = root / "pkg/mod.py"
    source.write_text(_MANY.replace("value + 7\n", "value + 70\n"))
    return root


def test_the_functions_a_change_touched_are_named(tmp_path: Path) -> None:
    root = _changed_repo(tmp_path)
    (root / "pkg/new.py").write_text("def g(value):\n    return value\n")

    assert mutated.changed_functions(
        root,
        base="HEAD",
        files=["pkg/mod.py", "pkg/new.py"],
    ) == frozenset({"pkg.mod.x_f7__mutmut_*", "pkg.new.x_g__mutmut_*"})


@pytest.mark.parametrize("most", [1, None])
@pytest.mark.parametrize("seed", ["", "1", "2", "seven"])
def test_a_changed_function_is_sampled_before_any_other(
    tmp_path: Path,
    most: int | None,
    seed: str,
) -> None:
    """Whatever the seed, the cap goes first to what the change touched."""
    root = _changed_repo(tmp_path)
    preferred = mutated.changed_functions(root, base="HEAD", files=["pkg/mod.py"])

    filters, _ = mutated.select_functions(
        root,
        files=["pkg/mod.py"],
        sources=("pkg",),
        seed=seed,
        rate=1 / 12,
        most=most,
        preferred=preferred,
    )

    assert filters == ("pkg.mod.x_f7__mutmut_*",)


def test_without_a_base_every_function_is_as_likely_as_any() -> None:
    """Run by hand, nothing is preferred: the sample is the seeded one alone."""
    parser = argparse.ArgumentParser()
    flags, _ = mutated._parse_args(parser, ["pkg"])
    assert flags.prefer_changed is None
    flags, _ = mutated._parse_args(
        argparse.ArgumentParser(),
        ["--prefer-changed", "HEAD"],
    )
    assert flags.prefer_changed == "HEAD"


def test_parallel_selection_forks_one_process_per_file_up_to_the_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path, body=_TWO_FUNCTIONS)
    (root / "pkg/other.py").write_text(_TWO_FUNCTIONS)
    pool = Mock(wraps=ProcessPoolExecutor)
    monkeypatch.setattr(mutated, "ProcessPoolExecutor", pool)
    files = ["pkg/mod.py", "pkg/other.py"]

    function_filters(root, files=files, sources=("pkg",))
    pool.assert_not_called()
    function_filters(root, files=files[:1], sources=("pkg",), workers=4)
    pool.assert_not_called()
    function_filters(root, files=files, sources=("pkg",), workers=4)

    (call,) = pool.call_args_list
    assert call.kwargs["max_workers"] == 2
    assert isinstance(call.kwargs["mp_context"], ForkContext)


def test_copy_snapshot_writes_both_configurations(tmp_path: Path) -> None:
    """Snapshot layout copies the project and rewrites mutmut settings."""
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)
    snapshot = tmp_path / "snapshot"
    copy_snapshot(
        root,
        snapshot=snapshot,
        sources=("pkg/mod.py",),
        tests=("pkg/mod_test.py",),
        template=_template(root, tmp_path),
    )
    assert (snapshot / "pkg/mod.py").is_file()
    assert not (snapshot / "mutants/pkg/mod.py").exists()
    assert mutable_sources(snapshot) == ("pkg/mod.py",)


def test_the_snapshot_overrides_the_projects_selection_settings(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)
    (root / "pkg/sub").mkdir()
    (root / "pkg/sub/deep.py").write_text(_TWO_FUNCTIONS)
    (root / "pkg/other.py").write_text(_TWO_FUNCTIONS)
    with (root / "pyproject.toml").open("a") as handle:
        handle.write(
            'also_copy = ["extra"]\n'
            'pytest_add_cli_args_test_selection = ["all"]\n'
            '[tool.coverage.run]\nomit = ["pkg/*"]\n',
        )
    snapshot = tmp_path / "missing" / "snapshot"

    copy_snapshot(
        root,
        snapshot=snapshot,
        sources=("pkg/sub/deep.py", "pkg/mod.py", "pkg/other.py"),
        tests=("pkg/mod_test.py",),
        template=_template(root, tmp_path),
    )

    for directory in (snapshot, snapshot / "mutants"):
        text = (directory / "pyproject.toml").read_text()
        assert mutable_sources(directory) == (
            "pkg/sub/deep.py",
            "pkg/mod.py",
            "pkg/other.py",
        )
        tool = from_plain(
            from_plain(tomllib.loads(text), dict[str, object])["tool"],
            dict[str, object],
        )
        mutmut = from_plain(tool["mutmut"], dict[str, object])
        run = from_plain(
            from_plain(tool["coverage"], dict[str, object])["run"],
            dict[str, object],
        )
        assert from_plain(mutmut["also_copy"], list[str]) == []
        assert from_plain(mutmut["pytest_add_cli_args_test_selection"], list[str]) == [
            "pkg/mod_test.py",
        ]
        assert from_plain(run["omit"], list[str]) == []
    assert (snapshot / "pkg/sub/deep.py").is_file()


def test_the_snapshot_finds_cp_off_the_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)
    template = _template(root, tmp_path)
    run = Mock(side_effect=subprocess.CalledProcessError(1, "cp"))
    monkeypatch.setattr(shutil, "which", Mock(return_value=None))
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        copy_snapshot(
            root,
            snapshot=tmp_path / "snapshot",
            sources=("pkg/mod.py",),
            tests=(),
            template=template,
        )

    assert cast("list[str]", run.call_args.args[0])[0] == "cp"


def test_a_failed_copy_stops_the_snapshot(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)

    with pytest.raises(subprocess.CalledProcessError):
        copy_snapshot(
            root,
            snapshot=tmp_path / "snapshot",
            sources=("pkg/mod.py",),
            tests=(),
            template=tmp_path / "absent",
        )


def test_snapshots_load_the_dynamo_plugin_after_the_projects_own_args(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path / "repo", body=_TWO_FUNCTIONS)
    _runnable(root)
    snapshot = tmp_path / "snapshot"

    copy_snapshot(
        root,
        snapshot=snapshot,
        sources=("pkg/mod.py",),
        tests=("pkg/mod_test.py",),
        template=_template(root, tmp_path),
    )

    for directory in (snapshot, snapshot / "mutants"):
        settings = tomllib.loads((directory / "pyproject.toml").read_text())
        assert settings["tool"]["mutmut"]["pytest_add_cli_args"] == [
            "-o",
            "addopts=",
            "-p",
            "codedoctor.mutation.mutmut_dynamo",
            "-p",
            "codedoctor.mutation.mutmut_param_ids",
            "-p",
            "no:py_pglite",
            "-p",
            "no:fast_array_utils",
            "-p",
            "no:zarr",
            "-p",
            "no:hydra_pytest",
        ]
        assert settings["tool"]["mutmut"]["mutate_only_covered_lines"] is False
        assert settings["tool"]["mutmut"]["process_isolation"] == "fork"


def test_argument_defaults_and_test_file_classification() -> None:
    """Argument defaults and test-file classification are stable."""
    parser = argparse.ArgumentParser()
    mutated._add_arguments(parser)
    assert parser.get_default("seed") == ""
    assert parser.get_default("function_sample_rate") == 1.0
    assert parser.get_default("max_functions_sampled") is None
    assert parser.get_default("workers") is None
    assert parser.get_default("paths") == [Path()]
    assert parser.get_default("emit") == "run"
    assert mutated._is_test("conftest.py")
    assert mutated._is_test("test_mod.py")
    assert mutated._is_test("mod_test.py")
    assert not mutated._is_test("module.py")


def test_the_seed_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI seeds each PR's sample; the variable is what the workflow sets."""
    monkeypatch.setenv("PRECOMMIT_MUTATION_SEED", "178")

    flags, _ = mutated._parse_args(
        argparse.ArgumentParser(),
        argv=["--max-functions-sampled", "2", "--workers", "3"],
    )

    assert (flags.seed, flags.max_functions_sampled, flags.workers) == ("178", 2, 3)


def test_flags_parse_to_their_types_and_choices() -> None:
    flags, _ = mutated._parse_args(
        argparse.ArgumentParser(),
        argv=["--emit", "filters", "--explain", "a.py", "b.py"],
    )

    assert (flags.emit, flags.explain, flags.paths) == (
        "filters",
        True,
        [Path("a.py"), Path("b.py")],
    )
    flags, _ = mutated._parse_args(argparse.ArgumentParser(), argv=["--emit", "run"])
    assert flags.emit == "run"
    with pytest.raises(SystemExit):
        mutated._parse_args(argparse.ArgumentParser(), argv=["--emit", "RUN"])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--function-sample-rate", "0"], "--function-sample-rate must be in (0, 1]"),
        (["--function-sample-rate", "1.5"], "--function-sample-rate must be in (0, 1]"),
        (["--function-sample-rate", "nan"], "--function-sample-rate must be in (0, 1]"),
        (["--max-functions-sampled", "0"], "--max-functions-sampled must be positive"),
    ],
)
def test_out_of_range_sampling_flags_are_rejected(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    with pytest.raises(SystemExit) as raised:
        mutated._parse_args(argparse.ArgumentParser(), argv=argv)

    assert raised.value.code == 2
    assert capsys.readouterr().err.endswith(f"error: {message}\n")


def test_the_boundary_sampling_flags_are_allowed() -> None:
    flags, _ = mutated._parse_args(
        argparse.ArgumentParser(),
        argv=["--function-sample-rate", "1", "--max-functions-sampled", "1"],
    )

    assert (flags.function_sample_rate, flags.max_functions_sampled) == (1.0, 1)


def test_sample_rejects_negative_sizes() -> None:
    with pytest.raises(ValueError, match=r"^sample size must be nonnegative$"):
        sample(("first", "second", "third"), seed="seed", size=-1)


def test_a_zero_size_samples_nothing() -> None:
    assert sample(("first",), seed="seed", size=0) == ()


def test_main_handles_selection_error_and_empty_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Selection errors fail cleanly, while no candidates succeed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        mutated,
        "function_filters",
        Mock(side_effect=ValueError("bad identifier")),
    )
    monkeypatch.setattr(sys, "argv", ["mutated", "pkg/mod.py"])
    assert mutated.main() == 1
    assert "bad identifier" in capsys.readouterr().err

    monkeypatch.setattr(mutated, "function_filters", Mock(return_value=()))
    assert mutated.main() == 0


def test_main_reports_snapshot_preparation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unusable snapshot fails its file, reported with the error's type."""
    _unusable_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["mutated", "--blocking", "pkg/mod.py"])
    assert mutated.main() == 1
    captured = capsys.readouterr()
    assert "mutation: pkg/mod.py: OSError: no snapshot\n" in captured.err
    assert "  pkg/mod.py\n" in captured.out


@pytest.mark.parametrize(
    ("argv", "variable", "status"),
    [
        ([], None, 0),
        ([], "0", 0),
        ([], "yes", 0),
        ([], "1", 1),
        (["--blocking"], None, 1),
    ],
    ids=["unset", "zero", "not-a-count", "set", "flag"],
)
def test_a_finding_fails_the_run_only_when_blocking_is_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    argv: list[str],
    variable: str | None,
    status: int,
) -> None:
    if variable is None:
        monkeypatch.delenv("PRECOMMIT_MUTATION_BLOCKING", raising=False)
    else:
        monkeypatch.setenv("PRECOMMIT_MUTATION_BLOCKING", variable)
    _unusable_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["mutated", *argv, "pkg/mod.py"])

    assert mutated.main() == status
    out = capsys.readouterr().out
    # Either way the finding is printed; advice says so instead of failing.
    assert "  pkg/mod.py\n" in out
    assert ("mutation: advisory, so nothing above fails this run" in out) is (
        status == 0
    )


def _unusable_snapshot(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Select one function of ``pkg/mod.py`` whose snapshot cannot be laid out."""
    monkeypatch.chdir(root)
    monkeypatch.setattr(mutated, "copy_template", Mock())
    monkeypatch.setattr(mutated, "_paired_tests", Mock(return_value=("t.py",)))
    monkeypatch.setattr(
        mutated,
        "function_filters",
        Mock(return_value=("pkg.mod.x_f__mutmut_*",)),
    )
    monkeypatch.setattr(
        mutated,
        "copy_snapshot",
        Mock(side_effect=OSError("no snapshot")),
    )


def test_an_unparsable_paired_test_fails_its_file_in_one_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mutmut never starts: one line names the file, not pytest's traceback."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "t_test.py").write_text("from m import a, b | c\n")
    monkeypatch.setattr(mutated, "copy_template", Mock())
    monkeypatch.setattr(
        mutated,
        "_paired_tests",
        # A deleted test file ahead of the broken one must not end the scan.
        Mock(
            return_value=(
                "gone_test.py::test_a",
                "t_test.py::test_a",
                "t_test.py::test_b",
            ),
        ),
    )
    monkeypatch.setattr(
        mutated,
        "function_filters",
        Mock(return_value=("pkg.mod.x_f__mutmut_*",)),
    )
    snapshot = Mock()
    monkeypatch.setattr(mutated, "copy_snapshot", snapshot)
    monkeypatch.setattr(sys, "argv", ["mutated", "--blocking", "pkg/mod.py"])

    assert mutated.main() == 1

    assert capsys.readouterr().err.endswith(
        "[mutated] 1/1 pkg/mod.py: FAILED\n"
        "mutation: pkg/mod.py: t_test.py:1: invalid syntax\n",
    )
    snapshot.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [OSError("disk full"), subprocess.CalledProcessError(128, "git")],
)
def test_main_fails_when_the_template_cannot_be_copied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    """A disk error, or a root git cannot list, fails the run without a traceback."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        mutated,
        "function_filters",
        Mock(return_value=("pkg.mod.x_f__mutmut_*",)),
    )
    monkeypatch.setattr(mutated, "copy_template", Mock(side_effect=error))
    monkeypatch.setattr(sys, "argv", ["mutated", "pkg/mod.py"])

    assert mutated.main() == 1
    assert capsys.readouterr().err == (
        f"mutation: cannot prepare the snapshot: {error}\n"
    )


def test_filters_skip_a_missing_file(tmp_path: Path) -> None:
    """A named path absent from disk contributes no function."""
    (tmp_path / "pyproject.toml").write_text('[tool.mutmut]\nsource_paths = ["pkg"]\n')
    assert function_filters(tmp_path, files=["pkg/missing.py"], sources=("pkg",)) == ()


def test_each_file_is_a_shard_and_the_biggest_go_first() -> None:
    """A long file started last would finish last, after every core idled."""
    assert plan_shards(
        {
            "one.py": ("one.x_a*",),
            "big.py": ("big.x_c*", "big.x_a*", "big.x_b*"),
            "mid.py": ("mid.x_a*", "mid.x_b*"),
            "tie.py": ("tie.x_a*",),
        },
    ) == (
        Shard(source="big.py", filters=("big.x_a*", "big.x_b*", "big.x_c*")),
        Shard(source="mid.py", filters=("mid.x_a*", "mid.x_b*")),
        Shard(source="one.py", filters=("one.x_a*",)),
        Shard(source="tie.py", filters=("tie.x_a*",)),
    )


def test_a_failing_file_does_not_stop_the_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One file's crash costs that file; every other file still reports."""

    def mutate(root: Path, *, shard: Shard, **_: object) -> ShardReport:
        del root
        if shard.source == "pkg/bad.py":
            raise subprocess.CalledProcessError(2, ["affected.py"])
        return ShardReport(
            source=shard.source,
            outcomes={"survived": ("pkg.ok.x_f__mutmut_1",)},
        )

    monkeypatch.setattr(mutated, "_mutate", mutate)
    monkeypatch.setattr(mutated, "copy_template", Mock())

    reports = mutated._run_shards(
        tmp_path,
        shards=(
            Shard(source="pkg/bad.py", filters=("pkg.bad.x_f__mutmut_*",)),
            Shard(source="pkg/ok.py", filters=("pkg.ok.x_f__mutmut_*",)),
        ),
        children=1,
        parallel=2,
    )

    by_source = {report.source: report for report in reports}
    assert by_source["pkg/ok.py"].outcomes == {"survived": ("pkg.ok.x_f__mutmut_1",)}
    assert by_source["pkg/bad.py"].failure.startswith(
        "mutation: pkg/bad.py: CalledProcessError: ",
    )
    err = capsys.readouterr().err
    assert "pkg/ok.py: 0 checked; 1 survived\n" in err
    assert "pkg/bad.py: FAILED\nmutation: pkg/bad.py: CalledProcessError" in err


def test_each_file_reports_as_it_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        mutated,
        "_mutate",
        Mock(return_value=ShardReport(source="a.py", checked=4)),
    )
    monkeypatch.setattr(mutated, "copy_template", Mock())

    mutated._run_shards(
        tmp_path,
        shards=(Shard(source="a.py", filters=("a.x_f*",)),),
        children=1,
        parallel=1,
    )

    assert capsys.readouterr().err == "[mutated] 1/1 a.py: 4 checked\n"


def test_a_killed_runs_snapshot_is_swept_by_the_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolate_mutation_environment: Path,
) -> None:
    """A run killed mid-shard (a hook timeout) must not leak its checkout copy."""
    orphan = isolate_mutation_environment / "mutated-killed"
    (orphan / "template").mkdir(parents=True)
    orphan.with_suffix(".lock").touch()
    seen: list[str] = []

    def record(root: Path, *, snapshot: Path, **_: object) -> ShardReport:
        del root
        seen.extend(path.name for path in isolate_mutation_environment.iterdir())
        return ShardReport(source=snapshot.name)

    monkeypatch.setattr(mutated, "copy_template", Mock())
    monkeypatch.setattr(mutated, "_mutate", record)

    mutated._run_shards(
        tmp_path,
        shards=(Shard(source="a.py", filters=("a.x_f*",)),),
        children=1,
        parallel=1,
    )

    assert "mutated-killed" not in seen
    assert len(seen) == 2
    assert list(isolate_mutation_environment.iterdir()) == []


def test_a_crash_tail_drops_spinner_frames_and_keeps_the_last_lines() -> None:
    output = "⠋ Generating\r⠙ Generating\r" + "".join(f"line {n}\n" for n in range(50))

    tail = mutated._tail(output, lines=3)

    assert tail == "line 47\nline 48\nline 49"


def test_snapshot_math_libraries_run_single_threaded() -> None:
    environ = {"PATH": "/bin", "OMP_NUM_THREADS": "64"}
    assert single_threaded_env(environ) == {
        "MKL_CBWR": "COMPATIBLE",
        "OPENBLAS_CORETYPE": "Haswell",
        "PATH": "/bin",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert environ == {"PATH": "/bin", "OMP_NUM_THREADS": "64"}


@pytest.mark.parametrize(
    "environ",
    [
        {"PATH": "/bin"},
        {"PATH": "/bin", "HTTPS_PROXY": "http://proxy:3128", "no_proxy": "*"},
    ],
)
def test_a_mac_sends_every_request_off_the_machine_to_a_dead_proxy(
    environ: dict[str, str],
) -> None:
    before = dict(environ)
    dead = "http://127.0.0.1:9"
    local = "localhost,127.0.0.1,::1"

    assert fork_safe_env(environ, platform="darwin") == {
        "PATH": "/bin",
        "http_proxy": dead,
        "https_proxy": dead,
        "all_proxy": dead,
        "no_proxy": local,
        "HTTP_PROXY": dead,
        "HTTPS_PROXY": dead,
        "ALL_PROXY": dead,
        "NO_PROXY": local,
    }
    assert environ == before


def test_another_platform_keeps_the_environment() -> None:
    environ = {"PATH": "/bin", "HTTPS_PROXY": "http://proxy:3128"}
    assert fork_safe_env(environ, platform="linux") == environ


def test_urllib_sends_a_remote_request_to_the_dead_proxy_and_localhost_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # HTTP clients such as httpx read their proxies through urllib.
    for name in [name for name in os.environ if name.lower().endswith("_proxy")]:
        monkeypatch.delenv(name)
    for name, value in fork_safe_env({}, platform="darwin").items():
        monkeypatch.setenv(name, value)

    assert request.getproxies()["https"] == "http://127.0.0.1:9"
    assert not request.proxy_bypass("github.com")
    assert request.proxy_bypass("localhost")


def test_a_crash_tail_keeps_pytest_failure_lines_above_the_tail() -> None:
    output = (
        "E   AssertionError: boom\n"
        "FAILED pkg/mod_test.py::test_x - AssertionError: boom\n"
        "ERROR pkg/mod_test.py::test_y\n"
        "noise\n" + "".join(f"line {n}\n" for n in range(5))
    )

    tail = mutated._tail(output, lines=2)

    assert tail == (
        "E   AssertionError: boom\n"
        "FAILED pkg/mod_test.py::test_x - AssertionError: boom\n"
        "ERROR pkg/mod_test.py::test_y\n"
        "line 3\nline 4"
    )


def test_filters_are_grouped_under_the_file_they_mutate() -> None:
    assert mutated._filters_by_source(
        ("pkg.a.x_f__mutmut_*", "pkg.b.x_f__mutmut_*", "pkg.a.x_g__mutmut_*"),
        files=["pkg/a.py", "pkg/b.py", "pkg/notes.txt"],
    ) == {
        "pkg/a.py": ("pkg.a.x_f__mutmut_*", "pkg.a.x_g__mutmut_*"),
        "pkg/b.py": ("pkg.b.x_f__mutmut_*",),
    }


def test_every_finding_of_every_shard_is_printed_with_its_diff(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No cap: a sweep needs the whole list, and the snapshot is gone after."""
    survivors = tuple(f"pkg.a.x_f__mutmut_{index}" for index in range(20))
    reports = [
        ShardReport(
            source="pkg/a.py",
            outcomes={"survived": survivors, "no tests": ("pkg.a.x_g__mutmut_1",)},
            diffs={name: f"diff of {name}" for name in survivors},
        ),
        ShardReport(
            source="pkg/b.py",
            outcomes={"unchecked": ("pkg.b.x_h__mutmut_*",)},
        ),
        ShardReport(source="pkg/c.py", failure="broken"),
    ]

    assert mutated._print_reports(reports) == 1

    out = capsys.readouterr().out
    assert out.startswith(
        "mutation: these files recorded no results; each reason is above:\n"
        "  pkg/c.py\n"
        "mutation: no test exercised these mutants, so nothing was checked:\n"
        "  pkg.a.x_g__mutmut_1\n"
        "mutation: these variants survived the selected tests:\n"
        + "".join(f"  {name}\n" for name in sorted(survivors))
        + "mutation: no mutant was checked for these selected functions:\n"
        "  pkg.b.x_h__mutmut_*\n"
        "\nCheck test coverage of these functions first.",
    )
    assert out.endswith(
        "".join(
            f"# {name}: survived\ndiff of {name}\n\n" for name in sorted(survivors)
        ),
    )


@pytest.mark.parametrize(
    ("report", "status"),
    [
        (ShardReport(source="a.py", outcomes={"survived": ()}), 0),
        (ShardReport(source="a.py", failure="x"), 1),
    ],
)
def test_a_clean_report_passes_and_a_failed_file_fails(
    report: ShardReport,
    status: int,
) -> None:
    assert mutated._print_reports([report]) == status


def test_a_mutmut_run_past_its_limit_is_stopped_with_what_it_started(
    tmp_path: Path,
) -> None:
    """A hung run would otherwise hold the hook, and its workers, forever."""
    job = tmp_path / "job.pid"
    script = f"sleep 60 & echo $! > {job}; wait"

    with pytest.raises(subprocess.TimeoutExpired):
        mutated._run_mutmut(
            ["/bin/sh", "-c", script],
            cwd=tmp_path,
            env=dict(os.environ),
            timeout_sec=0.2,
        )

    # Not this process's child, so once killed it is gone when its parent reaps it.
    pid = int(job.read_text())
    for _ in range(300):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"the job {pid} outlived its run's limit")


def test_a_mutmut_run_within_its_limit_returns_what_it_printed(tmp_path: Path) -> None:
    done = mutated._run_mutmut(
        ["/bin/sh", "-c", "echo out; echo err >&2; exit 3"],
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_sec=5.0,
    )
    assert (done.returncode, done.stdout, done.stderr) == (3, "out\n", "err\n")


def test_a_shard_past_its_limit_is_reported_and_costs_no_other(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hangs(argv: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        del options
        raise subprocess.TimeoutExpired(argv, 900.0)

    monkeypatch.setattr(mutated, "_run_mutmut", hangs)
    monkeypatch.setattr(mutated, "copy_snapshot", Mock())
    monkeypatch.setattr(mutated, "_paired_tests", Mock(return_value=("t.py",)))

    report = run_shard(
        tmp_path,
        shard=Shard(source="pkg/mod.py", filters=("pkg.mod.x_f__mutmut_*",)),
        snapshot=tmp_path / "snapshot",
        template=tmp_path,
        workers=1,
    )

    assert report.failure == "mutation: pkg/mod.py: mutmut ran past its 900 s limit"


def test_mutmut_retries_segfaulted_and_unrun_mutants_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    filters = ("pkg.mod.x_f__mutmut_*",)
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    load = Mock()
    data = Mock(
        load=load,
        exit_code_by_key={
            "pkg.mod.x_f__mutmut_1": -11,
            "pkg.mod.x_f__mutmut_2": None,
            "pkg.mod.x_f__mutmut_3": 1,
            "pkg.mod.x_other__mutmut_1": -11,
        },
    )
    loaded = Mock(return_value=data)
    monkeypatch.setattr(mutated, "_run_mutmut", run)
    monkeypatch.setattr(mutated, "copy_snapshot", Mock())
    monkeypatch.setattr(mutated, "_paired_tests", Mock(return_value=("t.py",)))
    monkeypatch.setitem(mutated.__dict__, "SourceFileMutationData", loaded)
    monkeypatch.setattr(
        mutated,
        "_collect",
        Mock(return_value=ShardReport(source="pkg/mod.py")),
    )

    mutated._mutate(
        tmp_path,
        shard=Shard(source="pkg/mod.py", filters=filters),
        snapshot=snapshot,
        template=tmp_path,
        workers=4,
    )

    assert run.call_count == 2
    single = fork_safe_env(single_threaded_env(os.environ))
    assert [call.kwargs["env"] for call in run.call_args_list] == [single, single]
    assert run.call_args_list[1].args[0] == [
        sys.executable,
        "-m",
        "mutmut",
        "run",
        "--max-children",
        "1",
        "pkg.mod.x_f__mutmut_1",
        "pkg.mod.x_f__mutmut_2",
    ]
    assert run.call_args_list[1].kwargs == {"cwd": snapshot, "env": single}
    loaded.assert_called_once_with(path=snapshot / "mutants" / "pkg/mod.py")
    load.assert_called_once_with()


def test_a_mutmut_crash_reports_the_tail_of_its_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "".join(f"out {n}\n" for n in range(60))
    run = Mock(return_value=subprocess.CompletedProcess([], 3, stdout, "err\n"))
    monkeypatch.setattr(mutated, "_run_mutmut", run)
    monkeypatch.setattr(mutated, "copy_snapshot", Mock())
    monkeypatch.setattr(mutated, "_paired_tests", Mock(return_value=("t.py",)))

    report = run_shard(
        tmp_path,
        shard=Shard(source="pkg/mod.py", filters=("pkg.mod.x_f__mutmut_*",)),
        snapshot=tmp_path / "snapshot",
        template=tmp_path,
        workers=5,
    )

    assert report.failure == (
        "".join(f"out {n}\n" for n in range(21, 60))
        + "err\nmutation: pkg/mod.py: mutmut exited 3 before recording results; "
        "its output is above."
    )
    argv = cast("list[str]", run.call_args.args[0])
    assert argv[:6] == [sys.executable, "-m", "mutmut", "run", "--max-children", "5"]
    assert run.call_args.kwargs == {
        "cwd": tmp_path / "snapshot",
        "env": fork_safe_env(single_threaded_env(os.environ)),
    }


def test_the_snapshot_mutates_only_the_selected_functions(tmp_path: Path) -> None:
    """Every unselected function is pragma-skipped; line numbers are unchanged."""
    body = (
        "def first(value):  # noqa: ANN001 -- kept.\n"
        "    return value + 1\n"
        "\n\n"
        "class Holder:\n"
        "    def other(self):\n"
        "        return 3\n"
        "\n"
        "    def method(self):  \n"
        "        return 2\n"
        "\n\n"
        "def second(value):\n"
        "    return value * 2\n"
        "\n\n"
        "def third(\n"
        "    value,\n"
        "):\n"
        "    # A comment between header and body.\n"
        "\n"
        "    return value - 1\n"
    )
    root = _repo(tmp_path / "repo", body=body)
    snapshot = tmp_path / "snapshot"

    copy_snapshot(
        root,
        template=_template(root, tmp_path),
        snapshot=snapshot,
        sources=("pkg/mod.py",),
        tests=("pkg/mod_test.py",),
        filters=(
            "pkg.mod.x_second__mutmut_*",
            f"pkg.mod.x{CLASS_NAME_SEPARATOR}Holder{CLASS_NAME_SEPARATOR}other__mutmut_*",
        ),
    )

    lines = (snapshot / "pkg/mod.py").read_text().splitlines()
    assert len(lines) == len(body.splitlines())
    skip = "# pragma: no mutate block"
    assert lines[0] == f"def first(value):  # noqa: ANN001 -- kept. {skip}"
    assert lines[5] == "    def other(self):"
    assert lines[8] == f"    def method(self):    {skip}"
    assert lines[12] == "def second(value):"
    assert lines[16:19] == ["def third(", "    value,", f"):  {skip}"]
    module, mutations, _, ignored = create_mutations(
        filename="pkg/mod.py",
        code="\n".join(lines) + "\n",
    )
    del module
    assert ignored == {"first", "method", "third"}
    owners = {m.contained_by_top_level_function for m in mutations}
    assert {owner.name.value for owner in owners if owner is not None} == {
        "other",
        "second",
    }
    assert None not in owners


def test_the_skip_pragma_keeps_each_line_ending(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/mod.py").write_bytes(
        b"def first(X):\r\n    return 1\r\ndef second():\n    return 2\n",
    )
    (root / "pyproject.toml").write_text('[tool.mutmut]\nsource_paths = ["pkg"]\n')
    snapshot = tmp_path / "snapshot"

    copy_snapshot(
        root,
        template=_template(root, tmp_path),
        snapshot=snapshot,
        sources=("pkg/mod.py",),
        tests=(),
        filters=("pkg.mod.x_second__mutmut_*",),
    )

    assert (snapshot / "pkg/mod.py").read_bytes() == (
        b"def first(X):  # pragma: no mutate block\r\n    return 1\r\n"
        b"def second():\n    return 2\n"
    )


def test_the_skip_pragma_counts_only_python_line_breaks(tmp_path: Path) -> None:
    """``str.splitlines`` also breaks at U+2028; Python's tokenizer does not."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/mod.py").write_bytes(
        'S = "x\u2028y\x0cz"\ndef a():\n    return S\ndef b():\n    return 2\n'.encode(),
    )
    (root / "pyproject.toml").write_text('[tool.mutmut]\nsource_paths = ["pkg"]\n')
    snapshot = tmp_path / "snapshot"

    copy_snapshot(
        root,
        template=_template(root, tmp_path),
        snapshot=snapshot,
        sources=("pkg/mod.py",),
        tests=(),
        filters=("pkg.mod.x_b__mutmut_*",),
    )

    assert (snapshot / "pkg/mod.py").read_bytes().decode() == (
        'S = "x\u2028y\x0cz"\ndef a():  # pragma: no mutate block\n'
        "    return S\ndef b():\n    return 2\n"
    )


def test_parsing_never_sizes_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every gate passes ``--workers``; probing the machine for a default is waste."""
    sized: list[int] = []
    monkeypatch.setattr(mutated, "default_workers", lambda: sized.append(1) or 1)
    parser = argparse.ArgumentParser()
    mutated._add_arguments(parser)

    parser.parse_args(["--workers", "2"])

    assert sized == []


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
