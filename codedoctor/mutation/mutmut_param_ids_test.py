"""Tests for the plugin keeping parametrize ids stable across sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import os
import subprocess
import sys

import pytest

from codedoctor.mutation import mutmut_param_ids


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


_MODULE = """\
import pytest


@pytest.mark.parametrize("v", [b"\\x00\\xff", "plain"], ids=repr)
def test_v(v):
    pass


@pytest.mark.parametrize("w", [b"\\x01"], ids=repr)
class TestShared:
    def test_a(self, w):
        pass

    def test_b(self, w):
        pass
"""

# Mutmut's sequence in one process: collect, then run the ids it recorded.
_PROBE = """\
import sys, pytest

class Ids:
    def __init__(self):
        self.ids = []

    def pytest_collection_modifyitems(self, items):
        self.ids = [item.nodeid for item in items]

plugins = sys.argv[1:]
first = Ids()
pytest.main(["-q", "--co", "-p", "no:cacheprovider", *plugins, "probe_test.py"], plugins=[first])
for _ in range(2):
    code = int(pytest.main(["-q", "-p", "no:cacheprovider", *plugins, *first.ids]))
    print("exit", code)
"""


@pytest.mark.cli_python_subprocess
@pytest.mark.parametrize(
    ("plugins", "exits"),
    [
        ([], ["exit 4", "exit 4"]),
        (["-p", "codedoctor.mutation.mutmut_param_ids"], ["exit 0", "exit 0"]),
    ],
    ids=["bare", "plugin"],
)
def test_ids_from_one_session_select_tests_in_the_next(
    tmp_path: Path,
    plugins: list[str],
    exits: list[str],
) -> None:
    """Without the plugin pytest re-escapes cached ids and selects nothing."""
    (tmp_path / "probe_test.py").write_text(_MODULE)
    (tmp_path / "probe.py").write_text(_PROBE)
    completed = subprocess.run(  # noqa: S603 -- Fixed interpreter and probe.
        [sys.executable, "probe.py", *plugins],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env={"PYTHONPATH": os.pathsep.join(sys.path), "PATH": ""},
        text=True,
    )

    assert [
        line for line in completed.stdout.splitlines() if line.startswith("exit")
    ] == exits, completed.stdout + completed.stderr


def test_each_session_forgets_cached_ids_once_per_mark() -> None:
    """In-process twin of the subprocess test, which the unit tier never runs."""
    source = pytest.mark.parametrize("v", [b"\x00"], ids=repr).mark
    # How pytest stores a mark whose ids came from another: a class's mark
    # reaches each method combined with the method's own.
    combined = pytest.mark.parametrize.mark.combined_with(source)
    bare = pytest.mark.parametrize("w", [1]).mark
    marks = [combined, combined, bare]

    for session in ("first", "second"):
        config = _Config()
        object.__setattr__(source, "_param_ids_generated", [f"{session}-cached"])
        _generate(config, marks=marks)
        assert source._param_ids_generated is None, session
        # Pytest refills the cache; the same session must keep what it generated.
        object.__setattr__(source, "_param_ids_generated", [f"{session}-fresh"])
        _generate(config, marks=marks)
        assert source._param_ids_generated == [f"{session}-fresh"]


@dataclass(slots=True, kw_only=True)
class _Config:
    """The part of ``pytest.Config`` the plugin reads: one stash per session."""

    stash: pytest.Stash = field(default_factory=pytest.Stash)


@dataclass(slots=True, kw_only=True)
class _Definition:
    marks: list[pytest.Mark]

    def iter_markers(self, name: str) -> Iterator[pytest.Mark]:
        return (mark for mark in self.marks if mark.name == name)


@dataclass(slots=True, kw_only=True)
class _Metafunc:
    """The part of ``pytest.Metafunc`` the plugin reads."""

    config: _Config
    definition: _Definition


def _generate(config: _Config, *, marks: list[pytest.Mark]) -> None:
    mutmut_param_ids.pytest_generate_tests(
        _Metafunc(config=config, definition=_Definition(marks=marks)),
    )


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
