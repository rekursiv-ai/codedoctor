"""Tests for the plugin keeping parametrize ids stable across sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import importlib
import sys

import pytest

from codedoctor.mutation import mutmut_param_ids


if TYPE_CHECKING:
    from collections.abc import Iterator


pytest_plugins = ("pytester",)

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


@pytest.mark.parametrize(
    ("plugins", "exits"),
    [
        ((), [pytest.ExitCode.USAGE_ERROR, pytest.ExitCode.USAGE_ERROR]),
        (
            ("-p", "codedoctor.mutation.mutmut_param_ids"),
            [pytest.ExitCode.OK, pytest.ExitCode.OK],
        ),
    ],
    ids=["bare", "plugin"],
)
def test_ids_from_one_session_select_tests_in_the_next(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    plugins: tuple[str, ...],
    exits: list[pytest.ExitCode],
) -> None:
    """Without the plugin pytest re-escapes cached ids and selects nothing.

    Mutmut's sequence in one process: collect, then run the ids it recorded. The
    test module is imported once, before the sessions, so all three meet the same
    marks -- and the ids those marks cached.
    """
    # The installed plugins are no part of the sequence.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makepyfile(probe_test=_MODULE)
    pytester.syspathinsert()
    monkeypatch.delitem(sys.modules, "probe_test", raising=False)
    _ = importlib.import_module("probe_test")

    collected = pytester.runpytest_inprocess(
        "--co",
        "-q",
        "-p",
        "no:cacheprovider",
        *plugins,
    )
    ids = [line for line in collected.outlines if "::" in line]
    ran = [
        pytester.runpytest_inprocess("-p", "no:cacheprovider", *plugins, *ids).ret
        for _ in range(2)
    ]

    assert len(ids) == 4
    assert ran == exits


def test_each_session_forgets_cached_ids_once_per_mark() -> None:
    """A session forgets the ids another cached, and keeps its own."""
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
