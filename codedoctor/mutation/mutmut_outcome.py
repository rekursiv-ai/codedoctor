"""Pytest plugin making a mutant run that ran no test a kill, not a survivor.

Mutmut reads a mutant's verdict from pytest's exit status, and pytest exits 0
when it skips every selected test and 5 when it collects or keeps none. Either
way no test body ran, yet 0 reads as "survived". A mutant of code the
repository's own collection hook calls -- the hook that skips a test whose
resource is absent -- can skip or deselect the very tests that would catch it.
Its selected tests all ran and passed without it, so a run that ran none of
them is the mutant's doing, and fails here.

The clean, forced-fail, and stats runs are left as pytest reports them.

``mutated.py`` loads this with ``-p`` in each snapshot; nothing else should.
"""

from __future__ import annotations

from typing import Protocol

from mutmut.mutation import trampoline

import pytest


def pytest_configure(config: _Configured) -> None:
    """Watch this session's tests.

    Args:
      config: The session's config.

    """
    mutant = "__mutmut_" in trampoline.get_mutant_under_test()
    config.pluginmanager.register(_Ran(mutant=mutant), "mutmut-outcome")


class _Registers(Protocol):
    def register(self, plugin: object, name: str) -> object: ...


class _Configured(Protocol):
    """The part of a pytest config this plugin uses."""

    @property
    def pluginmanager(self) -> _Registers: ...


class _Finished(Protocol):
    """The part of a pytest session its finish hook may rewrite."""

    exitstatus: int | pytest.ExitCode


class _Ran:
    """Whether any test body ran this session; fails a mutant run if none did."""

    def __init__(self, *, mutant: bool) -> None:
        self.mutant = mutant
        self.any = False

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # A skipped test reports setup and teardown only; an xfail one runs.
        if report.when == "call":
            self.any = True

    def pytest_sessionfinish(self, session: _Finished, exitstatus: int) -> None:
        idle = exitstatus in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
        if self.mutant and idle and not self.any:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
