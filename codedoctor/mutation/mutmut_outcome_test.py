"""Tests for the plugin making a mutant run that ran no test a kill."""

from __future__ import annotations

from typing import Literal

from mutmut.mutation import trampoline

import pytest

from codedoctor.mutation import mutmut_outcome


pytest_plugins = ("pytester",)

_MODULE = """\
import pytest


@pytest.mark.skip(reason="absent")
def test_skipped():
    pass


def test_passes():
    pass


@pytest.mark.xfail(strict=True)
def test_expected_to_fail():
    raise AssertionError


def test_fails():
    raise AssertionError
"""


@pytest.mark.parametrize(
    ("mutant", "selected", "status"),
    [
        ("pkg.mod.x_f__mutmut_1", "test_skipped", 1),
        ("pkg.mod.x_f__mutmut_1", "nothing", 1),
        ("pkg.mod.x_f__mutmut_1", "test_passes", 0),
        ("pkg.mod.x_f__mutmut_1", "test_expected_to_fail", 0),
        ("pkg.mod.x_f__mutmut_1", "test_fails", 1),
        ("", "test_skipped", 0),
        ("", "nothing", 5),
        ("stats", "test_skipped", 0),
        ("fail", "test_skipped", 0),
    ],
    ids=[
        "mutant-all-skipped",
        "mutant-none-selected",
        "mutant-passes",
        "mutant-xfails",
        "mutant-fails",
        "clean-all-skipped",
        "clean-none-selected",
        "stats-all-skipped",
        "forced-fail-all-skipped",
    ],
)
def test_only_a_mutant_run_that_ran_no_test_fails(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    mutant: str,
    selected: str,
    status: int,
) -> None:
    pytester.makepyfile(probe_test=_MODULE)
    monkeypatch.setenv("MUTANT_UNDER_TEST", mutant)
    # The installed plugins (asyncio, randomly, xdist) are no part of a mutant run's
    # verdict, and their own warnings would land in this session.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    result = pytester.runpytest_inprocess(
        "-p",
        "codedoctor.mutation.mutmut_outcome",
        "-k",
        selected,
        "probe_test.py",
    )

    assert result.ret == status, result.outlines


class _Session:
    def __init__(self, exitstatus: int | pytest.ExitCode) -> None:
        self.exitstatus = exitstatus


def _report(when: Literal["setup", "call", "teardown"]) -> pytest.TestReport:
    return pytest.TestReport("t", ("t.py", 0, "t"), {}, "passed", None, when)


@pytest.mark.parametrize(
    ("mutant", "phases", "exitstatus", "finished"),
    [
        # A mutant run whose every test skipped, or that kept none, is a kill.
        (True, ["setup", "teardown"], pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED),
        (True, [], pytest.ExitCode.NO_TESTS_COLLECTED, pytest.ExitCode.TESTS_FAILED),
        # A test body ran, so pytest's verdict stands.
        (True, ["setup", "call", "teardown"], pytest.ExitCode.OK, pytest.ExitCode.OK),
        # A failing or interrupted run keeps its own status.
        (True, [], pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.TESTS_FAILED),
        (True, [], pytest.ExitCode.INTERRUPTED, pytest.ExitCode.INTERRUPTED),
        # Not a mutant run: left as pytest reports it.
        (False, [], pytest.ExitCode.OK, pytest.ExitCode.OK),
        (
            False,
            [],
            pytest.ExitCode.NO_TESTS_COLLECTED,
            pytest.ExitCode.NO_TESTS_COLLECTED,
        ),
    ],
)
def test_a_mutant_run_that_ran_no_test_body_fails(
    *,
    mutant: bool,
    phases: list[Literal["setup", "call", "teardown"]],
    exitstatus: pytest.ExitCode,
    finished: pytest.ExitCode,
) -> None:
    ran = mutmut_outcome._Ran(mutant=mutant)
    session = _Session(exitstatus)

    for when in phases:
        ran.pytest_runtest_logreport(_report(when))
    ran.pytest_sessionfinish(session, exitstatus)

    assert session.exitstatus == finished


class _Plugins:
    def __init__(self) -> None:
        self.registered: dict[str, object] = {}

    def register(self, plugin: object, name: str) -> None:
        self.registered[name] = plugin


class _Config:
    def __init__(self) -> None:
        self.pluginmanager = _Plugins()


def test_configure_registers_a_watcher_for_the_run_under_way() -> None:
    config = _Config()

    mutmut_outcome.pytest_configure(config)

    watcher = config.pluginmanager.registered["mutmut-outcome"]
    assert isinstance(watcher, mutmut_outcome._Ran)
    # Under mutmut this is a mutant's name; under plain pytest it is empty.
    assert watcher.mutant is ("__mutmut_" in trampoline.get_mutant_under_test())
    assert watcher.any is False


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
