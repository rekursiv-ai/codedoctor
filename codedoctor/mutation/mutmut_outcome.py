"""Pytest plugin failing a mutation run in which no selected test ran.

Pytest exits 0 when it skips every test it collected. Under mutmut that reads
as "survived", yet the mutant was never tested: a mutant of a function the
repository's own collection hook calls can skip the very tests that would
catch it. A session that collected tests and ran none of them fails here, so
mutmut records the mutant killed.

``mutated.py`` loads this with ``-p`` in each snapshot; nothing else should.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import Generator


_RAN = pytest.StashKey[bool]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Note a test whose body ran, passing or failing.

    Args:
      item: The test.
      call: The phase that ran.

    Yields:
      nothing: Pytest makes the report meanwhile.

    Returns:
      report: Pytest's report of that phase, unchanged.

    """
    report = yield
    if call.when == "call" and not report.skipped:
        item.config.stash[_RAN] = True
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail a session that would pass having run no test.

    Args:
      session: The session ending.
      exitstatus: What pytest would exit with.

    """
    if exitstatus == pytest.ExitCode.OK and not session.config.stash.get(_RAN, False):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
