"""Pytest plugin refusing any signal a mutant could aim at every process.

``kill(-1)`` signals every process the user owns, ``kill(0)`` the caller's own
group, and ``killpg(1)`` is ``kill(-1)``. Code under test signals children it
started, so a mutant that turns a pid into one of those -- a ``-1`` sentinel
flipped to ``+1`` once killed a whole login -- would reach far beyond the test.
Here such a call raises instead, which fails the test and kills the mutant.

``mutated.py`` loads this with ``-p`` in each snapshot; nothing else should.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Swap in the guarded ``os.kill`` and ``os.killpg`` for the whole session.

    Args:
      config: The session's config.

    """
    del config
    kill, killpg = os.kill, os.killpg

    def guarded_kill(pid: int, sig: int, /) -> None:
        if pid <= 0:
            raise PermissionError(f"refused kill({pid}, {sig}) in a mutant run")
        kill(pid, sig)

    def guarded_killpg(pgid: int, sig: int, /) -> None:
        if pgid <= 1:
            raise PermissionError(f"refused killpg({pgid}, {sig}) in a mutant run")
        killpg(pgid, sig)

    # Never undone: the guard lasts as long as the mutant's test process.
    patch = pytest.MonkeyPatch()
    patch.setattr(os, "kill", guarded_kill)
    patch.setattr(os, "killpg", guarded_killpg)
