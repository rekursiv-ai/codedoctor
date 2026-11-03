"""Tests for the plugin refusing a signal aimed at every process.

The guard wraps recording fakes here, never the real calls: no test sends a
signal, and a mutant of the guard cannot be masked by the real guard that the
mutation run itself loads.
"""

from __future__ import annotations

import os
import signal

import pytest

from codedoctor.mutation import mutmut_kill_guard


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    """Install the guard over recorders; return what reached them."""
    reached: list[tuple[str, int, int]] = []

    def kill(pid: int, sig: int, /) -> None:
        reached.append(("kill", pid, sig))

    def killpg(pgid: int, sig: int, /) -> None:
        reached.append(("killpg", pgid, sig))

    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(os, "killpg", killpg)
    mutmut_kill_guard.pytest_configure(pytest.Config.__new__(pytest.Config))
    return reached


@pytest.mark.parametrize("pid", [-1, 0, -2])
def test_kill_of_everyone_or_a_group_is_refused(
    sent: list[tuple[str, int, int]],
    pid: int,
) -> None:
    with pytest.raises(PermissionError, match=rf"^refused kill\({pid}, "):
        os.kill(pid, signal.SIGCONT)
    assert sent == []


@pytest.mark.parametrize("pgid", [1, 0, -1])
def test_killpg_of_everyone_or_its_own_group_is_refused(
    sent: list[tuple[str, int, int]],
    pgid: int,
) -> None:
    with pytest.raises(PermissionError, match=rf"^refused killpg\({pgid}, "):
        os.killpg(pgid, signal.SIGCONT)
    assert sent == []


def test_a_signal_to_one_process_or_group_passes_through(
    sent: list[tuple[str, int, int]],
) -> None:
    os.kill(1, signal.SIGCONT)
    os.killpg(2, signal.SIGCONT)

    assert sent == [("kill", 1, signal.SIGCONT), ("killpg", 2, signal.SIGCONT)]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
