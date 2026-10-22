"""Tests for ``nbstripout_hook``: which binary runs, and what the hook returns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil
import subprocess
import sys

from codedoctor.nbstripout import nbstripout_hook


if TYPE_CHECKING:
    import pytest


class _FakeCompleted:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *installed: str,
    returncode: int = 0,
    argv: tuple[str, ...] = (),
) -> list[list[str]]:
    """Pretend only ``installed`` binaries exist; capture the argv ``main`` runs."""
    calls: list[list[str]] = []

    def fake_which(cmd: str) -> str | None:
        return f"/usr/bin/{cmd}" if cmd in installed else None

    def fake_run(args: list[str], *, check: bool) -> _FakeCompleted:
        del check
        calls.append(args)
        return _FakeCompleted(returncode)

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["nbstripout_hook.py", *argv])
    return calls


def test_prefers_the_fast_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """`nbstripout-fast` is a drop-in rewrite; prefer it whenever both exist.

    Choosing whichever `which` happens to resolve first would make hook latency
    depend on PATH order rather than on what is installed.
    """
    calls = _install(monkeypatch, "nbstripout-fast", "nbstripout")

    assert nbstripout_hook.main() == 0
    assert calls == [["/usr/bin/nbstripout-fast"]]


def test_falls_back_to_the_reference_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(monkeypatch, "nbstripout")

    assert nbstripout_hook.main() == 0
    assert calls == [["/usr/bin/nbstripout"]]


def test_missing_binary_reports_127(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """127 is the shell's "command not found", so the hook reads as unrunnable.

    Returning 1 would look like a notebook that failed to strip, sending a
    developer after the notebook instead of the missing dependency.
    """
    _install(monkeypatch)

    assert nbstripout_hook.main() == 127
    assert "nbstripout" in capsys.readouterr().err


def test_filenames_reach_the_underlying_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pre-commit appends the changed files; dropping them strips nothing."""
    calls = _install(monkeypatch, "nbstripout-fast", argv=("a.ipynb", "b.ipynb"))

    assert nbstripout_hook.main() == 0
    assert calls == [["/usr/bin/nbstripout-fast", "a.ipynb", "b.ipynb"]]


def test_the_tools_exit_status_is_the_hooks_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swallowing a nonzero status would let a dirty notebook through."""
    _install(monkeypatch, "nbstripout-fast", returncode=3)

    assert nbstripout_hook.main() == 3


def test_unknown_flags_pass_through_rather_than_erroring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every argument belongs to the wrapped tool, since this layer owns none.

    `parse_known_args` is what makes that true; `parse_args` would abort on the
    first nbstripout option a developer passes.
    """
    calls = _install(monkeypatch, "nbstripout-fast", argv=("--keep-count", "x.ipynb"))

    assert nbstripout_hook.main() == 0
    assert calls == [["/usr/bin/nbstripout-fast", "--keep-count", "x.ipynb"]]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
