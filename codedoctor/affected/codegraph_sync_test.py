"""Tests for the CodeGraph index sync."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil
import subprocess
import sys

import pytest

from codedoctor.affected import codegraph_sync


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _sync(tmp_path: Path, *, installed: bool = True) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv: Sequence[str]) -> int:
        calls.append(list(argv))
        return 0

    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    codegraph_sync.sync(
        root,
        caches=tmp_path / "caches",
        run=run,
        which=lambda name: f"/bin/{name}" if installed else None,
    )
    return calls


def test_a_missing_index_is_linked_into_the_cache_and_initialized(
    tmp_path: Path,
) -> None:
    """The ~387 MB index stays out of the checkout."""
    calls = _sync(tmp_path)

    assert (tmp_path / "repo" / ".codegraph").readlink() == tmp_path / "caches" / "repo"
    assert calls == [["codegraph", "init", "--yes"], ["codegraph", "sync", "--quiet"]]


def test_an_existing_index_is_only_synced(tmp_path: Path) -> None:
    (tmp_path / "repo" / ".codegraph").mkdir(parents=True)
    (tmp_path / "repo" / ".codegraph" / "codegraph.db").write_text("")

    assert _sync(tmp_path) == [["codegraph", "sync", "--quiet"]]


def test_without_the_cli_after_install_nothing_else_runs(tmp_path: Path) -> None:
    calls = _sync(tmp_path, installed=False)

    assert calls == [["npm", "install", "-g", "@colbymchenry/codegraph"]]


def test_an_unusable_cache_leaves_the_gates_on_their_floor(tmp_path: Path) -> None:
    """No index is a narrower check, never a failed commit."""
    (tmp_path / "caches").write_text("")

    assert _sync(tmp_path) == []
    assert not (tmp_path / "repo" / ".codegraph").is_symlink()


@pytest.mark.parametrize("value", ["0", "00", "", "false", "01", "-1", " 1 "])
def test_a_disabled_opt_in_is_off(value: str) -> None:
    """``PRECOMMIT_CODEGRAPH=0`` once re-synced a 417 MB index every commit."""
    assert not codegraph_sync.opted_in(
        {"PRECOMMIT_CODEGRAPH_PYTEST": value, "PRECOMMIT_CODEGRAPH_TYPECHECK": value},
    )


@pytest.mark.parametrize(
    "variable",
    ["PRECOMMIT_CODEGRAPH_PYTEST", "PRECOMMIT_CODEGRAPH_TYPECHECK"],
)
def test_either_gate_opting_in_is_on(variable: str) -> None:
    """A gate reading a never-synced index silently falls back to its floor."""
    assert codegraph_sync.opted_in({variable: "1"})


@pytest.mark.parametrize(("opt_in", "synced"), [("1", True), ("0", False)])
def test_the_cli_syncs_into_the_named_caches_only_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opt_in: str,
    synced: bool,
) -> None:
    calls: list[list[str]] = []

    def run(argv: Sequence[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PRECOMMIT_CODEGRAPH_PYTEST", opt_in)
    monkeypatch.delenv("PRECOMMIT_CODEGRAPH_TYPECHECK", raising=False)
    monkeypatch.setattr(shutil, "which", _on_path)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["codegraph_sync", "--caches", str(tmp_path / "caches")],
    )

    assert codegraph_sync.main() == 0
    assert (tmp_path / ".codegraph").is_symlink() is synced
    assert (calls[-1:] == [["codegraph", "sync", "--quiet"]]) is synced


def test_the_cli_reports_a_missing_npm_and_still_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """No npm leaves the gates on their floor; the commit goes through."""

    def missing(argv: Sequence[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(argv[0])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PRECOMMIT_CODEGRAPH_PYTEST", "1")
    monkeypatch.setattr(shutil, "which", _off_path)
    monkeypatch.setattr(subprocess, "run", missing)
    monkeypatch.setattr(
        sys,
        "argv",
        ["codegraph_sync", "--caches", str(tmp_path / "caches")],
    )

    assert codegraph_sync.main() == 0
    assert "npm: command not found" in capsys.readouterr().err


def _on_path(name: str) -> str:
    return f"/bin/{name}"


def _off_path(name: str) -> None:
    del name


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
