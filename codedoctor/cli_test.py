"""The product entry point finds the repository before installing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil
import subprocess
import sys

import pytest

from codedoctor import cli, install
from codedoctor.lib.testing.checks import assert_help_without_docstring


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import argparse


def test_help_without_docstring(monkeypatch: pytest.MonkeyPatch) -> None:
    assert_help_without_docstring(monkeypatch, cli, cli.main)


@pytest.mark.parametrize("git", ["/usr/bin/git", None])
def test_install_from_nested_directory(
    git: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    installed: list[Path] = []
    monkeypatch.setattr(sys, "argv", ["codedoctor", "install"])
    monkeypatch.setattr(install, "install", installed.append)

    def root(
        argv: Sequence[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert list(argv) == [git or "git", "rev-parse", "--show-toplevel"]
        assert check
        assert capture_output
        assert text
        return subprocess.CompletedProcess(argv, 0, stdout=str(tmp_path) + "\n")

    def which(name: str) -> str | None:
        assert name == "git"
        return git

    monkeypatch.setattr(subprocess, "run", root)
    monkeypatch.setattr(shutil, "which", which)
    assert cli.main() == 0
    assert installed == [tmp_path]
    assert capsys.readouterr().out == (
        "Codedoctor installed: Ruff, ty, paired tests, and pre-push checks.\n"
        "Check existing files: uv run --group pre-commit pre-commit run --all-files\n"
    )


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("uv"),
        ValueError("bad config"),
        subprocess.CalledProcessError(7, ["uv"]),
    ],
)
def test_install_reports_failure(
    failure: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["codedoctor", "install"])

    def fail(root: Path) -> None:
        assert root == tmp_path
        raise failure

    def root(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess([], 0, stdout=str(tmp_path))

    monkeypatch.setattr(install, "install", fail)
    monkeypatch.setattr(subprocess, "run", root)
    assert cli.main() == (
        failure.returncode if isinstance(failure, subprocess.CalledProcessError) else 1
    )
    assert capsys.readouterr().err


def test_install_outside_git_does_not_touch_files(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["codedoctor", "install"])

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.CalledProcessError(128, ["git"], stderr="not a git repository")

    monkeypatch.setattr(subprocess, "run", fail)
    assert cli.main() == 128
    assert capsys.readouterr().err == "codedoctor: not a git repository\n"


@pytest.mark.parametrize("doc", [None, " Explain setup. "])
def test_help_description_tracks_the_module(
    doc: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cli._add_arguments

    def arguments(parser: argparse.ArgumentParser) -> None:
        assert parser.description == (doc or "").strip()
        original(parser)

    monkeypatch.setattr(cli, "__doc__", doc)
    monkeypatch.setattr(cli, "_add_arguments", arguments)
    monkeypatch.setattr(sys, "argv", ["codedoctor", "--help"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
