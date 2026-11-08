"""The product entry point finds the repository before installing."""

from __future__ import annotations

from types import SimpleNamespace
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
@pytest.mark.parametrize("notebooks", [False, True])
def test_install_from_nested_directory(
    git: str | None,
    notebooks: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    installed: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["codedoctor", "install", *(["--notebooks"] if notebooks else [])],
    )

    def configure(root: Path, *, notebooks: bool = False) -> tuple[str, ...]:
        installed.append((root, notebooks))
        return ()

    monkeypatch.setattr(install, "install", configure)

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
    assert installed == [(tmp_path, notebooks)]
    assert capsys.readouterr().out == (
        "Codedoctor installed: development tools and pre-commit Git hook.\n"
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

    def fail(root: Path, *, notebooks: bool = False) -> None:
        assert root == tmp_path
        assert not notebooks
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


@pytest.mark.parametrize("current", [True, False])
def test_check_reports_stale_policy_without_installing(
    current: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["codedoctor", "check"])

    def root(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess([], 0, stdout=str(tmp_path))

    def check(path: Path) -> bool:
        assert path == tmp_path
        return current

    monkeypatch.setattr(subprocess, "run", root)
    monkeypatch.setattr(install, "current", check)
    assert cli.main() == (0 if current else 1)
    captured = capsys.readouterr()
    assert captured.out == ("Codedoctor configuration is current.\n" if current else "")
    assert captured.err == (
        ""
        if current
        else "Codedoctor configuration is stale; run `codedoctor install`.\n"
    )


def test_install_reports_preserved_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["codedoctor", "install"])

    def root(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess([], 0, stdout=str(tmp_path))

    def configure(path: Path, *, notebooks: bool = False) -> tuple[str, ...]:
        assert path == tmp_path
        assert not notebooks
        return ("ruff.line-length", "ty.rules.all")

    monkeypatch.setattr(subprocess, "run", root)
    monkeypatch.setattr(install, "install", configure)
    assert cli.main() == 0
    assert (
        capsys.readouterr().err
        == "Codedoctor preserved project overrides: ruff.line-length, ty.rules.all\n"
    )


@pytest.mark.parametrize(
    ("command", "module"),
    [
        ("lint", "codedoctor.lint.__main__"),
        ("typecheck", "codedoctor.typecheck.__main__"),
        ("testing", "codedoctor.testing.__main__"),
        ("mutation", "codedoctor.mutation.__main__"),
        ("gate", "codedoctor.gate.__main__"),
        ("affected", "codedoctor.affected.__main__"),
        ("coverage-floor", "codedoctor.coverage_floor.__main__"),
        ("assert-isinstance", "codedoctor.assert_isinstance.__main__"),
        ("nbstripout", "codedoctor.nbstripout.__main__"),
        ("yamlcheck", "codedoctor.yamlcheck.__main__"),
        ("export-leaks", "codedoctor.export_leaks.__main__"),
        ("codegraph-sync", "codedoctor.affected.codegraph_sync"),
        ("check-pr-text", "codedoctor.export_leaks.check_pr_text"),
    ],
)
@pytest.mark.parametrize("arguments", [["--help"], ["--fix", "", "a.py", "--", "b.py"]])
def test_stage_dispatch_preserves_arguments_and_exit_code(
    command: str,
    module: str,
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ["codedoctor", command, *arguments]
    monkeypatch.setattr(sys, "argv", original)
    calls: list[list[str]] = []

    def run() -> int:
        calls.append(sys.argv.copy())
        return 7

    def load(name: str) -> SimpleNamespace:
        assert name == module
        return SimpleNamespace(main=run)

    monkeypatch.setattr(cli, "import_module", load, raising=False)
    assert cli.main() == 7
    assert calls == [[f"codedoctor {command}", *arguments]]
    assert sys.argv is original


def test_stage_dispatch_restores_arguments_after_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ["codedoctor", "lint", "--help"]
    monkeypatch.setattr(sys, "argv", original)

    def run() -> int:
        raise SystemExit(2)

    def load(name: str) -> SimpleNamespace:
        assert name == "codedoctor.lint.__main__"
        return SimpleNamespace(main=run)

    monkeypatch.setattr(cli, "import_module", load, raising=False)
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
    assert sys.argv is original


@pytest.mark.parametrize("command", ["install", "check", "unknown"])
def test_setup_and_unknown_commands_reject_extra_arguments(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["codedoctor", command, "--unknown", "value"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
    error = capsys.readouterr().err
    assert (
        "invalid choice: 'unknown'"
        if command == "unknown"
        else "unrecognized arguments: --unknown value"
    ) in error


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
