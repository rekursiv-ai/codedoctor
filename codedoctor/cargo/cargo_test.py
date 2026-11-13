"""Cargo hooks run once per crate a commit touches."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import shutil
import subprocess
import sys

import pytest

from codedoctor.cargo import cargo


if TYPE_CHECKING:
    from collections.abc import Sequence


def _crate(root: Path, name: str) -> Path:
    (root / name / "src").mkdir(parents=True)
    manifest = root / name / "Cargo.toml"
    manifest.write_text(f'[package]\nname = "{name}"\n')
    return manifest


def test_each_file_maps_to_its_nearest_crate_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _crate(tmp_path, "outer")
    _crate(tmp_path / "outer", "inner")
    assert cargo.manifests(
        [
            Path("outer/inner/src/lib.rs"),
            Path("outer/src/main.rs"),
            Path("outer/inner/Cargo.toml"),
            Path("outer/Cargo.lock"),
            Path("loose.rs"),
        ],
    ) == [Path("outer/inner/Cargo.toml"), Path("outer/Cargo.toml")]


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("fmt", ["fmt"]),
        ("clippy", ["clippy", "--all-targets", "--locked", "--", "-D", "warnings"]),
    ],
)
@pytest.mark.parametrize("found", [True, False])
def test_each_crate_runs_once_and_any_failure_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    arguments: list[str],
    found: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    _crate(tmp_path, "a")
    _crate(tmp_path, "b")
    calls: list[list[str]] = []

    def run(argv: Sequence[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 3 if "a/Cargo.toml" in argv else 0)

    def which(name: str) -> str | None:
        return f"/bin/{name}" if found else None

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(
        sys,
        "argv",
        ["codedoctor cargo", command, "a/src/lib.rs", "b/src/lib.rs", "a/Cargo.toml"],
    )
    assert cargo.main() == 3
    executable = "/bin/cargo" if found else "cargo"
    assert calls == [
        [
            executable,
            arguments[0],
            "--manifest-path",
            f"{crate}/Cargo.toml",
            *arguments[1:],
        ]
        for crate in ("a", "b")
    ]


def test_files_outside_every_crate_run_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", pytest.fail)
    monkeypatch.setattr(sys, "argv", ["codedoctor cargo", "fmt", "loose.rs"])
    assert cargo.main() == 0


@pytest.mark.parametrize(
    ("docstring", "description"),
    [
        (cargo.__doc__, "Run a cargo command once per crate"),
        ("  Custom description.  \n", "\n\nCustom description.\n\n"),
        (None, "files ...]\n\npositional"),
    ],
)
def test_help_describes_the_command(
    docstring: str | None,
    description: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cargo, "__doc__", docstring)
    monkeypatch.setattr(sys, "argv", ["codedoctor cargo", "--help"])
    with pytest.raises(SystemExit):
        cargo.main()
    assert description in capsys.readouterr().out


def test_an_unknown_command_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["codedoctor cargo", "build", "lib.rs"])
    with pytest.raises(SystemExit):
        cargo.main()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
