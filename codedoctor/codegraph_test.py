"""Managed CodeGraph installation is verified, private, and reusable."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
from urllib import request

import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile

import pytest

from codedoctor import codegraph
from codedoctor.affected import backbone
from codedoctor.lib.userdirs import cache_dir


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@pytest.fixture(autouse=True)
def without_path_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    real = shutil.which

    def which(name: str) -> str | None:
        return None if name == "codegraph" else real(name)

    monkeypatch.setattr(shutil, "which", which)


def test_existing_executable_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def which(name: str) -> str | None:
        return "/tools/codegraph" if name == "codegraph" else None

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", _runner(calls))
    assert codegraph.executable() == "/tools/codegraph"
    codegraph.install()
    assert calls == [["/tools/codegraph", "--version"]]


def test_missing_executable_is_none() -> None:
    assert codegraph.executable() is None


def test_managed_executable_is_found_without_path() -> None:
    binary = (
        cache_dir()
        / "rekursiv-ai"
        / "codedoctor"
        / "codegraph"
        / "v1.6.2"
        / "bin"
        / "codegraph"
    )
    binary.parent.mkdir(parents=True)
    binary.write_text("managed binary")
    binary.chmod(0o755)
    assert codegraph.executable() == str(binary)


@pytest.mark.parametrize("valid_checksum", [False, True])
def test_download_is_checked_before_publishing_and_reused(
    monkeypatch: pytest.MonkeyPatch,
    valid_checksum: bool,
) -> None:
    payload = _archive()
    monkeypatch.setattr(
        codegraph,
        "_release",
        lambda: (
            "linux-x64",
            hashlib.sha256(payload if valid_checksum else b"wrong").hexdigest(),
        ),
    )
    requests: list[str] = []
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _runner(calls))

    def download(url: str, *, timeout: int) -> BytesIO:
        assert timeout > 0
        requests.append(url)
        return BytesIO(payload)

    monkeypatch.setattr(request, "urlopen", download)
    if not valid_checksum:
        with pytest.raises(ValueError, match="checksum"):
            codegraph.install()
        assert codegraph.executable() is None
        assert calls == []
        return
    codegraph.install()
    binary = codegraph.executable()
    assert binary is not None
    assert Path(binary).read_bytes() == b"standalone launcher"
    assert calls[0][1:] == ["--version"]
    assert calls[0][0] != binary
    codegraph.install()
    assert calls[-1] == [binary, "--version"]
    assert requests == [
        "https://github.com/colbymchenry/codegraph/releases/download/v1.6.2/codegraph-linux-x64.tar.gz",
    ]


def test_broken_existing_executable_fails_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def which(name: str) -> str | None:
        return "/broken/codegraph" if name == "codegraph" else None

    monkeypatch.setattr(shutil, "which", which)

    def fail(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert list(argv) == ["/broken/codegraph", "--version"]
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        codegraph.install()


@pytest.mark.parametrize(
    ("system", "machine", "target"),
    [
        ("linux", "x86_64", "linux-x64"),
        ("linux", "aarch64", "linux-arm64"),
        ("darwin", "arm64", "darwin-arm64"),
        ("darwin", "amd64", "darwin-x64"),
    ],
)
def test_supported_platforms_select_a_checksummed_release(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    target: str,
) -> None:
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    selected, checksum = codegraph._release()
    assert selected == target
    assert len(bytes.fromhex(checksum)) == 32


def test_unsupported_platform_has_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "unsupported")
    with pytest.raises(ValueError, match="unsupported"):
        codegraph._release()


def test_a_concurrent_install_is_reused_after_locking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = iter([None, "/managed/codegraph"])
    calls: list[list[str]] = []
    monkeypatch.setattr(codegraph, "executable", lambda: next(found))
    monkeypatch.setattr(subprocess, "run", _runner(calls))
    codegraph.install()
    assert calls == [["/managed/codegraph", "--version"]]


@pytest.mark.network_github
def test_standalone_release_installs_and_indexes_without_path(
    tmp_path: Path,
) -> None:
    assert codegraph.executable() is None
    codegraph.install()
    binary = codegraph.executable()
    assert binary is not None
    assert Path(binary).is_relative_to(cache_dir())
    codegraph.install()
    (tmp_path / "example.py").write_text("def example(): return 1\n")

    def run(argv: Sequence[str]) -> int:
        assert argv[0] == binary
        return subprocess.run(argv, check=False).returncode  # noqa: S603 -- Resolved managed CLI; fixed indexing arguments.

    assert backbone.sync_worktree(tmp_path, caches=tmp_path / "caches", run=run) == ""
    assert (tmp_path / ".codegraph" / "codegraph.db").is_file()


def _archive() -> bytes:
    stream = BytesIO()
    content = b"standalone launcher"
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        entry = tarfile.TarInfo("codegraph-linux-x64/bin/codegraph")
        entry.size = len(content)
        entry.mode = 0o755
        archive.addfile(entry, BytesIO(content))
    return stream.getvalue()


def _runner(calls: list[list[str]]) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(
        argv: Sequence[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check
        assert capture_output
        assert text
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="1.6.2\n")

    return run


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
