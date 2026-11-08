"""Provision a private standalone CodeGraph and resolve it independently of PATH."""

from __future__ import annotations

from http.client import HTTPResponse
from pathlib import Path
from typing import Final, cast
from urllib import request

import fcntl
import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

from codedoctor.lib.userdirs import cache_dir


_VERSION: Final = "v1.6.2"
_SHA256: Final = {
    "darwin-arm64": "d74d1bfb4060db63ec3c2b72c4e17f76c31978af3bad6ace4370d1501ac0662e",
    "darwin-x64": "53d1a4d1a9af31d6cec11b346df2ae792087081e13f623f583e27783c1e7f9bc",
    "linux-arm64": "c8c6be292be21d00dea26bad8b28d434731cf612fb8cc475dc10f5b3038b5b64",
    "linux-x64": "ef0af416092128fb1ccc723786000b7edf4a6971fe604acd08bf966e4f37b828",
}


def executable() -> str | None:
    """Find an existing CodeGraph, including the privately managed release.

    Returns:
        binary: Resolved executable path, or None when CodeGraph is absent.

    """
    return shutil.which("codegraph") or shutil.which(
        str(
            cache_dir()
            / "rekursiv-ai"
            / "codedoctor"
            / "codegraph"
            / _VERSION
            / "bin"
            / "codegraph",
        ),
    )


def install() -> None:
    """Install and verify CodeGraph without modifying global tools or shell profiles.

    Reuse an executable already on PATH. Otherwise install the pinned upstream
    bundle, including its Node runtime, after verifying its published checksum.
    Concurrent installers share a lock; incomplete downloads are never published.

    Raises:
        ValueError: The platform is unsupported or the downloaded checksum differs.
        OSError: Downloading or publishing the standalone bundle failed.
        subprocess.CalledProcessError: The executable failed its version check.

    """
    if (binary := executable()) is not None:
        _verify(binary)
        return
    home = cache_dir() / "rekursiv-ai" / "codedoctor" / "codegraph"
    home.mkdir(parents=True, exist_ok=True)
    with (home / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (binary := executable()) is not None:
            _verify(binary)
            return
        target, checksum = _release()
        with tempfile.TemporaryDirectory(prefix="codegraph-", dir=home) as temporary:
            scratch = Path(temporary)
            archive = scratch / "bundle.tar.gz"
            url = f"https://github.com/colbymchenry/codegraph/releases/download/{_VERSION}/codegraph-{target}.tar.gz"
            with (
                cast(HTTPResponse, request.urlopen(url, timeout=300)) as response,
                archive.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            with archive.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != checksum:
                raise ValueError(
                    "CodeGraph download checksum does not match its release.",
                )
            with tarfile.open(archive) as bundle:
                bundle.extractall(scratch, filter="data")
            incoming = scratch / f"codegraph-{target}"
            _verify(str(incoming / "bin" / "codegraph"))
            incoming.rename(home / _VERSION)


def _release() -> tuple[str, str]:
    """Select a standalone archive and its upstream SHA256 checksum."""
    machine = platform.machine()
    arch = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "x64", "amd64": "x64"}.get(
        machine,
        machine,
    )
    target = f"{sys.platform}-{arch}"
    if target not in _SHA256:
        raise ValueError(f"CodeGraph has no supported standalone bundle for {target}.")
    return target, _SHA256[target]


def _verify(binary: str) -> None:
    subprocess.run(  # noqa: S603 -- Resolved executable; only the fixed version flag.
        [binary, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
