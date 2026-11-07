"""Run every exported package's leak policy before the commit leaves this machine.

Copybarista enforces `[[leak_check.forbidden_text]]` during the EXPORT, which
runs on a push to main -- so a violation is reported ~50s after the commit is
already unrewritable, and every later export of that package stays red until
someone lands a follow-up. That happened twice in a row for one docstring
naming an internal provider class.

The check itself is cheap (~4.5s per package) and needs nothing a laptop lacks,
so there is no reason it should run only on a runner. This is the same move
`check-copybarista-pr-text` already makes for the PR-metadata half of the same
policy.

Scoped to packages whose files the commit actually touched: exporting all eight
costs ~35s, and a commit under one package cannot leak through another's rules.

Usage:
    check_export_leaks.py [--workers N] [<changed-file> ...]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import os
import subprocess
import sys
import tempfile
import tomllib

from codedoctor.lib.worker_count import capped_workers, worker_count


_CWD: Final = Path(__file__).resolve().parent
# The checkout this module lives in. Installed from a wheel there is none: no sync
# config lies under it, so every function below finds no package and the gate
# passes, as it should in a repository that exports nothing.
_ROOT: Final = _CWD.parents[2]


def package_configs(
    *,
    root: Path = _ROOT,
    sync_patterns: tuple[str, ...] = (
        "loop/*/copybarista.sync.toml",
        "loop/*/*/copybarista.sync.toml",
        "loop/*/*/*/copybarista.sync.toml",
        "website/*/copybarista.sync.toml",
    ),
) -> list[Path]:
    """Every live synced package's export config.

    Discovered rather than listed: a hand-maintained tuple in this repo had
    already drifted to three packages while seven were synced.

    Args:
      root: The checkout to search.
      sync_patterns: Globs locating a package's sync config. A package lives
        one to three levels under ``loop/``, a site one level under
        ``website/``; the recursive ``loop/**`` form also
        matches copybarista's own fixtures and doc examples, which carry a
        ``copy.barista.toml`` to exercise edge cases and are not repositories
        anyone syncs -- several deliberately fail to export. A neighbouring
        ``copybarista.sync.toml`` is what makes a package LIVE.

    Returns:
      configs: Each live package's ``copy.barista.toml``, in a stable order.

    """
    return [
        config
        for pattern in sync_patterns
        for sync in sorted(root.glob(pattern))
        if (config := sync.parent / "copy.barista.toml").is_file()
    ]


# The fleet checks of what a Python package ships (its pyproject, CI, wheel test)
# read their packages from here; the site exports no pyproject for them to read.
def python_package_configs(*, root: Path = _ROOT) -> list[Path]:
    """Return the ``package_configs`` whose sync config declares a Python package."""
    return [config for config in package_configs(root=root) if _python_package(config)]


def touched_configs(paths: Sequence[str], *, root: Path = _ROOT) -> list[Path]:
    """Configs whose package exports at least one of `paths`.

    A package's inputs are not the files under its directory: every config
    declares ``[[files.copy]]`` sources reaching outside it -- shared
    boilerplate, the repo-root pre-commit config, vendored library
    modules. Selecting by directory containment alone skipped the gate for
    every one of those, so a leak introduced in a file that ships to eight
    public repositories went unchecked.

    An empty `paths` means "check everything" -- that is what a manual run or
    a `--all-files` invocation wants.

    The repo-root ``conftest.py`` is an input to EVERY package: its
    ``pytest_plugins`` tuple decides which packages must vendor and declare a
    plugin module, yet no config copies it, so containment alone selects
    nothing.

    Args:
      paths: Relative repo paths that may have changed.
      root: The checkout the paths are relative to.

    Returns:
      configs: Package configs whose exports include at least one path.

    """
    configs = package_configs(root=root)
    if not paths:
        return configs
    # Lexically normalized, not ``resolve()``d: a symlink-free checkout makes them
    # equal, and resolving each path cost a ``realpath`` syscall chain per file --
    # 100s of the 104s an ``--all-files`` run of 10,675 paths took.
    changed = {Path(os.path.normpath(root / path)) for path in paths}
    if Path(os.path.normpath(root / "conftest.py")) in changed:
        return configs
    # Each package's input roots once, not once per changed path: the TOML parse
    # and source resolution were repeated paths x packages times.
    return [
        config
        for config in configs
        if _touches(
            changed,
            roots=(config.parent.resolve(), *_input_roots(config, root=root)),
        )
    ]


def check(config: Path, *, root: Path = _ROOT) -> str:
    """Export `config`'s package to a scratch tree; return stderr on failure.

    The leak policy runs as part of the export rather than standalone, because
    the terms are matched against the EXPORTED text -- a `[[transform]]` may
    rewrite a private symbol on the way out, so scanning the source tree would
    report leaks the public repo never receives.

    The exported tree is then format-checked for the same reason (see
    `_format_check`).

    Args:
      config: Path to copy.barista.toml (package export config).
      root: The checkout ``config`` lies in, where the export runs.

    Returns:
      error_msg: Stderr output on export/format failure; empty string if OK.

    """
    with tempfile.TemporaryDirectory(prefix="leak-check-") as scratch:
        completed = subprocess.run(  # noqa: S603 -- The export command uses a fixed executable and internally assembled arguments.
            [
                sys.executable,
                "-m",
                "copybarista",
                "export",
                str(config.relative_to(root)),
                ".",
                "--folder-dir",
                scratch,
                "--force",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return completed.stderr.strip() or f"export exited {completed.returncode}"
        return _format_check(Path(scratch))


def main() -> int:
    """Export each touched package and report any leak-policy violation.

    Returns:
      exit_code: 0 if all exports pass leak check, 1 on any failure.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        # The gate hands every path as one @argfile; argv overflows ARG_MAX.
        fromfile_prefix_chars="@",  # pragma: no mutate -- holds "@" either way.
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    paths = flags.paths
    workers = flags.workers

    configs = touched_configs(paths)
    if not configs:
        return 0
    # Each check is a subprocess doing its own work in a private scratch tree,
    # so a thread only waits on ``subprocess.run`` -- the GIL is released and
    # concurrency is bounded by the child processes, not by Python.
    with ThreadPoolExecutor(
        max_workers=capped_workers(workers, units=len(configs)),
    ) as pool:
        checks = {config: pool.submit(check, config) for config in configs}

    failures = {
        config: error for config, done in checks.items() if (error := done.result())
    }
    for config, error in failures.items():
        print(  # noqa: T201 -- CLI output is this command's user-facing error product.
            f"{config.relative_to(_ROOT)}:\n{error}",
            file=sys.stderr,
        )
    return 1 if failures else 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("paths", nargs="*")
    parser.add_argument(
        "--workers",
        type=worker_count,
        # Serial by default, and PASSED IN: the gate sizes the count, so these
        # exports share the budget every other gate gets.
        default=1,
        help="Packages exported concurrently. Empty or non-positive means serial.",
    )


# Reads the config's own ``[[files.copy]]`` sources rather than restating which files
# are shared, so a package that starts copying a new external file is covered without
# touching this gate.
def _input_roots(config: Path, *, root: Path) -> tuple[Path, ...]:
    """Return the repo paths `config` copies into its export, normalized."""
    return tuple(
        Path(os.path.normpath(root / source)) for source in _copy_sources(config)
    )


# Parsed with ``tomllib`` rather than copybarista's ``config`` so the gate stays
# importable without the monorepo package -- it runs from a pre-commit hook, before any
# project install is guaranteed.
def _copy_sources(config: Path) -> tuple[str, ...]:
    """Return the ``[[files.copy]]`` source paths declared by one config."""
    try:
        raw = tomllib.loads(config.read_bytes().decode())
    except (OSError, tomllib.TOMLDecodeError):
        return ()
    # ``tomllib`` types its result ``dict[str, Any]``, and ``isinstance`` on a
    # bare ``dict`` narrows to ``dict[Unknown, Unknown]`` -- either way every
    # ``.get`` below yields an unknown. Casting each level to a concrete
    # ``Mapping[str, object]`` keeps the checker tracing real types, and the
    # ``isinstance`` guards still verify the shape at runtime.
    files = raw.get("files")
    if not isinstance(files, dict):
        return ()
    copies = cast(Mapping[str, object], files).get("copy")
    if not isinstance(copies, list):
        return ()
    sources: list[str] = []
    for entry in cast(Sequence[object], copies):
        if not isinstance(entry, dict):
            continue
        source = cast(Mapping[str, object], entry).get("source")
        if isinstance(source, str):
            sources.append(source)
    return tuple(sources)


def _python_package(config: Path) -> bool:
    """Return whether ``config``'s sync config declares a Python package."""
    path = config.with_name("copybarista.sync.toml")
    # Parsed with ``tomllib`` for the reason ``_copy_sources`` gives. An absent key
    # takes ``SyncSettings.python_package``'s default.
    sync = cast(Mapping[str, object], tomllib.loads(path.read_bytes().decode())["sync"])
    return sync.get("python_package", True) is True


# A `[[transform]]` changes line LENGTH (`mono.pkg` -> `pkg`), and ruff's
# formatting is length-sensitive: a call that legitimately wraps across three lines in
# the monorepo fits on one after the rewrite. Both trees are then "correctly formatted"
# by their own measure, but the public repo's `ruff format` hook reformats on arrival
# and fails the export -- a red CI run for a file that is clean locally by construction.
# Only the post-transform text can settle it, so the check belongs here, on the tree
# already exported for the leak scan.
def _format_check(tree: Path) -> str:
    """Return ruff's complaint if the EXPORTED tree is not format-clean."""
    completed = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return ""
    # Paths are scratch-relative; the package-relative tail is what a reader
    # needs to find the source file.
    return (
        "exported tree is not format-clean (the export rewrites line lengths; "
        "run `ruff format` on the exported tree to see the wrapping it wants):\n"
        + completed.stdout.strip()
    )


class _Flags(Protocol):
    paths: Sequence[str]
    workers: int


def _touches(changed: set[Path], *, roots: Sequence[Path]) -> bool:
    """Return whether any ``changed`` path is a root or lies under one."""
    for path in changed:
        # Every ancestor, so containment is a set lookup per level, not a scan.
        if path in roots or any(parent in roots for parent in path.parents):
            return True
    return False
