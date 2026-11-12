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

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
import tomllib

from codedoctor.lib.worker_count import capped_workers, worker_count


_CWD: Final[Path] = Path(__file__).resolve().parent
# The checkout this module lives in. Installed from a wheel there is none: no sync
# config lies under it, so every function below finds no package and the gate
# passes, as it should in a repository that exports nothing.
_ROOT: Final = _CWD.parents[2]

_REQUIREMENT_NAME: Final = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

_IMPORT_LINE: Final = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][A-Za-z0-9_]*)|import\s+([^\n#]+))",
    re.MULTILINE,
)


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
    `_format_check`), its own imports resolved (see `_import_check`), and its
    stubbed imports matched to a stubs dependency (see `_stub_check`).

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
        tree = Path(scratch)
        errors = (_format_check(tree), _import_check(tree), _stub_check(tree, root))
        return "\n".join(error for error in errors if error)


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


def _import_check(tree: Path) -> str:
    """Return each import of the EXPORTED tree's own code that it does not ship."""
    unresolved = _unresolved_imports(tree)
    if not unresolved:
        return ""
    return "exported imports of unshipped code:\n" + "\n".join(unresolved)


def _stub_check(tree: Path, root: Path) -> str:
    """Return the stubbed packages the EXPORTED tree imports without their stubs."""
    stubs = root / "loop" / "typings" / "stubs"
    stubbed = frozenset(path.name for path in stubs.glob("*") if path.is_dir())
    undeclared = _undeclared_stubs(tree, stubbed)
    if not undeclared:
        return ""
    return (
        "exported tree imports stubbed packages without depending on "
        f"rekursiv-ai-typings: {', '.join(undeclared)}"
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


# A package's ``files.exclude`` can drop a module a shipped file imports, and an
# ``internal`` fence can strip a name one imports. The leak policy passes on both,
# because nothing forbidden remains; only the exported tree shows it. An import is the
# tree's own when its first dotted component names a top-level module or package the
# tree ships; every other is a dependency.
def _unresolved_imports(tree: Path) -> list[str]:
    """Return ``path: dotted`` for each import of the tree's own code it lacks."""
    module_names: dict[str, set[str]] = {}
    imports: list[tuple[Path, str, str | None]] = []
    for path in sorted(tree.rglob("*.py")):
        relative = path.relative_to(tree)
        if any(part.startswith(".") for part in relative.parts):
            continue
        text = path.read_text(
            encoding="utf-8",
        )  # pragma: no mutate -- this host's locale is UTF-8 too.
        source = ast.parse(text, filename=str(path))
        module_names.setdefault(_module_name(relative), set()).update(
            _top_level_names(source),
        )
        for node in _statements(source):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and not node.level
            ):
                imports.extend(
                    (relative, node.module, alias.name) for alias in node.names
                )
            elif isinstance(node, ast.Import):
                imports.extend((relative, alias.name, None) for alias in node.names)
    shipped = {name.split(".")[0] for name in module_names}
    module_dirs = {
        ".".join(directory.relative_to(tree).parts)
        for directory in tree.rglob("*")
        if directory.is_dir()
    }
    unresolved: list[str] = []
    for relative, module, name in imports:
        if module.split(".")[0] not in shipped:
            continue
        dotted = _unresolved(module, name, module_names, module_dirs)
        if dotted is not None:
            unresolved.append(f"{relative.as_posix()}: {dotted}")
    return unresolved


def _module_name(relative: Path) -> str:
    """Return a source file's import name, accounting for package initializers."""
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


# An import is a statement, and statements nest only in statements, handlers and match
# cases: skipping every expression skips nine nodes in ten of a full ``ast.walk``.
def _statements(source: ast.Module) -> Iterator[ast.stmt]:
    """Yield every statement of ``source``, nested ones included."""
    pending: deque[ast.AST] = deque(source.body)
    while pending:
        node = pending.popleft()
        if isinstance(node, ast.stmt):
            yield node
        pending.extend(
            child
            for child in ast.iter_child_nodes(node)
            if isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case))
        )


def _unresolved(
    module: str,
    name: str | None,
    module_names: dict[str, set[str]],
    module_dirs: set[str],
) -> str | None:
    """Return the dotted path that fails to resolve, or None when it is shipped."""
    if name is not None:
        child = f"{module}.{name}"
        if child in module_names or child in module_dirs:
            return None
    names = module_names.get(module)
    if names is None:
        return None if name is None and module in module_dirs else module
    if name is None or name == "*" or name in names:
        return None
    return f"{module}.{name}"


def _top_level_names(source: ast.Module) -> set[str]:
    """Names bound in module scope, including conditional branches."""
    names: set[str] = set()
    nodes: list[ast.AST] = list(source.body)
    while nodes:
        node = nodes.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(
            node,
            (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        nodes.extend(ast.iter_child_nodes(node))
    return names


def _undeclared_stubs(tree: Path, stubbed: frozenset[str]) -> list[str]:
    """Return the stubbed packages ``tree`` imports without declaring their stubs."""
    pyproject = tree / "pyproject.toml"
    if not pyproject.is_file() or "rekursiv-ai-typings" in _requirements(pyproject):
        return []
    return sorted(_imported_top_level_packages(tree) & stubbed)


# A line-anchored regex, not ``ast.parse``: parsing priml's 362 exported modules cost
# 0.64s for a set the caller intersects with the stub trees, which discards any
# docstring noise the regex picks up. Stub files count: an exported ``.pyi`` needs its
# imports' stubs as much as a ``.py``. The tree's own ``typings/`` does not: it holds
# stubs, not users of them.
def _imported_top_level_packages(tree: Path) -> set[str]:
    """Return the top-level names on ``import``/``from`` lines under ``tree``."""
    typings = tree / "typings"
    sources = (
        path
        for path in (*tree.rglob("*.py"), *tree.rglob("*.pyi"))
        if not path.is_relative_to(typings)
    )
    packages: set[str] = set()
    for path in sources:
        text = path.read_text(
            encoding="utf-8",
        )  # pragma: no mutate -- this host's locale is UTF-8 too.
        for match in _IMPORT_LINE.finditer(text):
            from_name, import_names = match.group(1), match.group(2)
            if from_name:
                packages.add(from_name)
            else:
                packages.update(
                    name.split(".")[0].split()[0] for name in import_names.split(",")
                )
    return packages


# The project's own name counts: the stubs distribution ships its stubs.
def _requirements(pyproject: Path) -> set[str]:
    """Return every distribution the project provides or requires."""
    text = pyproject.read_text(
        encoding="utf-8",
    )  # pragma: no mutate -- this host's locale is UTF-8 too.
    document = tomllib.loads(text)
    project = cast(Mapping[str, object], document.get("project", {}))
    lists: list[object] = [[project.get("name")], project.get("dependencies", [])]
    for table in (
        project.get("optional-dependencies", {}),
        document.get("dependency-groups", {}),
    ):
        lists.extend(cast(Mapping[str, object], table).values())
    names: set[str] = set()
    for entries in lists:
        for entry in cast(Sequence[object], entries):
            # A ``{include-group = ...}`` table names no distribution.
            if isinstance(entry, str) and (match := _REQUIREMENT_NAME.match(entry)):
                names.add(match.group(1).lower().replace("_", "-"))
    return names
