"""Where a module's tests live: the one pairing every gate selects tests by.

A source ``pkg/sub/mod.py`` pairs with ``mod_test.py`` or ``test_mod.py`` beside
it, in a ``tests/`` beside it, or in a root ``tests/`` mirroring its path below
the top-level package (``tests/sub/mod_test.py``), the layout a published
package keeps its tests in. A test file pairs with the same set, so its
siblings under another spelling run with it.

A source's split tests -- ``mod_<part>_test.py`` or ``test_mod_<part>.py``,
beside it or in its ``tests/`` -- are its tests too, unless a longer module
(``mod_<part>.py``) owns that name.

A test module is named as pytest collects one -- ``conftest.py``, ``*_test.py``,
``test_*.py`` -- or a hand-run ``*_parity.py`` check against a real system, or is
another suite run by hand, told by the tests it defines.
Neither is a source a test must cover.
"""

from __future__ import annotations

from pathlib import Path

import ast


def candidates(path: str | Path) -> tuple[Path, ...]:
    """Return every repo-relative path a test paired with ``path`` may have.

    Args:
      path: A repo-relative source or test module.

    Returns:
      candidates: The paths to probe, nearest first; none need exist.

    """
    source = Path(path)
    stem = source.stem.removesuffix("_test").removeprefix("test_")
    names = (f"{stem}_test.py", f"test_{stem}.py")
    folders = [source.parent, source.parent / "tests"]
    if len(source.parts) > 1 and source.parts[0] != "tests":
        folders.append(Path("tests", *source.parts[1:-1]))
    return tuple(folder / name for folder in folders for name in names)


def paired(path: str | Path, *, root: Path | None = None) -> list[Path]:
    """Return the tests of ``path`` that exist: its sidecars, then a source's splits.

    Args:
      path: A repo-relative source or test module.
      root: The directory ``path`` is relative to; the cwd when None.

    Returns:
      tests: Existing repo-relative test files, in probe order. A test pairs
        with its sidecars alone; a changed test reruns itself, never its
        module's splits.

    """
    base = root or Path()
    source = Path(path)
    found = [
        candidate for candidate in candidates(source) if (base / candidate).is_file()
    ]
    stem = source.stem
    if stem.endswith("_test") or stem.startswith("test_"):
        return found
    owners = [
        module.stem
        for module in (base / source.parent).glob(f"{stem}_*.py")
        if module.is_file() and not module.stem.endswith("_test")
    ]
    for directory in (source.parent, source.parent / "tests"):
        for pattern in (f"{stem}_*_test.py", f"test_{stem}_*.py"):
            for match in sorted((base / directory).glob(pattern)):
                name = match.stem.removeprefix("test_")
                if match.is_file() and not any(
                    name == owner or name.startswith(f"{owner}_") for owner in owners
                ):
                    found.append(directory / match.name)
    return found


def related(path: str | Path, *, root: Path) -> list[Path]:
    """Return ``path``, its source when it is a test, and that source's tests.

    Args:
      path: A repo-relative source or test file.
      root: The repository root.

    Returns:
      files: Sorted, existing source and test paths; an unpaired path stays selected.

    """
    named = Path(path)
    source = named
    if is_test(root / named):
        stem = named.stem.removesuffix("_test").removeprefix("test_")
        folders = [named.parent]
        if named.parent.name == "tests":
            folders.append(named.parent.parent)
        if named.parts[0] == "tests":
            folders.extend(
                package / Path(*named.parts[1:-1])
                for package in root.iterdir()
                if package.is_dir() and not package.name.startswith(".")
            )
        stems = [stem.rsplit("_", count)[0] for count in range(stem.count("_") + 1)]
        owners = (
            folder / f"{candidate}.py" for candidate in stems for folder in folders
        )
        source = next(
            (
                owner.relative_to(root) if owner.is_absolute() else owner
                for owner in owners
                if (root / owner).is_file()
                and not is_test(root / owner)
                and named
                in paired(
                    owner.relative_to(root) if owner.is_absolute() else owner,
                    root=root,
                )
            ),
            named,
        )
    return sorted({named, source, *paired(source, root=root)})


def is_test(path: str | Path) -> bool:
    """Return whether ``path`` is a test module rather than a source one.

    Args:
      path: A module on disk, or a name; a name that is not a file is judged by
        its name alone.

    Returns:
      test: Whether it is named as a test module, or defines a top-level
        ``test_*`` function or ``Test*`` class without opting out through
        ``__test__ = False``.

    """
    module = Path(path)
    name = module.name
    # A hand-run ``*_parity.py`` script is a test no other test pairs, whether or not
    # it defines a pytest test.
    if (
        name == "conftest.py"
        or name.endswith(("_test.py", "_parity.py"))
        or name.startswith("test_")
    ):
        return True
    if not module.is_file():
        return False
    try:
        tree = ast.parse(module.read_bytes())
    except SyntaxError:
        return False
    for node in tree.body:
        match node:
            case ast.Assign(targets=targets, value=ast.Constant(value=False)) if any(
                isinstance(target, ast.Name) and target.id == "__test__"
                for target in targets
            ):
                return False
            case ast.AnnAssign(
                target=ast.Name(id="__test__"),
                value=ast.Constant(value=False),
            ):
                return False
            case _:
                pass
    return any(
        (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name.startswith("test_")
        )
        or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
        for node in tree.body
    )
