"""Derive portable checker defaults and refresh only installer-owned settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, overload

from tomlkit.container import OutOfOrderTableProxy
from tomlkit.items import InlineTable, Table
from tomlkit.toml_document import TOMLDocument
from treekle import MutablePlainTree, from_plain

import tomlkit


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


_THIS: Final = Path(__file__).resolve()


type TomlTable = Table | InlineTable | OutOfOrderTableProxy


def defaults(*, name: str, python_version: str = "3.12") -> dict[str, MutablePlainTree]:
    """Read portable policy from the same configuration this package checks itself.

    Args:
      name: Consumer's first-party package and user-directory namespace.
      python_version: Oldest supported Python major/minor version.

    Returns:
      tools: Native checker tables, with checkout-specific settings removed.

    """
    template = _THIS.with_name("install_defaults.toml")
    if not template.is_file():
        template = next(
            folder / "pyproject.toml"
            for folder in _THIS.parents
            if (folder / "pyproject.toml").is_file()
        )
    content = template.read_text()
    document = tomlkit.parse(content)
    tools = from_plain(
        table(document, key="tool").unwrap(),
        dict[str, MutablePlainTree],
    )
    result = {
        key: value
        for key, value in tools.items()
        if key
        in {
            "ruff",
            "ty",
            "basedpyright",
            "pyrefly",
            "mutmut",
            "pytest",
            "coverage",
            "coverage-floor",
        }
    }
    table(result, key="mutmut")["source_paths"] = ["."]
    ruff = table(result, key="ruff")
    ruff["target-version"] = "py" + python_version.replace(".", "")
    lint = table(ruff, key="lint")
    table(lint, key="isort")["known-first-party"] = [name.replace("-", "_")]
    banned = table(table(lint, key="flake8-tidy-imports"), key="banned-api")
    banned.pop("pathlib.Path.home", None)
    based = table(result, key="basedpyright")
    based.pop("executionEnvironments", None)
    based.pop("stubPath", None)
    based["pythonVersion"] = python_version
    based["extraPaths"] = ["typings", ".venv/share/rekursiv-ai-typings"]
    table(table(result, key="ty"), key="environment")["python-version"] = python_version
    table(table(result, key="ty"), key="environment")["extra-paths"] = [
        "typings",
        ".venv/share/rekursiv-ai-typings",
    ]
    pyrefly = table(result, key="pyrefly")
    pyrefly["python-version"] = python_version + ".0"
    pyrefly["search-path"] = ["typings", ".venv/share/rekursiv-ai-typings"]
    pyrefly.pop("python-interpreter-path", None)
    pyrefly.pop("python-platform", None)
    exclusions = from_plain(
        table(tools, key="houselint").get("exclude"),
        list[str],
        default=[],
    )
    result["houselint"] = {
        "select": "ALL",
        "fix": "ALL",
        "exclude": [
            spec.replace("*_test.py", "*_test.py test_*.py")
            for spec in exclusions
            if set(spec.partition("=")[0].split())
            & {"*_test.py", "**/tests/test*.py", "conftest.py"}
        ],
        "set": [
            f"import-alias.first-party={name.replace('-', '_')}",
            "test-footer.runner=codedoctor.lib.testing.runner",
            f"userdirs-namespace.namespace={name}",
        ],
    }
    return from_plain(result, dict[str, MutablePlainTree])


def apply(document: TOMLDocument, *, defaults: Mapping[str, MutablePlainTree]) -> None:
    """Merge new defaults using the previous generated policy as a baseline.

    Args:
      document: Consumer project, modified in place without reformatting its policy.
      defaults: New canonical portable tool tables.

    """
    tools = table(document, key="tool")
    metadata = table(tools, key="codedoctor")
    previous = from_plain(
        table(metadata, key="managed").unwrap(),
        dict[str, MutablePlainTree],
    )
    old = dict(_leaves(previous))
    new = dict(_leaves(defaults))
    current = from_plain(tools.unwrap(), dict[str, MutablePlainTree])
    for path in sorted(old.keys() | new.keys()):
        present = _lookup(current, path=path)
        if path in new:
            if present != new[path] and (
                (path not in old and present is None)
                or (path in old and present is not None and present == old[path])
            ):
                _assign(tools, path=path, value=new[path])
        elif present is not None and present == old[path]:
            _assign(tools, path=path, value=None)
    if previous != defaults:
        metadata["managed"] = dict(defaults)


def overrides(document: TOMLDocument) -> tuple[str, ...]:
    """List native settings that differ from their recorded canonical defaults.

    Args:
      document: Configured consumer project.

    Returns:
      paths: Dotted setting names, including deliberately deleted defaults.

    """
    tools = table(document, key="tool")
    managed = table(table(tools, key="codedoctor"), key="managed")
    current = from_plain(tools.unwrap(), dict[str, MutablePlainTree])
    return tuple(
        sorted(
            ".".join(path)
            for path, value in _leaves(
                from_plain(managed.unwrap(), dict[str, MutablePlainTree]),
            )
            if _lookup(current, path=path) != value
        ),
    )


@overload
def table(
    parent: TOMLDocument | TomlTable,
    *,
    key: str,
) -> TomlTable: ...


@overload
def table(
    parent: dict[str, MutablePlainTree],
    *,
    key: str,
) -> dict[str, MutablePlainTree]: ...


def table(
    parent: TOMLDocument | TomlTable | dict[str, MutablePlainTree],
    *,
    key: str,
) -> TomlTable | dict[str, MutablePlainTree]:
    """Return a mutable table, rejecting malformed project configuration.

    Args:
      parent: TOML container or plain policy table.
      key: Child table to access or create.

    Returns:
      child: Existing table, or a newly inserted one.

    Raises:
      TypeError: The existing value is not a table.

    """
    if isinstance(parent, (TOMLDocument, Table, InlineTable, OutOfOrderTableProxy)):
        if key not in parent:
            parent[key] = tomlkit.table()
        value = cast(object, parent[key])
        if not isinstance(value, (Table, InlineTable, OutOfOrderTableProxy)):
            raise TypeError(f"Expected a TOML table for {key!r}.")
        return value
    child = parent.setdefault(key, {})
    if not isinstance(child, dict):
        raise TypeError(f"Expected a TOML table for {key!r}.")
    return child


def _leaves(
    tree: Mapping[str, MutablePlainTree],
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], MutablePlainTree]]:
    for key, value in tree.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from _leaves(value, path)
        else:
            yield path, value


def _lookup(
    tools: dict[str, MutablePlainTree],
    *,
    path: tuple[str, ...],
) -> MutablePlainTree:
    current: MutablePlainTree = tools
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _assign(
    tools: TomlTable,
    *,
    path: tuple[str, ...],
    value: MutablePlainTree,
) -> None:
    parent = tools
    for segment in path[:-1]:
        if segment in parent and not isinstance(
            parent[segment],
            (Table, InlineTable, OutOfOrderTableProxy),
        ):
            return
        parent = table(parent, key=segment)
    if value is None:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value
