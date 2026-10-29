"""Default argparse flags from a ``[tool.<name>]`` table in ``pyproject.toml``.

A tool configured in ``pyproject.toml`` runs bare -- ``tool FILES`` -- and
picks up the project's settings, as ruff and pytest do. Each key in the
table is the long flag it defaults, spelled without ``--``: ``min-digits``
defaults ``--min-digits``. The command line still wins: a plain flag
replaces the table's value, and an ``append`` flag adds to it.

The table is checked against the parser rather than declared twice: a key
naming no flag, or a value whose TOML type differs from what that flag
parses to, is a usage error. Silently ignoring a typo would run the
defaults while the author believes their setting is in force.

The nearest ``pyproject.toml`` at or above the start directory that HAS the
table wins, so a nested package's own project file does not hide the
repo's settings. Its directory is returned too: paths in the table are
relative to it, not to wherever the tool was run from.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import argparse
import inspect
import tomllib

from treekle import ReadError, from_plain


def parse_args(
    parser: argparse.ArgumentParser,
    /,
    *,
    tool: str,
    argv: Sequence[str] | None = None,
    start: Path | None = None,
) -> tuple[argparse.Namespace, Path]:
    """Parse ``argv`` over the defaults in ``[tool.<tool>]``.

    Args:
      parser: The tool's parser, every flag already registered.
      tool: Table name under ``[tool]``.
      argv: Arguments to parse; ``sys.argv[1:]`` when None.
      start: Directory to search upward from; the cwd when None.

    Returns:
      flags: The parsed namespace.
      root: Directory of the ``pyproject.toml`` read, which paths in the
        table are relative to; ``start`` when no file has the table.

    """
    origin = Path.cwd() if start is None else start
    found = _find_table(origin, tool=tool, parser=parser)
    if found is None:
        return parser.parse_args(argv), origin
    table, path = found
    actions = {
        option.removeprefix("--"): action
        for action in parser._actions  # noqa: SLF001 -- argparse exposes no public action list.
        for option in action.option_strings
        if option.startswith("--")
    }
    where = f"{path} [tool.{tool}]"
    unknown = sorted(table.keys() - actions.keys())
    if unknown:
        parser.error(f"{where}: unknown key(s) {unknown}; valid: {sorted(actions)}")
    for key, value in table.items():
        action = actions[key]
        default = cast(object, action.default)
        # An unset (None) default takes its type from the flag's converter: a
        # type itself, or the type a converter function returns.
        expected = _converted(action.type) if default is None else type(default)
        if expected is None:
            parser.error(
                f"{where} {key}: its flag declares no default or typed converter",
            )
        if isinstance(value, bool) is not (expected is bool) or not isinstance(
            value,
            expected,
        ):
            parser.error(
                f"{where} {key}: expected {expected.__name__}, "
                f"got {type(value).__name__}",
            )
        action.default = value
    return parser.parse_args(argv), path.parent


def _converted(converter: object) -> type | None:
    """Return the type ``converter`` produces, or None when it does not say."""
    if isinstance(converter, type):
        return converter
    if not callable(converter):
        return None
    returned = cast(
        object,
        inspect.signature(converter, eval_str=True).return_annotation,
    )
    return returned if isinstance(returned, type) else None


def _find_table(
    start: Path,
    /,
    *,
    tool: str,
    parser: argparse.ArgumentParser,
) -> tuple[dict[str, object], Path] | None:
    """Return the nearest ``[tool.<tool>]`` table and its file, or None."""
    for directory in (start, *start.parents):
        path = directory / "pyproject.toml"
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                document = tomllib.load(handle)
        except tomllib.TOMLDecodeError as err:
            parser.error(f"{path}: {err}")
        try:
            tools = from_plain(document.get("tool"), dict[str, object], default={})
        except ReadError:
            parser.error(f"{path} [tool]: expected a table")
        if tool not in tools:
            continue
        try:
            return from_plain(tools[tool], dict[str, object]), path
        except ReadError:
            parser.error(f"{path} [tool.{tool}]: expected a table")
    return None
