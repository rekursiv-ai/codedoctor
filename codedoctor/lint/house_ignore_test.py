"""Tests for the ``house-ignore`` marker shared by houselint and the hooks."""

from __future__ import annotations

from pathlib import Path

import ast

import pytest

from codedoctor.lint import house_ignore
from codedoctor.lint.houselint_types import Finding, Unit


def test_empty_match_header_has_a_single_line_range() -> None:
    """An empty match statement still has a meaningful header range."""
    node = ast.Match(subject=ast.Name(id="x"), cases=[])
    ast.fix_missing_locations(node)
    assert list(house_ignore.header_lines(node)) == [1]


def test_header_lines_of_compound_and_simple_statements() -> None:
    """A header stops before its body; a match before its first case."""
    module = ast.parse(
        "if (\n"
        "    a\n"
        "):\n"
        "    b = 1\n"
        "match (\n"
        "    x\n"
        "):\n"
        "    case 1:\n"
        "        pass\n"
        "y = (\n"
        "    1\n"
        ")\n",
    )
    if_node, match_node, assign = module.body

    assert list(house_ignore.header_lines(if_node)) == [1, 2, 3]
    assert list(house_ignore.header_lines(match_node)) == [5, 6, 7]
    assert list(house_ignore.header_lines(assign)) == [10, 11, 12]


def test_a_finding_outside_every_statement_is_its_own_line() -> None:
    """A comment line belongs to no statement; only a marker on it covers it."""
    unit = _unit("x = 1\n# BAD\ny = 2\n")

    assert house_ignore.statement_lines(unit, 2) == range(2, 3)


def test_the_innermost_header_wins_and_a_line_past_the_end_is_alone() -> None:
    """A wrapped call inside a def is its own statement, not the def's header."""
    unit = _unit(
        "def f(\n    a,\n):\n    return g(\n        a,\n    )\n\nz = 1\n",
    )

    assert house_ignore.statement_lines(unit, 2) == range(1, 4)
    assert house_ignore.statement_lines(unit, 5) == range(4, 7)
    assert house_ignore.statement_lines(unit, 6) == range(4, 7)
    assert house_ignore.statement_lines(unit, 7) == range(7, 8)


def test_a_case_line_belongs_to_no_statement_header() -> None:
    """``case`` is not a statement, and the match header ends above it."""
    unit = _unit("match x:\n    case 1:\n        pass\n")

    assert house_ignore.statement_lines(unit, 2) == range(2, 3)


def test_a_statement_sharing_its_headers_first_line_is_innermost() -> None:
    """``if x: y = (`` -- the wrapped body statement starts on the header line."""
    unit = _unit("z = 0\nif x: y = (\n    1\n)\n")

    assert house_ignore.statement_lines(unit, 2) == range(2, 5)


def test_a_malformed_marker_names_the_unit_and_says_what_is_missing() -> None:
    unit = _unit("x = 1  # house-ignore[]\n")

    assert house_ignore.marker_findings(unit) == [
        Finding(
            path=unit.path,
            line=1,
            rule="houselint",
            message="house-ignore[...] names no rule; a bare marker suppresses nothing",
        ),
        Finding(
            path=unit.path,
            line=1,
            rule="houselint",
            message="house-ignore[...] needs a trailing `-- Reason.`",
        ),
    ]


def test_a_marker_lists_its_rules_by_comma() -> None:
    assert house_ignore.rule_names(" a, b ,,c ") == {"a", "b", "c"}


def test_a_notebook_has_no_markers(tmp_path: Path) -> None:
    """A notebook carries no tokens, so a marker in its JSON is just text."""
    unit = Unit(
        path=tmp_path / "n.ipynb",
        source='{"cells": [{"source": ["# house-ignore[x] -- Reason.\\n"]}]}',
        kind="jupyter",
    )

    assert list(house_ignore.markers(unit)) == []


def _finding(line: int, rule: str = "r") -> Finding:
    return Finding(path=Path("mod.py"), line=line, rule=rule, message="m")


@pytest.mark.parametrize(
    ("source", "found", "kept"),
    [
        # A trailing marker covers its own statement, not its neighbours.
        ("x = 0\na = 1  # house-ignore[r] -- Why.\nb = 2\n", [2, 3], [3]),
        ("a = 1  # house-ignore[r] -- Why.\nb = 2\n", [1], []),
        # ...on any line of a wrapped statement.
        ("a = (\n    1\n)  # house-ignore[r] -- Why.\n", [2], []),
        # An own-line marker covers the whole file.
        ("# house-ignore[r] -- Why.\na = 1\nb = 2\n", [2, 3], []),
        # A finding on line 1 is a file verdict: any trailing marker covers it.
        ("a = 1\nb = 2  # house-ignore[r] -- Why.\n", [1, 2], []),
    ],
    ids=["neighbour", "own", "wrapped", "file", "line-one"],
)
def test_apply_drops_exactly_the_covered_findings(
    source: str,
    found: list[int],
    kept: list[int],
) -> None:
    result = house_ignore.apply(
        _unit(source),
        [_finding(line) for line in found],
        active=frozenset({"r"}),
    )
    assert result == [_finding(line) for line in kept]


def test_a_marker_for_another_rule_suppresses_nothing() -> None:
    unit = _unit("a = 1  # house-ignore[other] -- Why.\n")
    assert house_ignore.apply(unit, [_finding(1)], active=frozenset({"r"})) == [
        _finding(1),
    ]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "a = 1  # house-ignore[r] -- Why.\n",
            (
                "superfluous inline ignore: 'r' reports nothing on this statement; "
                "remove it from house-ignore[...]"
            ),
        ),
        (
            "# house-ignore[r] -- Why.\na = 1\n",
            (
                "superfluous file-level ignore: 'r' reports nothing here; "
                "remove it from the own-line house-ignore[...]"
            ),
        ),
    ],
    ids=["inline", "file"],
)
def test_a_marker_suppressing_nothing_is_reported_once_its_rule_ran(
    source: str,
    message: str,
) -> None:
    unit = _unit(source)
    assert house_ignore.apply(unit, [], active=frozenset({"r"})) == [
        Finding(path=Path("mod.py"), line=1, rule="houselint", message=message),
    ]
    assert house_ignore.apply(unit, [], active=frozenset({"other"})) == []


def _unit(source: str, name: str = "mod.py") -> Unit:
    """Build a Python Unit over ``source``."""
    return Unit(path=Path(name), source=source, kind="python")


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
