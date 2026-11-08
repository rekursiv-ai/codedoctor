"""Tests for ``custom_types``: the shared walk on ``Unit``."""

from __future__ import annotations

from pathlib import Path

import ast
import warnings

from codedoctor.lint.custom_types import Finding, Unit


def _unit(source: str, kind: str = "python") -> Unit:
    return Unit(path=Path("mod.py"), source=source, kind=kind)


_SOURCE = "import os\n\n\ndef f(x):\n    return g(x)\n\n\nclass C:\n    def m(self):\n        pass\n"


def test_walk_matches_ast_walk_order() -> None:
    unit = _unit(_SOURCE)
    tree = unit.tree
    assert tree is not None
    assert [type(n) for n in unit.walk] == [type(n) for n in ast.walk(tree)]
    assert unit.walk[0] is tree


def test_parents_maps_every_node_but_the_module() -> None:
    """``Load()``/``Store()`` are shared singletons, so they have no one parent."""
    unit = _unit(_SOURCE)
    tree = unit.tree
    assert tree is not None
    assert tree not in unit.parents
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if not isinstance(child, ast.expr_context):
                assert unit.parents[child] is parent


def test_nodes_selects_by_kind_including_subclasses_in_walk_order() -> None:
    unit = _unit(_SOURCE)
    defs = unit.nodes(ast.FunctionDef)
    assert [d.name for d in defs] == ["f", "m"]
    assert [type(s).__name__ for s in unit.nodes(ast.stmt)][:3] == [
        "Import",
        "FunctionDef",
        "ClassDef",
    ]
    assert unit.nodes(ast.FunctionDef) is defs
    assert unit.nodes(ast.Lambda) == ()
    mixed = unit.nodes(ast.ClassDef, ast.FunctionDef)
    assert [type(n).__name__ for n in mixed] == [
        "FunctionDef",
        "ClassDef",
        "FunctionDef",
    ]
    assert unit.nodes(ast.ClassDef, ast.FunctionDef) is mixed


def test_unparsable_and_notebook_units_have_empty_views() -> None:
    for unit in (_unit("def (\n"), _unit("{}", kind="jupyter")):
        assert unit.walk == ()
        assert dict(unit.parents) == {}
        assert unit.nodes(ast.AST) == ()


def test_a_parse_warning_names_the_units_file() -> None:
    unit = Unit(path=Path("pkg/mod.py"), source='x = "\\d"\n', kind="python")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert unit.tree is not None

    assert [warning.filename for warning in caught] == ["pkg/mod.py"]


def test_tokens_stop_quietly_where_tokenizing_fails() -> None:
    tokens = _unit("x = 1\ny = (\n").tokens

    assert [token.string for token in tokens][:3] == ["x", "=", "1"]
    assert all(token.start[0] <= 2 for token in tokens)


def test_lines_split_the_source_once() -> None:
    unit = _unit("a = 1\n# noqa\n")

    assert unit.lines == ("a = 1", "# noqa")
    assert unit.lines is unit.lines


def test_a_finding_renders_as_path_line_rule_message() -> None:
    finding = Finding(path=Path("pkg/m.py"), line=3, rule="no-any", message="Use T.")

    assert str(finding) == "pkg/m.py:3: [no-any] Use T."


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
