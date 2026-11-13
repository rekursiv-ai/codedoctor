"""Prove selective policies filter native candidates before IDs and execution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import ast
import importlib
import os
import runpy
import sys
import tokenize

from mutmut import (
    __main__ as mutmut_main,
    configuration,
)
from mutmut.mutation import file_mutation
from mutmut.mutation.data import SourceFileMutationData
from mutmut.mutation.diff_apply import read_mutant_function
from mutmut.mutation.pragma_handling import get_ignored_lines
from mutmut.state import state

import libcst as cst
import pytest

from codedoctor.mutation import mutmut_adapter


if TYPE_CHECKING:
    from collections.abc import Iterator

    from mutmut.mutation.file_mutation import MutatedFile


@pytest.fixture(autouse=True)
def adapter_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate native configuration, patches, generated files, and caches."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["mod.py"]\n',
    )
    monkeypatch.chdir(tmp_path)
    configuration.reset_config()
    monkeypatch.setattr(
        file_mutation,
        "create_mutations",
        file_mutation.create_mutations,
    )
    monkeypatch.setattr(
        file_mutation,
        "compute_function_hashes",
        file_mutation.compute_function_hashes,
    )
    monkeypatch.setattr(
        mutmut_main,
        "create_mutants_for_file",
        mutmut_main.create_mutants_for_file,
    )
    monkeypatch.setattr(
        file_mutation,
        "_compute_mutated_function_hashes",
        file_mutation._compute_mutated_function_hashes,
    )
    monkeypatch.setattr(
        mutmut_main,
        "setup_source_paths",
        mutmut_adapter._NATIVE_SETUP_PATHS,
    )
    mutmut_adapter.activate()
    yield
    configuration.reset_config()


def _generate(source: str, *, covered_lines: set[int] | None = None) -> MutatedFile:
    return file_mutation.mutate_file_contents(
        "mod.py",
        source,
        covered_lines=covered_lines,
    )


def _variants(generated: MutatedFile) -> list[str]:
    module = cst.parse_module(generated.code)
    return [
        cst.Module([read_mutant_function(module, name)]).code
        for name in generated.mutant_names
    ]


@pytest.mark.parametrize("prefix", ["", "# noqa: X -- note "])
def test_case_candidates_are_removed_before_ids_and_rendering(prefix: str) -> None:
    source = (
        'def f():\n    return read("Cookie", 1)  '
        f"{prefix}# pragma: no mutate[case-insensitive]\n"
    )
    native = _generate(source.replace("[case-insensitive]", ""))
    assert not native.mutant_names
    generated = _generate(source)
    variants = _variants(generated)
    assert variants
    assert any('"XXCookieXX"' in variant for variant in variants)
    assert any("read(None, 1)" in variant for variant in variants)
    assert any("read(1)" in variant for variant in variants)
    assert any('"Cookie", 2' in variant for variant in variants)
    assert all(
        '"cookie"' not in variant and '"COOKIE"' not in variant for variant in variants
    )
    assert list(generated.mutant_names) == [
        f"x_f__mutmut_{index}" for index in range(1, len(variants) + 1)
    ]
    assert '"cookie"' not in generated.code


@pytest.mark.parametrize(
    "source",
    [
        'def f():\n    raise ValueError("Bad input")  # pragma: no mutate[diagnostic-text]\n',
        'def f():  # pragma: no mutate[diagnostic-text] block\n    raise ValueError("Bad input")\n',
        'def f():\n    # pragma: no mutate[diagnostic-text] block\n    raise ValueError("Bad input")\n',
        '# pragma: no mutate[diagnostic-text] start\ndef f():\n    raise ValueError("Bad input")\n# pragma: no mutate end\n',
        'class C:  # pragma: no mutate[diagnostic-text] block\n    @staticmethod\n    def f():\n        raise ValueError("Bad input")\n',
        'def f():\n    raise ValueError(\n        "Bad input",  # pragma: no mutate[diagnostic-text]\n    )\n',
    ],
)
def test_native_scopes_remove_only_matching_strings(source: str) -> None:
    generated = _generate(source)
    variants = _variants(generated)
    assert variants
    assert all("XXBad inputXX" not in variant for variant in variants)
    assert all('"bad input"' not in variant for variant in variants)
    assert any(
        "ValueError(None)" in variant or "        None," in variant
        for variant in variants
    )


@pytest.mark.parametrize(
    "labels",
    [
        "diagnostic-text, case-insensitive",
        "case-insensitive, diagnostic-text",
        "diagnostic-text, case-insensitive, diagnostic-text",
    ],
)
def test_labels_are_match_any_sets_and_count_every_matching_label(labels: str) -> None:
    generated = _generate(
        f'def f():\n    return read("Cookie")  # pragma: no mutate[{labels}]\n',
    )
    assert _variants(generated) == [
        "def f():\n    return read(None)  # pragma: no mutate[" + labels + "]\n",
    ]


@pytest.mark.parametrize(
    "marker",
    [
        "[]",
        "[unknown]",
        "[case-insensitive,]",
        "[case-insensitive,,diagnostic-text]",
        "[case-insensitive",
        "[case-insensitive]]",
        "[case-insensitive] bogus",
        "[case-insensitive] start",
        "[case-insensitive] end",
        " [case-insensitive]",
    ],
)
@pytest.mark.parametrize("prefix", ["", "# noqa: X -- note "])
def test_invalid_selective_syntax_fails_closed(marker: str, prefix: str) -> None:
    with pytest.raises(ValueError, match=r"mod\.py:2"):
        _generate(
            f'def f():\n    return "Cookie"  {prefix}# pragma: no mutate{marker}\n',
        )


@pytest.mark.parametrize(
    "label",
    ["ignored-argument", "temporary-name", "equal-length-zip", "buffer-size"],
)
def test_deferred_labels_are_unknown(label: str) -> None:
    with pytest.raises(ValueError, match="unknown or empty mutation label"):
        _generate(f"def f():\n    return 1  # pragma: no mutate[{label}]\n")


@pytest.mark.parametrize(
    "label",
    [
        "diagnostic-text",
        "case-insensitive",
        "truthiness",
    ],
)
def test_unused_labels_are_checked_against_generated_candidates(label: str) -> None:
    with pytest.raises(ValueError, match=rf"mod.py:2: unused.*{label}"):
        _generate(f"def f():\n    return 1  # pragma: no mutate[{label}]\n")


def test_one_matching_label_does_not_hide_an_unused_label() -> None:
    with pytest.raises(ValueError, match=r"unused.*case-insensitive"):
        _generate(
            'def f():\n    return "123"  # pragma: no mutate[diagnostic-text, case-insensitive]\n',
        )


@pytest.mark.parametrize(
    "source",
    [
        "@cache\ndef f():\n    return 1  # pragma: no mutate[case-insensitive]\n",
        "def f():  # pragma: no mutate block\n    return 1  # pragma: no mutate[case-insensitive]\n",
    ],
)
def test_unexamined_native_scopes_defer_unused_labels(source: str) -> None:
    assert not _generate(source).mutant_names


def test_uncovered_scopes_defer_unused_labels() -> None:
    assert not _generate(
        "def f():\n    return 1  # pragma: no mutate[case-insensitive]\n",
        covered_lines=set(),
    ).mutant_names


@pytest.mark.parametrize(
    ("literal", "forbidden", "retained"),
    [
        ('"Bad {name}"', '"XXBad {name}XX"', '"BAD {NAME}"'),
        ('"Bad %s"', '"XXBad %sXX"', '"BAD %S"'),
        ('"Bad {value:.2f}"', '"XXBad {value:.2f}XX"', '"BAD {VALUE:.2F}"'),
    ],
)
def test_diagnostic_fields_remain_protected(
    literal: str,
    forbidden: str,
    retained: str,
) -> None:
    variants = _variants(
        _generate(
            f"def f():\n    return warn({literal})  # pragma: no mutate[diagnostic-text]\n",
        ),
    )
    assert all(forbidden not in variant for variant in variants)
    assert any(retained in variant for variant in variants)


def test_same_line_multiple_literals_keep_the_existing_predicate() -> None:
    with pytest.raises(ValueError, match=r"unused.*diagnostic-text"):
        _generate(
            'def f():\n    return warn("Bad", "Other")  # pragma: no mutate[diagnostic-text]\n',
        )


def test_policy_is_per_source_without_cross_visitor_leakage() -> None:
    selective = 'def f():\n    return "Cookie"  # pragma: no mutate[case-insensitive]\n'
    plain = selective.partition("  #")[0] + "\n"
    assert len(_generate(plain).mutant_names) == 3
    assert len(_generate(selective).mutant_names) == 1
    assert len(_generate(plain).mutant_names) == 3


def test_effective_policy_changes_native_hashes_including_inherited_scopes() -> None:
    plain = 'class C:\n    def f(self):\n        return "Cookie"\n'
    case = plain.replace(
        "class C:",
        "class C:  # pragma: no mutate[case-insensitive] block",
    )
    diagnostic = case.replace("case-insensitive", "diagnostic-text")
    assert (
        _generate(plain).hash_by_function_name != _generate(case).hash_by_function_name
    )
    assert file_mutation.compute_function_hashes(
        case,
    ) != file_mutation.compute_function_hashes(diagnostic)
    assert ast.dump(ast.parse(plain)) == ast.dump(ast.parse(case))


def test_set_order_duplicates_and_prose_do_not_change_native_hashes() -> None:
    source = 'def f():\n    return read("Cookie")  # pragma: no mutate[diagnostic-text, case-insensitive]\n'
    variant = source.replace(
        "diagnostic-text, case-insensitive]",
        "case-insensitive, diagnostic-text, case-insensitive] -- Explanation.",
    )
    assert (
        _generate(source).hash_by_function_name
        == _generate(variant).hash_by_function_name
    )


@pytest.mark.parametrize(
    "explanation",
    [
        " -- example # pragma: no mutate",
        " -- example # pragma: no mutate[unknown]",
    ],
)
def test_explanatory_markers_do_not_change_candidates_or_hashes(
    explanation: str,
) -> None:
    source = 'def f():\n    return read("Cookie", 1)  # pragma: no mutate[case-insensitive]\n'
    expected = _generate(source)
    explained = _generate(source.rstrip("\n") + explanation + "\n")
    assert explained.mutant_names == expected.mutant_names
    assert [variant.replace(explanation, "") for variant in _variants(explained)] == (
        _variants(expected)
    )
    assert explained.hash_by_function_name == expected.hash_by_function_name


def test_adapter_activation_is_idempotent() -> None:
    before = file_mutation.create_mutations
    mutmut_adapter.activate()
    assert file_mutation.create_mutations is before


@pytest.mark.parametrize("change", ["annotation", "implementation", "previously-empty"])
def test_policy_cache_invalidates_before_native_mtime_reuse(
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("mod.py")
    output = Path("mutants/mod.py")
    output.parent.mkdir()
    source.write_text(
        'def f():\n    return read("Cookie")  # pragma: no mutate[case-insensitive]\n',
    )
    if change == "previously-empty":
        source.write_text(
            'def f():\n    return "Cookie"  # pragma: no mutate[diagnostic-text]\n',
        )
    first = mutmut_main.create_mutants_for_file(source, output)
    assert first.error is None
    data = SourceFileMutationData(path=source)
    data.load()
    data.exit_code_by_key = dict.fromkeys(data.exit_code_by_key, 0)
    data.save()
    if change == "annotation":
        source.write_text(
            source.read_text().replace("case-insensitive", "diagnostic-text"),
        )
    elif change == "previously-empty":
        source.write_text(
            source.read_text().replace("diagnostic-text", "case-insensitive"),
        )
    else:
        monkeypatch.setattr(
            mutmut_adapter,
            "_implementation_hash",
            lambda: "changed-rule",
        )
    os.utime(source, (100, 100))
    os.utime(output, (200, 200))
    result = mutmut_main.create_mutants_for_file(source, output)
    assert not result.unmodified
    data.load()
    assert data.exit_code_by_key
    assert set(data.exit_code_by_key.values()) == {None}
    assert '"cookie"' not in output.read_text()


def test_unchanged_policy_reuses_native_results_and_mtime() -> None:
    source = Path("mod.py")
    output = Path("mutants/mod.py")
    output.parent.mkdir()
    source.write_text(
        'def f():\n    return read("Cookie")  # pragma: no mutate[case-insensitive]\n',
    )
    mutmut_main.create_mutants_for_file(source, output)
    data = SourceFileMutationData(path=source)
    data.load()
    data.exit_code_by_key = dict.fromkeys(data.exit_code_by_key, 1)
    data.save()
    os.utime(source, (100, 100))
    os.utime(output, (200, 200))
    result = mutmut_main.create_mutants_for_file(source, output)
    assert result.unmodified
    data.load()
    assert set(data.exit_code_by_key.values()) == {1}
    assert output.stat().st_mtime == 200


@pytest.mark.parametrize(
    "source",
    [
        '# pragma: no mutate[diagnostic-text] start\ndef f():\n    return "Bad"\n',
        '# pragma: no mutate end\ndef f():\n    return "Bad"\n',
        '# pragma: no mutate[diagnostic-text] start\n# pragma: no mutate[case-insensitive] start\ndef f():\n    return "Bad"\n# pragma: no mutate end\n',
    ],
)
def test_native_region_errors_fail_closed(source: str) -> None:
    with pytest.raises(ValueError, match=r"mod\.py"):
        _generate(source)


def test_sequential_regions_ignore_intervening_standalone_line_pragmas() -> None:
    generated = _generate(
        "# pragma: no mutate[diagnostic-text] start\n"
        'def first():\n    return read("Bad")\n'
        "# pragma: no mutate\n"
        "# pragma: no mutate end\n"
        "# pragma: no mutate[case-insensitive] start\n"
        'def second():\n    return "Cookie"\n'
        "# pragma: no mutate end\n",
    )
    variants = _variants(generated)
    assert any("read(None)" in variant for variant in variants)
    assert any('"XXCookieXX"' in variant for variant in variants)
    assert all(
        "XXBadXX" not in variant
        and '"cookie"' not in variant
        and '"COOKIE"' not in variant
        for variant in variants
    )


def test_distinct_scopes_do_not_consume_each_others_labels() -> None:
    variants = _variants(
        _generate(
            'def f():\n    first = read("Cookie")  # pragma: no mutate[case-insensitive]\n    return warn("Bad")  # pragma: no mutate[diagnostic-text]\n',
        ),
    )
    assert any("XXCookieXX" in variant for variant in variants)
    assert all(
        '"cookie"' not in variant and "XXBadXX" not in variant for variant in variants
    )


@pytest.mark.parametrize(
    ("original", "mutant"),
    [
        ('read("Cookie", 1)\n', 'read("cookie", 2)\n'),
        ('read("Cookie")\n', "read(None)\n"),
        ('read("Cookie")\n', 'read("Cookie")\n'),
        ('read(b"Cookie")\n', 'read(b"cookie")\n'),
    ],
)
def test_predicate_rejects_nonlocal_or_non_string_changes(
    original: str,
    mutant: str,
) -> None:
    tokens = mutmut_adapter._tokens(original)
    assert (
        mutmut_adapter._string_change(
            tokens,
            mutant=mutant,
            literal_counts=Counter(
                token.end[0] for token in tokens if token.type == tokenize.STRING
            ),
        )
        is None
    )


@pytest.mark.parametrize("value", ["Broken {", "Broken %", "Broken %("])
def test_malformed_diagnostic_fields_do_not_match(value: str) -> None:
    assert not mutmut_adapter._matches_label(
        value,
        mutant="XX" + value + "XX",
        label="diagnostic-text",
    )


def test_hashes_preserve_native_empty_and_accept_behavior() -> None:
    assert file_mutation.compute_function_hashes("x = 1\n") == {}
    assert file_mutation.compute_function_hashes("def invalid(") == {}
    assert (
        file_mutation.compute_function_hashes(
            "def f():\n    return 1\n",
            accept=lambda name: name == "x_missing",
        )
        == {}
    )


def test_reimport_preserves_native_generator_references() -> None:
    assert mutmut_adapter.__file__ is not None
    copied = cast(dict[str, object], runpy.run_path(mutmut_adapter.__file__))
    assert copied["_NATIVE_HASHES"] is mutmut_adapter._NATIVE_HASHES
    assert copied["_NATIVE_CREATE_FILE"] is mutmut_adapter._NATIVE_CREATE_FILE
    assert copied["_NATIVE_SETUP_PATHS"] is mutmut_adapter._NATIVE_SETUP_PATHS


def test_bootstrap_imports_generated_module_and_records_mutant_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [
        "bootstrap_probe",
        "bootstrap_probe.tools",
        "bootstrap_probe.tools.adapter",
    ]
    source = "def f():\n    return 1\n"
    for root in [tmp_path, tmp_path / "mutants"]:
        package = root / "bootstrap_probe/tools"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "adapter.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    before = importlib.import_module(names[-1])
    try:
        active_generator = file_mutation.create_mutations
        monkeypatch.setattr(mutmut_adapter, "__name__", names[-1])
        mutmut_adapter._setup_source_paths()
        generated = _generate(source)
        target = tmp_path / "mutants/bootstrap_probe/tools/adapter.py"
        target.write_text(generated.code)
        after = importlib.import_module(names[-1])
        assert after is not before
        assert after.__file__ == str(target)
        assert file_mutation.create_mutations is active_generator
        monkeypatch.setattr(state(), "_stats", set[str]())
        monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
        monkeypatch.setenv("MUTMUT_DEPENDENCY_DEPTH", "-1")
        function = cast(Callable[[], int], after.f)
        assert function() == 1
        assert state()._stats == {"bootstrap_probe.tools.adapter.x_f"}
        monkeypatch.setenv("MUTANT_UNDER_TEST", f"{names[-1]}.x_f__mutmut_1")
        assert function() == 2
    finally:
        for name in reversed(names):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_selective_policy_preserves_native_newline_formats(newline: str) -> None:
    source = 'def f():\n    return "Cookie"  # pragma: no mutate[case-insensitive]\n'.replace(
        "\n",
        newline,
    )
    generated = mutmut_adapter.mutate_file_contents("mod.py", source)
    assert len(generated.mutant_names) == 1
    assert "XXCookieXX" in generated.code
    assert '"cookie"' not in generated.code


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    "literal",
    [
        "x\u2028y\x0cz",
        *[
            f"x{separator}z"
            for separator in [
                "\v",
                "\f",
                "\x1c",
                "\x1d",
                "\x1e",
                "\x85",
                "\u2028",
                "\u2029",
            ]
        ],
    ],
)
def test_selective_policy_counts_only_python_physical_lines(
    literal: str,
    newline: str,
) -> None:
    source = f'S = "{literal}"\ndef f():\n    return "Cookie"  # pragma: no mutate[case-insensitive]\n'.replace(
        "\n",
        newline,
    )
    generated = mutmut_adapter.mutate_file_contents("mod.py", source)
    assert list(generated.mutant_names) == ["x_f__mutmut_1"]
    assert '"XXCookieXX"' in generated.code
    assert '"cookie"' not in generated.code
    assert '"COOKIE"' not in generated.code


@pytest.mark.parametrize(
    ("original", "mutant"),
    [
        ("False", "None"),
        ("0.0", "[]"),
        ("{}", "''"),
        ("()", "b''"),
        ("-0", "+0.0"),
        ("call(False)", "call(None)"),
        ("call(flag=False)", "call(flag=None)"),
    ],
)
def test_truthiness_accepts_only_falsy_literal_exchanges(
    original: str,
    mutant: str,
) -> None:
    source = "def f():\n    return "
    assert mutmut_adapter._structural_matches(
        ast.parse(source + original),
        mutant=ast.parse(source + mutant),
    ) == {"truthiness": {2}}


@pytest.mark.parametrize(
    ("original", "mutant"),
    [
        ("False", "True"),
        ("1", "2"),
        ("call()", "None"),
        ("[call()]", "[]"),
        ("value", "None"),
        ("value.attr", "None"),
        ("value[0]", "None"),
        ("False", "False"),
        ("{[1]}", "None"),
        ("{[]: 1}", "None"),
        ("call(False, 0)", "call(None, None)"),
        ("call(False)", "other(None)"),
        ("call(flag=False)", "call(other=None)"),
        ("call(False, value)", "call(value)"),
    ],
)
def test_truthiness_retains_nonliteral_truthy_and_nonlocal_changes(
    original: str,
    mutant: str,
) -> None:
    source = "def f(value):\n    return "
    assert (
        mutmut_adapter._structural_matches(
            ast.parse(source + original),
            mutant=ast.parse(source + mutant),
        )
        == {}
    )


def test_truthiness_does_not_match_an_unchanged_expression() -> None:
    expression = ast.parse("False", mode="eval").body
    assert mutmut_adapter._structural_matches(expression, mutant=expression) == {}


def test_truthiness_retains_optional_expression_removal() -> None:
    assert (
        mutmut_adapter._structural_matches(
            ast.parse("def f() -> bool:\n    return False\n"),
            mutant=ast.parse("def f():\n    return False\n"),
        )
        == {}
    )


def test_truthiness_filters_native_candidates_before_rendering() -> None:
    variants = _variants(
        _generate(
            "def f():\n    return call(False, 2)  # pragma: no mutate[truthiness]\n",
        ),
    )
    assert all("call(None, 2)" not in variant for variant in variants)
    assert any("call(True, 2)" in variant for variant in variants)
    assert any("call(False, 3)" in variant for variant in variants)


def test_nested_closing_marker_does_not_broaden_truthiness_scope() -> None:
    source = (
        "def f():\n"
        "    return (\n"
        "        outer(\n"
        "            inner(False),\n"
        "        )  # pragma: no mutate[truthiness]\n"
        "    )\n"
    )
    with pytest.raises(ValueError, match=r"unused.*truthiness"):
        _generate(source)


@pytest.mark.parametrize("argument", ["False", "flag=False"])
def test_call_closing_marker_keeps_direct_truthiness_scope(argument: str) -> None:
    source = (
        f"def f():\n    return call(\n        {argument},\n    )"
        "  # pragma: no mutate[truthiness]\n"
    )
    native = _variants(_generate(source.partition("  # pragma:")[0] + "\n"))
    variants = _variants(_generate(source))
    assert any("None," in variant for variant in native)
    assert all("None," not in variant for variant in variants)
    assert any("True," in variant for variant in variants)


def test_missing_candidate_metadata_never_filters() -> None:
    source = "def f():\n    return call(False)  # pragma: no mutate[truthiness]\n"
    wrapper = cst.MetadataWrapper(cst.parse_module(source))
    policy = mutmut_adapter._Policy.parse("mod.py", source=source)
    visitor = mutmut_adapter._SelectiveVisitor(
        policy=policy,
        module=wrapper.module,
        ignored=get_ignored_lines("mod.py", policy.blanket_source, wrapper),
        covered_lines=None,
    )
    with visitor.resolve(wrapper):
        assert not visitor._matches(
            file_mutation.Mutation(
                original_node=cst.Name("False"),
                mutated_node=cst.Name("None"),
                contained_by_top_level_function=None,
            ),
        )


@pytest.mark.parametrize("scope_line", [2, 4])
def test_outside_candidate_span_does_not_render(
    scope_line: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = ["def f():", '    first = "Bad"', "    number = 0", '    last = "Bad"']
    lines[scope_line - 1] += "  # pragma: no mutate[diagnostic-text]"
    source = "\n".join(lines) + "\n"
    wrapper = cst.MetadataWrapper(cst.parse_module(source))
    policy = mutmut_adapter._Policy.parse("mod.py", source=source)
    visitor = mutmut_adapter._SelectiveVisitor(
        policy=policy,
        module=wrapper.module,
        ignored=get_ignored_lines("mod.py", policy.blanket_source, wrapper),
        covered_lines=None,
    )
    function = wrapper.module.body[0]
    assert isinstance(function, cst.FunctionDef)
    assert isinstance(function.body, cst.IndentedBlock)
    statement = function.body.body[1]
    assert isinstance(statement, cst.SimpleStatementLine)
    assignment = statement.body[0]
    assert isinstance(assignment, cst.Assign)

    def reject_render(*nodes: object) -> None:
        pytest.fail(f"Rendered outside candidate span: {nodes}")

    monkeypatch.setattr(file_mutation, "deep_replace", reject_render)
    with visitor.resolve(wrapper):
        assert not visitor._matches(
            file_mutation.Mutation(
                original_node=assignment,
                mutated_node=assignment.with_changes(value=cst.Name("None")),
                contained_by_top_level_function=function,
            ),
        )


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
