"""Tests for ``houselint``: rule protocol, per-rule flags, and one parse."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing.context import ForkContext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, cast, override

import argparse
import ast
import json
import re
import subprocess
import sys

import pytest

from codedoctor.lib.worker_count import default_workers
from codedoctor.lint import (
    house_ignore,
    houselint,
    houselint_rules,
    houselint_types,
)
from codedoctor.lint.houselint import Exclusion, Rule, run
from codedoctor.lint.houselint_rules import (
    ConfigGlobalsRule,
    DataclassRule,
    DigitsRule,
    StaticMethodRule,
)
from codedoctor.lint.houselint_types import Finding, Unit


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from multiprocessing.context import BaseContext

    import tokenize


# ``main`` reads its defaults from the nearest configured ``pyproject.toml`` above the
# cwd, so from the repo root every test would inherit the repo's ``fix = "ALL"`` and
# exclusions.
@pytest.fixture(autouse=True)
def _outside_any_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run from a directory no ``[tool.houselint]`` governs."""
    monkeypatch.chdir(tmp_path)


def _write(tmp_path: Path, body: str, name: str = "mod.py") -> Path:
    """Write ``body`` to ``name`` in ``tmp_path``."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _exclusion(spec: str) -> Exclusion:
    """Parse one ``--exclude`` value against the shipped registry."""
    return houselint.parse_exclusion(
        spec,
        known={rule.name for rule in houselint.default_rules()} | {"counting"},
    )


# Goes through ``_add_arguments`` rather than constructing a Namespace so the flag
# SPELLING is under test too: a renamed flag breaks this, which is the point.
def _configured[R: Rule](rule: R, /, *argv: str) -> tuple[R, houselint._Flags]:
    """Parse ``argv`` through the real parser and apply them to ``rule``."""
    parser = argparse.ArgumentParser()
    houselint._add_arguments(parser)
    flags = cast(
        houselint._Flags,
        parser.parse_args(["--select", rule.name, *argv, "unused.py"]),
    )
    active, _ = houselint.select_rules(
        (rule,),
        select=flags.select,
        fix=flags.fix,
        settings=flags.settings,
    )
    return cast(R, active[0]), flags


@dataclass(frozen=True, kw_only=True, slots=True)
class _Counting:
    """A rule that records which units it saw and flags a marker line."""

    name: ClassVar[str] = "counting"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    marker: str = "BAD"
    """Substring whose presence in a line is a violation."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Flag every line containing ``marker``."""
        return [
            Finding(
                path=unit.path,
                line=n,
                rule=self.name,
                message=f"saw {self.marker}",
            )
            for n, text in enumerate(unit.source.splitlines(), start=1)
            if self.marker in text
        ]

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix."""
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class _CountingAnyKind:
    """Flags the marker in ``source``, for a file of EITHER kind.

    Reads ``source`` rather than ``lines`` because a notebook carries no
    line view. Exists so a test can show that a notebook body the
    python-only rule called clean is one this rule DOES report -- proving
    the kind guard, and not an empty fixture, is what silenced it.
    """

    name: ClassVar[str] = "counting-any"
    kinds: ClassVar[frozenset[str]] = frozenset({"python", "jupyter"})
    fixable: ClassVar[bool] = False

    marker: str = "BAD"
    """Substring whose presence in a line is a violation."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Flag every line of ``source`` containing ``marker``."""
        return [
            Finding(
                path=unit.path,
                line=n,
                rule=self.name,
                message=f"saw {self.marker}",
            )
            for n, text in enumerate(unit.source.splitlines(), start=1)
            if self.marker in text
        ]

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix."""
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class _Trim:
    """A fixable rule that strips one trailing ``z`` per pass."""

    name: ClassVar[str] = "trim"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report a trailing ``z``, which is exactly what ``fix`` strips."""
        if not unit.source.rstrip().endswith("z"):
            return ()
        return (Finding(path=unit.path, line=1, rule=self.name, message="z"),)

    def fix(self, unit: Unit, /) -> str | None:
        """Strip one trailing ``z``, or None when there is none."""
        if not unit.source.rstrip().endswith("z"):
            return None
        return unit.source.rstrip()[:-1] + "\n"


@dataclass(frozen=True, kw_only=True, slots=True)
class _Breaking:
    """A fixable rule whose fix leaves the file unparseable."""

    name: ClassVar[str] = "breaking"
    kinds: ClassVar[frozenset[str]] = frozenset({"python", "pyi"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report the first line whenever the file has one."""
        if not unit.source:
            return ()
        return (Finding(path=unit.path, line=1, rule=self.name, message="b"),)

    def fix(self, unit: Unit, /) -> str | None:
        """Append an unterminated bracket."""
        return unit.source + "x = (\n"


@pytest.mark.parametrize("name", ["m.py", "m.pyi"])
def test_a_fix_that_breaks_the_parse_raises(tmp_path: Path, name: str) -> None:
    """A fixer's output reached disk unchecked and left stubs unparseable.

    A broken fix is a houselint bug, not a finding in the user's file, so the
    run stops with the rule, the file, and the parse error that proves it.
    """
    path = _write(tmp_path, "x = 1\n", name=name)

    with pytest.raises(houselint.FixError) as raised:
        run([path], active=(_Breaking(),), fixing=frozenset({"breaking"}))

    assert path.read_text() == "x = 1\n"
    message = str(raised.value)
    assert "breaking" in message
    assert str(path) in message
    assert isinstance(raised.value.__cause__, SyntaxError)


def test_a_broken_fix_exits_nonzero_with_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main`` lets the error escape: a crash must not read as clean or as findings."""
    path = _write(tmp_path, "x = 1\n")
    monkeypatch.setattr(houselint, "default_rules", lambda: (_Breaking(),))
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "breaking", "--fix", "breaking", str(path)],
    )

    with pytest.raises(houselint.FixError):
        houselint.main()


def test_a_file_that_never_parsed_may_still_be_fixed(tmp_path: Path) -> None:
    """The gate rejects a regression, not a file that was already broken."""
    path = _write(tmp_path, "def (:\nz\n")

    _, written = run([path], active=(_Trim(),), fixing=frozenset({"trim"}))

    assert written == (path,)


def test_shipped_kinds_are_python_pyi_and_jupyter(tmp_path: Path) -> None:
    """The kinds are pre-commit's ``identify`` tags, as the hook's ``types_or``."""
    kinds = {
        name: houselint.read_unit(_write(tmp_path, "x = 1\n", name=name))
        for name in ("m.py", "m.pyi")
    }
    assert {name: getattr(unit, "kind", None) for name, unit in kinds.items()} == {
        "m.py": "python",
        "m.pyi": "pyi",
    }
    notebook = houselint.read_unit(_write(tmp_path, "{}", name="n.ipynb"))
    assert isinstance(notebook, Unit)
    assert notebook.kind == "jupyter"


def test_a_stub_still_gets_its_suppressions_checked(tmp_path: Path) -> None:
    stub = _write(tmp_path, "x: int  # type: ignore\n", name="m.pyi")

    findings, _ = run(
        [stub],
        active=(houselint_rules.SuppressionFormRule(),),
        fixing=frozenset(),
    )

    assert [(f.rule, f.line) for f in findings] == [("suppression-form", 1)]


def test_only_rules_safe_on_stubs_declare_pyi() -> None:
    """A stub mirrors another package's signatures; most fixers would rewrite them."""
    assert {rule.name for rule in houselint.default_rules() if "pyi" in rule.kinds} == {
        "digits",
        "emdash",
        "suppression-form",
    }


def test_every_shipped_rule_satisfies_the_protocol(tmp_path: Path) -> None:
    """A rule missing a member would fail only at dispatch, on a real commit."""
    for rule in houselint.default_rules():
        assert rule.kinds <= {"python", "pyi", "jupyter"}
        assert isinstance(rule.name, str)
        assert rule.name
        assert isinstance(rule.fixable, bool)

        # A rule whose ``check`` took the wrong arguments, or whose ``kinds``
        # was a plain str, fails at dispatch on a real commit. So call both
        # members and check what comes back.
        unit = Unit(path=tmp_path / "empty.py", source="", kind="python")
        findings = rule.check(unit)
        assert not isinstance(findings, str)
        assert all(isinstance(f, Finding) for f in findings)
        assert rule.fix(unit) is None


def test_a_rule_missing_a_member_is_not_a_rule() -> None:
    """The protocol has to REJECT something, or it asserts nothing."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class _NoFix:
        """Has every member but ``fix``."""

        name: ClassVar[str] = "nofix"
        kinds: ClassVar[frozenset[str]] = frozenset({"python"})
        fixable: ClassVar[bool] = False

        def check(self, unit: Unit, /) -> Sequence[Finding]:
            """Report nothing."""
            del unit
            return ()

    assert not isinstance(_NoFix(), Rule)


def test_shipped_rule_names_are_unique() -> None:
    """Two rules sharing a name would make ``--check-<name>`` ambiguous."""
    names = [rule.name for rule in houselint.default_rules()]
    assert len(names) == len(set(names))


def test_a_unit_parses_once_for_every_rule_that_needs_a_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of merging: N tree-reading rules pay ONE ``ast.parse``.

    Counting parses rather than timing them -- a timing assertion on a 3ms
    operation is a flake, and the claim is about call count, not speed.

    Both rules must actually READ ``tree``: the views are lazy, so a rule
    that only looks at ``source`` correctly causes zero parses and would
    make this assertion measure nothing.
    """
    calls = 0

    def counting_parse(source: str, path: Path) -> ast.Module | None:
        nonlocal calls
        calls += 1
        return ast.parse(source, filename=str(path))

    monkeypatch.setattr(houselint_types, "_parse", counting_parse)
    path = _write(tmp_path, "x = 1\n")
    rules = (DataclassRule(), ConfigGlobalsRule())

    run([path], active=rules, fixing=frozenset())

    assert calls == 1


def test_a_rule_only_sees_files_of_its_declared_kind(tmp_path: Path) -> None:
    """A python-only rule must never be handed a notebook.

    The body must CONTAIN the marker. With ``'{"cells": []}'`` the rule
    found nothing to report either way, so the test passed identically with
    the kind guard deleted from ``run`` -- it was satisfied by the empty
    case and proved nothing.
    """
    notebook = _write(tmp_path, '{"cells": [], "x": "BAD"}', name="nb.ipynb")
    findings, _ = run([notebook], active=(_Counting(),), fixing=frozenset())
    assert findings == ()


def test_the_kind_guard_is_what_makes_that_notebook_clean(tmp_path: Path) -> None:
    """The same body under a jupyter-reading rule DOES report.

    Pins that the previous test's cleanliness comes from the kind guard and
    not from the fixture being empty.
    """
    notebook = _write(tmp_path, '{"cells": [], "x": "BAD"}', name="nb.ipynb")

    findings, _ = run([notebook], active=(_CountingAnyKind(),), fixing=frozenset())

    assert [f.line for f in findings] == [1]


def test_findings_are_sorted_by_path_then_line(tmp_path: Path) -> None:
    """One log, many rules: unsorted output makes a multi-file run unreadable."""
    first = _write(tmp_path, "ok\nBAD\n", name="a.py")
    second = _write(tmp_path, "BAD\n", name="b.py")
    findings, _ = run([second, first], active=(_Counting(),), fixing=frozenset())
    assert [(f.path.name, f.line) for f in findings] == [("a.py", 2), ("b.py", 1)]


def test_a_fix_runs_before_the_check_that_would_report_it(tmp_path: Path) -> None:
    """Otherwise one invocation reports what it just repaired.

    Impossible today only because each hook is a separate process; merging
    them is what creates the ordering requirement.
    """
    path = _write(tmp_path, "value = 12345\n")
    digits = DigitsRule()

    findings, written = run([path], active=(digits,), fixing=frozenset({"digits"}))

    assert path.read_text() == "value = 12_345\n"
    assert written == (path,)
    assert findings == ()


def test_a_marked_finding_does_not_open_the_fix_gate(tmp_path: Path) -> None:
    """A fix over a suppressed finding would orphan the marker that kept it."""
    body = "value = 12345  # house-ignore[digits] -- Kept as written.\n"
    path = _write(tmp_path, body)

    findings, written = run(
        [path],
        active=(DigitsRule(),),
        fixing=frozenset({"digits"}),
    )

    assert path.read_text() == body
    assert written == ()
    assert findings == ()


def test_a_check_without_fix_reports_and_leaves_the_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "value = 12345\n")
    digits = DigitsRule()

    findings, written = run([path], active=(digits,), fixing=frozenset())

    assert path.read_text() == "value = 12345\n"
    assert written == ()
    assert len(findings) == 1
    assert findings[0].rule == "digits"


def test_configure_sets_a_rule_field_from_the_flag_value() -> None:
    rule, flags = _configured(DigitsRule(), "--set", "digits.min-digits=7")
    assert rule.min_digits == 7
    assert flags.fix == ""


def test_a_fix_flag_is_separate_from_the_settings() -> None:
    """``--fix`` names rules; it is never a token inside a settings value."""
    rule, flags = _configured(
        DigitsRule(),
        "--fix",
        "digits",
        "--set",
        "digits.min-digits=7",
    )
    assert rule.min_digits == 7
    assert flags.fix == "digits"


def test_a_setting_is_coerced_to_the_declared_field_type() -> None:
    """Values arrive as CLI text; the field's type decides what they become."""
    rule, _ = _configured(DigitsRule(), "--set", "digits.min-digits=7")
    assert rule.min_digits == 7
    assert not isinstance(rule.min_digits, str)


def test_a_misspelled_setting_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dropped typo silently reverts the knob to its default."""
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "houselint.py",
            "--select",
            "digits",
            "--set",
            "digits.min-digit=4",
            str(path),
        ],
    )

    with pytest.raises(SystemExit):
        houselint.main()

    assert "has no field(s) ['min_digit']" in capsys.readouterr().err


def test_no_rule_selected_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Off by default, so a run naming no rule checked nothing and exited 0.

    A hook entry that lost its ``--check-*`` would pass every commit while
    appearing to run, which is the failure mode silence cannot distinguish
    from success.
    """
    path = _write(
        tmp_path,
        "from dataclasses import dataclass\n\n@dataclass\nclass A:\n    x: int\n",
    )
    monkeypatch.setattr(sys, "argv", ["houselint.py", str(path)])

    with pytest.raises(SystemExit):
        houselint.main()

    assert "--select is required" in capsys.readouterr().err


def test_only_the_named_rule_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A file violating dataclass is clean under a globals-only invocation."""
    path = _write(
        tmp_path,
        "from dataclasses import dataclass\n\n@dataclass\nclass A:\n    x: int\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "globals", str(path)],
    )

    assert houselint.main() == 0
    assert capsys.readouterr().out == ""


def test_a_named_rule_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write(
        tmp_path,
        "from dataclasses import dataclass\n\n@dataclass\nclass A:\n    x: int\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "dataclass", str(path)],
    )

    assert houselint.main() == 1
    assert "[dataclass]" in capsys.readouterr().out


def test_two_rules_can_be_named_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path, "MAX = 500\nvalue = 12345\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "globals digits", str(path)],
    )

    assert houselint.main() == 1


def test_an_excluded_path_is_not_inspected(tmp_path: Path) -> None:
    kept = _write(tmp_path, "BAD\n", name="keep.py")
    skipped = _write(tmp_path, "BAD\n", name="drop_test.py")

    findings, _ = run(
        [kept, skipped],
        active=(_Counting(),),
        fixing=frozenset(),
        exclude=(_exclusion("*_test.py"),),
    )

    assert [f.path.name for f in findings] == ["keep.py"]


def test_an_unscoped_exclude_never_opens_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruff's ``exclude``: a path every rule skips is not even read."""
    path = _write(tmp_path, "BAD\n", name="drop_test.py")

    def refusing_read_unit(path: Path, /, *, reporter: str = "houselint") -> None:
        raise AssertionError(f"opened {path} for {reporter}")

    monkeypatch.setattr(houselint, "read_unit", refusing_read_unit)

    findings, _ = run(
        [path],
        active=(_Counting(),),
        fixing=frozenset(),
        exclude=(_exclusion("*_test.py"),),
    )

    assert findings == ()


def test_an_excluded_path_is_not_fixed_either(tmp_path: Path) -> None:
    """Exclusion gates the fixer too, or ``--exclude`` would still rewrite."""
    path = _write(tmp_path, "value = 12345\n", name="x_test.py")

    _, written = run(
        [path],
        active=(DigitsRule(),),
        fixing=frozenset({"digits"}),
        exclude=(_exclusion("*_test.py"),),
    )

    assert written == ()
    assert path.read_text() == "value = 12345\n"


def test_a_basename_glob_matches_in_any_directory(tmp_path: Path) -> None:
    """``conftest.py`` reads as ``.gitignore`` reads it: every directory."""
    deep = _write(tmp_path / "a" / "b", "BAD\n", name="conftest.py")
    top = _write(tmp_path, "BAD\n", name="conftest.py")
    other = _write(tmp_path / "a", "BAD\n", name="mod.py")

    findings, _ = run(
        [deep, top, other],
        active=(_Counting(),),
        fixing=frozenset(),
        exclude=(_exclusion("conftest.py"),),
        root=tmp_path,
    )

    assert [f.path.name for f in findings] == ["mod.py"]


def test_a_glob_with_a_slash_is_anchored_at_the_root(tmp_path: Path) -> None:
    """``lib/x.py`` names one file; ``other/lib/x.py`` is a different one."""
    named = _write(tmp_path / "lib", "BAD\n", name="x.py")
    nested = _write(tmp_path / "other" / "lib", "BAD\n", name="x.py")

    findings, _ = run(
        [named, nested],
        active=(_Counting(),),
        fixing=frozenset(),
        exclude=(_exclusion("lib/x.py"),),
        root=tmp_path,
    )

    assert [str(f.path.relative_to(tmp_path)) for f in findings] == ["other/lib/x.py"]


def test_a_double_star_crosses_directories(tmp_path: Path) -> None:
    shallow = _write(tmp_path / "vendor", "BAD\n", name="a.py")
    deep = _write(tmp_path / "vendor" / "sub", "BAD\n", name="b.py")
    outside = _write(tmp_path, "BAD\n", name="c.py")

    findings, _ = run(
        [shallow, deep, outside],
        active=(_Counting(),),
        fixing=frozenset(),
        exclude=(_exclusion("vendor/**"),),
        root=tmp_path,
    )

    assert [f.path.name for f in findings] == ["c.py"]


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("*_test.py", "b_test.py", True),
        ("*_test.py", "a/b_test.py", False),
        ("*_test.py", "b_tests.py", False),
        ("**/tests/test*.py", "pkg/tools/tests/test_x.py", True),
        ("**/tests/test*.py", "tests/test_x.py", True),
        ("**/tests/test*.py", "/abs/tests/test_x.py", True),
        ("**/tests/test*.py", "loop/tests/x_test.py", False),
        ("vendor/**", "vendor/a.py", True),
        ("vendor/**", "vendor/s/a.py", True),
        ("vendor/**", "vendors/a.py", False),
        ("lib/x.py", "lib/x.py", True),
        ("lib/x.py", "lib/x.pyi", False),
        ("a?c.py", "abc.py", True),
        ("a?c.py", "a/c.py", False),
    ],
)
def test_glob_regex_has_pathlib_semantics(
    pattern: str,
    path: str,
    expected: bool,
) -> None:
    """``*`` and ``?`` stop at ``/``; only ``**`` crosses a directory."""
    assert (houselint.glob_regex(pattern).match(path) is not None) is expected


@pytest.mark.parametrize(
    "spec",
    ["*_test.py conftest.py = digits", "*_test.py conftest.py=digits"],
)
def test_one_exclude_may_list_several_globs(spec: str) -> None:
    """Whitespace separates items; the ``=`` needs none around it."""
    entry = _exclusion(spec)
    assert entry.matches("a/b_test.py")
    assert entry.matches("a/conftest.py")
    assert not entry.matches("a/b.py")
    assert entry.rules == {"digits"}


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ("", "expected GLOBS [= RULES]"),
        ("= digits", "expected GLOBS [= RULES]"),
        ("*_test.py =", "'=' with no rule"),
        ("*_test.py = digit", "unknown rule(s) ['digit']"),
    ],
)
def test_a_malformed_exclude_is_rejected(spec: str, fragment: str) -> None:
    """A typo on the rule side would silently exclude nothing."""
    with pytest.raises(ValueError, match=re.escape(fragment)):
        _exclusion(spec)


def test_a_malformed_exclude_flag_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "digits", "--exclude", "*.py = digit", str(path)],
    )

    with pytest.raises(SystemExit):
        houselint.main()

    assert "unknown rule(s) ['digit']" in capsys.readouterr().err


def test_exclude_flag_is_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "from dataclasses import dataclass\n\n@dataclass\nclass A:\n    x: int\n"
    first = _write(tmp_path, body, name="a_test.py")
    second = _write(tmp_path, body, name="b_gen.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "houselint.py",
            "--select",
            "dataclass",
            "--exclude",
            "*_test.py",
            "--exclude",
            "*_gen.py",
            str(first),
            str(second),
        ],
    )

    assert houselint.main() == 0


@pytest.mark.parametrize(
    ("rule_name", "body"),
    [
        # Each body needs BOTH an unsuppressed violation (blanking the
        # tree/tokens must lose it) and a suppressed one (blanking the lines
        # must gain it). With only one half, blanking either view leaves the
        # result unchanged and the test passes without proving anything.
        (
            "dataclass",
            (
                "from dataclasses import dataclass\n\n"
                "@dataclass\nclass A:\n    x: int\n\n"
                "@dataclass  # house-ignore[dataclass] -- Plain record.\n"
                "class B:\n    y: int\n"
            ),
        ),
        (
            "globals",
            "MAX = 500\nMIN = 1  # house-ignore[globals] -- Shared default.\n",
        ),
        ("digits", "value = 12345\n"),
    ],
)
def test_a_rule_builds_only_the_views_it_reads(
    tmp_path: Path,
    rule_name: str,
    body: str,
) -> None:
    """A view a rule never touches must never be built for it.

    This is the regression guard on the reason ``needs`` was deleted rather
    than replaced. Rules once DECLARED their views and the dispatcher
    pre-built the union; nothing reconciled the declaration against the
    reads, so they drifted. Building every view instead made the dataclass
    and globals hooks ~370 ms and ~240 ms slower over 422 files, because
    neither reads ``tokens``. Laziness is what makes reading BE the
    declaration, and the memo boxes are what make the reads observable.

    ``tokens`` has no box -- it is an unmemoized iterator -- so it is
    observed by counting ``_tokenize`` calls instead.
    """
    path = _write(tmp_path, body)
    rule = next(r for r in houselint.default_rules() if r.name == rule_name)
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    tokenized: list[str] = []
    real_tokenize = houselint_types._tokenize

    def counting_tokenize(source: str) -> Iterator[tokenize.TokenInfo]:
        tokenized.append(source)
        return real_tokenize(source)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(houselint_types, "_tokenize", counting_tokenize)
        rule.check(unit)
        rule.fix(unit)

    built = {view for view in ("tree", "lines") if getattr(unit, f"_{view}")}
    if tokenized:
        built.add("tokens")
    assert built, f"{rule.name} reads no view of the Unit it was handed"
    # Every rule reads a strict SUBSET; a rule needing all three would make
    # this assertion vacuous and the laziness pointless.
    assert built != {"tree", "tokens", "lines"}


def test_exclude_matches_a_repo_relative_path(tmp_path: Path) -> None:
    """An anchored glob must match regardless of how the path was spelled.

    pre-commit passes relative paths, so an absolute-path caller silently
    matched nothing -- ``loop/x.py`` never matches ``/home/.../loop/x.py``.
    """
    path = _write(tmp_path / "sub", "BAD\n")

    findings, _ = run(
        [path.absolute()],
        active=(_Counting(),),
        fixing=frozenset(),
        exclude=(_exclusion("sub/mod.py"),),
        root=tmp_path,
    )

    assert findings == ()


def test_a_path_two_rules_fixed_is_listed_once(tmp_path: Path) -> None:
    """``written`` is the set of changed files, not one entry per fixer."""
    path = _write(tmp_path, "x = 1zz\n")

    _, written = run(
        [path],
        active=(_Trim(), _Trim()),
        fixing=frozenset({"trim"}),
    )

    assert path.read_text() == "x = 1\n"
    assert written == (path,)


def test_a_notebook_is_fixed_without_any_declared_view(tmp_path: Path) -> None:
    """``needs`` describes the PYTHON path; a notebook works from ``source``.

    A notebook has no single token stream -- each cell is tokenized on its
    own -- so ``read_unit`` leaves ``tokens`` empty for one. The rule must
    therefore not depend on a declared view to fix a notebook.
    """
    notebook = _write(
        tmp_path,
        json.dumps({"cells": [{"cell_type": "code", "source": ["x = 12345\n"]}]}),
        name="n.ipynb",
    )
    unit = houselint.read_unit(notebook)
    assert isinstance(unit, Unit)
    assert unit.tokens == ()

    assert DigitsRule().fix(unit) is not None


def test_a_finding_renders_as_path_line_rule_message(tmp_path: Path) -> None:
    finding = Finding(path=tmp_path / "m.py", line=4, rule="digits", message="hi")
    assert str(finding) == f"{tmp_path / 'm.py'}:4: [digits] hi"


def test_an_unparseable_file_is_skipped_silently(tmp_path: Path) -> None:
    """Ruff and ty both already report a syntax error; a third voice is noise."""
    path = _write(tmp_path, "def (:\n")
    findings, _ = run(
        [path],
        active=tuple(houselint.default_rules()),
        fixing=frozenset(),
    )
    assert findings == ()


def test_a_fixer_is_idempotent(tmp_path: Path) -> None:
    """``run`` is one pass, not a fixpoint loop; that requires idempotence."""
    path = _write(tmp_path, "value = 12345\n")
    digits = DigitsRule()

    run([path], active=(digits,), fixing=frozenset({"digits"}))
    _, written = run([path], active=(digits,), fixing=frozenset({"digits"}))

    assert written == ()


# -- Workers --------------------------------------------------------------
#
# A full-tree run is ~4000 files of pure-Python AST walking: CPU-bound, no
# shared state between files, and 30s serial on one core of a 128-core box.
# The pool splits PATHS, never rules, so a file is still read and parsed
# once and every fixer still sees the previous fixer's text.


@pytest.mark.cli_python_subprocess
def test_workers_report_the_same_findings_as_serial(tmp_path: Path) -> None:
    """Splitting the path list must not change what is reported, or its order."""
    paths = [
        _write(tmp_path, f"x = {n}\nBAD\n" * (n % 3 + 1), name=f"m{n:02}.py")
        for n in range(30)
    ]

    serial, _ = run(paths, active=(_Counting(),), fixing=frozenset())
    pooled, _ = run(paths, active=(_Counting(),), fixing=frozenset(), workers=3)

    assert pooled == serial
    assert len(pooled) == sum(n % 3 + 1 for n in range(30))


@pytest.mark.cli_python_subprocess
def test_workers_fix_every_file_and_list_each_once(tmp_path: Path) -> None:
    """``written`` is merged from every worker in path order, one entry per file."""
    dirty = [
        _write(tmp_path, f"v = 100{n:02}\n", name=f"d{n:02}.py") for n in range(25)
    ]
    clean = _write(tmp_path, "v = 1\n", name="clean.py")

    findings, written = run(
        [clean, *dirty],
        active=(DigitsRule(),),
        fixing=frozenset({"digits"}),
        workers=4,
    )

    assert findings == ()
    assert written == tuple(dirty)
    assert [p.read_text() for p in dirty] == [f"v = 10_0{n:02}\n" for n in range(25)]


@pytest.mark.cli_python_subprocess
def test_the_pool_is_sized_to_the_paths_not_the_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-file commit under a 24-worker hook must not start 24 processes.

    A worker costs a fork plus the rules import, so on a small commit the
    pool is slower than serial (measured: 5 files 0.16s vs 0.30s). Pinned by
    counting pool constructions rather than timing them.
    """
    sizes: list[int] = []

    class _Recording(ProcessPoolExecutor):
        def __init__(self, max_workers: int, mp_context: BaseContext) -> None:
            sizes.append(max_workers)
            super().__init__(max_workers=max_workers, mp_context=mp_context)

    monkeypatch.setattr(houselint, "ProcessPoolExecutor", _Recording)
    few = [_write(tmp_path, "BAD\n", name=f"f{n}.py") for n in range(5)]
    many = [_write(tmp_path, "BAD\n", name=f"m{n:02}.py") for n in range(40)]

    findings, _ = run(few, active=(_Counting(),), fixing=frozenset(), workers=24)
    assert len(findings) == 5
    assert sizes == []

    findings, _ = run(many, active=(_Counting(),), fixing=frozenset(), workers=24)
    assert len(findings) == 40
    assert sizes == [4]


@pytest.mark.cli_python_subprocess
def test_a_full_tree_argv_does_not_break_the_pool(tmp_path: Path) -> None:
    """``--all-files`` hands the CLI ~200 KB of paths; the pool must survive it.

    The default ``forkserver`` start method re-sends ``sys.argv`` to the server
    process on ITS command line, and Linux caps one argument at 128 KB -- so
    the server died at exec, the parent saw ``BrokenPipeError``, and the hook
    crashed only on a run large enough to matter. Reproduced end to end, in a
    subprocess, because an in-process pool may reuse a server that pytest's
    short argv already started.
    """
    paths = [
        _write(tmp_path, "value = 1\n", name=f"{'m' * 200}{n:04}.py")
        for n in range(700)
    ]
    assert sum(len(str(p)) for p in paths) > 131_072

    result = subprocess.run(  # noqa: S603 -- argv is built from tmp_path.
        [
            sys.executable,
            "-m",
            "codedoctor.lint",
            "--select",
            "digits",
            "--workers",
            "2",
            *map(str, paths),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(houselint.__file__).resolve().parents[3],
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr


def test_the_workers_flag_reaches_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--workers`` is how the hook hands ``$WORKERS`` to the pool."""
    seen: list[int] = []

    def spying_run(
        paths: Sequence[Path],
        *,
        active: Sequence[Rule],
        fixing: frozenset[str],
        exclude: Sequence[Exclusion] = (),
        root: Path = Path(),
        workers: int = 1,
    ) -> tuple[tuple[Finding, ...], tuple[Path, ...]]:
        del paths, active, fixing, exclude, root
        seen.append(workers)
        return (), ()

    monkeypatch.setattr(houselint, "run", spying_run)
    path = _write(tmp_path, "value = 1\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "digits", "--workers", "3", str(path)],
    )

    assert houselint.main() == 0
    assert seen == [3]


def test_an_unsized_workers_flag_means_serial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--workers ''`` is what a shell without the wrapper passes; it is not an error."""
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "digits", "--workers", "", str(path)],
    )

    assert houselint.main() == 1


# -- DataclassRule detection ---------------------------------------------
#
# Ported from check_dataclass_test.py when that module was drained into
# houselint. These pin the DETECTION logic; everything above pins dispatch.


def _dataclass_findings(tmp_path: Path, body: str) -> Sequence[Finding]:
    """Run DataclassRule over ``body`` and return its findings."""
    path = _write(tmp_path, body)
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    return DataclassRule().check(unit)


def test_missing_kwargs_are_reported(tmp_path: Path) -> None:
    findings = _dataclass_findings(
        tmp_path,
        "from dataclasses import dataclass\n\n@dataclass\nclass A:\n    x: int\n",
    )
    assert len(findings) == 1
    assert "kw_only=True" in findings[0].message
    assert "slots=True" in findings[0].message


def _staticmethod_findings(tmp_path: Path, body: str) -> Sequence[Finding]:
    """Run StaticMethodRule over ``body`` and return its findings."""
    path = _write(tmp_path, body)
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    return StaticMethodRule().check(unit)


def test_a_staticmethod_is_reported(tmp_path: Path) -> None:
    """STYLE.md: ``@classmethod`` over ``@staticmethod``."""
    findings = _staticmethod_findings(
        tmp_path,
        "class A:\n    @staticmethod\n    def f() -> int:\n        return 1\n",
    )
    assert len(findings) == 1
    assert findings[0].line == 2
    assert "@classmethod" in findings[0].message
    assert "f" in findings[0].message


def test_the_fix_rewrites_every_parameter_shape(tmp_path: Path) -> None:
    """``cls`` leads, and a bare ``()`` must not gain a stray separator."""
    body = (
        "class A:\n"
        "    @staticmethod\n"
        "    def none() -> int:\n"
        "        return 1\n\n"
        "    @staticmethod\n"
        "    def one(x: int) -> int:\n"
        "        return x\n\n"
        "    @staticmethod\n"
        "    def star(*args: int) -> int:\n"
        "        return len(args)\n\n"
        "    @staticmethod\n"
        "    def kwonly(*, k: int = 1) -> int:\n"
        "        return k\n"
    )
    path = _write(tmp_path, body)
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    fixed = StaticMethodRule().fix(unit)
    assert fixed is not None
    assert "@staticmethod" not in fixed
    assert "def none(cls) -> int:" in fixed
    assert "def one(cls, x: int) -> int:" in fixed
    assert "def star(cls, *args: int) -> int:" in fixed
    assert "def kwonly(cls, *, k: int = 1) -> int:" in fixed
    # The rewrite must still be importable, and `cls` must bind the class.
    namespace: dict[str, object] = {}
    exec(compile(fixed, str(path), "exec"), namespace)  # noqa: S102 -- The test executes source produced by the houselint rewrite itself to verify the transformed code.
    rewritten_class = namespace["A"]
    assert isinstance(rewritten_class, type)
    assert getattr(rewritten_class, "none")() == 1  # noqa: B009 -- The test accesses a dynamically generated fixture attribute whose name is supplied by the rewritten source.
    assert getattr(rewritten_class, "star")(1, 2) == 2  # noqa: B009 -- The test accesses a dynamically generated fixture attribute whose name is supplied by the rewritten source.


def test_a_staticmethod_call_binding_a_class_attribute_is_reported(
    tmp_path: Path,
) -> None:
    """``name = staticmethod(fn)`` is the decorator spelled as a call.

    The class-body form binds a module function as a method without the
    ``@``; it is the same construct and the same rule.
    """
    findings = _staticmethod_findings(
        tmp_path,
        "def price() -> int:\n    return 1\n\n"
        "class A:\n    cost = staticmethod(price)\n",
    )
    assert len(findings) == 1
    assert findings[0].line == 5
    assert "cost" in findings[0].message
    assert "@classmethod" in findings[0].message


def test_a_local_staticmethod_call_binding_becomes_a_forwarding_classmethod(
    tmp_path: Path,
) -> None:
    """The bound def's signature is in hand, so the forwarder copies it."""
    path = _write(
        tmp_path,
        "def price(a: int, /, b: int, *rest: int, k: int = 2, **kw: int) -> int:\n"
        '    """Sum everything."""\n'
        "    return a + b + sum(rest) + k + sum(kw.values())\n\n"
        "class A:\n"
        "    cost = staticmethod(price)\n",
    )
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    fixed = StaticMethodRule().fix(unit)
    assert fixed is not None
    assert "staticmethod(" not in fixed
    assert (
        "    @classmethod\n"
        "    def cost(cls, a: int, /, b: int, *rest: int, k: int=2, **kw: int) -> int:\n"
        '        """Sum everything."""\n'
        "        return price(a, b, *rest, k=k, **kw)\n"
    ) in fixed
    namespace: dict[str, object] = {}
    exec(compile(fixed, str(path), "exec"), namespace)  # noqa: S102 -- The test executes source produced by the houselint rewrite itself to verify the transformed code.
    rewritten_class = namespace["A"]
    assert isinstance(rewritten_class, type)
    assert getattr(rewritten_class, "cost")(1, 2, 3, k=4, z=5) == 15  # noqa: B009 -- The test accesses a dynamically generated fixture attribute whose name is supplied by the rewritten source.
    assert _write(tmp_path, fixed) == path
    refixed = houselint.read_unit(path)
    assert isinstance(refixed, Unit)
    assert StaticMethodRule().fix(refixed) is None


def test_an_imported_staticmethod_call_binding_is_not_autofixed(
    tmp_path: Path,
) -> None:
    """An imported function's signature is not in the file to copy."""
    path = _write(
        tmp_path,
        "from os.path import join\n\nclass A:\n    cost = staticmethod(join)\n",
    )
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert StaticMethodRule().fix(unit) is None


def test_an_autograd_staticmethod_is_not_autofixed(tmp_path: Path) -> None:
    """Under ``torch.compile`` a classmethod ``forward`` gets ``cls`` twice."""
    path = _write(
        tmp_path,
        "import torch\n\n"
        "class F(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx: object, x: int) -> int:\n"
        "        return x\n",
    )
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert StaticMethodRule().fix(unit) is None


def test_the_fix_is_idempotent(tmp_path: Path) -> None:
    """A file with no ``@staticmethod`` yields no rewrite."""
    path = _write(
        tmp_path,
        "class A:\n    @classmethod\n    def f(cls) -> int:\n        return 1\n",
    )
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    assert StaticMethodRule().fix(unit) is None


def test_a_scoped_exclude_skips_only_the_named_rules(tmp_path: Path) -> None:
    """Ruff's ``per-file-ignores``: the file is still read for every other rule."""
    body = (
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class A:\n"
        "    x: int = 1\n\n"
        "class B:\n"
        "    @staticmethod\n"
        "    def f() -> int:\n"
        "        return 1\n"
    )
    path = _write(tmp_path, body, name="vendored.py")
    findings, _ = run(
        [path],
        active=(DataclassRule(), StaticMethodRule()),
        fixing=frozenset(),
        exclude=(_exclusion("vendored.py = staticmethod"),),
        root=tmp_path,
    )
    rules = {finding.rule for finding in findings}
    assert rules == {"dataclass"}, f"staticmethod should have skipped: {findings}"


def test_scoped_excludes_union_their_rules(tmp_path: Path) -> None:
    body = (
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class A:\n"
        "    x: int = 1\n\n"
        "class B:\n"
        "    @staticmethod\n"
        "    def f() -> int:\n"
        "        return 1\n"
    )
    path = _write(tmp_path / "vendor", body, name="mod.py")
    findings, _ = run(
        [path],
        active=(DataclassRule(), StaticMethodRule()),
        fixing=frozenset(),
        exclude=(
            _exclusion("vendor/** = staticmethod"),
            _exclusion("mod.py = dataclass"),
        ),
        root=tmp_path,
    )
    assert not findings


def test_a_classmethod_is_clean(tmp_path: Path) -> None:
    assert not _staticmethod_findings(
        tmp_path,
        "class A:\n    @classmethod\n    def f(cls) -> int:\n        return 1\n",
    )


def test_the_word_staticmethod_in_a_string_is_clean(tmp_path: Path) -> None:
    """The substring prefilter must not decide; only a real decorator counts."""
    assert not _staticmethod_findings(tmp_path, 'DOC = "@staticmethod is banned"\n')


def test_compliant_dataclass_is_clean(tmp_path: Path) -> None:
    assert not _dataclass_findings(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass(kw_only=True, slots=True)\nclass A:\n    x: int\n",
    )


def test_zero_arg_super_in_a_slotted_dataclass_is_reported(tmp_path: Path) -> None:
    """``slots=True`` + zero-arg ``super()`` raises below Python 3.14.

    ``slots=True`` makes the decorator build a NEW class and discard the
    original, but zero-arg ``super()`` resolves through the ``__class__``
    cell compiled into the method, which still names the discarded one --
    ``TypeError: super(type, obj): obj must be an instance or subtype of
    type`` (python/cpython#90562, fixed in 3.14). A monorepo on 3.14 cannot
    see it; a ``requires-python = ">=3.12"`` consumer hits it on first call.
    """
    findings = _dataclass_findings(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass(kw_only=True, slots=True)\n"
        "class A:\n"
        "    def f(self) -> int:\n"
        "        return 1\n\n"
        "@dataclass(kw_only=True, slots=True)\n"
        "class B(A):\n"
        "    def f(self) -> int:\n"
        "        return super().f()\n",
    )
    assert len(findings) == 1
    assert "super()" in findings[0].message
    assert "B" in findings[0].message


def test_two_arg_super_in_a_slotted_dataclass_is_clean(tmp_path: Path) -> None:
    """The explicit form resolves at call time and is the sanctioned fix."""
    assert not _dataclass_findings(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass(kw_only=True, slots=True)\n"
        "class A:\n"
        "    def f(self) -> int:\n"
        "        return 1\n\n"
        "@dataclass(kw_only=True, slots=True)\n"
        "class B(A):\n"
        "    def f(self) -> int:\n"
        "        return super(B, self).f()\n",
    )


def test_zero_arg_super_outside_a_slotted_dataclass_is_clean(tmp_path: Path) -> None:
    """A plain class is unaffected: no new class object, no stale cell."""
    assert not _dataclass_findings(
        tmp_path,
        "class A:\n"
        "    def f(self) -> int:\n"
        "        return 1\n\n"
        "class B(A):\n"
        "    def f(self) -> int:\n"
        "        return super().f()\n",
    )


def test_a_nested_slotted_dataclass_is_still_checked(tmp_path: Path) -> None:
    """Test stubs declare these inside a function; the walk must reach them."""
    findings = _dataclass_findings(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass(kw_only=True, slots=True)\n"
        "class A:\n"
        "    def f(self) -> int:\n"
        "        return 1\n\n"
        "def test_x() -> None:\n"
        "    @dataclass(kw_only=True, slots=True)\n"
        "    class B(A):\n"
        "        def f(self) -> int:\n"
        "            return super().f()\n",
    )
    assert len(findings) == 1
    assert "B" in findings[0].message


def test_a_per_class_house_ignore_exempts_the_decorator(tmp_path: Path) -> None:
    """The dispatcher's marker is the only opt-out; the rule owns none of its own."""
    path = _write(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass()  # house-ignore[dataclass] -- Plain record.\n"
        "class A:\n    x: int\n\n"
        "@dataclass(slots=True)\n"
        "class B:\n    y: int\n",
    )
    findings, _ = run([path], active=(DataclassRule(),), fixing=frozenset())
    assert [f.line for f in findings] == [7]


def test_an_attribute_spelled_decorator_is_checked(tmp_path: Path) -> None:
    """``@dataclasses.dataclass`` is the same decorator by another path."""
    assert (
        len(
            _dataclass_findings(
                tmp_path,
                "import dataclasses\n\n@dataclasses.dataclass\nclass A:\n    x: int\n",
            ),
        )
        == 1
    )


def test_an_unrelated_decorator_is_ignored(tmp_path: Path) -> None:
    assert not _dataclass_findings(
        tmp_path,
        "from functools import cache\n\n@cache\nclass A:\n    x: int\n",
    )


# -- ConfigGlobalsRule detection -----------------------------------------
#
# Ported from check_config_globals_test.py when that module was drained
# into houselint.

_CONFIG_HEADER = (
    "from configgle import Fig\n\n\nclass Thing:\n    class Config(Fig['Thing']):\n"
)


def _globals_findings(tmp_path: Path, body: str) -> Sequence[Finding]:
    """Run ConfigGlobalsRule over ``body`` and return its findings."""
    path = _write(tmp_path, body)
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    return ConfigGlobalsRule().check(unit)


def test_module_scalar_const_in_config_file_flagged(tmp_path: Path) -> None:
    findings = _globals_findings(
        tmp_path,
        "DEFAULT_DIM = 256\n\n" + _CONFIG_HEADER + "        dim: int = 256\n",
    )
    assert len(findings) == 1
    assert "DEFAULT_DIM" in findings[0].message
    assert findings[0].line == 1


def test_string_and_collection_literals_flagged(tmp_path: Path) -> None:
    findings = _globals_findings(
        tmp_path,
        "DATA_ROOT = '/opt/scratch/x'\n"
        "SPLITS = ('train', 'test')\n\n" + _CONFIG_HEADER + "        a: int = 1\n",
    )
    names = " ".join(f.message for f in findings)
    assert "DATA_ROOT" in names
    assert "SPLITS" in names


def test_config_field_default_to_const_flagged(tmp_path: Path) -> None:
    findings = _globals_findings(
        tmp_path,
        _CONFIG_HEADER + "        dim: int = DEFAULT_DIM\n",
    )
    assert len(findings) == 1
    assert "'dim'" in findings[0].message
    assert "DEFAULT_DIM" in findings[0].message


def test_attribute_default_flagged(tmp_path: Path) -> None:
    findings = _globals_findings(
        tmp_path,
        _CONFIG_HEADER + "        dim: int = consts.DEFAULT_DIM\n",
    )
    assert any("DEFAULT_DIM" in f.message for f in findings)


def test_binop_literal_knob_flagged(tmp_path: Path) -> None:
    """A knob written as an arithmetic expression of literals is still a knob."""
    findings = _globals_findings(tmp_path, "CHUNK = 1 << 20\nTIMEOUT = 5 * 60\n")
    names = " ".join(f.message for f in findings)
    assert "CHUNK" in names
    assert "TIMEOUT" in names


def test_union_type_alias_binop_not_flagged(tmp_path: Path) -> None:
    """``X | Y`` is a type-union alias, not a value -- must stay exempt."""
    assert not _globals_findings(tmp_path, "Alias = int | str | None\n")


def test_inline_literal_default_clean(tmp_path: Path) -> None:
    assert not _globals_findings(tmp_path, _CONFIG_HEADER + "        dim: int = 256\n")


def test_injected_lowercase_callable_default_clean(tmp_path: Path) -> None:
    """``init_weight: InitFn = kaiming_uniform`` is injection, not a knob."""
    assert not _globals_findings(
        tmp_path,
        _CONFIG_HEADER + "        init_weight: object = kaiming_uniform\n",
    )


def test_non_config_file_still_scanned(tmp_path: Path) -> None:
    """A knob in a plain "constants" module is still the anti-pattern."""
    findings = _globals_findings(tmp_path, "DEFAULT_DIM = 256\nN_STEPS = 200\n")
    names = " ".join(f.message for f in findings)
    assert "DEFAULT_DIM" in names
    assert "N_STEPS" in names


def test_type_alias_and_logger_exempt(tmp_path: Path) -> None:
    assert not _globals_findings(
        tmp_path,
        "import logging\n"
        "from typing import Literal\n\n"
        "logger = logging.getLogger(__name__)\n"
        "Mode = Literal['a', 'b']\n"
        "Pair = tuple[int, int]\n"
        "Union = int | str\n"
        "__all__ = ['Thing']\n\n" + _CONFIG_HEADER + "        a: int = 1\n",
    )


def test_exempt_calls_not_flagged(tmp_path: Path) -> None:
    assert not _globals_findings(
        tmp_path,
        "import re\n"
        "from typing import TypeVar\n\n"
        "NAME_RE = re.compile('x')\n"
        "T = TypeVar('T')\n\n" + _CONFIG_HEADER + "        a: int = 1\n",
    )


def test_empty_container_not_flagged(tmp_path: Path) -> None:
    """An empty container is a registry populated at import, not a value."""
    assert not _globals_findings(
        tmp_path,
        "REGISTRY: dict[str, int] = {}\n"
        "CACHE = []\n\n" + _CONFIG_HEADER + "        a: int = 1\n",
    )


def _globals_run(tmp_path: Path, body: str) -> Sequence[Finding]:
    """Run ConfigGlobalsRule through the dispatcher, so markers apply."""
    path = _write(tmp_path, body)
    findings, _ = run([path], active=(ConfigGlobalsRule(),), fixing=frozenset())
    return findings


def test_suppression_silences_finding(tmp_path: Path) -> None:
    assert not _globals_run(
        tmp_path,
        "ANSI = '\\x1b[0m'  # house-ignore[globals] -- Terminal escape.\n\n"
        + _CONFIG_HEADER
        + "        a: int = 1\n",
    )


def test_suppression_on_multiline_assignment_trailing_line(tmp_path: Path) -> None:
    """The marker may sit on the closing line of a multi-line assignment."""
    assert not _globals_run(
        tmp_path,
        "PERM = (\n    0,\n    1,\n)  # house-ignore[globals] -- Fixed permutation.\n",
    )


def test_suppression_on_config_field(tmp_path: Path) -> None:
    assert not _globals_run(
        tmp_path,
        _CONFIG_HEADER
        + "        dim: int = DEFAULT_DIM  # house-ignore[globals] -- Legacy.\n",
    )


def test_final_annotated_constant_exempt(tmp_path: Path) -> None:
    """``Final`` is a type-checker-enforced constant declaration."""
    assert not _globals_findings(
        tmp_path,
        "from typing import Final\n\n"
        "SESSION_START_PATH: Final = '/api/sessions/start'\n"
        "MAX_ITEMS: Final[int] = 500\n"
        "OPS: Final[dict[str, str]] = {'is': '='}\n",
    )


def test_final_from_typing_extensions_exempt(tmp_path: Path) -> None:
    """``Final`` reached via attribute access must also exempt."""
    assert not _globals_findings(
        tmp_path,
        "import typing\n\nROUTE: typing.Final = '/api/x'\n",
    )


def test_non_final_annotated_constant_still_flagged(tmp_path: Path) -> None:
    """A plain type annotation is NOT an exemption -- only ``Final`` is."""
    findings = _globals_findings(tmp_path, "MAX_ITEMS: int = 500\n")
    assert any("MAX_ITEMS" in f.message for f in findings)


def test_final_with_redundant_suppression_is_error(tmp_path: Path) -> None:
    """``Final`` already exempts, so the marker is superfluous -- and the marker
    cannot silence that complaint, which is reported as ``houselint``.
    """
    findings = _globals_run(
        tmp_path,
        "from typing import Final\n\n"
        "ROUTE: Final = '/api/x'  # house-ignore[globals] -- Redundant.\n",
    )
    assert len(findings) == 1
    assert findings[0].rule == "houselint"
    assert findings[0].line == 3
    assert "superfluous" in findings[0].message.lower()


def test_suppression_without_final_on_real_tunable_ok(tmp_path: Path) -> None:
    """A genuine tunable that cannot be Final may still use the marker."""
    assert not _globals_run(
        tmp_path,
        "MAX_ITEMS = 500  # house-ignore[globals] -- Shared default across modules.\n",
    )


def test_syntax_error_returns_empty(tmp_path: Path) -> None:
    assert not _globals_findings(tmp_path, "def (:\n")


def test_config_bases_setting_narrows_which_classes_are_configs(
    tmp_path: Path,
) -> None:
    """The base list is a field, so a project with other bases can retarget."""
    path = _write(tmp_path, _CONFIG_HEADER + "        dim: int = DEFAULT_DIM\n")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)

    assert ConfigGlobalsRule().check(unit)
    assert not ConfigGlobalsRule(config_bases=("Other",)).check(unit)


# -- DigitsRule detection ------------------------------------------------
#
# Ported from group_digits_test.py when that module was drained into
# houselint.


def _regrouped(tmp_path: Path, body: str, name: str = "mod.py") -> str | None:
    """Run DigitsRule's fixer over ``body`` and return its rewrite."""
    path = _write(tmp_path, body, name=name)
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    return DigitsRule().fix(unit)


def test_check_mode_reports_needed_change_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(sys, "argv", ["houselint.py", "--select", "digits", str(path)])

    assert houselint.main() == 1
    assert path.read_text() == "value = 12345\n"
    assert "[digits]" in capsys.readouterr().out


def test_fix_mode_rewrites_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "digits", "--fix", "digits", str(path)],
    )

    assert houselint.main() == 0
    assert path.read_text() == "value = 12_345\n"


def test_a_short_literal_is_left_alone(tmp_path: Path) -> None:
    assert _regrouped(tmp_path, "value = 1234\n") is None


def test_an_already_grouped_literal_is_left_alone(tmp_path: Path) -> None:
    assert _regrouped(tmp_path, "value = 12_345\n") is None


def test_hex_octal_and_binary_are_left_alone(tmp_path: Path) -> None:
    """Non-decimal bases carry their own grouping conventions."""
    assert (
        _regrouped(tmp_path, "a = 0xDEADBEEF\nb = 0o777777\nc = 0b11111111\n") is None
    )


def test_a_float_is_left_alone(tmp_path: Path) -> None:
    assert _regrouped(tmp_path, "value = 12345.0\n") is None


def test_min_digits_setting_changes_the_threshold(tmp_path: Path) -> None:
    path = _write(tmp_path, "value = 1234\n")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)

    assert DigitsRule().fix(unit) is None
    assert DigitsRule(min_digits=4).fix(unit) == "value = 1_234\n"


def test_several_literals_on_one_line_all_regroup(tmp_path: Path) -> None:
    """Replacements apply right-to-left so earlier columns stay valid."""
    assert _regrouped(tmp_path, "a, b = 12345, 678901\n") == "a, b = 12_345, 678_901\n"


def test_a_notebook_code_cell_is_regrouped(tmp_path: Path) -> None:
    result = _regrouped(
        tmp_path,
        json.dumps({"cells": [{"cell_type": "code", "source": ["x = 12345\n"]}]}),
        name="n.ipynb",
    )
    assert result is not None
    assert json.loads(result)["cells"][0]["source"] == ["x = 12_345\n"]


def test_a_notebook_markdown_cell_is_untouched(tmp_path: Path) -> None:
    assert (
        _regrouped(
            tmp_path,
            json.dumps(
                {"cells": [{"cell_type": "markdown", "source": ["see 12345\n"]}]},
            ),
            name="n.ipynb",
        )
        is None
    )


def test_a_file_is_read_once_and_parsed_once_per_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One read and one parse per VERSION of a file, fixed or not.

    A rewritten file is new text, so a tree-reading rule must parse it
    again; that second parse is correctness, not waste. The waste was the
    second READ -- ``run`` re-read from disk the bytes it had just written.
    Counting both pins the contract the module docstring states.
    """
    parses = 0
    reads = 0
    real_read = Path.read_bytes

    def counting_parse(source: str, path: Path) -> ast.Module | None:
        nonlocal parses
        parses += 1
        return ast.parse(source, filename=str(path))

    def counting_read(self: Path) -> bytes:
        nonlocal reads
        reads += 1
        return real_read(self)

    monkeypatch.setattr(houselint_types, "_parse", counting_parse)
    monkeypatch.setattr(Path, "read_bytes", counting_read)
    # Both bodies declare a class: ``DataclassRule`` skips a source with no
    # ``class`` keyword without parsing it, so a class-free fixture would
    # count zero parses and measure nothing.
    clean = _write(tmp_path, "class A:\n    v = 12_345\n", name="clean.py")
    dirty = _write(tmp_path, "class B:\n    v = 12345\n", name="dirty.py")
    # ``DataclassRule`` is what reads ``tree``; pairing the fixer with a
    # rule that only reads ``source`` would count zero parses and prove
    # nothing, since the views are lazy.
    rules = (DigitsRule(), DataclassRule())

    run([clean, dirty], active=rules, fixing=frozenset({"digits"}))

    assert dirty.read_text() == "class B:\n    v = 12_345\n"
    # Two files, read once each; the rewritten one is never re-read.
    assert reads == 2
    # ONE parse per file, even the rewritten one. Fixes run before checks,
    # so the pre-fix text is never handed to a tree-reading rule and its
    # tree is therefore never built -- laziness turns "rebuild the Unit
    # after a write" from a second parse into no parse at all.
    assert parses == 2


def test_a_notebook_carries_no_python_only_view(tmp_path: Path) -> None:
    """A notebook has no module and no single token stream, whoever asks.

    The kind gate survives laziness because it is a CORRECTNESS gate, not
    an optimization: a notebook's raw JSON must never reach ``ast.parse``.
    ``lines`` needs no such gate -- nothing reads a notebook's lines, so
    nothing builds them, which is the same outcome the old kind guard on
    ``lines`` was added to produce.
    """
    notebook = _write(
        tmp_path,
        json.dumps({"cells": [{"cell_type": "code", "source": ["x = 12345\n"]}]}),
        name="n.ipynb",
    )

    unit = houselint.read_unit(notebook)

    assert isinstance(unit, Unit)
    assert unit.tokens == ()
    assert unit.tree is None
    # The fixer works from ``source``, tokenizing each cell on its own.
    assert DigitsRule().fix(unit) is not None
    assert not unit._lines, "nothing reads a notebook's lines"


def test_no_rule_declares_the_views_it_reads() -> None:
    """``needs`` is gone: a declaration nothing reconciles is a defect source."""
    for rule in houselint.default_rules():
        assert not hasattr(rule, "needs")


def test_a_non_ascii_line_does_not_corrupt_the_rewrite(tmp_path: Path) -> None:
    """Token columns are CHARACTER offsets; splicing bytes with them corrupts.

    A NUMBER token after a non-ASCII character on the same line splices at
    the wrong byte, cutting the multi-byte codepoint and duplicating a
    digit. Scanning ``loop/`` found 20 files already in that shape.
    """
    path = _write(tmp_path, 'label = "café"; n = 123456\n')

    run([path], active=(DigitsRule(),), fixing=frozenset({"digits"}))

    assert path.read_text(encoding="utf-8") == 'label = "café"; n = 123_456\n'


def test_leading_zeros_survive_grouping(tmp_path: Path) -> None:
    """Grouping is a textual regroup, not an ``int()`` round-trip."""
    path = _write(tmp_path, "value = 00000\n")

    run([path], active=(DigitsRule(),), fixing=frozenset({"digits"}))

    assert path.read_text() == "value = 00_000\n"


def test_a_literal_beyond_the_int_conversion_limit_is_grouped(tmp_path: Path) -> None:
    """``int()`` refuses past 4300 digits; a textual regroup has no limit."""
    path = _write(tmp_path, "value = " + "1" * 4301 + "\n")

    _, written = run([path], active=(DigitsRule(),), fixing=frozenset({"digits"}))

    assert written == (path,)
    assert path.read_text().count("_") == 1433
    assert path.read_text().replace("_", "") == "value = " + "1" * 4301 + "\n"


def test_a_file_that_is_not_utf8_is_reported_not_silently_clean(
    tmp_path: Path,
) -> None:
    """A file that cannot be decoded was reported CLEAN, which is a lie.

    ``_tokenize`` returned ``()`` on the decode error, so the fixer found no
    NUMBER tokens and returned None -- which ``check`` reads as "already
    grouped". The file passed the hook without ever being inspected.
    """
    path = tmp_path / "mod.py"
    path.write_bytes(b"# calf\xe9\nvalue = 123456\n")

    findings, written = run(
        [path],
        active=(DigitsRule(),),
        fixing=frozenset({"digits"}),
    )

    assert written == ()
    assert len(findings) == 1
    assert path.read_bytes() == b"# calf\xe9\nvalue = 123456\n"


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        # A value that cannot become an int must be reported AS a coercion
        # failure. Asserting only on the exit code would let argparse's
        # "unrecognized arguments" stand in for it, and the test would pass
        # unchanged the day the flag is renamed out of existence.
        (
            ("--set", "digits.min-digits=null"),
            "cannot read 'null' as int",
        ),
        (("--set", "digits.min-digits=[]"), "cannot read '[]' as int"),
        (("--set", "digits.min-digits=abc"), "cannot read 'abc' as int"),
        # A key naming no field must say so, and name the fields that exist.
        (("--set", "digits.bogus=1"), "has no field(s) ['bogus']"),
        # A key whose rule was never selected is a routing error, distinct
        # from a bad field: the field may well exist on that other rule.
        (("--set", "globals.config-bases=Fig"), "not in --select"),
        # A key with no dot cannot route at all.
        (("--set", "min-digits=5"), "expected RULE.FIELD"),
        # A --set with no '=' is not an assignment.
        (("--set", "digits.min-digits"), "expected RULE.FIELD=VALUE"),
        # Naming a rule that does not exist, in either list flag.
        (("--fix", "bogus"), "not in --select"),
    ],
)
def test_a_malformed_setting_exits_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: tuple[str, ...],
    expected: str,
) -> None:
    """Junk on the command line is a usage error, never a raw traceback.

    ``configure`` documented ``Raises: ValueError ... a value will not
    coerce`` and did not: ``min-digits=[]`` reached the fixer as a list,
    ``min-digits=null`` as None, and a bare ``min-digits`` silently became
    True -- turning the threshold into 1.
    """
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "digits", *flags, str(path)],
    )

    with pytest.raises(SystemExit):
        houselint.main()

    assert expected in capsys.readouterr().err
    assert path.read_text() == "value = 12345\n"


def test_a_bare_setting_flag_does_not_silently_become_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--digits-min-digits`` with no value must not set the threshold to 1."""
    path = _write(tmp_path, "value = 1234\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "digits", "--digits-min-digits", str(path)],
    )

    with pytest.raises(SystemExit):
        houselint.main()

    assert path.read_text() == "value = 1234\n"


def test_a_container_setting_takes_several_words(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tuple field must be expressible on the command line.

    An earlier mini-language split on ``,`` BEFORE ``json.loads``, and ``,``
    is JSON's own list separator, so ``config-bases=["Fig","Makes"]`` raised.
    Worse, the truncated ``'["Fig"'`` decoded to a STRING, turning the
    membership test into a substring test where ``"ig"`` matched.
    """
    path = _write(tmp_path, _CONFIG_HEADER + "        dim: int = DEFAULT_DIM\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "houselint.py",
            "--select",
            "globals",
            "--set",
            "globals.config-bases=Other Another",
            str(path),
        ],
    )

    assert houselint.main() == 0


def test_select_all_names_every_rule() -> None:
    active, fixing = houselint.select_rules(
        houselint.default_rules(),
        select="ALL",
        fix="",
        settings=[],
    )
    assert {r.name for r in active} == {r.name for r in houselint.default_rules()}
    assert fixing == frozenset()


def test_fix_all_names_every_fixable_selected_rule() -> None:
    active, fixing = houselint.select_rules(
        houselint.default_rules(),
        select="digits dataclass",
        fix="ALL",
        settings=[],
    )
    assert {r.name for r in active} == {"digits", "dataclass"}
    assert fixing == {"digits"}


def test_a_config_base_is_matched_whole_not_as_a_substring(tmp_path: Path) -> None:
    """``"ig" in ("Fig",)`` must never be what decides a class is a Config."""
    path = _write(tmp_path, _CONFIG_HEADER + "        dim: int = DEFAULT_DIM\n")
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)

    assert not ConfigGlobalsRule(config_bases=("ig",)).check(unit)
    assert ConfigGlobalsRule(config_bases=("Fig",)).check(unit)


def test_a_setting_flag_sets_the_rule_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rule gains a CLI knob by gaining a dataclass field, nothing else."""
    path = _write(tmp_path, "value = 1234\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "houselint.py",
            "--select",
            "digits",
            "--set",
            "digits.min-digits=4",
            str(path),
        ],
    )

    assert houselint.main() == 1


def test_a_file_named_fix_does_not_disable_the_fixer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whether a flag means "fix" must not depend on the filesystem.

    ``_Append`` decided "path or spec?" with ``Path(spec).exists()``, so a
    file named ``fix`` in the working directory made ``--check-digits=fix``
    silently drop the fix request.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fix").write_text("", encoding="utf-8")
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "digits", "--fix", "digits", str(path)],
    )

    assert houselint.main() == 0
    assert path.read_text() == "value = 12_345\n"


def test_a_nonexistent_path_is_not_stolen_as_a_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A path that does not exist is still a PATH, not a settings string."""
    missing = tmp_path / "nonexistent.py"
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "digits", str(missing)],
    )

    assert houselint.main() == 1
    assert "nonexistent.py" in capsys.readouterr().out


def test_there_is_no_fix_flag_for_an_unfixable_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise ``--fix-dataclass`` parses fine and does nothing."""
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "dataclass", "--fix", "dataclass", str(path)],
    )

    with pytest.raises(SystemExit):
        houselint.main()


def _repo(tmp_path: Path) -> dict[str, Path]:
    """Return a git root holding one file per kind, ignored ones, and a txt."""
    body = "value = 12345\n"
    (tmp_path / ".git").mkdir()
    _write(tmp_path, "ignored.py\nbuild/\n", name=".gitignore")
    _write(tmp_path, "*.ipynb\n!keep.ipynb\n", name="sub/deep/.gitignore")
    _write(tmp_path, body, name="ignored.py")
    _write(tmp_path, body, name="build/gen.py")
    _write(tmp_path, body, name="notes.txt")
    _write(tmp_path, body, name=".venv/lib/site.py")
    _write(tmp_path, body, name=".hidden/h.py")
    _write(tmp_path, "{}", name="sub/deep/dropped.ipynb")
    return {
        "py": _write(tmp_path, body, name="a.py"),
        "pyi": _write(tmp_path, body, name="sub/b.pyi"),
        "ipynb": _write(
            tmp_path,
            json.dumps({"cells": [{"cell_type": "code", "source": [body]}]}),
            name="sub/deep/keep.ipynb",
        ),
    }


def test_python_rules_skip_a_stub(tmp_path: Path) -> None:
    """A ``python`` fixer rewrote stub signatures and left stubs unparseable."""
    stub = _write(tmp_path, "def f(_x: int = ...) -> None: ...\nBAD\n", name="m.pyi")

    assert run([stub], active=(_Counting(),), fixing=frozenset()) == ((), ())


@dataclass(frozen=True, kw_only=True, slots=True)
class _StubRule(_Counting):
    name: ClassVar[str] = "stub-rule"
    kinds: ClassVar[frozenset[str]] = frozenset({"pyi"})
    marker: str = "int"


def test_a_rule_that_declares_pyi_parses_and_checks_stubs(tmp_path: Path) -> None:
    stub = _write(tmp_path, "x: int\n", name="m.pyi")

    unit = houselint.read_unit(stub)
    assert isinstance(unit, Unit)
    assert isinstance(unit.tree, ast.Module)
    assert unit.tokens
    findings, _ = run([stub], active=(_StubRule(),), fixing=frozenset())
    assert [(f.line, f.rule) for f in findings] == [(1, "stub-rule")]


def test_an_abbreviated_flag_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--sel`` must not resolve to ``--select``; a typo would configure a rule."""
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(sys, "argv", ["houselint.py", "--sel", "digits", str(path)])

    with pytest.raises(SystemExit) as raised:
        houselint.main()

    assert raised.value.code == 2


def _help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> str:
    """Return ``houselint.py --help``, whitespace-normalized."""
    # Argparse wraps at hyphens, which a whitespace join cannot undo.
    monkeypatch.setenv("COLUMNS", "10000")
    monkeypatch.setattr(sys, "argv", ["houselint.py", "--help"])
    with pytest.raises(SystemExit):
        houselint.main()
    return " ".join(capsys.readouterr().out.split())


def test_help_shows_every_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every flag states the value it takes when not given."""
    text = _help(monkeypatch, capsys)

    assert "exec uv" not in text
    for expected in (
        "Lint Python files and notebooks against the house rules.",
        "outside hidden directories and .gitignore patterns",
        "houselint --fix '' report only, rewrite nothing",
        "Exit status: 0 clean, 1 findings remain, 2 usage error.",
        (
            "[--select RULES] [--fix RULES] [--set RULE.FIELD=VALUE] "
            "[--exclude GLOBS[=RULES]] [--workers WORKERS] [paths ...]"
        ),
        "paths files and directories to lint (default: .)",
        "--select RULES space-separated rules to run, or ALL (default: none)",
        (
            "--fix RULES space-separated selected rules allowed to rewrite files, "
            "or ALL (default: none)"
        ),
        "--set RULE.FIELD=VALUE set one rule setting; repeatable (default: none)",
        (
            "--exclude GLOBS[=RULES] skip files matching GLOBS for every rule, or "
            "only for RULES; repeatable; a glob without '/' matches a basename "
            "anywhere (default: none)"
        ),
        (
            "--workers WORKERS processes to lint with; empty or below 1 means 1 "
            "(default: the machine's count)"
        ),
        "-h, --help show this help message and exit -",
    ):
        assert expected in text


def test_help_lists_every_rule_and_marks_the_fixers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = _help(monkeypatch, capsys)

    listed = ", ".join(
        f"{rule.name}{'*' * rule.fixable}" for rule in houselint.default_rules()
    )
    assert text.endswith(f"rules (* can fix): {listed}")
    assert "digits*" in text
    assert "dataclass," in text


def test_help_shows_the_pyproject_values_in_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A default from ``[tool.houselint]`` is what ``--help`` reports."""
    _write(
        tmp_path,
        '[tool.houselint]\nselect = "ALL"\nfix = "digits"\n'
        'exclude = ["vendor/**", """*_test.py =\n    globals"""]\nworkers = 3\n',
        name="pyproject.toml",
    )

    text = _help(monkeypatch, capsys)

    for expected in (
        "or ALL (default: ALL)",
        "or ALL (default: digits)",
        "(default: vendor/**; *_test.py = globals)",
        "(default: 3)",
    ):
        assert expected in text


def test_a_percent_in_a_default_is_shown_literally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(
        tmp_path,
        '[tool.houselint]\nexclude = ["100%(x)s/**"]\n',
        name="pyproject.toml",
    )

    assert "(default: 100%(x)s/**)" in _help(monkeypatch, capsys)


def test_workers_default_to_the_hook_sizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []

    def spying_run(
        paths: Sequence[Path],
        *,
        active: Sequence[Rule],
        fixing: frozenset[str],
        exclude: Sequence[Exclusion] = (),
        root: Path = Path(),
        workers: int = 1,
    ) -> tuple[tuple[Finding, ...], tuple[Path, ...]]:
        del paths, active, fixing, exclude, root
        seen.append(workers)
        return (), ()

    monkeypatch.setattr(houselint, "run", spying_run)
    path = _write(tmp_path, "value = 1\n")
    monkeypatch.setattr(sys, "argv", ["houselint.py", "--select", "digits", str(path)])

    assert houselint.main() == 0
    assert seen == [default_workers()]


def test_no_paths_lints_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bare ``houselint.py`` lints ``.``, as bare ``ruff check`` does."""
    _repo(tmp_path)
    monkeypatch.setattr(sys, "argv", ["houselint.py", "--select", "digits"])

    assert houselint.main() == 1

    flagged = {line.split(":")[0] for line in capsys.readouterr().out.splitlines()}
    assert flagged == {"a.py", "sub/b.pyi", "sub/deep/keep.ipynb"}


_PYPROJECT = """\
[project]
name = "x"

[tool.houselint]
select = "digits globals"
fix = "digits"
exclude = ["vendor/**", "*_test.py = globals"]
"""


def test_pyproject_supplies_the_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bare ``houselint.py FILES`` runs what ``[tool.houselint]`` names.

    Found by walking up from the cwd, and its globs are relative to the
    directory holding it, not to the cwd.
    """
    _write(tmp_path, _PYPROJECT, name="pyproject.toml")
    fixed = _write(tmp_path, "print(12345)\n", name="pkg/a.py")
    ignored = _write(tmp_path, "MAX = 500\n", name="pkg/b_test.py")
    flagged = _write(tmp_path, "MAX = 500\n", name="pkg/c.py")
    vendored = _write(tmp_path, "MAX = 500\nprint(12345)\n", name="vendor/d.py")
    monkeypatch.chdir(tmp_path / "pkg")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", *map(str, (fixed, ignored, flagged, vendored))],
    )

    assert houselint.main() == 1

    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"rewrote {fixed}"
    assert [line.split(":")[0] for line in out[1:]] == [str(flagged)]
    assert "[globals]" in out[1]
    assert fixed.read_text() == "print(12_345)\n"
    assert vendored.read_text() == "MAX = 500\nprint(12345)\n"


def test_the_command_line_overrides_pyproject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--select``/``--fix`` replace the config's lists; ``--exclude`` adds to them."""
    _write(tmp_path, _PYPROJECT, name="pyproject.toml")
    digits = _write(tmp_path, "MAX = 500\nvalue = 12345\n", name="a.py")
    skipped = _write(tmp_path, "value = 12345\n", name="skip.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "houselint.py",
            "--select",
            "digits",
            "--fix",
            "",
            "--exclude",
            "skip.py",
            str(digits),
            str(skipped),
        ],
    )

    assert houselint.main() == 1

    out = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in out] == [str(digits)]
    assert "[digits]" in out[0]
    assert digits.read_text() == "MAX = 500\nvalue = 12345\n"


def test_a_pyproject_without_the_table_is_passed_over(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nearest pyproject CONFIGURING houselint wins, as ruff finds its own."""
    _write(tmp_path, _PYPROJECT, name="pyproject.toml")
    _write(tmp_path, '[project]\nname = "inner"\n', name="inner/pyproject.toml")
    path = _write(tmp_path, "print(12345)\n", name="inner/a.py")
    monkeypatch.chdir(tmp_path / "inner")
    monkeypatch.setattr(sys, "argv", ["houselint.py", str(path)])

    assert houselint.main() == 0
    assert path.read_text() == "print(12_345)\n"


@pytest.mark.parametrize(
    ("table", "fragment"),
    [
        ('selec = "digits"\n', "unknown key(s) ['selec']"),
        ('select = ["digits"]\n', "select: expected str, got list"),
        ('select = "digits"\nexclude = ["*.py = nope"]\n', "unknown rule(s) ['nope']"),
        ('select = "digits"\nfix = "dataclass"\n', "not in --select"),
    ],
)
def test_a_bad_pyproject_table_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    table: str,
    fragment: str,
) -> None:
    """A typo in the table fails loudly; silently ignoring it would lint nothing."""
    _write(tmp_path, "[tool.houselint]\n" + table, name="pyproject.toml")
    path = _write(tmp_path, "value = 1\n")
    monkeypatch.setattr(sys, "argv", ["houselint.py", str(path)])

    with pytest.raises(SystemExit) as exit_info:
        houselint.main()

    assert exit_info.value.code == 2
    assert fragment in capsys.readouterr().err


def test_fixes_apply_in_active_order(tmp_path: Path) -> None:
    """``run`` applies fixers in ``active`` order, and each sees the last one's text.

    ``default_rules`` used to claim its tuple was a "fix-application
    order", but only one shipped rule is fixable, so the claim was
    unobservable and untested. The real guarantee is this one, and it takes
    two fixers to see it.
    """

    @dataclass(frozen=True, kw_only=True, slots=True)
    class _Append:
        """Appends its own letter to the file."""

        name: ClassVar[str] = "append"
        kinds: ClassVar[frozenset[str]] = frozenset({"python"})
        fixable: ClassVar[bool] = True

        letter: str = "a"

        def check(self, unit: Unit, /) -> Sequence[Finding]:
            """Report a missing trailing ``letter``, which ``fix`` appends."""
            if unit.source.rstrip().endswith(self.letter):
                return ()
            return (Finding(path=unit.path, line=1, rule=self.name, message="x"),)

        def fix(self, unit: Unit, /) -> str | None:
            """Append ``letter`` once."""
            if unit.source.rstrip().endswith(self.letter):
                return None
            return f"{unit.source.rstrip()}{self.letter}\n"

    path = _write(tmp_path, "x\n")

    run(
        [path],
        active=(_Append(letter="a"), _Append(letter="b")),
        fixing=frozenset({"append"}),
    )

    assert path.read_text() == "xab\n"


def test_a_fix_reports_what_it_rewrote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The hook this replaced printed ``grouped digits: <path>``.

    ``main`` discarded ``written``, so an autofix rewrote the file, printed
    nothing, and exited 0 -- leaving a dirty tree with no stated reason.
    """
    path = _write(tmp_path, "value = 12345\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["houselint.py", "--select", "digits", "--fix", "digits", str(path)],
    )

    assert houselint.main() == 0
    assert str(path) in capsys.readouterr().out


def test_a_notebook_with_a_non_object_cell_is_not_a_crash(tmp_path: Path) -> None:
    """``{"cells": [1]}`` is malformed input, not an AttributeError."""
    path = _write(tmp_path, '{"cells": [1]}', name="n.ipynb")

    findings, written = run(
        [path],
        active=(DigitsRule(),),
        fixing=frozenset({"digits"}),
    )

    assert written == ()
    assert findings == ()


def test_a_notebook_cell_with_a_non_string_source_is_not_a_crash(
    tmp_path: Path,
) -> None:
    """A cell whose ``source`` is not a list of strings must not raise."""
    path = _write(
        tmp_path,
        json.dumps({"cells": [{"cell_type": "code", "source": [1, None]}]}),
        name="n.ipynb",
    )

    findings, written = run(
        [path],
        active=(DigitsRule(),),
        fixing=frozenset({"digits"}),
    )

    assert written == ()
    assert findings == ()


def test_an_unwritable_file_is_reported_and_the_run_continues(tmp_path: Path) -> None:
    """A mid-loop PermissionError abandoned the files after it, with no record."""
    locked = _write(tmp_path, "value = 12345\n", name="locked.py")
    other = _write(tmp_path, "value = 67890\n", name="other.py")
    locked.chmod(0o444)
    try:
        findings, written = run(
            [locked, other],
            active=(DigitsRule(),),
            fixing=frozenset({"digits"}),
        )
    finally:
        locked.chmod(0o644)

    assert written == (other,)
    assert other.read_text() == "value = 67_890\n"
    assert any("locked.py" in str(f) and "cannot write" in f.message for f in findings)


def test_an_inline_ignore_anywhere_in_the_statement_suppresses(tmp_path: Path) -> None:
    """``ruff format`` moves a trailing comment to the closing paren of a wrapped.

    call.
    """
    path = _write(
        tmp_path,
        "import os\n"
        "x = os.environ.setdefault(\n"
        '    "A",\n'
        '    "/tmp",\n'
        ")  # house-ignore[tmp-literal] -- Pinned.\n"
        'y = "/tmp"  # house-ignore[other] -- Names another rule.\n',
    )
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert [f.line for f in findings] == [6]


def test_an_inline_ignore_on_a_compound_header_covers_the_header(
    tmp_path: Path,
) -> None:
    """A wrapped ``if (...):`` puts the marker after ``):``; it covers the test.

    It must NOT reach into the body: a marker on the header cannot silence a
    finding on a statement inside the block.
    """
    path = _write(
        tmp_path,
        "import os\n"
        "if (\n"
        '    os.environ.get("A")\n'
        '    == "/tmp"\n'
        "):  # house-ignore[tmp-literal] -- Pinned.\n"
        '    y = "/tmp"\n',
    )
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert [f.line for f in findings] == [6]


def test_an_inline_ignore_after_a_wrapped_decorator_covers_the_class(
    tmp_path: Path,
) -> None:
    """The formatter wraps ``@dataclass(...)``; the marker lands on its ``)`` line."""
    path = _write(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass(\n"
        "    frozen=True, slots=True\n"
        ")  # house-ignore[dataclass] -- Positional ctor.\n"
        "class A:\n    x: int\n",
    )
    findings, _ = run([path], active=(DataclassRule(),), fixing=frozenset())
    assert not findings


def test_an_inline_ignore_without_a_reason_is_a_finding(tmp_path: Path) -> None:
    path = _write(tmp_path, 'x = "/tmp"  # house-ignore[tmp-literal]\n')
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert [(f.line, f.rule) for f in findings] == [(1, "houselint")]
    assert "-- Reason." in findings[0].message


def test_a_bare_inline_ignore_is_a_finding_and_suppresses_nothing(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, 'x = "/tmp"  # house-ignore[] -- No rule named.\n')
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert sorted(f.rule for f in findings) == ["houselint", "tmp-literal"]
    assert any("names no rule" in f.message for f in findings)


def test_a_file_level_ignore_suppresses_the_listed_rules(tmp_path: Path) -> None:
    """A percent notebook is module scope by construction; one marker covers it."""
    path = _write(
        tmp_path,
        "# ---\n# jupyter:\n#   jupytext:\n#     formats: ipynb,py:percent\n# ---\n\n"
        "# %%\n"
        "# house-ignore[tmp-literal] -- Jupytext percent notebook.\n"
        "import os\n"
        'x = os.environ.setdefault("A", "/tmp")\n'
        'y = "/tmp"\n',
    )
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert not findings


def test_a_file_level_ignore_without_a_reason_is_a_finding(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "# house-ignore[tmp-literal]\n"
        "import os\n"
        'x = os.environ.setdefault("A", "/tmp")\n',
    )
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert [f.line for f in findings] == [1]
    assert "-- Reason." in findings[0].message


def test_a_superfluous_file_level_ignore_is_a_finding(tmp_path: Path) -> None:
    """A listed rule that would report nothing has rotted; say so."""
    path = _write(
        tmp_path,
        "# house-ignore[tmp-literal] -- Nothing here.\nx = 1\n",
    )
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert [f.line for f in findings] == [1]
    assert "superfluous" in findings[0].message


def test_a_superfluous_inline_ignore_is_a_finding(tmp_path: Path) -> None:
    """A marker that outlived the violation it excused is reported, not honored.

    The autofix for ``staticmethod`` rewrote a decorator but left its marker
    behind; the marker then read as a live exemption on a line that violated
    nothing. Reported as ``houselint`` so the marker cannot silence itself.
    """
    path = _write(
        tmp_path,
        'x = "fine"  # house-ignore[tmp-literal] -- Outlived its violation.\n'
        'y = "/tmp"  # house-ignore[tmp-literal] -- Still needed.\n',
    )
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert [(f.line, f.rule) for f in findings] == [(1, "houselint")]
    assert "superfluous inline ignore" in findings[0].message


def test_an_inline_ignore_for_an_inactive_rule_is_not_superfluous(
    tmp_path: Path,
) -> None:
    """Only a SELECTED rule can vouch that it reports nothing."""
    path = _write(
        tmp_path,
        'x = "fine"  # house-ignore[tmp-literal] -- Not under test here.\n',
    )
    findings, _ = run([path], active=(DigitsRule(),), fixing=frozenset())
    assert not findings


def test_an_own_line_marker_is_file_scoped_not_statement_scoped(
    tmp_path: Path,
) -> None:
    """Position decides scope, as ruff's ``noqa`` vs ``ruff: noqa`` does.

    An own-line marker above one statement covers the whole file, including a
    violation far below it; a trailing marker covers only its statement.
    """
    path = _write(
        tmp_path,
        "# house-ignore[tmp-literal] -- Whole file.\n"
        'x = "/tmp"\n\n\n'
        "def f() -> str:\n"
        '    return "/tmp/far"\n',
    )
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert not findings

    path = _write(
        tmp_path,
        'x = "/tmp"  # house-ignore[tmp-literal] -- One statement.\ny = "/tmp"\n',
        name="trailing.py",
    )
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert [f.line for f in findings] == [2]


def test_an_indented_own_line_marker_is_still_file_scoped(tmp_path: Path) -> None:
    """Inside a block the comment is indented; leading whitespace is not code."""
    path = _write(
        tmp_path,
        "def f() -> str:\n"
        "    # house-ignore[tmp-literal] -- Whole file.\n"
        '    return "/tmp"\n\n\n'
        'y = "/tmp"\n',
    )
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")
    findings, _ = run([path], active=(rule,), fixing=frozenset())
    assert not findings


def test_an_aliased_dataclass_import_is_checked(tmp_path: Path) -> None:
    """``from dataclasses import dataclass as dc`` is the same decorator."""
    findings = _dataclass_findings(
        tmp_path,
        "from dataclasses import dataclass as dc\n\n@dc\nclass A:\n    x: int\n",
    )
    assert len(findings) == 1
    assert "kw_only=True" in findings[0].message


def test_an_aliased_slotted_dataclass_reports_zero_arg_super(tmp_path: Path) -> None:
    """The slots detection must follow the same binding the decorator does."""
    findings = _dataclass_findings(
        tmp_path,
        "from dataclasses import dataclass as dc\n\n"
        "@dc(kw_only=True, slots=True)\n"
        "class A:\n"
        "    def f(self) -> int:\n"
        "        return 1\n\n"
        "@dc(kw_only=True, slots=True)\n"
        "class B(A):\n"
        "    def f(self) -> int:\n"
        "        return super().f()\n",
    )
    assert len(findings) == 1
    assert "super()" in findings[0].message


def test_an_aliased_dataclasses_module_is_checked(tmp_path: Path) -> None:
    """``import dataclasses as dcs`` then ``@dcs.dataclass`` is the same one."""
    findings = _dataclass_findings(
        tmp_path,
        "import dataclasses as dcs\n\n@dcs.dataclass\nclass A:\n    x: int\n",
    )
    assert len(findings) == 1


def test_an_unimported_name_spelled_dataclass_is_not_checked(tmp_path: Path) -> None:
    """A local ``dataclass`` that is not the decorator must not be flagged.

    The binding is what matters: matching the literal name would flag this,
    which is the same error as missing the aliased import, inverted.
    """
    assert not _dataclass_findings(
        tmp_path,
        "def dataclass(cls):\n    return cls\n\n@dataclass\nclass A:\n    x: int\n",
    )


def _rule_unit(tmp_path: Path, source: str, name: str = "mod.py") -> Unit:
    """Build a Unit for a rule fixture."""
    unit = houselint.read_unit(_write(tmp_path, source, name=name))
    assert isinstance(unit, Unit)
    return unit


def test_annotation_import_walks_all_annotation_shapes(tmp_path: Path) -> None:
    """Annotation imports account for aliases, defaults, casts, and guards."""
    unit = _rule_unit(
        tmp_path,
        "import typing\nfrom pkg import *\nfrom pkg import Thing\n"
        "if TYPE_CHECKING:\n    Thing = object\nelse:\n    fallback: Thing = 0\n"
        'value: "not valid [" = 0\n'
        "value2: Thing = 0\n"
        'def f(x: Thing = 1, y=cast("Thing", 2), z=cast("Thing", value=2)) -> Thing:\n'
        "    return x\n",
    )
    findings = houselint_rules.AnnotationImportRule(
        costs_ms={"pkg": 100},
    ).check(unit)
    assert len(findings) == 1
    assert findings[0].line == 3
    assert "pkg.Thing" in findings[0].message


def test_single_use_helper_rewrites_and_rejects_unsafe_shapes(tmp_path: Path) -> None:
    """The helper fixer substitutes safe calls and leaves unsafe calls unchanged."""
    rule = houselint_rules.SingleUseHelperRule()
    safe = _rule_unit(tmp_path, "def _add(x, y=1):\n    return x + y\n\nz = _add(2)\n")
    rewritten = rule.fix(safe)
    assert rewritten is not None
    assert "def _add" not in rewritten
    assert "z = 2 + 1" in rewritten
    unsafe = _rule_unit(
        tmp_path,
        "def _add(x):\n    return x\n\nz = _add(*items)\n",
        name="unsafe.py",
    )
    assert rule.check(unsafe) == ()
    assert rule.fix(unsafe) is None


def test_cwd_final_rewrites_path_shapes(tmp_path: Path) -> None:
    """Path expressions become the canonical Final bindings."""
    unit = _rule_unit(
        tmp_path,
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parent\n"
        "GRAND = Path(__file__).resolve().parents[2]\n"
        "HERE = Path(__file__).parent\n"
        "RAW = Path(__file__)\n",
    )
    rewritten = houselint_rules.CwdFinalRule().fix(unit)
    assert rewritten is not None
    assert "_CWD: Final" in rewritten
    assert "_THIS: Final" in rewritten
    assert "GRAND = _CWD.parents[1]" in rewritten
    assert "RAW = _THIS" in rewritten


def test_cwd_final_rewrites_every_parent_expression(tmp_path: Path) -> None:
    """CWD fixing handles resolve, parent, parents, division, and raw paths."""
    unit = _rule_unit(
        tmp_path,
        "from pathlib import Path\n"
        "A = Path(__file__).resolve()\n"
        "B = Path(__file__).resolve().parent\n"
        "C = Path(__file__).resolve().parents[1]\n"
        "D = Path(__file__).resolve().parents[2]\n"
        "E = Path(__file__).parent\n"
        "F = Path(__file__) / 'x'\n"
        "G = Path(__file__)\n",
        name="paths.py",
    )
    fixed = houselint_rules.CwdFinalRule().fix(unit)
    assert fixed is not None
    assert "A = _CWD" in fixed or "A = _THIS" in fixed
    assert "_CWD.parents[1]" in fixed
    assert "F = _CWD / 'x'" in fixed


def test_a_dotted_import_marker_keeps_the_import_and_stays_in_use(
    tmp_path: Path,
) -> None:
    """``--fix`` used to rewrite under the opt-out, then call the marker superfluous."""
    body = (
        "# house-ignore[dotted-import] -- Patched by path elsewhere.\n"
        "import a.b\nx = a.b.X\n"
    )
    path = _write(tmp_path, body)
    findings, written = run(
        [path],
        active=(houselint_rules.DottedImportRule(),),
        fixing=frozenset({"dotted-import"}),
    )
    assert (findings, written) == ((), ())
    assert path.read_text() == body


def test_import_rules_fix_an_alias_and_a_dotted_import_in_one_pass(
    tmp_path: Path,
) -> None:
    """``import-alias`` drops the alias; ``dotted-import`` shortens the rest."""
    path = _write(
        tmp_path,
        "import pkg.configgle.cli_override\n"
        "import pkg.lib.userdirs as dirs\n"
        "import numpy as np\n"
        "x = pkg.configgle.cli_override\n"
        "y = dirs.config_dir(), np\n",
    )
    findings, written = run(
        [path],
        active=(
            houselint_rules.ImportAliasRule(first_party=("pkg",)),
            houselint_rules.DottedImportRule(),
        ),
        fixing=frozenset({"import-alias", "dotted-import"}),
    )
    assert findings == ()
    assert written == (path,)
    assert path.read_text() == (
        "from pkg.configgle import cli_override\n"
        "from pkg.lib import userdirs\n"
        "import numpy as np\n"
        "x = cli_override\n"
        "y = userdirs.config_dir(), np\n"
    )


def test_import_hoist_handles_guarded_and_nested_imports(tmp_path: Path) -> None:
    """Only ordinary nested imports move; guarded imports remain in place."""
    unit = _rule_unit(
        tmp_path,
        "def f():\n"
        "    import os\n"
        "    return os.getcwd()\n"
        "if TYPE_CHECKING:\n    import typing\n"
        "try:\n    import optional\nexcept ImportError:\n    optional = None\n",
    )
    rule = houselint_rules.ImportHoistRule()
    findings = rule.check(unit)
    assert len(findings) == 1
    rewritten = rule.fix(unit)
    assert rewritten is not None
    assert rewritten.splitlines()[0] == "import os"
    assert "    import os" not in rewritten


def test_cli_shape_fix_repairs_main_and_polyglot_header(tmp_path: Path) -> None:
    """CLI fixes add returns, move flags, and repair the executable prelude."""
    rule = houselint_rules.CliShapeRule()
    path = _write(
        tmp_path,
        "import argparse\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description=__doc__)\n"
        "    parser.add_argument('--x')\n"
        "    return\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        name="tool.py",
    )
    unit = houselint.read_unit(path)
    assert isinstance(unit, Unit)
    rewritten = rule.fix(unit)
    assert rewritten is not None
    assert "def main() -> int:" in rewritten
    assert "return 0" in rewritten
    assert "_add_arguments(parser)" in rewritten
    polyglot = _rule_unit(
        tmp_path,
        "#!/bin/sh\n# ruff: noqa\n'''' 2>/dev/null #\nexec uv bad\n",
        name="poly.py",
    )
    fixed_polyglot = rule.fix(polyglot)
    assert fixed_polyglot is not None
    assert "# vim: ft=python" in fixed_polyglot
    assert "# fmt: off\n'''' 2>/dev/null #\nexec uv bad\n" in fixed_polyglot


def test_nested_import_shape_and_noqa_are_distinguished(tmp_path: Path) -> None:
    """A bare noqa suppresses one nested import, while another remains reportable."""
    unit = _rule_unit(
        tmp_path,
        "def f():\n"
        "    import os  # noqa: PLC0415 -- Deferred.\n"
        "    import sys\n"
        "    return os, sys\n",
    )
    rule = houselint_rules.ImportHoistRule()
    findings = rule.check(unit)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_dispatch_skips_unknown_suffix_and_reports_missing_path(tmp_path: Path) -> None:
    """Unknown suffixes skip silently; missing selected files report a finding."""
    unknown = _write(tmp_path, "BAD\n", name="notes.txt")
    assert houselint.read_unit(unknown) is None
    findings, written = run(
        [unknown],
        active=(_Counting(),),
        fixing=frozenset(),
    )
    assert findings == ()
    assert written == ()
    missing = tmp_path / "gone.py"
    findings, written = run(
        [missing],
        active=(_Counting(),),
        fixing=frozenset(),
    )
    assert written == ()
    assert len(findings) == 1
    assert "cannot read" in findings[0].message


def test_select_unknown_rule_names_are_usage_errors() -> None:
    """Selectors name the invalid rule and the known registry."""
    with pytest.raises(ValueError, match="unknown rule"):
        houselint.select_rules(
            (DigitsRule(),),
            select="missing",
            fix="",
            settings=[],
        )


def test_settings_and_helper_views_report_invalid_inputs() -> None:
    """Settings route before coercion, and marker helpers handle invalid trees."""
    with pytest.raises(ValueError, match="unknown rule"):
        houselint.select_rules(
            (DigitsRule(),),
            select="digits",
            fix="",
            settings=["missing.value=1"],
        )
    assert houselint._type_name("annotation") == "annotation"
    unit = Unit(path=Path("broken.py"), source="def (:\nx\n", kind="python")
    finding = Finding(path=unit.path, line=2, rule="digits", message="bad")
    assert house_ignore.suppressing_lines(unit, finding, {}) == ()


def test_config_globals_literal_and_type_shapes(tmp_path: Path) -> None:
    """Literal predicates distinguish values, aliases, and exempt calls."""
    unit = _rule_unit(
        tmp_path,
        "NEG = -1\n"
        "FLOAT = -1.5\n"
        "ALIAS = typing.Mapping[str, int]\n"
        "TYPE: TypeAlias\n"
        "UNSET: int\n"
        "obj.attr = 1\n",
    )
    findings = houselint_rules.ConfigGlobalsRule().check(unit)
    names = {f.message for f in findings}
    assert any("NEG" in message for message in names)
    assert any("FLOAT" in message for message in names)
    assert not any("ALIAS" in message for message in names)


def test_single_use_helper_rejects_edge_case_calls(tmp_path: Path) -> None:
    """Unsafe substitutions stay unchanged instead of changing evaluation order."""
    rule = houselint_rules.SingleUseHelperRule()
    cases = {
        "star.py": "def _x(a):\n    return a\ny = _x(*xs)\n",
        "missing.py": "def _x(a, b):\n    return a + b\ny = _x(1)\n",
        "empty.py": "def _x(a):\n    return a\ny = _x([])\n",
        "table.py": "def _x(a):\n    return {\n        'a': a,\n        'b': a,\n        'c': a,\n    }\ny = _x(1)\n",
    }
    for name, source in cases.items():
        unit = _rule_unit(tmp_path, source, name=name)
        assert rule.fix(unit) is None


def test_staticmethod_fix_leaves_existing_cls_and_handles_star_args(
    tmp_path: Path,
) -> None:
    """A pre-existing cls and a starred-only signature need no duplicate parameter."""
    unit = _rule_unit(
        tmp_path,
        "class A:\n"
        "    @staticmethod\n"
        "    def f(cls, *, value=1):\n"
        "        return value\n"
        "    @staticmethod\n"
        "    def g(*args):\n"
        "        return len(args)\n",
    )
    fixed = houselint_rules.StaticMethodRule().fix(unit)
    assert fixed is not None
    assert "def f(cls, *, value=1)" in fixed
    assert "def g(cls, *args)" in fixed


def test_module_side_effects_accepts_declarative_expression_shapes(
    tmp_path: Path,
) -> None:
    """Declarative unary, binary, boolean, conditional, and class calls are clean."""
    unit = _rule_unit(
        tmp_path,
        "A = -1\nB = 1 + 2\nC = True and False\nD = 1 if flag else 2\nE = Widget()\n",
    )
    assert not houselint_rules.ModuleSideEffectsRule().check(unit)


def test_xdg_rule_reports_home_lookup_forms(tmp_path: Path) -> None:
    """Both getenv and subscript HOME lookups violate the userdirs rule."""
    unit = _rule_unit(
        tmp_path,
        'import os\na = os.getenv("HOME")\nb = os.environ["HOME"]\n',
    )
    findings = houselint_rules.XdgLiteralRule().check(unit)
    assert len(findings) == 2
    assert all("userdirs" in finding.message for finding in findings)


def test_nested_try_ignores_nested_class_bodies(tmp_path: Path) -> None:
    """A try inside a class nested in a handler is outside the enclosing body scan."""
    unit = _rule_unit(
        tmp_path,
        "try:\n"
        "    class A:\n"
        "        try:\n"
        "            pass\n"
        "        except Exception:\n"
        "            pass\n"
        "except Exception:\n"
        "    pass\n",
    )
    assert not houselint_rules.NestedTryRule().check(unit)


def test_cli_polyglot_fix_repairs_short_and_bad_headers(tmp_path: Path) -> None:
    """Polyglot fixes add missing lines and preserve a valid exec fast path."""
    rule = houselint_rules.CliShapeRule()
    unit = _rule_unit(tmp_path, "#!/bin/sh\n", name="short.py")
    fixed = rule.fix(unit)
    assert fixed is not None
    assert fixed == "#!/bin/sh\n# vim: ft=python\n"


def test_rule_helpers_cover_attribute_and_unknown_call_shapes(tmp_path: Path) -> None:
    """Config and module-side-effect rules distinguish aliases from values."""
    globals_unit = _rule_unit(
        tmp_path,
        "import typing\n"
        "ROUTE: typing.Final = '/x'\n"
        "class Config(typing.Fig['Config']):\n    x: int = DEFAULT\n",
    )
    globals_findings = houselint_rules.ConfigGlobalsRule().check(globals_unit)
    assert len(globals_findings) == 1
    assert "DEFAULT" in globals_findings[0].message
    effects_unit = _rule_unit(
        tmp_path,
        "value = unknown_factory()\n",
        name="effects.py",
    )
    findings = houselint_rules.ModuleSideEffectsRule().check(effects_unit)
    assert len(findings) == 1
    assert "assignment from" in findings[0].message


def test_cli_fix_handles_invalid_and_implicit_main_returns(tmp_path: Path) -> None:
    """CLI fixing leaves invalid libraries alone and adds an implicit return."""
    rule = houselint_rules.CliShapeRule()
    invalid = _rule_unit(tmp_path, "def (:\n", name="invalid.py")
    assert rule.fix(invalid) is None
    unit = _rule_unit(
        tmp_path,
        "def main() -> int:\n    value = 1\n",
        name="main.py",
    )
    fixed = rule.fix(unit)
    assert fixed is not None
    assert "return 0" in fixed


def test_import_hoist_preserves_module_docstring(tmp_path: Path) -> None:
    """Hoisted imports follow the module docstring and remain top-level."""
    unit = _rule_unit(
        tmp_path,
        '"""Module."""\n\ndef f():\n    import os\n    return os.name\n',
    )
    fixed = houselint_rules.ImportHoistRule().fix(unit)
    assert fixed is not None
    assert fixed.startswith('"""Module."""\n\nimport os\n')


def test_lazy_import_cost_sees_an_import_under_a_runtime_if(tmp_path: Path) -> None:
    """An ``if`` that is not ``TYPE_CHECKING`` still imports eagerly."""
    unit = _rule_unit(
        tmp_path,
        "import sys\n"
        "if sys.version_info >= (3, 12):\n"
        "    import torch\n"
        "else:\n"
        "    x = 1\n\n\n"
        "def f():\n    return 1\n\n\n"
        "def g():\n    return 2\n\n\n"
        "def h():\n    return 3\n\n\n"
        "def k():\n    return torch.zeros(1)\n",
    )
    findings = houselint_rules.LazyImportCostRule().check(unit)
    assert [(f.line, f.rule) for f in findings] == [(3, "lazy-import-cost")]
    assert "'torch' costs ~1050 ms" in findings[0].message
    assert "only 1 of 4 defs" in findings[0].message


def test_single_use_helper_skips_an_undecodable_sibling(tmp_path: Path) -> None:
    """A sibling that is not UTF-8 shields nothing and does not crash the scan."""
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe\x00")
    unit = _rule_unit(tmp_path, "def _x(a):\n    return a + 1\n\n\ny = _x(2)\n")
    findings = houselint_rules.SingleUseHelperRule().check(unit)
    assert [(f.line, f.rule) for f in findings] == [(1, "single-use-helper")]
    assert "_x is a one-expression helper" in findings[0].message


@pytest.mark.parametrize(
    "source",
    [
        # A literal table the def names spans >= 3 lines.
        "def _x(a):\n    return (\n        a,\n        1,\n        2,\n    )\n\n\ny = _x(1)\n",
        # A wrapped expression holding an f-string cannot be unparsed safely.
        'def _x(a):\n    return f"{a}" + str(\n        1\n    )\n\n\ny = _x(1)\n',
        # More positional arguments than parameters.
        "def _x(a):\n    return a\n\n\ny = _x(1, 2)\n",
        # A keyword naming an already-bound parameter.
        "def _x(a):\n    return a\n\n\ny = _x(1, a=2)\n",
    ],
    ids=["literal-table", "wrapped-fstring", "extra-positional", "rebound-keyword"],
)
def test_single_use_helper_leaves_unsafe_substitutions(
    tmp_path: Path,
    source: str,
) -> None:
    """Each shape is refused by both check and fix, so no report lacks a fix."""
    unit = _rule_unit(tmp_path, source)
    rule = houselint_rules.SingleUseHelperRule()
    assert rule.check(unit) == ()
    assert rule.fix(unit) is None


def test_single_use_helper_inlines_a_dict_argument_unparenthesized(
    tmp_path: Path,
) -> None:
    """A dict display binds tightly enough to take ``.get`` directly."""
    unit = _rule_unit(
        tmp_path,
        "def _first(a):\n    return a.get(1)\n\n\ny = _first({1: 2})\n",
    )
    assert houselint_rules.SingleUseHelperRule().fix(unit) is None
    assert houselint_rules.ForwarderHelperRule().fix(unit) == "\n\ny = {1: 2}.get(1)\n"


def test_single_use_helper_inlines_into_a_def_with_locals(tmp_path: Path) -> None:
    """A caller's local bindings block only a body that reads one of them."""
    unit = _rule_unit(
        tmp_path,
        "def _inc(a):\n    return a + 1\n\n\ndef f():\n    b = 2\n    return _inc(b)\n",
    )
    assert (
        houselint_rules.SingleUseHelperRule().fix(unit)
        == "\n\ndef f():\n    b = 2\n    return b + 1\n"
    )


def test_final_inline_reports_a_binding_under_code_not_a_comment(
    tmp_path: Path,
) -> None:
    """The line above is code, so no comment block documents the binding."""
    unit = _rule_unit(
        tmp_path,
        "from typing import Final\n_A: Final = 1\nprint(_A)\n",
    )
    findings = houselint_rules.FinalInlineRule().check(unit)
    assert [(f.line, f.rule) for f in findings] == [(2, "final-inline")]
    assert "'_A' is a Final literal read at one site" in findings[0].message


def test_dataclass_imported_under_try_is_still_the_decorator(tmp_path: Path) -> None:
    """A binding inside a module-scope ``try`` reaches the class decorator."""
    findings = _dataclass_findings(
        tmp_path,
        "try:\n"
        "    from dataclasses import dataclass\n"
        "except ImportError:\n"
        "    pass\n\n\n"
        "@dataclass\n"
        "class A:\n"
        "    x: int\n",
    )
    assert [(f.line, f.message) for f in findings] == [
        (7, "@dataclass missing kw_only=True, slots=True on A"),
    ]


def test_config_globals_string_final_and_computed_callee(tmp_path: Path) -> None:
    """A quoted ``Final`` is not ``Final``; a computed callee is not a literal."""
    findings = _globals_findings(
        tmp_path,
        'X = handlers["k"](1)\nY: "Final" = 1\n',
    )
    assert [(f.line, f.rule) for f in findings] == [(2, "globals")]
    assert "module-level tunable value 'Y'" in findings[0].message


def test_digits_leaves_a_partly_grouped_literal(tmp_path: Path) -> None:
    """An underscore means the author chose the grouping; it is not redone."""
    assert _regrouped(tmp_path, "value = 1_23456\n") is None


@pytest.mark.parametrize(
    "source",
    [
        "class A:\n    cost = staticmethod(lambda: 1)\n",
        "def price(cls):\n    return 1\n\n\nclass A:\n    cost = staticmethod(price)\n",
    ],
    ids=["non-name-argument", "fn-takes-cls"],
)
def test_a_staticmethod_binding_the_forwarder_cannot_copy_is_left(
    tmp_path: Path,
    source: str,
) -> None:
    """Reported, but the forwarding classmethod would be wrong, so no rewrite."""
    unit = _rule_unit(tmp_path, source)
    rule = houselint_rules.StaticMethodRule()
    assert [f.rule for f in rule.check(unit)] == ["staticmethod"]
    assert rule.fix(unit) is None


def test_staticmethod_fix_finds_a_paren_on_a_continuation_line(
    tmp_path: Path,
) -> None:
    """A backslash-continued ``def`` header puts ``(`` on the next line."""
    unit = _rule_unit(
        tmp_path,
        "class A:\n"
        "    @staticmethod\n"
        "    def f \\\n"
        "    (*args):\n"
        "        return len(args)\n",
    )
    assert houselint_rules.StaticMethodRule().fix(unit) == (
        "class A:\n"
        "    @classmethod\n"
        "    def f \\\n"
        "    (cls, *args):\n"
        "        return len(args)\n"
    )


def test_module_side_effects_reports_a_comparison_assignment(tmp_path: Path) -> None:
    """A comparison is not one of the declarative value shapes."""
    unit = _rule_unit(tmp_path, "X = a < b\n")
    findings = houselint_rules.ModuleSideEffectsRule().check(unit)
    assert [(f.line, f.message) for f in findings] == [
        (1, "assignment from 'a < b' at module scope; move it into a function"),
    ]


_POLYGLOT_PRELUDE = (
    "#!/bin/sh\n"
    "# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.\n"
    "# fmt: off\n"
    "'''' 2>/dev/null #\n"
)
_POLYGLOT_EXEC = (
    'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync '
    'python3 "$0" "$@"\n'
)


def test_cli_fix_leaves_a_canonical_polyglot_alone(tmp_path: Path) -> None:
    """Header and trailer both already match, so nothing is rewritten."""
    unit = _rule_unit(
        tmp_path,
        _POLYGLOT_PRELUDE + _POLYGLOT_EXEC + "'''\nx = 1\n# vim: ft=python\n",
        name="tool.py",
    )
    assert houselint_rules.CliShapeRule().fix(unit) is None


def test_cli_fix_terminates_the_last_line_before_the_modeline(tmp_path: Path) -> None:
    """A file without a final newline must not glue the modeline onto code."""
    body = _POLYGLOT_PRELUDE + _POLYGLOT_EXEC + "'''\nx = 1"
    unit = _rule_unit(tmp_path, body, name="tool.py")
    assert houselint_rules.CliShapeRule().fix(unit) == body + "\n# vim: ft=python\n"


def test_cli_fix_keeps_a_custom_exec_uv_line(tmp_path: Path) -> None:
    """Any ``exec uv`` line is the author's; only a missing one is replaced."""
    exec_line = 'exec uv run python3 "$0" "$@"\n'
    body = _POLYGLOT_PRELUDE + exec_line + "'''\nx = 1\n"
    unit = _rule_unit(tmp_path, body, name="tool.py")
    assert houselint_rules.CliShapeRule().fix(unit) == body + "# vim: ft=python\n"


def test_cli_fix_of_a_vanished_file_still_returns_the_rewrite(tmp_path: Path) -> None:
    """The mode bit is best-effort; a missing path cannot block the text fix."""
    path = tmp_path / "gone.py"
    unit = Unit(path=path, source="#!/bin/sh\n", kind="python")
    assert houselint_rules.CliShapeRule().fix(unit) == "#!/bin/sh\n# vim: ft=python\n"
    assert not path.exists()


def test_cli_fix_skips_other_keywords_and_a_wrapped_description(
    tmp_path: Path,
) -> None:
    """``prog=`` is not a description, and a wrapped value is not spliced."""
    unit = _rule_unit(
        tmp_path,
        "import argparse\n\n\n"
        "def main() -> int:\n"
        "    parser = argparse.ArgumentParser(\n"
        '        prog="t", description=(__doc__ or "").split(\n'
        '            "x"\n'
        "        )[0]\n"
        "    )\n"
        "    return 0\n",
        name="tool.py",
    )
    assert houselint_rules.CliShapeRule().fix(unit) is None


def test_cwd_final_fix_reuses_existing_bindings(tmp_path: Path) -> None:
    """With ``Final`` and ``_CWD`` already bound, only the use site changes."""
    unit = _rule_unit(
        tmp_path,
        "from pathlib import Path\n"
        "from typing import Final\n"
        "_CWD: Final = Path(__file__).resolve().parent\n"
        "Y = Path(__file__).parent\n",
    )
    assert houselint_rules.CwdFinalRule().fix(unit) == (
        "from pathlib import Path\n"
        "from typing import Final\n"
        "_CWD: Final = Path(__file__).resolve().parent\n"
        "Y = _CWD\n"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("X = Path(__file__).resolve(\n).parent\n", "_CWD"),
        ("X = Path(__file__).resolve(\n).parents[2]\n", "_CWD.parents[1]"),
        ("X = (Path(__file__)\n.parent)\n", "_CWD"),
    ],
    ids=["resolve-parent", "resolve-parents", "parent"],
)
def test_cwd_final_fix_rewrites_a_wrapped_path_expression(
    tmp_path: Path,
    source: str,
    expected: str,
) -> None:
    unit = _rule_unit(tmp_path, "from pathlib import Path\n" + source)
    fixed = houselint_rules.CwdFinalRule().fix(unit)
    assert fixed is not None
    [value] = [
        stmt.value
        for stmt in ast.parse(fixed).body
        if isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "X"
    ]
    assert ast.unparse(value) == expected


def test_cwd_final_fix_rewrites_an_uncalled_resolve_attribute(tmp_path: Path) -> None:
    """``.resolve.parent`` never matches the text patterns; the parent walk does."""
    unit = _rule_unit(
        tmp_path,
        "from pathlib import Path\n"
        "X = Path(__file__).resolve.parent\n"
        "Y = Path(__file__).resolve.parents[2]\n",
    )
    assert houselint_rules.CwdFinalRule().fix(unit) == (
        "from pathlib import Path\n"
        "from typing import Final\n\n"
        "_CWD: Final = Path(__file__).resolve().parent\n\n"
        "X = _CWD\n"
        "Y = _CWD.parents[1]\n"
    )


def test_import_alias_fix_avoids_def_class_and_parameter_names(tmp_path: Path) -> None:
    """A local class named like the leaf leaves the import for the author."""
    unit = _rule_unit(
        tmp_path,
        "import codedoctor.lib.userdirs as dirs\n\n\n"
        "class userdirs:\n"
        "    def f(self, x):\n"
        "        return dirs.config_dir()\n",
    )
    assert houselint_rules.ImportAliasRule().fix(unit) is None


def test_dotted_import_fix_leaves_unrelated_and_call_rooted_attributes(
    tmp_path: Path,
) -> None:
    """Only attribute chains rooted at the shortened import are rewritten."""
    unit = _rule_unit(
        tmp_path,
        "import pkg.configgle.cli_override\n"
        "import os\n"
        "x = pkg.configgle.cli_override\n"
        "y = os.sep\n"
        "z = f().a.b\n",
    )
    assert houselint_rules.DottedImportRule().fix(unit) == (
        "from pkg.configgle import cli_override\n"
        "import os\n"
        "x = cli_override\n"
        "y = os.sep\n"
        "z = f().a.b\n"
    )


@pytest.mark.parametrize(
    ("handler", "expected"),
    [("ImportError", []), ("ValueError", [3])],
)
def test_import_hoist_honours_an_optional_import_under_a_module_if(
    tmp_path: Path,
    handler: str,
    expected: list[int],
) -> None:
    """A ``try``/``except ImportError`` nested in a module ``if`` is still optional."""
    unit = _rule_unit(
        tmp_path,
        "if flag:\n"
        "    try:\n"
        "        import optional\n"
        f"    except {handler}:\n"
        "        optional = None\n",
    )
    findings = houselint_rules.ImportHoistRule().check(unit)
    assert [f.line for f in findings] == expected


def test_import_hoist_places_the_import_after_existing_imports(
    tmp_path: Path,
) -> None:
    unit = _rule_unit(
        tmp_path,
        "import sys\n\n\ndef f():\n    import os\n    return os, sys\n",
    )
    assert houselint_rules.ImportHoistRule().fix(unit) == (
        "import sys\n\nimport os\n\n\ndef f():\n    return os, sys\n"
    )


def test_a_marker_on_the_line_after_a_statement_does_not_cover_it(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        'z = 0\nx = "/tmp"\ny = 1  # house-ignore[tmp-literal] -- Next statement.\n',
    )
    rule = next(r for r in houselint.default_rules() if r.name == "tmp-literal")

    findings, _ = run([path], active=(rule,), fixing=frozenset())

    assert [(f.line, f.rule) for f in findings] == [
        (2, "tmp-literal"),
        (3, "houselint"),
    ]


def test_a_line_one_finding_is_covered_by_a_marker_anywhere(tmp_path: Path) -> None:
    """A whole-file verdict lands on line 1; its marker may sit at the end."""
    path = _write(
        tmp_path,
        "BAD\nx = 1\ny = 2  # house-ignore[counting] -- Whole-file verdict.\n",
    )

    findings, _ = run([path], active=(_Counting(),), fixing=frozenset())

    assert findings == ()


def test_a_file_level_ignore_silences_every_finding_of_its_rule(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "# house-ignore[counting] -- Whole file.\nBAD\nBAD\n",
    )

    findings, _ = run([path], active=(_Counting(), _Trim()), fixing=frozenset())

    assert findings == ()


def test_a_rule_of_another_kind_does_not_stop_the_rules_after_it(
    tmp_path: Path,
) -> None:
    """Both loops skip a rule whose kinds exclude the file and go on."""
    notebook = _write(
        tmp_path,
        json.dumps({"cells": [{"cell_type": "code", "source": ["BAD 123456"]}]}),
        name="n.ipynb",
    )

    findings, written = run(
        [notebook],
        active=(_Counting(), DigitsRule(), _CountingAnyKind()),
        fixing=frozenset({"digits"}),
    )

    assert written == (notebook,)
    assert "123_456" in notebook.read_text()
    assert [f.rule for f in findings] == ["counting-any"]


def test_a_clean_fixer_does_not_stop_the_fixers_after_it(tmp_path: Path) -> None:
    path = _write(tmp_path, "value = 12345\n")

    _, written = run(
        [path],
        active=(_Trim(), DigitsRule()),
        fixing=frozenset({"trim", "digits"}),
    )

    assert written == (path,)
    assert path.read_text() == "value = 12_345\n"


def test_a_fully_excluded_path_reports_nothing_written(tmp_path: Path) -> None:
    path = _write(tmp_path, "value = 12345\n", name="x_test.py")

    _, written = run(
        [path],
        active=(DigitsRule(),),
        fixing=frozenset({"digits"}),
        exclude=(_exclusion("*_test.py = digits"),),
    )

    assert written == ()


def test_a_fix_is_reported_written_only_when_a_rule_rewrote(tmp_path: Path) -> None:
    clean = _write(tmp_path, "value = 1\n", name="clean.py")
    dirty = _write(tmp_path, "value = 12345\n", name="dirty.py")

    _, written = run(
        [clean, dirty],
        active=(DigitsRule(),),
        fixing=frozenset({"digits"}),
    )

    assert written == (dirty,)


def test_a_failed_write_is_one_dispatcher_finding_on_line_one(tmp_path: Path) -> None:
    locked = _write(tmp_path, "value = 12345\n", name="locked.py")
    locked.chmod(0o444)
    try:
        findings, written = run(
            [locked],
            active=(DigitsRule(),),
            fixing=frozenset({"digits"}),
        )
    finally:
        locked.chmod(0o644)

    assert written == ()
    assert [str(f) for f in findings if f.rule == "houselint"] == [
        f"{locked}:1: [houselint] cannot write: Permission denied",
    ]


def test_an_unreadable_path_is_one_finding_naming_the_os_error(tmp_path: Path) -> None:
    missing = tmp_path / "gone.py"

    assert houselint.read_unit(missing) == Finding(
        path=missing,
        line=1,
        rule="houselint",
        message="cannot read: No such file or directory",
    )
    assert houselint.read_unit(missing, reporter="other") == Finding(
        path=missing,
        line=1,
        rule="other",
        message="cannot read: No such file or directory",
    )


def test_an_undecodable_file_is_one_finding_naming_the_byte(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_bytes(b"x = 1\n# \xe9\n")

    assert houselint.read_unit(path, reporter="other") == Finding(
        path=path,
        line=1,
        rule="other",
        message="cannot decode as UTF-8: invalid continuation byte at byte 8",
    )


def test_a_fixed_file_is_written_as_utf8(tmp_path: Path) -> None:
    path = _write(tmp_path, '"""Calf\u00e9."""\nvalue = 12345\n')

    run([path], active=(DigitsRule(),), fixing=frozenset({"digits"}))

    assert path.read_bytes() == b'"""Calf\xc3\xa9."""\nvalue = 12_345\n'


def test_a_fix_error_names_the_rule_path_and_parse_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = 1\n")

    with pytest.raises(houselint.FixError) as raised:
        run([path], active=(_Breaking(),), fixing=frozenset({"breaking"}))

    assert str(raised.value) == (
        f"breaking fix produced unparseable source for {path}; fix the rule in "
        "houselint_rules.py ('(' was never closed, line 2)"
    )
    cause = raised.value.__cause__
    assert isinstance(cause, SyntaxError)
    assert cause.filename == str(path)


def test_superfluous_markers_are_dispatcher_findings_on_their_file(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "# house-ignore[counting] -- Nothing here.\n"
        "x = 1  # house-ignore[counting] -- Nothing here either.\n",
    )

    findings, _ = run([path], active=(_Counting(),), fixing=frozenset())

    assert [(f.path, f.line, f.rule) for f in findings] == [
        (path, 1, "houselint"),
        (path, 2, "houselint"),
    ]


def test_an_unset_rule_keeps_its_defaults_while_another_is_set() -> None:
    active, _ = houselint.select_rules(
        (DigitsRule(), houselint_rules.NestedDefRule()),
        select="digits nested-def",
        fix="",
        settings=["digits.min-digits=7"],
    )

    assert active == (DigitsRule(min_digits=7), houselint_rules.NestedDefRule())


def test_a_setting_value_may_hold_an_equals_and_a_dot() -> None:
    """Split at the FIRST ``=`` and the FIRST ``.``: the value is free text."""
    grouped = houselint._settings_by_rule(
        ["a.b=c=d.e", "a.f.g=1"],
        selected={"a"},
        known={"a"},
    )

    assert grouped == {"a": {"b": "c=d.e", "f.g": "1"}}


def test_fixing_an_unfixable_rule_names_it() -> None:
    with pytest.raises(ValueError, match=r"^--fix: \['dataclass'\] cannot fix; "):
        houselint.select_rules(
            (DataclassRule(),),
            select="dataclass",
            fix="dataclass",
            settings=[],
        )


def test_a_trailing_double_star_matches_only_below_its_directory() -> None:
    regex = houselint.glob_regex("vendor/**")

    assert regex.match("vendor/a/b.py")
    assert not regex.match("vendor")
    assert not regex.match("vendorx/a.py")


def test_the_pool_starts_at_twenty_paths_and_splits_one_worker_per_ten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured break-even is ~10 files per worker; one worker is just serial.

    Each worker takes about eight chunks, so a slow file pins one chunk, not a
    contiguous eighth of the paths. ``fork`` because ``forkserver`` re-sends the
    parent's argv, which an ``--all-files`` run pushes past the 128 KB limit.
    """
    pools: list[tuple[int, bool]] = []
    chunks: list[int] = []

    # Records the pool ``run`` asks for and maps in this process: the sizing is
    # what is under test, and forking five real pools only measured the fork.
    class _Recording(ProcessPoolExecutor):
        def __init__(self, max_workers: int, mp_context: BaseContext) -> None:
            pools.append((max_workers, isinstance(mp_context, ForkContext)))

        @override
        def shutdown(
            self,
            wait: bool = True,
            *,
            cancel_futures: bool = False,
        ) -> None:
            del wait, cancel_futures

        @override
        def map[T](
            self,
            fn: Callable[[Path], T],
            *iterables: Iterable[Path],
            timeout: float | None = None,
            chunksize: int = 1,
        ) -> Iterator[T]:
            del timeout
            chunks.append(chunksize)
            return map(fn, *iterables)

    monkeypatch.setattr(houselint, "ProcessPoolExecutor", _Recording)
    paths = [_write(tmp_path, "BAD\n", name=f"m{n:02}.py") for n in range(48)]

    for count, workers in ((19, 24), (20, 24), (29, 24), (32, 24), (20, 1), (48, 2)):
        findings, _ = run(
            paths[:count],
            active=(_Counting(),),
            fixing=frozenset(),
            workers=workers,
        )
        assert [f.path for f in findings] == paths[:count]
    run(paths, active=(_Counting(),), fixing=frozenset())

    assert pools == [(2, True), (2, True), (3, True), (2, True)]
    assert chunks == [1, 1, 1, 3]


def test_help_wraps_the_rule_list_at_79_columns_without_splitting_names(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``b`` would end the first line at 80 columns; it starts an indented second."""
    rules = (
        SimpleNamespace(name="a" * 57, fixable=False),
        SimpleNamespace(name="b", fixable=True),
    )
    monkeypatch.setattr(houselint, "default_rules", lambda: rules)
    monkeypatch.setattr(sys, "argv", ["houselint.py", "--help"])
    with pytest.raises(SystemExit):
        houselint.main()

    assert capsys.readouterr().out.endswith(
        "\n\nrules (* can fix): " + "a" * 57 + ",\n  b*\n",
    )


def test_a_later_check_reports_against_the_fixed_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "BAD\nz\n")

    findings, _ = run(
        [path],
        active=(_Trim(), _Counting()),
        fixing=frozenset({"trim"}),
    )

    assert [(f.path, f.line) for f in findings] == [(path, 1)]


@dataclass(frozen=True, kw_only=True, slots=True)
class _CleanFixer:
    """A fixable rule that is always clean and records each ``check``."""

    name: ClassVar[str] = "clean-fixer"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    calls: list[Path]
    """Paths ``check`` was called on, in order."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Record the call; report nothing."""
        self.calls.append(unit.path)
        return ()

    def fix(self, unit: Unit, /) -> str | None:
        """Never needed: ``check`` is always clean."""
        del unit
        return None


def test_a_fixers_check_is_reused_for_the_report(tmp_path: Path) -> None:
    """The fix pass already ran ``check`` on this exact text; it is not run twice."""
    path = _write(tmp_path, "x = 1\n")
    calls: list[Path] = []

    run([path], active=(_CleanFixer(calls=calls),), fixing=frozenset({"clean-fixer"}))

    assert calls == [path]


def test_parsing_never_sizes_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every gate passes ``--workers``; probing the machine for a default is waste."""
    sized: list[int] = []
    monkeypatch.setattr(houselint, "default_workers", lambda: sized.append(1) or 1)
    parser = argparse.ArgumentParser()
    houselint._add_arguments(parser)

    parser.parse_args(["--workers", "2"])

    assert sized == []


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
