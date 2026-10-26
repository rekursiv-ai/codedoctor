"""Houselint rules beyond the four founding ones; see ``houselint``.

Every rule here follows ``houselint.Rule`` and is registered by
``houselint.default_rules``. Split out of ``houselint`` only for size:
that module owns the dispatcher and protocol; this one owns rules that read
shared Units and say what is wrong with them.

Rules that need only ``unit.source`` prefilter on a substring before
touching ``unit.tree``, the same trade the founding rules make: most files
hold none of the marker and never pay to parse.
"""

from collections import Counter
from collections.abc import (
    Iterator,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path
from typing import ClassVar, Final, Self, cast
from typing_extensions import TypeIs

import ast
import builtins
import inspect
import itertools
import json
import os
import re
import stat
import textwrap
import tokenize

from treekle import ReadError, from_plain, json_loads

from codedoctor.lint import house_ignore
from codedoctor.lint.houselint_types import Finding, Unit, _tokenize


_USERDIRS_HELPERS = frozenset({"config_dir", "state_dir", "cache_dir", "data_dir"})

_DEF_KINDS: Final = (ast.FunctionDef, ast.AsyncFunctionDef)

_NOTEBOOK_JSON: Final = json.JSONEncoder(ensure_ascii=False, indent=1)

_BUILTIN_NAMES: Final = frozenset(dir(builtins))


@dataclass(frozen=True, kw_only=True, slots=True)
class _DataclassBindings:
    """The local names through which ``dataclasses.dataclass`` is reachable.

    Defined ahead of its caller rather than after it, against the usual
    private-follows-public order, because ``DataclassRule`` names it in a
    method signature and the annotation must already exist.

    A decorator is matched by what it is BOUND to, not by how it is spelled.
    ``from dataclasses import dataclass as dc`` makes ``@dc`` the very same
    decorator, and matching the literal name ``dataclass`` missed it: the
    class shipped without ``slots``/``kw_only`` and the rule called the file
    clean.

    Attributes:
      direct: Names bound to the decorator itself (``dataclass``, ``dc``).
      modules: Names bound to the ``dataclasses`` module, whose
        ``.dataclass`` attribute is the decorator.

    """

    direct: frozenset[str]
    modules: frozenset[str]


@dataclass(frozen=True, kw_only=True, slots=True)
class DataclassRule:
    """Require @dataclass(kw_only=True, slots=True) on every dataclass.

    Also rejects zero-arg ``super()`` inside a slotted dataclass:
    ``slots=True`` rebuilds the class, leaving the method's ``__class__``
    cell pointing at the discarded original, which raises TypeError below
    Python 3.14 (cpython#124455). Use the explicit ``super(Cls, self)``.

    Per-class opt-out via the house marker on the decorator line::

        @dataclass(slots=True)  # house-ignore[dataclass] -- Positional ctor.
    """

    name: ClassVar[str] = "dataclass"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    required: tuple[str, ...] = ("kw_only", "slots")
    """Decorator kwargs that must be present and True."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each dataclass missing a required kwarg or using bare super().

        Every finding this rule makes hangs off a ``ClassDef``, so a source
        with no ``class`` keyword can hold none, and settling that by
        substring costs ~1 ms over 422 files against the ~105 ms of parsing
        the 212 that have no class at all. The scan is a necessary
        condition only -- ``class`` also appears in strings and comments --
        so a match parses and decides properly.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not _CLASS_KEYWORD.search(unit.source):
            return ()
        tree = unit.tree
        if tree is None:
            return ()
        bound = _dataclass_bindings(tree)
        findings: list[Finding] = []
        for node in unit.nodes(ast.ClassDef):
            findings.extend(self._decorator_findings(node, unit, bound))
            if _is_slotted(node, bound):
                findings.extend(
                    Finding(
                        path=unit.path,
                        line=lineno,
                        rule=self.name,
                        message=(
                            f"zero-arg super() in slotted dataclass {node.name}; "
                            f"use super({node.name}, self) -- slots=True rebuilds "
                            "the class and the zero-arg form raises below "
                            "Python 3.14"
                        ),
                    )
                    for lineno in _zero_arg_supers(node)
                )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: which kwarg a class needs is a judgement call.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None

    def _decorator_findings(
        self,
        node: ast.ClassDef,
        unit: Unit,
        bound: _DataclassBindings,
    ) -> Sequence[Finding]:
        """Report each ``@dataclass`` decorator missing a required kwarg."""
        return [
            Finding(
                path=unit.path,
                line=dec.lineno,
                rule=self.name,
                message=(
                    f"@dataclass missing {', '.join(f'{k}=True' for k in missing)} "
                    f"on {node.name}"
                ),
            )
            for dec in node.decorator_list
            if (missing := _missing_kwargs(dec, self.required, bound))
        ]


@dataclass(frozen=True, kw_only=True, slots=True)
class StaticMethodRule:
    """Reject ``@staticmethod``; the house form is ``@classmethod``.

    ``STYLE.md``: "``@classmethod`` over ``@staticmethod``; functions over
    stateless classes." A staticmethod that never needs the class is a
    module-level function wearing a class as a namespace; one that might
    later need it cannot reach it without a signature change every caller
    sees. ``@classmethod`` costs one unused parameter and keeps both doors
    open.

    The call form ``name = staticmethod(fn)`` in a class body is the same
    construct without the ``@`` and is reported the same way. It is
    autofixed when ``fn`` is an undecorated def in the same module, whose
    signature the forwarding classmethod can copy.

    ``forward``/``backward`` of a ``torch.autograd.Function`` are exempt:
    under ``torch.compile`` a classmethod ``forward`` receives ``cls`` twice
    and raises ``TypeError`` (measured on torch 2.11).
    """

    name: ClassVar[str] = "staticmethod"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report every ``@staticmethod`` decorator and ``staticmethod(...)`` binding.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "staticmethod" not in unit.source:
            return ()
        decorated, bound = _staticmethod_sites(unit)
        return tuple(
            Finding(
                path=unit.path,
                line=line,
                rule=self.name,
                message=(
                    f"@staticmethod on {name}; use @classmethod "
                    "(STYLE.md) or a module-level function"
                ),
            )
            for line, name in sorted(
                [(dec.lineno, node.name) for dec, node in decorated]
                + [(stmt.lineno, name) for stmt, name in bound],
            )
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Rewrite each ``staticmethod`` to a ``@classmethod`` taking ``cls``.

        The rename alone is not enough -- a ``classmethod`` is called with the
        class bound to its first positional, so a body that kept the old
        parameter list would receive the class as its first real argument.
        A ``name = staticmethod(fn)`` binding becomes a classmethod that
        forwards to ``fn`` with ``fn``'s own signature; one whose ``fn`` is
        not a local undecorated def is left for the author.

        Decorator edits keep the line count, so they apply first; binding
        replacements then apply bottom-up so earlier line numbers stay valid.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        tree = unit.tree
        if tree is None or "staticmethod" not in unit.source:
            return None
        decorated, bound = _staticmethod_sites(unit)
        functions = {
            node.name: node for node in tree.body if isinstance(node, _DEF_KINDS)
        }
        forwarders = [
            (stmt, text)
            for stmt, name in bound
            if (text := _forwarder(stmt, name=name, functions=functions)) is not None
        ]
        if not decorated and not forwarders:
            return None
        lines = unit.source.splitlines(keepends=True)
        for dec, node in decorated:
            _insert_cls(lines, node)
            index = dec.lineno - 1
            lines[index] = lines[index].replace("@staticmethod", "@classmethod", 1)
        for stmt, text in sorted(
            forwarders,
            key=lambda entry: entry[0].lineno,
            reverse=True,
        ):
            lines[stmt.lineno - 1 : stmt.end_lineno or stmt.lineno] = [
                textwrap.indent(text, " " * stmt.col_offset) + "\n",
            ]
        return "".join(lines)


@dataclass(frozen=True, kw_only=True, slots=True)
class ConfigGlobalsRule:
    """Flag module-level tunable values that belong in a config.

    The project rule: a tunable value -- a model/training/task knob --
    belongs in a configgle ``Config`` field (or a frozen dataclass wrapping
    a cohesive group of knobs), never as a loose module-level constant.
    Shared values flow through ``finalize()`` / ``make()`` injection, not a
    global a Config or ``__init__`` reads. Even a value fixed for today's
    task (``ARC_MAX_GRID = 30``) is a config field: a new task would tune
    it, and the config is the single source of truth for an experiment.

    Two checks: every module-level assignment whose value is a literal
    scalar or literal collection of scalars, and every Config field whose
    default names an ALL-CAPS module constant instead of an inline literal.

    Definitionally-non-config bindings (type aliases, loggers, compiled
    regexes, registry containers) are auto-exempt, as is anything annotated
    ``Final``. Suppress a genuine exception with the house marker::

        ANSI_RESET = _ESC + "[0m"  # house-ignore[globals] -- Escape, not a knob.
    """

    name: ClassVar[str] = "globals"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    config_bases: tuple[str, ...] = ("Fig", "Makes", "Maker")
    """configgle base classes; subscripting one declares a Config."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each unsuppressed module-level tunable value.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if unit.tree is None:
            return ()
        findings: list[Finding] = []
        for stmt in unit.tree.body:
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            findings.extend(self._binding_findings(stmt, unit))
        findings.extend(self._config_field_findings(unit))
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: relocating a knob into a config is a design change.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None

    def _binding_findings(
        self,
        stmt: ast.Assign | ast.AnnAssign,
        unit: Unit,
    ) -> Sequence[Finding]:
        """Report one module-level binding that is a knob."""
        if not _is_knob(stmt):
            return ()
        return (
            Finding(
                path=unit.path,
                line=stmt.lineno,
                rule=self.name,
                message=(
                    f"module-level tunable value {_target_name(stmt)!r}; move "
                    "it into the call site. "
                    "This is a global and globals invite confusion. "
                    "Meanwhile functions' kwargs (with this as a default) "
                    "should be bearing the load of this value thus being a "
                    "stateless path for user respecfication. "
                    "(If you have to repeat this default value, that's less "
                    "bad than global.) "
                    "If you're using configgle then its particularly silly to "
                    "use a global -- the configgle Config _is_ the container "
                    "of defaults. "
                ),
            ),
        )

    def _config_field_findings(self, unit: Unit) -> Sequence[Finding]:
        """Report each Config field defaulting to a module-level constant."""
        findings: list[Finding] = []
        for node in unit.nodes(ast.ClassDef):
            if not any(_base_name(base) in self.config_bases for base in node.bases):
                continue
            for member in node.body:
                if not isinstance(member, ast.AnnAssign):
                    continue
                const = _constant_default(member)
                if const is None:
                    continue
                findings.append(
                    Finding(
                        path=unit.path,
                        line=member.lineno,
                        rule=self.name,
                        message=(
                            f"Config field {ast.unparse(member.target)!r} "
                            "defaults to module "
                            f"constant {const!r}; inline the literal so the "
                            "default lives in the Config (or suppress: "
                            "house-ignore[globals] -- Reason.)"
                        ),
                    ),
                )
        return findings


@dataclass(frozen=True, kw_only=True, slots=True)
class DigitsRule:
    """Insert underscore separators into long integer literals.

    Must run before ``ruff format``: grouping lengthens a line, which can
    change how the formatter wraps it.
    """

    name: ClassVar[str] = "digits"
    kinds: ClassVar[frozenset[str]] = frozenset({"python", "pyi", "jupyter"})
    fixable: ClassVar[bool] = True
    min_digits: int = 5
    """Shortest integer literal to group."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report the file once when any literal wants grouping.

        One finding per FILE, not per literal: the fix is mechanical and
        whole-file, so listing every literal would bury the other rules'
        findings under a rewrite nobody has to read.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if self.fix(unit) is None:
            return ()
        return (
            Finding(
                path=unit.path,
                line=1,
                rule=self.name,
                message=(
                    f"integer literal of >= {self.min_digits} digits without "
                    "underscore separators (rerun with --fix digits)"
                ),
            ),
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Return the regrouped source, or None when already grouped.

        A notebook re-tokenizes per cell: it has no single token stream, so
        ``unit.tokens`` is empty for one and the notebook path works from
        ``source`` alone.

        A run of ``min_digits`` digits ANYWHERE in the text is necessary for
        a groupable literal to exist, and a plain substring scan settles it
        far cheaper than tokenizing: 86% of real files hold no such run, and
        the scan costs 26 ms over 422 files against ~190 ms of tokenizing.
        It cannot be sufficient -- the digits may be inside a string, a
        comment, or an already-grouped literal -- so a match still tokenizes
        and the answer is unchanged either way.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        if not _digit_run(self.min_digits).search(unit.source):
            return None
        if unit.kind == "jupyter":
            return self._regroup_notebook(unit.source)
        return _regroup(unit.source, unit.tokens, self.min_digits)

    # A notebook is input, not a trusted structure: a malformed document or
    # cell is left alone for the tool that owns it, never raised out of the hook.
    def _regroup_notebook(self, source: str) -> str | None:
        """Regroup every code cell, or None when none changed."""
        try:
            notebook = json_loads(source, dict[str, object])
        except (json.JSONDecodeError, ReadError):
            return None
        try:
            cells = from_plain(notebook.get("cells"), list[object])
        except ReadError:
            return None
        before = json.dumps(cells)
        for index, raw_cell in enumerate(cells):
            try:
                cell = from_plain(raw_cell, dict[str, object])
            except ReadError:
                continue
            if cell.get("cell_type") != "code":
                continue
            text = _multiline_string(cell.get("source"))
            result = _regroup(text, tuple(_tokenize(text)), self.min_digits)
            if result is not None:
                cell["source"] = result.splitlines(keepends=True)
                cells[index] = cell
        if json.dumps(cells) == before:
            return None
        notebook["cells"] = cells
        # Jupyter's own on-disk form: one-space indent, non-ASCII kept raw.
        return _NOTEBOOK_JSON.encode(notebook) + "\n"


@dataclass(frozen=True, kw_only=True, slots=True)
class EmDashRule:
    """Reject U+2014; STYLE.md Markdown/Comments: ``--`` is the house dash."""

    name: ClassVar[str] = "emdash"
    kinds: ClassVar[frozenset[str]] = frozenset({"python", "pyi", "jupyter"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each line holding an em dash.

        A notebook stores non-ASCII as JSON escapes, so for one the escape
        spelling counts too; in a .py the escape is a string the author
        wrote on purpose (this rule's own source, for one).

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        probe = _emdash_probe(unit.kind)
        if not probe.search(unit.source):
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=n,
                rule=self.name,
                message="em dash (U+2014); write `--` (rerun with --fix emdash)",
            )
            for n, text in enumerate(unit.source.splitlines(), start=1)
            if probe.search(text)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Replace every em dash with ``--``.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        probe = _emdash_probe(unit.kind)
        if not probe.search(unit.source):
            return None
        return probe.sub("--", unit.source)


@dataclass(frozen=True, kw_only=True, slots=True)
class PytestFooterRule:
    """Require the standard ``test_main`` footer on every ``*_test.py``.

    STYLE.md Testing: the footer imports ``codedoctor.lib.testing.runner`` (the
    module), never the ``codedoctor.lib.testing`` facade, because the facade pulls
    in ``imports`` and every test file would pay for it.
    """

    name: ClassVar[str] = "test-footer"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True
    runner: str = "codedoctor.lib.testing.runner"
    """Module the footer imports ``test_main`` from; an exported package sets its own."""

    @property
    def _footer(self) -> str:
        return (
            'if __name__ == "__main__":\n'
            f"    from {self.runner} import test_main\n"
            "\n"
            "    test_main(__file__)\n"
        )

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report a missing footer or a facade import.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not _is_test_file(unit.path):
            return ()
        findings: list[Finding] = []
        facade = self.runner.rpartition(".")[0]
        for n, text in enumerate(unit.source.splitlines(), start=1):
            if f"from {facade} import test_main" in text:
                findings.append(
                    Finding(
                        path=unit.path,
                        line=n,
                        rule=self.name,
                        message=(
                            "footer imports test_main from the package facade; "
                            f"use `from {self.runner} import test_main`"
                        ),
                    ),
                )
        if not unit.source.rstrip("\n").endswith(self._footer.removesuffix("\n")):
            findings.append(
                Finding(
                    path=unit.path,
                    line=max(1, unit.source.count("\n")),
                    rule=self.name,
                    message=(
                        "test file lacks the standard footer "
                        "(STYLE.md Testing); rerun with --fix test-footer"
                    ),
                ),
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """Append the footer, replacing any existing ``__main__`` guard.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        if not _is_test_file(unit.path) or not self.check(unit):
            return None
        tree = unit.tree
        body = unit.source
        if tree is not None:
            guard = next(
                (
                    stmt
                    for stmt in tree.body
                    if isinstance(stmt, ast.If) and _is_main_guard(stmt)
                ),
                None,
            )
            if guard is not None and guard.end_lineno is not None:
                lines = body.splitlines(keepends=True)
                del lines[guard.lineno - 1 : guard.end_lineno]
                body = "".join(lines)
        return body.rstrip("\n") + "\n\n\n" + self._footer


@dataclass(frozen=True, kw_only=True, slots=True)
class PytestNamingRule:
    """Tests are ``<module>_test.py``; ``test_*.py`` only under ``tools/tests``."""

    name: ClassVar[str] = "test-naming"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report a ``test_*.py`` outside the one sanctioned directory.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not unit.path.name.startswith("test_") or unit.path.suffix != ".py":
            return ()
        parts = unit.path.resolve().parts
        if parts[-3:-1] == ("tools", "tests"):
            return ()
        stem = unit.path.name.removeprefix("test_").removesuffix(".py")
        return (
            Finding(
                path=unit.path,
                line=1,
                rule=self.name,
                message=(
                    f"test file named test_*.py; rename to {stem}_test.py "
                    "(STYLE.md Testing)"
                ),
            ),
        )

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: renaming a file is a git operation.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class XdgLiteralRule:
    """Reject re-deriving the userdirs layout (AGENTS.md Per-User Directories)."""

    name: ClassVar[str] = "xdg-literal"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report XDG literals, ``Path.home()``, and ``$HOME`` lookups.

        Docstrings are prose about the layout, not a re-derivation of it, so
        a string that is a statement of its own is skipped.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not re.search(
            _XDG_PREFIX + r"|Path\.home\(|expanduser|\"HOME\"|'HOME'",
            unit.source,
        ):
            return ()
        if unit.tree is None:
            return ()
        docstrings = {
            id(stmt.value)
            for node in unit.nodes(
                ast.Module,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
            for stmt in node.body[:1]
            if _is_docstring(stmt)
            if isinstance(stmt, ast.Expr)
        }
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=f"{reason}; go through codedoctor.lib.userdirs",
            )
            for node in unit.nodes(ast.expr)
            if id(node) not in docstrings and (reason := _xdg_violation(node))
        )

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: which helper replaces the literal is a judgement call.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class UserdirsNamespaceRule:
    """First segment after a userdirs helper is ``rekursiv-ai`` (or a vendor's)."""

    name: ClassVar[str] = "userdirs-namespace"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    vendor_namespaces: tuple[str, ...] = (
        "uv",
        "ms-playwright",
        "goa-1.0",
        "Google",
        "google-chrome",
    )
    """Other vendors' directories, sanctioned by AGENTS.md rule 1.

    ``Google`` and ``google-chrome`` are Chrome's user data dirs on macOS
    and Linux, where steno registers its native messaging host.
    """

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each ``<x>_dir() / <segment>`` whose segment is not sanctioned.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "_dir()" not in unit.source:
            return ()
        findings: list[Finding] = []
        for node in unit.nodes(ast.BinOp):
            if not (isinstance(node.op, ast.Div) and _is_userdirs_call(node.left)):
                continue
            segment = node.right
            if isinstance(segment, ast.Constant) and isinstance(segment.value, str):
                if (
                    segment.value == "rekursiv-ai"
                    or segment.value in self.vendor_namespaces
                ):
                    continue
                what = f"namespace {segment.value!r}"
            else:
                what = "non-literal namespace"
            findings.append(
                Finding(
                    path=unit.path,
                    line=node.lineno,
                    rule=self.name,
                    message=(
                        f"{what} after {_call_text(node.left)}; the first segment "
                        'must be "rekursiv-ai" (AGENTS.md Per-User Directories)'
                    ),
                ),
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class UserdirsWrapperRule:
    """No wrapper around ``<x>_dir() / leaf``: inline it at every call site."""

    name: ClassVar[str] = "userdirs-wrapper"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report one-line wrapper defs and module globals holding a userdirs path.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "_dir()" not in unit.source:
            return ()
        tree = unit.tree
        if tree is None:
            return ()
        findings: list[Finding] = []
        findings.extend(
            Finding(
                path=unit.path,
                line=stmt.lineno,
                rule=self.name,
                message="module global holds a userdirs path; build it inline at each use (AGENTS.md rule 2)",
            )
            for stmt in tree.body
            if isinstance(stmt, (ast.Assign, ast.AnnAssign))
            and _is_userdirs_path(stmt.value)
        )
        for node in unit.nodes(ast.FunctionDef, ast.AsyncFunctionDef):
            body = _without_docstring(node.body)
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and _is_userdirs_path(body[0].value)
            ):
                findings.append(
                    Finding(
                        path=unit.path,
                        line=node.lineno,
                        rule=self.name,
                        message=(
                            f"{node.name} wraps a userdirs path; inline "
                            "`<x>_dir() / ...` at each call site (AGENTS.md rule 2)"
                        ),
                    ),
                )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class CacheKwargRule:
    """Reject ``cache_dir=``/``model_dir=``/``torch.hub.set_dir`` (AGENTS.md)."""

    name: ClassVar[str] = "cache-kwarg"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each explicit cache location handed to a library.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not re.search(r"cache_dir=|model_dir=|hub\.set_dir", unit.source):
            return ()
        # A kwarg on a function THIS module defines is its own API, not a
        # library's cache override.
        local_defs = {d.name for d in unit.nodes(*_DEF_KINDS)}
        findings: list[Finding] = []
        for node in unit.nodes(ast.Call):
            if _call_text(node).rpartition(".")[2] in local_defs:
                continue
            findings.extend(
                Finding(
                    path=unit.path,
                    line=node.lineno,
                    rule=self.name,
                    message=f"explicit {kw.arg}= overrides the provisioned cache; let the library read its env var (AGENTS.md Provisioned Paths)",
                )
                for kw in node.keywords
                if kw.arg in ("cache_dir", "model_dir")
            )
            if _call_text(node).endswith("hub.set_dir"):
                findings.append(
                    Finding(
                        path=unit.path,
                        line=node.lineno,
                        rule=self.name,
                        message=(
                            "torch.hub.set_dir overrides TORCH_HOME; delete it "
                            "(AGENTS.md Provisioned Paths)"
                        ),
                    ),
                )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class LazyImportBindingRule:
    """``lazy_import`` is bound only by ``from wrapt import lazy_import``."""

    name: ClassVar[str] = "lazy-import-binding"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report aliased or attribute-spelled lazy_import.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "lazy_import" not in unit.source:
            return ()
        findings: list[Finding] = []
        for node in unit.nodes(ast.ImportFrom, ast.Attribute):
            if isinstance(node, ast.ImportFrom) and node.module == "wrapt":
                findings.extend(
                    _lazy_finding(unit, node.lineno, self.name)
                    for alias in node.names
                    if alias.name == "lazy_import" and alias.asname is not None
                )
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == "lazy_import"
                and isinstance(node.value, ast.Name)
                and node.value.id == "wrapt"
            ):
                findings.append(_lazy_finding(unit, node.lineno, self.name))
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class TmpLiteralRule:
    """Reject ``/tmp`` literals; ``tempfile`` owns temporary locations."""

    name: ClassVar[str] = "tmp-literal"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each string literal that is a ``/tmp`` path.

        The banned prefix is assembled rather than spelled so this module
        passes its own rule.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        prefix = "/" + "tmp"
        if prefix not in unit.source:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=f"literal {prefix} path; use tempfile (STYLE.md Executables)",
            )
            for node in unit.nodes(ast.Constant)
            if isinstance(node.value, str)
            and (node.value == prefix or node.value.startswith(prefix + "/"))
        )

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class TempfilePrefixRule:
    """Every tempfile allocation names itself with ``prefix=``."""

    name: ClassVar[str] = "tempfile-prefix"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each tempfile call missing ``prefix=``.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "tempfile" not in unit.source:
            return ()
        tree = unit.tree
        if tree is None:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=f"{_call_text(node)} without prefix=; name the temporary",
            )
            for node in _unprefixed_tempfile_calls(unit, tree)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Insert a filename-derived prefix into each recognized call.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        if "tempfile" not in unit.source:
            return None
        tree = unit.tree
        if tree is None:
            return None
        prefix = f'prefix="{unit.path.stem.replace("_", "-")}-"'
        starts = _cached_line_starts(unit.source)
        # Before the first keyword when there is one, so a trailing ``**kw``
        # never ends up ahead of it; else right after the last positional, so
        # a trailing comma stays last; else just inside the closing paren.
        inserts = sorted(
            (_offset(starts, kw.lineno, kw.col_offset), f"{prefix}, ")
            if (kw := next(iter(node.keywords), None)) is not None
            else (_span(starts, node.args[-1])[1], f", {prefix}")
            if node.args
            else (_offset(starts, node.end_lineno, node.end_col_offset) - 1, prefix)
            for node in _unprefixed_tempfile_calls(unit, tree)
        )
        if not inserts:
            return None
        source = unit.source
        for at, text in reversed(inserts):
            source = source[:at] + text + source[at:]
        return source


@dataclass(frozen=True, kw_only=True, slots=True)
class CwdFinalRule:
    """``Path(__file__)`` appears only as ``_CWD: Final = Path(__file__).resolve().parent``.

    A file that needs its own path (a test excluding itself from a directory
    scan) binds ``_THIS: Final = Path(__file__).resolve()`` and derives
    ``_CWD: Final = _THIS.parent`` from it. Anything else spelled with
    ``Path(__file__)`` is a violation.
    """

    name: ClassVar[str] = "cwd-final"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report every ``Path(__file__)`` not in the canonical binding.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "Path(__file__)" not in unit.source:
            return ()
        tree = unit.tree
        if tree is None:
            return ()
        canonical: set[int] = set()
        for stmt in tree.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and ast.unparse(stmt.annotation) == "Final"
                and stmt.value is not None
                and (stmt.target.id, ast.unparse(stmt.value))
                in (
                    ("_CWD", "Path(__file__).resolve().parent"),
                    ("_THIS", "Path(__file__).resolve()"),
                )
            ):
                canonical.add(stmt.lineno)
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=(
                    "Path(__file__) outside `_CWD: Final = "
                    "Path(__file__).resolve().parent`; bind _CWD once and build "
                    "subpaths from it"
                ),
            )
            for node in unit.nodes(ast.Call)
            if _call_text(node) == "Path"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "__file__"
            and node.lineno not in canonical
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Bind the module path once and rewrite mechanical path expressions.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        tree = unit.tree
        if tree is None or not (
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "pathlib"
                and any(
                    alias.name == "Path" and (alias.asname or "Path") == "Path"
                    for alias in node.names
                )
                for node in tree.body
            )
        ):
            return None
        replacements, needs_this = _cwd_replacements(unit, tree)
        if not replacements:
            return None
        source = unit.source
        for start, end, replacement in sorted(replacements, reverse=True):
            source = source[:start] + replacement + source[end:]
        return _insert_cwd_bindings(source, needs_this=needs_this)


@dataclass(frozen=True, kw_only=True, slots=True)
class ImportAliasRule:
    """Alias a first-party import only when its own name is taken (STYLE.md).

    Third-party conventions (``numpy as np``, ``torch.distributed as dist``)
    are the ecosystem's, not ours; a first-party alias has no such warrant
    and hides which module a name came from -- unless the plain name already
    means something here. Twenty-five game packages each holding a ``kernel``
    module, or a module defining its own ``render`` beside a ``render``
    submodule it needs, can only be spelled with an alias, and the
    alternative (``import a.b.kernel``) is what ``dotted-import`` exists to
    remove. Name such an alias ``<qualifier>_<leaf>`` (``ft09_kernel``).

    A name counts as taken when anything in the module binds it, in any
    scope, when it is a builtin, or when another imported module ends in it
    -- the same test ``fix`` applies, so ``check`` reports exactly the
    aliases ``fix`` would drop.
    """

    name: ClassVar[str] = "import-alias"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True
    first_party: tuple[str, ...] = ("loop", "ops")
    """Top-level packages whose imports may not be aliased."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each first-party ``import x as y`` whose plain name is free.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "import" not in unit.source or unit.tree is None:
            return ()
        imports = self._imports(unit)
        claims = self._claims(imports, _module_bindings(unit))
        return tuple(
            sorted(
                Finding(
                    path=unit.path,
                    line=node.lineno,
                    rule=self.name,
                    message=(
                        f"import {alias.name} as {alias.asname}; "
                        f"`{claims[alias][-1]}` is free here, so import it "
                        "unaliased (a first-party alias is only for a taken name)"
                    ),
                )
                for node in imports
                for alias in node.names
                if alias in claims
            ),
        )

    def _is_first_party(
        self,
        node: ast.Import | ast.ImportFrom,
        alias: ast.alias,
    ) -> bool:
        """Whether the imported module is ours: relative, or under ``first_party``."""
        if isinstance(node, ast.ImportFrom):
            if node.level:
                return True
            root = (node.module or "").partition(".")[0]
        else:
            root = alias.name.partition(".")[0]
        return root in self.first_party

    def fix(self, unit: Unit, /) -> str | None:
        """Drop first-party aliases, pointing every read at the real name.

        An alias whose leaf name is already bound -- by the module or by an
        earlier rewrite -- stays: there it is the only spelling.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        if unit.tree is None:
            return None
        imports = self._imports(unit)
        claims = self._claims(imports, _module_bindings(unit))
        # A relative alias is reported but kept: its claimed path drops the
        # dots, so ``from .mod import x as y`` would come out ``from mod``.
        for node in imports:
            if isinstance(node, ast.ImportFrom) and node.level:
                for alias in node.names:
                    claims.pop(alias, None)
        if not claims:
            return None
        starts = _line_starts(unit.source)
        renames = {
            a.asname: parts[-1] for a, parts in claims.items() if a.asname is not None
        }
        edits = [
            *_import_edits(
                unit.source,
                starts=starts,
                imports=imports,
                statements={
                    alias: [
                        f"from {'.'.join(parts[:-1])} import {parts[-1]}"
                        if len(parts) > 1
                        else f"import {'.'.join(parts)}",
                    ]
                    for alias, parts in claims.items()
                },
            ),
            *(
                (*_span(starts, node), renames[node.id])
                for node in unit.nodes(ast.Name)
                if isinstance(node.ctx, ast.Load) and node.id in renames
            ),
        ]
        source = unit.source
        for start, end, replacement in sorted(edits, reverse=True):
            source = source[:start] + replacement + source[end:]
        return source

    def _imports(self, unit: Unit) -> list[ast.Import | ast.ImportFrom]:
        """Every import statement, in source order, so first come is first served."""
        return sorted(
            [*unit.nodes(ast.Import), *unit.nodes(ast.ImportFrom)],
            key=lambda node: (node.lineno, node.col_offset),
        )

    # A leaf ending two different imports is taken for both: claiming it for the first
    # would leave the rest reading as the only ``kernel``.
    def _claims(
        self,
        imports: Sequence[ast.Import | ast.ImportFrom],
        bindings: set[str],
    ) -> dict[ast.alias, tuple[str, ...]]:
        """Map each renaming alias whose plain name is free to its dotted path."""
        paths = {
            alias: (
                *(
                    (node.module or "").split(".")
                    if isinstance(node, ast.ImportFrom)
                    else ()
                ),
                *alias.name.split("."),
            )
            for node in imports
            for alias in node.names
        }
        shared = Counter(parts[-1] for parts in set(paths.values()))
        claims: dict[ast.alias, tuple[str, ...]] = {}
        for node in imports:
            for alias in node.names:
                leaf = paths[alias][-1]
                if (
                    alias.asname in {None, leaf}
                    or leaf in bindings
                    or leaf in _BUILTIN_NAMES
                    or shared[leaf] > 1
                    or not self._is_first_party(node, alias)
                ):
                    continue
                bindings.add(leaf)
                claims[alias] = paths[alias]
        return claims


@dataclass(frozen=True, kw_only=True, slots=True)
class ImportHoistRule:
    """Move unsuppressed nested imports into the module import block."""

    name: ClassVar[str] = "import-hoist"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report imports outside the module body unless ``PLC0415`` is suppressed.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per unsuppressed nested import.

        """
        tree = unit.tree
        if tree is None:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=(
                    f"{ast.unparse(node).splitlines()[0]} is outside module scope; "
                    "add `# noqa: PLC0415` only when deferral is intentional"
                ),
            )
            for node in _nested_imports(unit)
            if not _noqa_plc0415(unit, node)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Move unsuppressed nested imports above the first module statement.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The rewritten source, or None when no import moves.

        """
        tree = unit.tree
        if tree is None:
            return None
        lines = unit.source.splitlines(keepends=True)
        # Lines move whole, so an import sharing one (``if x: import a``) stays
        # reported but unfixed: moving it would carry the other statement along.
        candidates = [
            node
            for node in _nested_imports(unit)
            if not _noqa_plc0415(unit, node)
            and not lines[node.lineno - 1][: node.col_offset].strip()
            and lines[(node.end_lineno or node.lineno) - 1][
                node.end_col_offset :
            ].strip()[:1]
            in {"", "#"}
        ]
        if not candidates:
            return None
        moved = [
            textwrap.dedent("".join(lines[node.lineno - 1 : node.end_lineno]))
            for node in candidates
        ]
        for node in sorted(
            candidates,
            key=lambda candidate: candidate.lineno,
            reverse=True,
        ):
            del lines[node.lineno - 1 : node.end_lineno]
        source = "".join(lines)
        try:
            remaining = ast.parse(source)
        except SyntaxError:
            # Removing the only statement from a compound block would leave
            # invalid Python; report the import instead of corrupting the file.
            return None
        insert_at = _module_import_end(remaining)
        # Blank lines at the seam are rebuilt, not kept: one blank line after
        # the existing block, two before the code that follows.
        prefix = "".join(lines[:insert_at]).rstrip()
        suffix = "".join(lines[insert_at:]).lstrip()
        imports = "".join(f"{text.rstrip()}\n" for text in moved)
        header = f"{prefix}\n\n" if prefix else ""
        return f"{header}{imports}\n\n{suffix}"


@dataclass(frozen=True, kw_only=True, slots=True)
class DottedImportRule:
    """Import a submodule as ``from a import b``, never as ``import a.b``.

    ``import a.b`` binds ``a``, and basedpyright proves it used by matching
    the symbol each ``a.b.x`` read resolved to. A worker near its heap cap
    empties its type cache mid-check (``Program._handleMemoryHighUsage``); the
    modules it rebuilds carry new symbols, ``Checker._isMultipartImportUnused``
    no longer finds the reads recorded before the flush, and a used import is
    reported unused -- in a file nobody touched, a different one each run.
    ``from a import b`` binds ``b`` itself, a one-name import that check never
    consults. This narrows the exposure; it does not stop the flush.

    The fix rewrites the import and every read that reaches it (``a.b.x`` to
    ``b.x``), following Python's scoping: a read in a scope that rebinds
    ``a`` -- a parameter, a comprehension variable, another function's local
    import -- is not this import's. And following its paths: ``a.b.x`` needs
    only some import of ``a`` to have run, ``b.x`` this one, so each read must
    run after it -- later in its block, or in a def body while the import is
    unconditional, or in an annotation that never runs -- unless it is the
    only import binding ``a``, when every other path failed before too. A root
    still read another way (``a.other``, ``print(a)``) gets an ``import a``
    before each rewritten import no other import of ``a`` already ran through.
    What it cannot prove safe it leaves, and the finding says why:

    - The leaf already names something where the import would be read (a
      local, a parameter beside the read, a loop variable), is a builtin, or
      ends two different imports. Rename a local that can move; otherwise
      alias the import (``from x.b import kernel as b_kernel``), which
      ``import-alias`` allows exactly when the plain name is taken.
    - ``a`` is bound another way in the import's scope, or is declared
      ``global`` or ``nonlocal``.
    - Nothing reads through ``a.b``: it is imported for a side effect, or to
      load a submodule a read through ``a`` relies on. Delete it, or say what
      it is for in a ``house-ignore`` reason.
    - ``a.b`` itself is assigned or deleted, or swapped on ``a`` at runtime
      (``setattr(a, "b", ...)``, ``patch("a.b")``): a bound ``b`` keeps the
      module it saw at import.
    - A read through ``a.b`` can run where this import has not.
    - The import is in a class body, or carries a ``house-ignore`` marker.

    A quoted annotation (``"a.b.T"``) is not parsed and keeps reading ``a``;
    basedpyright reports it if the rewrite left ``a`` unbound. Nor can a
    one-file rule see another file reaching through the module's ``a``: a
    ``patch("pkg.mod.a.b.X")`` string fails only when its test runs. Point it
    at the leaf the module now binds (``"pkg.mod.b.X"``), the same object.
    """

    name: ClassVar[str] = "dotted-import"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report every unaliased dotted import, and why the fix would leave one.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per dotted import, in source order.

        """
        if "import" not in unit.source:
            return ()
        found = _DottedImports.build(unit)
        if found is None:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=(
                    f"import {alias.name} binds {alias.name.partition('.')[0]}; "
                    f"write `from {alias.name.rpartition('.')[0]} import "
                    f"{alias.name.rpartition('.')[2]}"
                    + (
                        f" as <qualifier>_{alias.name.rpartition('.')[2]}"
                        if aliased
                        else ""
                    )
                    + "` "
                    + (
                        f"(not autofixed: {reason})"
                        if reason
                        else f"(rerun with --fix {self.name})"
                    )
                ),
            )
            for node, alias, reason, aliased in found.verdicts()
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Rewrite every dotted import ``check`` clears, until none is left.

        One rewrite can free a name another needs: ``import c.d`` binds ``c``
        until it becomes ``from c import d``, and only then may
        ``import loop.a.b.c`` become ``from loop.a.b import c``. The dispatcher
        runs each fixer once, so this repeats until a pass changes nothing.
        Every pass removes a dotted import, so it ends.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        current = unit
        while (found := _DottedImports.build(current)) is not None:
            rewritten = found.rewrite()
            if rewritten is None:
                break
            current = Unit(path=unit.path, source=rewritten, kind=unit.kind)
        return None if current is unit else current.source


@dataclass(frozen=True, kw_only=True, slots=True)
class NestedDefRule:
    """No nested function longer than ``max_lines`` (STYLE.md Function Design).

    A nested def that is only CALLED by its enclosing function is a helper
    hiding from the module namespace; hoist it. One that escapes -- returned,
    passed as a callback, decorated -- is a closure, and hoisting it means
    threading every captured variable through a parameter list or a class,
    which STYLE.md's inlining preference does not ask for.
    """

    name: ClassVar[str] = "nested-def"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    max_lines: int = 3
    """Longest nested def tolerated, decorators excluded."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each nested def whose body span exceeds the limit.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        findings: list[Finding] = []
        for outer in unit.nodes(*_DEF_KINDS):
            for inner in _direct_nested_defs(outer):
                span = (inner.end_lineno or inner.lineno) - inner.lineno + 1
                if span > self.max_lines and not _escapes(inner, outer):
                    findings.append(
                        Finding(
                            path=unit.path,
                            line=inner.lineno,
                            rule=self.name,
                            message=(
                                f"nested function {inner.name} spans {span} lines "
                                f"(> {self.max_lines}); hoist it to module level"
                            ),
                        ),
                    )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: hoisting changes closure semantics.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class GetLoggerNameRule:
    """A module's OWN logger is ``getLogger(__name__)`` (STYLE.md Code Organization).

    The rule is about the logger a module binds for itself, so it fires on
    a module-level ``logger = getLogger(<not __name__>)``. Reaching for the
    root logger (``getLogger()``) or another library's logger by name inside
    a function is configuration, not identity, and is left alone.
    """

    name: ClassVar[str] = "getlogger-name"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each module-level logger bound to a name other than ``__name__``.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "getLogger" not in unit.source:
            return ()
        tree = unit.tree
        if tree is None:
            return ()
        findings: list[Finding] = []
        for stmt in tree.body:
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
                continue
            call = stmt.value
            if not (
                isinstance(call, ast.Call) and _call_text(call).endswith("getLogger")
            ):
                continue
            if (
                len(call.args) == 1
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "__name__"
            ):
                continue
            findings.append(
                Finding(
                    path=unit.path,
                    line=stmt.lineno,
                    rule=self.name,
                    message="module logger is not getLogger(__name__); loggers are per-module",
                ),
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class NestedTryRule:
    """Sequential try blocks, never nested (STYLE.md Error Handling)."""

    name: ClassVar[str] = "nested-try"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each ``try``/``except`` inside another ``try``/``except`` body.

        A ``try``/``finally`` is resource scoping, not error handling; one
        wrapping (or wrapped by) a handler is the cleanup-plus-handling shape
        and is left alone. Two handlers nested is what STYLE.md forbids.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "try:" not in unit.source:
            return ()
        findings: list[Finding] = []
        for outer in unit.nodes(ast.Try):
            if not outer.handlers:
                continue
            findings.extend(
                Finding(
                    path=unit.path,
                    line=inner.lineno,
                    rule=self.name,
                    message=(
                        "try nested inside a try body; make them sequential "
                        "(STYLE.md Error Handling)"
                    ),
                )
                for stmt in outer.body
                for inner in _statements_in(stmt)
                if isinstance(inner, ast.Try) and inner.handlers
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class NegativeBoolParamRule:
    """Affirmative boolean names (STYLE.md Function Design)."""

    name: ClassVar[str] = "negative-bool-param"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    prefixes: tuple[str, ...] = ("no_", "not_", "without_", "dont_")
    """Name prefixes that negate.

    ``skip_``/``disable_`` are verbs: ``skip_missing=True`` DOES skip, which is
    the "True = do it" reading STYLE.md asks for, so they are not listed.
    """

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each ``bool`` parameter with a negating prefix.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "bool" not in unit.source:
            return ()
        findings: list[Finding] = []
        for node in unit.nodes(*_DEF_KINDS):
            findings.extend(
                Finding(
                    path=unit.path,
                    line=arg.lineno,
                    rule=self.name,
                    message=f"negative boolean parameter {arg.arg}; name the affirmative (True = do it)",
                )
                for arg in _all_args(node)
                if arg.annotation is not None
                and ast.unparse(arg.annotation) == "bool"
                and arg.arg.startswith(self.prefixes)
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: renaming a parameter has call sites.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class AllReexportRule:
    """A non-``__init__`` ``__all__`` lists only names the module defines."""

    name: ClassVar[str] = "all-reexport"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each imported name re-exported through ``__all__``.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "__all__" not in unit.source or unit.path.name in (
            "__init__.py",
            "conftest.py",
        ):
            return ()
        tree = unit.tree
        if tree is None:
            return ()
        imported = {
            (alias.asname or alias.name.rpartition(".")[2])
            for node in unit.nodes(ast.Import, ast.ImportFrom)
            for alias in node.names
        }
        findings: list[Finding] = []
        for stmt in tree.body:
            if not (
                isinstance(stmt, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets
                )
                and isinstance(stmt.value, (ast.List, ast.Tuple))
            ):
                continue
            findings.extend(
                Finding(
                    path=unit.path,
                    line=elt.lineno,
                    rule=self.name,
                    message=f"__all__ re-exports imported {elt.value!r}; only a package __init__ re-exports (STYLE.md)",
                )
                for elt in stmt.value.elts
                if isinstance(elt, ast.Constant) and elt.value in imported
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class InitFacadeRule:
    """A package ``__init__`` holds only a docstring, imports, and ``__all__``."""

    name: ClassVar[str] = "init-facade"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each statement an ``__init__`` may not hold.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if unit.path.name != "__init__.py":
            return ()
        tree = unit.tree
        if tree is None:
            return ()
        findings: list[Finding] = []
        for index, stmt in enumerate(tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            if index == 0 and _is_docstring(stmt):
                continue
            # ``__version__`` is packaging metadata the release doc reads from
            # here; it is the one dunder a facade may bind besides ``__all__``.
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("__all__", "__version__")
                for t in stmt.targets
            ):
                continue
            if isinstance(stmt, ast.If) and "TYPE_CHECKING" in ast.unparse(stmt.test):
                continue
            findings.append(
                Finding(
                    path=unit.path,
                    line=stmt.lineno,
                    rule=self.name,
                    message=(
                        f"{type(stmt).__name__} in __init__.py; a package facade "
                        "holds only docstring, imports, __all__ (STYLE.md)"
                    ),
                ),
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class DefOrderRule:
    """Public API before private helpers (STYLE.md Code Organization)."""

    name: ClassVar[str] = "def-order"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each private def/class a public one follows without needing it.

        A private definition may precede a public one only when something
        after it needs the name at DEFINITION time -- a base, decorator,
        signature, class-attribute default, or module-level statement. A
        reference from inside a function body resolves at call time and is
        no excuse.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        tree = unit.tree
        if tree is None:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=stmt.lineno,
                rule=self.name,
                message=(
                    f"private {stmt.name} precedes public {public}, which does "
                    "not need it at definition time; private members go at the "
                    "end of the module"
                ),
            )
            for stmt, public in _misplaced_privates(tree)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Move each misplaced private def, with its leading comment, to the end.

        Order among the moved definitions is preserved. They land after the
        last statement, or before a trailing ``if __name__`` guard.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        tree = unit.tree
        if tree is None:
            return None
        moved = [stmt for stmt, _ in _misplaced_privates(tree)]
        if not moved:
            return None
        lines = unit.source.splitlines(keepends=True)
        spans = [_def_span(stmt, lines) for stmt in moved]
        chunks = ["".join(lines[start:end]) for start, end in spans]
        tail_at = len(lines)
        last = tree.body[-1]
        if isinstance(last, ast.If) and _is_main_guard(last):
            tail_at = _def_span(last, lines)[0]
        # Every moved private precedes the tail: a private is misplaced only
        # when a public def follows it, and the guard, when present, is last.
        for start, end in sorted(spans, reverse=True):
            del lines[start:end]
            tail_at -= end - start
        # Two blank lines between top-level definitions, as the formatter wants;
        # deleting a span leaves the blanks on both sides of it behind.
        head = re.sub(r"\n{4,}", "\n\n\n", "".join(lines[:tail_at])).rstrip("\n")
        body = "".join("\n\n\n" + chunk.strip("\n") for chunk in chunks)
        # `rest` starts at the guard's first line, never at a blank one.
        rest = "".join(lines[tail_at:])
        return head + body + "\n" + ("\n\n" + rest if rest else "")


@dataclass(frozen=True, kw_only=True, slots=True)
class ModuleSideEffectsRule:
    """Module scope declares; it does not act (STYLE.md Code Organization).

    Two module-level shapes are declarations even though they run code:

    - An ``assert`` or ``if ...: raise`` over module constants. That is a
      consistency check between two tables in the same file, and STYLE.md
      asks for it at import, where the author sees it, not at first use.
    - A call to a PRIVATE function defined in this module whose result is
      bound to a constant (``TABLE = _build_table()``) or that installs into
      a table this module owns (``_bind_hooks()``). The work is declarative
      -- it derives one table from another -- and the function exists only
      so the derivation has a name and a docstring.

    Environment side effects (``mpl.use``, ``os.environ[...] =``, network,
    filesystem) are the target, and none of those shapes is exempt.
    """

    name: ClassVar[str] = "module-side-effects"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    declarative_calls: tuple[str, ...] = (
        "getLogger",
        "compile",
        "TypeVar",
        "ParamSpec",
        "NewType",
        "TypeAliasType",
        "ContextVar",
        "Lock",
        "RLock",
        "APIRouter",
        "partial",
        "UUID",
        "Option",
        "lazy_import",
        "frozenset",
        "tuple",
        "dict",
        "set",
        "list",
        "Path",
        "namedtuple",
        "NamedTuple",
        "MappingProxyType",
        "field",
        "cast",
        "object",
        "deque",
        "defaultdict",
        "Counter",
        "OrderedDict",
        "import_module",
        "mark",
        "skipif",
        "usefixtures",
        "fixture",
        "param",
        "parametrize",
        "datetime",
        "timedelta",
        "date",
        "Enum",
        "auto",
        "Config",
        "dataclass",
        "Final",
        "join",
        "len",
        "maketrans",
        "fromkeys",
        "sorted",
        "range",
        "default_rng",
        "count",
    )
    """Callee names whose module-level call builds a value rather than acting."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each module-scope statement that acts.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        tree = unit.tree
        if tree is None:
            return ()
        findings: list[Finding] = []
        local_defs = {
            d.name
            for d in tree.body
            if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef))
            and d.name.startswith("_")
        }
        for index, stmt in enumerate(tree.body):
            # A bare string right after an assignment is that global's
            # attribute docstring, which STYLE.md sanctions.
            after_assign = index != 0 and isinstance(
                tree.body[index - 1],
                (ast.Assign, ast.AnnAssign, ast.TypeAlias),
            )
            if after_assign and _is_docstring(stmt):
                continue
            reason = self._violation(stmt, first=index == 0, local_defs=local_defs)
            if reason:
                findings.append(
                    Finding(
                        path=unit.path,
                        line=stmt.lineno,
                        rule=self.name,
                        message=f"{reason} at module scope; move it into a function",
                    ),
                )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None

    def _violation(
        self,
        stmt: ast.stmt,
        *,
        first: bool,
        local_defs: frozenset[str] | set[str],
    ) -> str:
        """Name the offence, or empty when the statement is declarative."""
        if isinstance(
            stmt,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.TypeAlias,
                ast.Assert,
            ),
        ):
            return ""
        if first and _is_docstring(stmt):
            return ""
        if isinstance(stmt, ast.If):
            test = ast.unparse(stmt.test)
            if "TYPE_CHECKING" in test or _is_main_guard(stmt):
                return ""
            # ``if <constant check>: raise`` is an import-time consistency
            # assertion that survives ``-O``; same standing as ``assert``.
            if all(isinstance(s, ast.Raise) for s in stmt.body) and not stmt.orelse:
                return ""
            # ``if sys.platform == ...:`` choosing between two defs is
            # declaring which implementation this platform gets.
            if all(
                isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for s in (*stmt.body, *stmt.orelse)
            ):
                return ""
            return "runtime if"
        if isinstance(stmt, ast.Try):
            # ``try: from x import y`` / ``except ImportError: ...`` is an
            # optional-dependency binding; the handler's fallback assignment,
            # re-raise with a better message, or ``TYPE_CHECKING`` split are
            # all the fallback half of that import.
            if all(
                isinstance(s, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
                for s in stmt.body
            ) and all(
                isinstance(s, (ast.Assign, ast.AnnAssign, ast.Pass, ast.Raise, ast.If))
                for h in stmt.handlers
                for s in h.body
            ):
                return ""
            return "try"
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            return self._assign_violation(stmt, local_defs)
        if isinstance(stmt, ast.Expr):
            if (
                isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id in local_defs
                and not stmt.value.args
                and not stmt.value.keywords
            ):
                return ""
            return f"expression {ast.unparse(stmt.value)[:40]!r}"
        return type(stmt).__name__.lower()

    def _assign_violation(
        self,
        stmt: ast.Assign | ast.AnnAssign,
        local_defs: frozenset[str] | set[str],
    ) -> str:
        """Name the offence in a module-level assignment, or empty."""
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if not all(isinstance(t, ast.Name) for t in targets):
            return "attribute/subscript assignment"
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if names & {"__all__", "logger", "log", "_log"}:
            return ""
        if isinstance(stmt, ast.AnnAssign):
            ann = ast.unparse(stmt.annotation)
            if ann.split("[")[0].rpartition(".")[2] in (
                "Final",
                "TypeAlias",
                "ClassVar",
            ):
                return ""
        value = stmt.value
        if value is None or _is_declarative_value(value, self.declarative_calls):
            return ""
        # ``TABLE = _derive_table(OTHER)``: a private local builder deriving
        # one constant from others is the named form of a comprehension.
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in local_defs
            and all(
                _is_declarative_value(a, self.declarative_calls)
                for a in (*value.args, *(k.value for k in value.keywords))
            )
        ):
            return ""
        return f"assignment from {ast.unparse(value)[:40]!r}"


_XDG_PREFIX: Final = "XDG" + "_"
_TEMPFILE_CALLS = frozenset(
    {
        "tempfile.mkdtemp",
        "tempfile.mkstemp",
        "tempfile.TemporaryDirectory",
        "tempfile.NamedTemporaryFile",
    },
)


_FLIPPED: dict[type[ast.cmpop], str] = {
    ast.Lt: ">=",
    ast.LtE: ">",
    ast.Gt: "<=",
    ast.GtE: "<",
}

# Every single-operator comparison and its negation as an AST operator class.
# The ordering four mirror ``_FLIPPED``, which ``negated-compare`` shares.
_INVERTED: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}

_POLYGLOT_HEADER: Final = (
    "# fmt: off",
    "'''' 2>/dev/null #",
    'exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"',
)
_EXEC_LINE = re.compile(
    r'^exec uv --quiet --project "\$\(dirname "\$0"\)" run --frozen --no-sync\s+'
    r'python3 (?:"\$0"|-m [\w.]+) "\$@"$',
)


@dataclass(frozen=True, kw_only=True, slots=True)
class UnderscoreParamRule:
    """No ``_``-prefixed parameters on a ``def``: keep the real name and ``del`` it.

    A lambda has no body to ``del`` in, so ``lambda _path, _obj: True`` is the
    only way to say "ignored" there and is exempt.
    """

    name: ClassVar[str] = "underscore-param"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each ``_name`` parameter.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not _UNDERSCORE_PARAM_PROBE.search(unit.source):
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=arg.lineno,
                rule=self.name,
                message=(
                    f"parameter {arg.arg} is underscore-prefixed; keep the name "
                    f"{arg.arg.lstrip('_')} and `del` it (STYLE.md Deprecation)"
                ),
            )
            for node in unit.nodes(*_DEF_KINDS)
            if not _is_triton_kernel(node)
            for arg in _params(node)
            if _is_underscored(arg.arg)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Rename unused ``_name`` params to ``name`` and ``del`` them.

        Only a def whose body never mentions the name is rewritten; a used
        one is reported and left, since renaming would change references. So
        is a def whose body shares the header line (``def f(_x): ...``): there
        is no line for the ``del`` to go on.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        if not _UNDERSCORE_PARAM_PROBE.search(unit.source):
            return None
        starts = _cached_line_starts(unit.source)
        edits: list[tuple[int, int, str]] = []
        for node in unit.nodes(*_DEF_KINDS):
            first = node.body[0]
            if (
                _is_triton_kernel(node)
                or unit.lines[first.lineno - 1].encode()[: first.col_offset].strip()
            ):
                continue
            used = {
                n.id for s in node.body for n in ast.walk(s) if isinstance(n, ast.Name)
            }
            params = {x.arg for x in _params(node)}
            targets = [
                a
                for a in _params(node)
                if _is_underscored(a.arg)
                and a.arg not in used
                and a.arg.lstrip("_") not in used | params
            ]
            if not targets:
                continue
            edits.extend(
                (
                    start := _offset(starts, a.lineno, a.col_offset),
                    start + len(a.arg),
                    a.arg.lstrip("_"),
                )
                for a in targets
            )
            body = _without_docstring(node.body)
            names = ", ".join(a.arg.lstrip("_") for a in targets)
            if body:
                # `del` is a statement, so a placeholder `pass` becomes dead
                # weight that ruff PIE790 then reports; it is replaced outright.
                lone_pass = len(body) == 1 and isinstance(body[0], ast.Pass)
                # A decorated def starts at its first ``@``, not at ``def``.
                decorators = (
                    body[0].decorator_list
                    if isinstance(body[0], (*_DEF_KINDS, ast.ClassDef))
                    else []
                )
                first_line = min(
                    (d.lineno for d in decorators),
                    default=body[0].lineno,
                )
                at = _offset(starts, first_line, body[0].col_offset)
                end = _offset(starts, body[0].end_lineno, body[0].end_col_offset)
                indent = " " * body[0].col_offset
                edits.append(
                    (at, end, f"del {names}")
                    if lone_pass
                    else (at, at, f"del {names}\n{indent}"),
                )
            else:
                doc = node.body[0]
                at = _offset(starts, doc.end_lineno, doc.end_col_offset)
                edits.append((at, at, f"\n{' ' * doc.col_offset}del {names}"))
        if not edits:
            return None
        source = unit.source
        for start, end, text in sorted(edits, reverse=True):
            source = source[:start] + text + source[end:]
        return source


@dataclass(frozen=True, kw_only=True, slots=True)
class NegatedCompareRule:
    """``x < 0 or x > 1``, never ``not 0 <= x <= 1`` (STYLE.md Code Style)."""

    name: ClassVar[str] = "negated-compare"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each ``not a <op> b`` ordering comparison.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "not " not in unit.source:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=(
                    f"negated comparison `{ast.unparse(node)}`; flip the operator "
                    "(single) or split into `a < lo or a > hi` (chained); "
                    "the negated form also rejects NaN, so prepend "
                    "`math.isnan(a) or` if NaN can reach here"
                ),
            )
            for node in unit.nodes(ast.UnaryOp)
            if _is_negated_ordering(node)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Flip single-comparator negations in place; chains are reported only.

        Only a comparison against a NUMERIC LITERAL is flipped: ``not x <= y``
        on two sets is subset, whose negation is not ``x > y``, and on two
        floats the flip drops the NaN rejection STYLE.md wants spelled as
        ``math.isnan``. A literal bound tells us the operand is a number, and
        a single-op flip cannot lose a NaN case there because ``not x < 0``
        and ``x >= 0`` agree on every non-NaN input -- the NaN case is
        reported, not rewritten.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        if "not " not in unit.source:
            return None
        # Unparsed as one node so an operand binding looser than the comparison
        # (``not (a or b) < 0``) keeps its parentheses.
        return _splice_nodes(
            unit,
            {
                n: ast.unparse(
                    ast.Compare(
                        left=n.operand.left,
                        ops=[_INVERTED[type(n.operand.ops[0])]()],
                        comparators=n.operand.comparators,
                    ),
                )
                for n in unit.nodes(ast.UnaryOp)
                if _is_negated_ordering(n)
                and isinstance(n.operand, ast.Compare)
                and len(n.operand.ops) == 1
                and n.lineno == n.end_lineno
                and _is_int_literal(n.operand.comparators[0])
            },
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class NegatedTernaryRule:
    """``bar if x else foo``, never ``foo if not x else bar``."""

    name: ClassVar[str] = "negated-ternary"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each ``A if not C else B``.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "if not " not in unit.source:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message="negated ternary; swap the arms and drop the `not`",
            )
            for node in unit.nodes(ast.IfExp)
            if _is_negated_ternary(node)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Swap the arms of single-line negated ternaries.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        if "if not " not in unit.source:
            return None
        # Unparsed as one node: a ternary or lambda arm moved to the front
        # needs parentheses the original position did not.
        return _splice_nodes(
            unit,
            {
                n: ast.unparse(
                    ast.IfExp(test=n.test.operand, body=n.orelse, orelse=n.body),
                )
                for n in unit.nodes(ast.IfExp)
                if _is_negated_ternary(n)
                and isinstance(n.test, ast.UnaryOp)
                and n.lineno == n.end_lineno
            },
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class SuperOnlyRule:
    """No ``super()``-only overrides; rely on inheritance (STYLE.md)."""

    name: ClassVar[str] = "super-only"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each method whose body only forwards to ``super()``.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "super(" not in unit.source:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=f"{node.name} only calls super().{node.name}; delete the override",
            )
            for node in unit.nodes(ast.FunctionDef, ast.AsyncFunctionDef)
            if _is_super_only(node)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Delete each super-only method, decorators included.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        if "super(" not in unit.source:
            return None
        found = [
            n
            for n in unit.nodes(ast.FunctionDef, ast.AsyncFunctionDef)
            if _is_super_only(n)
        ]
        # Deleting every statement of a class body leaves ``class A(B):`` with
        # no body, a SyntaxError; such a class keeps its overrides and stays
        # reported for the author to give it a body.
        emptied = {
            id(parent)
            for n in found
            if isinstance(parent := unit.parents.get(n), ast.ClassDef)
            and all(s in found for s in parent.body)
        }
        edits = [n for n in found if id(unit.parents.get(n)) not in emptied]
        if not edits:
            return None
        lines = unit.source.splitlines(keepends=True)
        for node in sorted(edits, key=lambda n: -n.lineno):
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)]) - 1
            end = node.end_lineno or node.lineno
            while end < len(lines) and not lines[end].strip():
                end += 1
            del lines[start:end]
        return "".join(lines)


@dataclass(frozen=True, kw_only=True, slots=True)
class TensorValueGuardRule:
    """Never inspect tensor VALUES to validate (STYLE.md Error Handling)."""

    name: ClassVar[str] = "tensor-value-guard"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    methods: tuple[str, ...] = ("item", "all", "any")
    """Method names whose call in a guard synchronizes the device."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each VALIDATION guard whose test calls a synchronizing method.

        A validation guard is an ``assert`` or an ``if`` that raises. A branch
        that acts on the values (``if done.any(): restart()``) is the
        algorithm, not a precondition, and is left alone.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not re.search(r"\.(item|all|any)\(\)", unit.source):
            return ()
        findings: list[Finding] = []
        # A ``while`` or a ternary is never a precondition; only these two are.
        for node in unit.nodes(ast.If, ast.Assert):
            if not _is_validation_guard(node):
                continue
            test = node.test
            findings.extend(
                Finding(
                    path=unit.path,
                    line=sub.lineno,
                    rule=self.name,
                    message=f".{sub.func.attr}() inside a guard synchronizes the device; let out-of-domain input yield NaN/inf (STYLE.md Error Handling)",
                )
                for sub in ast.walk(test)
                if isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and (sub.func.attr in self.methods)
                and (not sub.args)
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class MkdirModeRule:
    """Secrets are 0o700/0o600; scratch takes the umask; umask is restored."""

    name: ClassVar[str] = "mkdir-mode"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each mode violation (see the four cases in the class doc).

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not re.search(
            r"mode=|umask|\.chmod\(|\.mkdir\(|atomic_write_bytes\(",
            unit.source,
        ):
            return ()
        findings = [
            finding
            for node in unit.nodes(ast.Try)
            for finding in self._umask_findings(node, unit)
        ]
        findings.extend(
            Finding(path=unit.path, line=node.lineno, rule=self.name, message=reason)
            for node in unit.nodes(ast.Call)
            if (reason := _mode_violation(node))
        )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None

    def _umask_findings(self, node: ast.Try, unit: Unit) -> Sequence[Finding]:
        """Report ``os.umask`` calls in a try whose finally does not restore."""
        restores = any(
            isinstance(c, ast.Call) and _call_text(c).endswith("umask")
            for s in node.finalbody
            for c in ast.walk(s)
        )
        if restores:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=c.lineno,
                rule=self.name,
                message="os.umask without a finally that restores the previous mask",
            )
            for s in node.body
            for c in ast.walk(s)
            if isinstance(c, ast.Call) and _call_text(c).endswith("umask")
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class ReturnsNamedRule:
    """``Returns:``/``Yields:`` entries are ``name: description`` (STYLE.md)."""

    name: ClassVar[str] = "returns-named"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each unnamed entry under a public def's Returns/Yields.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not re.search(r"(Returns|Yields):", unit.source):
            return ()
        findings: list[Finding] = []
        for node in _public_defs(unit):
            for section, entries in _sections(ast.get_docstring(node) or ""):
                if section not in ("Returns", "Yields"):
                    continue
                findings.extend(
                    Finding(
                        path=unit.path,
                        line=node.lineno,
                        rule=self.name,
                        message=f"{node.name}: {section}: entry {entry[:40]!r} is not `name: description`",
                    )
                    for entry in entries
                    if not re.match(r"^\w+:\s", entry)
                )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class ArgsSectionRule:
    """A public def with parameters documents them under ``Args:``.

    A short def is exempt: at ``max_oneliner_body`` body lines or fewer the
    signature IS the documentation, and a section per parameter would say
    ``x: X.`` back to a reader who can see ``x`` two lines down. A def that
    long or shorter may carry a one-line docstring; one that already carries
    sections is held to them.
    """

    name: ClassVar[str] = "args-section"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    max_oneliner_body: int = 5
    """Body lines (docstring excluded) at or below which a one-line docstring
    needs no sections."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each public documented def with parameters but no ``Args:``.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        findings: list[Finding] = []
        for node, present in _sectioned_defs(unit, self.max_oneliner_body):
            has_params = any(a.arg not in ("self", "cls") for a in _all_args(node))
            has_params = (
                has_params
                or node.args.vararg is not None
                or node.args.kwarg is not None
            )
            if has_params and "Args" not in present:
                findings.append(
                    Finding(
                        path=unit.path,
                        line=node.lineno,
                        rule=self.name,
                        message=f"{node.name} docstring lacks Args:",
                    ),
                )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class ReturnsSectionRule:
    """A public def returning a value documents it under ``Returns:``.

    A generator documents its output under ``Yields:`` instead. The same
    short-def exemption as ``args-section`` applies. Separate from that rule
    so a tree can be exempt from one and not the other: an experiment factory
    returning ``TrainLoop.Config`` has nothing to say under ``Returns:`` that
    its annotation does not, while its parameters still want ``Args:``.
    """

    name: ClassVar[str] = "returns-section"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    max_oneliner_body: int = 5
    """Body lines (docstring excluded) at or below which a one-line docstring
    needs no sections."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each public documented def missing ``Returns:`` or ``Yields:``.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        findings: list[Finding] = []
        for node, present in _sectioned_defs(unit, self.max_oneliner_body):
            is_gen = any(
                isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node)
            )
            if is_gen:
                missing = "Yields" not in present
            else:
                missing = (
                    node.returns is not None
                    and ast.unparse(node.returns) not in ("None", "NoReturn", "Never")
                    and "Returns" not in present
                    # A stub declaring a generator type documents its output
                    # under Yields, as the concrete implementation will.
                    and "Yields" not in present
                    and not _has_decorator(node, "property")
                )
            if missing:
                findings.append(
                    Finding(
                        path=unit.path,
                        line=node.lineno,
                        rule=self.name,
                        message=(
                            f"{node.name} docstring lacks "
                            f"{'Yields:' if is_gen else 'Returns:'}"
                        ),
                    ),
                )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class PrivateOnelinerRule:
    """Private defs get a one-line docstring or none (STYLE.md Documentation)."""

    name: ClassVar[str] = "private-oneliner"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each private def with a multi-line docstring.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=f"private {node.name} has a multi-line docstring; one line or none",
            )
            for node in unit.nodes(*_DEF_KINDS)
            if node.name.startswith("_")
            and not _is_dunder(node.name)
            and _is_multiline_docstring(node)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Keep the summary; move the rest of the prose to a comment above the def.

        ``Args:``/``Returns:``-style sections are dropped: a private def's
        callers are all in this file, and the signature already says what
        the section restates. Any other paragraph is rationale, which
        STYLE.md wants in a comment guarding the site -- so it becomes one,
        placed above the decorators, at the def's indent.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        edits: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Constant]] = []
        for node in unit.nodes(*_DEF_KINDS):
            if not (node.name.startswith("_") and not _is_dunder(node.name)):
                continue
            const = _docstring_node(node)
            if (
                const is not None
                and isinstance(const.value, str)
                and "\n" in const.value
            ):
                edits.append((node, const))
        if not edits:
            return None
        lines = unit.source.splitlines(keepends=True)
        for node, const in sorted(edits, key=lambda e: -e[1].lineno):
            assert isinstance(const.value, str)
            indent = " " * node.col_offset
            summary, prose = _split_private_docstring(
                const.value,
                summary_width=88 - const.col_offset - 6,
                comment_width=88 - node.col_offset - 2,
            )
            opener_line = lines[const.lineno - 1]
            opener = opener_line[const.col_offset :]
            doc_indent = opener_line.removesuffix(opener)
            prefix = opener[: len(opener) - len(opener.lstrip("rRbBuU"))]
            lines[const.lineno - 1 : const.end_lineno or const.lineno] = [
                f'{doc_indent}{prefix}"""{summary}"""\n',
            ]
            if prose:
                first = node.decorator_list[0] if node.decorator_list else node
                lines[first.lineno - 1 : first.lineno - 1] = [
                    f"{indent}#{' ' + ln if ln else ''}\n" for ln in prose
                ]
        return "".join(lines)


@dataclass(frozen=True, kw_only=True, slots=True)
class AttrBlankLineRule:
    """A documented class attribute's docstring is followed by a blank line.

    An attribute docstring sits BELOW the attribute it documents, the
    opposite of every other docstring; the blank line after it is what
    stops a reader from attaching it to the attribute that follows. An
    undocumented attribute needs no separator.
    """

    name: ClassVar[str] = "attr-blank-line"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each attribute run-on.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        return tuple(
            Finding(
                path=unit.path,
                line=line,
                rule=self.name,
                message="attribute docstring not followed by a blank line",
            )
            for line in self._missing_blank_after(unit)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Insert the missing blank lines.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        missing = self._missing_blank_after(unit)
        if not missing:
            return None
        lines = unit.source.splitlines(keepends=True)
        for line in sorted(missing, reverse=True):
            lines.insert(line, "\n")
        return "".join(lines)

    def _missing_blank_after(self, unit: Unit) -> list[int]:
        """Line numbers after which a blank line is missing."""
        tree = unit.tree
        if tree is None:
            return []
        lines = unit.lines
        out: list[int] = []
        for cls in unit.nodes(ast.ClassDef):
            body = cls.body
            for i, stmt in enumerate(body[:-2]):
                if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    continue
                doc = body[i + 1]
                if not _is_docstring(doc):
                    continue
                end = doc.end_lineno or doc.lineno
                if lines[end].strip():
                    out.append(end)
        return out


@dataclass(frozen=True, kw_only=True, slots=True)
class AttrDocFormRule:
    """One attribute-doc form per class: inline strings or ``Attributes:``."""

    name: ClassVar[str] = "attr-doc-form"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each class mixing both forms.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "Attributes:" not in unit.source:
            return ()
        findings: list[Finding] = []
        for cls in unit.nodes(ast.ClassDef):
            doc = ast.get_docstring(cls)
            has_section = doc is not None and any(
                s == "Attributes" for s, _ in _sections(doc)
            )
            inline = any(
                isinstance(a, (ast.Assign, ast.AnnAssign)) and _is_docstring(b)
                for a, b in itertools.pairwise(cls.body)
            )
            if has_section and inline:
                findings.append(
                    Finding(
                        path=unit.path,
                        line=cls.lineno,
                        rule=self.name,
                        message=(
                            f"{cls.name} documents attributes both inline and in "
                            "Attributes:; pick one form"
                        ),
                    ),
                )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class CommentSentenceRule:
    """Comments are sentences: capitalized, punctuated (STYLE.md Comments)."""

    name: ClassVar[str] = "comment-sentence"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each comment block that is not a sentence.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "#" not in unit.source:
            return ()
        findings: list[Finding] = []
        for block in _comment_blocks(unit.tokens):
            first, last = block[0], block[-1]
            head = _comment_text(first)
            tail = _comment_text(last)
            if not head or _is_exempt_comment(head) or _is_marker(block):
                continue
            problems: list[str] = []
            # A first token that is a plain lowercase word is a sentence
            # start; one with a dot, underscore, paren, or mixed case
            # (``x.ndim``, ``arXiv``, ``git's``) is an identifier.
            if _LOWER_WORD.fullmatch(head.split()[0]):
                problems.append("starts lowercase")
            if not tail.endswith(_SENTENCE_END):
                problems.append("lacks terminal punctuation")
            if problems:
                findings.append(
                    Finding(
                        path=unit.path,
                        line=first.start[0],
                        rule=self.name,
                        message=f"comment {' and '.join(problems)} (STYLE.md Comments)",
                    ),
                )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """Terminate each unpunctuated comment block and capitalize its start.

        A lowercase first word that names something bound or imported in
        this file (``spec``, ``asyncpg``) is an identifier, so it is wrapped
        in backticks rather than capitalized; any other lowercase word is a
        sentence start.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        if "#" not in unit.source:
            return None
        original = unit.source.splitlines(keepends=True)
        lines = list(original)
        names = _bound_names(unit)
        for block in _comment_blocks(unit.tokens):
            head, tail = _comment_text(block[0]), _comment_text(block[-1])
            if not head or _is_exempt_comment(head) or _is_marker(block):
                continue
            first_word = head.split()[0]
            match = _LOWER_WORD.fullmatch(first_word)
            if match is not None:
                row, start = block[0].start[0] - 1, block[0].start[1]
                # ``_LOWER_WORD`` is letters plus at most one trailing mark, so
                # the word's first occurrence after ``#`` is the one to rewrite.
                stem = match.group(1)
                new_word = (
                    f"`{stem}`" if stem in names else stem.capitalize()
                ) + first_word[len(stem) :]
                col = lines[row].index(first_word, start)
                lines[row] = (
                    lines[row][:col] + new_word + lines[row][col + len(first_word) :]
                )
            if not tail.endswith(_SENTENCE_END):
                row = block[-1].start[0] - 1
                line = lines[row]
                body = line.rstrip()
                lines[row] = body.rstrip(",;") + "." + line[len(body) :]
        return "".join(lines) if lines != original else None


@dataclass(frozen=True, kw_only=True, slots=True)
class SuppressionFormRule:
    """Every suppression comment carries a ``-- Reason.`` tail.

    Applies to ruff (``noqa``), basedpyright (``pyright: ignore``), ty
    (``ty: ignore``), pyrefly (``pyrefly: ignore``), and the house marker
    itself. A bare suppression tells
    the next reader that a check was silenced and nothing else; the reason is
    what lets them decide whether it still holds. The fix-types skill requires
    the tail already; this makes the gate enforce it.

    ``type: ignore`` is a mypy directive and is reported outright: neither
    checker we run reads it, so it can only be dead.
    """

    name: ClassVar[str] = "suppression-form"
    kinds: ClassVar[frozenset[str]] = frozenset({"python", "pyi"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each suppression lacking a reason, and each ``type: ignore``.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if not _SUPPRESSION_PROBE.search(unit.source):
            return ()
        findings: list[Finding] = []
        for tok in unit.tokens:
            if tok.type != tokenize.COMMENT:
                continue
            text = tok.string
            if re.search(r"#\s*type:\s*ignore\b", text):
                findings.append(
                    Finding(
                        path=unit.path,
                        line=tok.start[0],
                        rule=self.name,
                        message=(
                            "`type: ignore` is a mypy directive; neither ty nor "
                            "basedpyright reads it -- remove it, or use "
                            "`ty: ignore[code]` / `pyright: ignore[code]`"
                        ),
                    ),
                )
            findings.extend(
                Finding(
                    path=unit.path,
                    line=tok.start[0],
                    rule=self.name,
                    message=(
                        f"`{match.group(0).strip()}` needs a trailing "
                        "`-- Reason.` saying why the check cannot be satisfied"
                    ),
                )
                for match in _SUPPRESSION.finditer(text)
                if _REASON_AFTER.match(text, match.end()) is None
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: the reason is the one thing a tool cannot supply.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class FmtDirectiveRule:
    """No ``# fmt: skip`` / ``# fmt: off``: shape the code so ruff's layout reads.

    A formatter opt-out exempts the line from the 88-column budget too, since
    E501 is off and ruff format is what enforces the width; a skipped line
    can grow without bound. The one exemption is the polyglot header, where
    ``# fmt: off`` keeps ruff from rewriting the shell preamble the ``cli``
    skill pins (``_POLYGLOT_HEADER``).
    """

    name: ClassVar[str] = "fmt-directive"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each formatter directive outside the polyglot header.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per directive, in source order.

        """
        if "fmt:" not in unit.source:
            return ()
        header = _polyglot_header_rows(unit.source.splitlines())
        return tuple(
            Finding(
                path=unit.path,
                line=tok.start[0],
                rule=self.name,
                message=(
                    "formatter opt-out; delete it and reshape the code so "
                    "`ruff format`'s own layout reads (name a constant, split "
                    "a helper, build the table from data)"
                ),
            )
            for tok in unit.tokens
            if tok.type == tokenize.COMMENT
            and _FMT_DIRECTIVE.search(tok.string)
            and tok.start[0] not in header
        )

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: dropping the directive reflows code ruff must format.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class AnnotationImportRule:
    """A heavy module imported only for annotations lives under ``TYPE_CHECKING``.

    Ruff's TC001-TC003 do this for imports it can prove are annotation-only,
    but they stop at the ``from __future__ import annotations`` boundary and
    never weigh the import. This rule counts uses and weighs: a name whose
    every use sits inside an annotation (parameter, return, variable
    annotation, quoted ``cast`` target) is annotation-only, and when its
    module costs ``threshold_ms`` or more to import, deferring it is a
    measurable win.

    Cheap modules are left alone on purpose. ``TYPE_CHECKING`` has a runtime
    cost of its own: ``typing.get_type_hints`` raises ``NameError`` on a name
    that was never bound, and configgle's override traversal
    (``configgle/cli_override.py``) and sagent's tool-schema builder
    (``sagent/tools/core.py``) both resolve hints at runtime. Trading that
    breakage for a ``pathlib`` import nobody can measure is a loss, so only a
    heavy module clears the bar.

    A root is skipped when any other name bound from it is used at runtime:
    moving ``Tensor`` under ``TYPE_CHECKING`` while ``torch`` itself loads
    eagerly two lines up defers nothing.
    """

    name: ClassVar[str] = "annotation-import"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    threshold_ms: int = 50
    """Modules costing at least this much are reported."""

    # A factory, not a default: the table is defined below the rule classes
    # (private helpers follow their callers) and a dataclass default is
    # evaluated at class-definition time.
    costs_ms: Mapping[str, int] = field(default_factory=lambda: _IMPORT_COSTS_MS)
    """Measured cold-import cost per top-level module, in milliseconds."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each heavy runtime import whose every use is an annotation.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        tree = unit.tree
        if tree is None or "import " not in unit.source:
            return ()
        imported = _runtime_imports(tree)
        if not imported:
            return ()
        usage = _name_usage(tree)
        runtime_roots = {
            spelled.partition(".")[0]
            for local, (_, spelled) in imported.items()
            if (uses := usage.get(local)) is not None and uses.runtime
        }
        findings: list[Finding] = []
        for local, (lineno, spelled) in imported.items():
            root = spelled.partition(".")[0]
            cost = self.costs_ms.get(root, 0)
            if cost < self.threshold_ms or root in runtime_roots:
                continue
            uses = usage.get(local)
            if uses is None or not uses.annotation:
                continue
            findings.append(
                Finding(
                    path=unit.path,
                    line=lineno,
                    rule=self.name,
                    message=(
                        f"{spelled!r} costs ~{cost} ms to import and is used only in "
                        f"{uses.annotation} annotation(s); import it under "
                        "`if TYPE_CHECKING:` (STYLE.md Code Organization)"
                    ),
                ),
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: moving an import can create a cycle only a human sees.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class FinalInlineRule:
    """A private ``Final`` module constant read once is an inline expression.

    STYLE.md: "Prefer inline expressions over module-level convenience
    constants. Readers shouldn't bounce to the top of the file to understand
    a value." A ``_NAME: Final = <literal>`` read at exactly one site is that
    bounce with nothing bought; write the literal where it is used. The same
    holds for a single-use ``_PAT = re.compile(...)``: the module-level
    binding exists to amortize compilation across calls, and one call site
    amortizes nothing -- ``re`` caches compiled patterns itself.

    Five shapes are left alone because one local read proves nothing about
    them. A public name may be imported by another module, so its read count
    here is not its read count. A binding that spans lines is prose or data
    (a help text, a table) whose name IS the documentation. A binding with
    an attribute docstring is a documented knob; inlining it would orphan the
    docstring. A binding introduced by a comment block is the same documented
    knob in the other house spelling. A name the sibling ``_test.py`` mentions
    is read there too -- a ``monkeypatch.setattr`` target or an asserted
    bound -- and inlining it breaks the test. A test module's own constants
    are fixture vocabulary shared across cases by intent; the hook excludes
    test files from this rule.
    """

    name: ClassVar[str] = "final-inline"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each single-use private one-line ``Final`` or ``re.compile``.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        tree = unit.tree
        if tree is None:
            return ()
        if not re.search(r"\bFinal\b|re\.compile\(", unit.source):
            return ()
        reads = _module_name_reads(unit)
        sibling = unit.path.with_name(f"{unit.path.stem}_test.py")
        test_source = sibling.read_text() if sibling.is_file() else unit.source[:0]
        # A trailing sentinel pairs the last statement with "no docstring after".
        return tuple(
            Finding(
                path=unit.path,
                line=stmt.lineno,
                rule=self.name,
                message=(
                    f"{name!r} is a {kind} read at one site; inline it there "
                    "(STYLE.md: no module-level convenience constants)"
                ),
            )
            for stmt, following in itertools.pairwise([*tree.body, ast.Pass()])
            if (binding := _inlineable_binding(stmt)) is not None
            for name, kind in [binding]
            if name.startswith("_")
            and not name.startswith("__")
            and stmt.end_lineno == stmt.lineno
            and reads[name] == 1
            and not _is_docstring(following)
            and name not in test_source
            and not _has_leading_comment(unit.lines, stmt.lineno)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: the inlined form may need reflowing at the use site.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class SingleUseHelperRule:
    """A one-expression private helper called once is a name for an expression.

    STYLE.md: "Prefer inlining over a shared helper unless the shared body is
    substantial." A module-level ``_helper`` whose body is a single ``return
    <expr>`` and whose name is read at exactly one site shares nothing and
    hides the expression behind a bounce to the bottom of the file. Write the
    expression where it is used.

    Exempt, because inlining would change what the call means: a return
    annotated ``TypeGuard``/``TypeIs`` (the checker narrows only through a
    call), any decorator, ``async``, ``*args``/``**kwargs``, and type
    parameters (``def f[T]``: ``T`` is bound only inside the def, and each
    call solves it). Also exempt: a name the sibling ``_test.py`` mentions,
    which is a monkeypatch target or under direct test.

    The fix substitutes the call's arguments for the parameters in the return
    expression, replaces the call, and deletes the def. A helper is reported
    only when that substitution is a pure rename, so every finding is
    autofixable. Silent otherwise: the name is passed as a value rather than
    called (a typed callback, which a lambda would untype), a parameter is
    read other than exactly once, an argument would run other than once, in
    call order, ahead of the body's calls (read under a branch, in a
    comprehension or lambda, or after a call; a constant may sit anywhere,
    and a name anywhere but in a lambda), a missing argument's default is not
    a constant (it ran once, with the def), the call splats, a name would
    bind differently where it lands (a global some scope around the call
    rebinds, an argument the body's own lambda or comprehension rebinds), a
    comment or suppression would be lost, a return annotation is
    load-bearing for the checker, or the body is a literal table the def
    exists to name. A report the fixer cannot honour is an invitation to
    hand-edit or suppress; both were measured worse than the helper.
    """

    name: ClassVar[str] = "single-use-helper"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each single-use, single-expression private helper.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        return tuple(
            Finding(
                path=unit.path,
                line=helper.node.lineno,
                rule=self.name,
                message=(
                    f"{helper.node.name} is a one-expression helper called once; "
                    "inline it at the call site (STYLE.md Function Design)"
                ),
            )
            for helper in _single_use_helpers(unit)
            if helper.call is not None
            and _inline_call(helper.node, helper.call, unit) is not None
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Inline each helper whose substitution is a pure rename; delete its def.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        helpers = [h for h in _single_use_helpers(unit) if h.call is not None]
        # Every edit is an absolute offset into the ORIGINAL source and they are
        # applied bottom-up, so a call-site rewrite that shortens the file
        # cannot shift a def deletion below it onto the wrong lines. Two
        # helpers whose spans overlap -- one called from inside the other --
        # cannot both be applied against the original text; the second waits
        # for the next run, which the dispatcher's idempotence check permits.
        edits: list[tuple[int, int, str]] = []
        starts = _line_starts(unit.source)
        for helper in helpers:
            call = helper.call
            if call is None:
                raise ValueError("Expected call is not None.")
            replacement = _inline_call(helper.node, call, unit)
            if replacement is None:
                continue
            first, last = _def_span(helper.node, unit.lines)
            proposed = (
                (
                    _offset(starts, call.lineno, call.col_offset),
                    _offset(starts, call.end_lineno, call.end_col_offset),
                    replacement,
                ),
                (starts.starts[first], starts.starts[last], ""),
            )
            if any(
                max(s, s2) < min(e, e2) for s, e, _ in proposed for s2, e2, _ in edits
            ):
                continue
            edits.extend(proposed)
        if not edits:
            return None
        source = unit.source
        for start, end, text in sorted(edits, reverse=True):
            source = source[:start] + text + source[end:]
        # Deleting a def leaves the blank lines on both sides of it behind;
        # two between top-level definitions is what the formatter wants.
        return re.sub(r"\n{4,}", "\n\n\n", source).rstrip("\n") + "\n"


@dataclass(frozen=True, kw_only=True, slots=True)
class ForwarderHelperRule:
    """A def only this file calls, whose whole body calls another, is that function.

    STYLE.md: a function "whose whole body is ``return _helper(...)``" is a
    smell -- inline it. A module-level def whose body is one line, ``return
    g(...)``, adds a name and a bounce and no logic, however many sites call
    it. ``single-use-helper`` defers every such def to this rule.

    Public names qualify too, when nothing outside this file reaches them:
    not listed in ``__all__``, not in a package ``__init__``, and named by no
    other module under the project root (a test, a sibling, an importer).

    The same exemptions and the same per-call substitution as
    ``single-use-helper`` apply, so the rule reports a helper only when every
    call site inlines -- one unsafe site and the def stays whole.
    """

    name: ClassVar[str] = "forwarder-helper"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each forwarder every call site can inline.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=(
                    f"{node.name} only forwards to another function; inline it "
                    "at every call site (STYLE.md Function Design)"
                ),
            )
            for node, _ in _inlinable_forwarders(unit)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Inline every call of each forwarder and delete its def.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        edits: list[tuple[int, int, str]] = []
        starts = _line_starts(unit.source)
        for node, replacements in _inlinable_forwarders(unit):
            first, last = _def_span(node, unit.lines)
            sites = [
                (
                    _offset(starts, call.lineno, call.col_offset),
                    _offset(starts, call.end_lineno, call.end_col_offset),
                    text,
                )
                for call, text in replacements
            ]
            # Every edit is an offset into the ORIGINAL source, so no two may
            # overlap. A call nested in another call of the same forwarder
            # (``_m(_m(x)["a"])``) is left for the next run: the outer one is
            # inlined now, and the def stays until no call remains. The
            # dispatcher reruns fixes to a fixed point.
            outer = [
                site
                for site in sites
                if not any(
                    o[0] <= site[0] and site[1] <= o[1] and o != site for o in sites
                )
            ]
            proposed = outer
            if len(outer) == len(sites):
                proposed = [*outer, (starts.starts[first], starts.starts[last], "")]
            if any(
                max(s, s2) < min(e, e2) for s, e, _ in proposed for s2, e2, _ in edits
            ):
                continue
            edits.extend(proposed)
        if not edits:
            return None
        source = unit.source
        for start, end, text in sorted(edits, reverse=True):
            source = source[:start] + text + source[end:]
        return re.sub(r"\n{4,}", "\n\n\n", source).rstrip("\n") + "\n"


@dataclass(frozen=True, kw_only=True, slots=True)
class _SingleUseHelper:
    """A private helper the rule reports, and its one call site when it is one.

    Attributes:
      node: The helper's def.
      call: The single call, or None when the name is passed as a value.

    """

    node: ast.FunctionDef

    call: ast.Call | None


@dataclass(frozen=True, kw_only=True, slots=True)
class LazyImportCostRule:
    """A heavy module few functions need goes through ``lazy_import``.

    STYLE.md: "Imports not on the main path and which take >50ms to load
    should be guarded as ``wrapt.lazy_import``." The costs in
    ``_IMPORT_COSTS_MS`` were measured cold, best of three, on the dev box
    (``import X`` in a fresh interpreter minus the bare-interpreter floor).

    "Not on the main path" is decided from the module itself: an eager import
    of a heavy root is a finding when at most ``max_use_fraction`` of the
    module's function bodies reach a name bound from it. A module where every
    def touches torch IS a torch module, and deferring torch there buys
    nothing but a proxy. Two further shapes are exempt because the lazy
    binding cannot replace them:

    - A name used at module scope (a base class, a decorator, a constant
      built from the module) needs the real object at import time.
    - A root some sibling eager heavy import already loads -- ``import torch``
      brings numpy with it -- costs nothing extra; ``_IMPLIED_IMPORTS`` lists
      the measured pairs.

    A test module pays the import once per process and its collection is the
    main path; the hook excludes test files from this rule.
    """

    name: ClassVar[str] = "lazy-import-cost"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False
    threshold_ms: int = 50
    """Modules costing at least this much are reported."""

    max_use_fraction: float = 0.25
    """A root reached from more than this share of the defs is main-path."""

    costs_ms: Mapping[str, int] = field(default_factory=lambda: _IMPORT_COSTS_MS)
    """Measured cold-import cost per top-level module, in milliseconds."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each eager heavy import that few of the module's defs reach.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "import " not in unit.source:
            return ()
        tree = unit.tree
        if tree is None:
            return ()
        eager = [
            (node, module, module.partition(".")[0])
            for node in _eager_imports(tree)
            for module in _imported_modules(node)
            if self.costs_ms.get(module.partition(".")[0], 0) >= self.threshold_ms
        ]
        defs = unit.nodes(*_DEF_KINDS)
        if not eager or len(defs) < 2:
            return ()
        implied = {
            implied_root
            for _, _, root in eager
            for implied_root in _IMPLIED_IMPORTS.get(root, frozenset())
        }
        locals_by_root: dict[str, set[str]] = {}
        for node, _, root in eager:
            locals_by_root.setdefault(root, set()).update(_import_locals(node))
        module_scope = _module_scope_loads(tree)
        # Each def's names once, instead of re-walking every body per import.
        def_names = [
            {n.id for stmt in d.body for n in ast.walk(stmt) if isinstance(n, ast.Name)}
            for d in defs
        ]
        type_checked = {
            argument.id
            for call in unit.nodes(ast.Call)
            if isinstance(call.func, ast.Name) and call.func.id == "isinstance"
            for argument in call.args[1:]
            if isinstance(argument, ast.Name)
        }
        findings: list[Finding] = []
        for node, module, root in eager:
            names = locals_by_root[root]
            if (
                root in implied
                or names & (module_scope | type_checked)
                or _house_ignore(unit, node, self.name)
            ):
                continue
            using = sum(bool(names & seen) for seen in def_names)
            if using > self.max_use_fraction * len(defs):
                continue
            findings.append(
                Finding(
                    path=unit.path,
                    line=node.lineno,
                    rule=self.name,
                    message=(
                        f"{module!r} costs ~{self.costs_ms[root]} ms to import and "
                        f"only {using} of {len(defs)} defs reach it; bind it with "
                        "`lazy_import` (STYLE.md Code Organization)"
                    ),
                ),
            )
        return tuple(findings)

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: the lazy binding shape depends on how the name is used.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class AssertRule:
    """``assert`` is for narrowing; a runtime check raises (STYLE.md Error Handling).

    ``assert isinstance(...)`` (or an ``and`` of them) is type narrowing the
    checker reads and ``-O`` may drop without changing a correct program. Any
    other assert is a validation that ``-O`` silently removes, so it is
    reported; the fix rewrites it to ``if <not test>: raise ValueError(msg)``.

    The negation follows the house spelling: a comparison flips its operator
    (``==`` to ``!=``, ``is`` to ``is not``); an ordering flips only against
    a numeric literal, since on floats ``not a <= b`` also rejects NaN and
    ``a > b`` does not (STYLE.md Code Style); ``and``/``or`` distribute by De
    Morgan; anything else is wrapped in ``not``. Without a message the
    exception says what was expected. A statement carrying a comment, or one
    whose test spans lines, is reported but left for the author: the rewrite
    would drop the comment or reflow the test. So is one whose negation would
    be a ``not a <= b`` that ``negated-compare`` forbids.
    """

    name: ClassVar[str] = "assert"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each assert that is not an isinstance narrowing.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        tree = unit.tree
        if tree is None:
            return ()
        if "assert " not in unit.source:
            return ()
        return tuple(
            Finding(
                path=unit.path,
                line=node.lineno,
                rule=self.name,
                message=(
                    "assert used as a runtime check; `-O` drops it. Raise "
                    "ValueError instead (rerun with --fix assert), or narrow "
                    "with isinstance"
                ),
            )
            for node in unit.nodes(ast.Assert)
            if not _is_isinstance_test(node.test)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Rewrite each single-line assert to ``if <not test>: raise ValueError``.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        tree = unit.tree
        if tree is None:
            return None
        if "assert " not in unit.source:
            return None
        comments = {
            tok.start[0]: tok.string
            for tok in unit.tokens
            if tok.type == tokenize.COMMENT
        }
        starts = _cached_line_starts(unit.source)
        edits: list[tuple[ast.Assert, str, str]] = []
        for node in unit.nodes(ast.Assert):
            if _is_isinstance_test(node.test):
                continue
            # A marked assert is kept on purpose -- a check ``-O`` may drop,
            # such as the non-negative base ahead of ``cast(float, x ** k)``.
            here = Finding(path=unit.path, line=node.lineno, rule=self.name, message="")
            if not house_ignore.apply(unit, [here], active=frozenset()):
                continue
            end = node.end_lineno or node.lineno
            # The rewrite replaces whole lines, so an assert sharing a line
            # with other code (``if x: assert y``, ``assert y; z = 1``) is
            # reported only; replacing its lines would break or drop that code.
            before = unit.source[
                starts.starts[node.lineno - 1] : _offset(
                    starts,
                    node.lineno,
                    node.col_offset,
                )
            ]
            after = starts.lines[end - 1][
                _offset(starts, end, node.end_col_offset) - starts.starts[end - 1] :
            ].strip()
            if before.strip() or (after and not after.startswith("#")):
                continue
            inside = [line for line in comments if node.lineno <= line <= end]
            # One trailing comment on the statement's last line moves to the
            # ``if`` header. A comment anywhere else in the statement has no
            # mechanical home in the rewrite, so the site is reported only.
            if inside and inside != [end]:
                continue
            negated = _negated(node.test, unit.source)
            # ``not a <= b`` is what ``negated-compare`` forbids, and which
            # spelling is right turns on whether NaN can reach it.
            if any(
                map(_is_negated_ordering, ast.walk(ast.parse(negated, mode="eval"))),
            ):
                continue
            edits.append((node, negated, f"  {comments[end]}" if inside else ""))
        if not edits:
            return None
        lines = unit.source.splitlines(keepends=True)
        for node, negated, trailing in sorted(edits, key=lambda e: -e[0].lineno):
            indent = lines[node.lineno - 1][: node.col_offset]
            test = (
                ast.unparse(node.test)
                if node.test.lineno != node.test.end_lineno
                else _segment(unit.source, node.test)
            )
            if node.msg is None:
                message = (
                    f'"Expected {test}."'
                    if '"' not in test
                    else repr(
                        f"Expected {test}.",
                    )
                )
            else:
                # Verbatim, so quoting and implicit concatenation survive; a
                # message that spans lines keeps its line breaks and the
                # formatter re-indents them.
                message = _segment(unit.source, node.msg)
            # A type check that fails is a TypeError (ruff TRY004); the
            # ``isinstance`` exemption above leaves only the negated and
            # ``type(x) is T`` spellings to reach here.
            exception = "TypeError" if _is_type_test(node.test) else "ValueError"
            lines[node.lineno - 1 : node.end_lineno or node.lineno] = [
                f"{indent}if {negated}:{trailing}\n",
                f"{indent}    raise {exception}({message})\n",
            ]
        return "".join(lines)


@dataclass(frozen=True, kw_only=True, slots=True)
class TreekleTorchLoadRule:
    """A ``torch.load`` payload is cast to its type, never read with ``from_plain``.

    ``torch.load`` returns what ``torch.save`` pickled: tensors, not a plain
    tree. ``from_plain`` over it checks only that a leaf is a ``Tensor`` --
    never its shape, dtype or device -- while it drops keys the target does
    not declare and refuses a file that lacks one, so a corpus written by an
    older or newer writer stops loading. One ``cast(T, torch.load(...))`` to
    the writer's ``TypedDict`` states the contract and reads every version; a
    lone tensor narrows with ``isinstance``. The older spelling,
    ``from_plain(cast(object, torch.load(...)), T)``, is the same read, and
    so is a payload bound to a name first and read through it
    (``state = torch.load(...)``, then ``from_plain(state["model"], T)``).
    ``from_plain`` reads only what ``to_plain`` wrote, so a read with
    ``allow_imports=True`` passes: only ``to_plain`` writes the ``py/*`` tags
    that flag lets name code, as a payload's ``to_plain(config)`` field does.
    """

    name: ClassVar[str] = "treekle-torch-load"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = False

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each ``from_plain`` whose input is a ``torch.load`` payload.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        if "torch.load" not in unit.source or "from_plain" not in unit.source:
            return ()
        loaded = {
            target.id
            for node in unit.nodes(ast.Assign, ast.AnnAssign)
            if node.value is not None and _is_torch_load(node.value)
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        return tuple(
            Finding(
                path=unit.path,
                line=call.lineno,
                rule=self.name,
                message=(
                    "from_plain over a torch.load payload checks no shape and "
                    "rejects older files; write cast(T, torch.load(...)) with the "
                    "writer's TypedDict"
                ),
            )
            for call in unit.nodes(ast.Call)
            if isinstance(call.func, ast.Name)
            and call.func.id == "from_plain"
            and call.args
            and (_is_torch_load(call.args[0]) or _read_root(call.args[0]) in loaded)
            and not any(
                keyword.arg == "allow_imports"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in call.keywords
            )
        )

    def fix(self, unit: Unit, /) -> str | None:
        """No autofix: the target type moves into a cast the author places.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        del unit
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class LoggingFstringRule:
    """A logging call takes a ``%`` format and arguments, not an f-string (G004).

    ``logger.info(f"a {x}")`` formats eagerly whether or not INFO is enabled;
    ``logger.info("a %s", x)`` defers to the handler. The fix maps each
    placeholder to a ``%`` spec: no spec is ``%s``; ``!r``/``!s``/``!a`` are
    ``%r``/``%s``/``%a``; a numeric spec (``.2f``, ``d``, ``>8``) is carried
    over. A spec ``%`` cannot express (``,``, ``%``, ``=``) keeps that
    placeholder as an f-string ARGUMENT, so the message still defers.
    """

    name: ClassVar[str] = "logging-fstring"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True
    methods: tuple[str, ...] = (
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "exception",
        "critical",
        "fatal",
        "log",
    )
    """Logger methods whose first positional is a lazily-formatted message."""

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report each logging call whose message is an f-string.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        return tuple(
            Finding(
                path=unit.path,
                line=call.lineno,
                rule=self.name,
                message=(
                    "f-string in a logging call formats eagerly; pass a %-format "
                    "and arguments (rerun with --fix logging-fstring)"
                ),
            )
            for call, _ in self._sites(unit)
        )

    def fix(self, unit: Unit, /) -> str | None:
        """Rewrite each f-string message to a ``%`` format plus arguments.

        Args:
          unit: The file under inspection.

        Returns:
          rewritten: The full new source, or None when nothing changes.

        """
        edits: list[tuple[int, int, str]] = []
        starts = _line_starts(unit.source)
        for call, message in self._sites(unit):
            # A call already passing arguments has a message we cannot rebuild.
            if call.args[-1] is not message:
                continue
            fmt, args = _percent_format(message, unit.source)
            quoted = '"' + fmt.replace("\\", "\\\\").replace('"', '\\"') + '"'
            edits.append(
                (
                    _offset(starts, message.lineno, message.col_offset),
                    _offset(starts, message.end_lineno, message.end_col_offset),
                    ", ".join([quoted, *args]),
                ),
            )
        if not edits:
            return None
        source = unit.source
        for start, end, text in sorted(edits, reverse=True):
            source = source[:start] + text + source[end:]
        return source

    def _sites(self, unit: Unit) -> Iterator[tuple[ast.Call, ast.JoinedStr]]:
        """Yield each logging call and its f-string message."""
        if 'f"' not in unit.source and "f'" not in unit.source:
            return
        for node in unit.nodes(ast.Call):
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in self.methods
                and node.args
            ):
                continue
            receiver = _call_text(node.func.value).rpartition(".")[2]
            if not (
                receiver in ("logging", "logger", "log") or receiver.startswith("_log")
            ):
                continue
            index = 1 if node.func.attr == "log" else 0
            if len(node.args) > index:
                message = node.args[index]
                if isinstance(message, ast.JoinedStr):
                    yield node, message


@dataclass(frozen=True, kw_only=True, slots=True)
class CliShapeRule:
    """Mechanical checks from the ``cli`` skill, one lettered finding each."""

    name: ClassVar[str] = "cli-shape"
    kinds: ClassVar[frozenset[str]] = frozenset({"python"})
    fixable: ClassVar[bool] = True

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Run every lettered check and collect their findings.

        Args:
          unit: The file under inspection.

        Returns:
          findings: One per violation, in source order.

        """
        lines = unit.lines
        polyglot = bool(lines) and lines[0] == "#!/bin/sh"
        findings: list[str | tuple[int, str]] = []
        if polyglot:
            findings.extend(self._header(lines, unit))
        tree = unit.tree
        if tree is None:
            return self._emit(unit, findings)
        guard = next(
            (s for s in tree.body if isinstance(s, ast.If) and _is_main_guard(s)),
            None,
        )
        # Shape 2 (a library with a run-hook) keeps a bare guard doing one
        # call that is NOT ``main``; a guard that launches ``main`` claims to
        # be an executable and needs the header. Tests keep ``test_main``.
        has_main = any(
            isinstance(d, ast.FunctionDef) and d.name == "main" for d in tree.body
        )
        if (
            guard is not None
            and not polyglot
            and has_main
            and not _is_test_file(unit.path)
        ):
            findings.append(
                "(b) `if __name__` guard launching main without the polyglot "
                "header: half-executable",
            )
        if unit.path.name == "__main__.py":
            findings.extend(self._main_module(tree))
        # Module-level only, as ``_fix_cli`` rewrites: a ``main`` method on a
        # class or Protocol is not the entry point.
        mains = [
            d for d in tree.body if isinstance(d, ast.FunctionDef) and d.name == "main"
        ]
        for node in mains:
            findings.extend(self._main_def(node, noreturn=_noreturn_defs(tree)))
        if (
            guard is not None
            and len(guard.body) == 1
            and isinstance(guard.body[0], ast.Expr)
            and ast.unparse(guard.body[0]) == "main()"
        ):
            findings.append(
                (
                    guard.body[0].lineno,
                    "(d) guard drops main's exit code; `raise SystemExit(main())`",
                ),
            )
        findings.extend(self._add_argument_sites(mains))
        findings.extend(self._description(unit, polyglot=polyglot))
        findings.extend(self._t201(unit, lines))
        return self._emit(unit, findings)

    def fix(self, unit: Unit, /) -> str | None:
        """Apply mechanical CLI-shape rewrites; leave design choices alone."""
        return _fix_cli(unit)

    def _emit(
        self,
        unit: Unit,
        findings: Sequence[str | tuple[int, str]],
    ) -> Sequence[Finding]:
        """Turn collected messages into Findings."""
        return tuple(
            Finding(
                path=unit.path,
                line=f[0] if isinstance(f, tuple) else 1,
                rule=self.name,
                message=f[1] if isinstance(f, tuple) else f,
            )
            for f in findings
        )

    # Extra ruff-noqa lines after line 2 are allowed (a file may need ``N999``
    # for a kebab-case name); the fixed lines follow them. The ``exec`` line may span a
    # backslash continuation and may run ``python3 -m <module>`` instead of ``"$0"``
    # (shape 2b, where the module must import as itself); ``--frozen --no-sync`` are
    # mandatory and the project is ``$(dirname "$0")`` with no path walk.
    def _header(self, lines: Sequence[str], unit: Unit) -> Iterator[str]:
        """Check (a): the polyglot header, trailer, and executable bit."""
        if len(lines) < 5 or not _RUFF_NOQA_HEADER.match(lines[1]):
            yield "(a) line 2 must be `# ruff: noqa: EXE003, D300, D205[, CODE...]`"
        start = 2
        while (
            lines[start:]
            and lines[start].startswith("# ")
            and (not lines[start].startswith("# fmt:"))
        ):
            start += 1
        for offset, want in enumerate(_POLYGLOT_HEADER[:2], start=start):
            if len(lines) <= offset or lines[offset] != want:
                yield f"(a) line {offset + 1} must be `{want}`"
        # The shell prelude may carry comments and a fast path before the
        # ``exec uv`` line (trax walks to the venv interpreter first); the
        # check is on the ``exec uv`` line itself, wherever it sits.
        exec_at = start + 2
        while exec_at < len(lines) and not lines[exec_at].startswith("exec uv "):
            if lines[exec_at].startswith(3 * "'"):
                break
            exec_at += 1
        exec_line = ""
        while exec_at < len(lines):
            line = lines[exec_at].rstrip()
            exec_line += line.removesuffix("\\").rstrip()
            if line == line.removesuffix("\\"):
                break
            exec_at += 1
            exec_line += " "
        if not _EXEC_LINE.match(exec_line):
            yield (
                f"(a) line {exec_at + 1} must be `{_POLYGLOT_HEADER[2]}` "
                '(or `... python3 -m <module> "$@"` for a coupled entry)'
            )
        if not lines or lines[-1] != "# vim: ft=python":
            yield "(a) last line must be `# vim: ft=python`"
        try:
            mode = unit.path.stat().st_mode
        except OSError:
            return
        if not mode & stat.S_IXUSR:
            yield "(a) polyglot file lacks the executable bit; chmod +x"

    def _main_module(self, tree: ast.Module) -> Iterator[str]:
        """Check (c): ``__main__.py`` is a pure dispatcher."""
        body = _without_docstring(tree.body)
        imports = [s for s in body if isinstance(s, ast.ImportFrom)]
        guards = [s for s in body if isinstance(s, ast.If) and _is_main_guard(s)]
        others = [s for s in body if s not in imports and s not in guards]
        if others or len(guards) != 1:
            yield "(c) __main__.py holds more than `from X import main` + the guard"
            return
        main_imports = [s for s in imports if any(a.name == "main" for a in s.names)]
        if len(main_imports) != 1:
            yield "(c) __main__.py must import `main` exactly once"
        guard_body = guards[0].body
        if not (
            len(guard_body) == 1
            and ast.unparse(guard_body[0]) == "raise SystemExit(main())"
        ):
            yield "(c) guard body must be exactly `raise SystemExit(main())`"

    def _main_def(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        noreturn: frozenset[str],
    ) -> Iterator[tuple[int, str]]:
        """Check (d): ``main() -> int``, every path returns a code, no ``sys.exit``."""
        if node.returns is None or ast.unparse(node.returns) != "int":
            yield (node.lineno, "(d) main must be annotated `-> int`")
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and _call_text(sub) in ("sys.exit", "exit"):
                yield (sub.lineno, "(d) sys.exit inside main; return the code instead")
            if isinstance(sub, ast.Return) and sub.value is None:
                yield (sub.lineno, "(d) bare return in main; return an exit code")
        if _falls_through(node.body, noreturn=noreturn):
            last = node.body[-1]
            yield (
                last.end_lineno or last.lineno,
                "(d) main can fall off its end; finish with `return 0`",
            )

    # The skill's rule is that ``main`` stays orchestration and flag registration lives
    # behind a seam. ``_add_arguments`` / ``_parse_args`` are the canonical seams; any
    # other helper (a ``make_parser`` classmethod per subcommand, a ``_parser()``
    # builder with subparsers) is still a seam and still leaves ``main`` pure.
    def _add_argument_sites(
        self,
        mains: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> Iterator[tuple[int, str]]:
        """Check (e): ``add_argument`` never inline in ``main``."""
        for node in mains:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "add_argument"
                ):
                    yield (
                        sub.lineno,
                        (
                            "(e) add_argument inline in main; group flags in "
                            "_add_arguments"
                        ),
                    )

    def _description(
        self,
        unit: Unit,
        *,
        polyglot: bool,
    ) -> Iterator[tuple[int, str]]:
        """Check (f): the ``description=`` docstring idiom matches the file shape."""
        want = _description_idiom(polyglot=polyglot)
        for call, _ in _wrong_descriptions(unit, polyglot=polyglot):
            yield (call.lineno, f"(f) description= must be `{want}`")
        for node in unit.nodes(ast.Call):
            if _call_text(node).endswith("cleandoc") and "__doc__" in ast.unparse(node):
                yield (
                    node.lineno,
                    "(f) inspect.cleandoc on __doc__; use the slice idiom",
                )
        for node in unit.nodes(ast.Assign):
            if any(
                isinstance(t, ast.Name) and t.id == "doc" for t in node.targets
            ) and ast.unparse(node.value).startswith("__doc__"):
                yield (node.lineno, "(f) local `doc = __doc__...`; inline the idiom")

    # Only a polyglot file has a header to carry the file-level form; a library module
    # with several suppressed prints keeps them inline (STYLE.md forbids file-level
    # suppressions outside that header). Counted are ``print(`` calls that are
    # suppressed at all -- a ``console.print`` is not the builtin.
    def _t201(self, unit: Unit, lines: Sequence[str]) -> Iterator[tuple[int, str]]:
        """Check (g): T201 suppression placement follows the print count."""
        suppressed = [n for n, ln in enumerate(lines, start=1) if "noqa: T201" in ln]
        header_has = any(
            ln.startswith("# ruff: noqa:") and "T201" in ln for ln in lines[:8]
        )
        polyglot = bool(lines) and lines[0] == "#!/bin/sh"
        prints = len(re.findall(r"(?<![\w.])print\(", unit.source))
        if header_has and prints < 2:
            yield (1, "(g) file-level T201 with fewer than 2 prints; use inline noqa")
        if polyglot and prints >= 2 and suppressed:
            yield (
                suppressed[0],
                "(g) inline T201 with 2+ prints; use the file-level header",
            )


def _params(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> Sequence[ast.arg]:
    """Every positional and keyword parameter of ``node``."""
    args = node.args
    return [*args.posonlyargs, *args.args, *args.kwonlyargs]


def _is_underscored(name: str) -> bool:
    """Whether a parameter name is ``_x``-style (not ``_``, not dunder)."""
    return name.startswith("_") and name != "_" and not _is_dunder(name)


# Triton's code generator rejects `del`, and an autotune `key=` names the
# parameter verbatim, so a jitted kernel can neither drop nor rename it.
def _is_triton_kernel(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``node`` carries a ``triton.jit``/``@jit`` decorator."""
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Attribute) and target.attr == "jit":
            return True
        if isinstance(target, ast.Name) and target.id == "jit":
            return True
    return False


def _is_dunder(name: str) -> bool:
    """Whether ``name`` is ``__x__``."""
    return name.startswith("__") and name.endswith("__")


def _is_int_literal(node: ast.expr) -> bool:
    """Whether ``node`` is an integer constant, possibly negated."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) is int


def _is_negated_ordering(node: ast.AST) -> TypeIs[ast.UnaryOp]:
    """Whether ``node`` is ``not <compare>`` using only ordering operators."""
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Compare)
        and all(type(op) in _FLIPPED for op in node.operand.ops)
    )


def _is_negated_ternary(node: ast.AST) -> TypeIs[ast.IfExp]:
    """Whether ``node`` is ``A if not C else B`` with a simple ``C``."""
    return (
        isinstance(node, ast.IfExp)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, (ast.Name, ast.Attribute, ast.Call))
    )


def _is_super_only(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``node``'s body is exactly a forwarding ``super().<name>(...)``."""
    body = _without_docstring(node.body)
    # A default is behavior: deleting the override changes what a call that
    # omits the argument passes the parent.
    if len(body) != 1 or node.args.defaults or any(node.args.kw_defaults):
        return False
    stmt = body[0]
    call = stmt.value if isinstance(stmt, (ast.Return, ast.Expr)) else None
    if isinstance(call, ast.Await):
        call = call.value
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == node.name
        and isinstance(call.func.value, ast.Call)
        and _call_text(call.func.value) == "super"
    ):
        return False
    params = [a.arg for a in _params(node)][1:]
    forwarded = [ast.unparse(a) for a in call.args] + [
        f"{k.arg}={ast.unparse(k.value)}"
        if k.arg is not None
        else f"**{ast.unparse(k.value)}"
        for k in call.keywords
    ]
    if node.args.vararg:
        params.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        params.append(f"**{node.args.kwarg.arg}")
    return forwarded == params or forwarded == [f"{p}={p}" for p in params]


def _is_torch_load(node: ast.expr) -> bool:
    """Whether ``node`` is ``torch.load(...)``, bare or inside a ``cast``."""
    if not isinstance(node, ast.Call):
        return False
    if _call_text(node) == "cast" and len(node.args) == 2:
        return _is_torch_load(node.args[1])
    return _call_text(node) == "torch.load"


def _read_root(node: ast.expr) -> str:
    """Return the name a read of ``node`` starts from, through subscripts, ``.get`` and casts."""
    while True:
        if isinstance(node, ast.Subscript | ast.Attribute):
            node = node.value
        elif (
            isinstance(node, ast.Call)
            and _call_text(node) == "cast"
            and len(node.args) == 2
        ):
            node = node.args[1]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        ):
            node = node.func.value
        else:
            return node.id if isinstance(node, ast.Name) else ""


def _is_isinstance_test(test: ast.expr) -> bool:
    """Whether ``test`` is ``isinstance(...)`` or an ``and`` of only those."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return all(_is_isinstance_test(value) for value in test.values)
    return isinstance(test, ast.Call) and _call_text(test) == "isinstance"


def _is_type_test(test: ast.expr) -> bool:
    """Whether ``test`` checks a type: ``not isinstance``, ``type(x) is T``, etc."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _is_type_test(test.operand)
    if isinstance(test, ast.BoolOp):
        return all(_is_type_test(value) for value in test.values)
    if isinstance(test, ast.Call):
        return _call_text(test) in ("isinstance", "issubclass", "callable")
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        return any(
            isinstance(side, ast.Call) and _call_text(side) == "type"
            for side in (test.left, test.comparators[0])
        )
    return False


# Where the negation only wraps or strips, the operand's original text is kept so
# an author's spacing and quoting survive; a flipped comparison is re-spelled
# from the AST, which the formatter normalizes anyway. A test spanning lines is
# re-spelled too: its original wrapping was chosen for ``assert (`` and the
# formatter re-wraps the ``if`` header.
def _negated(test: ast.expr, source: str) -> str:
    """Spell ``not test`` the way STYLE.md wants it written; ``source`` is the module."""
    if test.lineno != test.end_lineno:
        text = ast.unparse(test)
        return _negated(ast.parse(text, mode="eval").body, text)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _segment(source, test.operand)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op = test.ops[0]
        flipped = _INVERTED.get(type(op))
        if type(op) in _FLIPPED and not (
            _is_number_literal(test.left) or _is_number_literal(test.comparators[0])
        ):
            flipped = None
        if flipped is not None:
            # Unparsed as one node so an operand that binds looser than the
            # comparison (``x in (a or b)``) keeps its parentheses.
            return ast.unparse(
                ast.Compare(
                    left=test.left,
                    ops=[flipped()],
                    comparators=test.comparators,
                ),
            )
    if isinstance(test, ast.BoolOp):
        joiner = " or " if isinstance(test.op, ast.And) else " and "
        return joiner.join(
            f"({negated})" if isinstance(value, ast.BoolOp) else negated
            for value in test.values
            for negated in (_negated(value, source),)
        )
    text = _segment(source, test)
    if isinstance(
        test,
        ast.Name | ast.Attribute | ast.Call | ast.Subscript | ast.Compare,
    ):
        return f"not {text}"
    return f"not ({text})"


# Conversions and specs ``%`` can carry directly. Alignment ``>N`` is ``%Ns`` and
# ``<N`` is ``%-Ns``; anything else (``,`` grouping, ``%`` percent, ``=`` debug,
# ``^`` centring, nested ``{width}``) stays an f-string argument.
_PERCENT_SPEC = re.compile(
    r"^(?P<flags>[-+ 0#]*)(?P<width>\d*)(?P<prec>\.\d+)?(?P<type>[dieEfFgGxXos])?$",
)
_ALIGN_SPEC = re.compile(r"^(?P<align>[<>])(?P<width>\d+)$")


def _percent_format(message: ast.JoinedStr, source: str) -> tuple[str, list[str]]:
    """Split an f-string into a ``%`` format and the argument texts it consumes."""
    fmt: list[str] = []
    args: list[str] = []
    for part in message.values:
        if isinstance(part, ast.Constant):
            fmt.append(str(part.value).replace("%", "%%"))
            continue
        if not isinstance(part, ast.FormattedValue):
            continue
        spec = ""
        if isinstance(part.format_spec, ast.JoinedStr):
            # A nested field (``{x:>{w}}``) keeps its own text, so the spec
            # matches neither ``%`` pattern and stays an f-string argument.
            spec = "".join(_segment(source, v) for v in part.format_spec.values)
        conversion = "" if part.conversion == -1 else chr(part.conversion)
        value = _segment(source, part.value)
        if part.conversion == -1 and (match := _PERCENT_SPEC.match(spec)):
            kind = match["type"] or "s"
            fmt.append(f"%{match['flags']}{match['width']}{match['prec'] or ''}{kind}")
            args.append(value)
        elif part.conversion == -1 and (match := _ALIGN_SPEC.match(spec)):
            sign = "-" if match["align"] == "<" else ""
            fmt.append(f"%{sign}{match['width']}s")
            args.append(value)
        elif not spec:
            # An empty spec with no conversion took the first branch.
            fmt.append(f"%{conversion}")
            args.append(value)
        else:
            fmt.append("%s")
            bang = f"!{conversion}" if conversion else ""
            args.append(f'f"{{{value}{bang}:{spec}}}"')
    return "".join(fmt), args


def _segment(source: str, node: ast.expr) -> str:
    """Return ``node``'s original text, or ``ast.unparse`` when it has no span."""
    if node.end_col_offset is None:
        return ast.unparse(node)
    # ``ast.get_source_segment`` re-splits the whole file on every call; the
    # assert fixer asks once per assert, which made it quadratic in file size.
    starts = _cached_line_starts(source)
    return source[
        _offset(starts, node.lineno, node.col_offset) : _offset(
            starts,
            node.end_lineno,
            node.end_col_offset,
        )
    ] or ast.unparse(node)


# One entry: every call in a fix pass indexes the same file's text, and a larger
# cache would hold every linted file's source alive for the whole run.
@lru_cache(maxsize=1)
def _cached_line_starts(source: str) -> "_LineStarts":
    """``_line_starts`` memoized on the source text it indexes."""
    return _line_starts(source)


def _is_number_literal(node: ast.expr) -> bool:
    """Whether ``node`` is an int/float constant, possibly negated."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def _is_validation_guard(node: ast.If | ast.Assert) -> bool:
    """Whether ``node`` is an ``assert`` or an ``if`` whose body raises."""
    return isinstance(node, ast.Assert) or any(
        isinstance(stmt, ast.Raise) for stmt in node.body
    )


def _mode_violation(node: ast.Call) -> str:
    """Name a permission-mode violation on ``node``, or empty."""
    method = _call_text(node).rpartition(".")[2]
    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    kwargs = {k.arg: k.value for k in node.keywords if k.arg is not None}
    on_userdirs = receiver is not None and _contains_userdirs_call(receiver)
    on_scratch = receiver is not None and (
        any(
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith("/opt/scratch")
            for n in ast.walk(receiver)
        )
    )
    if method == "mkdir" and on_userdirs and not _mode_is(kwargs.get("mode"), 0o700):
        return "mkdir under a userdirs path without mode=0o700"
    if (
        method == "atomic_write_bytes"
        and node.args
        and _contains_userdirs_call(node.args[0])
        and not _mode_is(kwargs.get("file_mode"), 0o600)
    ):
        return "atomic_write_bytes under a userdirs path without file_mode=0o600"
    if "mode" in kwargs and on_scratch:
        return "mode= on a /opt/scratch path; the provisioned umask governs"
    if method == "chmod" and node.args and not on_userdirs:
        arg = node.args[0]
        # Owner-only modes (no group/other bits) mark a secret; a secret
        # belongs under userdirs. 0o755 and 0o644 are not private.
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, int)
            and arg.value & 0o677 in {0o200, 0o400, 0o600}
        ):
            return "chmod to a private mode on a non-userdirs path"
    return ""


def _mode_is(value: ast.expr | None, want: int) -> bool:
    """Whether ``value`` is the integer constant ``want``."""
    return isinstance(value, ast.Constant) and value.value == want


def _contains_userdirs_call(node: ast.AST) -> bool:
    """Whether any sub-expression is ``<x>_dir()``, at any nesting."""
    return any(_is_userdirs_call(n) for n in ast.walk(node) if isinstance(n, ast.expr))


def _public_defs(unit: Unit) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every def whose name is public and non-dunder."""
    for node in unit.nodes(*_DEF_KINDS):
        if not node.name.startswith("_"):
            yield node


def _sectioned_defs(
    unit: Unit,
    max_oneliner_body: int,
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]]:
    """Yield each public documented def held to sections, with those it has."""
    for node in _public_defs(unit):
        doc = ast.get_docstring(node) or ""
        if not doc or _has_decorator(node, "overload"):
            continue
        # An override inherits its contract; the base's docstring is the
        # one that documents the parameters.
        if _has_decorator(node, "setter") or _has_decorator(node, "override"):
            continue
        if "\n" not in doc.strip() and _body_lines(node) <= max_oneliner_body:
            continue
        yield node, {name for name, _ in _sections(doc)}


def _has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """Whether a decorator's dotted text ends in ``name``."""
    return any(_call_text(d).rpartition(".")[2] == name for d in node.decorator_list)


def _docstring_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Constant | None:
    """Return the docstring's Constant node, or None."""
    if node.body and _is_docstring(node.body[0]):
        expr = node.body[0]
        assert isinstance(expr, ast.Expr)
        assert isinstance(expr.value, ast.Constant)
        return expr.value
    return None


def _is_multiline_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``node``'s docstring spans more than one source line."""
    const = _docstring_node(node)
    return const is not None and isinstance(const.value, str) and "\n" in const.value


def _body_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Source lines the body spans, docstring excluded; 0 for a docstring-only def."""
    body = _without_docstring(node.body)
    if not body:
        return 0
    last = body[-1]
    return (last.end_lineno or last.lineno) - body[0].lineno + 1


def _split_private_docstring(
    text: str,
    *,
    summary_width: int,
    comment_width: int,
) -> tuple[str, list[str]]:
    """Split a docstring into a one-line summary and wrapped rationale lines."""
    paragraphs = [p for p in inspect.cleandoc(text).split("\n\n") if p.strip()]
    if not paragraphs:
        return text.strip(), []
    first = " ".join(paragraphs[0].split())
    rest: list[str] = []
    if len(first) > summary_width:
        sentence, sep, tail = first.partition(". ")
        if not sep:
            sentence, sep, tail = first.partition("; ")
            sentence += "."
        if tail:
            first, rest = sentence, [tail[0].upper() + tail[1:]]
    section = None
    for para in paragraphs[1:]:
        head = para.partition("\n")[0]
        section_head = _SECTION_HEAD.match(head)
        if section_head:
            section = section_head
            continue
        if section is not None and para.startswith(" "):
            continue
        section = None
        rest.append(" ".join(para.split()))
    prose: list[str] = []
    for para in rest:
        if prose:
            prose.append("")
        prose.extend(textwrap.wrap(para, width=comment_width))
    return first, prose


def _sections(doc: str) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(name, entries)`` for each Google-style section in ``doc``."""
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        m = _SECTION_HEAD.match(lines[i])
        if not m:
            i += 1
            continue
        name = str(m.group(1))
        entries: list[str] = []
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].startswith(" ")):
            text = lines[i]
            if text.strip():
                if text.startswith("    ") and entries:
                    entries[-1] += " " + text.strip()
                else:
                    entries.append(text.strip())
            i += 1
        yield name, entries


def _comment_blocks(
    tokens: Sequence[tokenize.TokenInfo],
) -> list[list[tokenize.TokenInfo]]:
    """Group consecutive full-line comments at one indent into blocks."""
    blocks: list[list[tokenize.TokenInfo]] = []
    previous: tokenize.TokenInfo | None = None
    for tok in tokens:
        if tok.type in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            continue
        if tok.type == tokenize.COMMENT:
            if (
                previous is not None
                and previous.type == tokenize.COMMENT
                and tok.start[0] == previous.start[0] + 1
                and tok.start[1] == previous.start[1]
                and tok.line[: tok.start[1]].strip() == ""
                and not _is_exempt_comment(_comment_text(tok))
            ):
                blocks[-1].append(tok)
            else:
                blocks.append([tok])
        previous = tok
    return blocks


def _comment_text(tok: tokenize.TokenInfo) -> str:
    """Comment body after the ``#`` and one space."""
    return tok.string.lstrip("#").strip()


def _bound_names(unit: Unit) -> frozenset[str]:
    """Every name the module imports, assigns, defines, or takes as a parameter."""
    return frozenset(
        (
            *(n.id for n in unit.nodes(ast.Name)),
            *((a.asname or a.name).partition(".")[0] for a in unit.nodes(ast.alias)),
            *(d.name for d in unit.nodes(*_DEF_KINDS, ast.ClassDef)),
            *(a.arg for a in unit.nodes(ast.arg)),
            *(a.attr for a in unit.nodes(ast.Attribute)),
        ),
    )


# Judged per block, never per line: a line is also what splits blocks, and
# a prose line opening on a capitalized word (``IGNORED: it was ...``) would
# split its paragraph and leave the line above judged as unterminated.
def _is_marker(block: Sequence[tokenize.TokenInfo]) -> bool:
    """Whether ``block`` is one all-caps line a tool matches verbatim."""
    return len(block) == 1 and _MARKER.match(_comment_text(block[0])) is not None


def _is_exempt_comment(text: str) -> bool:
    """Whether a comment is a directive, tag, URL, or section banner, not prose."""
    return (
        # PEP 723 inline script metadata opens with ``# /// script``; ``uv run``
        # refuses the block once its closing ``# ///`` gains a period.
        text.startswith((*_EXEMPT_PREFIXES, "///"))
        or re.match(r"^https?://\S+$", text) is not None
        or _BANNER.match(text) is not None
    )


_UNDERSCORE_PARAM_PROBE = re.compile(r"[(,]\s*\*{0,2}_[A-Za-z]")
# ruff's own S105/S106 name test (flake8_bandit/helpers.rs).
_SECRET_KEYWORD = re.compile(
    r"(^|_)(?i:pas+wo?r?d|pass(phrase)?|pwd|token|secrete?)($|_)",
)
_SECTION_HEAD = re.compile(
    r"^(Args|Returns|Yields|Raises|Attributes|References|Derivation|Examples?|Note|Notes):\s*$",
)
# The ``-- reason`` tail is required, as for every suppression; the fixer adds
# one, so a check that accepted its absence would leave that rewrite unreported.
_RUFF_NOQA_HEADER = re.compile(
    r"^# ruff: noqa: EXE003, D300, D205(, [A-Z]+\d+)* -- .+$",
)
_SENTENCE_END: Final = (".", ":", "?", "!", ")", '"', "`", "--")
_EXEMPT_PREFIXES: Final = (
    "!",
    "noqa",
    "type:",
    "pyright:",
    "ruff:",
    "fmt:",
    "vim:",
    "copybarista:",
    "house-ignore",
    "TODO",
    "FIXME",
    "NOTE",
    "isort:",
    "pragma:",
    "shellcheck",
    "mypy:",
    "ty:",
    "pyrefly:",
    "nosec",
    "codespell:",
)
_LOWER_WORD = re.compile(r"([a-z]+)[,.:;]?")
_BANNER = re.compile(r"^[-=\u2500]{2,}(?:\s.*\s[-=\u2500]{2,})?\.?$")
"""A ``# -- title ----`` section divider; a ruler, not a sentence."""

_MARKER = re.compile(r"^[A-Z][A-Z0-9_]*(?: [A-Z][A-Z0-9_]*)*(?::.*)?$")
"""An all-caps marker a tool matches verbatim (``# FULL_REPLACE``, ``# BEGIN X``)."""


_CLASS_KEYWORD = re.compile(r"\bclass\b")
"""Necessary-condition probe for the class rules; see ``DataclassRule.check``."""


_SUPPRESSION_PROBE = re.compile(
    r"noqa|pyright:|ty:|pyrefly:|type:\s*ignore|house-ignore",
)
"""Necessary-condition probe; a file without any of these has no suppression."""

# One alternation for every checker's inline form. ``noqa`` without a code list
# is still a suppression (ruff treats it as blanket) and still needs a reason.
_SUPPRESSION = re.compile(
    r"#\s*(?:noqa(?::\s*[A-Z]+\d+(?:\s*,\s*[A-Z]+\d+)*)?"
    r"|(?:pyright|ty|pyrefly):\s*ignore\[[^\]]*\]"
    r"|house-ignore\[[^\]]*\])",
)
_FMT_DIRECTIVE = re.compile(r"#\s*fmt:\s*(?:skip|off|on)\b")
"""A ruff/black formatter directive, ``on`` included: it only pairs an ``off``."""

_REASON_AFTER = re.compile(r"\s*--\s+\S.*?\.\s*(?:#|$)")
"""The tail a suppression must carry: ``-- Reason.`` ending before the line or next ``#``."""


# Measured cold, best of three, ``import X`` minus a bare interpreter on the
# dev box (2026-09). Re-measure when a floor moves; the numbers are the rule.
_IMPORT_COSTS_MS: Final[Mapping[str, int]] = {
    "torch": 1050,
    "torchvision": 400,
    "torchaudio": 250,
    "transformers": 900,
    "jax": 700,
    "flax": 450,
    "optax": 300,
    "numpy": 90,
    "scipy": 200,
    "pandas": 350,
    "sklearn": 400,
    "matplotlib": 300,
    "plotly": 250,
    "wandb": 600,
    "PIL": 60,
    "cv2": 120,
    "h5py": 80,
    "pyarrow": 150,
    "datasets": 500,
    "tiktoken": 70,
    "triton": 200,
    "einops": 60,
    "lpips": 400,
    "anthropic": 120,
    "openai": 150,
    "google": 200,
    "httpx": 60,
    "aiohttp": 80,
    "fastapi": 120,
    "uvicorn": 90,
    "pydantic": 70,
    "yaml": 30,
    "rich": 60,
    "prompt_toolkit": 80,
    "asyncpg": 60,
    "psycopg": 60,
    "duckdb": 90,
    "sqlalchemy": 120,
    "networkx": 150,
    "sympy": 500,
}

# Which heavy roots a heavy root's own import already loads, measured as
# ``sys.modules`` membership after ``import X`` in a fresh interpreter
# (2026-09). Only pairs where both sides are in ``_IMPORT_COSTS_MS`` matter.
_IMPLIED_IMPORTS: Final[Mapping[str, frozenset[str]]] = {
    "torch": frozenset({"numpy"}),
    "torchvision": frozenset({"PIL", "numpy", "torch"}),
    "jax": frozenset({"numpy"}),
    "flax": frozenset({"jax", "numpy", "rich"}),
    "optax": frozenset({"jax", "numpy"}),
    "scipy": frozenset({"numpy"}),
    "pandas": frozenset({"numpy", "pyarrow"}),
    "sklearn": frozenset({"numpy", "pandas", "pyarrow", "scipy"}),
    "matplotlib": frozenset({"PIL", "numpy"}),
    "pyarrow": frozenset({"numpy"}),
    "fastapi": frozenset({"pydantic"}),
    "httpx": frozenset({"rich"}),
    "transformers": frozenset({"PIL", "httpx", "numpy", "rich"}),
    "datasets": frozenset({"aiohttp", "httpx", "numpy", "pandas", "pyarrow", "rich"}),
    "wandb": frozenset({"numpy", "pydantic"}),
    "openai": frozenset({"aiohttp", "pydantic"}),
    "anthropic": frozenset({"aiohttp", "pydantic"}),
}


@dataclass(frozen=True, kw_only=True, slots=True)
class _NameUses:
    """How a module-scope name is used: in annotations vs. at runtime."""

    annotation: int = 0
    runtime: int = 0


# ``typing`` and ``typing_extensions`` are exempt: their names are the
# annotation vocabulary, and a reader expects them at the top of the imports.
_ANNOTATION_VOCABULARY = frozenset({"typing", "typing_extensions", "__future__"})


def _runtime_imports(tree: ast.Module) -> dict[str, tuple[int, str]]:
    """Map each name bound by a module-scope import outside ``TYPE_CHECKING``."""
    out: dict[str, tuple[int, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.partition(".")[0] in _ANNOTATION_VOCABULARY:
                    continue
                local = (alias.asname or alias.name).partition(".")[0]
                out[local] = (node.lineno, alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.partition(".")[0] not in _ANNOTATION_VOCABULARY
        ):
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                out[local] = (node.lineno, f"{node.module}.{alias.name}")
    return out


@dataclass(kw_only=True, slots=True)
class _UsageWalk:
    """Mutable tallies for ``_name_usage``; a class so the walker is not nested."""

    annotation: dict[str, int] = field(default_factory=dict[str, int])
    runtime: dict[str, int] = field(default_factory=dict[str, int])

    def visit(self, node: ast.AST, *, in_annotation: bool = False) -> None:
        """Tally ``node`` and recurse, flipping the flag at annotation sites.

        Args:
          node: Subtree to walk.
          in_annotation: Whether ``node`` sits inside an annotation.

        """
        if isinstance(node, ast.Name):
            tally = self.annotation if in_annotation else self.runtime
            tally[node.id] = tally.get(node.id, 0) + 1
        elif isinstance(node, ast.Constant):
            if in_annotation and isinstance(node.value, str):
                self._visit_string_annotation(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_function(node)
        elif isinstance(node, ast.AnnAssign):
            self.visit(node.annotation, in_annotation=True)
            if node.value is not None:
                self.visit(node.value)
            self.visit(node.target)
        elif _is_quoted_cast(node):
            # Only a QUOTED cast target is annotation-only; a bare name is
            # evaluated at call time and needs the runtime import.
            self.visit(node.args[0], in_annotation=True)
            for child in node.args[1:]:
                self.visit(child)
            for kw in node.keywords:
                self.visit(kw.value)
        elif isinstance(node, ast.If) and _is_type_checking_test(node.test):
            # Imports under TYPE_CHECKING are not runtime uses of anything.
            for child in node.orelse:
                self.visit(child, in_annotation=in_annotation)
        else:
            for child in ast.iter_child_nodes(node):
                self.visit(child, in_annotation=in_annotation)

    def _visit_string_annotation(self, text: str) -> None:
        """Tally names inside a string annotation, which names things too."""
        try:
            inner = ast.parse(text, mode="eval").body
        except SyntaxError:
            return
        self.visit(inner, in_annotation=True)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Annotations flip the flag; defaults, decorators, and the body do not."""
        args = node.args
        for arg in (
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            args.vararg,
            args.kwarg,
        ):
            if arg is not None and arg.annotation is not None:
                self.visit(arg.annotation, in_annotation=True)
        if node.returns is not None:
            self.visit(node.returns, in_annotation=True)
        for child in (*args.defaults, *args.kw_defaults):
            if child is not None:
                self.visit(child)
        for child in (*node.decorator_list, *node.body):
            self.visit(child)


def _name_usage(tree: ast.Module) -> dict[str, _NameUses]:
    """Count annotation vs. runtime uses of every name in ``tree``."""
    walk = _UsageWalk()
    walk.visit(tree)
    return {
        name: _NameUses(
            annotation=walk.annotation.get(name, 0),
            runtime=walk.runtime.get(name, 0),
        )
        for name in walk.annotation.keys() | walk.runtime.keys()
    }


def _is_quoted_cast(node: ast.AST) -> TypeIs[ast.Call]:
    """Whether ``node`` is ``cast("T", ...)`` with a string first argument."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cast"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
    )


def _is_type_checking_test(test: ast.expr) -> bool:
    """Whether ``test`` is ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_name_reads(unit: Unit) -> Counter[str]:
    """Count ``Load`` occurrences of each bare name anywhere in the module."""
    return Counter(n.id for n in unit.nodes(ast.Name) if isinstance(n.ctx, ast.Load))


def _inlineable_binding(stmt: ast.stmt) -> tuple[str, str] | None:
    """Name and kind of a ``Final`` literal or ``re.compile`` module binding."""
    if (
        isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and _annotation_is_final(stmt.annotation)
        and stmt.value is not None
        and _is_literal_value(stmt.value)
    ):
        return stmt.target.id, "Final literal"
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "compile"
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id == "re"
    ):
        return stmt.targets[0].id, "single-use re.compile"
    return None


def _eager_imports(tree: ast.Module) -> Iterator[ast.Import | ast.ImportFrom]:
    """Yield module-scope imports outside ``if TYPE_CHECKING:`` blocks."""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.Try):
            for child in (*node.body, *node.orelse, *node.finalbody):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    yield child
        elif isinstance(node, ast.If) and not _is_type_checking_test(node.test):
            for child in (*node.body, *node.orelse):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    yield child


def _imported_modules(node: ast.Import | ast.ImportFrom) -> Iterator[str]:
    """Yield the dotted module each alias of ``node`` loads."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name
    elif node.module is not None and node.level == 0:
        yield node.module


# A comment introduces the run of bindings under it, not only the first: a block
# of related bits (``_IN_MODIFY`` / ``_IN_CREATE`` / ``_IN_MOVED_TO``) is one
# documented group.
def _single_use_helpers(unit: Unit) -> list[_SingleUseHelper]:
    """Private top-level defs read once whose body is one ``return <expr>``."""
    tree = unit.tree
    if tree is None or "def _" not in unit.source:
        return []
    reads = _module_name_reads(unit)
    candidates: list[ast.FunctionDef] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name
        if not name.startswith("_") or name.startswith("__") or reads.get(name) != 1:
            continue
        if node.decorator_list or node.args.vararg or node.args.kwarg:
            continue
        if node.returns is not None and _base_name(node.returns) in (
            "TypeGuard",
            "TypeIs",
        ):
            continue
        body = _without_docstring(node.body)
        # A one-line ``return g(...)`` is ``forwarder-helper``'s at any count.
        if (
            len(body) == 1
            and isinstance(body[0], ast.Return)
            and body[0].value
            and not _is_forwarder(node)
        ):
            candidates.append(node)
    if not candidates:
        return []
    # One local read is not one read: the sibling test may monkeypatch the
    # name, and a package sibling may reach it as ``module._helper`` behind an
    # SLF001 suppression. Both are read from disk only once a candidate exists.
    names = {node.name for node in candidates}
    reached = _names_reached_by_siblings(unit.path, names)
    return [
        _SingleUseHelper(node=node, call=_single_call(unit, node.name))
        for node in candidates
        if node.name not in reached
    ]


def _inlinable_forwarders(
    unit: Unit,
) -> list[tuple[ast.FunctionDef, list[tuple[ast.Call, str]]]]:
    """Return forwarders called only in this file, with each call's inlined text."""
    tree = unit.tree
    if tree is None or unit.path.name == "__init__.py":
        return []
    reads = _module_name_reads(unit)
    exported = _dunder_all(tree)
    candidates = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and not node.name.startswith("__")
        and node.name not in exported
        and reads.get(node.name, 0) >= 1
        and not (node.decorator_list or node.args.vararg or node.args.kwarg)
        and not (
            node.returns is not None
            and _base_name(node.returns) in ("TypeGuard", "TypeIs")
        )
        and _is_forwarder(node)
    ]
    if not candidates:
        return []
    names = {n.name for n in candidates}
    # Deleting a name every caller of which is rewritten here is safe only if
    # no other module reaches it: a sibling (monkeypatch, ``mod._x``) or, as
    # ``live_reference_test`` does with ``games/*/kernel_test._action``, a
    # module in another directory that imports this one.
    reached = _names_reached_by_siblings(
        unit.path,
        names,
    ) | _names_reached_by_importers(
        unit.path,
        names,
    )
    found: list[tuple[ast.FunctionDef, list[tuple[ast.Call, str]]]] = []
    for node in candidates:
        if node.name in reached:
            continue
        replacements: list[tuple[ast.Call, str]] = []
        for use in unit.nodes(ast.Name):
            if use.id != node.name or not isinstance(use.ctx, ast.Load):
                continue
            call = unit.parents.get(use)
            if not (isinstance(call, ast.Call) and call.func is use):
                break
            # A sync helper is the boundary ruff's ASYNC rules check from;
            # inlined into a coroutine, its blocking I/O becomes a finding.
            if isinstance(_enclosing_def(unit, call), ast.AsyncFunctionDef):
                break
            # A secret-named keyword fed from a parameter is clean in the
            # helper; inlined beside a literal it becomes ruff S106.
            if any(
                k.arg is not None and _SECRET_KEYWORD.search(k.arg)
                for k in _forwarded_call(node).keywords
            ):
                break
            text = _inline_call(node, call, unit)
            if text is None:
                break
            replacements.append((call, text))
        else:
            found.append((node, replacements))
    return found


def _is_forwarder(node: ast.FunctionDef) -> bool:
    """Return whether ``node``'s body is one line, ``return g(...)``."""
    body = _without_docstring(node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    ret = body[0]
    if (ret.end_lineno or ret.lineno) != ret.lineno:
        return False
    call = ret.value
    if not isinstance(call, ast.Call):
        return False
    # ``cast(T, value)`` over a parameter is typed by that parameter's annotation;
    # inlined over ``"one"`` it becomes ty's ``disjoint-cast``.
    params = {
        a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    }
    if (
        isinstance(call.func, ast.Name)
        and call.func.id == "cast"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id in params
    ):
        return False
    return _call_states_return_type(node, call)


# The checker types the inlined call, not the deleted annotation. That is the
# same type only when the call spells it: ``Foo(...) -> Foo``, a target argument
# as in ``json_loads(x, T) -> T``, or ``cast(T, ...) -> T``. Otherwise the
# annotation narrowed an ``Any``/``Unknown`` (``-> Tensor`` over a numpy call)
# and inlining surfaces it at every call site.
def _call_states_return_type(node: ast.FunctionDef, call: ast.Call) -> bool:
    """Return whether ``call`` alone determines ``node``'s declared return type."""
    if node.returns is None or (
        isinstance(node.returns, ast.Name) and node.returns.id == "object"
    ):
        return True
    declared = ast.dump(node.returns)
    return any(
        ast.dump(_unquoted(part)) == declared
        for part in (call.func, *call.args, *(k.value for k in call.keywords))
    )


def _unquoted(expr: ast.expr) -> ast.expr:
    """Return the annotation a string literal spells, else ``expr`` itself."""
    if not (isinstance(expr, ast.Constant) and isinstance(expr.value, str)):
        return expr
    try:
        return ast.parse(expr.value, mode="eval").body
    except SyntaxError:
        return expr


def _forwarded_call(node: ast.FunctionDef) -> ast.Call:
    """Return the call a forwarder's ``return`` makes."""
    ret = _without_docstring(node.body)[0]
    assert isinstance(ret, ast.Return)
    assert isinstance(ret.value, ast.Call)
    return ret.value


def _names_reached_by_importers(path: Path, names: set[str]) -> frozenset[str]:
    """Names among ``names`` some tracked module importing ``path``'s module reads."""
    root = _repo_root(path)
    if root is None:
        return frozenset[str]()
    resolved = path.resolve()
    module = ".".join(resolved.relative_to(root).with_suffix("").parts)
    parent = module.rpartition(".")[0]
    reached: set[str] = set()
    for other, text in _python_texts(root):
        if other == resolved:
            continue
        # The stem is the cheap filter; an importer names the module's path.
        if path.stem not in text or (
            module not in text and not (parent and parent in text)
        ):
            continue
        reached.update(
            n for n in names if re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", text)
        )
    return frozenset(reached)


# The OUTERMOST project root: a nested package (``pkg/treekle/.export``) has
# its own ``pyproject.toml``, but its importers live in the enclosing monorepo.
def _repo_root(path: Path) -> Path | None:
    """Return the outermost project root (a ``pyproject.toml``) above ``path``."""
    roots = [p for p in path.resolve().parents if (p / "pyproject.toml").is_file()]
    return roots[-1] if roots else None


def _dunder_all(tree: ast.Module) -> frozenset[str]:
    """Names a module's ``__all__`` lists; a public API is reached by name elsewhere."""
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "__all__"
            and isinstance(node, ast.Assign | ast.AnnAssign)
            and isinstance(node.value, ast.List | ast.Tuple)
        ):
            return frozenset(
                e.value
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            )
    return frozenset[str]()


# Pruned while walking: ``rglob`` descended into ``.venv`` (63k files) only to
# filter it out, 3s per worker process against 0.1s here.
@cache
def _python_files(root: Path) -> tuple[Path, ...]:
    """Every ``.py`` file under ``root``, outside hidden and dependency trees."""
    found: list[Path] = []
    for directory, subdirectories, files in os.walk(root.resolve()):
        subdirectories[:] = sorted(
            d for d in subdirectories if not d.startswith(".") and d != "node_modules"
        )
        found.extend(
            Path(directory, name).resolve()
            for name in sorted(files)
            if name.endswith(".py") and not name.startswith(".")
        )
    return tuple(found)


# Read once per process: each linted forwarder module scanned every file again.
@cache
def _python_texts(root: Path) -> tuple[tuple[Path, str], ...]:
    """Each readable ``.py`` file under ``root`` with its UTF-8 text."""
    texts: list[tuple[Path, str]] = []
    for path in _python_files(root):
        try:
            texts.append((path, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return tuple(texts)


# ``_test.py`` siblings are matched on the bare name (a monkeypatch spells it as a
# string); other modules on the attribute spelling ``.name`` or an explicit
# ``from ... import name``, so a same-named private in an unrelated sibling does
# not shield this one.
def _names_reached_by_siblings(path: Path, names: set[str]) -> frozenset[str]:
    """Names among ``names`` that another module in ``path``'s directory mentions."""
    reached: set[str] = set()
    for sibling in sorted(path.parent.glob("*.py")):
        if sibling == path:
            continue
        try:
            text = sibling.read_bytes().decode()
        except (OSError, UnicodeDecodeError):
            continue
        mentioned = {name for name in names if name in text}
        if sibling.name.endswith("_test.py"):
            reached.update(mentioned)
            continue
        # A name absent from the text can be neither imported nor dotted, so
        # most siblings are never parsed.
        dotted = {name for name in mentioned if f".{name}" in text}
        reached.update(dotted)
        if not mentioned - dotted:
            continue
        try:
            sibling_tree = ast.parse(text)
        except (SyntaxError, ValueError):
            # Unreadable imports cannot be ruled out, so every mention counts.
            reached.update(mentioned)
            continue
        imported = {
            alias.name
            for node in ast.walk(sibling_tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        reached.update(mentioned & imported)
    return frozenset(reached)


def _single_call(unit: Unit, name: str) -> ast.Call | None:
    """Return the ``Call`` whose callee is the bare ``name``; None if used as a value."""
    child = next(
        (
            n
            for n in unit.nodes(ast.Name)
            if n.id == name and isinstance(n.ctx, ast.Load)
        ),
        None,
    )
    if child is None:
        return None
    parent = unit.parents.get(child)
    return parent if isinstance(parent, ast.Call) and parent.func is child else None


# Precedence is handled by parenthesizing conservatively: an argument that is not
# atomic is wrapped before substitution, and the whole expression is wrapped when
# the call sits inside a larger expression. ``ruff format`` strips what turns out
# to be redundant.
def _inline_call(node: ast.FunctionDef, call: ast.Call, unit: Unit) -> str | None:
    """Source text replacing ``call`` with ``node``'s body, or None if unsafe."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(
        k.arg is None for k in call.keywords
    ):
        return None
    # An empty container literal is typed by the parameter it is passed to;
    # substituted into ``{}.get(k)`` it is ``dict[Unknown, Unknown]`` and the
    # checker loses what the annotation said.
    if any(
        (isinstance(a, ast.Dict) and not a.keys)
        or (isinstance(a, ast.List | ast.Set | ast.Tuple) and not a.elts)
        for a in (*call.args, *(k.value for k in call.keywords))
    ):
        return None
    params = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    bound = _bind_arguments(node, call)
    if bound is None:
        return None
    body = _without_docstring(node.body)
    ret = body[0]
    assert isinstance(ret, ast.Return)
    if ret.value is None:
        raise ValueError("Expected ret.value is not None.")
    expr = ret.value
    tree = unit.tree
    if tree is None:
        raise ValueError("Expected tree is not None.")
    # A spelling is the parameter only where the def's own scope resolves it:
    # a lambda or comprehension in the body may bind the same name, and a
    # walrus that rebinds the parameter is a second occurrence of it.
    scopes = _cached_scopes(tree)
    refs = [
        n
        for n in ast.walk(expr)
        if isinstance(n, ast.Name) and n.id in bound and scopes.resolve(n.id, n) is node
    ]
    uses = Counter(n.id for n in refs)
    if any(uses[p.arg] != 1 for p in params):
        return None
    if not _keeps_bindings(node, call, unit) or not _keeps_evaluation(
        node,
        expr,
        {n: bound[n.id] for n in refs},
        unit,
    ):
        return None
    # A helper's return annotation can be load-bearing for the checker, and
    # the inlined expression would infer differently. Two shapes seen: a local
    # Protocol (``-> _TokenizerLike`` over an ``Any``-returning factory) and a
    # widened container (``-> dict[str, object]`` over ``{"a": s, "n": 1}``,
    # which infers the invariant ``dict[str, str | int]`` and no longer fits
    # a ``list[dict[str, object]]``).
    if node.returns is not None and (
        _base_name(node.returns) in _local_classes(unit)
        or isinstance(
            expr,
            ast.Dict
            | ast.List
            | ast.Set
            | ast.Tuple
            | ast.DictComp
            | ast.ListComp
            | ast.SetComp,
        )
    ):
        return None
    # Comments do not survive substitution: a suppression on the return line
    # would be dropped and the rule it silenced would fire at the call site,
    # and a rationale block above the def has no mechanical home in the caller.
    first_line, last_line = _def_span(node, unit.lines)
    if first_line + 1 != node.lineno or any(
        tok.type == tokenize.COMMENT and node.lineno <= tok.start[0] <= last_line
        for tok in unit.tokens
    ):
        return None
    # A string that itself spans lines (implicit concatenation, a wrapped
    # f-string) reflows into one literal the formatter cannot rewrap; a dict
    # or call that merely CONTAINS strings reflows fine.
    if any(
        (
            isinstance(n, ast.JoinedStr)
            or (isinstance(n, ast.Constant) and isinstance(n.value, str))
        )
        and n.end_lineno != n.lineno
        for n in ast.walk(expr)
    ):
        return None
    # A literal table the def existed to NAME (a language list, a dispatch
    # dict) is data, not an expression; inlining it buries the call site.
    if (
        isinstance(expr, ast.Tuple | ast.List | ast.Set | ast.Dict)
        and (expr.end_lineno or expr.lineno) - expr.lineno >= 3
    ):
        return None
    # A call site already inside an f-string cannot host a reflowed
    # expression: the quotes it carries would close the enclosing string.
    if isinstance(_parent_of(unit, call), ast.FormattedValue):
        return None
    starts = _line_starts(unit.source)
    text = ast.get_source_segment(unit.source, expr)
    if text is None:
        raise ValueError("Expected text is not None.")
    expr_start = _offset(starts, expr.lineno, expr.col_offset)
    pieces: list[tuple[int, int, str]] = []
    for n in refs:
        s, e = _span(starts, n)
        pieces.append((s - expr_start, e - expr_start, _atomic(bound[n.id])))
    for s, e, arg_text in sorted(pieces, reverse=True):
        text = text[:s] + arg_text + text[e:]
    if "\n" in text:
        # Reflow a wrapped expression onto one line; ``ruff format`` rewraps.
        # Re-parsing keeps bracket spacing canonical where a whitespace join
        # would leave ``{ i: v }``. Parenthesized so a wrapped implicit
        # string concatenation, indented on its continuation lines, parses.
        # ``ast.unparse`` re-quotes every string, so an f-string whose
        # quotes it would have to nest is left to the author.
        if any(isinstance(n, ast.JoinedStr) for n in ast.walk(expr)):
            return None
        text = ast.unparse(ast.parse(f"({text})"))
    # The whole body is wrapped only where the call site can see it: as an
    # operand of a larger expression, or when the body is itself a
    # conditional or lambda that any neighbour would rebind. As a call
    # argument, a subscript index, or a statement's value it is already
    # delimited, and a redundant pair leaves ``UP034`` for the author.
    parent = _parent_of(unit, call)
    delimited = isinstance(
        parent,
        ast.Call
        | ast.Subscript
        | ast.stmt
        | ast.keyword
        | ast.FormattedValue
        | ast.Tuple
        | ast.List
        | ast.Set
        | ast.Dict,
    )
    # A call body binds tighter than any operator, so it never needs the pair.
    if (
        isinstance(parent, ast.expr)
        and not delimited
        and not isinstance(expr, ast.Call)
    ) or isinstance(expr, ast.IfExp | ast.Lambda):
        text = f"({text})"
    return text


# A default ran once, when the def did. Copied to the call site it runs at every
# call, so only a constant, which evaluates to the same immutable value each time,
# may be.
def _bind_arguments(
    node: ast.FunctionDef,
    call: ast.Call,
) -> dict[str, ast.expr] | None:
    """Map each parameter to its argument, or None if the call cannot be bound."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    bound: dict[str, ast.expr] = {}
    if len(call.args) > len(positional):
        return None
    for index, arg in enumerate(call.args):
        bound[positional[index].arg] = arg
    keywordable = {p.arg for p in [*args.args, *args.kwonlyargs]}
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError("Expected kw.arg is not None.")
        if kw.arg not in keywordable or kw.arg in bound:
            return None
        bound[kw.arg] = kw.value
    first_default = len(positional) - len(args.defaults)
    defaults = {
        positional[first_default + index].arg: d
        for index, d in enumerate(args.defaults)
    }
    defaults.update(
        (args.kwonlyargs[index].arg, d)
        for index, d in enumerate(args.kw_defaults)
        if d is not None
    )
    for param in [*positional, *args.kwonlyargs]:
        if param.arg in bound:
            continue
        default = defaults.get(param.arg)
        if not isinstance(default, ast.Constant):
            return None
        bound[param.arg] = default
    return bound


def _atomic(expr: ast.expr) -> str:
    """Source for ``expr``, parenthesized unless it binds tighter than anything."""
    text = ast.unparse(expr)
    # ``ast.unparse`` parenthesizes every tuple, a subscript's index included.
    if isinstance(
        expr,
        ast.Name
        | ast.Constant
        | ast.Attribute
        | ast.Call
        | ast.Subscript
        | ast.List
        | ast.Tuple
        | ast.Set
        | ast.Dict
        | ast.JoinedStr,
    ):
        return text
    return f"({text})"


def _parent_of(unit: Unit, target: ast.AST) -> ast.AST | None:
    """Return the node whose direct child is ``target``."""
    return unit.parents.get(target)


def _enclosing_def(
    unit: Unit,
    target: ast.AST,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the innermost function whose body contains ``target``."""
    node = unit.parents.get(target)
    while node is not None and not isinstance(
        node,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ):
        node = unit.parents.get(node)
    return node


def _local_classes(unit: Unit) -> frozenset[str]:
    """Names of classes defined at this module's top level."""
    tree = unit.tree
    if tree is None:
        raise ValueError("Expected tree is not None.")
    return frozenset(n.name for n in tree.body if isinstance(n, ast.ClassDef))


# Inlining moves the helper's text into the call site's scopes and the call's
# arguments into the body's, so every name must bind there as it did here. A
# type parameter binds only inside the def. The body's other names resolve at
# module scope, and any def, lambda, comprehension, or class body around the call
# that binds one captures it. An argument that lands inside the body's own lambda
# or comprehension is captured by a name that scope binds, and stops seeing a
# class body around the call. A walrus in the body binds into the caller. A
# ``__name`` in the body would be mangled in a class around the call. That is
# not checked: nothing outside a class is mangled, so only a ``setattr`` or a
# module global named so on purpose answers to one.
def _keeps_bindings(node: ast.FunctionDef, call: ast.Call, unit: Unit) -> bool:
    """Return whether every name inlining ``call`` moves resolves as it did."""
    tree = unit.tree
    if tree is None:
        raise ValueError("Expected tree is not None.")
    ret = _without_docstring(node.body)[0]
    assert isinstance(ret, ast.Return)
    if ret.value is None:
        raise ValueError("Expected ret.value is not None.")
    scopes = _cached_scopes(tree)
    params = {a.arg for a in _all_args(node)}
    if node.type_params or scopes.bound[node] - params:
        return False
    body = [n for n in ast.walk(ret.value) if isinstance(n, ast.Name)]
    if any(
        scopes.resolve(n.id, n) is tree and scopes.resolve(n.id, call) is not tree
        for n in body
    ):
        return False
    site = {
        n.id
        for part in (*call.args, *call.keywords)
        for n in ast.walk(part)
        if isinstance(n, ast.Name)
    }
    for n in body:
        scope = scopes.owner[n]
        if n.id not in params or scope is node:
            continue
        if isinstance(scopes.owner[call], ast.ClassDef):
            return False
        while scope is not node:
            if site & scopes.bound[scope]:
                return False
            scope = scopes.enclosing[scope]
    return True


# A call evaluates each argument once, in order, before the body runs; inlined, an
# argument evaluates where its parameter is read. That matches only where the def's
# own scope reads it unconditionally and ahead of every call the body makes. A
# branch of ``if``/``else``, the right of ``and``/``or``, or a later link of a
# chained comparison runs it never or once; a comprehension runs it per element;
# a lambda, whenever it is called. A constant is the same value wherever it runs,
# and so is a name, except in a lambda: called later, it sees whatever the caller
# rebound since, the very bug a closure factory (``lambda v: v + k`` per loop
# iteration) exists to prevent. A generator reads late too, but every one the
# rules inline across the repo is consumed where it is built.
def _keeps_evaluation(
    node: ast.FunctionDef,
    expr: ast.expr,
    reads: Mapping[ast.Name, ast.expr],
    unit: Unit,
) -> bool:
    """Return whether each argument ``reads`` maps to runs as the call ran it."""
    tree = unit.tree
    if tree is None:
        raise ValueError("Expected tree is not None.")
    scopes = _cached_scopes(tree)
    starts = _cached_line_starts(unit.source)
    computed: list[ast.Name] = []
    for read, value in reads.items():
        scope = scopes.owner[read]
        if isinstance(value, ast.Name):
            while scope is not node:
                if isinstance(scope, ast.Lambda):
                    return False
                scope = scopes.enclosing[scope]
        elif not isinstance(value, ast.Constant):
            if scope is not node or not _unconditional(read, expr, unit):
                return False
            computed.append(read)
    if not computed:
        return True
    # On the unconditional path of one scope, source order is evaluation order. A
    # comprehension runs its calls when it is reached, after its first iterable.
    computed.sort(key=lambda n: _span(starts, n)[0])
    order = [_span(starts, reads[n])[0] for n in computed]
    last = _span(starts, computed[-1])[0]
    return order == sorted(order) and not any(
        isinstance(
            n,
            ast.Call | ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        )
        and scopes.owner[n] is node
        and _span(starts, n)[1] <= last
        and any(isinstance(m, ast.Call) for m in ast.walk(n))
        for n in ast.walk(expr)
    )


def _unconditional(node: ast.AST, root: ast.expr, unit: Unit) -> bool:
    """Whether ``node`` runs every time ``root`` does, under no branch between them."""
    child = node
    while child is not root:
        parent = unit.parents[child]
        if (
            (isinstance(parent, ast.IfExp) and child is not parent.test)
            or (isinstance(parent, ast.BoolOp) and child is not parent.values[0])
            or (
                isinstance(parent, ast.Compare)
                and child is not parent.left
                and child is not parent.comparators[0]
            )
        ):
            return False
        child = parent
    return True


# One entry, as for ``_cached_line_starts``: every call site a pass inlines
# resolves its names against the same file.
@lru_cache(maxsize=1)
def _cached_scopes(tree: ast.Module) -> "_Scopes":
    """``_Scopes.build`` memoized on the tree it indexes."""
    return _Scopes.build(tree)


def _has_leading_comment(lines: Sequence[str], lineno: int) -> bool:
    """Whether a ``#`` comment sits above ``lineno`` (1-based) or its run."""
    row = lineno - 2
    while row >= 0 and lines[row].strip():
        if lines[row].lstrip().startswith("#"):
            return True
        row -= 1
    return False


def _import_locals(node: ast.Import | ast.ImportFrom) -> Iterator[str]:
    """Yield the local name each alias of ``node`` binds."""
    for alias in node.names:
        if alias.name != "*":
            yield (alias.asname or alias.name).partition(".")[0]


# A def's signature (annotations, defaults) and decorators are evaluated when the
# ``def`` statement runs, so they count as module scope; only the body is deferred.
# A class body runs at import time too, so its statements are module scope while its
# methods are not.
def _module_scope_loads(tree: ast.Module) -> frozenset[str]:
    """Names loaded when the module is imported, outside every function body."""
    out: set[str] = set()
    pending: list[ast.AST] = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pending.extend(node.decorator_list)
            pending.append(node.args)
            if node.returns is not None:
                pending.append(node.returns)
            continue
        if isinstance(node, ast.ClassDef):
            pending.extend((*node.decorator_list, *node.bases, *node.body))
            pending.extend(kw.value for kw in node.keywords)
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.add(node.id)
        pending.extend(ast.iter_child_nodes(node))
    return frozenset(out)


# Cached rather than a module constant because ``min_digits`` is a setting: a constant
# would pin the default. See ``DigitsRule.fix``.
@cache
def _digit_run(min_digits: int) -> re.Pattern[str]:
    """Return the compiled run-of-N-digits probe for one threshold."""
    return re.compile(rf"\d{{{min_digits}}}")


# Scans the module body and the body of any block statement in it, rather than
# ``ast.walk``: an import binds a name usable by a decorator, and a decorator is
# evaluated at class-definition time, so only imports at module scope or inside a
# module-scope ``if``/``try`` are reachable. Walking every node instead cost 196 ms over
# 422 files -- the whole tree, to find statements that live in the first screen of it.
def _dataclass_bindings(tree: ast.Module) -> _DataclassBindings:
    """Resolve every local name that reaches ``dataclasses.dataclass``."""
    direct: set[str] = set()
    modules: set[str] = set()
    for node in _module_scope_statements(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "dataclasses"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
            direct.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "dataclass"
            )
    return _DataclassBindings(direct=frozenset(direct), modules=frozenset(modules))


# ``if TYPE_CHECKING:`` and ``try: ... except ImportError:`` are the two shapes that
# routinely hold an import at module scope, so a name bound there is as usable by a
# decorator as one bound at the top level. A function or class body is not descended
# into: a name bound there is local to it and cannot be the decorator on a module-scope
# class.
def _module_scope_statements(tree: ast.Module) -> Iterator[ast.AST]:
    """Yield module-body statements, descending only into block statements."""
    for node in tree.body:
        yield node
        if isinstance(node, (ast.If, ast.Try)):
            yield from ast.walk(node)


def _is_dataclass_decorator(node: ast.expr, bound: _DataclassBindings) -> bool:
    """Whether ``node`` names the dataclass decorator (call already stripped)."""
    if isinstance(node, ast.Name):
        return node.id in bound.direct
    if isinstance(node, ast.Attribute):
        return node.attr == "dataclass" and _base_name(node.value) in bound.modules
    return False


def _missing_kwargs(
    node: ast.expr,
    required: tuple[str, ...],
    bound: _DataclassBindings,
) -> tuple[str, ...]:
    """Return the required kwargs a ``@dataclass`` decorator does not set True."""
    if _is_dataclass_decorator(node, bound):
        return required
    if not isinstance(node, ast.Call) or not _is_dataclass_decorator(node.func, bound):
        return ()
    present = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
    return tuple(
        key
        for key in required
        if not (
            isinstance(value := present.get(key), ast.Constant) and value.value is True
        )
    )


def _is_slotted(node: ast.ClassDef, bound: _DataclassBindings) -> bool:
    """Whether a ``@dataclass`` decorator on ``node`` sets ``slots=True``."""
    for dec in node.decorator_list:
        if not isinstance(dec, ast.Call) or not _is_dataclass_decorator(
            dec.func,
            bound,
        ):
            continue
        for kw in dec.keywords:
            if (
                kw.arg == "slots"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ):
                return True
    return False


# Calls inside a nested class are that class's problem, not this one's: the nested
# definition is visited separately by the caller's walk.
def _zero_arg_supers(node: ast.ClassDef) -> tuple[int, ...]:
    """Line numbers of zero-arg ``super()`` calls owned by ``node``."""
    nested = {n for n in ast.walk(node) if isinstance(n, ast.ClassDef)} - {node}
    disowned = {id(n) for cls in nested for n in ast.walk(cls)}
    return tuple(
        sub.lineno
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "super"
        and not sub.args
        and id(sub) not in disowned
    )


def _is_knob(stmt: ast.Assign | ast.AnnAssign) -> bool:
    """Whether a module-level assignment is a tunable-value-as-global."""
    name = _target_name(stmt)
    if name is None:
        return False
    if _is_dunder(name):
        return False
    if name in {"logger", "log", "_log"}:
        return False
    if (
        isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.annotation, ast.Name)
        and stmt.annotation.id == "TypeAlias"
    ):
        return False
    if isinstance(stmt, ast.AnnAssign) and _annotation_is_final(stmt.annotation):
        # ``NAME: Final = ...`` explicitly (and type-checker-enforced) declares
        # a constant, not a tunable to relocate -- the typed alternative to a
        # ``house-ignore[globals]`` marker.
        return False
    value = stmt.value
    if value is None:
        return False
    # A literal is never a call, so only a ``X | Y`` of literals needs exempting.
    return _is_literal_value(value) and not _is_type_expression(value)


def _target_name(stmt: ast.Assign | ast.AnnAssign) -> str | None:
    """Return the single bound name, or None for tuple/attribute targets."""
    if isinstance(stmt, ast.AnnAssign):
        return stmt.target.id if isinstance(stmt.target, ast.Name) else None
    if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id
    return None


# This is the shape a config field default should hold. A non-empty literal container
# (``("a", "b")`` / ``{1: 2}``) counts; an EMPTY one does not -- an empty container is
# almost always a mutable registry populated at import, not a value, and needs no
# suppression.
def _is_literal_value(value: ast.expr) -> bool:
    """Whether the RHS is a literal scalar or a literal collection of scalars."""
    if isinstance(value, ast.Constant):
        return isinstance(value.value, (int, float, bool, str, bytes))
    if isinstance(value, ast.UnaryOp) and isinstance(value.operand, ast.Constant):
        return isinstance(value.operand.value, (int, float))  # ``-7.0``
    if isinstance(value, ast.BinOp):
        # A knob written as an arithmetic/bit expression of literals
        # (``1 << 20``, ``5 * 60``) is still a literal value. Both operands
        # must themselves be literals; ``NAME + 10`` (a derived global ref) is
        # caught by the separate global-reference signal, not here.
        return _is_literal_value(value.left) and _is_literal_value(value.right)
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return bool(value.elts) and all(_is_literal_value(e) for e in value.elts)
    if isinstance(value, ast.Dict):
        return bool(value.keys) and all(
            k is not None and _is_literal_value(k) and _is_literal_value(v)
            for k, v in zip(value.keys, value.values, strict=True)
        )
    return False


# Any uppercase-initial subscript base is also treated as a type (``Mapping[...]`` / a
# user alias), since config knobs are scalars.
def _is_type_expression(
    value: ast.expr,
    *,
    type_bases: frozenset[str] = frozenset(
        {
            "Literal",
            "Callable",
            "Union",
            "Optional",
            "Annotated",
            "ClassVar",
            "Final",
            "dict",
            "list",
            "tuple",
            "set",
            "frozenset",
            "type",
        },
    ),
) -> bool:
    """Whether the RHS is a type alias rather than a runtime value."""
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
        return True  # ``X | Y`` union alias.
    if isinstance(value, ast.Subscript):
        head = value.value
        if isinstance(head, ast.Attribute):
            return True  # ``typing.Mapping[...]`` etc.
        if isinstance(head, ast.Name):
            return head.id in type_bases or head.id[:1].isupper()
    return False


def _annotation_is_final(annotation: ast.expr) -> bool:
    """Whether an AnnAssign is annotated ``Final`` / ``Final[...]``."""
    node = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    if isinstance(node, ast.Name):
        return node.id == "Final"
    if isinstance(node, ast.Attribute):
        return node.attr == "Final"
    return False


def _base_name(base: ast.expr) -> str:
    """Return the head name of a (possibly subscripted) base expression."""
    node = base.value if isinstance(base, ast.Subscript) else base
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


# A Config field default must be an inline literal (or ``field(...)`` / a sentinel / an
# injected lower-case strategy), never a reference to a module-level constant. The
# discriminator is the ALL-CAPS convention: ``hidden_dim: int = DEFAULT_HIDDEN_DIM`` is
# a violation, while ``init_weight: InitFn = kaiming_uniform`` is dependency injection.
def _constant_default(field: ast.AnnAssign) -> str | None:
    """Return the offending constant name if a Config field defaults to one."""
    value = field.value
    if isinstance(value, ast.Name):
        name = value.id
    elif isinstance(value, ast.Attribute):
        name = value.attr  # ``module.DEFAULT_X``
    else:
        return None
    return name if name.isupper() else None


# Nbformat's ``multiline_string``: one string, or a list of lines. A non-string
# line is dropped rather than rejecting the cell.
def _multiline_string(value: object) -> str:
    """Join a notebook cell ``source`` into one text, or ``""`` when malformed."""
    if isinstance(value, str):
        return value
    lines = _listed(value)
    return "".join(line for line in lines if isinstance(line, str))


# Takes an already-tokenized stream so the dispatcher's one tokenization serves both the
# check and the fix. Splices TEXT: a token's columns are character offsets, so the same
# arithmetic against bytes shifts by one per non-ASCII character earlier on the line.
def _regroup(
    source: str,
    tokens: Sequence[tokenize.TokenInfo],
    min_digits: int,
) -> str | None:
    """Rewrite long integer literals in ``source``, or None when unchanged."""
    replacements: list[tuple[int, int, int, str]] = []
    for tok in tokens:
        if tok.type != tokenize.NUMBER:
            continue
        raw = tok.string
        if len(raw.replace("_", "")) < min_digits:
            continue
        grouped = _group(raw)
        if grouped != raw:
            replacements.append((tok.start[0], tok.start[1], tok.end[1], grouped))
    if not replacements:
        return None
    lines = source.split("\n")
    for row, start, end, grouped in reversed(replacements):
        line = lines[row - 1]
        lines[row - 1] = line[:start] + grouped + line[end:]
    return "\n".join(lines)


# Non-decimal bases keep their own grouping conventions, and a float or complex literal
# is not an integer, so both pass through untouched.
#
# Groups the digit STRING from the right rather than formatting ``int(literal)``: the
# round-trip dropped leading zeros (``00000`` came back ``0``) and raised above
# CPython's 4300-digit integer-conversion limit, which reached the command line as a
# traceback.
def _group(literal: str) -> str:
    """Return ``literal`` with underscore separators, or unchanged."""
    if not literal.isdigit():
        return literal
    head = len(literal) % 3 or 3
    return "_".join(
        [literal[:head], *(literal[i : i + 3] for i in range(head, len(literal), 3))],
    )


def _staticmethod_sites(
    unit: Unit,
    /,
) -> tuple[
    list[tuple[ast.Name, ast.FunctionDef | ast.AsyncFunctionDef]],
    list[tuple[ast.Assign | ast.AnnAssign, str]],
]:
    """Return non-autograd ``@staticmethod`` decorators and ``staticmethod()`` binds."""
    decorated = [
        (dec, node)
        for node in unit.nodes(ast.FunctionDef, ast.AsyncFunctionDef)
        if not (
            node.name in {"forward", "backward"}
            and _is_autograd_function_class(unit.parents.get(node))
        )
        for dec in node.decorator_list
        if isinstance(dec, ast.Name) and dec.id == "staticmethod"
    ]
    bound = [
        (stmt, target.id)
        for cls in unit.nodes(ast.ClassDef)
        if not _is_autograd_function_class(cls)
        for stmt in cls.body
        if isinstance(stmt, ast.Assign | ast.AnnAssign)
        and _is_staticmethod_call(stmt.value)
        for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
        if isinstance(target, ast.Name)
    ]
    return decorated, bound


def _forwarder(
    stmt: ast.Assign | ast.AnnAssign,
    /,
    *,
    name: str,
    functions: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> str | None:
    """Return a ``@classmethod`` forwarding to the def ``stmt`` binds, if local."""
    call = stmt.value
    assert isinstance(call, ast.Call)
    if (
        (isinstance(stmt, ast.Assign) and len(stmt.targets) != 1)
        or len(call.args) != 1
        or call.keywords
        or not isinstance(call.args[0], ast.Name)
    ):
        return None
    fn = functions.get(call.args[0].id)
    if fn is None or fn.decorator_list:
        return None
    params = fn.args
    positional = [*params.posonlyargs, *params.args]
    every = [*positional, *params.kwonlyargs, params.vararg, params.kwarg]
    if any(param is not None and param.arg == "cls" for param in every):
        return None
    args: list[ast.expr] = [ast.Name(param.arg) for param in positional]
    if params.vararg:
        args.append(ast.Starred(ast.Name(params.vararg.arg)))
    forwarded = ast.Call(
        func=ast.Name(fn.name),
        args=args,
        keywords=[
            ast.keyword(param.arg, ast.Name(param.arg)) for param in params.kwonlyargs
        ]
        + ([ast.keyword(None, ast.Name(params.kwarg.arg))] if params.kwarg else []),
    )
    first = fn.body[0]
    docstring = (
        [first]
        if isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
        else []
    )
    is_async = isinstance(fn, ast.AsyncFunctionDef)
    kind = ast.AsyncFunctionDef if is_async else ast.FunctionDef
    method = kind(
        name=name,
        args=ast.arguments(
            posonlyargs=[ast.arg("cls"), *params.posonlyargs]
            if params.posonlyargs
            else [],
            args=params.args if params.posonlyargs else [ast.arg("cls"), *params.args],
            vararg=params.vararg,
            kwonlyargs=params.kwonlyargs,
            kw_defaults=params.kw_defaults,
            kwarg=params.kwarg,
            defaults=params.defaults,
        ),
        body=[*docstring, ast.Return(ast.Await(forwarded) if is_async else forwarded)],
        decorator_list=[ast.Name("classmethod")],
        returns=fn.returns,
        type_params=fn.type_params,
    )
    return ast.unparse(ast.fix_missing_locations(method))


def _is_staticmethod_call(value: ast.expr | None) -> bool:
    """Whether ``value`` is a ``staticmethod(...)`` call."""
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "staticmethod"
    )


def _insert_cls(
    lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    /,
) -> None:
    """Add a leading ``cls`` parameter to ``node``, in place on ``lines``."""
    args = node.args
    first = (args.posonlyargs or args.args or [None])[0]
    if first is not None:
        if first.arg == "cls":
            return
        index = first.lineno - 1
        column = first.col_offset
        lines[index] = lines[index][:column] + "cls, " + lines[index][column:]
        return
    # No positional parameter: `cls` still leads, but `*args` / `*` /
    # `**kwargs` after it need the separator. They may start on the line
    # below the `(`, so the signature decides, not the rest of the line.
    index = node.lineno - 1
    while "(" not in lines[index]:
        index += 1
    column = lines[index].index("(") + 1
    if args.vararg is None and not args.kwonlyargs and args.kwarg is None:
        separator = "cls"
    elif lines[index][column:].strip():
        separator = "cls, "
    else:
        separator = "cls,"
    lines[index] = lines[index][:column] + separator + lines[index][column:]


def _is_declarative_value(value: ast.expr, calls: Sequence[str]) -> bool:
    """Whether a module-level RHS builds a value rather than performing work."""
    if isinstance(
        value,
        (
            ast.Constant,
            ast.Name,
            ast.Attribute,
            ast.Subscript,
            ast.Tuple,
            ast.List,
            ast.Set,
            ast.Dict,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Lambda,
            ast.JoinedStr,
        ),
    ):
        return True
    if isinstance(value, ast.UnaryOp):
        return _is_declarative_value(value.operand, calls)
    if isinstance(value, ast.BinOp):
        return _is_declarative_value(value.left, calls) and _is_declarative_value(
            value.right,
            calls,
        )
    if isinstance(value, ast.BoolOp):
        return all(_is_declarative_value(v, calls) for v in value.values)
    if isinstance(value, ast.IfExp):
        return all(
            _is_declarative_value(v, calls)
            for v in (value.test, value.body, value.orelse)
        )
    if isinstance(value, ast.Call):
        # ``ContextVar[bool](...)`` and ``dict[str, int]()`` are the same
        # constructors with a subscript; judge the head name beneath it.
        func = value.func
        if isinstance(func, ast.Subscript):
            func = func.value
        head = _call_text(func).rpartition(".")[2]
        # A CapWords callee is a class: constructing one is declaring a value.
        # A private class (``_Spec()``) is still a class.
        return head in calls or head.lstrip("_")[:1].isupper()
    return False


def _is_test_file(path: Path) -> bool:
    """Whether ``path`` is a pytest module by house naming."""
    return path.name.endswith("_test.py") or (
        path.name.startswith("test_") and path.suffix == ".py"
    )


# The fix covers the mechanical letters only: (a) header/trailer/mode, (d) `main`
# return type and guard body, (e) inline `add_argument` -> `_add_arguments`, (f) the
# `description=` idiom. (b) adding a header and (c) `__main__.py` contents are
# design decisions and stay with `check()`. `sys.exit` inside `main` and a parser
# not bound from `argparse.ArgumentParser(...)` are left alone too: the first
# needs the author to decide the code, the second may register flags elsewhere.
def _fix_cli(unit: Unit) -> str | None:
    """Rewrite one file toward the cli skill's shape; None when nothing changes."""
    tree = unit.tree
    lines = unit.source.splitlines(keepends=True)
    polyglot = unit.source.startswith("#!/bin/sh\n")
    main = (
        None
        if tree is None
        else next(
            (
                d
                for d in tree.body
                if isinstance(d, ast.FunctionDef) and d.name == "main"
            ),
            None,
        )
    )
    if main is not None and any(
        isinstance(s, ast.Call) and _call_text(s) in ("sys.exit", "exit")
        for s in ast.walk(main)
    ):
        return None
    # An unparseable polyglot file can still have its header and trailer
    # repaired and its mode set; everything else needs the AST.
    if tree is not None:
        # In-place edits first, while the original line numbers still hold;
        # moving the flags shifts every line below, so ``main`` is re-read
        # from the moved text before its trailing ``return 0`` is placed.
        guard = next(
            (s for s in tree.body if isinstance(s, ast.If) and _is_main_guard(s)),
            None,
        )
        if (
            guard is not None
            and len(guard.body) == 1
            and ast.unparse(guard.body[0]) == "main()"
        ):
            stmt = guard.body[0]
            row = stmt.lineno - 1
            lines[row] = (
                lines[row][: stmt.col_offset]
                + "raise SystemExit(main())"
                + lines[row][stmt.end_col_offset :]
            )
        if main is not None:
            _fix_description(unit, lines, polyglot=polyglot)
            _fix_add_arguments(main, lines)
            main = next(
                d
                for d in ast.parse("".join(lines)).body
                if isinstance(d, ast.FunctionDef) and d.name == "main"
            )
            _fix_main_returns(main, lines, noreturn=_noreturn_defs(tree))
    if polyglot:
        _fix_polyglot_header(lines)
        _fix_polyglot_trailer(lines)
        _ensure_executable(unit.path)
    rewritten = "".join(lines)
    return rewritten if rewritten != unit.source else None


def _fix_polyglot_trailer(lines: list[str]) -> None:
    """End ``lines`` with the ``# vim: ft=python`` modeline."""
    # A modeline with trailing whitespace is replaced, not followed by a second.
    if lines and lines[-1].strip() == "# vim: ft=python":
        lines.pop()
    if lines:
        lines.append(lines.pop().rstrip("\n") + "\n")
    lines.append("# vim: ft=python\n")


# The executable bit is part of shape (a); it is the one edit a returned source
# string cannot carry, so the fixer sets it on disk as a side effect.
def _ensure_executable(path: Path) -> None:
    """Set the executable bits on ``path`` when it lacks them."""
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if not mode & stat.S_IXUSR:
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _noreturn_defs(tree: ast.Module) -> frozenset[str]:
    """Names of this module's top-level functions annotated ``NoReturn``/``Never``."""
    return frozenset(
        d.name
        for d in tree.body
        if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef))
        and d.returns is not None
        and ast.unparse(d.returns).rpartition(".")[2] in {"NoReturn", "Never"}
    )


def _fix_main_returns(
    main: ast.FunctionDef,
    lines: list[str],
    *,
    noreturn: frozenset[str],
) -> None:
    """Annotate ``main -> int`` and make every exit path return a code."""
    # Rewriting an existing ``-> int`` is a no-op, so no test of the annotation.
    for r in range(main.lineno - 1, main.body[0].lineno - 1):
        line = lines[r]
        if line.rstrip().endswith((":", ":  # noqa")):
            if main.returns is None:
                head, colon, tail = line.rpartition(":")
                lines[r] = f"{head} -> int{colon}{tail}"
            else:
                # The return annotation is the last arrow; a source spelling
                # ``unparse`` does not reproduce is left for the author.
                head, arrow, tail = line.rpartition(f"-> {ast.unparse(main.returns)}")
                lines[r] = f"{head}-> int{tail}" if arrow else line
            break
    for node in ast.walk(main):
        if isinstance(node, ast.Return) and node.value is None:
            row = node.lineno - 1
            # ``end_col_offset`` counts UTF-8 bytes, not characters.
            col = len(lines[row].encode()[: node.end_col_offset].decode())
            lines[row] = lines[row][:col] + " 0" + lines[row][col:]
    last = main.body[-1]
    if _falls_through(main.body, noreturn=noreturn):
        end = last.end_lineno or last.lineno
        indent = " " * main.body[0].col_offset
        lines.insert(end, f"{indent}return 0\n")


# ``noreturn`` names this module's ``NoReturn`` functions; ``exit``-named calls
# (``sys.exit``, ``parser.exit``, ``os._exit``) never return either.
def _falls_through(body: Sequence[ast.stmt], *, noreturn: frozenset[str]) -> bool:
    """Whether control can run off the end of ``body``; unknown shapes can."""
    if not body:
        return True
    last = body[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return False
    if isinstance(last, ast.Expr) and isinstance(last.value, ast.Call):
        callee = _call_text(last.value)
        return callee not in noreturn and callee.rpartition(".")[2] not in {
            "exit",
            "_exit",
        }
    if isinstance(last, ast.If):
        return _falls_through(last.body, noreturn=noreturn) or _falls_through(
            last.orelse,
            noreturn=noreturn,
        )
    if isinstance(last, (ast.With, ast.AsyncWith)):
        return _falls_through(last.body, noreturn=noreturn)
    if isinstance(last, (ast.Try, ast.TryStar)):
        if last.finalbody and not _falls_through(last.finalbody, noreturn=noreturn):
            return False
        main_path = last.orelse or last.body
        return _falls_through(main_path, noreturn=noreturn) or any(
            _falls_through(handler.body, noreturn=noreturn) for handler in last.handlers
        )
    return True


def _description_idiom(*, polyglot: bool) -> str:
    """Return the ``description=`` expression the cli skill pins for this shape."""
    # Total under ``python -OO``, where ``__doc__`` is None: ``(__doc__ or "")``
    # sliced at ``[2]`` raises IndexError, and argparse takes ``description=None``.
    return (
        '__doc__.split("\\n", 2)[2] if __doc__ else None'
        if polyglot
        else '(__doc__ or "").strip()'
    )


# The rule is about how the MODULE docstring is sliced; a literal or another
# object's docstring (``cls.__doc__``) is not the module's and is not subject to
# the idiom. ``check`` reports these and ``fix`` rewrites the same ones.
def _wrong_descriptions(
    unit: Unit,
    *,
    polyglot: bool,
) -> Iterator[tuple[ast.Call, ast.expr]]:
    """Yield each ``ArgumentParser`` call and its off-idiom module-doc description."""
    want = ast.unparse(ast.parse(_description_idiom(polyglot=polyglot)).body[0])
    for node in unit.nodes(ast.Call):
        if not _call_text(node).endswith("ArgumentParser"):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "description"
                and any(
                    isinstance(n, ast.Name) and n.id == "__doc__"
                    for n in ast.walk(kw.value)
                )
                and ast.unparse(kw.value) != want
            ):
                yield node, kw.value


def _fix_description(unit: Unit, lines: list[str], *, polyglot: bool) -> None:
    """Rewrite each single-line off-idiom ``description=`` to the skill's idiom."""
    want = _description_idiom(polyglot=polyglot)
    for _, value in _wrong_descriptions(unit, polyglot=polyglot):
        if value.lineno != value.end_lineno:
            continue
        row = value.lineno - 1
        lines[row] = (
            lines[row][: value.col_offset] + want + lines[row][value.end_col_offset :]
        )


def _fix_add_arguments(main: ast.FunctionDef, lines: list[str]) -> None:
    """Move every ``parser.add_argument(...)`` in ``main`` into ``_add_arguments``."""
    parsers = [
        s.targets[0].id
        for s in main.body
        if isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Name)
        and isinstance(s.value, ast.Call)
        and _call_text(s.value).endswith("ArgumentParser")
    ]
    if not parsers:
        return
    parser_name = parsers[-1]
    sites = [
        s
        for s in main.body
        if isinstance(s, ast.Expr)
        and isinstance(s.value, ast.Call)
        and isinstance(s.value.func, ast.Attribute)
        and s.value.func.attr == "add_argument"
        and isinstance(s.value.func.value, ast.Name)
        and s.value.func.value.id == parser_name
    ]
    if not sites:
        return
    indent = " " * sites[0].col_offset
    moved = ["".join(lines[s.lineno - 1 : (s.end_lineno or s.lineno)]) for s in sites]
    first_row = sites[0].lineno - 1
    for s in reversed(sites):
        del lines[s.lineno - 1 : (s.end_lineno or s.lineno)]
    lines.insert(first_row, f"{indent}_add_arguments({parser_name})\n")
    removed = sum((s.end_lineno or s.lineno) - s.lineno + 1 for s in sites) - 1
    end = (main.end_lineno or main.lineno) - removed
    helper = [
        "\n",
        "\n",
        f"def _add_arguments({parser_name}: argparse.ArgumentParser) -> None:\n",
        *moved,
    ]
    lines[end:end] = helper


def _fix_polyglot_header(lines: list[str]) -> None:
    """Repair lines 2-5 of a ``#!/bin/sh`` header toward the pinned template."""
    if len(lines) < 2:
        return
    noqa = lines[1].rstrip()
    match = re.match(
        r"^# ruff: noqa: EXE003, D300, D205((?:, [A-Z]+\d+)*)( -- .*)?$",
        noqa,
    )
    if match is None:
        extra: list[str] = []
        if noqa.startswith("# ruff: noqa"):
            extra = [match.group(1) for match in re.finditer(r"\b([A-Z]+\d+)\b", noqa)]
        extra = [c for c in extra if c not in ("EXE003", "D300", "D205")]
        tail = ("".join(f", {c}" for c in extra)) + " -- Polyglot shell/Python script."
        fixed = f"# ruff: noqa: EXE003, D300, D205{tail}\n"
        # A missing noqa line is inserted; overwriting line 2 would shift the
        # pinned lines onto the docstring summary below them.
        if noqa.startswith("# ruff: noqa"):
            lines[1] = fixed
        else:
            lines.insert(1, fixed)
    elif match.group(2) is None:
        lines[1] = noqa + " -- Polyglot shell/Python script.\n"
    # Same skips as ``CliShapeRule._header``: extra ruff file-level suppression
    # lines and their continuation comments precede the pinned lines, and the
    # shell prelude may hold lines before ``exec uv``.
    start = 2
    while (
        start < len(lines)
        and lines[start].startswith("# ")
        and not lines[start].startswith("# fmt:")
    ):
        start += 1
    # As with the noqa line, a pinned line is rewritten only over a drifted copy
    # of itself (a comment, a ``''''`` opener). Anything else in its slot is the
    # author's, so the pinned line goes in above it; overwriting would shift each
    # line onto the next, down to the docstring summary.
    for offset, (want, stem) in enumerate(
        zip(_POLYGLOT_HEADER[:2], ("#", 4 * "'"), strict=True),
        start=start,
    ):
        if offset == len(lines) or not lines[offset].startswith(stem):
            lines.insert(offset, want + "\n")
        elif lines[offset].rstrip() != want:
            lines[offset] = want + "\n"
    exec_at = start + 2
    while exec_at < len(lines) and not lines[exec_at].startswith("exec uv "):
        if lines[exec_at].startswith(3 * "'"):
            break
        exec_at += 1
    if exec_at < len(lines) and lines[exec_at].startswith("exec uv "):
        return
    # Another ``exec`` before the docstring is the launcher to replace, with the
    # lines its trailing backslashes continue it onto (as ``_header`` joins them).
    # With none, the house line goes in above what follows the opener: with no
    # shell prelude, that is the docstring summary.
    launch = next(
        (row for row in range(start + 2, exec_at) if lines[row].startswith("exec ")),
        None,
    )
    if launch is None:
        lines.insert(start + 2, _POLYGLOT_HEADER[2] + "\n")
        return
    end = launch + 1
    while end < len(lines) and lines[end - 1].rstrip().endswith("\\"):
        end += 1
    lines[launch:end] = [_POLYGLOT_HEADER[2] + "\n"]


def _is_main_guard(stmt: ast.If) -> bool:
    """Whether ``stmt`` is ``if __name__ == "__main__":``."""
    test = stmt.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


def _is_docstring(stmt: ast.stmt) -> bool:
    """Whether ``stmt`` is a bare string expression."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _without_docstring(body: Sequence[ast.stmt]) -> Sequence[ast.stmt]:
    """Return ``body`` minus a leading docstring."""
    return body[1:] if body and _is_docstring(body[0]) else body


def _call_text(node: ast.expr) -> str:
    """Return the dotted callee of a call, or the dotted name of an expression."""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return f"{_call_text(target.value)}.{target.attr}"
    return ""


def _is_userdirs_call(node: ast.expr) -> bool:
    """Whether ``node`` is ``<x>_dir()`` for a userdirs helper (any spelling)."""
    return (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and _call_text(node).rpartition(".")[2] in _USERDIRS_HELPERS
    )


def _is_userdirs_path(node: ast.expr | None) -> bool:
    """Whether ``node`` is ``<x>_dir() / ...`` at any depth of ``/``."""
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        node = node.left
    return node is not None and _is_userdirs_call(node)


def _xdg_violation(node: ast.expr) -> str:
    """Name how ``node`` re-derives the userdirs layout, or empty."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if _XDG_PREFIX in node.value:
            return f"literal {node.value!r}"
        return ""
    if isinstance(node, ast.Call):
        text = _call_text(node)
        if text == "Path.home":
            return "Path.home()"
        if text in ("os.path.expanduser", "expanduser"):
            return "expanduser"
        if text in ("os.environ.get", "environ.get", "os.getenv", "getenv") and (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "HOME"
        ):
            return "$HOME lookup"
        return ""
    if (
        isinstance(node, ast.Subscript)
        and _call_text(node.value).endswith("environ")
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "HOME"
    ):
        return "$HOME lookup"
    return ""


def _lazy_finding(unit: Unit, line: int, rule: str) -> Finding:
    """Build the one lazy_import finding shape."""
    return Finding(
        path=unit.path,
        line=line,
        rule=rule,
        message=(
            "lazy_import bound other than `from wrapt import lazy_import` "
            "(STYLE.md Code Organization)"
        ),
    )


def _escapes(
    inner: ast.FunctionDef | ast.AsyncFunctionDef,
    outer: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Whether ``inner`` is used as a value (a closure) rather than only called."""
    if inner.decorator_list:
        return True
    callees = {
        id(node.func)
        for node in ast.walk(outer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return any(
        isinstance(node, ast.Name)
        and node.id == inner.name
        and isinstance(node.ctx, ast.Load)
        and id(node) not in callees
        for node in ast.walk(outer)
    )


def _direct_nested_defs(
    outer: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield defs nested in ``outer``'s body, not descending into classes."""
    stack: list[ast.AST] = list(outer.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.ClassDef):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
            continue
        stack.extend(ast.iter_child_nodes(node))


def _statements_in(stmt: ast.stmt) -> Iterator[ast.stmt]:
    """Yield ``stmt`` and every statement beneath it, not crossing a def."""
    stack: list[ast.AST] = [stmt]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.stmt):
            yield node
        stack.extend(ast.iter_child_nodes(node))


# A private def is excused when any LATER top-level statement needs its name at
# definition time: a def/class header, a class-body attribute, or a module-level
# statement (a constant built from it, a call to it).
def _misplaced_privates(
    tree: ast.Module,
) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, str]]:
    """Return ``(private_def, public_name)`` for each private def a public one follows."""
    defs = ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
    out: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, str]] = []
    body = tree.body
    # A def under a module-level `if`/`try` is still a module-scope binding, so
    # the closure must follow calls into it too.
    by_name = {s.name: s for s in _module_scope_defs(body) if isinstance(s, defs)}
    for index, stmt in enumerate(body):
        if not isinstance(stmt, defs) or not stmt.name.startswith("_"):
            continue
        later = body[index + 1 :]
        public = next(
            (
                s.name
                for s in later
                if isinstance(s, defs) and not s.name.startswith("_")
            ),
            None,
        )
        if public is not None and stmt.name not in _needed_at_module_time(
            later,
            by_name,
        ):
            out.append((stmt, public))
    return out


def _module_scope_defs(statements: Sequence[ast.stmt]) -> list[ast.stmt]:
    """Statements at module scope, descending into compound statements' branches."""
    out: list[ast.stmt] = []
    for s in statements:
        out.append(s)
        if isinstance(s, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            out.extend(
                _module_scope_defs(
                    [c for c in ast.iter_child_nodes(s) if isinstance(c, ast.stmt)],
                ),
            )
    return out


# A module-level ``_bind()`` runs ``_bind``'s body at import, so every name that body
# reads is needed then too; the closure is taken to a fixpoint.
def _needed_at_module_time(
    statements: Sequence[ast.stmt],
    by_name: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef],
) -> frozenset[str]:
    """Names ``statements`` evaluate at module time, following calls into local defs."""
    # The ``if __name__`` guard is the program's entry, not a module-time
    # dependency: everything is reachable from it, which would excuse anything.
    needed: set[str] = set()
    for s in statements:
        if not (isinstance(s, ast.If) and _is_main_guard(s)):
            needed |= _definition_time_names(s)
    frontier = [n for n in needed if n in by_name]
    while frontier:
        name = frontier.pop()
        body_names = {n.id for n in ast.walk(by_name[name]) if isinstance(n, ast.Name)}
        for other in body_names - needed:
            needed.add(other)
            if other in by_name:
                frontier.append(other)
    return frozenset(needed)


def _definition_time_names(stmt: ast.stmt) -> frozenset[str]:
    """Names ``stmt`` evaluates when the module runs, not when something is called."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _names_in_header(stmt)
    if isinstance(stmt, ast.ClassDef):
        # A class body runs at definition time, nested classes included; only
        # method BODIES defer.
        return _names_in_header(stmt).union(
            *(_definition_time_names(member) for member in stmt.body),
        )
    if isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
        # Branch bodies run at module time too; walk them the same way rather
        # than treating the compound statement as one opaque expression.
        parts = [
            _definition_time_names(child)
            for child in ast.iter_child_nodes(stmt)
            if isinstance(child, ast.stmt)
        ]
        exprs = [
            frozenset(n.id for n in ast.walk(child) if isinstance(n, ast.Name))
            for child in ast.iter_child_nodes(stmt)
            if isinstance(child, ast.expr)
        ]
        names: set[str] = set()
        for part in (*parts, *exprs):
            names |= part
        return frozenset(names)
    return frozenset(n.id for n in ast.walk(stmt) if isinstance(n, ast.Name))


def _def_span(stmt: ast.stmt, lines: Sequence[str]) -> tuple[int, int]:
    """0-based ``[start, end)`` line span of ``stmt`` plus its leading comment block."""
    first = stmt
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        first = stmt.decorator_list[0] if stmt.decorator_list else stmt
    start = first.lineno - 1
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    end = stmt.end_lineno or stmt.lineno
    # A bare string right after a def is not its docstring; leave it.
    return start, end


def _names_in_header(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> frozenset[str]:
    """Names a definition evaluates before its body: bases, decorators, signature."""
    parts: list[ast.AST] = list(node.decorator_list)
    if isinstance(node, ast.ClassDef):
        parts.extend(node.bases)
        parts.extend(kw.value for kw in node.keywords)
    else:
        parts.append(node.args)
        if node.returns is not None:
            parts.append(node.returns)
    return frozenset(
        sub.id for part in parts for sub in ast.walk(part) if isinstance(sub, ast.Name)
    )


def _all_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Sequence[ast.arg]:
    """Every parameter of ``node`` in declaration order."""
    args = node.args
    return [*args.posonlyargs, *args.args, *args.kwonlyargs]


@cache
def _emdash_probe(kind: str) -> re.Pattern[str]:
    """Return the em dash, plus its JSON escape when the unit is a notebook."""
    codepoint = chr(0x2014)
    escape = "\\" + "u2014"
    return re.compile(
        re.escape(codepoint) + ("|" + re.escape(escape) if kind == "jupyter" else ""),
    )


def _is_file_path_call(node: ast.AST) -> TypeIs[ast.Call]:
    """Whether ``node`` is exactly ``Path(__file__)``."""
    return (
        isinstance(node, ast.Call)
        and _call_text(node) == "Path"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "__file__"
    )


def _cwd_replacements(
    unit: Unit,
    tree: ast.Module,
) -> tuple[list[tuple[int, int, str]], bool]:
    """Return ``(start, end, text)`` source splices and whether ``_THIS`` is needed."""
    parents = unit.parents
    starts = _cached_line_starts(unit.source)
    canonical = {
        stmt.lineno
        for stmt in tree.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and ast.unparse(stmt.annotation) == "Final"
        and stmt.value is not None
        and (stmt.target.id, ast.unparse(stmt.value))
        in (
            ("_CWD", "Path(__file__).resolve().parent"),
            ("_THIS", "Path(__file__).resolve()"),
        )
    }
    # Each ``Path(__file__)`` call is the root of exactly one rewrite, chosen by
    # walking up its parents: ``.resolve()`` may sit between it and ``.parent``
    # or ``.parents[n]`` (called or not), and a ``/`` makes it a directory.
    replacements: list[tuple[int, int, str]] = []
    for node in unit.nodes(ast.Call):
        if not _is_file_path_call(node) or node.lineno in canonical:
            continue
        top: ast.expr = node
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.attr == "resolve":
            top = parent
            called = parents.get(parent)
            if isinstance(called, ast.Call) and called.func is parent:
                top = called
            parent = parents.get(top)
        if isinstance(parent, ast.Attribute) and parent.attr == "parent":
            top, text = parent, "_CWD"
        elif (
            isinstance(parent, ast.Attribute)
            and parent.attr == "parents"
            and isinstance(subscript := parents.get(parent), ast.Subscript)
            and isinstance(subscript.slice, ast.Constant)
            and type(subscript.slice.value) is int
            and subscript.slice.value >= 1
        ):
            depth = subscript.slice.value
            top = subscript
            text = "_CWD.parent" if depth == 1 else f"_CWD.parents[{depth - 1}]"
        elif (
            top is node
            and isinstance(parent, ast.BinOp)
            and isinstance(parent.op, ast.Div)
            and parent.left is node
        ):
            text = "_CWD"
        else:
            text = "_THIS"
        replacements.append(
            (
                _offset(starts, top.lineno, top.col_offset),
                _offset(starts, top.end_lineno, top.end_col_offset),
                text,
            ),
        )
    return replacements, any(text == "_THIS" for *_, text in replacements)


def _insert_cwd_bindings(source: str, *, needs_this: bool) -> str:
    """Insert missing typing and path bindings after the import block."""
    tree = ast.parse(source)
    names = {
        stmt.target.id
        for stmt in tree.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    bindings: list[str] = []
    if needs_this and "_THIS" not in names:
        bindings.append("_THIS: Final = Path(__file__).resolve()")
    if "_CWD" not in names:
        bindings.append(
            "_CWD: Final = _THIS.parent"
            if needs_this
            else "_CWD: Final = Path(__file__).resolve().parent",
        )
    if not bindings:
        return source
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_final = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "typing"
        and any(alias.name == "Final" for alias in node.names)
        for node in imports
    )
    lines = source.splitlines(keepends=True)
    insert_at = max((node.end_lineno or node.lineno for node in imports), default=0)
    head = (
        lines[:insert_at]
        if imported_final
        else [
            *lines[:insert_at],
            "from typing import Final\n",
        ]
    )
    prefix = "".join(head).rstrip("\n")
    suffix = "".join(lines[insert_at:]).lstrip("\n")
    return prefix + "\n\n" + "\n".join(bindings) + "\n\n" + suffix


@dataclass(frozen=True, kw_only=True, slots=True)
class _LineStarts:
    """Character offset of each one-based line, plus the lines themselves.

    Attributes:
      starts: ``starts[i]`` is the character offset where line ``i + 1`` begins.
      lines: Source lines without terminators, for byte-to-character conversion.

    """

    starts: tuple[int, ...]

    lines: tuple[str, ...]


def _line_starts(source: str) -> _LineStarts:
    """Return absolute character offsets for one-based source lines."""
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(source) if char == "\n")
    return _LineStarts(starts=tuple(starts), lines=tuple(source.split("\n")))


# ``ast`` columns count UTF-8 BYTES; ``str`` slicing counts characters. Every
# non-ASCII character before the column shifts the two apart by one or more, so
# the byte prefix is decoded to find the character column. Measured: a ``→`` in
# a log message put the splice one character short and cut the closing paren.
def _offset(starts: _LineStarts, line: int | None, column: int | None) -> int:
    """Convert an AST line and byte column to an absolute character offset."""
    if line is None:
        raise ValueError("Expected line is not None.")
    if column is None:
        raise ValueError("Expected column is not None.")
    text = starts.lines[line - 1]
    if text.isascii():
        return starts.starts[line - 1] + column
    return starts.starts[line - 1] + len(text.encode()[:column].decode())


def _span(starts: _LineStarts, node: ast.expr) -> tuple[int, int]:
    """Return absolute offsets for an expression node."""
    return (
        _offset(starts, node.lineno, node.col_offset),
        _offset(starts, node.end_lineno, node.end_col_offset),
    )


# Spans never overlap here -- each caller's nodes are disjoint sites -- so a
# bottom-up splice keeps every earlier offset valid.
def _splice_nodes(unit: Unit, replacements: Mapping[ast.expr, str]) -> str | None:
    """Replace each node's source text; None when ``replacements`` is empty."""
    if not replacements:
        return None
    starts = _cached_line_starts(unit.source)
    source = unit.source
    for (start, end), text in sorted(
        ((_span(starts, node), text) for node, text in replacements.items()),
        reverse=True,
    ):
        source = source[:start] + text + source[end:]
    return source


def _module_bindings(unit: Unit) -> set[str]:
    """Collect names bound anywhere in the module."""
    return {
        *(n.id for n in unit.nodes(ast.Name) if isinstance(n.ctx, ast.Store)),
        *(a.arg for a in unit.nodes(ast.arg)),
        *(d.name for d in unit.nodes(ast.FunctionDef)),
        *(d.name for d in unit.nodes(ast.AsyncFunctionDef)),
        *(d.name for d in unit.nodes(ast.ClassDef)),
        *(
            (alias.asname or alias.name).partition(".")[0]
            for node in unit.nodes(ast.Import) + unit.nodes(ast.ImportFrom)
            for alias in node.names
        ),
    }


def _attribute_chains(
    unit: Unit,
) -> list[tuple[tuple[str, ...], list[ast.Attribute | ast.Name]]]:
    """Return every outermost dotted-name chain with its per-segment nodes."""
    chains: list[tuple[tuple[str, ...], list[ast.Attribute | ast.Name]]] = []
    for node in unit.nodes(ast.Attribute):
        parent = unit.parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue
        result = _dotted_parts(node)
        if result is not None:
            chains.append((tuple(result[0]), result[1]))
    return chains


def _longest_import(
    chain: tuple[str, ...],
    *,
    dotted: AbstractSet[tuple[str, ...]],
) -> tuple[str, ...] | None:
    """Return the longest dotted import that prefixes ``chain``."""
    return next(
        (chain[:k] for k in range(len(chain), 0, -1) if chain[:k] in dotted),
        None,
    )


def _import_edits(
    source: str,
    *,
    starts: _LineStarts,
    imports: Sequence[ast.Import | ast.ImportFrom],
    statements: Mapping[ast.alias, Sequence[str]],
) -> Iterator[tuple[int, int, str]]:
    """Yield one splice per import statement, swapping in each alias's ``statements``."""
    for node in imports:
        if not any(alias in statements for alias in node.names):
            continue
        texts = [
            text
            for alias in node.names
            for text in statements.get(alias)
            or [
                f"from {node.module} import {ast.unparse(alias)}"
                if isinstance(node, ast.ImportFrom)
                else f"import {ast.unparse(alias)}",
            ]
        ]
        start = _offset(starts, node.lineno, node.col_offset)
        prefix = source[_offset(starts, node.lineno, 0) : start]
        # An import sharing its line (``if X: import a.b``) has no line of its
        # own to break: a newline would move every statement after the first
        # out of the block.
        yield (
            start,
            _offset(starts, node.end_lineno, node.end_col_offset),
            ("; " if prefix.strip() else f"\n{prefix}").join(texts),
        )


def _dotted_parts(
    node: ast.Attribute | ast.Name,
) -> tuple[list[str], list[ast.Attribute | ast.Name]] | None:
    """Return dotted names and their AST nodes from root through ``node``."""
    if isinstance(node, ast.Name):
        return [node.id], [node]
    if not isinstance(node.value, (ast.Attribute, ast.Name)):
        return None
    result = _dotted_parts(node.value)
    if result is None:
        return None
    parts, nodes = result
    return [*parts, node.attr], [*nodes, node]


@dataclass(frozen=True, kw_only=True, slots=True)
class _DottedImports:
    """One file's unaliased dotted imports and the reads that reach each.

    Attributes:
      unit: The file.
      scopes: Its scopes, for deciding which binding a read sees.
      dotted: Each ``import a.b`` statement and alias, in source order.
      paths: Every unaliased import path, plain ``import a`` included, keyed
        by the scope its root is bound in and that root.
      chains: Each attribute chain, keyed by the binding its root reads.
      lazy_annotations: Whether ``from __future__ import annotations`` keeps
        every annotation from running.

    """

    unit: Unit

    scopes: "_Scopes"

    dotted: tuple[tuple[ast.Import, ast.alias], ...]

    paths: dict[tuple[ast.AST, str], set[tuple[str, ...]]]

    chains: dict[
        tuple[ast.AST, str],
        list[tuple[tuple[str, ...], list[ast.Attribute | ast.Name]]],
    ]

    lazy_annotations: bool

    @classmethod
    def build(cls, unit: Unit) -> Self | None:
        """Index ``unit``; None when it does not parse or has no dotted import.

        Args:
          unit: The file under inspection.

        Returns:
          found: The index, or None when there is no dotted import to judge.

        """
        tree = unit.tree
        dotted = sorted(
            (
                (node, alias)
                for node in unit.nodes(ast.Import)
                for alias in node.names
                if alias.asname is None and "." in alias.name
            ),
            key=lambda pair: (pair[0].lineno, pair[0].col_offset),
        )
        if tree is None or not dotted:
            return None
        scopes = _Scopes.build(tree)
        paths: dict[tuple[ast.AST, str], set[tuple[str, ...]]] = {}
        for node in unit.nodes(ast.Import):
            for alias in node.names:
                if alias.asname is None:
                    parts = tuple(alias.name.split("."))
                    paths.setdefault((scopes.owner[node], parts[0]), set()).add(parts)
        chains: dict[
            tuple[ast.AST, str],
            list[tuple[tuple[str, ...], list[ast.Attribute | ast.Name]]],
        ] = {}
        for chain, nodes in _attribute_chains(unit):
            key = (scopes.resolve(chain[0], nodes[0]), chain[0])
            if key in paths:
                chains.setdefault(key, []).append((chain, nodes))
        return cls(
            unit=unit,
            scopes=scopes,
            dotted=tuple(dotted),
            paths=paths,
            chains=chains,
            lazy_annotations=any(
                node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
                for node in unit.nodes(ast.ImportFrom)
            ),
        )

    def verdicts(self) -> list[tuple[ast.Import, ast.alias, str, bool]]:
        """Pair each dotted import with why it must stay.

        Returns:
          verdicts: Each statement, alias, reason, and whether the remedy is
            an alias, in source order; the reason is empty when the fix may
            rewrite the import.

        """
        leaves = Counter(
            name.rpartition(".")[2] for name in {alias.name for _, alias in self.dotted}
        )
        return [
            (
                node,
                alias,
                *self._reason(
                    node,
                    parts=tuple(alias.name.split(".")),
                    shared=leaves[alias.name.rpartition(".")[2]] > 1,
                ),
            )
            for node, alias in self.dotted
        ]

    def rewrite(self) -> str | None:
        """Rewrite every import ``verdicts`` clears, and each read reaching it.

        Returns:
          rewritten: The full new source, or None when nothing is cleared.

        """
        groups: dict[tuple[ast.AST, str], list[tuple[ast.Import, ast.alias]]] = {}
        for node, alias, reason, _ in self.verdicts():
            if not reason:
                key = (self.scopes.owner[node], alias.name.partition(".")[0])
                groups.setdefault(key, []).append((node, alias))
        if not groups:
            return None
        source = self.unit.source
        starts = _line_starts(source)
        comments = {
            node: _trailing_comment(source, starts=starts, node=node)
            for group in groups.values()
            for node, _ in group
        }
        statements: dict[ast.alias, list[str]] = {}
        edits = [(start, end, "") for start, end, _ in filter(None, comments.values())]
        for (scope, root), group in groups.items():
            covered = self._read_edits(scope, root=root, group=group, edits=edits)
            # A trailing comment (a deferred import's ``noqa: PLC0415``)
            # outgrows ``from a import b``'s line; ruff's isort then wraps it
            # and moves the comment onto ``b,``, where ruff deems the noqa
            # unused and deletes it. After ``(`` it stays on the first line,
            # and the trailing comma stops isort joining the lines back.
            # Each statement replacing the import keeps its comment.
            for node, alias in group:
                parent, _, leaf = alias.name.rpartition(".")
                comment = comments[node]
                indent = source[
                    _offset(starts, node.lineno, 0) : _offset(
                        starts,
                        node.lineno,
                        node.col_offset,
                    )
                ]
                statements[alias] = [
                    f"from {parent} import (  {comment[2]}\n{indent}    {leaf},\n"
                    f"{indent})"
                    if comment
                    else f"from {parent} import {leaf}",
                ]
            if not self._stranded(scope, root=root, covered=covered):
                continue
            for node, alias in self._unbound_at(scope, root=root, group=group):
                comment = comments[node]
                statements[alias].insert(
                    0,
                    f"import {root}  {comment[2]}" if comment else f"import {root}",
                )
        edits.extend(
            _import_edits(
                source,
                starts=starts,
                imports=list(dict.fromkeys(node for node, _ in self.dotted)),
                statements=statements,
            ),
        )
        for start, end, replacement in sorted(edits, reverse=True):
            source = source[:start] + replacement + source[end:]
        return source

    def _read_edits(
        self,
        scope: ast.AST,
        *,
        root: str,
        group: Sequence[tuple[ast.Import, ast.alias]],
        edits: list[tuple[int, int, str]],
    ) -> set[ast.AST]:
        """Append a splice for each read the group's rewrite reaches; return roots."""
        starts = _cached_line_starts(self.unit.source)
        done = {tuple(alias.name.split(".")) for _, alias in group}
        covered: set[ast.AST] = set()
        # Subscript, not ``.get(..., [])`` as in ``_reason``: only a group
        # ``verdicts`` cleared reaches here, and clearing requires a read
        # through this ``(scope, root)``, so the key is always present. A
        # default here would be unreachable, and an unreachable default hides
        # the day that stops being true.
        for chain, nodes in self.chains[scope, root]:
            match = _longest_import(chain, dotted=self.paths[scope, root])
            if match is None or match not in done:
                continue
            # The span of ``a.b`` itself, not from ``a``: in ``(a).b`` the
            # attribute starts at the ``(``, which ``a`` does not cover.
            edits.append((*_span(starts, nodes[len(match) - 1]), match[-1]))
            covered.add(nodes[0])
        return covered

    # ``import a.b`` bound ``a`` wherever it ran; each rewritten import whose
    # block no kept or earlier rewritten import has already run through needs
    # its own ``import a``, or a read of ``a`` on that path finds it unbound.
    def _unbound_at(
        self,
        scope: ast.AST,
        *,
        root: str,
        group: Sequence[tuple[ast.Import, ast.alias]],
    ) -> list[tuple[ast.Import, ast.alias]]:
        """Return the rewritten imports that need an ``import root`` before them."""
        rewritten = {alias for _, alias in group}
        binders = [
            node
            for node in self.unit.nodes(ast.Import)
            if self.scopes.owner[node] is scope
            and any(
                alias.asname is None
                and alias.name.partition(".")[0] == root
                and alias not in rewritten
                for alias in node.names
            )
        ]
        needing: list[tuple[ast.Import, ast.alias]] = []
        for node, alias in group:
            if not any(
                _dominates(binder, later=node, parents=self.unit.parents)
                for binder in binders
            ):
                needing.append((node, alias))
                binders.append(node)
        return needing

    def _reason(
        self,
        node: ast.Import,
        *,
        parts: tuple[str, ...],
        shared: bool,
    ) -> tuple[str, bool]:
        """Why ``import <parts>`` must stay, and whether an alias is the remedy."""
        scope = self.scopes.owner[node]
        root, leaf, name = parts[0], parts[-1], ".".join(parts)
        reads = [
            nodes
            for chain, nodes in self.chains.get((scope, root), [])
            if _longest_import(chain, dotted=self.paths[scope, root]) == parts
        ]
        if _ignored(self.unit, node=node, rule="dotted-import"):
            return "it carries a house-ignore[dotted-import] marker", False
        if isinstance(scope, ast.ClassDef):
            return "it is in a class body", False
        if root in self.scopes.declared:
            return f"`{root}` is declared global or nonlocal", False
        if root in self.scopes.other[scope]:
            return f"`{root}` is also bound another way in this scope", False
        if leaf in _BUILTIN_NAMES:
            return f"`{leaf}` is a builtin", True
        # Binding ``leaf`` here must capture no read of another ``leaf``, and
        # every rewritten read must see this binding, not one nearer to it.
        # A loop variable of that name in this scope or one inside it is
        # ruff's F402 (import shadowed by loop variable).
        if (
            leaf in self.scopes.bound[scope]
            or leaf in self.scopes.declared
            or any(
                isinstance(target, ast.Name)
                and target.id == leaf
                and self.scopes.within(target, scope=scope)
                for loop in self.unit.nodes(ast.For, ast.AsyncFor)
                for target in ast.walk(loop.target)
            )
            or any(
                other.id == leaf
                and isinstance(other.ctx, ast.Load)
                and self.scopes.resolve(leaf, other, binding=scope) is scope
                for other in self.unit.nodes(ast.Name)
            )
            or any(self._shadowed(leaf, node=nodes[0], scope=scope) for nodes in reads)
        ):
            return f"`{leaf}` already names something else where it would be read", True
        if shared:
            return f"another import also ends in `{leaf}`", True
        if not reads:
            return (
                f"nothing reads through `{name}`; delete it, or name the side "
                "effect it is for in a house-ignore reason"
            ), False
        if any(
            len(nodes) == len(parts) and isinstance(nodes[-1].ctx, ast.Store | ast.Del)
            for nodes in reads
        ):
            return f"`{name}` itself is assigned or deleted", False
        if self._patched(parts):
            return (
                f"`{name}` is swapped on `{'.'.join(parts[:-1])}` at runtime "
                f"(setattr or patch), which a bound `{leaf}` would not see"
            ), False
        if not self._sole_binder(node, root=root) and not all(
            self._runs_after(node, read=nodes[0]) for nodes in reads
        ):
            return f"a read of `{name}` can run where this import has not", False
        return "", False

    # One import binding ``a`` makes every path that skips it fail before and
    # after alike. With a second, ``a.b.x`` can run after only the other ran
    # (``a.b`` loaded by ``a``'s own ``__init__``), where a bare ``b`` would not.
    def _sole_binder(self, node: ast.Import, *, root: str) -> bool:
        """Whether ``node`` is the one import that binds ``root`` in its scope."""
        scope = self.scopes.owner[node]
        return not any(
            other is not node
            and self.scopes.owner[other] is scope
            and any(
                alias.asname is None and alias.name.partition(".")[0] == root
                for alias in other.names
            )
            for other in self.unit.nodes(ast.Import)
        )

    # A read later in the import's block, or under a later statement of it, runs after
    # it. A read in a def or lambda body runs when called, after the scope's body ran --
    # if the import is unconditional in that body. An annotation under ``from __future__
    # import annotations`` never runs.
    def _runs_after(self, node: ast.Import, *, read: ast.AST) -> bool:
        """Whether ``read`` can only run once ``node`` has bound its name."""
        parents = self.unit.parents
        if self.lazy_annotations and _in_annotation(read, parents=parents):
            return True
        scope = self.scopes.owner[node]
        current = self.scopes.owner[read]
        while current is not scope:
            if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                return parents.get(node) is scope
            current = self.scopes.enclosing[current]
        return _dominates(node, later=read, parents=parents)

    # Only a submodule swap matters: ``patch("a.b.X")`` patches an attribute of
    # the module object a bound ``b`` still is.
    def _patched(self, parts: tuple[str, ...]) -> bool:
        """Whether a call swaps ``a.b`` on ``a``: ``setattr(a, "b")``, ``patch("a.b")``."""
        name, parent = ".".join(parts), parts[:-1]
        for call in self.unit.nodes(ast.Call):
            match call.args:
                case [ast.Constant(value=str() as target), *_] if target == name:
                    return True
                case [
                    ast.Attribute() | ast.Name() as owner,
                    ast.Constant(value=str() as attr),
                    *_,
                ] if attr == parts[-1]:
                    dotted = _dotted_parts(owner)
                    if dotted is not None and tuple(dotted[0]) == parent:
                        return True
                case _:
                    pass
        return False

    # Lexical, not semantic: in ``def f(pool: multiprocessing.pool.Pool)`` the
    # annotation is read in the module, so ``pool: pool.Pool`` would run, but a
    # reader takes ``pool.Pool`` for an attribute of the parameter beside it.
    def _shadowed(self, name: str, *, node: ast.AST, scope: ast.AST) -> bool:
        """Whether a def, lambda, or class around ``node`` within ``scope`` binds it."""
        parents = self.unit.parents
        current = parents.get(node)
        while current is not None and current is not scope:
            if name in self.scopes.bound.get(current, ()):
                return True
            current = parents.get(current)
        return False

    def _stranded(self, scope: ast.AST, *, root: str, covered: set[ast.AST]) -> bool:
        """Whether ``scope``'s ``root`` is read other than through a rewrite."""
        return any(
            node.id == root
            and isinstance(node.ctx, ast.Load)
            and node not in covered
            and self.scopes.resolve(root, node) is scope
            for node in self.unit.nodes(ast.Name)
        )


def _dominates(
    first: ast.stmt,
    *,
    later: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
) -> bool:
    """Whether ``first`` always runs before ``later``: earlier in a block holding it."""
    child = later
    while (parent := parents.get(child)) is not None:
        if parent is parents.get(first):
            # ``body``/``orelse``/``finalbody``: the same parent, but only the
            # same list runs both, in order.
            for _, value in cast(
                "Iterator[tuple[str, object]]",
                ast.iter_fields(parent),
            ):
                block = cast("list[object]", value) if isinstance(value, list) else []
                if first in block and child in block:
                    return block.index(first) < block.index(child)
            return False
        child = parent
    return False


def _in_annotation(node: ast.AST, *, parents: Mapping[ast.AST, ast.AST]) -> bool:
    """Whether ``node`` sits in a parameter, return, or variable annotation."""
    child, parent = node, parents.get(node)
    while parent is not None:
        match parent:
            case ast.arg(annotation=annotation) if annotation is child:
                return True
            case (
                ast.FunctionDef(returns=returns)
                | ast.AsyncFunctionDef(
                    returns=returns,
                )
            ) if returns is child:
                return True
            case ast.AnnAssign(annotation=annotation) if annotation is child:
                return True
            case ast.stmt():
                return False
            case _:
                child, parent = parent, parents.get(parent)
    return False


# Read from comment tokens' text, as the dispatcher reads markers: one trailing
# the statement covers it, one on a line of its own covers the file.
def _ignored(unit: Unit, *, node: ast.stmt, rule: str) -> bool:
    """Whether a ``house-ignore`` marker naming ``rule`` covers ``node``."""
    for token in unit.tokens:
        if token.type != tokenize.COMMENT:
            continue
        match = re.search(r"#\s*house-ignore\[([^\]]*)\]", token.string)
        if match is None or rule not in {
            name.strip() for name in match.group(1).split(",")
        }:
            continue
        line, column = token.start
        own_line = not unit.lines[line - 1][:column].strip()
        if own_line or node.lineno <= line <= (node.end_lineno or node.lineno):
            return True
    return False


def _trailing_comment(
    source: str,
    *,
    starts: _LineStarts,
    node: ast.stmt,
) -> tuple[int, int, str] | None:
    """Span and text of the comment after ``node``, when ``node`` starts its line."""
    start = _offset(starts, node.end_lineno, node.end_col_offset)
    end = source.find("\n", start)
    if end < 0:
        end = len(source)
    comment = source[start:end].strip()
    prefix = source[
        _offset(starts, node.lineno, 0) : _offset(starts, node.lineno, node.col_offset)
    ]
    if prefix.strip() or not comment.startswith("#"):
        return None
    return start, end, comment


@dataclass(frozen=True, kw_only=True, slots=True)
class _Scopes:
    """Which scope evaluates each node, and the names each scope binds.

    A scope is the module, a def, a lambda, a class body, or a comprehension:
    a comprehension's targets are its own, and folding them into the scope
    around it would send that scope's reads of a same-named root nowhere.

    Attributes:
      owner: The scope each node is evaluated in.
      enclosing: Each nested scope's parent scope.
      bound: Each scope's bound names, however bound.
      other: Each scope's names bound by anything but an unaliased
        ``import a`` or ``import a.b``.
      declared: Every name a ``global`` or ``nonlocal`` statement names.

    """

    owner: dict[ast.AST, ast.AST]

    enclosing: dict[ast.AST, ast.AST]

    bound: dict[ast.AST, set[str]]

    other: dict[ast.AST, set[str]]

    declared: set[str]

    @classmethod
    def build(cls, tree: ast.Module) -> Self:
        """Walk ``tree`` once, iteratively: a deep expression overflows recursion.

        Args:
          tree: The parsed module.

        Returns:
          scopes: Every node's scope and every scope's bindings.

        """
        scopes = cls(
            owner={},
            enclosing={},
            bound={tree: set()},
            other={tree: set()},
            declared=set(),
        )
        stack: list[tuple[ast.AST, ast.AST]] = [(child, tree) for child in tree.body]
        while stack:
            node, scope = stack.pop()
            scopes.owner[node] = scope
            if isinstance(
                node,
                ast.FunctionDef
                | ast.AsyncFunctionDef
                | ast.Lambda
                | ast.ClassDef
                | ast.ListComp
                | ast.SetComp
                | ast.DictComp
                | ast.GeneratorExp,
            ):
                stack.extend(scopes._enter(node, scope))
                continue
            for name, plain in _bindings(node):
                scopes.bound[scope].add(name)
                if not plain:
                    scopes.other[scope].add(name)
            if isinstance(node, ast.NamedExpr):
                # ``(x := ...)`` in a comprehension binds in the scope around it.
                home = scope
                while isinstance(
                    home,
                    ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
                ):
                    home = scopes.enclosing[home]
                scopes.bound[home].add(node.target.id)
                scopes.other[home].add(node.target.id)
            if isinstance(node, ast.Global | ast.Nonlocal):
                scopes.declared.update(node.names)
            stack.extend((child, scope) for child in ast.iter_child_nodes(node))
        return scopes

    def resolve(
        self,
        name: str,
        node: ast.AST,
        *,
        binding: ast.AST | None = None,
    ) -> ast.AST:
        """Return the scope whose binding of ``name`` a read at ``node`` sees.

        A class body's names are invisible to the defs and lambdas inside it,
        so only the read's own scope may be a class.

        Args:
          name: The name read.
          node: Where it is read.
          binding: A scope to treat as binding ``name`` too: what a read
            would see once a rewrite binds it there.

        Returns:
          scope: The binding scope; the module when no inner scope binds it.

        """
        scope = self.owner[node]
        own = True
        while scope in self.enclosing:
            if (name in self.bound[scope] or scope is binding) and (
                own or not isinstance(scope, ast.ClassDef)
            ):
                return scope
            own = False
            scope = self.enclosing[scope]
        return scope

    def within(self, node: ast.AST, *, scope: ast.AST) -> bool:
        """Whether ``node`` is evaluated in ``scope`` or a scope nested in it.

        Args:
          node: Any node of the module.
          scope: The module, a def, a lambda, a class, or a comprehension.

        Returns:
          inside: Whether ``scope`` lies on ``node``'s chain of scopes.

        """
        current = self.owner[node]
        while current is not scope:
            if current not in self.enclosing:
                return False
            current = self.enclosing[current]
        return True

    def _enter(
        self,
        node: ast.FunctionDef
        | ast.AsyncFunctionDef
        | ast.Lambda
        | ast.ClassDef
        | ast.ListComp
        | ast.SetComp
        | ast.DictComp
        | ast.GeneratorExp,
        scope: ast.AST,
    ) -> list[tuple[ast.AST, ast.AST]]:
        """Register the scope ``node`` opens; return its children and their scopes."""
        self.enclosing[node] = scope
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            self.bound[scope].add(node.name)
            self.other[scope].add(node.name)
        # Decorators, defaults, annotations, and bases run in the scope around
        # the def or class; only the parameters and the body are inside it. A
        # comprehension's first iterable runs outside it, and the rest inside.
        if isinstance(
            node,
            ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        ):
            first, *rest = node.generators
            params: list[ast.arg] = []
            outside: list[ast.AST] = [first.iter]
            inside: list[ast.AST] = [
                *(
                    (node.key, node.value)
                    if isinstance(node, ast.DictComp)
                    else (node.elt,)
                ),
                first.target,
                *first.ifs,
                *rest,
            ]
        elif isinstance(node, ast.ClassDef):
            params = []
            outside = [
                *node.decorator_list,
                *node.bases,
                *node.keywords,
                *node.type_params,
            ]
            inside = list(node.body)
        else:
            args = node.args
            params = [
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                *filter(None, (args.vararg, args.kwarg)),
            ]
            if isinstance(node, ast.Lambda):
                outside = [*args.defaults, *filter(None, args.kw_defaults)]
                inside = [node.body]
            else:
                outside = [
                    *node.decorator_list,
                    *args.defaults,
                    *filter(None, args.kw_defaults),
                    *filter(None, (arg.annotation for arg in params)),
                    *filter(None, (node.returns,)),
                    *node.type_params,
                ]
                inside = list(node.body)
        self.bound[node] = {arg.arg for arg in params}
        self.other[node] = {arg.arg for arg in params}
        return [(child, scope) for child in outside] + [
            (child, node) for child in inside
        ]


def _bindings(node: ast.AST) -> Iterator[tuple[str, bool]]:
    """Yield each name ``node`` binds, and whether a plain ``import a[.b]`` binds it."""
    match node:
        case ast.Name(id=name, ctx=ast.Store() | ast.Del()):
            yield name, False
        case ast.Import(names=names):
            for alias in names:
                yield alias.asname or alias.name.partition(".")[0], alias.asname is None
        case ast.ImportFrom(names=names):
            for alias in names:
                yield alias.asname or alias.name, False
        case (
            ast.ExceptHandler(name=str() as name)
            | ast.MatchAs(name=str() as name)
            | ast.MatchStar(name=str() as name)
            | ast.MatchMapping(rest=str() as name)
        ):
            yield name, False
        case _:
            pass


def _is_autograd_function_class(node: ast.AST | None) -> bool:
    """Whether ``node`` subclasses PyTorch's custom autograd Function."""
    return isinstance(node, ast.ClassDef) and any(
        ast.unparse(base) in {"Function", "torch.autograd.Function"}
        for base in node.bases
    )


def _nested_imports(unit: Unit) -> tuple[ast.Import | ast.ImportFrom, ...]:
    """Return imports not directly owned by the module body."""
    parents = unit.parents
    return tuple(
        node
        for node in unit.nodes(ast.Import, ast.ImportFrom)
        if not isinstance(parents.get(node), ast.Module)
        and not _is_main_guard_import(node, parents)
        and not _is_type_checking_import(node, parents)
        and not _is_optional_import(node, parents)
    )


def _is_module_scope(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
) -> bool:
    """Whether ``node`` is not nested inside a callable or class."""
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.Module):
            return True
        if isinstance(
            parent,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            return False
        parent = parents.get(parent)
    return False


def _is_optional_import(
    node: ast.Import | ast.ImportFrom,
    parents: Mapping[ast.AST, ast.AST],
) -> bool:
    """Whether an import is guarded by a module-scope ``ImportError`` handler."""
    parent = parents.get(node)
    if not isinstance(parent, ast.Try) or not _is_module_scope(parent, parents):
        return False
    return any(
        handler.type is None
        or (isinstance(handler.type, ast.Name) and handler.type.id == "ImportError")
        or (
            isinstance(handler.type, ast.Tuple)
            and any(
                isinstance(item, ast.Name) and item.id == "ImportError"
                for item in handler.type.elts
            )
        )
        for handler in parent.handlers
    )


def _is_main_guard_import(
    node: ast.Import | ast.ImportFrom,
    parents: Mapping[ast.AST, ast.AST],
) -> bool:
    """Whether an import belongs directly to the standard main guard."""
    parent = parents.get(node)
    return isinstance(parent, ast.If) and _is_main_guard(parent)


def _is_type_checking_import(
    node: ast.Import | ast.ImportFrom,
    parents: Mapping[ast.AST, ast.AST],
) -> bool:
    """Whether an import belongs directly to an ``if TYPE_CHECKING`` block."""
    parent = parents.get(node)
    return isinstance(parent, ast.If) and _is_type_checking_test(parent.test)


def _house_ignore(
    unit: Unit,
    node: ast.Import | ast.ImportFrom,
    rule: str,
) -> bool:
    """Whether a node carries a houselint suppression for ``rule``."""
    end_lineno = node.end_lineno or node.lineno
    lines = unit.lines[node.lineno - 1 : end_lineno]
    marker = f"house-ignore[{rule}]"
    return any(marker in line for line in lines)


def _noqa_plc0415(unit: Unit, node: ast.Import | ast.ImportFrom) -> bool:
    """Whether an import statement carries a bare or PLC0415 noqa."""
    lines = unit.lines[node.lineno - 1 : node.end_lineno]
    return any(
        "# noqa" in line.lower() and (":" not in line or "plc0415" in line.lower())
        for line in lines
    )


def _module_import_end(tree: ast.Module) -> int:
    """Return the line index after the module's initial import block."""
    head = list(
        itertools.takewhile(
            lambda node: isinstance(node, (ast.Import, ast.ImportFrom)),
            _without_docstring(tree.body),
        ),
    )
    if not head and tree.body and _is_docstring(tree.body[0]):
        head = [tree.body[0]]
    return max((node.end_lineno or node.lineno for node in head), default=0)


# ``check`` and ``fix`` share this so a fix never rewrites a call the check
# called clean: the fix gate in ``houselint._lint_path`` relies on it.
def _unprefixed_tempfile_calls(unit: Unit, tree: ast.Module) -> Iterator[ast.Call]:
    """Yield each tempfile call, dotted or directly imported, without ``prefix=``."""
    imported = frozenset(
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "tempfile"
        for alias in node.names
        if alias.name in {name.removeprefix("tempfile.") for name in _TEMPFILE_CALLS}
    )
    for node in unit.nodes(ast.Call):
        text = _call_text(node)
        if text not in _TEMPFILE_CALLS and text not in imported:
            continue
        if not any(kw.arg == "prefix" for kw in node.keywords):
            yield node


def _listed(value: object) -> list[object]:
    """Return ``value`` when it is a JSON array, else ``[]``."""
    try:
        return from_plain(value, list[object], default=[])
    except ReadError:
        return []


def _polyglot_header_rows(lines: Sequence[str]) -> frozenset[int]:
    """Return the 1-based rows of a polyglot header's ``fmt: off`` and ``fmt: on``."""
    start = next(
        (
            i
            for i, line in enumerate(lines[:-1])
            if line.startswith(_POLYGLOT_HEADER[0])
            and lines[i + 1] == _POLYGLOT_HEADER[1]
        ),
        None,
    )
    if start is None:
        return frozenset[int]()
    close = next(
        (j for j in range(start + 2, len(lines)) if lines[j].startswith("'''")),
        len(lines),
    )
    after = close + 1
    paired = after < len(lines) and _FMT_DIRECTIVE.search(lines[after]) is not None
    return frozenset({start + 1, after + 1} if paired else {start + 1})
