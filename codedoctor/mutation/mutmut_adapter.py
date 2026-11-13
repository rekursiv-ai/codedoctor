"""Filter selective mutation policies before native mutmut assigns candidate IDs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from io import StringIO
from pathlib import Path
from string import Formatter
from typing import TYPE_CHECKING, Final, TypeGuard, cast, override

import ast
import hashlib
import inspect
import os
import re
import sys
import tokenize

from libcst.metadata import PositionProvider
from mutmut import __main__ as mutmut_main
from mutmut.__main__ import FileMutationResult
from mutmut.mutation import file_mutation
from mutmut.mutation.data import SourceFileMutationData
from mutmut.mutation.file_mutation import MutationVisitor
from mutmut.mutation.mutators import mutation_operators
from mutmut.mutation.pragma_handling import (
    PragmaParseError,
    PragmaVisitor,
    get_ignored_lines,
)
from mutmut.mutation.trampoline_templates import mangle_function_name
from mutmut.utils.file_utils import setup_source_paths

import libcst as cst


_THIS: Final[Path] = Path(__file__).resolve()
_CWD: Final[Path] = _THIS.parent

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from mutmut.mutation.file_mutation import MutatedFile, Mutation
    from mutmut.mutation.pragma_handling import IgnoredCode


_NATIVE_HASHES = cast(
    Callable[[str, Callable[[str], bool] | None], dict[str, str]],
    inspect.unwrap(file_mutation.compute_function_hashes),
)
_NATIVE_CREATE_FILE = cast(
    Callable[[Path, Path], FileMutationResult],
    inspect.unwrap(mutmut_main.create_mutants_for_file),
)
_NATIVE_SETUP_PATHS = setup_source_paths
_LABELS = frozenset(
    {
        "diagnostic-text",
        "case-insensitive",
        "truthiness",
    },
)


def activate() -> None:
    """Install the source-local policy and cache adapters in native mutmut."""
    wraps(_NATIVE_HASHES, assigned=(), updated=())(_function_hashes)
    wraps(_NATIVE_CREATE_FILE, assigned=(), updated=())(_create_mutants_for_file)
    mutmut_main.setup_source_paths = _setup_source_paths  # ty: ignore[invalid-assignment] -- Native module-function replacement is this adapter's extension seam.  # pyright: ignore[reportPrivateImportUsage] -- Native CLI's imported alias is the required patch site.
    file_mutation.create_mutations = create_mutations  # ty: ignore[invalid-assignment] -- Native module-function replacement is this adapter's extension seam.
    file_mutation.compute_function_hashes = _function_hashes  # ty: ignore[invalid-assignment] -- Native module-function replacement is this adapter's extension seam.
    file_mutation._compute_mutated_function_hashes = _cache_hashes  # ty: ignore[invalid-assignment] -- Native module-function replacement is this adapter's extension seam.  # noqa: SLF001 -- Native cache hook must retain hashes for empty owners.
    mutmut_main.create_mutants_for_file = _create_mutants_for_file  # ty: ignore[invalid-assignment] -- Native module-function replacement is this adapter's extension seam.


def mutate_file_contents(
    filename: str,
    code: str,
    covered_lines: set[int] | None = None,
    coverage_excluded_lines: set[int] | None = None,
) -> MutatedFile:
    """Activate selective policy and render native candidates for one source.

    Args:
      filename: Source path used in diagnostics.
      code: Original Python source, including comments.
      covered_lines: Native coverage selection, or every line.
      coverage_excluded_lines: Native coverage exclusions.

    Returns:
      generated: Native source, candidate IDs, and cache hashes.

    """
    activate()
    return file_mutation.mutate_file_contents(
        filename,
        code,
        covered_lines,
        coverage_excluded_lines,
    )


def create_mutations(
    filename: str,
    code: str,
    covered_lines: set[int] | None = None,
    coverage_excluded_lines: set[int] | None = None,
) -> tuple[cst.Module, list[Mutation], set[str], set[str]]:
    """Generate native candidates, filtering selective matches before rendering.

    Args:
      filename: Source path used in diagnostics.
      code: Original Python source, including comments.
      covered_lines: Native coverage selection, or every line.
      coverage_excluded_lines: Native coverage exclusions.

    Returns:
      module: Original native source tree.
      mutations: Retained original/replacement pairs, without candidate IDs.
      ignored_classes: Native blanket-excluded classes.
      ignored_functions: Native blanket-excluded functions.

    Raises:
      ValueError: A selective declaration is invalid or demonstrably unused.

    """
    policy = _Policy.parse(filename, source=code)
    wrapper = cst.MetadataWrapper(cst.parse_module(code))
    blanket_wrapper = cst.MetadataWrapper(cst.parse_module(policy.blanket_source))
    ignored = get_ignored_lines(
        filename,
        code,
        blanket_wrapper,
        coverage_excluded_lines,
    )
    visitor = _SelectiveVisitor(
        policy=policy,
        module=wrapper.module,
        ignored=ignored,
        covered_lines=covered_lines,
    )
    module = wrapper.visit(visitor)
    policy.check_usage(examined=visitor.examined, matched=visitor.matched)
    return module, visitor.mutations, visitor.ignored_classes, visitor.ignored_functions


@dataclass(frozen=True, slots=True, kw_only=True)
class _Directive:
    line: int
    column: int
    end_column: int
    labels: frozenset[str]
    mode: str
    standalone: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class _Scope:
    line: int
    labels: frozenset[str]
    lines: frozenset[int]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Policy:
    filename: str
    blanket_source: str
    scopes: tuple[_Scope, ...]

    @classmethod
    def parse(cls, filename: str, *, source: str) -> _Policy:
        directives = _directives(filename, source=source)
        normalized = _pragma_source(
            source,
            directives=directives,
            active=set(range(len(directives))),
        )
        _ignored_scope(normalized, filename=filename)
        ends = {
            index: next(
                end
                for end in range(index + 1, len(directives))
                if directives[end].standalone and directives[end].mode == "end"
            )
            for index, directive in enumerate(directives)
            if directive.standalone and directive.mode == "start"
        }
        selective = {
            index for index, directive in enumerate(directives) if directive.labels
        }
        selective_ends = {ends[index] for index in selective if index in ends}
        blanket = set(range(len(directives))) - selective - selective_ends
        scopes = tuple(
            _Scope(
                line=directives[index].line,
                labels=directives[index].labels,
                lines=_ignored_scope(
                    _pragma_source(
                        source,
                        directives=directives,
                        active={index, ends[index]} if index in ends else {index},
                    ),
                ),
            )
            for index in sorted(selective)
        )
        return cls(
            filename=filename,
            scopes=scopes,
            blanket_source=_pragma_source(
                source,
                directives=directives,
                active=blanket,
            ),
        )

    def check_usage(
        self,
        *,
        examined: set[int],
        matched: dict[int, set[str]],
    ) -> None:
        for index, scope in enumerate(self.scopes):
            if not scope.lines.intersection(examined):
                continue
            unused = scope.labels - matched.get(index, set())
            if unused:
                raise ValueError(
                    f"{self.filename}:{scope.line}: unused mutation labels: {', '.join(sorted(unused))}",
                )


def _directives(filename: str, *, source: str) -> tuple[_Directive, ...]:
    directives: list[_Directive] = []
    for token in tokenize.generate_tokens(StringIO(source, newline=None).readline):
        if token.type != tokenize.COMMENT:
            continue
        markers = list(re.finditer(r"# pragma:\s*no mutate", token.string))
        for index, marker in enumerate(markers):
            end = (
                markers[index + 1].start()
                if index + 1 < len(markers)
                else len(token.string)
            )
            tail = token.string[marker.end() : end]
            syntax, separator, _ = tail.partition("--")
            syntax = syntax.rstrip()
            labels = frozenset[str]()
            mode = syntax.strip().split(",", 1)[0].split(" ", 1)[0]
            standalone = not token.line[: token.start[1]].strip()
            if "[" in syntax or "]" in syntax:
                match = re.fullmatch(r"\[([^\[\]]*)\](?:\s+(block|start))?", syntax)
                if match is None:
                    raise ValueError(
                        f"{filename}:{token.start[0]}: invalid selective mutation pragma",
                    )
                labels = frozenset(label.strip() for label in match[1].split(","))
                if not labels or labels - _LABELS:
                    raise ValueError(
                        f"{filename}:{token.start[0]}: unknown or empty mutation label",
                    )
                mode = match[2] or "bare"
                if mode == "start" and not standalone:
                    raise ValueError(
                        f"{filename}:{token.start[0]}: start must be standalone",
                    )
            if separator:
                end = len(token.string)
            directives.append(
                _Directive(
                    line=token.start[0],
                    column=token.start[1] + marker.start(),
                    end_column=token.start[1] + end,
                    labels=labels,
                    mode=mode if mode in {"block", "start", "end"} else "bare",
                    standalone=standalone,
                ),
            )
            if separator:
                break
    return tuple(directives)


def _pragma_source(
    source: str,
    *,
    directives: tuple[_Directive, ...],
    active: set[int],
) -> str:
    lines = StringIO(source, newline="").readlines()
    for index in reversed(range(len(directives))):
        directive = directives[index]
        replacement = (
            f"# pragma: no mutate {directive.mode}" if index in active else "#"
        )
        line = lines[directive.line - 1]
        lines[directive.line - 1] = (
            line[: directive.column] + replacement + " " + line[directive.end_column :]
        )
    return "".join(lines)


def _ignored_scope(
    source: str,
    *,
    filename: str = "",  # pragma: no mutate[diagnostic-text]
) -> frozenset[int]:
    visitor = PragmaVisitor(filename)
    try:
        cst.MetadataWrapper(cst.parse_module(source)).visit(visitor)
    except PragmaParseError as error:
        raise ValueError(str(error)) from error
    return frozenset(visitor.no_mutate_lines | visitor.ignore_node_lines)


class _SelectiveVisitor(MutationVisitor):
    def __init__(
        self,
        *,
        policy: _Policy,
        module: cst.Module,
        ignored: IgnoredCode,
        covered_lines: set[int] | None,
    ) -> None:
        super().__init__(mutation_operators, ignored, covered_lines)
        self.policy = policy
        self.module = module
        self.tree = ast.parse(module.code)
        self.tokens = _tokens(module.code)
        self.literal_counts = Counter(
            token.end[0] for token in self.tokens if token.type == tokenize.STRING
        )
        self.examined: set[int] = set()
        self.matched: dict[int, set[str]] = {}

    @override
    def _create_mutations(self, node: cst.CSTNode) -> None:
        position = self.get_metadata(PositionProvider, node, None)
        if position is not None and isinstance(
            self.get_metadata(file_mutation.OuterFunctionProvider, node, None),
            cst.FunctionDef,
        ):
            self.examined.add(position.start.line)
        start = len(self.mutations)
        super()._create_mutations(node)
        candidates = self.mutations[start:]
        del self.mutations[start:]
        for candidate in candidates:
            if not self._matches(candidate):
                self.mutations.append(candidate)

    def _matches(self, mutation: Mutation) -> bool:
        position = self.get_metadata(PositionProvider, mutation.original_node, None)
        if position is None:
            return False
        if not any(
            scope.lines.intersection(range(position.start.line, position.end.line + 1))
            for scope in self.policy.scopes
        ):
            return False
        changed = file_mutation.deep_replace(
            self.module,
            mutation.original_node,
            mutation.mutated_node,
        )
        assert isinstance(changed, cst.Module)
        difference = _string_change(
            self.tokens,
            mutant=changed.code,
            literal_counts=self.literal_counts,
        )
        matches = _structural_matches(self.tree, mutant=ast.parse(changed.code))
        if difference is not None:
            line, original, mutant = difference
            for label in ("diagnostic-text", "case-insensitive"):
                if _matches_label(original, mutant=mutant, label=label):
                    matches[label] = {line}
        matched = False  # pragma: no mutate[truthiness]
        for index, scope in enumerate(self.policy.scopes):
            labels = {
                label
                for label in scope.labels
                if scope.lines.intersection(matches.get(label, set()))
            }
            if labels:
                self.matched.setdefault(index, set()).update(labels)
                matched = True
        return matched


def _structural_matches(
    original: ast.AST,
    *,
    mutant: ast.AST,
) -> dict[str, set[int]]:
    if _ast_key(original) == _ast_key(mutant):
        return {}
    if _falsy_literal(original) and _falsy_literal(mutant):
        return {"truthiness": {original.lineno}}
    if type(original) is not type(mutant):
        return {}
    updated = dict(cast("Iterator[tuple[str, object]]", ast.iter_fields(mutant)))
    differences = [
        (value, updated[name])
        for name, value in cast(
            "Iterator[tuple[str, object]]",
            ast.iter_fields(original),
        )
        if _ast_key(value) != _ast_key(updated[name])
    ]
    if len(differences) != 1:
        return {}
    old, new = differences[0]
    if isinstance(old, list):
        assert isinstance(new, list)
        old_items, new_items = cast(list[object], old), cast(list[object], new)
        if len(old_items) != len(new_items):
            return {}
        differences = [
            (left, new_items[index])
            for index, left in enumerate(old_items)
            if _ast_key(left) != _ast_key(new_items[index])
        ]
        if len(differences) != 1:
            return {}
        old, new = differences[0]
    if not isinstance(old, ast.AST) or new is None:
        return {}
    assert isinstance(new, ast.AST)
    matches = _structural_matches(old, mutant=new)
    value = old.value if isinstance(old, ast.keyword) else old
    if isinstance(original, ast.Call) and matches and _falsy_literal(value):
        matches["truthiness"].add(original.lineno)
    return matches


def _ast_key(value: object) -> str:
    if isinstance(value, ast.AST):
        return ast.dump(value)
    if isinstance(value, list):
        return repr([_ast_key(item) for item in cast(list[object], value)])
    return repr(value)


def _falsy_literal(node: ast.AST) -> TypeGuard[ast.expr]:
    if isinstance(node, ast.Constant):
        return not bool(node.value)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in {int, float, complex}
        and node.operand.value == 0
    )


def _tokens(source: str) -> list[tokenize.TokenInfo]:
    return [
        token
        for token in tokenize.generate_tokens(StringIO(source, newline=None).readline)
        if token.type != tokenize.COMMENT
    ]


def _string_change(
    original: list[tokenize.TokenInfo],
    *,
    mutant: str,
    literal_counts: Counter[int],
) -> tuple[int, str, str] | None:
    updated = _tokens(mutant)
    if len(original) != len(updated):
        return None
    changes = [
        (old, new)
        for old, new in zip(original, updated, strict=True)
        if (old.type, old.string) != (new.type, new.string)
    ]
    if len(changes) != 1:
        return None
    old, new = changes[0]
    if (
        old.type != tokenize.STRING
        or new.type != tokenize.STRING
        or literal_counts[old.end[0]] != 1
    ):
        return None
    old_value = cast(object, ast.literal_eval(old.string))
    new_value = cast(object, ast.literal_eval(new.string))
    if not isinstance(old_value, str) or not isinstance(new_value, str):
        return None
    return old.end[0], old_value, new_value


def _matches_label(original: str, *, mutant: str, label: str) -> bool:
    if label == "case-insensitive":
        return original != mutant and original.lower() == mutant.lower()
    fields = _diagnostic_fields(original)
    return fields is not None and fields == _diagnostic_fields(mutant)


def _diagnostic_fields(
    value: str,
) -> tuple[list[tuple[str, str | None, str | None]], list[str]] | None:
    parts = cast(
        list[str],
        re.split(
            r"(%(?:\([^()]+\))?[#0 +\-]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?[a-zA-Z%])",
            value,
        ),
    )
    if any("%" in part for part in parts[::2]):
        return None
    try:
        fields = [
            (field, spec, conversion)
            for _, field, spec, conversion in Formatter().parse(value)
            if field is not None
        ]
    except ValueError:
        return None
    return fields, parts[1::2]


def _implementation_hash() -> str:
    return hashlib.sha256(_THIS.read_bytes()).hexdigest()


def _function_hashes(
    source_code: str,
    accept: Callable[[str], bool] | None = None,
) -> dict[str, str]:
    hashes = _NATIVE_HASHES(source_code, accept)
    if not hashes:
        return hashes
    policy = _Policy.parse("<source>", source=source_code)
    blanket = _ignored_scope(policy.blanket_source)
    identity = _implementation_hash()
    _hash_functions(
        ast.parse(source_code).body,
        hashes=hashes,
        policy=policy,
        blanket=blanket,
        identity=identity,
    )
    return hashes


def _hash_functions(
    body: Sequence[ast.stmt],
    *,
    hashes: dict[str, str],
    policy: _Policy,
    blanket: frozenset[int],
    identity: str,
    class_name: str = "",
) -> None:
    for node in body:
        if isinstance(node, ast.ClassDef):
            _hash_functions(
                node.body,
                hashes=hashes,
                policy=policy,
                blanket=blanket,
                identity=identity,
                class_name=f"{class_name}.{node.name}" if class_name else node.name,
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = mangle_function_name(name=node.name, class_name=class_name or None)
            if key in hashes:
                effective = [
                    (
                        line - node.lineno,
                        line in blanket,
                        sorted(
                            {
                                label
                                for scope in policy.scopes
                                if line in scope.lines
                                for label in scope.labels
                            },
                        ),
                    )
                    for line in range(node.lineno, (node.end_lineno or node.lineno) + 1)
                ]
                hashes[key] = hashlib.sha256(
                    repr((hashes[key], identity, effective)).encode(),
                ).hexdigest()[:12]


def _cache_hashes(
    source_code: str,
    module: cst.Module,
    mutations: Sequence[Mutation],
) -> dict[str, str]:
    # Zero retained candidates must still invalidate when a policy later exposes them.
    del module, mutations
    return _function_hashes(source_code)


def _setup_source_paths() -> None:
    """Let tests resolve generated packages without replacing active native hooks."""
    _NATIVE_SETUP_PATHS()
    # Native path setup cannot replace packages imported by the adapter bootstrap.
    name = __name__
    while name:
        sys.modules.pop(name, None)
        name = name.rpartition(".")[0]


def _create_mutants_for_file(filename: Path, output_path: Path) -> FileMutationResult:
    data = SourceFileMutationData(path=filename)
    data.load()
    hashes = _function_hashes(filename.read_text())
    # Native mtime reuse precedes hash checks; force generation before stale results escape.
    if hashes != data.hash_by_function_name:
        with output_path.open("a"):
            pass
        os.utime(output_path, (filename.stat().st_atime, filename.stat().st_mtime))
    return _NATIVE_CREATE_FILE(filename, output_path)
