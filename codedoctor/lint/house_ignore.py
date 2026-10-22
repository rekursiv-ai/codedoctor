"""The ``house-ignore`` marker every house check honors, houselint's and the hooks'.

One marker for every rule. Its POSITION decides its scope, as ruff's ``noqa`` vs
``ruff: noqa`` does: a marker trailing code covers that statement; one on a line
of its own covers the whole file. The rule list is required and so is the
``-- Reason.`` tail; a marker missing either is itself a finding. Checks never
parse markers -- :func:`apply` filters their findings against them.

    x = 1  # house-ignore[globals, digits] -- Reason.   (this statement)
    # house-ignore[module-side-effects] -- Reason.       (whole file)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ast
import re
import tokenize

from codedoctor.lint.houselint_types import Finding


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from codedoctor.lint.houselint_types import Unit


IGNORE = re.compile(r"#\s*house-ignore\[([^\]]*)\](.*)$")


def apply(
    unit: Unit,
    found: Iterable[Finding],
    /,
    *,
    active: frozenset[str],
) -> list[Finding]:
    """Drop the findings ``unit``'s markers suppress; report markers that rotted.

    Args:
      unit: The file the findings are in.
      found: Every finding the ``active`` rules reported in ``unit``.
      active: Rules that ran on ``unit``; only a marker naming one of them can
        be judged superfluous.

    Returns:
      findings: The unsuppressed findings, then one per marker naming an
        ``active`` rule that suppressed nothing.

    """
    marks = tuple(markers(unit))
    file_ignores = [
        (lineno, rule_names(match.group(1))) for lineno, match, own in marks if own
    ]
    inline_ignores = {
        lineno: rule_names(match.group(1)) for lineno, match, own in marks if not own
    }
    silenced = {name for _, names in file_ignores for name in names}
    covered: set[str] = set()
    used_inline: set[tuple[int, str]] = set()
    kept: list[Finding] = []
    for finding in found:
        if finding.rule in silenced:
            covered.add(finding.rule)
        elif marker_lines := suppressing_lines(unit, finding, inline_ignores):
            used_inline.update((line, finding.rule) for line in marker_lines)
        else:
            kept.append(finding)
    # A marker naming a rule that reports nothing there has rotted -- the code
    # was fixed and the marker outlived it. Reported as ``houselint`` so the
    # marker cannot silence the complaint about itself.
    kept.extend(
        Finding(
            path=unit.path,
            line=line,
            rule="houselint",
            message=(
                f"superfluous file-level ignore: {name!r} reports nothing here; "
                "remove it from the own-line house-ignore[...]"
            ),
        )
        for line, names in file_ignores
        for name in sorted(names)
        if name in active and name not in covered
    )
    kept.extend(
        Finding(
            path=unit.path,
            line=line,
            rule="houselint",
            message=(
                f"superfluous inline ignore: {name!r} reports nothing on this "
                "statement; remove it from house-ignore[...]"
            ),
        )
        for line, names in sorted(inline_ignores.items())
        for name in sorted(names)
        if name in active and (line, name) not in used_inline
    )
    return kept


# Markers are read from COMMENT tokens, never from raw lines: a test fixture or a
# rule's own example quotes the marker inside a string literal, and a line scan
# cannot tell that prose from a live marker. A comment token at column 0 of its
# logical content is own-line; one after code is trailing.
def markers(unit: Unit) -> Iterator[tuple[int, re.Match[str], bool]]:
    """Yield each house marker in ``unit``.

    Args:
      unit: The file to scan.

    Yields:
      marker: ``(line, match, own_line)``: the 1-based line, the ``IGNORE``
        match, and whether the comment stands on a line of its own.

    """
    for tok in unit.tokens:
        if tok.type != tokenize.COMMENT:
            continue
        match = IGNORE.search(tok.string)
        if match is None:
            continue
        lineno, col = tok.start
        own_line = not unit.lines[lineno - 1][:col].strip()
        yield lineno, match, own_line


def rule_names(listing: str) -> frozenset[str]:
    """Split the bracketed rule list of a marker."""
    return frozenset(n.strip() for n in listing.split(",") if n.strip())


# One grammar, one failure mode: a bare marker with no rules suppresses nothing
# and reads as if it did, and a marker with no reason leaves the next reader to
# rediscover why. Both are findings, not warnings. Only houselint reports them,
# so a marker is complained about once however many tools read it.
def marker_findings(unit: Unit) -> list[Finding]:
    """Report each ``house-ignore`` marker missing its rule list or ``-- Reason.``.

    Args:
      unit: The file to scan.

    Returns:
      findings: One per malformed marker, as rule ``houselint``.

    """
    findings: list[Finding] = []
    for lineno, match, _ in markers(unit):
        if not rule_names(match.group(1)):
            findings.append(
                Finding(
                    path=unit.path,
                    line=lineno,
                    rule="houselint",
                    message="house-ignore[...] names no rule; a bare marker suppresses nothing",
                ),
            )
        if re.match(r"^\s*--\s+\S.*\.\s*$", match.group(2)) is None:
            findings.append(
                Finding(
                    path=unit.path,
                    line=lineno,
                    rule="houselint",
                    message="house-ignore[...] needs a trailing `-- Reason.`",
                ),
            )
    return findings


# The marker may sit on any line of the statement the finding is in, not only the
# reported line: the formatter moves a trailing comment to the closing paren when it
# wraps a call, and a marker that stopped working after ``ruff format`` is worse than
# none.
# A finding on line 1 is a whole-file verdict (the polyglot header, a test footer), so
# for those the marker may be on any line of the file.
# Every marker in range is returned, not the first: a wrapped ``if`` with several
# ``.any()`` calls carries one marker per call, and each covers the same header.
def suppressing_lines(
    unit: Unit,
    finding: Finding,
    ignores: Mapping[int, frozenset[str]],
) -> tuple[int, ...]:
    """Lines of the trailing markers covering ``finding``; empty if unsuppressed."""
    covering = ignores if finding.line == 1 else statement_lines(unit, finding.line)
    return tuple(line for line in covering if finding.rule in ignores.get(line, ()))


# A simple statement is its own header. A compound statement's header is its
# keyword through the colon -- a wrapped ``if (... or ...):`` spans several lines,
# and a marker on any of them (the formatter puts it after the ``):``) covers a
# finding on any other. The body is excluded: a marker on the ``if`` line must
# not silence findings inside the block.
def statement_lines(unit: Unit, line: int) -> range:
    """Return the innermost statement header containing ``line``.

    Args:
      unit: The parsed file.
      line: 1-based line of a finding.

    Returns:
      header: Its 1-based lines; just ``line`` when no statement holds it.

    """
    best: range | None = None
    for node in unit.nodes(ast.stmt):
        # Cheap span test first; a header never extends past its node's end.
        if line > (node.end_lineno or node.lineno):
            continue
        header = header_lines(node)
        if header.start <= line < header.stop and (
            best is None or header.start >= best.start
        ):
            best = header
    return best if best is not None else range(line, line + 1)


# A compound header runs from the keyword to the line before its first body
# statement -- that is where a wrapped header's closing ``):`` and any trailing
# comment live. ``ast`` records no position for the colon itself.
def header_lines(node: ast.stmt) -> range:
    """Return ``node``'s header lines.

    Args:
      node: A statement.

    Returns:
      header: 1-based lines from its first decorator to its colon; the whole
        node for a simple statement.

    """
    end = node.end_lineno or node.lineno
    if isinstance(
        node,
        ast.If
        | ast.While
        | ast.For
        | ast.AsyncFor
        | ast.With
        | ast.AsyncWith
        | ast.FunctionDef
        | ast.AsyncFunctionDef
        | ast.ClassDef
        | ast.Try
        | ast.TryStar,
    ):
        end = _body_start(node.body) - 1
    elif isinstance(node, ast.Match):
        end = node.cases[0].pattern.lineno - 1 if node.cases else node.lineno
    # A decorator is part of the header it decorates: the formatter wraps a
    # long ``@dataclass(...)`` over several lines and the marker lands on the
    # closing paren, while the finding points at the ``@``.
    start = node.lineno
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        start = min((d.lineno for d in node.decorator_list), default=start)
    return range(start, max(end, start) + 1)


def _body_start(body: Sequence[ast.stmt]) -> int:
    """First line of a block, counting a leading decorator as part of it."""
    first = body[0]
    if isinstance(first, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return min((d.lineno for d in first.decorator_list), default=first.lineno)
    return first.lineno
