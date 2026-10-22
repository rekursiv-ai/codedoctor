"""Shared values for the houselint dispatcher and rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import ast
import io
import tokenize


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


_CODE = frozenset({"python", "pyi"})


@dataclass(frozen=True, kw_only=True, slots=True, order=True)
class Finding:
    """One violation, at one line, from one rule.

    ``order=True`` sorts by path then line with no key function; the field
    order below IS the sort order, so ``rule`` and ``message`` only break
    ties between two findings on the same line.

    Attributes:
      path: File the violation is in.
      line: 1-based line number.
      rule: Name of the rule that reported it. One process runs several
        rules into one log, so every line has to say which one fired.
      message: Human-readable explanation.

    """

    path: Path
    line: int
    rule: str
    message: str

    @override
    def __str__(self) -> str:
        """Render as ``path:line: [rule] message``."""
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


@dataclass(frozen=True, kw_only=True, slots=True)
class Unit:
    """One file's text and the views derived from it, built on first read."""

    path: Path
    source: str
    kind: str

    _tree: list[ast.Module | None] = field(
        default_factory=list,
        init=False,
        repr=False,
        compare=False,
    )
    _tokens: list[tuple[tokenize.TokenInfo, ...]] = field(
        default_factory=list,
        init=False,
        repr=False,
        compare=False,
    )
    _lines: list[tuple[str, ...]] = field(
        default_factory=list,
        init=False,
        repr=False,
        compare=False,
    )
    _index: list[_Index] = field(
        default_factory=list,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def tree(self) -> ast.Module | None:
        """Parsed module or stub; None for a notebook, and when it does not parse."""
        if not self._tree:
            self._tree.append(
                _parse(self.source, self.path) if self.kind in _CODE else None,
            )
        return self._tree[0]

    @property
    def tokens(self) -> tuple[tokenize.TokenInfo, ...]:
        """Token stream; empty for a notebook and on a tokenization error."""
        if not self._tokens:
            self._tokens.append(
                tuple(_tokenize(self.source)) if self.kind in _CODE else (),
            )
        return self._tokens[0]

    @property
    def lines(self) -> tuple[str, ...]:
        """Source lines, for scanning suppression comments."""
        if not self._lines:
            self._lines.append(tuple(self.source.splitlines()))
        return self._lines[0]

    @property
    def walk(self) -> tuple[ast.AST, ...]:
        """Every node of ``tree`` in ``ast.walk`` order; empty when unparsed."""
        return self._indexed().walk

    @property
    def parents(self) -> Mapping[ast.AST, ast.AST]:
        """Each node's parent; the module itself has none."""
        return self._indexed().parents

    def nodes[T: ast.AST](self, *kinds: type[T]) -> tuple[T, ...]:
        """Every node that is an instance of any of ``kinds``, in ``walk`` order.

        Args:
          *kinds: Node classes to select; subclasses match, as with
            ``isinstance``. Several kinds keep their interleaved walk order.

        Returns:
          nodes: The matching nodes, cached per ``kinds`` for this unit.

        """
        index = self._indexed()
        found = index.by_kind.get(kinds)
        if found is None:
            # Union the exact-type buckets of every concrete type matching
            # ``kinds``, then restore walk order -- far cheaper than an
            # ``isinstance`` pass over every node for each new query.
            positions = [
                position
                for concrete, members in index.by_type.items()
                if issubclass(concrete, kinds)
                for position in members
            ]
            positions.sort()
            found = tuple(index.walk[position] for position in positions)
            index.by_kind[kinds] = found
        return cast("tuple[T, ...]", found)

    def _indexed(self) -> _Index:
        """Build the one walk every rule shares, on first use."""
        if not self._index:
            self._index.append(_Index.build(self.tree))
        return self._index[0]


# One pass over the tree yields the node order, the parent map, and each node's
# walk position grouped by exact type; ``by_kind`` memoizes the queries rules make.
@dataclass(frozen=True, kw_only=True, slots=True)
class _Index:
    """The shared walk of one parsed module."""

    walk: tuple[ast.AST, ...]
    parents: dict[ast.AST, ast.AST]
    by_type: dict[type[ast.AST], list[int]]
    by_kind: dict[tuple[type[ast.AST], ...], tuple[ast.AST, ...]]

    @classmethod
    def build(cls, tree: ast.Module | None) -> _Index:
        """Walk ``tree`` once, breadth-first as ``ast.walk`` does.

        Args:
          tree: Parsed module, or None when the source did not parse.

        Returns:
          index: Node order and parent map; empty when ``tree`` is None.

        """
        if tree is None:
            return cls(walk=(), parents={}, by_type={}, by_kind={})
        order: list[ast.AST] = [tree]
        parents: dict[ast.AST, ast.AST] = {}
        by_type: dict[type[ast.AST], list[int]] = {}
        for position, node in enumerate(order):
            by_type.setdefault(type(node), []).append(position)
            for child in ast.iter_child_nodes(node):
                parents[child] = node
                order.append(child)
        return cls(walk=tuple(order), parents=parents, by_type=by_type, by_kind={})


# The error has to be caught around the ITERATION, not around building a tuple:
# ``generate_tokens`` is lazy, so a ``try`` wrapped only over the call returns a
# generator that raises later, at the consumer.
def _tokenize(source: str) -> Iterator[tokenize.TokenInfo]:
    """Yield ``source``'s tokens, stopping quietly at a tokenization error."""
    try:
        yield from tokenize.generate_tokens(io.StringIO(source).readline)
    except (tokenize.TokenError, SyntaxError):
        return


def _parse(source: str, path: Path) -> ast.Module | None:
    """Parse ``source``, or None when it does not parse."""
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None
