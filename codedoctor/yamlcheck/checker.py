"""Validate YAML without executing constructors or accepting unknown tags."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import argparse
import sys

from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.nodes import MappingNode, Node, SequenceNode


def main() -> int:
    """Validate the supplied YAML files and return the process exit code."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    return check(flags.files)


class Flags(Protocol):
    """YAML paths supplied by pre-commit."""

    files: list[Path]


def check(paths: list[Path]) -> int:
    """Validate YAML files, allowing only the two Compose merge tags.

    Args:
        paths: YAML files to check without modifying their contents.

    Returns:
        exit_code: One if any file fails validation, otherwise zero.

    """
    result = 0
    for path in paths:
        try:
            loader = YAML(typ="safe", pure=True)
            node = loader.compose(path.read_text(encoding="utf-8"))
            if node is not None:
                _normalize_compose_tags(node)
                loader.constructor.construct_document(node)
        except (OSError, ValueError, YAMLError) as error:
            sys.stderr.write(f"{path}: {error}\n")
            result = 1
    return result


def _normalize_compose_tags(root: Node) -> None:
    """Retag Compose values before the unchanged safe constructor validates them."""
    pending = [root]
    visited: set[Node] = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        kind = "str"
        if isinstance(node, MappingNode):
            kind = "map"
            pending.extend(child for pair in node.value for child in pair)
        elif isinstance(node, SequenceNode):
            kind = "seq"
            pending.extend(node.value)
        if node.tag in {"!reset", "!override"}:
            node.tag = f"tag:yaml.org,2002:{kind}"


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("files", type=Path, nargs="*")
