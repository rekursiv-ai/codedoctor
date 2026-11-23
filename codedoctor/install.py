"""Configure a repository's Codedoctor gates and development tools."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import fnmatch
import itertools
import json
import re
import shutil
import subprocess
import sys
import textwrap

from treekle import MutablePlainTree, from_plain
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

import tomlkit
import yaml

from codedoctor import codegraph, policy
from codedoctor.lib.yaml import position


_THIS: Final[Path] = Path(__file__).resolve()
_CWD: Final = _THIS.parent


def install(root: Path, *, notebooks: bool = False) -> tuple[str, ...]:
    """Configure and activate Codedoctor in a Git repository.

    Args:
      root: Repository root; explicit project overrides are preserved.
      notebooks: Install notebook dependencies and output-stripping/sync hooks.

    Returns:
      overrides: Native settings differing from the canonical policy.

    Raises:
      ValueError: Existing configuration cannot be merged safely.
      subprocess.CalledProcessError: Dependency resolution or hook installation failed.

    """
    project = root / "pyproject.toml"
    hooks = root / ".pre-commit-config.yaml"
    configured_project = project_config(
        project.read_text() if project.exists() else "",
        name=root.name,
    )
    configured_hooks = hook_config(
        hooks.read_text() if hooks.exists() else "",
        notebooks=notebooks,
    )
    project.write_text(configured_project)
    subprocess.run(  # noqa: S603 -- Fixed uv command and package names.
        [
            shutil.which("uv") or "uv",
            "add",
            "--group",
            "pre-commit",
            "--no-sync",
            "codedoctor[coverage,mutation,notebook]"
            if notebooks
            else "codedoctor[coverage,mutation]",
            "pre-commit",
            "ruff",
            "codespell",
            "ty",
            "pyrefly",
            "basedpyright",
            "pytest",
            "pytest-cov",
            "pytest-timeout",
            "pytest-testmon",
            "pytest-xdist",
            "wrapt",
        ],
        cwd=root,
        check=True,
    )
    hooks.write_text(configured_hooks)
    subprocess.run(  # noqa: S603 -- Fixed uv command and package names.
        [
            shutil.which("uv") or "uv",
            "run",
            "--locked",
            "--group",
            "pre-commit",
            "pre-commit",
            "install",
            "--hook-type",
            "pre-commit",
        ],
        cwd=root,
        check=True,
    )
    codegraph.install()
    _bootstrap_hooks(root)
    _update_gitignore(root)
    _write_biome(root)
    return policy.overrides(tomlkit.parse(configured_project))


def current(root: Path) -> bool:
    """Check generated policy freshness without changing files or dependencies."""
    path = root / "pyproject.toml"
    text = path.read_text() if path.exists() else ""
    return bool(text) and project_config(text, name=root.name) == text


def project_config(text: str, *, name: str) -> str:
    """Return TOML with missing setup defaults, preserving existing policy.

    Args:
      text: Existing pyproject content, or empty for a new project.
      name: Directory name used only when creating a project.

    Returns:
      configured: Comment-preserving project configuration.

    """
    document = tomlkit.parse(text)
    dependencies = policy.table(document, key="dependency-groups")
    dependencies.setdefault("pre-commit", [])
    tool = policy.table(document, key="tool")
    uv = policy.table(tool, key="uv")
    if "project" not in document:
        project = policy.table(document, key="project")
        project["name"] = "-".join(re.findall(r"[a-z0-9]+", name.lower())) or "project"
        project["version"] = "0.1.0"
        project["requires-python"] = ">=3.12"
        project["dependencies"] = []
        uv.setdefault("package", False)
    groups = policy.table(uv, key="dependency-groups")
    group = policy.table(groups, key="pre-commit")
    group.setdefault("requires-python", ">=3.12")
    project_fields = from_plain(
        policy.table(document, key="project").unwrap(),
        dict[str, MutablePlainTree],
    )
    project_name = from_plain(project_fields.get("name", name), str)
    floor = (
        re.search(
            r"(?:>=|==|~=)\s*(3\.\d+)",
            from_plain(project_fields["requires-python"], str),
        )
        if "requires-python" in project_fields
        else None
    )
    policy.apply(
        document,
        defaults=policy.defaults(
            name=project_name,
            python_version=floor[1] if floor else "3.12",
        ),
    )
    return tomlkit.dumps(document)


def hook_config(text: str, *, notebooks: bool = False) -> str:
    """Return YAML with missing Codedoctor hooks, preserving existing entries.

    Args:
      text: Existing pre-commit configuration, or empty for a new project.
      notebooks: Include notebook output stripping and paired-file synchronization.

    Returns:
      configured: Hook configuration, unchanged when already installed.

    Raises:
      ValueError: Configuration is malformed or a hook ID conflicts.

    """
    expected = [
        {
            **repo,
            "hooks": [
                hook
                for hook in from_plain(repo["hooks"], list[dict[str, object]])
                if notebooks or hook["id"] not in {"nbstripout", "jupytext-sync"}
            ],
        }
        for repo in _hook_repos()
    ]
    text = _replace_owned_hooks(
        _upgrade_hooks(text, expected=expected),
        expected=expected,
    )
    document = from_plain(
        _document(text),
        dict[str, object],
        default=dict[str, object](),
    )
    repos = (
        from_plain(document["repos"], list[dict[str, object]])
        if "repos" in document
        else []
    )
    wanted = [
        _hook_identity(repo, hook=hook)
        for repo in expected
        for hook in from_plain(repo["hooks"], list[dict[str, object]])
    ]
    existing = [
        _hook_identity(repo, hook=hook)
        for repo in repos
        for hook in from_plain(repo.get("hooks", []), list[dict[str, object]])
    ]
    for present in existing:
        if (
            any(present[1]["id"] == hook["id"] for _, hook in wanted)
            and present not in wanted
        ):
            raise ValueError(
                f"Hook {present[1]['id']!r} conflicts with Codedoctor's definition.",
            )
    missing: list[dict[str, object]] = []
    for repo in expected:
        hooks = [
            hook
            for hook in from_plain(repo["hooks"], list[dict[str, object]])
            if _hook_identity(repo, hook=hook) not in existing
        ]
        if hooks:
            missing.append({**repo, "hooks": hooks})
    if missing:
        text = _append_repos(text, repos=missing)
    return _order_owned_repos(text, expected=expected)


def _upgrade_hooks(text: str, *, expected: list[dict[str, object]]) -> str:
    commands: dict[str, str] = {}
    for repo in expected:
        for hook in from_plain(repo["hooks"], list[dict[str, object]]):
            if "entry" not in hook:
                continue
            entry = from_plain(hook["entry"], str)
            match = re.search(r" -m codedoctor ([\w-]+)", entry)
            if match:
                command = match[1]
                module = {
                    "codegraph-sync": "affected.codegraph_sync",
                    "check-pr-text": "export_leaks.check_pr_text",
                }.get(command, command.replace("-", "_"))
                commands[module] = command
    changes: list[tuple[int, int, str]] = []
    pending = [_compose(text)]
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, SequenceNode):
            pending.extend(node.value)
        elif isinstance(node, MappingNode):
            fields = {
                key.value: value
                for key, value in node.value
                if isinstance(key, ScalarNode)
            }
            identifier = fields.get("id")
            entry = fields.get("entry")
            if isinstance(identifier, ScalarNode) and isinstance(entry, ScalarNode):
                match = re.match(r"(.*? -m )codedoctor\.([\w.]+)(?=\s|$)", entry.value)
                if match and match[2] in commands:
                    values = {
                        "entry": match[1]
                        + "codedoctor "
                        + commands[match[2]]
                        + entry.value[match.end() :],
                    }
                    if identifier.value == "houselint":
                        values["id"] = "codedoctor-lint"
                        name = fields.get("name")
                        if isinstance(name, ScalarNode) and name.value == "houselint":
                            values["name"] = "codedoctor-lint"
                    for key, value in values.items():
                        scalar = fields[key]
                        assert isinstance(scalar, ScalarNode)
                        # JSON strings are YAML scalars without line wrapping.
                        replacement = json.dumps(value)
                        if scalar.style in {"|", ">"}:
                            replacement += "\n"
                        changes.append(
                            (
                                position(scalar).index,
                                position(scalar, end=True).index,
                                replacement,
                            ),
                        )
            pending.extend(fields.values())
    for start, end, replacement in sorted(changes, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


# Codedoctor owns what its hooks run, `args` included, so a hook an older release wrote
# is rewritten to this release's definition. Refusing left every client of 0.1.5 unable
# to install 0.1.6; keeping a project's `args` kept `biome-check` without the flags the
# monorepo runs it with (2026-11-18/19). Which paths a hook skips is the project's:
# `exclude` stays as the project wrote it. Hooks pair by repository and id, in order,
# since one id may run twice (ruff-format before and after the fix).
def _replace_owned_hooks(text: str, *, expected: list[dict[str, object]]) -> str:
    """Rewrite in place each Codedoctor hook whose definition differs, but `exclude`."""
    wanted: dict[
        tuple[object, object],
        list[tuple[dict[str, object], dict[str, object]]],
    ]
    wanted = {}
    for repo in expected:
        for hook in from_plain(repo["hooks"], list[dict[str, object]]):
            wanted.setdefault((repo["repo"], hook["id"]), []).append((repo, hook))
    seen: dict[tuple[object, object], int] = {}
    changes: dict[int, tuple[int, str]] = {}
    for repo_node in _block_repos(text):
        repo_fields = _scalar_fields(repo_node)
        hooks = {
            key.value: value
            for key, value in repo_node.value
            if isinstance(key, ScalarNode)
        }.get("hooks")
        if "repo" not in repo_fields or not isinstance(hooks, SequenceNode):
            continue
        for hook_node in hooks.value:
            if not isinstance(hook_node, MappingNode) or hook_node.flow_style:
                continue
            present = from_plain(
                _document(_node_text(text, hook_node)),
                dict[str, object],
            )
            key = (repo_fields["repo"].value, present.get("id"))
            index = seen.get(key, 0)
            seen[key] = index + 1
            if index >= len(wanted.get(key, [])):
                continue
            owner, hook = wanted[key][index]
            for field, value in owner.items():
                scalar = repo_fields.get(field)
                if (
                    isinstance(value, str)
                    and scalar is not None
                    and scalar.value != value
                ):
                    changes[position(scalar).index] = (
                        position(scalar, end=True).index,
                        json.dumps(value),
                    )
            replacement = {k: v for k, v in hook.items() if k != "exclude"}
            if "exclude" in present:
                replacement["exclude"] = present["exclude"]
            if replacement != present:
                changes[position(hook_node).index] = _hook_text(
                    text,
                    node=hook_node,
                    hook=replacement,
                )
    for start, (end, value) in sorted(changes.items(), reverse=True):
        text = text[:start] + value + text[end:]
    return text


def _order_owned_repos(text: str, *, expected: list[dict[str, object]]) -> str:
    """Move the repositories holding only Codedoctor hooks into the packaged order."""
    positions: dict[tuple[object, object], list[int]] = {}
    for rank, key in enumerate(
        (repo["repo"], hook["id"])
        for repo in expected
        for hook in from_plain(repo["hooks"], list[dict[str, object]])
    ):
        positions.setdefault(key, []).append(rank)
    # An id can appear twice (ruff-format); each copy gets its own rank.
    ranks = {key: iter(value) for key, value in positions.items()}
    nodes = _block_repos(text)
    owned: list[tuple[int, int]] = []
    for index, node in enumerate(nodes):
        repo = _scalar_fields(node).get("repo")
        hooks = {
            key.value: value for key, value in node.value if isinstance(key, ScalarNode)
        }.get("hooks")
        node_ranks: list[int | None] = []
        for hook in hooks.value if isinstance(hooks, SequenceNode) else []:
            assert isinstance(hook, MappingNode)
            key = (repo and repo.value, _scalar_fields(hook)["id"].value)
            node_ranks.append(next(ranks.get(key, iter(())), None))
        first = node_ranks[0] if node_ranks else None
        if first is not None and None not in node_ranks:
            owned.append((first, index))
    slots = [index for _, index in owned]
    moved = [index for _, index in sorted(owned)]
    if moved == slots:
        return text
    first = position(nodes[0])
    bounds = [
        _content_end(text, index=first.index - first.column),
        *(_content_end(text, index=position(node, end=True).index) for node in nodes),
    ]
    chunks = [text[start:end] for start, end in itertools.pairwise(bounds)]
    if not chunks[-1].endswith("\n"):
        chunks[-1] += "\n"
    order = iter(moved)
    return (
        text[: bounds[0]]
        + "".join(
            chunks[next(order)] if index in slots else chunk
            for index, chunk in enumerate(chunks)
        )
        + text[bounds[-1] :]
    )


def _content_end(text: str, *, index: int) -> int:
    """Return the index past the last line before ``index`` that is not blank or comment."""
    # A node's end mark lies past any comments after it.
    lines = text[:index].splitlines(keepends=True)
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("#")):
        lines.pop()
    return sum(map(len, lines))


def _block_repos(text: str) -> list[MappingNode]:
    """Return the block-style repository mappings under the top-level `repos`."""
    tree = _compose(text)
    if not isinstance(tree, MappingNode):
        return []
    repos = {
        key.value: value for key, value in tree.value if isinstance(key, ScalarNode)
    }.get("repos")
    if not isinstance(repos, SequenceNode) or repos.flow_style:
        return []
    return [
        node
        for node in repos.value
        if isinstance(node, MappingNode) and not node.flow_style
    ]


def _safe_loader() -> type[yaml.SafeLoader | yaml.CSafeLoader]:
    """Return libyaml's safe loader where PyYAML has it, else the pure-Python one."""
    # libyaml's marks count characters, as the pure loader's do, so the edits that
    # slice the text at them hold under either. The pure loader cost each
    # `hook_config` a quarter second under the gate's coverage tracer.
    return yaml.CSafeLoader if yaml.__with_libyaml__ else yaml.SafeLoader


def _compose(text: str) -> yaml.Node | None:
    """Return the YAML document ``text`` as nodes, each marked where it is written."""
    return yaml.compose(text, Loader=_safe_loader())


def _document(text: str) -> object:
    """Return the YAML document ``text`` as plain data, as ``yaml.safe_load`` does."""
    loader = _safe_loader()(text)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _scalar_fields(node: MappingNode) -> dict[str, ScalarNode]:
    """Return a mapping's scalar-valued fields by key."""
    return {
        key.value: value
        for key, value in node.value
        if isinstance(key, ScalarNode) and isinstance(value, ScalarNode)
    }


def _node_text(text: str, node: MappingNode) -> str:
    """Return a block mapping's source, its first line dedented to match the rest."""
    start, end = position(node), position(node.value[-1][1], end=True)
    return " " * start.column + text[start.index : end.index]


def _hook_text(
    text: str,
    *,
    node: MappingNode,
    hook: dict[str, object],
) -> tuple[int, str]:
    """Return the end index and block YAML replacing ``node`` with ``hook``."""
    # A block list's end mark sits past its newline and the next line's indent;
    # end the span at its last character so what follows keeps its line.
    end = len(text[: position(node.value[-1][1], end=True).index].rstrip())
    # Unbounded width: a long `entry` stays on its line.
    dumped = yaml.safe_dump(
        hook,
        sort_keys=False,  # pragma: no mutate[truthiness] -- PyYAML reads only its truth.
        width=sys.maxsize,
    )
    # The first line takes the hook's own place; each after it, the hook's indent.
    return end, dumped.removesuffix("\n").replace(
        "\n",
        "\n" + " " * position(node).column,
    )


def _hook_identity(
    repo: dict[str, object],
    *,
    hook: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Match hook definitions field for field, but the project's ``exclude``."""
    return (
        {key: value for key, value in repo.items() if key != "hooks"},
        {key: value for key, value in hook.items() if key != "exclude"},
    )


def _hook_repos() -> list[dict[str, object]]:
    """Read the hooks every installed repository runs: the monorepo's, as exported."""
    packaged = _THIS.with_name("install_hooks.yaml")
    text = packaged.read_text() if packaged.is_file() else ""
    config = from_plain(_document(text), dict[str, object])
    return from_plain(config["repos"], list[dict[str, object]])


def _append_repos(text: str, *, repos: list[dict[str, object]]) -> str:
    """Insert repositories without reserializing existing configuration."""
    tree = _compose(text)
    if tree is None:
        return (
            text
            + "\n"
            + yaml.safe_dump(
                {
                    "fail_fast": False,
                    "default_stages": ["pre-commit"],
                    "default_install_hook_types": ["pre-commit"],
                    "repos": repos,
                },
            )
        )
    assert isinstance(tree, MappingNode)
    for key, value in tree.value:
        if isinstance(key, ScalarNode) and key.value == "repos":
            assert isinstance(value, SequenceNode)
            return _append_repo(text, node=value, repos=repos)
    if tree.flow_style:
        fragment = yaml.safe_dump(
            {"repos": repos},
            default_flow_style=True,
        ).strip()[1:-1]
        at = (
            position(tree.value[-1][1], end=True).index
            if tree.value
            else position(tree).index + 1
        )
        return text[:at] + (", " if tree.value else "") + fragment + text[at:]
    return text.rstrip() + "\nrepos:\n" + textwrap.indent(yaml.safe_dump(repos), "  ")


def _append_repo(
    text: str,
    *,
    node: SequenceNode,
    repos: list[dict[str, object]],
) -> str:
    """Insert repositories without reserializing the user's YAML."""
    if node.flow_style:
        at = (
            position(node.value[-1], end=True).index
            if node.value
            else position(node).index + 1
        )
        fragment = yaml.safe_dump(
            repos,
            default_flow_style=True,
        ).strip()[1:-1]
        return text[:at] + (", " if node.value else "") + fragment + text[at:]
    at = position(node, end=True).index
    fragment = textwrap.indent(
        yaml.safe_dump(repos),
        " " * position(node).column,
    )
    return text[:at].rstrip() + "\n" + fragment + text[at:]


def _bootstrap_hooks(root: Path) -> None:
    """Keep generated Git hooks usable after an exact uv sync."""
    git = subprocess.run(  # noqa: S603 -- Fixed Git command, no user input.
        [shutil.which("git") or "git", "rev-parse", "--git-common-dir"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    path = root / git.stdout.strip() / "hooks" / "pre-commit"
    content = path.read_text()
    # Retain pre-commit's identity so reinstall/uninstall preserves legacy hooks.
    header = content[: content.index("# start templated\n")]
    path.write_text(
        header
        + "exec uv run --frozen --group pre-commit pre-commit hook-impl "
        + "--config=.pre-commit-config.yaml --hook-type=pre-commit "
        + '--hook-dir="$(dirname "$0")" -- "$@"\n',
    )


def _write_biome(root: Path) -> None:
    """Give the biome-check hook Codedoctor's settings unless the project has its own."""
    if not any((root / name).exists() for name in ("biome.json", "biome.jsonc")):
        shutil.copyfile(_THIS.with_name("install_biome.json"), root / "biome.json")


def _update_gitignore(root: Path) -> None:
    """Append rules for generated state the existing rules do not already ignore."""
    path = root / ".gitignore"
    text = path.read_text() if path.exists() else ""
    lines = text.splitlines()
    missing = [
        rule
        for rule in (
            ".testmondata*",
            ".codegraph/",
            ".coverage",
            ".coverage.*",
            ".coverage-*",
        )
        if not _covered(lines, rule=rule)
    ]
    if missing:
        with path.open("ab") as stream:
            if text:
                stream.write(b"\n" if text.endswith("\n") else b"\n\n")
            stream.write(("\n".join(["# Codedoctor", *missing]) + "\n").encode())


# Gitignore semantics for root-level paths: the last matching line wins, `!` re-includes,
# a trailing `/` matches only directories, and an inner `/` names a deeper path. Every
# undecided case reads as uncovered, so the cost of imprecision is a redundant line.
def _covered(lines: list[str], *, rule: str) -> bool:
    """Whether `lines` ignore every root-level path `rule` ignores."""
    directory = rule.endswith("/")
    rule = rule.removesuffix("/")
    stem = rule.removesuffix("*")
    covered = False
    for line in lines:
        pattern = line.rstrip(" ")
        if not pattern or pattern.startswith("#"):
            continue
        negated = pattern.startswith("!")
        pattern = pattern.removeprefix("!")
        if pattern.endswith("/") and not directory:
            continue
        pattern = pattern.removesuffix("/").removeprefix("/").removeprefix("**/")
        if negated:
            literal = re.split(r"[*?\[\\]", pattern)[0]
            if (
                "\\" in pattern
                or (stem == rule and fnmatch.fnmatchcase(rule, pattern))
                or (
                    stem != rule
                    and (stem.startswith(literal) or literal.startswith(stem))
                )
            ):
                covered = False
        elif "/" not in pattern and _matches_every(pattern, rule=rule):
            covered = True
    return covered


def _matches_every(pattern: str, *, rule: str) -> bool:
    """Whether glob `pattern` matches every name `rule` (literal or `stem*`) matches."""
    stem = rule.removesuffix("*")
    if stem == rule:
        return fnmatch.fnmatchcase(rule, pattern)
    # A trailing `*` that matches `stem` stretches over any suffix after it.
    return pattern.endswith("*") and fnmatch.fnmatchcase(stem, pattern)
