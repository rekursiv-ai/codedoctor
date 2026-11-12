"""Configure a repository's Codedoctor gates and development tools."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import fnmatch
import json
import re
import shutil
import subprocess
import textwrap

from treekle import MutablePlainTree, from_plain
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

import tomlkit
import yaml

from codedoctor import codegraph, policy


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
    text = _upgrade_hooks(text, expected=expected)
    document = from_plain(
        yaml.safe_load(text),
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
    if not missing:
        return text
    return _append_repos(text, repos=missing)


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
    loader = yaml.SafeLoader(text)
    try:
        tree = loader.get_single_node()
    finally:
        loader.dispose()
    changes: list[tuple[int, int, str]] = []
    pending = [tree]
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
                            values["name"] = "codedoctor lint"
                    for key, value in values.items():
                        scalar = fields[key]
                        assert isinstance(scalar, ScalarNode)
                        # JSON strings are YAML scalars without line wrapping.
                        replacement = json.dumps(value)
                        if scalar.style in {"|", ">"}:
                            replacement += "\n"
                        changes.append(
                            (
                                scalar.start_mark.index,
                                scalar.end_mark.index,
                                replacement,
                            ),
                        )
            pending.extend(fields.values())
    for start, end, replacement in sorted(changes, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _hook_identity(
    repo: dict[str, object],
    *,
    hook: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Match hook definitions without taking ownership of project arguments."""
    return (
        {key: value for key, value in repo.items() if key != "hooks"},
        {key: value for key, value in hook.items() if key != "args"},
    )


def _hook_repos() -> list[dict[str, object]]:
    """Read the same hook configuration this package uses itself."""
    template = _THIS.with_name("install_hooks.yaml")
    content = template.read_text()
    config = from_plain(yaml.safe_load(content), dict[str, object])
    repos = from_plain(config["repos"], list[dict[str, object]])
    portable: list[dict[str, object]] = []
    for repo in repos:
        hooks = [
            hook
            for hook in from_plain(repo["hooks"], list[dict[str, object]])
            if hook["id"]
            not in {
                "cargo-fmt",
                "cargo-clippy",
                "check-copybarista-pr-text",
                "check-export-leaks-spool",
                "check-export-leaks",
                "priml-golden-owners",
                "arcagi3-gold-isolation",
            }
        ]
        if hooks:
            portable.append({**repo, "hooks": hooks})
    return portable


def _append_repos(text: str, *, repos: list[dict[str, object]]) -> str:
    """Insert repositories without reserializing existing configuration."""
    loader = yaml.SafeLoader(text)
    try:
        tree = loader.get_single_node()
    finally:
        loader.dispose()
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
        position = (
            tree.value[-1][1].end_mark.index
            if tree.value
            else tree.start_mark.index + 1
        )
        return (
            text[:position] + (", " if tree.value else "") + fragment + text[position:]
        )
    return text.rstrip() + "\nrepos:\n" + textwrap.indent(yaml.safe_dump(repos), "  ")


def _append_repo(
    text: str,
    *,
    node: SequenceNode,
    repos: list[dict[str, object]],
) -> str:
    """Insert repositories without reserializing the user's YAML."""
    if node.flow_style:
        position = (
            node.value[-1].end_mark.index if node.value else node.start_mark.index + 1
        )
        fragment = yaml.safe_dump(
            repos,
            default_flow_style=True,
        ).strip()[1:-1]
        return (
            text[:position] + (", " if node.value else "") + fragment + text[position:]
        )
    position = node.end_mark.index
    fragment = textwrap.indent(
        yaml.safe_dump(repos),
        " " * node.start_mark.column,
    )
    return text[:position].rstrip() + "\n" + fragment + text[position:]


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
