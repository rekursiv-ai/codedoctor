"""Configure a repository's Codedoctor gates and development tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict, cast

import re
import shutil
import subprocess
import textwrap
import tomllib

from tomlkit.items import InlineTable, Table
from treekle import from_plain
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

import tomlkit
import yaml


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent

if TYPE_CHECKING:
    from tomlkit.toml_document import TOMLDocument


def install(root: Path) -> None:
    """Configure and activate Codedoctor in a Git repository.

    Args:
      root: Repository root; no source files or existing policy are rewritten.

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
    configured_hooks = hook_config(hooks.read_text() if hooks.exists() else "")
    project.write_text(configured_project)
    subprocess.run(  # noqa: S603 -- Fixed uv command and package names.
        [
            shutil.which("uv") or "uv",
            "add",
            "--group",
            "pre-commit",
            "--no-sync",
            "codedoctor[coverage,mutation,notebook]",
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
            "--hook-type",
            "pre-push",
        ],
        cwd=root,
        check=True,
    )
    _bootstrap_hooks(root)
    _update_gitignore(root)


def project_config(text: str, *, name: str) -> str:
    """Return TOML with missing setup defaults, preserving existing policy.

    Args:
      text: Existing pyproject content, or empty for a new project.
      name: Directory name used only when creating a project.

    Returns:
      configured: Comment-preserving project configuration.

    """
    document = tomlkit.parse(text)
    dependencies = _table(document, key="dependency-groups")
    dependencies.setdefault("pre-commit", [])
    tool = _table(document, key="tool")
    uv = _table(tool, key="uv")
    if "project" not in document:
        project = _table(document, key="project")
        project["name"] = "-".join(re.findall(r"[a-z0-9]+", name.lower())) or "project"
        project["version"] = "0.1.0"
        project["requires-python"] = ">=3.12"
        project["dependencies"] = []
        uv.setdefault("package", False)
    groups = _table(uv, key="dependency-groups")
    group = _table(groups, key="pre-commit")
    group.setdefault("requires-python", ">=3.12")
    pytest = _table(tool, key="pytest")
    options = _table(pytest, key="ini_options")
    options.setdefault("python_files", ["test_*.py", "*_test.py"])
    houselint = _table(tool, key="houselint")
    houselint.setdefault("select", "ALL")
    houselint.setdefault("fix", "ALL")
    houselint.setdefault("exclude", _test_exclusions())
    return tomlkit.dumps(document)


def hook_config(text: str) -> str:
    """Return YAML with missing Codedoctor hooks, preserving existing entries.

    Args:
      text: Existing pre-commit configuration, or empty for a new project.

    Returns:
      configured: Hook configuration, unchanged when already installed.

    Raises:
      ValueError: Configuration is malformed or a hook ID conflicts.

    """
    document = from_plain(yaml.safe_load(text), dict[str, object], default={})
    repos = (
        from_plain(document["repos"], list[dict[str, object]])
        if "repos" in document
        else []
    )
    expected = _hook_repos()
    wanted = [
        ({key: value for key, value in repo.items() if key != "hooks"}, hook)
        for repo in expected
        for hook in from_plain(repo["hooks"], list[dict[str, object]])
    ]
    existing = [
        ({key: value for key, value in repo.items() if key != "hooks"}, hook)
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
        metadata = {key: value for key, value in repo.items() if key != "hooks"}
        hooks = [
            hook
            for hook in from_plain(repo["hooks"], list[dict[str, object]])
            if (metadata, hook) not in existing
        ]
        if hooks:
            missing.append({**metadata, "hooks": hooks})
    if not missing:
        return text
    return _append_repos(text, repos=missing)


class _HouselintDefaults(TypedDict):
    exclude: list[str]


class _ToolDefaults(TypedDict):
    houselint: _HouselintDefaults


class _ProjectDefaults(TypedDict):
    tool: _ToolDefaults


def _test_exclusions() -> list[str]:
    """Read portable test exemptions from the package's canonical policy."""
    template = _THIS.with_name("install_defaults.toml")
    # Only the wheel holds the copy; a checkout, monorepo or export, reads its own.
    if not template.is_file():
        template = next(
            folder / "pyproject.toml"
            for folder in _THIS.parents
            if (folder / "pyproject.toml").is_file()
        )
    config = from_plain(tomllib.loads(template.read_text()), _ProjectDefaults)
    return [
        spec.replace("*_test.py", "*_test.py test_*.py")
        for spec in config["tool"]["houselint"]["exclude"]
        if set(spec.partition("=")[0].split())
        & {"*_test.py", "**/tests/test*.py", "conftest.py"}
    ]


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
            }
        ]
        if hooks:
            portable.append({**repo, "hooks": hooks})
    return portable


def _append_repos(text: str, *, repos: list[dict[str, object]]) -> str:
    """Insert repositories without reserializing existing configuration."""
    tree = yaml.compose(text, Loader=yaml.SafeLoader)
    if tree is None:
        return (
            text
            + "\n"
            + yaml.safe_dump(
                {
                    "fail_fast": False,
                    "default_stages": ["pre-commit"],
                    "default_install_hook_types": ["pre-commit", "pre-push"],
                    "repos": repos,
                },
                sort_keys=False,
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
            sort_keys=False,
        ).strip()[1:-1]
        position = (
            tree.value[-1][1].end_mark.index
            if tree.value
            else tree.start_mark.index + 1
        )
        return (
            text[:position] + (", " if tree.value else "") + fragment + text[position:]
        )
    return (
        text.rstrip()
        + "\nrepos:\n"
        + textwrap.indent(yaml.safe_dump(repos, sort_keys=False), "  ")
    )


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
            sort_keys=False,
        ).strip()[1:-1]
        return (
            text[:position] + (", " if node.value else "") + fragment + text[position:]
        )
    position = node.end_mark.index
    fragment = textwrap.indent(
        yaml.safe_dump(repos, sort_keys=False),
        " " * node.start_mark.column,
    )
    return text[:position].rstrip() + "\n" + fragment + text[position:]


def _table(
    parent: TOMLDocument | Table | InlineTable,
    *,
    key: str,
) -> Table | InlineTable:
    """Return a TOML table, creating it only when absent."""
    item = cast(object, parent.setdefault(key, tomlkit.table()))
    if not isinstance(item, Table | InlineTable):
        raise TypeError(f"Expected a TOML table for {key!r}.")
    return item


def _bootstrap_hooks(root: Path) -> None:
    """Keep generated Git hooks usable after an exact uv sync."""
    git = subprocess.run(  # noqa: S603 -- Fixed Git command, no user input.
        [shutil.which("git") or "git", "rev-parse", "--git-common-dir"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    directory = root / git.stdout.strip() / "hooks"
    for name in ("pre-commit", "pre-push"):
        path = directory / name
        content = path.read_text()
        # Retain pre-commit's identity so reinstall/uninstall preserves legacy hooks.
        header = content[: content.index("# start templated\n")]
        path.write_text(
            header
            + "exec uv run --frozen --group pre-commit pre-commit hook-impl "
            + f"--config=.pre-commit-config.yaml --hook-type={name} "
            + '--hook-dir="$(dirname "$0")" -- "$@"\n',
        )


def _update_gitignore(root: Path) -> None:
    """Append missing generated-state rules without rewriting existing entries."""
    path = root / ".gitignore"
    text = path.read_text() if path.exists() else ""
    existing = set(text.splitlines())
    missing = [
        pattern
        for pattern in (
            ".testmondata*",
            ".codegraph/",
            ".coverage",
            ".coverage.*",
            ".coverage-*",
        )
        if pattern not in existing
    ]
    if missing:
        with path.open("ab") as stream:
            if text and not text.endswith("\n"):
                stream.write(b"\n")
            stream.write(("\n".join(missing) + "\n").encode())
