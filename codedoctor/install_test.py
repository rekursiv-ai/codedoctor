"""Repository setup preserves existing policy and is repeatable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil
import subprocess
import tomllib

from ruamel.yaml import YAML
from treekle import MutablePlainTree, from_plain

import pytest
import rignore
import tomlkit
import yaml

from codedoctor import codegraph, install, policy


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def test_project_config_creates_a_virtual_project() -> None:
    text = install.project_config("", name="--A new.repo--")
    document = tomllib.loads(text)
    assert document["project"] == {
        "name": "a-new-repo",
        "version": "0.1.0",
        "requires-python": ">=3.12",
        "dependencies": [],
    }
    assert document["dependency-groups"]["pre-commit"] == []
    assert document["tool"]["uv"]["package"] is False
    assert document["tool"]["codedoctor"]["lint"]["select"] == "ALL"
    assert document["tool"]["codedoctor"]["lint"]["fix"] == "ALL"
    settings = from_plain(document, dict[str, MutablePlainTree])
    tools = policy.table(settings, key="tool")
    lint = policy.table(policy.table(tools, key="codedoctor"), key="lint")
    exclusions = from_plain(lint["exclude"], list[str])
    assert any(
        "*_test.py" in spec and "assert" in spec.split("=")[1].split()
        for spec in exclusions
    )
    assert any("test_*.py" in spec for spec in exclusions)
    assert any("conftest.py" in spec for spec in exclusions)
    assert document["tool"]["uv"]["dependency-groups"]["pre-commit"] == {
        "requires-python": ">=3.12",
    }
    assert install.project_config(text, name="ignored") == text


def test_project_config_preserves_policy_and_comments() -> None:
    original = (
        '# Keep Python 3.10 support.\n[project]\nname = "old"\n'
        'requires-python = ">=3.10"\n\n[tool.ruff]\nline-length = 100\n'
        '[tool.pytest.ini_options]\npython_files = ["*_test.py"]\n'
        '[tool.uv]\ndefault-groups = ["dev"]\n'
        '[tool.uv.dependency-groups]\npre-commit = {requires-python = ">=3.13"}\n'
        '[tool.codedoctor.lint]\nselect = "digits"\nfix = ""\nexclude = ["custom.py = digits"]\n'
    )
    text = install.project_config(original, name="ignored")
    assert "# Keep Python 3.10 support." in text
    assert tomllib.loads(text)["tool"]["ruff"]["line-length"] == 100
    assert tomllib.loads(text)["tool"]["codedoctor"]["lint"]["select"] == "digits"
    assert tomllib.loads(text)["project"]["requires-python"] == ">=3.10"
    assert tomllib.loads(text)["tool"]["uv"]["dependency-groups"]["pre-commit"] == {
        "requires-python": ">=3.13",
    }


def test_project_config_accepts_interleaved_project_and_tool_tables() -> None:
    original = (
        '[project]\nname = "consumer"\nrequires-python = ">=3.12"\n'
        "[tool.ruff]\nline-length = 100\n"
        '[build-system]\nrequires = ["hatchling"]\n'
        '[project.scripts]\nconsumer = "consumer:main"\n'
        '[tool.ty.rules]\nall = "warn"\n'
        '[tool.ruff.lint]\nselect = ["ALL"]\n'
    )
    configured = install.project_config(original, name="ignored")
    parsed = tomllib.loads(configured)
    assert parsed["project"]["scripts"] == {"consumer": "consumer:main"}
    assert parsed["tool"]["ruff"]["line-length"] == 100
    assert parsed["tool"]["ruff"]["lint"]["select"] == ["ALL"]
    assert install.project_config(configured, name="ignored") == configured


def test_project_config_keeps_pytest_discovery_defaults() -> None:
    text = install.project_config('[project]\nname = "demo"\n', name="demo")
    config = tomllib.loads(text)
    defaults = tomllib.loads(tomlkit.dumps(policy.defaults(name="demo")))
    assert config["tool"]["pytest"] == defaults["pytest"]
    assert policy.overrides(tomlkit.parse(text)) == ()


def test_hook_config_preserves_existing_hooks_and_comments() -> None:
    original = (
        "# Custom checks stay.\ndefault_stages: [manual]\nrepos:\n"
        "  - repo: local\n    hooks:\n      - id: custom\n"
        "        name: custom\n        entry: custom\n        language: system\n"
    )
    configured = install.hook_config(original)
    assert configured.startswith(original)
    document = from_plain(yaml.safe_load(configured), dict[str, object])
    repos = from_plain(document["repos"], list[dict[str, object]])
    hooks = [
        hook
        for repo in repos[1:]
        for hook in from_plain(repo["hooks"], list[dict[str, object]])
    ]
    ids = [hook["id"] for hook in hooks]
    assert "ty-check-spool" in ids
    assert "ty-check" in ids
    assert "pytest-spool" in ids
    assert "pytest" in ids
    assert "pytest-global" in ids
    assert all("stages" in hook for hook in hooks)
    assert install.hook_config(configured) == configured


@pytest.mark.parametrize("notebooks", [False, True])
def test_hook_config_creates_valid_empty_config(notebooks: bool) -> None:
    text = install.hook_config("", notebooks=notebooks)
    assert "repos:" in text
    assert "uv --quiet run --frozen --group pre-commit" in text
    assert "loop." not in text
    assert "loop/" not in text
    assert "--no-fix" not in text
    assert "format --check" not in text
    config = from_plain(yaml.safe_load(text), dict[str, object])
    repos = from_plain(config["repos"], list[dict[str, object]])
    hooks = [
        hook
        for repo in repos
        for hook in from_plain(repo["hooks"], list[dict[str, object]])
    ]
    ids = {hook["id"] for hook in hooks}
    assert ids & {"nbstripout", "jupytext-sync"} == (
        {"nbstripout", "jupytext-sync"} if notebooks else set()
    )
    assert {
        "end-of-file-fixer",
        "trailing-whitespace",
        "check-added-large-files",
        "check-merge-conflict",
        "check-yaml",
        "check-toml",
        "actionlint",
        "uv-lock",
        "codedoctor-lint",
        "codespell",
        "ruff-check",
        "ruff-format",
        "codegraph-sync",
        "ty-check",
        "pyrefly-check",
        "basedpyright",
        "mutation",
        "coverage-floor",
        "ty-check-global",
        "pyrefly-check-global",
        "pytest-global",
        "pytest-slow-global",
        "pytest-ci-smoke-global",
        "pytest-integration-global",
    } <= ids
    assert (
        not {
            "cargo-fmt",
            "cargo-clippy",
            "check-export-leaks",
            "check-export-leaks-spool",
            "check-copybarista-pr-text",
        }
        & ids
    )
    ruff = [hook for hook in hooks if hook["id"] in {"ruff-check", "ruff-format"}]
    assert [hook["id"] for hook in ruff] == ["ruff-format", "ruff-check", "ruff-format"]
    assert ruff[1]["args"] == ["--fix", "--extend-select", "COM812"]
    checker = next(hook for hook in hooks if hook["id"] == "ty-check")
    assert checker["entry"] == (
        "uv --quiet run --frozen --group pre-commit python -m codedoctor typecheck ty-check"
    )
    assert config["fail_fast"] is False
    assert config["default_stages"] == ["pre-commit"]
    assert config["default_install_hook_types"] == ["pre-commit"]
    assert install.hook_config(text, notebooks=notebooks) == text


@pytest.mark.parametrize(
    "original",
    [
        "repos: []\n",
        "# Just a comment.\n",
        "fail_fast: true\n",
        "{}\n",
        "{fail_fast: true,}\n",
        "repos: [{repo: local, hooks: []},]\n",
        "repos: [{repo: local, hooks: [{id: custom, entry: custom, language: system, name: custom}]}]\n",
        "repos:\n- repo: local\n  hooks: []\nfail_fast: true\n",
        "repos: [{repo: local}]\n",
        "\n# Keep the leading blank line.\nfail_fast: true\n\n",
        "\n# Keep the leading blank line.\nrepos:\n- repo: local\n  hooks: []\n\n",
    ],
)
def test_hook_config_accepts_empty_repositories(original: str) -> None:
    text = install.hook_config(original)
    configured = from_plain(YAML(typ="safe", pure=True).load(text), dict[str, object])
    assert "repos" in configured
    assert install.hook_config(text) == text
    assert "ty-check-spool" in text
    if original.startswith("\n"):
        assert text.startswith(original.rstrip() + "\n")
        assert "\n\n\n" not in text


@pytest.mark.parametrize(
    "original",
    ["repos: broken\n", "repos: null\n", "repos: [\n", "[]\n"],
)
def test_hook_config_rejects_invalid_config(original: str) -> None:
    with pytest.raises((ValueError, yaml.YAMLError)):
        install.hook_config(original)


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (
            "\nfail_fast: true\n\n",
            "\nfail_fast: true\nrepos:\n  - hooks: []\n    repo: local\n",
        ),
        (
            "\nrepos:\n- repo: old\n  hooks: []\n\n",
            "\nrepos:\n- repo: old\n  hooks: []\n- hooks: []\n  repo: local\n",
        ),
    ],
)
def test_appending_repositories_preserves_leading_whitespace(
    original: str,
    expected: str,
) -> None:
    assert (
        install._append_repos(original, repos=[{"repo": "local", "hooks": []}])
        == expected
    )


def test_hook_config_preserves_project_owned_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults: list[dict[str, object]] = [
        {
            "repo": "checks",
            "rev": "v1",
            "hooks": [{"id": "size", "args": ["--maxkb=800", "--enforce-all"]}],
        },
    ]
    monkeypatch.setattr(install, "_hook_repos", lambda: defaults)
    original = (
        "repos:\n- repo: checks\n  rev: v1\n  hooks:\n"
        "  - id: size\n    args: [--maxkb=5000, --enforce-all]\n"
    )
    assert install.hook_config(original) == original


@pytest.mark.parametrize("flow", [False, True])
@pytest.mark.parametrize(
    "prefix",
    ["uv run", "uv --quiet run --frozen --group pre-commit"],
)
def test_hook_config_upgrades_stage_commands_without_losing_custom_arguments(
    monkeypatch: pytest.MonkeyPatch,
    flow: bool,
    prefix: str,
) -> None:
    hooks: list[dict[str, object]] = [
        {
            "id": "codedoctor-lint",
            "name": "codedoctor lint",
            "entry": "uv run python -m codedoctor lint",
            "language": "system",
        },
        {"id": "codegraph-sync", "entry": "uv run python -m codedoctor codegraph-sync"},
        {"id": "pytest", "entry": "uv run python -m codedoctor testing pytest"},
    ]
    for hook in hooks:
        hook["entry"] = from_plain(hook["entry"], str).replace("uv run", prefix)
    defaults: list[dict[str, object]] = [{"repo": "local", "hooks": hooks}]
    monkeypatch.setattr(install, "_hook_repos", lambda: defaults)
    original = (
        "# Keep this comment.\nrepos:\n- repo: local\n  hooks:\n"
        "  - &lint\n    id: houselint\n    name: houselint\n    language: system\n"
        "    entry: uv run python -m codedoctor.lint\n"
        "    args: [--fix, '']  # Report only.\n"
        "  - *lint\n"
        "  - id: codegraph-sync\n    entry: >-\n"
        "      uv run python -m codedoctor.affected.codegraph_sync\n"
        "  - id: pytest\n    entry: uv run python -m codedoctor.testing pytest\n"
    ).replace("uv run", prefix)
    if flow:
        original = "# Keep this comment.\n" + yaml.safe_dump(
            yaml.safe_load(original),
            default_flow_style=True,
        )
    configured = install.hook_config(original)
    expected = [
        {
            "repo": "local",
            "hooks": [
                {**hooks[0], "args": ["--fix", ""]},
                {**hooks[0], "args": ["--fix", ""]},
                *hooks[1:],
            ],
        },
    ]
    assert yaml.safe_load(configured) == {"repos": expected}
    assert "# Keep this comment." in configured
    if not flow:
        assert "# Report only." in configured
    assert install.hook_config(configured) == configured


@pytest.mark.parametrize("style", ["|-", ">-"])
@pytest.mark.parametrize("name", ["", "    name: Project lint\n"])
@pytest.mark.parametrize(
    ("module", "command"),
    [
        ("export_leaks.check_pr_text", "check-pr-text"),
        ("assert_isinstance", "assert-isinstance"),
    ],
)
def test_hook_migration_preserves_scalar_boundaries_and_custom_names(
    style: str,
    name: str,
    module: str,
    command: str,
) -> None:
    expected: list[dict[str, object]] = [
        {
            "hooks": [
                {"id": "unrelated"},
                {"entry": "uv run python -m codedoctor lint"},
                {"entry": f"uv run python -m codedoctor {command}"},
            ],
        },
    ]
    original = (
        f"repos:\n- hooks:\n  - id: houselint\n{name}    entry: {style}\n"
        "      uv run python -m codedoctor.lint\n"
        "    args: [--fix, '', --plugin, codedoctor.testing.listing]\n"
        f"  - id: custom\n    entry: uv run python -m codedoctor.{module}\n"
    )
    migrated = from_plain(
        yaml.safe_load(install._upgrade_hooks(original, expected=expected)),
        dict[str, object],
    )
    repos = from_plain(migrated["repos"], list[dict[str, object]])
    hooks = from_plain(repos[0]["hooks"], list[dict[str, object]])
    assert hooks[0] == {
        "id": "codedoctor-lint",
        "entry": "uv run python -m codedoctor lint",
        "args": ["--fix", "", "--plugin", "codedoctor.testing.listing"],
        **({"name": "Project lint"} if name else {}),
    }
    assert hooks[1] == {
        "id": "custom",
        "entry": f"uv run python -m codedoctor {command}",
    }


def test_hook_migration_continues_after_a_shared_node() -> None:
    expected: list[dict[str, object]] = [
        {
            "hooks": [
                {"entry": "uv run python -m codedoctor lint"},
                {"entry": "uv run python -m codedoctor testing"},
            ],
        },
    ]
    original = (
        "hooks:\n- {id: pytest, entry: uv run python -m codedoctor.testing}\n"
        "- &lint {id: houselint, entry: uv run python -m codedoctor.lint}\n- *lint\n"
    )
    migrated = install._upgrade_hooks(original, expected=expected)
    assert yaml.safe_load(migrated) == {
        "hooks": [
            {"id": "pytest", "entry": "uv run python -m codedoctor testing"},
            {"id": "codedoctor-lint", "entry": "uv run python -m codedoctor lint"},
            {"id": "codedoctor-lint", "entry": "uv run python -m codedoctor lint"},
        ],
    }


def test_hook_migration_visits_recursive_metadata_once() -> None:
    original = "metadata: &metadata {self: *metadata}\n"
    assert install._upgrade_hooks(original, expected=[]) == original


def test_hook_config_rejects_conflicting_hook_before_writing() -> None:
    original = (
        "repos:\n  - repo: local\n    hooks:\n      - id: ty-check\n"
        "        entry: different\n"
    )
    with pytest.raises(ValueError, match="ty-check"):
        install.hook_config(original)


@pytest.mark.parametrize("found_executables", [True, False])
@pytest.mark.parametrize("existing_pre_push", [True, False])
@pytest.mark.parametrize("notebooks", [False, True])
def test_install_locks_dependencies_then_installs_only_pre_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    found_executables: bool,
    existing_pre_push: bool,
    notebooks: bool,
) -> None:
    calls: list[list[str]] = []
    uv = "/tools/uv" if found_executables else "uv"
    git = "/tools/git" if found_executables else "git"
    hook_dir = tmp_path / ".git" / "hooks"
    hook_dir.mkdir(parents=True)
    pre_push = hook_dir / "pre-push"
    if existing_pre_push:
        pre_push.write_text("custom pre-push hook\n")
    ignored = tmp_path / ".gitignore"
    original_ignore = "# Keep existing rules.\ncustom/"
    ignored.write_text(original_ignore)
    original_input = '[project]\nname = "existing"\nversion = "2.0.0"\n'
    (tmp_path / "pyproject.toml").write_text(original_input)

    def run(
        argv: Sequence[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path
        assert check
        calls.append(list(argv))
        if argv[0] == git:
            assert list(argv) == [git, "rev-parse", "--git-common-dir"]
            assert capture_output
            assert text
        elif "install" in argv:
            for name in argv[argv.index("install") + 2 :: 2]:
                path = hook_dir / name
                path.write_text(
                    "#!/bin/sh\n# pre-commit identity\n# start templated\n"
                    "old runner\n# start templated\nlegacy runner\n",
                )
                path.chmod(0o755)
        return subprocess.CompletedProcess(argv, 0, stdout=str(tmp_path / ".git"))

    def which(name: str) -> str | None:
        return f"/tools/{name}" if found_executables else None

    ensured: list[str] = []
    monkeypatch.setattr(codegraph, "install", lambda: ensured.append("codegraph"))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shutil, "which", which)
    install.install(tmp_path, notebooks=notebooks)
    assert ignored.is_file()
    configured_ignore = ignored.read_text()
    assert configured_ignore.startswith(original_ignore)
    assert configured_ignore != original_ignore
    original_project = (tmp_path / "pyproject.toml").read_text()
    assert original_project.startswith(original_input)
    assert (
        tomllib.loads(original_project)["tool"]["codedoctor"]["lint"]["select"] == "ALL"
    )
    original_hooks = (tmp_path / ".pre-commit-config.yaml").read_text()
    assert ("id: nbstripout" in original_hooks) == notebooks
    assert ("id: jupytext-sync" in original_hooks) == notebooks
    install.install(tmp_path, notebooks=notebooks)
    assert (tmp_path / "pyproject.toml").read_text() == original_project
    assert (tmp_path / ".pre-commit-config.yaml").read_text() == original_hooks
    assert ignored.read_text() == configured_ignore
    assert calls[0] == [
        uv,
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
    ]
    assert calls[1] == [
        uv,
        "run",
        "--locked",
        "--group",
        "pre-commit",
        "pre-commit",
        "install",
        "--hook-type",
        "pre-commit",
    ]
    assert calls[2] == [git, "rev-parse", "--git-common-dir"]
    assert ensured == ["codegraph", "codegraph"]
    assert (tmp_path / "pyproject.toml").is_file()
    assert (tmp_path / ".pre-commit-config.yaml").is_file()
    if existing_pre_push:
        assert pre_push.read_text() == "custom pre-push hook\n"
    else:
        assert not pre_push.exists()
    path = hook_dir / "pre-commit"
    assert path.read_text() == (
        "#!/bin/sh\n# pre-commit identity\n"
        "exec uv run --frozen --group pre-commit pre-commit hook-impl "
        "--config=.pre-commit-config.yaml --hook-type=pre-commit "
        '--hook-dir="$(dirname "$0")" -- "$@"\n'
    )
    assert path.stat().st_mode & 0o111


@pytest.mark.parametrize(
    "backend",
    ["rignore", pytest.param("git", marks=pytest.mark.cli_git)],
)
@pytest.mark.parametrize(
    ("original_ignore", "custom_ignored"),
    [
        (None, False),
        ("", False),
        ("# Keep existing rules.\ncustom/", True),
        ("# Existing coverage rule.\r\n.coverage\r\n", False),
    ],
)
def test_git_ignores_generated_state_without_hiding_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original_ignore: str | None,
    custom_ignored: bool,
    backend: str,
) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    ignored = tmp_path / ".gitignore"
    if original_ignore is not None:
        ignored.write_bytes(original_ignore.encode())
    install._update_gitignore(tmp_path)
    configured = ignored.read_bytes()
    original = (original_ignore or "").encode()
    assert configured.startswith(original)
    appended = configured[len(original) :]
    if original and not original.endswith(b"\n"):
        assert appended.startswith(b"\n")
        appended = appended[1:]
    assert appended.endswith(b"\n")
    assert all(appended.splitlines())
    assert len(configured.splitlines()) == len(set(configured.splitlines()))
    generated = {
        ".testmondata",
        ".testmondata-wal",
        ".codegraph/index.db",
        ".coverage",
        ".coverage.worker.123",
        ".coverage-journal",
    }
    paths = [
        *sorted(generated),
        ".coveragerc",
        ".codegraph.py",
        "app.py",
        "custom/file.txt",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    if backend == "git":
        git = shutil.which("git")
        assert git is not None
        subprocess.run(  # noqa: S603 -- Git resolved from PATH; fixed arguments.
            [git, "init", "--quiet"],
            cwd=tmp_path,
            check=True,
        )
        result = subprocess.run(  # noqa: S603 -- Git resolved from PATH; fixture paths on stdin.
            [git, "check-ignore", "--no-index", "--stdin"],
            cwd=tmp_path,
            input="\n".join(paths) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        excluded = set(result.stdout.splitlines())
    else:
        visible = set(
            rignore.walk(
                tmp_path,
                ignore_hidden=False,
                require_git=False,
                read_parents_ignores=False,
                read_global_git_ignore=False,
                read_git_exclude=False,
                read_ignore_files=False,
            ),
        )
        excluded = {name for name in paths if tmp_path / name not in visible}
    assert excluded == generated | ({"custom/file.txt"} if custom_ignored else set())
    install._update_gitignore(tmp_path)
    assert ignored.read_bytes() == configured


def test_invalid_config_prevents_partial_setup(tmp_path: Path) -> None:
    path = tmp_path / ".pre-commit-config.yaml"
    path.write_text("repos: broken\n")
    with pytest.raises(ValueError, match=r"repos|list"):
        install.install(tmp_path)
    assert not (tmp_path / "pyproject.toml").exists()
    assert path.read_text() == "repos: broken\n"


def test_install_propagates_dependency_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(
        argv: Sequence[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path
        assert check
        raise subprocess.CalledProcessError(7, argv)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        install.install(tmp_path)
    assert caught.value.returncode == 7


@pytest.mark.parametrize("text", ["tool = 4\n", '[tool]\nuv = "broken"\n'])
def test_project_config_rejects_invalid_tables(text: str) -> None:
    with pytest.raises(TypeError, match="TOML table"):
        install.project_config(text, name="demo")


def test_packaged_hook_template_is_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = tmp_path / "install.py"
    module.with_name("install_hooks.yaml").write_text(
        "repos:\n- repo: https://example.test/hooks\n  rev: v1\n"
        "  hooks:\n  - id: custom\n    stages: [pre-commit]\n",
    )
    monkeypatch.setattr(install, "_THIS", module)
    text = install.hook_config("")
    assert "repo: https://example.test/hooks" in text
    assert "rev: v1" in text
    assert "id: custom" in text
    assert install.hook_config(text) == text


def test_project_name_without_letters_is_valid() -> None:
    text = install.project_config("", name="...")
    assert tomllib.loads(text)["project"]["name"] == "project"


def test_packaged_defaults_preserve_test_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = tmp_path / "install.py"
    module.with_name("install_defaults.toml").write_text(
        '[tool.codedoctor.lint]\nexclude = ["*_test.py **/tests/test*.py = assert globals", '
        '"conftest.py = module-side-effects", "private.py = globals"]\n',
    )
    monkeypatch.setattr(policy, "_THIS", module)
    configured = tomllib.loads(install.project_config("", name="demo"))
    assert configured["tool"]["codedoctor"]["lint"] == {
        "select": "ALL",
        "fix": "ALL",
        "set": [
            "import-alias.first-party=demo",
            "test-footer.runner=codedoctor.lib.testing.runner",
            "userdirs-namespace.namespace=demo",
        ],
        "exclude": [
            "*_test.py test_*.py **/tests/test*.py = assert globals",
            "conftest.py = module-side-effects",
        ],
    }


def test_a_checkout_without_the_packaged_defaults_reads_its_nearest_pyproject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the wheel carries ``install_defaults.toml``; a source tree has none."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.codedoctor.lint]\nexclude = ["conftest.py = module-side-effects"]\n',
    )
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    # A test pattern named only after the ``=`` is a rule's argument, not its files.
    (tmp_path / "pkg" / "pyproject.toml").write_text(
        "[tool.codedoctor.lint]\nexclude = ["
        '"*_test.py = assert", "**/tests/test*.py = globals", '
        '"private.py = conftest.py = globals"]\n',
    )
    monkeypatch.setattr(policy, "_THIS", tmp_path / "pkg" / "sub" / "install.py")

    configured = tomllib.loads(install.project_config("", name="demo"))

    assert configured["tool"]["codedoctor"]["lint"]["exclude"] == [
        "*_test.py test_*.py = assert",
        "**/tests/test*.py = globals",
    ]


@pytest.mark.parametrize("supported", [">=3.10", "==3.14.*", "~=3.13", "<4"])
def test_checker_versions_follow_the_project_floor(supported: str) -> None:
    text = install.project_config(
        f'[project]\nname = "my-project"\nrequires-python = "{supported}"\n',
        name="directory",
    )
    config = tomllib.loads(text)
    version = {">=3.10": "3.10", "==3.14.*": "3.14", "~=3.13": "3.13", "<4": "3.12"}[
        supported
    ]
    assert config["tool"]["ruff"]["target-version"] == "py" + version.replace(".", "")
    assert config["tool"]["ruff"]["lint"]["isort"]["known-first-party"] == [
        "my_project",
    ]
    assert config["tool"]["ty"]["environment"]["python-version"] == version
    assert config["tool"]["pyrefly"]["python-version"] == version + ".0"
    assert config["tool"]["basedpyright"]["pythonVersion"] == version
    assert (
        config["tool"]["codedoctor"]["lint"]["set"][0]
        == "import-alias.first-party=my_project"
    )
    assert install.project_config(text, name="different-directory") == text


def test_check_detects_new_and_stale_configuration(tmp_path: Path) -> None:
    assert not install.current(tmp_path)
    path = tmp_path / "pyproject.toml"
    path.write_text(install.project_config("", name=tmp_path.name))
    assert install.current(tmp_path)
    document = tomlkit.parse(path.read_text())
    document["tool"]["ruff"]["show-fixes"] = False
    path.write_text(tomlkit.dumps(document))
    assert install.current(tmp_path)
    path.write_text('[project]\nname = "demo"\n')
    assert not install.current(tmp_path)


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
