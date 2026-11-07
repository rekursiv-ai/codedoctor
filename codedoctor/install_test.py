"""Repository setup preserves existing policy and is repeatable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil
import subprocess
import tomllib

from treekle import from_plain

import pytest
import yaml

from codedoctor import install


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
    assert document["tool"]["houselint"]["select"] == "ALL"
    assert document["tool"]["houselint"]["fix"] == "ALL"
    policy = from_plain(tomllib.loads(text), install._ProjectDefaults)
    exclusions = policy["tool"]["houselint"]["exclude"]
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
        '[tool.houselint]\nselect = "digits"\nfix = ""\nexclude = ["custom.py = digits"]\n'
    )
    text = install.project_config(original, name="ignored")
    assert text.startswith(original)
    assert "# Keep Python 3.10 support." in text
    assert tomllib.loads(text)["project"]["requires-python"] == ">=3.10"
    assert tomllib.loads(text)["tool"]["uv"]["dependency-groups"]["pre-commit"] == {
        "requires-python": ">=3.13",
    }


def test_project_config_keeps_pytest_discovery_defaults() -> None:
    text = install.project_config('[project]\nname = "demo"\n', name="demo")
    config = tomllib.loads(text)
    assert config["tool"]["pytest"]["ini_options"]["python_files"] == [
        "test_*.py",
        "*_test.py",
    ]


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


def test_hook_config_creates_valid_empty_config() -> None:
    text = install.hook_config("")
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
    assert {
        "end-of-file-fixer",
        "trailing-whitespace",
        "check-added-large-files",
        "check-merge-conflict",
        "check-yaml",
        "check-toml",
        "actionlint",
        "uv-lock",
        "houselint",
        "codespell",
        "nbstripout",
        "jupytext-sync",
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
        "uv --quiet run --frozen --group pre-commit python -m codedoctor.typecheck ty-check"
    )
    assert config["default_stages"] == ["pre-commit"]
    assert config["default_install_hook_types"] == ["pre-commit", "pre-push"]
    assert install.hook_config(text) == text


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
    ],
)
def test_hook_config_accepts_empty_repositories(original: str) -> None:
    text = install.hook_config(original)
    assert install.hook_config(text) == text
    assert "ty-check-spool" in text


@pytest.mark.parametrize(
    "original",
    ["repos: broken\n", "repos: null\n", "repos: [\n", "[]\n"],
)
def test_hook_config_rejects_invalid_config(original: str) -> None:
    with pytest.raises((ValueError, yaml.YAMLError)):
        install.hook_config(original)


def test_hook_config_rejects_conflicting_hook_before_writing() -> None:
    original = (
        "repos:\n  - repo: local\n    hooks:\n      - id: ty-check\n"
        "        entry: different\n"
    )
    with pytest.raises(ValueError, match="ty-check"):
        install.hook_config(original)


def test_install_locks_dependencies_then_installs_both_git_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    hook_dir = tmp_path / ".git" / "hooks"
    hook_dir.mkdir(parents=True)

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
        if argv[0] == "git":
            assert list(argv) == ["git", "rev-parse", "--git-common-dir"]
            assert capture_output
            assert text
        elif "install" in argv:
            for name in ("pre-commit", "pre-push"):
                path = hook_dir / name
                path.write_text(
                    "#!/bin/sh\n# pre-commit identity\n# start templated\nold runner\n",
                )
                path.chmod(0o755)
        return subprocess.CompletedProcess(argv, 0, stdout=str(tmp_path / ".git"))

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shutil, "which", str)
    install.install(tmp_path)
    original_project = (tmp_path / "pyproject.toml").read_text()
    original_hooks = (tmp_path / ".pre-commit-config.yaml").read_text()
    install.install(tmp_path)
    assert (tmp_path / "pyproject.toml").read_text() == original_project
    assert (tmp_path / ".pre-commit-config.yaml").read_text() == original_hooks
    assert calls[0][:6] == [
        "uv",
        "add",
        "--group",
        "pre-commit",
        "--no-sync",
        "codedoctor[coverage,mutation,notebook]",
    ]
    assert {
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
    } <= set(calls[0])
    assert "rekursiv-ai-typings" not in calls[0]
    assert calls[1] == [
        "uv",
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
    ]
    assert (tmp_path / "pyproject.toml").is_file()
    assert (tmp_path / ".pre-commit-config.yaml").is_file()
    for name in ("pre-commit", "pre-push"):
        path = hook_dir / name
        assert path.read_text() == (
            "#!/bin/sh\n# pre-commit identity\n"
            "exec uv run --frozen --group pre-commit pre-commit hook-impl "
            f"--config=.pre-commit-config.yaml --hook-type={name} "
            '--hook-dir="$(dirname "$0")" -- "$@"\n'
        )
        assert path.stat().st_mode & 0o111


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
        '[tool.houselint]\nexclude = ["*_test.py **/tests/test*.py = assert globals", '
        '"conftest.py = module-side-effects", "private.py = globals"]\n',
    )
    monkeypatch.setattr(install, "_THIS", module)
    configured = tomllib.loads(install.project_config("", name="demo"))
    assert configured["tool"]["houselint"] == {
        "select": "ALL",
        "fix": "ALL",
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
        '[tool.houselint]\nexclude = ["conftest.py = module-side-effects"]\n',
    )
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    # A test pattern named only after the ``=`` is a rule's argument, not its files.
    (tmp_path / "pkg" / "pyproject.toml").write_text(
        "[tool.houselint]\nexclude = ["
        '"*_test.py = assert", "**/tests/test*.py = globals", '
        '"private.py = conftest.py = globals"]\n',
    )
    monkeypatch.setattr(install, "_THIS", tmp_path / "pkg" / "sub" / "install.py")

    configured = tomllib.loads(install.project_config("", name="demo"))

    assert configured["tool"]["houselint"]["exclude"] == [
        "*_test.py test_*.py = assert",
        "**/tests/test*.py = globals",
    ]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
