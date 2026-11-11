"""Portable policy tracks upstream defaults without erasing project decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tomllib

import pytest
import tomlkit

from codedoctor import policy


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import MutablePlainTree


def test_defaults_use_the_canonical_checker_policy() -> None:
    defaults = tomllib.loads(tomlkit.dumps(policy.defaults(name="netv")))
    assert defaults["ruff"]["lint"]["select"] == ["ALL"]
    assert "E501" in defaults["ruff"]["lint"]["ignore"]
    assert defaults["ruff"]["lint"]["isort"]["known-first-party"] == ["netv"]
    assert defaults["ruff"]["target-version"] == "py312"
    assert defaults["basedpyright"]["typeCheckingMode"] == "recommended"
    assert defaults["basedpyright"]["failOnWarnings"] is True
    assert defaults["basedpyright"]["reportUnusedCallResult"] is False
    assert defaults["ty"]["rules"]["all"] == "warn"
    assert defaults["ty"]["terminal"]["error-on-warning"] is True
    assert defaults["pyrefly"]["preset"] == "strict"
    assert defaults["pyrefly"]["errors"]["unused-call-result"] == "ignore"
    assert defaults["mutmut"]["source_paths"] == ["."]
    assert defaults["coverage-floor"]["floor"] == ["=100"]
    assert defaults["pytest"]["ini_options"]["filterwarnings"][0] == "error"
    assert "*_test.py" in defaults["coverage"]["run"]["omit"]
    assert "executionEnvironments" not in defaults["basedpyright"]
    assert (
        "pathlib.Path.home"
        not in defaults["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"]
    )
    assert defaults["codedoctor"]["lint"]["set"] == [
        "import-alias.first-party=netv",
        "test-footer.runner=codedoctor.lib.testing.runner",
        "userdirs-namespace.namespace=netv",
    ]
    # Mutation snapshots inject their own plugins into this table.
    del defaults["mutmut"]
    rendered = tomlkit.dumps(defaults)
    assert "loop/" not in rendered
    assert "loop." not in rendered
    assert "website/" not in rendered


def test_packaged_policy_is_the_source_of_new_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = tmp_path / "policy.py"
    module.with_name("install_defaults.toml").write_text(
        '[project]\nname = "codedoctor"\n[tool.ruff.lint]\n'
        'select = ["FUTURE"]\n[tool.ty.rules]\nnew-rule = "error"\n'
        '[tool.pyrefly.errors]\nnew-error = "warn"\n'
        "[tool.basedpyright]\nreportFutureError = true\n"
        '[tool.mutmut]\nsource_paths = ["upstream"]\nmutate_only_covered_lines = true\n',
    )
    monkeypatch.setattr(policy, "_THIS", module)
    defaults = tomllib.loads(tomlkit.dumps(policy.defaults(name="consumer")))
    assert defaults["ruff"]["lint"]["select"] == ["FUTURE"]
    assert defaults["ty"]["rules"]["new-rule"] == "error"
    assert defaults["pyrefly"]["errors"]["new-error"] == "warn"
    assert defaults["basedpyright"]["reportFutureError"] is True
    assert defaults["mutmut"]["source_paths"] == ["."]
    assert defaults["mutmut"]["mutate_only_covered_lines"] is True


def test_defaults_drop_checkout_settings_and_name_the_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every setting the checkout owns is replaced, and nothing else is kept."""
    module = tmp_path / "policy.py"
    module.with_name("install_defaults.toml").write_text(
        "[tool.ruff.lint.flake8-tidy-imports.banned-api]\n"
        '"pathlib.Path.home".msg = "Use userdirs."\n'
        '"os.system".msg = "Use subprocess."\n'
        "[tool.basedpyright]\n"
        'executionEnvironments = [{root = "src"}]\n'
        'stubPath = "src/typings"\n'
        'typeCheckingMode = "recommended"\n'
        "[tool.pyrefly]\n"
        'python-interpreter-path = ".venv/bin/python"\n'
        'python-platform = "linux"\n'
        'preset = "strict"\n'
        "[tool.codedoctor.lint]\n"
        'exclude = ["*_test.py = assert", "conftest.py = globals", '
        '"**/tests/test*.py = globals", "private.py = globals", '
        # A test pattern after the ``=`` is a rule's argument, not the files.
        '"private.py = conftest.py = globals"]\n'
        "[tool.unrelated]\nkept = false\n",
    )
    monkeypatch.setattr(policy, "_THIS", module)
    stubs = ["typings", ".venv/share/rekursiv-ai-typings"]

    assert policy.defaults(name="my-project", python_version="3.10") == {
        "ruff": {
            "target-version": "py310",
            "lint": {
                "flake8-tidy-imports": {
                    "banned-api": {"os.system": {"msg": "Use subprocess."}},
                },
                "isort": {"known-first-party": ["my_project"]},
            },
        },
        "basedpyright": {
            "typeCheckingMode": "recommended",
            "pythonVersion": "3.10",
            "extraPaths": stubs,
        },
        "pyrefly": {
            "preset": "strict",
            "python-version": "3.10.0",
            "search-path": stubs,
        },
        "ty": {"environment": {"python-version": "3.10", "extra-paths": stubs}},
        "mutmut": {"source_paths": ["."]},
        "codedoctor": {
            "lint": {
                "select": "ALL",
                "fix": "ALL",
                "exclude": [
                    "*_test.py test_*.py = assert",
                    "conftest.py = globals",
                    "**/tests/test*.py = globals",
                ],
                "set": [
                    "import-alias.first-party=my_project",
                    "test-footer.runner=codedoctor.lib.testing.runner",
                    "userdirs-namespace.namespace=my-project",
                ],
            },
        },
    }


def test_source_policy_finds_the_nearest_pyproject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 77\n")
    monkeypatch.setattr(policy, "_THIS", tmp_path / "pkg" / "policy.py")
    defaults = tomllib.loads(tomlkit.dumps(policy.defaults(name="app")))
    assert defaults["ruff"]["line-length"] == 77


def test_merge_adopts_missing_defaults_and_preserves_comments_and_overrides() -> None:
    document = tomlkit.parse("# Project decision.\n[tool.ruff]\nline-length = 100\n")
    defaults: dict[str, MutablePlainTree] = {
        "ruff": {"line-length": 88, "show-fixes": True},
    }
    policy.apply(document, defaults=defaults)
    assert document["tool"]["ruff"]["line-length"] == 100
    assert document["tool"]["ruff"]["show-fixes"] is True
    assert "# Project decision." in tomlkit.dumps(document)
    once = tomlkit.dumps(document)
    policy.apply(document, defaults=defaults)
    assert tomlkit.dumps(document) == once


def test_serialized_policy_is_byte_identical_after_reinstallation() -> None:
    document = tomlkit.document()
    defaults: dict[str, MutablePlainTree] = {
        "ruff": {"lint": {"select": ["ALL"]}},
        "ty": {"rules": {"all": "warn"}},
    }
    policy.apply(document, defaults=defaults)
    once = tomlkit.dumps(document)
    restored = tomlkit.parse(once)
    policy.apply(restored, defaults=defaults)
    assert tomlkit.dumps(restored) == once


def test_refresh_changes_managed_values_without_overwriting_project_edits() -> None:
    document = tomlkit.document()
    policy.apply(
        document,
        defaults={"ruff": {"line-length": 88, "lint": {"select": ["E"]}}},
    )
    document["tool"]["ruff"]["line-length"] = 100
    policy.apply(
        document,
        defaults={"ruff": {"line-length": 90, "lint": {"select": ["ALL"]}}},
    )
    assert document["tool"]["ruff"]["line-length"] == 100
    assert document["tool"]["ruff"]["lint"]["select"] == ["ALL"]
    assert policy.overrides(document) == ("ruff.line-length",)
    assert (
        tomllib.loads(tomlkit.dumps(document))["tool"]["codedoctor"]["managed"]["ruff"][
            "line-length"
        ]
        == 90
    )


def test_refresh_removes_retired_defaults_but_keeps_changed_and_deleted_settings() -> (
    None
):
    document = tomlkit.document()
    policy.apply(document, defaults={"ty": {"a": True, "b": True, "c": True}})
    document["tool"]["ty"]["b"] = False
    del document["tool"]["ty"]["c"]
    policy.apply(document, defaults={"ty": {"c": False, "d": True}})
    assert tomllib.loads(tomlkit.dumps(document))["tool"]["ty"] == {
        "b": False,
        "d": True,
    }
    assert policy.overrides(document) == ("ty.c",)


def test_refresh_preserves_an_overridden_table() -> None:
    document = tomlkit.parse('[tool.ruff]\nlint = "custom"\n')
    policy.apply(document, defaults={"ruff": {"lint": {"select": ["ALL"]}}})
    assert document["tool"]["ruff"]["lint"] == "custom"
    assert policy.overrides(document) == ("ruff.lint.select",)


def test_refresh_removes_nested_retired_settings() -> None:
    document = tomlkit.document()
    policy.apply(document, defaults={"ruff": {"lint": {"select": ["ALL"]}}})
    policy.apply(document, defaults={"ruff": {"lint": {"ignore": ["E501"]}}})
    assert tomllib.loads(tomlkit.dumps(document))["tool"]["ruff"]["lint"] == {
        "ignore": ["E501"],
    }


def test_table_accepts_inline_toml_and_rejects_plain_scalars() -> None:
    document = tomlkit.parse('[tool]\nruff = {lint = {select = ["ALL"]}}\n')
    tools = policy.table(document, key="tool")
    ruff = policy.table(tools, key="ruff")
    assert policy.table(ruff, key="lint")["select"] == ["ALL"]
    malformed: dict[str, MutablePlainTree] = {"ruff": 1}
    with pytest.raises(TypeError, match="Expected a TOML table for 'ruff'"):
        policy.table(malformed, key="ruff")
    with pytest.raises(TypeError, match="Expected a TOML table for 'ty'"):
        policy.table(tomlkit.parse("ty = 1\n"), key="ty")


@pytest.mark.parametrize(
    "text",
    ["tool = 1\n", "[tool]\ncodedoctor = 1\n", "[tool.codedoctor]\nmanaged = 1\n"],
)
def test_invalid_managed_configuration_is_rejected(text: str) -> None:
    with pytest.raises(TypeError):
        policy.apply(tomlkit.parse(text), defaults={"ruff": {"line-length": 88}})


@pytest.mark.parametrize("deleted", [False, True])
def test_lint_schema_migration_preserves_overrides_and_deleted_settings(
    deleted: bool,
) -> None:
    document = tomlkit.parse(
        '# Project lint policy.\n[tool.houselint]\nselect = "digits"\n'
        'fix = ""  # Report only.\nexclude = ["private.py"]\n'
        '[tool.codedoctor.managed.houselint]\nselect = "ALL"\n'
        'fix = "ALL"\nexclude = []\nset = ["old"]\n',
    )
    if deleted:
        del document["tool"]["houselint"]
    defaults: dict[str, MutablePlainTree] = {
        "codedoctor": {
            "lint": {"select": "ALL", "fix": "ALL", "exclude": [], "set": ["new"]},
        },
    }
    policy.apply(document, defaults=defaults)
    parsed = tomllib.loads(tomlkit.dumps(document))
    assert "houselint" not in parsed["tool"]
    assert parsed["tool"]["codedoctor"]["managed"] == defaults
    if deleted:
        assert "lint" not in parsed["tool"]["codedoctor"]
    else:
        assert parsed["tool"]["codedoctor"]["lint"] == {
            "select": "digits",
            "fix": "",
            "exclude": ["private.py"],
        }
        assert "# Report only." in tomlkit.dumps(document)
    assert policy.overrides(document) == tuple(
        f"codedoctor.lint.{key}" for key in ("exclude", "fix", "select", "set")
    )
    once = tomlkit.dumps(document)
    policy.apply(document, defaults=defaults)
    assert tomlkit.dumps(document) == once


def test_lint_schema_migration_rejects_competing_tables() -> None:
    document = tomlkit.parse(
        '[tool.houselint]\nselect = "digits"\n[tool.codedoctor.lint]\nselect = "ALL"\n',
    )
    with pytest.raises(
        ValueError,
        match=r"^Cannot migrate both houselint and codedoctor\.lint\.$",
    ):
        policy.apply(document, defaults={"codedoctor": {"lint": {"select": "ALL"}}})


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
