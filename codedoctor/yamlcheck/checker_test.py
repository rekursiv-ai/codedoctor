"""Compose tags preserve strict YAML validation."""

from __future__ import annotations

from pathlib import Path

import sys

from ruamel.yaml import YAML, YAMLError

import pytest

from codedoctor.yamlcheck import checker


@pytest.mark.parametrize(
    "text",
    [
        "",
        "services: {web: {image: example}}",
        "ports: !reset []",
        "ports: !override [8080:80]",
        "value: !reset null",
        "value: !override {nested: [one, two]}",
        "value: !reset {nested: !override [one]}",
        "value: &value !override [*value]",
        "ports: !reset []\nshared: &shared [one]\nreferences: [*shared, *shared]",
        "base: &base {image: example}\nweb: {<<: *base, ports: !reset []}",
    ],
)
def test_accepts_safe_yaml_and_compose_tags(
    text: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "compose.yml"
    path.write_text(text)
    assert checker.check([path]) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "text",
    [
        "ports: [",
        "value: *missing",
        "value: !unknown []",
        "value: !reset [!unknown hidden]",
        "value: !!python/object/apply:os.system ['false']",
        "value: !override [!!python/object/apply:os.system ['false']]",
        "key: first\nkey: second",
        "value: !reset {key: first, key: second}",
        "first: document\n---\nsecond: document",
        "value: !!int invalid",
    ],
)
def test_rejects_invalid_yaml(
    text: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "invalid.yml"
    path.write_text(text)
    assert checker.check([path]) == 1
    assert str(path) in capsys.readouterr().err


def test_reports_missing_file_and_continues(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.yml"
    invalid = tmp_path / "invalid.yml"
    invalid.write_text("key: !unknown value")
    assert checker.check([missing, invalid]) == 1
    errors = capsys.readouterr().err
    assert str(missing) in errors
    assert str(invalid) in errors


def test_does_not_change_other_yaml_loaders(tmp_path: Path) -> None:
    path = tmp_path / "compose.yml"
    path.write_text("ports: !reset []")
    assert checker.check([path]) == 0
    with pytest.raises(YAMLError):
        YAML(typ="safe").load("ports: !reset []")


def test_uses_pure_safe_loader_and_explicit_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unicode.yml"
    path.write_text("label: café\n", encoding="utf-8")
    loaders: list[tuple[str, bool]] = []

    def loader(*, typ: str, pure: bool) -> YAML:
        loaders.append((typ, pure))
        return YAML(typ=typ, pure=pure)

    reads: list[tuple[Path, str | None]] = []
    read = Path.read_text

    def read_text(self: Path, encoding: str | None = None) -> str:
        reads.append((self, encoding))
        return read(self, encoding=encoding)

    with monkeypatch.context() as patch:
        patch.setattr(checker, "YAML", loader)
        patch.setattr(Path, "read_text", read_text)
        assert checker.check([path]) == 0
    assert loaders == [("safe", True)]
    assert reads == [(path, "utf-8")]


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "invalid.yml"
    path.write_text("value: !unknown []")
    monkeypatch.setattr(sys, "argv", ["check-yaml", str(path)])
    assert checker.main() == 1


def test_empty_selection() -> None:
    assert checker.check([]) == 0


@pytest.mark.parametrize(
    ("docstring", "description"),
    [
        (
            checker.__doc__,
            "Validate YAML without executing constructors or accepting unknown tags.",
        ),
        ("  Custom YAML description.  \n", "Custom YAML description."),
        (None, ""),
        ("", ""),
        (" \n ", ""),
    ],
)
def test_help_description(
    docstring: str | None,
    description: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(checker, "__doc__", docstring)
    monkeypatch.setattr(sys, "argv", ["check-yaml", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        checker.main()
    assert exit_info.value.code == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == (
        "usage: check-yaml [-h] [files ...]\n\n"
        + (f"{description}\n\n" if description else "")
        + "positional arguments:\n"
        "  files\n\n"
        "options:\n"
        "  -h, --help  show this help message and exit\n"
    )


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
