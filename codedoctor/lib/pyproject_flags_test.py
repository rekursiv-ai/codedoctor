"""Tests for ``pyproject_flags``: TOML defaults under argparse flags."""

from pathlib import Path
from typing import Protocol, cast

import argparse

import pytest

from codedoctor.lib import pyproject_flags


class _Flags(Protocol):
    select: str
    min_digits: int
    exclude: list[str]


def _parse(argv: list[str], *, start: Path) -> tuple[_Flags, Path]:
    """Parse ``argv`` through ``_parser`` over the ``[tool.tool]`` defaults."""
    flags, root = pyproject_flags.parse_args(
        _parser(),
        tool="tool",
        argv=argv,
        start=start,
    )
    return cast(_Flags, flags), root


def _parser() -> argparse.ArgumentParser:
    """Build a parser with one flag of each shape the table can default."""
    parser = argparse.ArgumentParser(prog="tool")
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--select", default="")
    parser.add_argument("--min-digits", type=int, default=5)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--strict", action="store_true")
    return parser


def _pyproject(directory: Path, body: str) -> Path:
    """Write ``body`` as ``directory/pyproject.toml``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pyproject.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_table_supplies_every_default(tmp_path: Path) -> None:
    _pyproject(
        tmp_path,
        "[tool.tool]\n"
        'select = "a b"\n'
        "min-digits = 7\n"
        'exclude = ["x/**", "*_test.py = a"]\n'
        "strict = true\n",
    )

    flags, root = pyproject_flags.parse_args(
        _parser(),
        tool="tool",
        argv=["f.py"],
        start=tmp_path / "deep" / "er",
    )

    assert vars(flags) == {
        "paths": ["f.py"],
        "select": "a b",
        "min_digits": 7,
        "exclude": ["x/**", "*_test.py = a"],
        "strict": True,
    }
    assert root == tmp_path


def test_a_plain_flag_replaces_and_an_append_flag_adds(tmp_path: Path) -> None:
    _pyproject(tmp_path, '[tool.tool]\nselect = "a"\nexclude = ["x/**"]\n')

    flags, _ = _parse(["--select", "b", "--exclude", "y/**"], start=tmp_path)

    assert (flags.select, flags.exclude) == ("b", ["x/**", "y/**"])


def test_a_pyproject_without_the_table_is_passed_over(tmp_path: Path) -> None:
    """A nested package's own project file must not hide the outer settings."""
    _pyproject(tmp_path, '[tool.tool]\nselect = "outer"\n')
    _pyproject(tmp_path / "inner", '[project]\nname = "inner"\n')

    flags, root = _parse([], start=tmp_path / "inner")

    assert (flags.select, root) == ("outer", tmp_path)


def test_no_table_anywhere_keeps_the_parser_defaults(tmp_path: Path) -> None:
    flags, root = _parse([], start=tmp_path)

    assert (flags.select, flags.min_digits, root) == ("", 5, tmp_path)


@pytest.mark.parametrize(
    ("table", "fragment"),
    [
        ('selec = "a"\n', "unknown key(s) ['selec']"),
        ('select = ["a"]\n', "select: expected str, got list"),
        ('min-digits = "7"\n', "min-digits: expected int, got str"),
        ('exclude = "x"\n', "exclude: expected list, got str"),
    ],
)
def test_a_bad_table_is_a_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    table: str,
    fragment: str,
) -> None:
    """A typo fails loudly; ignoring it would silently run the defaults."""
    path = _pyproject(tmp_path, f"[tool.tool]\n{table}")

    with pytest.raises(SystemExit) as exit_info:
        pyproject_flags.parse_args(_parser(), tool="tool", argv=[], start=tmp_path)

    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert f"{path} [tool.tool]" in err
    assert fragment in err


@pytest.mark.parametrize(
    "body",
    ["[tool.tool\n", "[tool]\ntool = 3\n", "tool = 1\n"],
)
def test_an_unreadable_table_is_a_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    body: str,
) -> None:
    path = _pyproject(tmp_path, body)

    with pytest.raises(SystemExit) as exit_info:
        pyproject_flags.parse_args(_parser(), tool="tool", argv=[], start=tmp_path)

    assert exit_info.value.code == 2
    assert str(path) in capsys.readouterr().err


def test_a_bool_is_not_accepted_for_an_int_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``True`` is an ``int`` to ``isinstance``; the table must still say int."""
    _pyproject(tmp_path, "[tool.tool]\nmin-digits = true\n")

    with pytest.raises(SystemExit):
        _parse([], start=tmp_path)

    assert "min-digits: expected int, got bool" in capsys.readouterr().err


def test_an_unset_flag_takes_its_table_type_from_its_converter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def parse() -> argparse.Namespace:
        parser = _parser()
        parser.add_argument("--most", type=int)
        flags, _ = pyproject_flags.parse_args(
            parser,
            tool="tool",
            argv=[],
            start=tmp_path,
        )
        return flags

    _pyproject(tmp_path, "[tool.tool]\nmost = 4\n")
    assert vars(parse())["most"] == 4
    _pyproject(tmp_path, '[tool.tool]\nmost = "4"\n')
    with pytest.raises(SystemExit):
        parse()
    assert "most: expected int, got str" in capsys.readouterr().err


def test_an_unset_flag_takes_its_table_type_from_a_converter_function(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--workers`` converts with a function, not a type, and defaults to None."""

    def count(text: str) -> int:
        return int(text)

    def parse() -> argparse.Namespace:
        parser = _parser()
        parser.add_argument("--workers", type=count)
        flags, _ = pyproject_flags.parse_args(
            parser,
            tool="tool",
            argv=[],
            start=tmp_path,
        )
        return flags

    _pyproject(tmp_path, "[tool.tool]\nworkers = 3\n")
    assert vars(parse())["workers"] == 3
    _pyproject(tmp_path, '[tool.tool]\nworkers = "3"\n')
    with pytest.raises(SystemExit):
        parse()
    assert "workers: expected int, got str" in capsys.readouterr().err


def test_an_int_is_not_accepted_for_a_bool_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _pyproject(tmp_path, "[tool.tool]\nstrict = 1\n")

    with pytest.raises(SystemExit):
        _parse([], start=tmp_path)

    assert "strict: expected bool, got int" in capsys.readouterr().err


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
