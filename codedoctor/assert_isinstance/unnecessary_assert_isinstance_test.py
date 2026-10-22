"""Tests for the unnecessary ``assert isinstance`` check."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import argparse
import json
import subprocess
import sys
import tempfile

from treekle import json_loads

import pytest

from codedoctor.assert_isinstance import unnecessary_assert_isinstance
from codedoctor.assert_isinstance.unnecessary_assert_isinstance import Diagnostic


if TYPE_CHECKING:
    from collections.abc import Sequence

_SOURCE = """\
def f(x: int, y: object) -> int:
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert isinstance(y, (int, str))
    assert isinstance(
        x, int
    )
    return x + y
"""

_PROBE = "isinstance(x, int); "

_OVERLAP = """\
from typing import Protocol, runtime_checkable


@runtime_checkable
class Named(Protocol):
    name: str


class Thing:
    name: int = 0


def f(x: Thing, z: object) -> None:
    assert isinstance(x, Named)
    assert isinstance(z, Missing)
"""


def test_probe_prefixes_only_single_line_name_asserts() -> None:
    lines = unnecessary_assert_isinstance.probe(_SOURCE, checker="basedpyright").split(
        "\n",
    )
    assert lines[1] == f"    {_PROBE}assert isinstance(x, int)"
    assert lines[2] == ("    isinstance(y, int); assert isinstance(y, int)")
    assert lines[3:] == _SOURCE.split("\n")[3:]


def _tree(root: Path, *paths: str) -> None:
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).touch()


def test_cover_names_a_directory_only_when_every_source_in_it_is_chosen(
    tmp_path: Path,
) -> None:
    _tree(
        tmp_path,
        "top.py",
        "pkg/full/a.py",
        "pkg/full/b.pyi",
        "pkg/full/deep/c.py",
        "pkg/full/notes.txt",
        "pkg/full/data/blob.json",
        "pkg/part/a.py",
        "pkg/part/other.py",
    )
    chosen = ["top.py", "pkg/full/a.py", "pkg/full/b.pyi", "pkg/full/deep/c.py"]

    assert unnecessary_assert_isinstance.cover(
        [tmp_path / p for p in [*chosen, "pkg/part/a.py"]],
        root=tmp_path,
    ) == [tmp_path / "pkg/full", tmp_path / "pkg/part/a.py", tmp_path / "top.py"]


def test_cover_rises_to_just_below_the_root_and_never_to_it(tmp_path: Path) -> None:
    _tree(tmp_path, "pkg/a.py", "pkg/sub/b.py")
    files = [tmp_path / "pkg/a.py", tmp_path / "pkg/sub/b.py"]
    assert unnecessary_assert_isinstance.cover(files, root=tmp_path) == [
        tmp_path / "pkg",
    ]


@pytest.mark.parametrize(
    "extra",
    ["pkg/hidden/x.py", "pkg/link"],
    ids=["unchosen-nested-source", "symlink"],
)
def test_cover_keeps_files_when_a_directory_would_add_sources(
    tmp_path: Path,
    extra: str,
) -> None:
    _tree(tmp_path, "pkg/a.py")
    if extra == "pkg/link":
        (tmp_path / extra).symlink_to(tmp_path / "pkg/a.py")
    else:
        _tree(tmp_path, extra)
    assert unnecessary_assert_isinstance.cover(
        [tmp_path / "pkg/a.py"],
        root=tmp_path,
    ) == [tmp_path / "pkg/a.py"]


def test_main_hands_basedpyright_the_covering_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    _tree(tmp_path, "pkg/a.py", "pkg/b.py", "lone/c.py", "lone/d.py")
    seen: list[Sequence[Path]] = []

    def record(
        argv: Sequence[Path],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, '{"generalDiagnostics": []}', "")

    monkeypatch.setattr(subprocess, "run", record)

    assert _main(monkeypatch, "pkg/a.py", "pkg/b.py", "lone/c.py") == 0

    assert seen[0][3:] == [Path("lone/c.py"), Path("pkg")]


def test_a_run_that_analyzes_no_file_is_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(subprocess, "run", _silent_basedpyright)

    assert _main(monkeypatch, "m.py") == 0
    assert capsys.readouterr().out == "0 errors, 0 warnings, 0 notes\n"


def test_a_report_with_exit_0_is_still_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(subprocess, "run", _notes_only_basedpyright)

    assert _main(monkeypatch, "m.py") == 0
    assert capsys.readouterr().out.endswith("0 errors, 0 warnings, 1 notes\n")


def test_probed_mirrors_the_root_and_rewrites_only_probed_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "sub" / "m.py").write_text(_SOURCE)
    (root / "pkg" / "other.py").write_text(_SOURCE)
    (root / "plain.py").write_text("x = 1\n")

    files = [root / "pkg" / "sub" / "m.py", root / "plain.py"]
    with unnecessary_assert_isinstance.probed(
        root,
        files=files,
        checker="basedpyright",
    ) as overlay:
        assert (
            overlay / "pkg" / "sub" / "m.py"
        ).read_text() == unnecessary_assert_isinstance.probe(
            _SOURCE,
            checker="basedpyright",
        )
        assert not (overlay / "pkg").is_symlink()
        assert (overlay / "pkg" / "other.py").is_symlink()
        assert (overlay / "plain.py").is_symlink()
    assert (root / "pkg" / "sub" / "m.py").read_text() == _SOURCE
    assert not overlay.exists()


def test_diagnostics_map_the_overlay_back_onto_the_checkout(tmp_path: Path) -> None:
    """Probe verdicts land on the assert; other findings shift left past the probe."""
    root = tmp_path / "root"
    overlay = tmp_path / "overlay"
    root.mkdir()
    (root / "m.py").write_text(_SOURCE)
    outside = tmp_path / "typeshed.pyi"
    outside.write_text("x: int\n")
    width = len(_PROBE)
    report = {
        "generalDiagnostics": [
            _item(
                overlay / "m.py",
                line=1,
                character=4,
                rule="reportUnnecessaryIsInstance",
            ),
            _item(overlay / "m.py", line=1, character=4 + width, rule="reportX"),
            _item(
                overlay / "m.py",
                line=1,
                character=4 + 5,
                rule="reportUnsafeOverlap",
            ),
            _item(overlay / "m.py", line=7, character=11, rule=""),
            _item(outside, line=0, character=0, rule="reportY"),
        ],
    }

    found = unnecessary_assert_isinstance.diagnostics(
        json.dumps(report),
        overlay=overlay,
        root=root,
    )

    def expect(path: Path, line: int, column: int, rule: str) -> Diagnostic:
        return Diagnostic(
            path=path,
            line=line,
            column=column,
            severity="warning",
            message="m",
            rule=rule,
            redundant=rule == "reportUnnecessaryIsInstance",
        )

    assert found == [
        expect(root / "m.py", 2, 5, "reportUnnecessaryIsInstance"),
        expect(root / "m.py", 2, 5, "reportX"),
        expect(root / "m.py", 8, 12, ""),
        expect(outside, 1, 1, "reportY"),
    ]
    assert str(found[0]) == (
        f"  {root / 'm.py'}:2:5 - warning: m (reportUnnecessaryIsInstance)\n"
    )
    assert str(found[2]) == f"  {root / 'm.py'}:8:12 - warning: m\n"


def test_a_diagnostic_without_a_range_sits_at_the_file_start(tmp_path: Path) -> None:
    """Basedpyright omits ``range`` for a file-level finding, such as a cycle."""
    (tmp_path / "m.py").write_text(_SOURCE)
    item = _item(tmp_path / "m.py", line=0, character=0, rule="reportImportCycles")
    del item["range"]

    found = unnecessary_assert_isinstance.diagnostics(
        json.dumps({"generalDiagnostics": [item]}),
        overlay=tmp_path / "overlay",
        root=tmp_path,
    )

    assert found == [
        Diagnostic(
            path=tmp_path / "m.py",
            line=1,
            column=1,
            severity="warning",
            message="m",
            rule="reportImportCycles",
            redundant=False,
        ),
    ]


def test_a_multiline_message_indents_like_basedpyright() -> None:
    found = Diagnostic(
        path=Path("a.py"),
        line=1,
        column=1,
        severity="error",
        message="one\ntwo",
        rule="",
        redundant=False,
    )
    assert str(found) == "  a.py:1:1 - error: one\n    two\n"


def test_delete_lines_removes_only_asserts(tmp_path: Path) -> None:
    path = tmp_path / "m.py"
    path.write_text(_SOURCE)
    unnecessary_assert_isinstance.delete_lines(path, [2, 3])
    assert path.read_text() == _SOURCE.replace(
        "    assert isinstance(x, int)\n    assert isinstance(y, int)\n",
        "",
        1,
    )
    with pytest.raises(
        ValueError,
        match=r"m\.py:1 is not an isinstance assert: 'def f\(x: int, y: object\)",
    ):
        unnecessary_assert_isinstance.delete_lines(path, [1])


def test_a_commented_assert_is_probed_and_deleted_whole(tmp_path: Path) -> None:
    line = "    assert isinstance(x, int)  # why"
    assert (
        unnecessary_assert_isinstance.probe(line, checker="basedpyright")
        == f"    {_PROBE}assert isinstance(x, int)  # why"
    )
    path = tmp_path / "m.py"
    path.write_text(f"def f(x: int) -> None:\n{line}\n    print(x)\n")
    unnecessary_assert_isinstance.delete_lines(path, [2])
    assert path.read_text() == "def f(x: int) -> None:\n    print(x)\n"


def test_deleting_a_blocks_only_statement_leaves_pass(tmp_path: Path) -> None:
    path = tmp_path / "m.py"
    path.write_text("for x in (1,):\n    assert isinstance(x, int)\ny = 1\n")
    unnecessary_assert_isinstance.delete_lines(path, [2])
    assert path.read_text() == "for x in (1,):\n    pass\ny = 1\n"


_FINDING = "m.py:2: [unnecessary-assert-isinstance] unnecessary assert isinstance\n"
_SUMMARY = "0 errors, 0 warnings, 0 notes\n"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], (_SOURCE, "")),
        (
            ["--fix"],
            (
                _SOURCE.replace("    assert isinstance(x, int)\n", "", 1),
                "Deleted 1 unnecessary assert(s).\n",
            ),
        ),
    ],
    ids=["report", "fix"],
)
def test_an_unnecessary_assert_fails_and_fix_deletes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected: tuple[str, str],
) -> None:
    """``x: int`` is flagged at the assert's line; ``y: object`` is not."""
    after, stderr = expected
    _project(tmp_path, monkeypatch=monkeypatch)
    _replay(monkeypatch, flagged=True)

    assert _main(monkeypatch, *argv, "m.py") == 1

    assert capsys.readouterr() == (_FINDING + _SUMMARY, stderr)
    assert (tmp_path / "m.py").read_text() == after


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("  # house-ignore[unnecessary-assert-isinstance] -- Kept on purpose.", ""),
        (
            "  # house-ignore[digits] -- Other rule.",
            _FINDING,
        ),
    ],
    ids=["named", "other-rule"],
)
def test_a_house_ignore_marker_naming_the_rule_suppresses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    marker: str,
    expected: str,
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    source = _SOURCE.replace(
        "    assert isinstance(x, int)\n",
        f"    assert isinstance(x, int){marker}\n",
        1,
    )
    (tmp_path / "m.py").write_text(source)
    _replay(monkeypatch, flagged=True)

    assert _main(monkeypatch, "--fix", "m.py") == int(bool(expected))

    assert capsys.readouterr().out == expected + _SUMMARY
    if not expected:
        assert (tmp_path / "m.py").read_text() == source


def test_a_marker_that_suppresses_nothing_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "m.py").write_text(
        "y = 1  # house-ignore[unnecessary-assert-isinstance] -- Stale.\n",
    )
    _replay(monkeypatch, flagged=False)

    assert _main(monkeypatch, "m.py") == 1

    assert capsys.readouterr().out == (
        "m.py:1: [houselint] superfluous inline ignore: "
        "'unnecessary-assert-isinstance' reports nothing on this statement; "
        "remove it from house-ignore[...]\n" + _SUMMARY
    )


@pytest.mark.parametrize(
    ("entry", "skipped"),
    [
        ('"m.py = unnecessary-assert-isinstance"', True),
        ('"m.py"', True),
        ('"m.py = digits"', False),
    ],
    ids=["names-rule", "every-rule", "other-rule"],
)
def test_houselint_excludes_skip_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entry: str,
    skipped: bool,
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch, houselint=f"exclude = [{entry}]\n")
    _replay(monkeypatch, flagged=True)

    assert _main(monkeypatch, "m.py") == int(not skipped)

    assert capsys.readouterr().out == ("" if skipped else _FINDING) + _SUMMARY


_TY = Path(sys.executable).parent / "ty"
_NEEDED = "def f(x: object) -> int:\n    assert isinstance(x, int)\n    return x\n"


def test_ty_needs_only_the_asserts_whose_removal_it_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "need.py").write_text(_NEEDED)
    needed = (tmp_path / "need.py", 2)
    assert unnecessary_assert_isinstance.still_needed(
        {needed, (tmp_path / "m.py", 2)},
        root=tmp_path,
        checkers=[_TY],
    ) == {needed}


def test_ty_needs_an_assert_that_changes_a_diagnostic_on_an_already_flagged_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the assert turns ``tuple[int, str]`` into ``tuple[object, str]``."""
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "need.py").write_text(
        "def f(x: object, y: str) -> tuple[int, int]:\n"
        "    assert isinstance(x, int)\n"
        "    return x, y\n",
    )
    needed = (tmp_path / "need.py", 2)
    assert unnecessary_assert_isinstance.still_needed(
        {needed},
        root=tmp_path,
        checkers=[_TY],
    ) == {
        needed,
    }


@pytest.mark.parametrize("status", [2, 101])
def test_a_crashed_checker_raises_instead_of_clearing_every_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    ty = tmp_path / "bin" / "ty"
    ty.parent.mkdir()
    ty.write_text(f"#!/bin/sh\necho panic >&2; exit {status}\n")
    ty.chmod(0o755)
    with pytest.raises(unnecessary_assert_isinstance.CheckerCrashError) as raised:
        unnecessary_assert_isinstance.still_needed(
            {(tmp_path / "m.py", 2)},
            root=tmp_path,
            checkers=[ty],
        )
    assert (raised.value.returncode, raised.value.stderr) == (status, "panic\n")


def test_a_crashed_ty_fails_the_run_and_deletes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    _replay(monkeypatch, flagged=True)
    ty = tmp_path / "bin" / "ty"
    ty.parent.mkdir()
    ty.write_text("#!/bin/sh\necho panic >&2; exit 2\n")
    ty.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(ty.parent / "python"))

    assert _main(monkeypatch, "--fix", "m.py") == 2

    assert capsys.readouterr() == ("", "panic\n")
    assert (tmp_path / "m.py").read_text() == _SOURCE


@pytest.mark.cli_node
def test_the_probe_binds_no_name_the_source_may_use(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.basedpyright]\ntypeCheckingMode = "recommended"\n',
    )
    source = (
        "def f(x: object) -> str:\n"
        '    __probe__ = "ok"\n'
        "    assert isinstance(x, int)\n"
        "    return __probe__\n"
    )
    (tmp_path / "m.py").write_text(source)
    with unnecessary_assert_isinstance.probed(
        tmp_path,
        files=[tmp_path / "m.py"],
        checker="basedpyright",
    ) as overlay:
        done = subprocess.run(  # noqa: S603 -- no shell; fixed argv.
            [
                Path(sys.executable).parent / "basedpyright",
                "--threads=1",
                "--outputjson",
                "m.py",
            ],
            cwd=overlay,
            capture_output=True,
            text=True,
            check=False,
        )
        found = unnecessary_assert_isinstance.diagnostics(
            done.stdout,
            overlay=overlay,
            root=tmp_path,
        )
    assert found == []


def test_a_pyproject_without_a_tool_table_is_not_the_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    assert (
        unnecessary_assert_isinstance.project_root(
            tmp_path,
            tool="basedpyright",
        )
        == tmp_path.resolve()
    )


def test_checker_lines_maps_each_ty_diagnostic_to_its_absolute_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def f() -> int:\n    return ''\n")
    (tmp_path / "b.py").write_text("x: int = ''\n\n\ny: str = 1\n")
    monkeypatch.chdir(tmp_path.parent)
    found = unnecessary_assert_isinstance._checker_lines(
        _TY,
        tmp_path,
        files=[Path("pkg/a.py"), Path("b.py")],
    )
    assert {
        path: sorted(key.split(" ", 1)[0] for key in counts)
        for path, counts in found.items()
    } == {
        (tmp_path / "pkg" / "a.py"): ["2:12:"],
        (tmp_path / "b.py"): ["1:10:", "4:10:"],
    }


def test_an_assert_ty_needs_is_not_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Basedpyright's verdict is replayed; real ty still needs the narrowing."""
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "m.py").write_text(_NEEDED)
    _replay(monkeypatch, flagged=True)

    assert _main(monkeypatch, "--fix", "m.py") == 0

    assert capsys.readouterr().out == _SUMMARY
    assert (tmp_path / "m.py").read_text() == _NEEDED


def test_excluded_and_unlintable_sources_do_not_end_the_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch, houselint='exclude = ["a.py"]\n')
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "notes.txt").write_text("x\n")
    _replay(monkeypatch, flagged=True)

    assert _main(monkeypatch, "a.py", "notes.txt", "m.py") == 1

    assert capsys.readouterr().out == _FINDING + _SUMMARY


@pytest.mark.cli_node
def test_no_paths_checks_the_projects_loop_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "loop").mkdir()
    (tmp_path / "m.py").rename(tmp_path / "loop" / "m.py")
    monkeypatch.chdir(tmp_path / "loop")
    run = subprocess.run
    seen: list[Sequence[Path]] = []

    def record(
        argv: Sequence[Path],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert text
        seen.append(argv)
        return run(
            argv,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=check,
        )

    monkeypatch.setattr(subprocess, "run", record)

    assert _main(monkeypatch) == 1

    assert seen[0][-1] == Path("loop")
    assert capsys.readouterr().out == "loop/" + _FINDING + _SUMMARY


def test_help_shows_the_docstring_after_the_shell_preamble(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--help")
    out = capsys.readouterr().out
    assert "\nReport ``assert isinstance(x, T)`` statements no type" in out
    assert "\nPyright deliberately skips ``reportUnnecessaryIsInstance``" in out
    assert "\nExamples:\n" in out
    assert "exec uv" not in out


@pytest.mark.parametrize(
    "case",
    [
        ("warning", "0 errors, 1 warnings, 0 notes\n", 1),
        ("information", "0 errors, 0 warnings, 1 notes\n", 0),
    ],
)
def test_warnings_fail_and_notes_are_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: tuple[str, str, int],
) -> None:
    severity, summary, status = case
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "m.py").write_text("x = 1\n")
    item = _item(Path("m.py"), line=0, character=0, rule="reportX")
    item["severity"] = severity
    _replay(monkeypatch, flagged=False, extra=[item])

    assert _main(monkeypatch, "m.py") == status
    assert capsys.readouterr().out.endswith(summary)


@pytest.mark.cli_node
def test_check_reports_what_a_plain_run_would_at_checkout_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only the probe's verdict is new; its other findings are not."""
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "m.py").write_text("x = 1\n")
    (tmp_path / "overlap.py").write_text(_OVERLAP)

    assert _main(monkeypatch, "m.py", "overlap.py") == 1

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"  {tmp_path / 'overlap.py'}:15:26 - error:")
    assert lines[1:] == ["1 errors, 0 warnings, 0 notes"]


@pytest.mark.parametrize("status", [0, 1])
def test_it_fails_exactly_when_basedpyright_would(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "m.py").write_text("x = 1\n")
    error = _item(Path("m.py"), line=0, character=0, rule="reportX")
    error["severity"] = "error"
    _replay(monkeypatch, flagged=False, extra=[error] * status)

    assert _main(monkeypatch, "m.py") == status
    assert capsys.readouterr().out.endswith(f"{status} errors, 0 warnings, 0 notes\n")


@pytest.mark.parametrize("status", [2, 3])
def test_a_crashed_checker_passes_its_output_and_status_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    checker = tmp_path / "bin" / "basedpyright"
    checker.parent.mkdir()
    checker.write_text(f"#!/bin/sh\necho out; echo err >&2; exit {status}\n")
    checker.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(checker.parent / "python"))

    assert _main(monkeypatch, "m.py") == status
    assert capsys.readouterr() == ("out\n", "err\n")


@pytest.mark.parametrize(
    ("argv", "threads", "fix"),
    [([], None, False), (["--threads", "3", "--fix"], 3, True)],
)
def test_the_defaults_check_loop_and_leave_sizing_to_the_run(
    argv: list[str],
    threads: int | None,
    fix: bool,
) -> None:
    parser = argparse.ArgumentParser()
    unnecessary_assert_isinstance._add_arguments(parser)
    assert vars(parser.parse_args(argv)) == {
        "fix": fix,
        "threads": threads,
        "checker": "pyrefly",
        "typechecker": ["pyrefly"],
        "paths": [],
    }


def test_a_run_from_a_subdirectory_checks_the_project_it_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "bin").mkdir()
    monkeypatch.chdir(tmp_path / "bin")
    _replay(monkeypatch, flagged=True)
    assert _main(monkeypatch, "--fix", "..") == 1
    assert capsys.readouterr().out == _FINDING + _SUMMARY
    assert (tmp_path / "m.py").read_text() == _SOURCE.replace(
        "    assert isinstance(x, int)\n",
        "",
        1,
    )


def test_a_temp_dir_behind_a_symlink_maps_back_onto_the_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The checker reports the overlay's real path, as on macOS's ``/var``."""
    root = tmp_path / "root"
    root.mkdir()
    _project(root, monkeypatch=monkeypatch)
    (tmp_path / "private").mkdir()
    (tmp_path / "var").symlink_to(tmp_path / "private")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "var"))
    _replay(monkeypatch, flagged=True)

    assert _main(monkeypatch, "m.py") == 1

    assert capsys.readouterr().out == _FINDING + _SUMMARY


def test_the_project_root_is_the_nearest_basedpyright_config(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.basedpyright]\n")
    (tmp_path / "pkg" / "deep").mkdir(parents=True)
    (tmp_path / "pkg" / "pyproject.toml").write_text("[tool.ruff]\n")
    assert unnecessary_assert_isinstance.project_root(
        tmp_path / "pkg" / "deep",
        tool="basedpyright",
    ) == (tmp_path.resolve())
    bare = tmp_path / "pkg" / "deep"
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
    assert (
        unnecessary_assert_isinstance.project_root(
            bare,
            tool="basedpyright",
        )
        == bare.resolve()
    )


@pytest.mark.cli_node
def test_the_replayed_report_is_what_basedpyright_says(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canned report the ``main`` tests replay matches a real run."""
    _project(tmp_path, monkeypatch=monkeypatch)
    with unnecessary_assert_isinstance.probed(
        tmp_path,
        files=[tmp_path / "m.py"],
        checker="basedpyright",
    ) as overlay:
        done = subprocess.run(  # noqa: S603 -- no shell; fixed argv.
            [
                Path(sys.executable).parent / "basedpyright",
                "--threads=1",
                "--outputjson",
                "m.py",
            ],
            cwd=overlay,
            capture_output=True,
            text=True,
            check=False,
        )
        real = json_loads(done.stdout, dict[str, object])["generalDiagnostics"]
        mapped = unnecessary_assert_isinstance.diagnostics(
            done.stdout,
            overlay=overlay,
            root=tmp_path,
        )
        replayed = unnecessary_assert_isinstance.diagnostics(
            json.dumps({"generalDiagnostics": _found(overlay, flagged=True)}),
            overlay=overlay,
            root=tmp_path,
        )
    assert real == [*_found(overlay, flagged=True), *_unused_probes(overlay)]
    assert mapped == replayed


def _pyrefly_project(root: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    """Make ``root`` the cwd: a pyrefly project holding ``m.py``."""
    (root / "pyproject.toml").write_text("[tool.pyrefly]\n")
    (root / "m.py").write_text(source)
    monkeypatch.chdir(root)
    monkeypatch.setenv("WORKERS_DEFAULT", "1")


def test_the_pyrefly_probe_reveals_the_asserted_name() -> None:
    assert (
        unnecessary_assert_isinstance.probe(
            "    assert isinstance(x, int)  # why",
            checker="pyrefly",
        )
        == "    isinstance(x, int) or reveal_type(x); assert isinstance(x, int)  # why"
    )


@pytest.mark.parametrize("fix", [False, True], ids=["report", "fix"])
def test_pyrefly_flags_only_the_redundant_assert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fix: bool,
) -> None:
    """Real pyrefly: ``x: int`` is redundant, ``y: object`` narrows."""
    source = (
        "def f(x: int, y: object) -> int:\n"
        "    assert isinstance(x, int)\n"
        "    assert isinstance(y, int)\n"
        "    return x + y\n"
    )
    _pyrefly_project(tmp_path, monkeypatch, source)
    argv = ["check.py", "--threads", "1", *(["--fix"] * fix), "m.py"]
    monkeypatch.setattr(sys, "argv", argv)

    assert unnecessary_assert_isinstance.main() == 1

    assert capsys.readouterr().out == _FINDING + _SUMMARY
    expected = source.replace("    assert isinstance(x, int)\n", "") if fix else source
    assert (tmp_path / "m.py").read_text() == expected


_FLAGS_NARROWED = """\
import argparse
from typing import Protocol, runtime_checkable


@runtime_checkable
class _Flags(Protocol):
    seed: int


def f() -> int:
    flags = argparse.ArgumentParser().parse_args([])
    assert isinstance(flags, _Flags)
    return flags.seed
"""


@pytest.mark.parametrize(
    ("typechecker", "kept"),
    [
        ([], False),
        pytest.param(
            ["--typechecker", "basedpyright"],
            True,
            marks=pytest.mark.cli_node,
        ),
    ],
    ids=["default", "typechecker-basedpyright"],
)
def test_typechecker_basedpyright_keeps_an_assert_it_still_narrows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    typechecker: list[str],
    kept: bool,
) -> None:
    """Pyrefly calls it redundant; basedpyright reads ``Namespace`` fields as ``Any``."""
    _pyrefly_project(tmp_path, monkeypatch, _FLAGS_NARROWED)
    with (tmp_path / "pyproject.toml").open("a") as config:
        config.write('[tool.basedpyright]\ntypeCheckingMode = "all"\n')
    argv = ["check.py", "--threads", "1", *typechecker, "--fix", "m.py"]
    monkeypatch.setattr(sys, "argv", argv)

    assert unnecessary_assert_isinstance.main() == int(not kept)

    removed = _FLAGS_NARROWED.replace("    assert isinstance(flags, _Flags)\n", "")
    assert (tmp_path / "m.py").read_text() == (_FLAGS_NARROWED if kept else removed)
    assert capsys.readouterr().out.endswith(_SUMMARY)


def test_pyrefly_drops_the_probes_own_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The unimported ``reveal_type`` and a non-``Never`` reveal are not reported."""
    _pyrefly_project(
        tmp_path,
        monkeypatch,
        "def f(y: object) -> int:\n    assert isinstance(y, int)\n    return y\n",
    )
    monkeypatch.setattr(sys, "argv", ["check.py", "--threads", "1", "m.py"])

    assert unnecessary_assert_isinstance.main() == 0
    assert capsys.readouterr().out == _SUMMARY


def test_pyrefly_errors_fail_the_run_at_checkout_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A finding after the probe on the assert's line shifts back past it."""
    _pyrefly_project(
        tmp_path,
        monkeypatch,
        "def f(y: object) -> int:\n    assert isinstance(y, str)\n    return y\n",
    )
    monkeypatch.setattr(sys, "argv", ["check.py", "--threads", "1", "m.py"])

    assert unnecessary_assert_isinstance.main() == 1

    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"  {tmp_path.resolve() / 'm.py'}:3:12 - error: ")
    assert out[0].endswith("(bad-return)")
    assert out[1:] == ["1 errors, 0 warnings, 0 notes"]


def test_a_pyrefly_run_that_prints_no_report_passes_its_failure_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pyrefly exits 1 with no JSON when a path is missing."""
    _pyrefly_project(tmp_path, monkeypatch, "x = 1\n")
    checker = tmp_path / "bin" / "pyrefly"
    checker.parent.mkdir()
    checker.write_text("#!/bin/sh\necho missing >&2; exit 1\n")
    checker.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(checker.parent / "python"))
    monkeypatch.setattr(sys, "argv", ["check.py", "--threads", "1", "m.py"])

    assert unnecessary_assert_isinstance.main() == 1
    assert capsys.readouterr() == ("", "missing\n")


def test_pyrefly_diagnostics_map_severities_and_shift_past_the_probe(
    tmp_path: Path,
) -> None:
    (tmp_path / "m.py").write_text(
        "def f(x: int) -> None:\n    assert isinstance(x, int)\n",
    )
    width = len("isinstance(x, int) or reveal_type(x); ")

    def item(column: int, name: str, severity: str, line: int = 1) -> dict[str, object]:
        return {
            "path": "m.py",
            "line": line,
            "column": column,
            "name": name,
            "description": "revealed type: Never" if name == "reveal-type" else "d",
            "severity": severity,
        }

    report = {
        "errors": [
            item(1, "a", "error"),
            item(1, "b", "warn"),
            item(1, "c", "info"),
            item(5 + width, "d", "error", line=2),
            item(5 + 2, "reveal-type", "info", line=2),
            item(5 + 1, "unimported-directive", "error", line=2),
        ],
    }
    found = unnecessary_assert_isinstance.pyrefly_diagnostics(
        json.dumps(report),
        overlay=tmp_path,
        root=tmp_path,
    )
    assert [(d.line, d.column, d.severity, d.redundant) for d in found] == [
        (1, 1, "error", False),
        (1, 1, "warning", False),
        (1, 1, "information", False),
        (2, 5, "error", False),
        (2, 5, "information", True),
    ]


def test_pyrefly_runs_with_info_reports_and_no_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below ``info`` the probe's ``reveal-type`` verdict is never reported."""
    _pyrefly_project(tmp_path, monkeypatch, "x = 1\n")
    seen: list[Sequence[Path | str]] = []

    def record(
        argv: Sequence[Path | str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, '{"errors": []}', "")

    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr(sys, "argv", ["check.py", "--threads", "1", "m.py"])

    assert unnecessary_assert_isinstance.main() == 0
    assert seen[0][1:] == [
        "check",
        "--threads=1",
        "--output-format=json",
        "--min-severity=info",
        "--summary=none",
        Path("m.py"),
    ]


def test_the_project_root_follows_the_checker(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pyrefly]\n")
    (tmp_path / "pkg").mkdir()
    assert (
        unnecessary_assert_isinstance.project_root(
            tmp_path / "pkg",
            tool="pyrefly",
        )
        == tmp_path.resolve()
    )
    assert (
        unnecessary_assert_isinstance.project_root(
            tmp_path / "pkg",
            tool="basedpyright",
        )
        == (tmp_path / "pkg").resolve()
    )


def _found(overlay: Path, *, flagged: bool) -> list[dict[str, object]]:
    """Return basedpyright's probe verdicts for ``_SOURCE`` probed in ``overlay``."""
    message = 'Unnecessary isinstance call; "int" is always an instance of "int"'
    return [_at_probe(overlay, 1, message, "reportUnnecessaryIsInstance")] * flagged


def _unused_probes(overlay: Path) -> list[dict[str, object]]:
    """Return what each bare probe in ``_SOURCE`` also draws; ``diagnostics`` drops it."""
    message = (
        'Result of call expression is of type "bool" and is not used; '
        'assign to variable "_" if this is intentional'
    )
    return [
        _at_probe(overlay, line, message, "reportUnusedCallResult") for line in (1, 2)
    ]


def _at_probe(overlay: Path, line: int, message: str, rule: str) -> dict[str, object]:
    """Return one diagnostic spanning the probe on 0-based ``line`` of ``m.py``."""
    return {
        "file": str(overlay / "m.py"),
        "severity": "warning",
        "message": message,
        "range": {
            "start": {"line": line, "character": 4},
            "end": {"line": line, "character": 22},
        },
        "rule": rule,
    }


_CLEAN_REPORTS: Final = {"pyrefly": '{"errors": []}'}
"""What a checker the test does not fake prints for ``_SOURCE``: nothing."""


# Only ``test_the_replayed_report_is_what_basedpyright_says`` and
# ``test_check_reports_what_a_plain_run_would_at_checkout_columns`` pay it.
def _replay(
    monkeypatch: pytest.MonkeyPatch,
    *,
    flagged: bool,
    extra: Sequence[dict[str, object]] = (),
) -> None:
    """Answer the basedpyright run with ``_found``; node startup costs ~0.6s."""
    run = subprocess.run

    def replay(
        argv: Sequence[Path],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert text
        # A failing check is a report, never an exception.
        assert check is False
        # Pyrefly finds nothing in ``_SOURCE``; ty runs, as tests fake it.
        if argv[0].name in _CLEAN_REPORTS:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=_CLEAN_REPORTS[argv[0].name],
                stderr="",
            )
        if argv[0].name != "basedpyright":
            return run(
                argv,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                check=check,
            )
        # Basedpyright names files under its real cwd: ``/var/folders`` on macOS
        # is reported as ``/private/var/folders``.
        real = cwd.resolve()
        items = [
            *_found(real, flagged=flagged),
            *({**item, "file": str(real / str(item["file"]))} for item in extra),
        ]
        return subprocess.CompletedProcess(
            argv,
            int(bool(items)),
            stdout=json.dumps({"generalDiagnostics": items}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", replay)


def _item(path: Path, *, line: int, character: int, rule: str) -> dict[str, object]:
    """Return one ``--outputjson`` diagnostic."""
    item: dict[str, object] = {
        "file": str(path),
        "severity": "warning",
        "message": "m",
        "range": {"start": {"line": line, "character": character}},
    }
    if rule:
        item["rule"] = rule
    return item


def _silent_basedpyright(
    argv: Sequence[Path],
    *,
    cwd: Path,
    capture_output: bool,
    text: bool,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Answer as basedpyright 1.40 does above one thread when it analyzes no file."""
    del cwd, capture_output, text, check
    return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")


def _notes_only_basedpyright(
    argv: Sequence[Path],
    *,
    cwd: Path,
    capture_output: bool,
    text: bool,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Answer with one note, which leaves basedpyright's exit code at 0."""
    del capture_output, text, check
    note = _item(cwd.resolve() / "m.py", line=0, character=4, rule="")
    report = {"generalDiagnostics": [{**note, "severity": "information"}]}
    return subprocess.CompletedProcess(
        argv,
        returncode=0,
        stdout=json.dumps(report),
        stderr="",
    )


def _project(
    root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    houselint: str = "",
) -> None:
    """Make ``root`` the cwd: a basedpyright project holding ``m.py``."""
    (root / "pyproject.toml").write_text(
        '[tool.basedpyright]\ntypeCheckingMode = "recommended"\n'
        + (f"[tool.houselint]\n{houselint}" if houselint else ""),
    )
    (root / "m.py").write_text(_SOURCE)
    monkeypatch.chdir(root)


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Run ``check.main`` with ``argv`` on one checker thread."""
    # The ``--threads`` default is computed even when overridden, and on macOS
    # it asks ``sysctl`` through ``subprocess.run``, which tests fake.
    monkeypatch.setenv("WORKERS_DEFAULT", "1")
    argv0 = ["check.py", "--threads", "1", "--checker", "basedpyright"]
    monkeypatch.setattr(sys, "argv", [*argv0, *argv])
    return unnecessary_assert_isinstance.main()


def test_a_bare_run_checks_loop_at_the_machines_thread_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unpinned ``--threads`` is sized when the run starts; no paths means ``loop``."""
    _project(tmp_path, monkeypatch=monkeypatch)
    (tmp_path / "loop").mkdir()
    (tmp_path / "m.py").rename(tmp_path / "loop" / "m.py")
    monkeypatch.setattr(unnecessary_assert_isinstance, "default_workers", lambda: 6)
    seen: list[list[str]] = []

    def record(
        argv: Sequence[Path],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        seen.append([str(arg) for arg in argv])
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr(sys, "argv", ["check.py", "--checker", "basedpyright"])

    assert unnecessary_assert_isinstance.main() == 0
    assert [Path(seen[0][0]).name, *seen[0][1:]] == [
        "basedpyright",
        "--threads=6",
        "--outputjson",
        "loop",
    ]


def test_parsing_never_sizes_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every gate passes ``--threads``; probing the machine for a default is waste."""
    sized: list[int] = []
    monkeypatch.setattr(
        unnecessary_assert_isinstance,
        "default_workers",
        lambda: sized.append(1) or 1,
    )
    parser = argparse.ArgumentParser()
    unnecessary_assert_isinstance._add_arguments(parser)

    parser.parse_args(["--threads", "2"])

    assert sized == []


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
