"""Report ``assert isinstance(x, T)`` statements no type checker needs; ``--fix`` deletes them.

Pyright deliberately skips ``reportUnnecessaryIsInstance`` inside an ``assert``,
so a narrowing assert that became unnecessary (its operand is already ``T`` after
a declaration fix upstream) is invisible to the gate and lingers as noise. This
tool makes the checker score those sites. Each single-line
``assert isinstance(NAME, DOTTED_NAME)`` gains a probe on its own line,
``isinstance(...); assert isinstance(...)``, in a symlink overlay of
the working directory; ONE basedpyright run over the overlay then reports every
diagnostic a plain run would, plus each unnecessary assert, and the assert still
narrows for everything after it. That run replaces the plain one in pre-commit.

``--checker pyrefly`` (the default) runs pyrefly instead. Pyrefly has no
unnecessary-isinstance rule, so its probe is ``isinstance(...) or
reveal_type(NAME); ``: the revealed type is ``Never`` exactly when the assert
is redundant.

An assert the checker calls unnecessary may still narrow for another gate
checker, so it is only unnecessary when ty and each ``--typechecker`` other
than ``--checker`` (default: pyrefly), run once over copies with every
candidate replaced by ``pass``, report nothing new on those files. Those runs
happen only when the checker flagged something, and a crash fails the run.

Findings follow houselint's contract: each is rule ``unnecessary-assert-isinstance``,
a ``house-ignore`` marker naming it suppresses it (and is reported once it
suppresses nothing), and ``[tool.houselint] exclude`` entries naming it or every
rule skip the file. Exits 1 on any diagnostic, as basedpyright does; with
``--fix`` the unnecessary asserts are deleted, and the run still fails so the
fix gets reviewed.

Examples:
  unnecessary_assert_isinstance a.py b.py   # report
  unnecessary_assert_isinstance --fix loop  # delete them

"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, override

import argparse
import ast
import contextlib
import dataclasses
import os
import re
import subprocess
import sys
import tempfile
import tomllib

from treekle import from_plain, json_loads

from codedoctor.lib import pyproject_flags
from codedoctor.lib.files import walk
from codedoctor.lib.worker_count import default_workers, worker_count
from codedoctor.lint import house_ignore, houselint
from codedoctor.lint.houselint_types import Finding, Unit


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


_SOURCE_SUFFIXES = frozenset({".py", ".pyi"})
# A trailing comment is allowed: it is where a ``house-ignore`` marker goes.
_ASSERT = re.compile(
    r"(?P<indent>[ \t]*)assert isinstance\((?P<args>(?P<name>\w+), [A-Za-z_][\w.]*)\)"
    r"[ \t]*(?:#.*)?",
)


def main() -> int:
    """Run the program.

    Returns:
      exit_code: 1 when a diagnostic or an unnecessary assert remains or was fixed.

    """
    # pragma: no mutate start -- the module docstring is never None.
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # pragma: no mutate end
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    checker = flags.checker
    threads = flags.threads or default_workers()
    root = project_root(Path.cwd(), tool=checker)
    paths = [p.resolve() for p in flags.paths] or [root / "loop"]
    sources = walk.expand(paths, suffixes={".py"})
    targets = [
        *(p for p in paths if p.is_dir()),
        *cover([p for p in paths if not p.is_dir()], root=root),
    ]
    options = (
        [
            "check",
            f"--threads={threads}",
            "--output-format=json",
            # Info, so the probe's ``reveal-type`` verdict is reported.
            "--min-severity=info",
            "--summary=none",
        ]
        if checker == "pyrefly"
        else [f"--threads={threads}", "--outputjson"]
    )
    with probed(root, files=sources, checker=checker) as overlay:
        result = subprocess.run(  # noqa: S603 -- no shell; fixed argv.
            [
                Path(sys.executable).parent / checker,
                *options,
                # Relative, so the checker resolves them inside the overlay.
                *(path.relative_to(root) for path in targets),
            ],
            cwd=overlay,
            capture_output=True,
            text=True,
            check=False,
        )
    sys.stderr.write(result.stderr)
    # 0 clean, 1 diagnostics; anything else is a crash with no JSON report, as is
    # pyrefly's exit 1 on a missing path, which prints no report at all.
    if result.returncode > 1 or (
        checker == "pyrefly" and result.returncode and not result.stdout
    ):
        sys.stdout.write(result.stdout)
        return result.returncode
    parse = pyrefly_diagnostics if checker == "pyrefly" else diagnostics
    # Above one thread, basedpyright prints no report at all, not an empty one, when
    # it analyzes no file: every target excluded or under a hidden directory.
    found = (
        parse(result.stdout, overlay=overlay, root=root)
        if result.stdout or result.returncode
        else []
    )
    rule = "unnecessary-assert-isinstance"
    try:
        needed = still_needed(
            {(d.path, d.line) for d in found if d.redundant},
            root=root,
            checkers=[
                Path(sys.executable).parent / name
                for name in ("ty", *flags.typechecker)
                if name != checker
            ],
        )
    except CheckerCrashError as crash:
        sys.stderr.write(crash.stderr)
        return crash.returncode
    candidates = [
        Finding(
            path=d.path,
            line=d.line,
            rule=rule,
            message="unnecessary assert isinstance",
        )
        for d in found
        if d.redundant and (d.path, d.line) not in needed
    ]
    findings = _honor_house_ignore(candidates, sources=sources, rule=rule)
    hits: dict[Path, list[int]] = {}
    for finding in findings:
        if finding.rule == rule:
            hits.setdefault(finding.path, []).append(finding.line)
    # Repo-relative, as houselint prints a finding.
    sys.stdout.writelines(
        f"{dataclasses.replace(f, path=f.path.absolute().relative_to(root))}\n"
        for f in findings
    )
    if flags.fix:
        for path, lines in hits.items():
            delete_lines(path, lines)
    count = len(findings)
    if hits and flags.fix:
        sys.stderr.write(
            f"Deleted {sum(map(len, hits.values()))} unnecessary assert(s).\n",
        )
    rest = [d for d in found if not d.redundant]
    counts = Counter(d.severity for d in rest)
    sys.stdout.writelines(map(str, rest))
    sys.stdout.write(
        f"{counts['error']} errors, {counts['warning']} warnings, "
        f"{counts['information']} notes\n",
    )
    # A deletion fails the run like any auto-fix, so the fixed file is reviewed.
    return int(count > 0 or counts["error"] + counts["warning"] > 0)


class Flags(Protocol):
    """Parsed command-line flags."""

    fix: bool
    threads: int | None
    checker: str
    typechecker: list[str]
    paths: list[Path]


class CheckerCrashError(RuntimeError):
    """A checker exited without a report, so its result is unknown."""

    def __init__(self, returncode: int, stderr: str) -> None:
        """Record the checker's exit status and error output."""
        super().__init__(f"checker exited {returncode}")
        self.returncode = returncode
        self.stderr = stderr


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Diagnostic:
    """One checker diagnostic, located in the checkout rather than the overlay.

    Attributes:
      path: File the diagnostic is in.
      line: 1-based line.
      column: 1-based column.
      severity: ``error``, ``warning``, or ``information``.
      message: The checker's message.
      rule: The rule that fired, or empty.
      redundant: Whether this is a probe proving its assert redundant.

    """

    path: Path
    line: int
    column: int
    severity: str
    message: str
    rule: str
    redundant: bool

    @override
    def __str__(self) -> str:
        """Render as basedpyright's text report renders one diagnostic."""
        rule = f" ({self.rule})" if self.rule else ""
        message = self.message.replace("\n", "\n    ")
        return (
            f"  {self.path}:{self.line}:{self.column} - "
            f"{self.severity}: {message}{rule}\n"
        )


def project_root(start: Path, *, tool: str) -> Path:
    """Return the checker project ``start`` lies in.

    Args:
      start: Directory the tool runs from.
      tool: The checker, as its ``[tool.<name>]`` table names it.

    Returns:
      root: Nearest directory at or above ``start`` whose ``pyproject.toml``
        has ``[tool.<tool>]``; ``start`` itself when none does.

    """
    start = start.resolve()
    for directory in (start, *start.parents):
        path = directory / "pyproject.toml"
        if path.is_file() and tool in from_plain(
            tomllib.loads(path.read_text()).get("tool"),
            dict[str, object],
            default={},
        ):
            return directory
    return start


def cover(files: Sequence[Path], *, root: Path) -> list[Path]:
    """Return the fewest paths under ``root`` that name exactly ``files``.

    Basedpyright makes each named path an ``include`` entry and matches every
    import's source file against every entry, so naming 6323 files took a
    `transformers` importer from 3s to 47s. A directory replaces its files only
    when it holds no other ``.py``/``.pyi`` file and no symlink, so the checked
    set never grows.

    Args:
      files: Absolute files to check.
      root: Directory every returned path lies strictly below, unless a file
        sits directly in it.

    Returns:
      targets: Each file's highest ancestor below ``root`` whose every source
        is in ``files``, or the file itself; sorted, deduplicated.

    """
    chosen = {f.absolute() for f in files}
    holding = {parent for path in chosen for parent in path.parents}
    full: dict[Path, bool] = {}

    def is_full(directory: Path) -> bool:
        if directory not in full:
            full[directory] = all(
                _covered(child, chosen=chosen, holding=holding, is_full=is_full)
                for child in directory.iterdir()
            )
        return full[directory]

    targets: set[Path] = set()
    for path in chosen:
        top = path
        for parent in path.parents:
            # Never ``root`` itself: it holds ``.git`` and ``.venv``.
            if parent == root or not parent.is_relative_to(root) or not is_full(parent):
                break
            top = parent
        targets.add(top)
    return sorted(targets)


def probe(text: str, *, checker: str) -> str:
    """Prefix every single-line ``assert isinstance`` in ``text`` with a probe.

    Args:
      text: Python source.
      checker: ``basedpyright`` or ``pyrefly``; picks the probe's form.

    Returns:
      probed: ``text`` with the checker's probe before each matched assert, on
        the assert's own line.

    """
    return "\n".join(
        line
        if (match := _ASSERT.fullmatch(line)) is None
        else f"{match['indent']}{_probe(match, checker=checker)}{line.lstrip()}"
        for line in text.split("\n")
    )


@contextlib.contextmanager
def probed(root: Path, *, files: Sequence[Path], checker: str) -> Generator[Path]:
    """Yield a symlink mirror of ``root`` whose ``files`` carry probes.

    Args:
      root: Directory to mirror; the checker reads its configuration from here.
      files: Files to probe, relative to the working directory or absolute.
      checker: ``basedpyright`` or ``pyrefly``; picks the probe's form.

    Yields:
      overlay: Temporary directory; every entry links into ``root`` except the
        probed files and the directories leading to them.

    """
    with tempfile.TemporaryDirectory(prefix="unnecessary-assert-") as temp:
        # Real, as basedpyright reports it: otherwise ``diagnostics`` cannot map a
        # file under macOS's ``/var`` (``/private/var``) back onto the checkout.
        overlay = Path(temp).resolve()
        _link_children(root, into=overlay)
        for path in files:
            text = path.read_text()
            probes = probe(text, checker=checker)
            if probes == text:
                continue
            _unlink_to(overlay, path.absolute().relative_to(root)).write_text(probes)
        yield overlay


def diagnostics(report: str, *, overlay: Path, root: Path) -> list[Diagnostic]:
    """Map basedpyright's JSON ``report`` on ``overlay`` back onto ``root``.

    Pyright skips more than ``reportUnnecessaryIsInstance`` inside an
    ``assert`` -- an unsafe protocol overlap, for one -- so on a probe only the
    redundancy verdict is kept. A diagnostic on the assert after it moves left
    by the probe's width.

    Args:
      report: ``basedpyright --outputjson`` output from a run over ``overlay``.
      overlay: Directory the run checked.
      root: Directory ``overlay`` mirrors.

    Returns:
      found: The diagnostics a run over ``root`` would report, plus one
        ``redundant`` diagnostic at each assert its probe proved redundant.

    """
    found: list[Diagnostic] = []
    for item in from_plain(
        json_loads(report, dict[str, object]).get(
            "generalDiagnostics",
        ),
        list[dict[str, object]],
    ):
        checked = Path(from_plain(item.get("file"), str))
        path = (
            root / checked.relative_to(overlay)
            if checked.is_relative_to(overlay)
            else checked
        )
        # A file-level finding (an import cycle) carries no ``range``.
        where = from_plain(item.get("range"), dict[str, object], default={})
        start = from_plain(where.get("start"), dict[str, object], default={})
        line = from_plain(start.get("line"), int, default=0) + 1
        column = from_plain(start.get("character"), int, default=0)
        rule = from_plain(item.get("rule"), str, default="")
        probed_at = _probe_span(path, line=line, checker="basedpyright")
        redundant = column in probed_at
        if redundant and rule != "reportUnnecessaryIsInstance":
            continue
        found.append(
            Diagnostic(
                path=path,
                line=line,
                column=1 + (probed_at.start if redundant else column - len(probed_at)),
                severity=from_plain(item.get("severity"), str),
                message=from_plain(item.get("message"), str),
                rule=rule,
                redundant=redundant,
            ),
        )
    return found


def pyrefly_diagnostics(report: str, *, overlay: Path, root: Path) -> list[Diagnostic]:
    """Map pyrefly's JSON ``report`` on ``overlay`` back onto ``root``.

    On a probe only a ``reveal-type`` of ``Never`` is kept, as the redundancy
    verdict; the probe's other findings (the unimported ``reveal_type``, any
    revealed type but ``Never``) are dropped. A diagnostic on the assert after
    it moves left by the probe's width.

    Args:
      report: ``pyrefly check --output-format=json`` output, run in ``overlay``.
      overlay: Directory the run checked, and its working directory.
      root: Directory ``overlay`` mirrors.

    Returns:
      found: The diagnostics a run over ``root`` would report, plus one
        ``redundant`` diagnostic at each assert its probe proved redundant.

    """
    severities = {"error": "error", "warn": "warning", "info": "information"}
    found: list[Diagnostic] = []
    for item in from_plain(
        json_loads(report, dict[str, object]).get("errors"),
        list[dict[str, object]],
    ):
        # Relative to the working directory; ``normpath`` keeps a symlinked
        # overlay entry inside the overlay, where ``resolve`` would leave it.
        checked = Path(os.path.normpath(overlay / from_plain(item.get("path"), str)))
        path = (
            root / checked.relative_to(overlay)
            if checked.is_relative_to(overlay)
            else checked
        )
        line = from_plain(item.get("line"), int)
        column = from_plain(item.get("column"), int) - 1
        rule = from_plain(item.get("name"), str)
        message = from_plain(item.get("description"), str)
        probed_at = _probe_span(path, line=line, checker="pyrefly")
        redundant = column in probed_at
        if redundant and (rule, message) != ("reveal-type", "revealed type: Never"):
            continue
        found.append(
            Diagnostic(
                path=path,
                line=line,
                column=1 + (probed_at.start if redundant else column - len(probed_at)),
                severity=severities[from_plain(item.get("severity"), str)],
                message=message,
                rule=rule,
                redundant=redundant,
            ),
        )
    return found


def still_needed(
    sites: set[tuple[Path, int]],
    *,
    root: Path,
    checkers: Sequence[Path],
) -> set[tuple[Path, int]]:
    """Return the ``sites`` whose assert some of ``checkers`` may still need.

    Every candidate becomes ``pass`` in a copy of its file under one overlay,
    so one run per checker answers for all of them; a file where a checker
    then reports a diagnostic it did not before keeps every candidate it holds.

    Args:
      sites: ``(file, 1-based line)`` asserts the probing checker calls redundant.
      root: Directory the checkers read their configuration from.
      checkers: The ``ty``, ``basedpyright`` or ``pyrefly`` executables to ask.

    Returns:
      needed: The subset any checker reports a new diagnostic for once removed.

    Raises:
      CheckerCrashError: A checker crashed, so every candidate is kept.

    """
    if not sites:
        return set()
    files = sorted({path for path, _ in sites})
    before = {c: _checker_lines(c, root, files=files) for c in checkers}
    with tempfile.TemporaryDirectory(prefix="unnecessary-assert-typecheck-") as temp:
        # Real, as basedpyright reports it (macOS's ``/var`` is ``/private/var``).
        overlay = Path(temp).resolve()
        _link_children(root, into=overlay)
        for path in files:
            target = _unlink_to(overlay, path.absolute().relative_to(root))
            source = path.read_text().split("\n")
            for site_path, line in sites:
                if site_path == path:
                    match = _ASSERT.fullmatch(source[line - 1])
                    if match is None:
                        raise ValueError("Expected match is not None.")
                    source[line - 1] = f"{match['indent']}pass"
            target.write_text("\n".join(source))
        copies = [overlay / p.absolute().relative_to(root) for p in files]
        after = {c: _checker_lines(c, overlay, files=copies) for c in checkers}
    # Which assert a new diagnostic needed is not recoverable from its line
    # (two asserts can narrow one return), so a file with any keeps them all.
    # Whole diagnostics are compared, not lines: removing an assert can change
    # the message on a line that already had one.
    changed = {
        path
        for path in files
        for c in checkers
        if after[c].get(overlay / path.absolute().relative_to(root), Counter())
        - before[c].get(path.absolute(), Counter())
    }
    return {site for site in sites if site[0] in changed}


def delete_lines(path: Path, lines: Sequence[int]) -> None:
    """Delete asserts from ``path`` in place.

    Args:
      path: File to edit.
      lines: 1-based line numbers, each holding a single-line isinstance assert.

    """
    source = path.read_text().split("\n")
    for line in sorted(set(lines), reverse=True):
        if not _ASSERT.fullmatch(source[line - 1]):
            raise ValueError(
                f"{path}:{line} is not an isinstance assert: {source[line - 1]!r}",
            )
        indent = source[line - 1][: -len(source[line - 1].lstrip())]
        del source[line - 1]
        # An assert that was its block's only statement leaves the block empty.
        with contextlib.suppress(SyntaxError):
            ast.parse("\n".join(source))
            continue
        source.insert(line - 1, f"{indent}pass")
    path.write_text("\n".join(source))


# Read through houselint's own loader, so the excludes mean here what they mean
# to every houselint rule: an entry naming this rule, or naming none, skips it.
def _honor_house_ignore(
    candidates: Sequence[Finding],
    *,
    sources: Sequence[Path],
    rule: str,
) -> list[Finding]:
    """Drop excluded and marker-suppressed findings; report markers that rotted."""
    parser = argparse.ArgumentParser()
    houselint._add_arguments(parser)  # noqa: SLF001 -- one ``[tool.houselint]`` reader.
    # From the cwd, as houselint itself reads it.
    namespace, config_root = pyproject_flags.parse_args(
        parser,
        tool="houselint",
        argv=[],
    )
    known = {r.name for r in houselint.default_rules()} | {rule}
    exclude = [
        entry
        for spec in cast("list[str]", namespace.exclude)
        if not (entry := houselint.parse_exclusion(spec, known=known)).rules
        or rule in entry.rules
    ]
    kept: list[Finding] = []
    flagged = {f.path.absolute() for f in candidates}
    for path in sources:
        # Tokenizing is the cost; only a candidate or a marker naming the rule
        # can yield a finding, and a marker names it literally.
        if path.absolute() not in flagged and rule not in path.read_text():
            continue
        relative = houselint._relative(path, config_root)  # noqa: SLF001 -- same path view.
        if any(entry.matches(relative) for entry in exclude):
            continue
        unit = houselint.read_unit(path)
        if not isinstance(unit, Unit):
            continue
        absolute = path.absolute()
        kept.extend(
            house_ignore.apply(
                unit,
                (f for f in candidates if f.path.absolute() == absolute),
                active=frozenset({rule}),
            ),
        )
    return kept


def _probe(match: re.Match[str], *, checker: str) -> str:
    """Return the probe statement inserted before a matched assert."""
    # A bare expression: binding a name could shadow one the source reads.
    if checker == "pyrefly":
        return f"isinstance({match['args']}) or reveal_type({match['name']}); "
    return f"isinstance({match['args']}); "


def _probe_span(path: Path, *, line: int, checker: str) -> range:
    """Return the 0-based columns the probe occupies on ``path``'s 1-based ``line``."""
    match = _ASSERT.fullmatch(path.read_text().split("\n")[line - 1])
    if match is None:
        return range(0)
    start = len(match["indent"])
    return range(start, start + len(_probe(match, checker=checker)))


# Each directory on the way becomes real, holding links to its children, so writing the
# file never reaches through a link into the checkout.
def _unlink_to(overlay: Path, relative: Path) -> Path:
    """Make ``overlay / relative`` a writable path, unlinking the file it mirrors."""
    target = overlay
    for part in relative.parts:
        target /= part
        if target.is_symlink() and target.is_dir():
            real = target.resolve()
            target.unlink()
            target.mkdir()
            _link_children(real, into=target)
    target.unlink()
    return target


def _link_children(directory: Path, *, into: Path) -> None:
    """Symlink every entry of ``directory`` into the existing directory ``into``."""
    for child in directory.iterdir():
        (into / child.name).symlink_to(child)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Delete the unnecessary asserts.",
    )
    parser.add_argument(
        "--threads",
        type=worker_count,
        help="Checker worker threads (default: the machine's count).",
    )
    parser.add_argument(
        "--checker",
        choices=["pyrefly", "basedpyright"],
        default="pyrefly",
        help="Type checker to run (default: %(default)s).",
    )
    parser.add_argument(
        "--typechecker",
        nargs="+",
        choices=["pyrefly", "basedpyright"],
        default=["pyrefly"],
        help="Checkers that, with ty, must agree an assert is unnecessary "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to check (default: the project's loop/).",
    )


def _checker_lines(
    checker: Path,
    cwd: Path,
    *,
    files: Sequence[Path],
) -> dict[Path, Counter[str]]:
    """Return ``checker``'s diagnostics, each ``line:column: rule message``, per file."""
    options = {
        "ty": ["check", "--output-format", "concise"],
        "basedpyright": ["--threads=1", "--outputjson"],
        "pyrefly": ["check", "--output-format=json", "--summary=none"],
    }[checker.name]
    done = subprocess.run(  # noqa: S603 -- no shell; fixed argv.
        [checker, *options, *map(str, files)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    # Pyrefly exits 1 with no report on a missing path.
    if done.returncode > 1 or (done.returncode and not done.stdout):
        raise CheckerCrashError(done.returncode, done.stderr)
    found: dict[Path, Counter[str]] = {}
    for path, key in _diagnostic_keys(checker.name, done.stdout, cwd=cwd):
        found.setdefault(path, Counter())[key] += 1
    return found


def _diagnostic_keys(
    checker: str,
    report: str,
    *,
    cwd: Path,
) -> Generator[tuple[Path, str]]:
    """Yield ``(absolute file, "line:column: rule message")`` for each diagnostic."""
    if checker == "ty":
        for row in report.splitlines():
            file, _, rest = row.partition(":")
            if file and re.match(r"\d+:\d+: ", rest):
                yield (cwd / file).absolute(), rest
        return
    if checker == "basedpyright":
        items = from_plain(
            json_loads(report, dict[str, object]).get("generalDiagnostics"),
            list[dict[str, object]],
        )
        for item in items:
            where = from_plain(item.get("range"), dict[str, object], default={})
            start = from_plain(where.get("start"), dict[str, object], default={})
            line = from_plain(start.get("line"), int, default=0)
            column = from_plain(start.get("character"), int, default=0)
            rule = from_plain(item.get("rule"), str, default="")
            message = from_plain(item.get("message"), str)
            path = Path(from_plain(item.get("file"), str))
            yield path, f"{line}:{column}: {rule} {message}"
        return
    items = from_plain(
        json_loads(report, dict[str, object]).get("errors"),
        list[dict[str, object]],
    )
    for item in items:
        path = Path(os.path.normpath(cwd / from_plain(item.get("path"), str)))
        line = from_plain(item.get("line"), int)
        column = from_plain(item.get("column"), int)
        rule = from_plain(item.get("name"), str)
        message = from_plain(item.get("description"), str)
        yield path, f"{line}:{column}: {rule} {message}"


def _covered(
    child: Path,
    *,
    chosen: set[Path],
    holding: set[Path],
    is_full: Callable[[Path], bool],
) -> bool:
    """Whether naming ``child``'s parent would check nothing outside ``chosen``."""
    if child in chosen:
        return True
    if child.is_symlink():
        return False
    if child.is_dir():
        if child in holding:
            return is_full(child)
        return not _holds_source(child)
    return child.suffix not in _SOURCE_SUFFIXES


# Pyright's default excludes (hidden directories among them) apply only when a
# config sets no `exclude`, so no directory is skipped here.
def _holds_source(directory: Path) -> bool:
    """Whether basedpyright would find a source or symlink under ``directory``."""
    for parent, dirs, names in directory.walk():
        if any(Path(name).suffix in _SOURCE_SUFFIXES for name in names) or any(
            (parent / name).is_symlink() for name in dirs
        ):
            return True
    return False
