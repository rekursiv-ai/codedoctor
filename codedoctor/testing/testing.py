"""Run the tests a change reaches, or one whole-suite pytest tier.

Commands:
  pytest          The tests paired with the spooled files, then their coverage.
  coverage-floor  Floor each spooled file by its own tests, from ``pytest``'s run.
  tier NAME       One whole-suite pytest tier from ``TIERS``.

Examples:
  python -m codedoctor.gate spool pytest a.py b.py
  python -m codedoctor.testing pytest
  python -m codedoctor.testing coverage-floor
  python -m codedoctor.testing tier pytest-global

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import contextlib
import shlex
import shutil
import sys
import tempfile
import time

from _pytest.config import hookimpl


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from codedoctor.gate import gate
    from codedoctor.lib import sidecars
    from codedoctor.testing import coverage_summary, listing, tiers
    from codedoctor.testing.tiers import Tier
else:
    from wrapt import lazy_import

    # Pytest loads this plugin before coverage starts; defer the gate's imports.
    gate = lazy_import("codedoctor.gate.gate")
    sidecars = lazy_import("codedoctor.lib.sidecars")
    coverage_summary = lazy_import("codedoctor.testing.coverage_summary")
    listing = lazy_import("codedoctor.testing.listing")
    tiers = lazy_import("codedoctor.testing.tiers")


@hookimpl(wrapper=True)
def pytest_report_teststatus() -> Generator[
    None,
    tuple[str, str, str | tuple[str, dict[str, bool]]],
    tuple[str, str, str],
]:
    """Keep outcome counts without per-test progress characters or words."""
    category, _, _ = yield
    return category, "", ""


def main() -> int:
    """Run the paired tests or a tier.

    Returns:
      exit_code: pytest's, or the coverage summary's.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    flags = _parse_args(parser)
    return flags.handler(flags, gate.context())


def pytest(context: gate.Context, *, dependents: str | None) -> int:
    """Run the tests paired with the ``pytest`` spool, then gate their coverage.

    Pairs ``x.py`` with ``x_test.py`` or ``test_x.py``, beside it or under
    ``tests/``. Green with no pair found means "untested", not "tested".

    Args:
      context: What the gate reads and runs.
      dependents: Module whose CLI names tests within the resolved affected:N
        scope. They run in a second pytest, unmeasured; the selected own tests
        run first, measured per test for ``coverage-floor``.
        Both runs keep the project's default marker selection.

    Returns:
      exit_code: pytest's, 0 for an empty collection, or the coverage summary's;
        1 when the spool was never written.

    """
    name = "pytest"
    files = gate.drained(name, context=context)
    if files is None:
        return gate.unspooled(name)
    # Until this run's coverage replaces it, the hand-off says why there is none,
    # for ``coverage-floor`` to print. Absent means ``pytest`` never ran, as under
    # ``SKIP``; an earlier run's data never survives into this one.
    left = handoff(context)
    left.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _drop_stale_handoffs(left.parent, now=time.time())
    left.write_text("the pytest hook did not finish\n")
    count = gate.workers(name, environ=context.environ)
    own, extra = _selected(context, files=files, dependents=dependents, workers=count)
    if not own and not extra:
        left.write_text("the pytest hook found no tests\n")
        return 0
    sources = [path for path in files if not sidecars.is_test(context.root / path)]
    directories = sorted({Path(path).parent.as_posix() for path in sources})
    measured = context.installed("pytest_cov")
    cov: list[str] = (
        [
            *(f"--cov={directory}" for directory in directories),
            # Which test ran each line: ``coverage-floor`` counts a file's own.
            "--cov-context=test",
            "--no-cov-on-fail",
            "--cov-fail-under=0",
            "--cov-report=term-missing:skip-covered"
            if context.environ.get("PRECOMMIT_COVERAGE_DETAIL") == "1"
            else "--cov-report=",
        ]
        if measured and directories
        else []
    )
    # A private data file: `coverage erase` or a combine in the checkout deletes
    # every `.coverage.*` there, killing a concurrent `pytest --cov`'s workers.
    with tempfile.TemporaryDirectory(prefix="precommit-") as temp:
        # pragma: no mutate start -- any name in a private dir.
        data = Path(temp) / ".coverage"
        # pragma: no mutate end
        # Measure only own tests: graph imports would otherwise count as own.
        # Keep pytest's default marker selection in both passes.
        passes: list[tuple[list[str], list[str], list[str]]] = [
            (own, [], cov),
            (extra, [], []),
        ]
        for index, (tests, marker, measuring) in enumerate(passes):
            if not tests:
                continue
            status = _run_pytest(
                context,
                tests=tests,
                marker=marker,
                cov=measuring,
                data=data,
                listing_file=Path(temp) / f"selected-{index}.txt",
                count=count,
            )
            if status not in {0, 5}:
                left.write_text(f"pytest exited {status}\n")
                return status
        if not cov or not data.is_file():
            left.write_text(
                "pytest passed but recorded no coverage"
                + ("" if cov else ": pytest-cov is not installed")
                + "\n",
            )
            return 0
        below = coverage_summary.shortfall(sources, data=data)
        shutil.move(data, left)
    sys.stderr.writelines(f"{line}\n" for line in below)
    return int(bool(below))


def floor(context: gate.Context) -> int:
    """Floor the ``coverage-floor`` spool's files from the ``pytest`` gate's run.

    The tests run once, in ``pytest``; this reads the data it left, so skipping
    ``pytest`` measures nothing here either.

    Args:
      context: What the gate reads and runs.

    Returns:
      exit_code: The floor's; 1 when the spool was never written.

    """
    name = "coverage-floor"
    files = gate.drained(name, context=context)
    if files is None:
        return gate.unspooled(name)
    data = handoff(context)
    try:
        if not files:
            return 0
        with gate.argfile(files) as named:
            return context.run(
                [sys.executable, "-m", _FLOOR, "--data", str(data), "--", named],
            )
    finally:
        data.unlink(missing_ok=True)


def handoff(context: gate.Context) -> Path:
    """Return where this run's ``pytest`` gate leaves its coverage data."""
    return context.directory / f"coverage.{context.key}"


# A run whose ``coverage-floor`` never drains its hand-off -- ``SKIP=coverage-floor``
# -- would leave one data file per run. Age, not key: a concurrent run's is live.
_STALE_SEC: Final = 24 * 60 * 60


# The floor imports coverage, which an export without pytest-cov lacks; named, not
# imported, so this module still loads there.
_FLOOR: Final = f"{__name__.rsplit('.', 2)[0]}.coverage_floor"


def tier(spec: Tier, *, context: gate.Context, plugin: str | None) -> int:
    """Run a whole-suite pytest tier, selecting with testmon when that is safe.

    Testmon and pytest-split both partition one collection, and which wins is
    plugin order: a sharded run with a warm database once ran 8 of 118 tests and
    stayed green. So selection is derived, never configured: a non-empty
    ``PYTEST_SPLIT_ARGS`` (a shard) or ``PYTEST_FORCE_ALL`` turns it off. It is
    ``--testmon-forceselect``, not ``--testmon``: the tier passes ``-m``, and
    plain ``--testmon`` then selects nothing (tarpas/pytest-testmon#238).
    ``COVERAGE_CORE=ctrace`` because sysmon cannot carry testmon's per-test
    contexts, and testmon reads only the environment.

    ``PYTHONUNBUFFERED=1``: pre-commit reads the hook through a pipe, so a tier
    killed for memory otherwise leaves no progress at all.

    Args:
      spec: The tier.
      context: What the gate reads and runs.
      plugin: Module loaded with ``-p`` when collecting a selection, if
        importable; it writes the selected node ids.

    Returns:
      exit_code: pytest's; 0 when the tier is skipped, its selection is empty,
        or, under ``allow_empty``, nothing was collected.

    """
    environ = context.environ
    if spec.requires is not None and context.which(spec.requires) is None:
        return 0
    nodeids: list[str] = []
    if gate.flagged(environ, "PYTEST_NODEIDS_FILE"):
        listed = Path(environ["PYTEST_NODEIDS_FILE"]).read_text().splitlines()
        nodeids = [line for line in listed if "::" in line]
        if not nodeids:
            return 0
    split = shlex.split(environ.get("PYTEST_SPLIT_ARGS", ""))
    selects = (
        not gate.flagged(environ, "PYTEST_FORCE_ALL")
        and not split
        and (
            spec.ci_testmon
            or gate.flagged(environ, "PYTEST_COLLECT_SELECTED")
            or not gate.flagged(environ, "GITHUB_ACTIONS")
        )
        and context.installed("testmon")
    )
    select = ["--testmon-forceselect"] if selects else []
    measured = context.installed("pytest_cov")
    cov: list[str] = []
    if not selects and not gate.flagged(environ, "PYTEST_TESTMON_ARGS") and measured:
        cov = ["--no-cov"]
    # Bare `--cov` measures the project, as `[tool.coverage.run]` scopes it.
    if gate.flagged(environ, "PYTEST_COVERAGE_REPORT") and measured:
        cov = ["--cov", "--cov-report=term:skip-covered"]
    marker = ["-m", spec.marker] if spec.marker else []
    pytest_argv = [
        "env",
        "PYTHONUNBUFFERED=1",
        "COVERAGE_CORE=ctrace",
        sys.executable,
        "-m",
        "pytest",
        f"-n={gate.workers(spec.name, environ=environ)}",
    ]
    if gate.flagged(environ, "PYTEST_COLLECT_SELECTED"):
        loaded = (
            ["-p", plugin] if plugin is not None and context.installed(plugin) else []
        )
        status = context.run(
            [
                *pytest_argv,
                *select,
                *loaded,
                "-q",
                "--tb=short",
                *spec.extra,
                "--collect-only",
                *marker,
            ],
        )
        return 0 if status in {0, 5} else status
    status = context.run(
        [
            *pytest_argv,
            *cov,
            *select,
            *shlex.split(environ.get("PYTEST_TESTMON_ARGS", "")),
            "-q",
            "--tb=short",
            "-p",
            __name__,
            *spec.extra,
            *shlex.split(environ.get("PYTEST_PROGRESS_ARGS", "")),
            *marker,
            *split,
            *nodeids,
        ],
    )
    return 0 if spec.allow_empty and status == 5 else status


class Flags(Protocol):
    """Parsed command-line flags."""

    handler: Callable[[Flags, gate.Context], int]
    tier: str
    dependents: str | None
    plugin: str | None


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> Flags:
    """Register ``pytest`` and ``tier`` and parse ``argv``."""
    commands = parser.add_subparsers(dest="command", required=True)
    tests = commands.add_parser("pytest", help="Run the tests paired with the spool.")
    tests.set_defaults(handler=_test)
    gate.add_dependents(tests)
    floored = commands.add_parser(
        "coverage-floor",
        help="Floor each spooled file by its own tests, from pytest's run.",
    )
    floored.set_defaults(handler=_floor)
    suite = commands.add_parser("tier", help="Run one whole-suite pytest tier.")
    suite.set_defaults(handler=_run_tier)
    suite.add_argument("tier", choices=sorted(tiers.TIERS), help="The tier.")
    suite.add_argument(
        "--plugin",
        metavar="MODULE",
        help="Load this module with -p when collecting a selection.",
    )
    return cast(Flags, parser.parse_args(argv))


def _test(flags: Flags, context: gate.Context) -> int:
    return pytest(context, dependents=flags.dependents)


def _floor(flags: Flags, context: gate.Context) -> int:
    del flags
    return floor(context)


def _run_tier(flags: Flags, context: gate.Context) -> int:
    return tier(tiers.TIERS[flags.tier], context=context, plugin=flags.plugin)


# The own tests are what ``coverage-floor`` floors a file by, so the graph may widen the
# run but never replace them.
def _selected(
    context: gate.Context,
    *,
    files: Sequence[str],
    dependents: str | None,
    workers: int,
) -> tuple[list[str], list[str]]:
    """Return each file's own tests, then the graph's tests in no own file."""
    depth = gate.scope("pytest", environ=context.environ)
    selected = (
        {Path(path) for path in files}
        if depth < 0
        else {
            file for path in files for file in sidecars.related(path, root=context.root)
        }
    )
    own = sorted(
        path.as_posix() for path in selected if sidecars.is_test(context.root / path)
    )
    walked = gate.walk(
        context,
        files=files,
        dependents=dependents,
        hops=depth,
        emit="nodeids",
        workers=workers,
    )
    owned = set(own)
    # Export overlays are public-root templates, not runnable monorepo tests.
    runnable = {
        test.partition("::")[0]
        for test in (*own, *walked)
        if ".export" not in Path(test.partition("::")[0]).parts
    }
    return [test for test in own if test in runnable], sorted(
        {test for test in walked if test.partition("::")[0] in runnable - owned},
    )


def _run_pytest(
    context: gate.Context,
    *,
    tests: Sequence[str],
    marker: Sequence[str],
    cov: Sequence[str],
    data: Path,
    listing_file: Path,
    count: int,
) -> int:
    """Run one pytest over ``tests``, into ``data`` when ``cov``; return its status."""
    # Files, not nodeids: the threshold was measured in files, and dozens of
    # nodeids from one file paid xdist's ~5s startup for nothing.
    spread = len({test.partition("::")[0] for test in tests}) >= (
        gate.positive(context.environ.get("SIDECAR_XDIST_MIN")) or 16
    )
    # Serial needs ``--dist=no`` where xdist is installed: a ``--dist`` in addopts
    # makes pytest-cov its xdist master, which records no per-test contexts.
    serial = ["--dist=no"] if context.installed("xdist") else []
    # Files as arguments and the nodeids as a listing: once one nodeid argument
    # points into a module that skips at module level, pytest drops every other
    # argument and the gate passes on "1 skipped". Whole files need no listing,
    # and none is loaded then: a ``-p`` plugin imports before coverage starts, so
    # every line it imports would read as never run.
    files_run = sorted({test.partition("::")[0] for test in tests})
    narrowed: list[str] = []
    if any("::" in test for test in tests):
        listing_file.write_text("".join(f"{test}\n" for test in tests))
        narrowed = ["-p", listing.__name__, f"--test-files={listing_file}"]
    with gate.argfile([*cov, *files_run]) as named:
        return context.run(
            [
                "env",
                f"COVERAGE_FILE={data}",
                # Per-test contexts need it: sysmon stops reporting a line once
                # seen, so a line first run at import has no test. Slower, so
                # only where something is measured.
                *(["COVERAGE_CORE=ctrace"] if cov else []),
                sys.executable,
                "-m",
                "pytest",
                *(["-n", str(count)] if spread else serial),
                "-q",
                "--tb=short",
                "-p",
                __name__,
                *marker,
                *narrowed,
                named,
            ],
        )


def _drop_stale_handoffs(directory: Path, *, now: float) -> None:
    """Remove hand-offs in ``directory`` last written over a day before ``now``."""
    for stale in directory.glob("coverage.*"):
        with contextlib.suppress(FileNotFoundError):
            if now - stale.stat().st_mtime > _STALE_SEC:
                stale.unlink()
