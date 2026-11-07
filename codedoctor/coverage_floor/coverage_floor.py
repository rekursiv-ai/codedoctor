r"""Floor each changed source's coverage by its own sidecar tests.

Takes the changed paths as arguments. A source under a floored prefix (one
``--floor PREFIX/=PERCENT`` each, added to the ``floor`` list of
``[tool.coverage-floor]`` in ``pyproject.toml``; an empty prefix floors every
path) with a paired test (as ``lib.sidecars`` pairs them) has that test run
with its directory measured. The run fails when a test fails or a floored unit
is below its floor: each changed source under ``--scope file``, or the changed
sources of one directory pooled under ``--scope directory``. A floored source
without a sidecar fails, save ``__init__.py`` and ``__main__.py``. Vendored
sources, anything under a ``third_party/`` directory, are never floored, nor
is a path the coverage configuration's ``omit`` drops.

With ``--data`` it runs no tests: it reads the ``pytest`` gate's run, whose lines
carry the test that ran them, and counts only each source's own tests and its
import. The ``coverage-floor`` hook runs it so, through ``testing coverage-floor``.

Examples:
  python -m codedoctor.coverage_floor --floor pkg/=100 -- pkg/a.py
  python -m codedoctor.coverage_floor --data .coverage -- pkg/a.py
  python -m codedoctor.coverage_floor --scope directory \
    --floor pkg/=90 -- pkg/a.py

"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from importlib import util
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import argparse
import ast
import os
import re
import socket
import subprocess
import sys
import tempfile

from coverage import results
from coverage.files import GlobMatcher, prep_patterns

import coverage

from codedoctor.lib import pyproject_flags
from codedoctor.lib.kernel_pins import kernel_pins
from codedoctor.lib.sidecars import is_test, paired
from codedoctor.lib.worker_count import capped_workers, default_workers, worker_count
from codedoctor.testing.tiers import RUNNABLE


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def main() -> int:
    """Run the program.

    Returns:
      exit_code: 1 on a failed sidecar or a changed source below the floor.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # The gate hands every path as one @argfile; argv overflows ARG_MAX.
        fromfile_prefix_chars="@",  # pragma: no mutate -- holds "@" either way.
    )
    _add_arguments(parser)
    namespace, _ = pyproject_flags.parse_args(parser, tool="coverage-floor")
    flags = cast(Flags, namespace)
    try:
        floors = dict(map(_floor, flags.floor))
    except ValueError as err:
        parser.error(str(err))
    sources, tests, untested = sidecars(flags.paths, floored=tuple(floors))
    # Reported, then measured anyway: stopping here hid every under-covered
    # source behind the first untested one.
    sys.stdout.writelines(
        f"coverage-floor: {path} has no sidecar test\n" for path in untested
    )
    if flags.data is not None:
        return int(bool(untested) or _report_run(flags.data, sources, flags, floors))
    if not sources:
        return int(bool(untested))
    # One line per broken sidecar: pytest's collection traceback is 30 lines
    # for a single SyntaxError, which ruff has already reported.
    if broken := _unparsable(tests):
        sys.stdout.writelines(f"coverage-floor: {error}\n" for error in broken)
        return 1
    with tempfile.TemporaryDirectory(prefix="coverage-floor-") as temp:
        data = Path(temp) / ".coverage"
        failed = run_groups(
            groups(tests, count=flags.workers or default_workers()),
            cov=sorted({f"--cov={Path(source).parent}" for source in sources}),
            data=data,
        )
        below = report(data, sources=sources, floors=floors, scope=flags.scope)
    return int(bool(untested) or failed or below)


class Flags(Protocol):
    """Parsed command-line flags."""

    paths: list[str]
    workers: int | None
    scope: Literal["file", "directory"]
    floor: list[str]
    data: Path | None


def sidecars(
    paths: Iterable[str],
    *,
    floored: tuple[str, ...],
) -> tuple[list[str], list[str], list[str]]:
    """Return the changed floored sources that have a sidecar, those sidecars, and the rest.

    Args:
      paths: Changed repo-relative paths.
      floored: Directory prefixes, each ending in ``/`` or empty for every
        path, whose sources count, save those under a ``third_party/``
        directory or omitted by the coverage configuration of the cwd.

    Returns:
      sources: Each floored, non-test source with at least one sidecar.
      tests: Every sidecar of those sources, in order, deduplicated.
      untested: Each floored, non-test source on disk with no sidecar, save
        ``__init__.py`` and ``__main__.py``.

    """
    omitted = GlobMatcher(prep_patterns(coverage.Coverage().config.run_omit))
    sources: list[str] = []
    tests: dict[str, None] = {}
    untested: list[str] = []
    for path in paths:
        name = Path(path)
        if (
            is_test(name)
            or not path.startswith(floored)
            or "third_party" in name.parts
            # No run measures what coverage's `omit` drops -- fixtures, stubs --
            # so demanding its sidecar would fail every one.
            or omitted.match(str(name.resolve()))
        ):
            continue
        found = [str(test) for test in paired(name)]
        if found:
            sources.append(path)
            tests.update(dict.fromkeys(found))
        elif name.is_file() and name.name not in ("__init__.py", "__main__.py"):
            untested.append(path)
    return sources, list(tests), untested


def groups(tests: Sequence[str], *, count: int) -> list[list[str]]:
    """Split ``tests`` into up to ``count`` groups of near-equal total size.

    Each group is one serial pytest. xdist would pay ~8s of worker startup and
    re-collection on top of the run; size stands in for runtime, which on 472
    sidecars balanced 24 groups to within 1.3x of each other.

    Args:
      tests: Test files.
      count: Most groups.

    Returns:
      groups: Nonempty lists of test files, each in path order.

    """
    sizes = {test: Path(test).stat().st_size for test in tests}
    members: list[list[str]] = [
        [] for _ in range(capped_workers(count, units=len(tests)))
    ]
    for test in sorted(tests, key=lambda t: (-sizes[t], t)):
        min(members, key=lambda group: sum(sizes[t] for t in group)).append(test)
    # Path order: pytest 9 splits a package whose files are interleaved with
    # another directory's, and the second copy loses the package's conftest
    # fixtures (`fixture 'prepared_manifest' not found`).
    return [sorted(group) for group in members if group]


def run_groups(
    groups: Sequence[Sequence[str]],
    *,
    cov: Sequence[str],
    data: Path,
) -> bool:
    """Run each group as one pytest, concurrently, writing ``data`` combined.

    Args:
      groups: Test files per pytest process.
      cov: ``--cov=DIR`` arguments every process measures.
      data: Coverage data file to combine every group's measurements into.

    Returns:
      failed: Whether any group's pytest exited nonzero.

    """
    ports = _distinct_ports(len(groups))
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        runs = list(
            pool.map(
                partial(_pytest, cov=cov, data=data),
                range(len(groups)),
                groups,
                [_database_args(port) for port in ports],
            ),
        )
    # Only a failing group's output: 24 passing groups' dots say nothing.
    for run in runs:
        if run.returncode:
            sys.stdout.write(run.stdout)
    combined = coverage.Coverage(data_file=str(data))
    combined.combine()
    combined.save()
    return any(run.returncode for run in runs)


def report(
    data: Path,
    *,
    sources: Sequence[str],
    floors: Mapping[str, float],
    scope: Literal["file", "directory"],
    owners: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """Print the report for ``sources`` and whether any unit is below its floor.

    Args:
      data: Combined coverage data file.
      sources: Changed sources to floor, each under a prefix of ``floors``.
      floors: Least passing percent per path prefix; the longest match wins.
      scope: ``file`` floors each source; ``directory`` pools the sources of
        each directory, so a weak file passes when its changed siblings carry
        the directory over the floor.
      owners: Each source's own test files, when ``data`` holds a wider run
        whose lines carry pytest-cov's per-test contexts; only those tests'
        lines count. None counts every line: the run held only own tests.

    Returns:
      below: Whether any measured unit is under its floor; with no source
        measured, False.

    """
    measurement = coverage.Coverage(data_file=str(data))
    measurement.load()
    measured = set(measurement.get_data().measured_files())
    # `morfs`, not `include`: an include list is matched as one glob per
    # source against every measured file, 3.7s for 472 sources. A source the
    # data never saw is left out, exactly as `include` leaves it out.
    morfs = [source for source in sources if str(Path(source).resolve()) in measured]
    # No table: printing one re-read every source, 10s of an --all-files run, and
    # each unit below its floor already names its missed lines.
    units: dict[str, tuple[float, int, int, list[str]]] = {}
    for source in morfs:
        if owners is not None:
            # The empty context is import: a module's own lines run then, before
            # any test, whichever test file imported it.
            measurement.get_data().set_query_contexts(
                ["^$", *(f"^{re.escape(test)}::" for test in owners[source])],
            )
        _, statements, _, missing, gaps = measurement.analysis2(source)
        unit = source if scope == "file" else str(Path(source).parent)
        floor = floors[
            max(
                (p for p in floors if source.startswith(p)),
                key=len,  # pragma: no mutate -- nested prefixes sort longest last.
            )
        ]
        _, total, missed, where = units.get(unit, (floor, 0, 0, []))
        if gaps:
            where = [*where, gaps if scope == "file" else f"{source}: {gaps}"]
        units[unit] = (floor, total + len(statements), missed + len(missing), where)
    precision = measurement.config.precision
    below = False
    for unit, (floor, total, missed, where) in units.items():
        percent = 100 * (total - missed) / total if total else 100.0
        if results.should_fail_under(percent, floor, precision):
            sys.stdout.write(
                f"coverage-floor: {unit} at "
                f"{results.display_covered(percent, precision)}% "
                f"is below {floor:g}%; missed {'; '.join(where)}\n",
            )
            below = True
    return below


# No file is a ``pytest`` gate that never ran, as under ``SKIP``: nothing to floor, and
# the tests are not run here. A file that is not coverage data is the gate's own line
# saying why it measured nothing -- it failed, crashed or recorded no coverage -- which
# must not pass.
def _report_run(
    data: Path,
    sources: Sequence[str],
    flags: Flags,
    floors: Mapping[str, float],
) -> bool:
    """Floor ``sources`` by their own tests' lines in the ``pytest`` gate's ``data``."""
    if not sources or not data.is_file():
        return False
    with data.open("rb") as handle:
        header = handle.read(len(_SQLITE))
    if header != _SQLITE:
        reason = data.read_text(errors="replace").strip() or "an empty hand-off"
        sys.stdout.write(
            f"coverage-floor: {reason}; nothing to floor {len(sources)} file(s) by\n",
        )
        return True
    return report(
        data,
        sources=sources,
        floors=floors,
        scope=flags.scope,
        owners={source: [str(test) for test in paired(source)] for source in sources},
    )


# Coverage data is an SQLite database; anything else is the gate's reason it has none.
_SQLITE: Final = b"SQLite format 3\x00"


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("paths", nargs="*", help="Changed repository-relative paths.")
    parser.add_argument(
        "--data",
        type=Path,
        help=(
            "Floor from this pytest run's per-test coverage data instead of "
            "running the tests; absent means pytest did not run."
        ),
    )
    parser.add_argument(
        "--workers",
        type=worker_count,
        help="Concurrent pytest processes (default: the machine's count).",
    )
    parser.add_argument(
        "--scope",
        choices=("file", "directory"),
        default="file",
        help=(
            "Floor each changed file, or each directory's changed files pooled "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--floor",
        action="append",
        default=[],
        metavar="PREFIX/=PERCENT",
        help="Add or override one prefix's floor; repeatable (default: %(default)s).",
    )


def _floor(text: str) -> tuple[str, float]:
    """Parse ``PREFIX/=PERCENT`` into its prefix and percent; ``=PERCENT`` floors all."""
    prefix, sep, percent = text.rpartition("=")
    if sep and (prefix.endswith("/") or not prefix):
        with suppress(ValueError):
            return prefix, float(percent)
    msg = f"Floor {text!r} is not PREFIX/=PERCENT."
    raise ValueError(msg)


def _unparsable(paths: Iterable[str]) -> list[str]:
    """Return ``path:line: message`` for each of ``paths`` Python cannot parse."""
    errors: list[str] = []
    for path in paths:
        try:
            ast.parse(Path(path).read_bytes())
        except FileNotFoundError:
            continue
        except SyntaxError as error:
            errors.append(f"{path}:{error.lineno}: {error.msg}")
    return errors


# Every socket is held open until all are bound, so no two can be the same.
def _distinct_ports(count: int) -> list[int]:
    """Return ``count`` free local ports, all different from one another."""
    held = [socket.socket() for _ in range(count)]
    try:
        for held_socket in held:
            held_socket.bind(("127.0.0.1", 0))
        return [held_socket.getsockname()[1] for held_socket in held]
    finally:
        for held_socket in held:
            held_socket.close()


# Groups start together, and pytest-postgresql picks each server's port at random
# from the free ones: two groups once picked the same and every database test of the
# second errored. A port per group, chosen here, cannot collide.
def _database_args(port: int) -> list[str]:
    """Return the pytest flags pinning a group's Postgres server to ``port``."""
    if util.find_spec("pytest_postgresql") is None:
        return []
    return [f"--postgresql-port={port}"]


def _pytest(
    index: int,
    tests: Sequence[str],
    extra: Sequence[str] = (),
    *,
    cov: Sequence[str],
    data: Path,
) -> subprocess.CompletedProcess[str]:
    """Run one serial pytest over ``tests`` into its own parallel data file."""
    return subprocess.run(  # noqa: S603 -- this interpreter; derived arguments.
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:randomly",
            "--tb=line",
            "-m",
            RUNNABLE,
            "--cov-report=",
            # One group measures part of a package; the floor is judged after
            # every group's data is combined, never by the project's `fail_under`.
            "--cov-fail-under=0",
            *extra,
            *cov,
            "--",
            *tests,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # Groups already fill the cores; a torch test at its default thread
        # count would oversubscribe them, as `mutated.single_threaded_env` says.
        env={
            **kernel_pins(),
            **os.environ,
            "COVERAGE_FILE": f"{data}.{index}",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
