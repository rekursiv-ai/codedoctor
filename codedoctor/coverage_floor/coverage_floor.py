r"""Floor each changed source's coverage by its own sidecar tests.

Takes the changed paths as arguments. A source under a floored prefix (one
``--floor PREFIX/=PERCENT`` each) with a paired test (``x_test.py``,
``test_x.py``, or either under ``tests/``) has that test run with its directory measured. The
run fails when a test fails or a floored unit is below its floor: each
changed source under ``--scope file``, or the changed sources of one
directory pooled under ``--scope directory``. A source without a sidecar is
skipped.

Examples:
  python -m codedoctor.coverage_floor --floor pkg/=100 -- pkg/a.py
  python -m codedoctor.coverage_floor --scope directory \
    --floor pkg/=90 -- pkg/a.py

"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import argparse
import ast
import os
import subprocess
import sys
import tempfile

from coverage import results

import coverage

from codedoctor.lib.testing.resources import kernel_pins
from codedoctor.lib.worker_count import capped_workers, default_workers, worker_count


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
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    floors = dict(flags.floor)
    sources, tests = sidecars(flags.paths, floored=tuple(floors))
    if not sources:
        return 0
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
    return int(failed or below)


class Flags(Protocol):
    """Parsed command-line flags."""

    paths: list[str]
    workers: int | None
    scope: Literal["file", "directory"]
    floor: list[tuple[str, float]]


def sidecars(
    paths: Iterable[str],
    *,
    floored: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Return the changed floored sources that have a sidecar, and those sidecars.

    Args:
      paths: Changed repo-relative paths.
      floored: Directory prefixes, each ending in ``/``, whose sources count.

    Returns:
      sources: Each floored, non-test source with at least one sidecar.
      tests: Every sidecar of those sources, in order, deduplicated.

    """
    sources: list[str] = []
    tests: dict[str, None] = {}
    for path in paths:
        name = Path(path)
        if _is_test(name) or not path.startswith(floored):
            continue
        found = [
            str(candidate)
            for candidate in (
                name.with_name(f"{name.stem}_test.py"),
                name.with_name(f"test_{name.stem}.py"),
                name.parent / "tests" / f"{name.stem}_test.py",
                name.parent / "tests" / f"test_{name.stem}.py",
            )
            if candidate.is_file()
        ]
        if found:
            sources.append(path)
            tests.update(dict.fromkeys(found))
    return sources, list(tests)


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
    loads = [0] * capped_workers(count, units=len(tests))
    members: list[list[str]] = [[] for _ in loads]
    for test in sorted(tests, key=lambda t: (-Path(t).stat().st_size, t)):
        lightest = loads.index(min(loads))
        loads[lightest] += Path(test).stat().st_size
        members[lightest].append(test)
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
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        runs = list(
            pool.map(
                partial(_pytest, cov=cov, data=data),
                range(len(groups)),
                groups,
            ),
        )
    # Only a failing group's output: 24 passing groups' dots say nothing.
    for run in runs:
        if run.returncode:
            sys.stdout.write(run.stdout)
    combined = coverage.Coverage(data_file=str(data))
    combined.combine([str(data.parent)])
    combined.save()
    return any(run.returncode for run in runs)


def report(
    data: Path,
    *,
    sources: Sequence[str],
    floors: Mapping[str, float],
    scope: Literal["file", "directory"],
) -> bool:
    """Print the report for ``sources`` and whether any unit is below its floor.

    Args:
      data: Combined coverage data file.
      sources: Changed sources to floor, each under a prefix of ``floors``.
      floors: Least passing percent per path prefix; the longest match wins.
      scope: ``file`` floors each source; ``directory`` pools the sources of
        each directory, so a weak file passes when its changed siblings carry
        the directory over the floor.

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
    if not morfs:
        return False
    # No table: printing one re-read every source, 10s of an --all-files run, and
    # each unit below its floor already names its missed lines.
    units: dict[str, tuple[float, int, int, list[str]]] = {}
    for source in morfs:
        _, statements, _, missing, gaps = measurement.analysis2(source)
        unit = source if scope == "file" else str(Path(source).parent)
        floor = floors[max((p for p in floors if source.startswith(p)), key=len)]
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


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("paths", nargs="*", help="Changed repository-relative paths.")
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
        type=_floor,
        action="append",
        default=[],
        metavar="PREFIX/=PERCENT",
        help="Add or override one prefix's floor; repeatable (default: %(default)s).",
    )


def _floor(text: str) -> tuple[str, float]:
    """Parse ``PREFIX/=PERCENT`` into its prefix and percent."""
    prefix, _, percent = text.rpartition("=")
    if not prefix.endswith("/"):
        msg = f"Floor {text!r} is not PREFIX/=PERCENT."
        raise argparse.ArgumentTypeError(msg)
    return prefix, float(percent)


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


def _is_test(path: Path) -> bool:
    """Return whether ``path`` is a test module or conftest."""
    return (
        path.name == "conftest.py"
        or path.name.endswith("_test.py")
        or path.name.startswith("test_")
    )


def _pytest(
    index: int,
    tests: Sequence[str],
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
            "--cov-report=",
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
