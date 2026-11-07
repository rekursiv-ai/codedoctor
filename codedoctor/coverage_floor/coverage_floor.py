r"""Floor each changed source's coverage by its own sidecar tests.

Runs no tests: it reads the ``pytest`` gate's run (``--data``), whose lines carry
the test that ran them, and counts only each source's own tests -- as
``lib.sidecars`` pairs them -- and its import. The ``coverage-floor`` hook runs
it so, through ``testing coverage-floor``.

Takes the changed paths as arguments. A source under a floored prefix (one
``--floor PREFIX/=PERCENT`` each, added to the ``floor`` list of
``[tool.coverage-floor]`` in ``pyproject.toml``; an empty prefix floors every
path) fails below its floor: each changed source under ``--scope file``, or the
changed sources of one directory pooled under ``--scope directory``. A floored
source without a sidecar fails, save ``__init__.py`` and ``__main__.py``.
Vendored sources, anything under a ``third_party/`` directory, are never
floored, nor is a path the coverage configuration's ``omit`` drops.

Examples:
  python -m codedoctor.coverage_floor --data .coverage -- pkg/a.py
  python -m codedoctor.coverage_floor --data .coverage --scope directory \
    --floor pkg/=90 -- pkg/a.py

"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import argparse
import os
import re
import sys

from coverage import results
from coverage.files import GlobMatcher, prep_patterns

import coverage

from codedoctor.gate.gate import skipped
from codedoctor.lib import pyproject_flags
from codedoctor.lib.sidecars import is_test, paired


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
    sources, untested = sidecars(flags.paths, floored=tuple(floors))
    # Reported, then measured anyway: stopping here hid every under-covered
    # source behind the first untested one.
    sys.stdout.writelines(
        f"coverage-floor: {path} has no sidecar test\n" for path in untested
    )
    below = _report_run(flags.data, sources, flags, floors)
    return int(bool(untested) or below)


class Flags(Protocol):
    """Parsed command-line flags."""

    paths: list[str]
    scope: Literal["file", "directory"]
    floor: list[str]
    data: Path


def sidecars(
    paths: Iterable[str],
    *,
    floored: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Return the changed floored sources that have a sidecar, and those without.

    Args:
      paths: Changed repo-relative paths.
      floored: Directory prefixes, each ending in ``/`` or empty for every
        path, whose sources count, save those under a ``third_party/``
        directory or omitted by the coverage configuration of the cwd.

    Returns:
      sources: Each floored, non-test source with at least one sidecar.
      untested: Each floored, non-test source on disk with no sidecar, save
        ``__init__.py`` and ``__main__.py``.

    """
    omitted = GlobMatcher(prep_patterns(coverage.Coverage().config.run_omit))
    sources: list[str] = []
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
        if paired(name):
            sources.append(path)
        elif name.is_file() and name.name not in ("__init__.py", "__main__.py"):
            untested.append(path)
    return sources, untested


def report(
    data: Path,
    *,
    sources: Sequence[str],
    floors: Mapping[str, float],
    scope: Literal["file", "directory"],
    owners: Mapping[str, Sequence[str]],
) -> bool:
    """Print the report for ``sources`` and whether any unit is below its floor.

    Args:
      data: The ``pytest`` gate's coverage data, its lines carrying per-test
        contexts.
      sources: Changed sources to floor, each under a prefix of ``floors``.
      floors: Least passing percent per path prefix; the longest match wins.
      scope: ``file`` floors each source; ``directory`` pools the sources of
        each directory, so a weak file passes when its changed siblings carry
        the directory over the floor.
      owners: Each source's own test files; only their lines, and the
        source's import, count.

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


# No file is a ``pytest`` gate that never ran. Under a ``SKIP`` naming it that is
# nothing to floor, and the tests are not run here; otherwise it is a floor run alone,
# which floors nothing and so must not pass. A file that is not coverage data is the gate's own line
# saying why it measured nothing -- it failed, crashed or recorded no coverage -- which
# must not pass.
def _report_run(
    data: Path,
    sources: Sequence[str],
    flags: Flags,
    floors: Mapping[str, float],
) -> bool:
    """Floor ``sources`` by their own tests' lines in the ``pytest`` gate's ``data``."""
    if not sources:
        return False
    if not data.is_file():
        if any(skipped(os.environ, token) for token in PYTEST_SKIP_TOKENS):
            sys.stdout.write(
                "coverage-floor: skipped with the pytest hook; nothing floored\n",
            )
            return False
        sys.stdout.write(
            "coverage-floor: the pytest hook did not run; run it with this one\n",
        )
        return True
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


# What a ``SKIP`` names the ``pytest`` hook by: its id, and its alias in
# ``.pre-commit-config.yaml``, which a test keeps in step.
PYTEST_SKIP_TOKENS: Final = ("pytest", "superseded-at-pre-push")

# Coverage data is an SQLite database; anything else is the gate's reason it has none.
_SQLITE: Final = b"SQLite format 3\x00"


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("paths", nargs="*", help="Changed repository-relative paths.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="The pytest gate's per-test coverage data; absent if it did not run.",
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
