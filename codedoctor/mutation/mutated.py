# ruff: noqa: T201 -- CLI output is this command's user-facing product.
"""Mutation-test the functions in the given files.

Paths work as in ruff: none means the current directory, a directory means
every .py file under it outside hidden directories and .gitignore patterns,
and a named file is mutated as given. Mutates every function under
[tool.mutmut] source_paths by default; --function-sample-rate keeps a seeded
fraction of them, rounded up, and --max-functions-sampled caps that count.
Mutmut skips decorated functions other than class and static methods.
--prefer-changed BASE samples the functions changed since commit BASE before
any other; without it, every function is as likely as any.
The project root is the nearest
pyproject.toml; flag defaults come from [tool.mutated] in the nearest
pyproject.toml that has one, and flags override them.

Each file is mutated alone, in its own snapshot against its own paired
tests, so a broken test or crash costs only that file. Files run concurrently
within --workers: one mutmut parent dispatches and records every mutant on a
single core, so more parents, not more children per parent, raise
throughput. A line is written as each file finishes, and every surviving
mutant is reported with its diff.

Findings are advice: they are printed and the run exits 0, unless --blocking
is given or PRECOMMIT_MUTATION_BLOCKING holds a positive integer; then any
finding exits 1. A run that cannot select functions or lay out its snapshots
exits 1 either way.

Examples:
  mutated
  mutated pkg/lib
  mutated pkg/serialize.py
  mutated --explain pkg/serialize.py
  mutated --emit filters pkg/serialize.py
  mutated --blocking pkg/serialize.py

"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import chdir
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from multiprocessing.context import ForkContext
from pathlib import Path
from typing import Final, Protocol, cast
from unicodedata import normalize

import argparse
import ast
import bisect
import contextlib
import hashlib
import io
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import tokenize
import tomllib

from libcst import ClassDef, FunctionDef
from mutmut import configuration
from mutmut.configuration import config, reset_config
from mutmut.mutation.data import SourceFileMutationData
from mutmut.mutation.diff_apply import (
    get_diff_for_mutant,
    read_functions_from_index,
    read_mutant_function,
    read_original_function,
)
from mutmut.mutation.file_mutation import create_mutations
from mutmut.mutation.trampoline_templates import (
    CLASS_NAME_SEPARATOR,
    mangle_function_name,
)
from mutmut.stats import status_by_exit_code
from mutmut.utils.format_utils import get_mutant_name
from tomlkit.items import Table
from treekle import ReadError, from_plain

import libcst as cst
import tomlkit

from codedoctor.affected.affected import harvest
from codedoctor.gate.gate import positive
from codedoctor.lib import pyproject_flags
from codedoctor.lib.files import tempclaim, walk
from codedoctor.lib.kernel_pins import kernel_pins
from codedoctor.lib.worker_count import capped_workers, default_workers, worker_count


_DESCRIPTION: Final = (__doc__ or "").strip()


def main() -> int:
    """Mutate-test the functions in the named files.

    Returns:
      status: Nonzero when the run could not start, or when blocking is on and
        it found failed or inconclusive mutations; zero otherwise.

    """
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    flags, root = _parse_args(parser)
    seed, workers = flags.seed, flags.workers or default_workers()
    try:
        named = _project_relative(
            walk.expand(flags.paths, suffixes={".py"}),
            root=root,
        )
        preferred = (
            frozenset[str]()
            if flags.prefer_changed is None
            else changed_functions(root, base=flags.prefer_changed, files=named)
        )
        filters, candidates = select_functions(
            root,
            files=named,
            sources=mutable_sources(root),
            seed=seed,
            rate=flags.function_sample_rate,
            most=flags.max_functions_sampled,
            workers=workers,
            preferred=preferred,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.stderr.write(f"mutation: cannot select functions: {error}\n")
        return 1
    if skipped := candidates - len(filters):
        sys.stderr.write(
            f"[mutated] {len(filters)} selected; {skipped} skipped (seed {seed!r})\n",
        )
    if flags.explain:
        sys.stderr.write(
            f"[mutated] {len(named)} file(s) -> {len(filters)} function(s)\n",
        )
        for mutant_filter in filters:
            sys.stderr.write(f"  {mutant_filter}\n")
    if flags.emit == "filters":
        for mutant_filter in filters:
            print(mutant_filter)
        return 0
    if not filters:
        return 0
    shards = plan_shards(_filters_by_source(filters, files=named))
    # Measured on 1178 mutants: one mutmut parent records 34 mutants/s with 8
    # children and 37/s with 64. Past 8 children, only more parents help, so
    # the budget is split across shards first: 24 workers ran 4 shards 3 at a
    # time at 8 children each, the fourth waiting for a whole shard's run.
    children = max(1, min(8, workers // len(shards)))
    parallel = min(len(shards), max(1, workers // children))
    if flags.explain:
        sys.stderr.write(
            f"[mutated] {len(shards)} file(s), {parallel} at a time x "
            f"{children} worker(s)\n",
        )
    try:
        reports = _run_shards(
            root,
            shards=shards,
            children=children,
            parallel=parallel,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        sys.stderr.write(f"mutation: cannot prepare the snapshot: {error}\n")
        return 1
    found = _print_reports(reports)
    if found and not flags.blocking:
        print(
            "mutation: advisory, so nothing above fails this run; "
            "PRECOMMIT_MUTATION_BLOCKING=1 makes it fail.",
        )
        return 0
    return found


def mutable_sources(root: Path) -> tuple[str, ...]:
    """Return the ``[tool.mutmut] source_paths`` entries, repo-relative.

    Args:
      root: Repository root containing the configuration.

    Returns:
      sources: Configured paths, or empty when mutmut is unconfigured.

    """
    config = root / "pyproject.toml"
    if not config.is_file():
        return ()
    with config.open("rb") as handle:
        table = from_plain(tomllib.load(handle), dict[str, object])
    for name in ("tool", "mutmut"):
        table = from_plain(table.get(name), dict[str, object], default={})
    raw = table.get("source_paths", [])
    # One ``convert`` for both shapes: a non-list and a non-string entry are
    # the same typo, not "no source". Narrowing with ``isinstance`` first would
    # hand ``convert`` a ``list[Unknown]`` and lose the element type.
    try:
        return tuple(from_plain(raw, list[str]))
    except ReadError as error:
        raise TypeError(
            f"[tool.mutmut] source_paths must be strings: {raw!r}",
        ) from error


def function_filters(
    root: Path,
    *,
    files: list[str],
    sources: tuple[str, ...],
    workers: int = 1,
) -> tuple[str, ...]:
    """Return one mutmut filter per mutable function in ``files``.

    Args:
      root: Project root, the directory holding ``pyproject.toml``.
      files: Root-relative paths to mutate.
      sources: Configured ``source_paths``; a file outside them is skipped.
      workers: Cap on processes to parse the files over.

    Returns:
      filters: Sorted patterns ending in ``__mutmut_*`` per function:
        ``<module>.x_<function>`` or ``<module>.xǁ<Class>ǁ<method>``.

    """
    root = root.resolve()
    # Mutmut's stats phase reads this global on every trampoline hit, so a run
    # inside a mutmut session must hand back the configuration it found.
    previous = configuration._config  # noqa: SLF001 -- mutmut exposes no way to restore its configuration.
    with chdir(root):
        reset_config()
        try:
            return _filters(root, files=files, sources=sources, workers=workers)
        finally:
            configuration._config = previous  # noqa: SLF001 -- mutmut exposes no way to restore its configuration.


def changed_functions(root: Path, *, base: str, files: list[str]) -> frozenset[str]:
    """Return a filter for each function in ``files`` whose lines differ from ``base``.

    A file git does not track is new, so every function in it differs.

    Args:
      root: Project root, inside a git checkout.
      base: The commit to compare the working tree with.
      files: Root-relative paths to look in.

    Returns:
      filters: One ``__mutmut_*`` pattern per changed function.

    Raises:
      subprocess.CalledProcessError: Git could not compare against ``base``.

    """
    names = [path for path in files if path.endswith(".py")]
    if not names:
        return frozenset[str]()
    diff = subprocess.run(  # noqa: S603 -- git from PATH; derived arguments.
        ["git", "diff", "--unified=0", "--no-color", base, "--", *names],  # noqa: S607 -- git from PATH.
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked = subprocess.run(  # noqa: S603 -- git from PATH; derived arguments.
        ["git", "ls-files", "--", *names],  # noqa: S607 -- git from PATH.
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    changed: defaultdict[str, list[range]] = defaultdict(list)
    path = ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line.removeprefix("+++ b/") if line != "+++ /dev/null" else ""
        elif line.startswith("@@") and path:
            start, _, count = line.split()[2].removeprefix("+").partition(",")
            first = int(start)
            # A deletion (count 0) changed the line it sits between.
            changed[path].append(range(first, first + max(int(count or 1), 1)))
    for untracked in set(names) - set(tracked):
        changed[untracked].append(range(1, 1 << 31))
    return frozenset(
        get_mutant_name(Path(path), mutant_method_name=f"{name}__mutmut_*")
        for path, spans in changed.items()
        if (root / path).is_file()
        for name, lines in _function_lines(root / path)
        if any(span.start < lines.stop and lines.start < span.stop for span in spans)
    )


def select_functions(
    root: Path,
    *,
    files: list[str],
    sources: tuple[str, ...],
    seed: str,
    rate: float,
    most: int | None,
    workers: int = 1,
    preferred: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], int]:
    """Return ``sample(function_filters(...))`` and how many functions it drew from.

    Mutmut's own check of a file, which ``function_filters`` runs on every
    named file, costs ~0.15 s per file: 24 s of a 37 s ``--all-files`` run that
    sampled 4 of 32,150 functions. Under a cap, functions are instead ranked
    by the same seeded hash from a parse alone, and mutmut checks only the
    files that ranking reaches; the sample is the same one.

    Args:
      root: Project root, the directory holding ``pyproject.toml``.
      files: Root-relative paths to mutate.
      sources: Configured ``source_paths``; a file outside them is skipped.
      seed: Varies the sample between changes.
      rate: Fraction of the mutable functions to keep, rounded up.
      most: Most functions to keep, or None for no cap.
      workers: Cap on processes to parse the files over.
      preferred: Filters ranked ahead of every other, such as the functions a
        change touched; the seed orders each group.

    Returns:
      filters: The sampled filters, sorted.
      candidates: Functions sampled from: every mutable one when all were
        checked, else every function the parse found, which mutmut's check
        only ever narrows.

    """
    check = partial(function_filters, root, sources=sources, workers=workers)
    if most is None:
        found = check(files=files)
        size = sample_size(len(found), rate=rate, most=most)
        return sample(found, seed=seed, size=size, preferred=preferred), len(found)
    by_file = _declared_by_file(root, files=files, sources=sources, workers=workers)
    owner = {name: path for path, names in by_file.items() for name in names}
    ranked = sorted(owner, key=partial(_rank, seed=seed, preferred=preferred))
    # The least count of mutable functions whose sample is ``most`` long.
    enough = math.floor((most - 1) / Fraction(str(rate))) + 1
    order = list(dict.fromkeys(owner[name] for name in ranked))
    position = {path: index for index, path in enumerate(order)}
    checked = 0
    mutable: set[str] = set()
    kept: list[str] = []
    for name in ranked:
        if position[owner[name]] >= checked:
            batch = order[checked : checked + max(1, workers)]
            checked += len(batch)
            mutable.update(check(files=batch))
        if name in mutable:
            kept.append(name)
        if len(kept) >= most and len(mutable) >= enough:
            return tuple(sorted(kept[:most])), len(ranked)
    size = sample_size(len(mutable), rate=rate, most=most)
    return tuple(sorted(kept[:size])), len(mutable)


def sample_size(candidates: int, *, rate: float, most: int | None) -> int:
    """Return ``min(most, ceil(rate * candidates))``; None means no cap.

    The rate is read as the decimal it was written as, so 0.1 of 30 is 3,
    not the 4 that binary float rounding up would give.

    Args:
      candidates: Functions available to sample.
      rate: Fraction of them to keep.
      most: Most functions to keep, or None for no cap.

    Returns:
      size: How many functions to sample.

    """
    size = math.ceil(Fraction(str(rate)) * candidates)
    return size if most is None else min(most, size)


def sample(
    filters: tuple[str, ...],
    *,
    seed: str,
    size: int,
    preferred: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Keep the ``size`` filters whose seeded hashes are lowest (bottom-k).

    Every function is equally likely whatever its module path, so many files
    are not always sampled from their alphabetically first corner. A later
    run displaces a pick only when a newly named function hashes lower;
    ``random.sample`` would redraw the whole sample instead.

    Args:
      filters: Candidate filters.
      seed: Varies the sample between changes; CI passes the PR number.
      size: Most filters to keep.
      preferred: Filters kept before any other.

    Returns:
      sample: The kept filters, sorted.

    Raises:
      ValueError: The size is negative.

    References:
      https://doi.org/10.1145/1281100.1281133
        Cohen and Kaplan. Summarizing data using bottom-k sketches. PODC 2007.

    """
    if size < 0:
        raise ValueError("sample size must be nonnegative")
    ranked = sorted(filters, key=partial(_rank, seed=seed, preferred=preferred))
    return tuple(sorted(ranked[:size]))


@dataclass(frozen=True, slots=True, kw_only=True)
class Shard:
    """One source file, mutated alone in its own snapshot against its own tests.

    A file is the unit of failure: a broken paired test, or a crash, costs
    that file's results and no other's.
    """

    source: str
    filters: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ShardReport:
    """What one shard found.

    Attributes:
      source: The file mutated.
      outcomes: Mutant names, or filters for ``unchecked``, by status.
      diffs: Each surviving mutant's diff against its source.
      checked: Mutants with a recorded result, killed or not.
      failure: Why the shard recorded no results; empty when it did.

    """

    source: str
    outcomes: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict[str, tuple[str, ...]],
    )
    diffs: Mapping[str, str] = field(default_factory=dict[str, str])
    checked: int = 0
    failure: str = ""


def plan_shards(filters_by_source: Mapping[str, tuple[str, ...]]) -> tuple[Shard, ...]:
    """Return one shard per file, most functions first.

    A long file started last finishes last, after every other core idled.

    Args:
      filters_by_source: Selected filters, keyed by the source they mutate.

    Returns:
      shards: One per source, its filters sorted.

    """
    return tuple(
        Shard(source=source, filters=tuple(sorted(names)))
        for source, names in sorted(
            filters_by_source.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    )


def run_shard(
    root: Path,
    *,
    shard: Shard,
    snapshot: Path,
    template: Path,
    workers: int,
) -> ShardReport:
    """Mutate ``shard`` in ``snapshot``, collect every outcome, then delete it.

    Args:
      root: Project root.
      shard: The file and filters to mutate.
      snapshot: Missing directory to run in; removed before returning.
      template: The project as laid out by ``copy_template``.
      workers: Mutants this shard runs concurrently.

    Returns:
      report: Outcomes and survivor diffs, or why there are none.

    """
    try:
        return _mutate(
            root,
            shard=shard,
            snapshot=snapshot,
            template=template,
            workers=workers,
        )
    except subprocess.TimeoutExpired as error:
        return ShardReport(
            source=shard.source,
            failure=f"mutation: {shard.source}: mutmut ran past its "
            f"{error.timeout:g} s limit",
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return ShardReport(
            source=shard.source,
            failure=f"mutation: {shard.source}: {type(error).__name__}: {error}",
        )
    finally:
        shutil.rmtree(snapshot, ignore_errors=True)


def copy_template(root: Path, *, template: Path) -> None:
    """Copy the project into ``template`` once, for every shard to clone.

    The project is what git lists: every tracked file, plus untracked ones
    ``.gitignore`` does not match, such as a module not yet committed. Caches,
    virtualenvs, and runtime output stay out. Symlinks are copied as links.

    Args:
      root: Project root, inside a git work tree.
      template: Missing directory to fill.

    """
    # Not ``walk``: tests read tracked dotfiles (``.pre-commit-config.yaml``,
    # ``.github/``) and force-added files under ignored paths, and a file
    # missing here fails the shard's stats run before any mutant is checked.
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],  # noqa: S607 -- git from PATH.
        cwd=root,
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    ).stdout
    for name in listing.split("\0"):
        source = root / name
        # The index still lists a tracked file deleted from the work tree.
        if not name or not (source.is_file() or source.is_symlink()):
            continue
        destination = template / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst=destination, follow_symlinks=False)


def copy_snapshot(
    root: Path,
    *,
    snapshot: Path,
    sources: tuple[str, ...],
    tests: tuple[str, ...],
    template: Path,
    filters: tuple[str, ...] = (),
) -> None:
    """Lay out ``snapshot`` so mutmut mutates ``sources`` and runs ``tests``.

    Mutmut copies ``source_paths`` into ``mutants/`` and runs the tests from
    there. Every other project file goes straight into ``mutants/``: the
    project's packages are regular packages, so a module missing beside the
    mutated one cannot be imported from the checkout instead.

    Args:
      root: Project root.
      snapshot: Missing directory to fill.
      sources: Root-relative modules to mutate.
      tests: Pytest node ids that exercise ``sources``.
      template: The project as laid out by ``copy_template``.
      filters: Selected ``__mutmut_*`` patterns; every other function in
        ``sources`` is pragma-skipped. Empty mutates every function.

    """
    snapshot.mkdir(parents=True)
    # `cp` holds no GIL: concurrent shards copying file by file in Python
    # threads serialized, 24 s for eight shards of a 16k-file checkout.
    subprocess.run(  # noqa: S603 -- Fixed executable, derived paths.
        [shutil.which("cp") or "cp", "-a", str(template), str(snapshot / "mutants")],
        check=True,
    )
    for name in sources:
        (snapshot / "mutants" / name).unlink()
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dst=snapshot / name)
        if filters:
            _skip_unselected(snapshot / name, name=name, filters=filters)
    config = tomlkit.parse((root / "pyproject.toml").read_text())
    tool = cast(Table, config["tool"])
    settings = cast(Table, tool["mutmut"])
    settings["source_paths"] = list(sources)
    settings["pytest_add_cli_args_test_selection"] = list(tests)
    plain = cast(Mapping[str, object], settings.unwrap())
    settings["pytest_add_cli_args"] = [
        *_strings(plain.get("pytest_add_cli_args", [])),
        "-p",
        "codedoctor.mutation.mutmut_dynamo",
        "-p",
        "codedoctor.mutation.mutmut_param_ids",
        # py_pglite marks every test whose path says "benchmark" `performance`,
        # which the default ``-m`` deselects, leaving such modules untestable.
        "-p",
        "no:py_pglite",
        # Installed by dependencies, never used here, and each pytest start
        # imported zarr, anndata, pandas, scipy, and numba for them: one
        # mutmut shard went 10.8s -> 5.6s with these and torch dropped.
        "-p",
        "no:fast_array_utils",
        "-p",
        "no:zarr",
        "-p",
        "no:hydra_pytest",
    ]
    settings["also_copy"] = []
    settings["mutate_only_covered_lines"] = False
    # ``forkserver`` re-imports pytest, conftest and torch in a fresh child for
    # stats, clean, forced-fail and warm-up alike; ``fork`` imports once. With
    # identical verdicts: custom_json 65 s -> 28 s, priml embedding 300 s -> 36 s.
    settings["process_isolation"] = "fork"
    # Reporting omits tools/, but mutation targets there still need measured lines.
    coverage = cast(Table, tool.setdefault("coverage", tomlkit.table()))
    run = cast(Table, coverage.setdefault("run", tomlkit.table()))
    run["omit"] = []
    # Mutmut reads its config from the root and pytest reads it from mutants/.
    for directory in (snapshot, snapshot / "mutants"):
        (directory / "pyproject.toml").write_text(tomlkit.dumps(config))


def single_threaded_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Return ``environ`` with every math library pinned to one thread.

    Shards already fill the cores; a torch test left at its default thread
    count oversubscribes them, measured at 0.14 s alone and 48 s beside the
    other shards, past pytest's 60 s timeout once mutmut's stats tracing
    adds its own cost.

    Args:
      environ: The environment to copy.

    Returns:
      env: A copy whose OpenMP, MKL, and OpenBLAS thread counts are 1, with
        each ``kernel_pins`` entry ``environ`` does not set.

    """
    return {
        **kernel_pins(),
        **environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def fork_safe_env(
    environ: Mapping[str, str],
    *,
    platform: str = sys.platform,
) -> dict[str, str]:
    """Return ``environ`` with every request off the machine sent to a dead proxy.

    Mutmut forks one child per mutant from a process that has already run the
    tests, and on macOS the system code a request reaches in that child is not
    fork-safe, so the child dies and the mutant is reported inconclusive:

    - Reading the system proxy settings through SystemConfiguration dies with
      SIGSEGV, as every mutant whose tests build knowbuddy's Discord client
      did. ``urllib`` reads them only when no ``*_proxy`` variable is set.
    - A mutant that drops a test's fake transport sends a real request, and
      checking its certificate through ``truststore`` and the Security
      framework dies with SIGABRT, though the tests would kill that mutant.

    Each proxy variable names a port on this machine nothing listens on, so
    such a request is refused before it resolves a name or starts TLS, and the
    mutant fails its tests as it would anywhere else. ``localhost`` stays
    direct for tests that serve on it. A proxy the environment names is
    replaced: requests through it would reach the Security framework too.

    Args:
      environ: The environment to copy.
      platform: The ``sys.platform`` to build it for.

    Returns:
      env: A copy whose proxy variables, on macOS, name the dead proxy.

    """
    if platform != "darwin":
        return dict(environ)
    dead = "http://127.0.0.1:9"
    proxies = {
        "http_proxy": dead,
        "https_proxy": dead,
        "all_proxy": dead,
        "no_proxy": "localhost,127.0.0.1,::1",
    }
    return {
        **{
            name: value
            for name, value in environ.items()
            if not name.lower().endswith("_proxy")
        },
        **proxies,
        **{name.upper(): value for name, value in proxies.items()},
    }


def mutant_outcomes(
    exit_codes: Mapping[str, int | None],
    *,
    filters: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Group the selected mutants that need attention by why.

    Args:
      exit_codes: Mutmut's recorded exit code per mutant; ``None`` is unrun.
      filters: Selected ``__mutmut_*`` patterns.

    Returns:
      outcomes: Mutant names under ``survived``, ``no tests``, and
        ``inconclusive``; filter patterns under ``unchecked`` when no mutant
        of that function ran. Killed and timed-out mutants are omitted.

    """
    prefixes = _prefixes(filters)
    outcomes: dict[str, list[str]] = {
        "no tests": [],
        "inconclusive": [],
        "survived": [],
        "unchecked": [],
    }
    for name, code in sorted(exit_codes.items()):
        status = status_by_exit_code[code]
        if not name.startswith(prefixes) or status in {
            "killed",
            "timeout",
            "caught by type check",
        }:
            continue
        if status in outcomes:
            outcomes[status].append(name)
        else:
            outcomes["inconclusive"].append(f"{name} ({status}, exit {code})")
    outcomes["unchecked"] = [
        pattern
        for pattern in filters
        if not any(
            name.startswith(pattern.removesuffix("*")) and code is not None
            for name, code in exit_codes.items()
        )
    ]
    return {status: tuple(names) for status, names in outcomes.items()}


class _Flags(Protocol):
    paths: list[Path]
    seed: str
    prefer_changed: str | None
    function_sample_rate: float
    max_functions_sampled: int | None
    workers: int | None
    emit: str
    explain: bool
    blocking: bool


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> tuple[_Flags, Path]:
    """Parse ``argv`` over ``[tool.mutated]``; return the flags and project root."""
    _add_arguments(parser)
    namespace, root = pyproject_flags.parse_args(parser, tool="mutated", argv=argv)
    flags = cast(_Flags, namespace)
    rate = flags.function_sample_rate
    if math.isnan(rate) or rate <= 0 or rate > 1:
        parser.error("--function-sample-rate must be in (0, 1]")
    if flags.max_functions_sampled is not None and flags.max_functions_sampled < 1:
        parser.error("--max-functions-sampled must be positive")
    # Without a [tool.mutated] table the root is the nearest pyproject.toml,
    # which holds the [tool.mutmut] settings.
    root = next(
        (
            directory
            for directory in (root, *root.absolute().parents)
            if (directory / "pyproject.toml").is_file()
        ),
        root,
    )
    return flags, root


def _project_relative(files: Sequence[Path], *, root: Path) -> list[str]:
    """Rewrite cwd-relative ``files`` relative to ``root``."""
    base = root.resolve()
    # Resolve the directory, not the file: a symlinked source keeps its name.
    return [
        (name.absolute().parent.resolve() / name.name).relative_to(base).as_posix()
        for name in files
    ]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path()],
        help="Files and directories to mutate.",
    )
    parser.add_argument(
        "--seed",
        default=os.environ.get("PRECOMMIT_MUTATION_SEED", ""),
        help="Sampling seed (default: PRECOMMIT_MUTATION_SEED, or empty).",
    )
    parser.add_argument(
        "--prefer-changed",
        metavar="BASE",
        help="Sample the functions changed since commit BASE first (default: none).",
    )
    parser.add_argument(
        "--function-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of functions to mutate, rounded up; in (0, 1].",
    )
    parser.add_argument(
        "--max-functions-sampled",
        type=int,
        help="Most functions to mutate (default: no cap).",
    )
    parser.add_argument(
        "--workers",
        type=worker_count,
        help="Mutants run concurrently; empty or below 1 means 1.",
    )
    parser.add_argument(
        "--emit",
        choices=("run", "filters"),
        default="run",
        help="Run mutations, or only print the selected filters.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Write the selected functions to stderr.",
    )
    parser.add_argument(
        "--blocking",
        action="store_true",
        default=bool(positive(os.environ.get("PRECOMMIT_MUTATION_BLOCKING", ""))),
        help="Exit 1 on any finding (default: PRECOMMIT_MUTATION_BLOCKING, or off).",
    )


def _filters_by_source(
    filters: tuple[str, ...],
    *,
    files: list[str],
) -> dict[str, tuple[str, ...]]:
    """Group ``filters`` under the named source file each one mutates."""
    source_of = {
        get_mutant_name(Path(source), mutant_method_name="").removesuffix("."): (
            Path(source).as_posix()
        )
        for source in files
        if source.endswith(".py")
    }
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for name in filters:
        grouped[source_of[name.rsplit(".", 1)[0]]].append(name)
    return {source: tuple(names) for source, names in grouped.items()}


def _run_shards(
    root: Path,
    *,
    shards: tuple[Shard, ...],
    children: int,
    parallel: int,
) -> list[ShardReport]:
    """Run ``parallel`` shards at a time; report each one as it finishes."""
    # Mutmut writes `mutants/` into its working directory, so each shard runs
    # in a disposable copy rather than the checkout. A claim, not a
    # TemporaryDirectory: a pre-commit timeout kills this process, and each
    # orphaned copy of the checkout costs ~37k inodes until the next run.
    with (
        tempclaim.claimed_directory(
            Path(tempfile.gettempdir()),
            prefix="mutated-",
        ) as temporary,
        ThreadPoolExecutor(max_workers=parallel) as pool,
    ):
        # Coverage resolves symlinks; mutmut uses lexical paths. Run in the
        # physical directory so covered-line lookups use matching filenames.
        base = temporary.resolve()
        copy_template(root, template=base / "template")
        futures = [
            pool.submit(
                run_shard,
                root,
                shard=shard,
                snapshot=base / str(index),
                template=base / "template",
                workers=children,
            )
            for index, shard in enumerate(shards)
        ]
        reports: list[ShardReport] = []
        for report in map(Future[ShardReport].result, as_completed(futures)):
            reports.append(report)
            _progress(report, done=len(reports), total=len(shards))
        return reports


def _progress(report: ShardReport, *, done: int, total: int) -> None:
    """Write one line per finished shard, and its failure at once."""
    counts = ", ".join(
        f"{len(names)} {status}" for status, names in report.outcomes.items() if names
    )
    status = "FAILED" if report.failure else f"{report.checked} checked"
    sys.stderr.write(
        f"[mutated] {done}/{total} {report.source}: {status}"
        f"{'; ' + counts if counts else ''}\n",
    )
    if report.failure:
        sys.stderr.write(f"{report.failure}\n")


def _mutate(
    root: Path,
    *,
    shard: Shard,
    snapshot: Path,
    template: Path,
    workers: int,
) -> ShardReport:
    """Lay out ``snapshot``, run mutmut there, and read back what it recorded."""
    tests = _paired_tests(root, sources=(shard.source,))
    if not tests:
        return ShardReport(
            source=shard.source,
            failure=f"mutation: {shard.source}: no paired tests found",
        )
    # One line per broken test file, before mutmut starts: otherwise pytest's
    # 30-line collection traceback stands in for a SyntaxError ruff reported.
    if broken := _unparsable(root, tests=tests):
        return ShardReport(
            source=shard.source,
            failure="\n".join(f"mutation: {shard.source}: {e}" for e in broken),
        )
    copy_snapshot(
        root,
        snapshot=snapshot,
        sources=(shard.source,),
        tests=tests,
        template=template,
        filters=shard.filters,
    )
    # Wildcards select mutant keys; mutmut also matches them against bare
    # function keys for clean tests. Exact names take its suffix-stripping
    # branch, so include one representative per function for that lookup.
    selectors = (*shard.filters, *(f.removesuffix("*") + "1" for f in shard.filters))
    completed = _run_mutmut(
        [
            sys.executable,
            "-m",
            "mutmut",
            "run",
            "--max-children",
            str(workers),
            *selectors,
        ],
        cwd=snapshot,
        env=fork_safe_env(single_threaded_env(os.environ)),
    )
    if completed.returncode:
        return ShardReport(
            source=shard.source,
            failure=f"{_tail(completed.stdout + completed.stderr)}\n"
            f"mutation: {shard.source}: mutmut exited {completed.returncode} "
            "before recording results; its output is above.",
        )
    _retry_inconclusive(snapshot, shard=shard)
    return _collect(snapshot, shard=shard)


def _unparsable(root: Path, *, tests: tuple[str, ...]) -> list[str]:
    """Return ``file:line: message`` for each test file Python cannot parse."""
    errors: list[str] = []
    for name in dict.fromkeys(test.split("::")[0] for test in tests):
        try:
            ast.parse((root / name).read_bytes())
        except FileNotFoundError:
            continue
        except SyntaxError as error:
            errors.append(f"{name}:{error.lineno}: {error.msg}")
    return errors


def _retry_inconclusive(snapshot: Path, *, shard: Shard) -> None:
    """Retry selected segfaulted and unrun mutants serially once."""
    data = SourceFileMutationData(path=snapshot / "mutants" / shard.source)
    data.load()
    names = tuple(
        name
        for name, code in sorted(data.exit_code_by_key.items())
        if name.startswith(_prefixes(shard.filters))
        and (code is None or status_by_exit_code[code] == "segfault")
    )
    if names:
        _ = _run_mutmut(
            [sys.executable, "-m", "mutmut", "run", "--max-children", "1", *names],
            cwd=snapshot,
            env=fork_safe_env(single_threaded_env(os.environ)),
        )


# Its own session, so a run past its limit is killed with every worker it
# forked: killing mutmut alone would leave them testing, and holding its pipes.
def _run_mutmut(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_sec: float = 900.0,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` to its end, or kill it and all it started after ``timeout_sec``."""
    with subprocess.Popen(  # noqa: S603 -- fixed executable, derived arguments.
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            out, err = process.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            _ = process.communicate()
            raise
    return subprocess.CompletedProcess(argv, process.returncode, out, err)


# Spinner frames are dropped. A failed stats run dumps pytest's whole log before
# mutmut's own messages, so the ``FAILED``/``ERROR`` lines naming the cause would
# otherwise fall outside the tail.
def _tail(output: str, *, lines: int = 40) -> str:
    """Return pytest's failure lines, then the last ``lines`` lines of ``output``."""
    # Mutmut redraws a spinner with carriage returns; each frame is one line.
    kept = [
        line
        for line in re.split(r"[\r\n]+", output)
        if line.strip() and line.lstrip()[:1] not in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    ]
    tail = kept[-lines:]
    causes = [
        line
        for line in kept[: len(kept) - len(tail)]
        if line.startswith(("FAILED ", "ERROR ", "E   "))
    ]
    return "\n".join([*causes, *tail])


def _print_reports(reports: list[ShardReport]) -> int:
    """Print every finding of every shard; return nonzero if there is one."""
    if failed := sorted(r.source for r in reports if r.failure):
        print("mutation: these files recorded no results; each reason is above:")
        print("".join(f"  {source}\n" for source in failed), end="")
    merged = {
        status: sorted(name for r in reports for name in r.outcomes.get(status, ()))
        for status in ("no tests", "inconclusive", "survived", "unchecked")
    }
    for status, heading in (
        ("no tests", "no test exercised these mutants, so nothing was checked:"),
        ("inconclusive", "these mutants have inconclusive results:"),
        ("survived", "these variants survived the selected tests:"),
        ("unchecked", "no mutant was checked for these selected functions:"),
    ):
        if merged[status]:
            print(f"mutation: {heading}")
            print("".join(f"  {name}\n" for name in merged[status]), end="")
    if merged["no tests"]:
        print(
            "\nCheck test coverage of these functions first. Missing test "
            "selection or sandbox imports can also prevent coverage.",
        )
    diffs = {name: diff for r in reports for name, diff in r.diffs.items()}
    for name in merged["survived"]:
        print(f"# {name}: survived\n{diffs[name]}\n")
    return int(any(r.failure for r in reports) or any(merged.values()))


def _paired_tests(root: Path, *, sources: tuple[str, ...]) -> tuple[str, ...]:
    """Return the node ids of the tests paired with ``sources``."""
    # In-process: depth 0 opens no index, and a subprocess per file paid a
    # whole interpreter start for a handful of ``is_file`` probes.
    with chdir(root):
        return tuple(
            harvest(
                Path(".codegraph/codegraph.db"),
                changed=list(sources),
                depth=0,
                emit="nodeids",
            ),
        )


def _collect(snapshot: Path, *, shard: Shard) -> ShardReport:
    """Read the outcomes mutmut recorded in ``snapshot`` and each survivor's diff."""
    # An absolute path replaces mutmut's cwd-relative `mutants/` prefix.
    data = SourceFileMutationData(path=snapshot / "mutants" / shard.source)
    data.load()
    outcomes = mutant_outcomes(data.exit_code_by_key, filters=shard.filters)
    mutated = snapshot / "mutants" / shard.source
    diffs = {
        name: _diff(snapshot, name=name, source=shard.source)
        for name in outcomes["survived"]
        if not _only_unobservable(*_functions(name, path=mutated))
    }
    return ShardReport(
        source=shard.source,
        outcomes={**outcomes, "survived": tuple(diffs)},
        diffs=diffs,
        checked=sum(
            code is not None and name.startswith(_prefixes(shard.filters))
            for name, code in data.exit_code_by_key.items()
        ),
    )


def _prefixes(filters: tuple[str, ...]) -> tuple[str, ...]:
    """Return the mutant-name prefixes that ``filters`` select."""
    return tuple(name.removesuffix("*") for name in filters)


def _diff(snapshot: Path, *, name: str, source: str) -> str:
    """Return mutant ``name``'s diff, read from ``snapshot`` without a subprocess."""
    mutated = snapshot / "mutants" / source
    # An absolute path replaces mutmut's cwd-relative `mutants/` prefix; the
    # header then names the snapshot, which is deleted, so name the source.
    return get_diff_for_mutant(name, path=mutated).replace(str(mutated), source)


# ``typing.cast`` returns its value untouched, and argparse ``help=`` text is
# presentation, so no test can kill a mutant confined to either. The mutmut
# config skips lines holding only ``help=``; this covers shared lines. In
# ``(doc or "").split("\n", 2)[2]`` the fallback is read only when ``doc`` is
# falsy, and then every fallback the index rejects raises the same IndexError.
def _functions(name: str, *, path: Path) -> tuple[str, str]:
    """Return the source of mutant ``name``'s original function and of the mutant."""
    functions = read_functions_from_index(name, path)
    if functions is None:
        module = cst.parse_module(path.read_text(encoding="utf-8"))
        functions = (
            read_original_function(module, name),
            read_mutant_function(module, name),
        )
    original, mutant = (cst.Module([function]).code for function in functions)
    return original, mutant


def _only_unobservable(original: str, mutant: str) -> bool:
    """Return whether ``mutant`` differs only in casts, help, flags, or fallbacks."""
    return original != mutant and _erase_unobservable(original) == _erase_unobservable(
        mutant,
    )


# ``usedforsecurity`` only declares a hashlib digest's purpose (FIPS builds
# check it); the bytes it returns are the same either way, so no test can see it.
_UNOBSERVABLE_FLAGS: Final = frozenset({"usedforsecurity"})


def _erase_unobservable(code: str) -> str:
    """Dump ``code`` with casts, ``help=`` values, flags, and raising fallbacks blanked."""
    tree = ast.parse(textwrap.dedent(code))
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and (fallback := _raising_fallback(node)):
            fallback.values[-1] = ast.Constant(value=None)
        if not isinstance(node, ast.Call):
            continue
        if _callee(node) == "cast" and node.args:
            node.args[0] = ast.Constant(value=None)
        node.keywords = [k for k in node.keywords if k.arg not in _UNOBSERVABLE_FLAGS]
        for keyword in node.keywords:
            if keyword.arg == "help" and isinstance(keyword.value, ast.Constant):
                keyword.value = ast.Constant(value=None)
    return ast.dump(tree)


def _raising_fallback(node: ast.Subscript) -> ast.BoolOp | None:
    """Return ``x or "s"`` in ``(x or "s").split(...)[i]`` if ``"s"`` raises there."""
    call = node.value
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "split"
        and isinstance(either := call.func.value, ast.BoolOp)
        and isinstance(either.op, ast.Or)
        and isinstance(fallback := either.values[-1], ast.Constant)
        and isinstance(fallback.value, str)
        and not call.keywords
        and isinstance(node.slice, ast.Constant)
        and isinstance(index := node.slice.value, int)
    ):
        return None
    match call.args:
        case [ast.Constant(value=str(sep)), ast.Constant(value=int(maxsplit))] if sep:
            parts = fallback.value.split(sep, maxsplit)
        case [ast.Constant(value=str(sep))] if sep:
            parts = fallback.value.split(sep)
        case _:
            return None
    return None if -len(parts) <= index < len(parts) else either


def _callee(call: ast.Call) -> str:
    """Return the bare name a call invokes: ``cast`` for ``typing.cast(...)``."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _strings(value: object) -> tuple[str, ...]:
    """Validate an argparse or TOML list of strings at its untyped boundary."""
    assert isinstance(value, list)
    result: list[str] = []
    for item in cast("list[object]", value):
        assert isinstance(item, str)
        result.append(item)
    return tuple(result)


def _filters(
    root: Path,
    *,
    files: list[str],
    sources: tuple[str, ...],
    workers: int,
) -> tuple[str, ...]:
    """Select the functions in ``files`` under the project's mutation settings."""
    # Mutmut renders every mutant of a file to list its functions, ~0.75 s per
    # file: a 76-file library took 57 s serially.
    found = _map_files(
        partial(_file_filters, root=root),
        _mutable_paths(root, files=files, sources=sources),
        workers=workers,
    )
    return tuple(sorted({name for names in found for name in names}))


def _declared_by_file(
    root: Path,
    *,
    files: list[str],
    sources: tuple[str, ...],
    workers: int,
) -> dict[str, tuple[str, ...]]:
    """Return, per mutable path, a filter for every function its parse declares."""
    root = root.resolve()
    previous = configuration._config  # noqa: SLF001 -- mutmut exposes no way to restore its configuration.
    with chdir(root):
        reset_config()
        try:
            paths = _mutable_paths(root, files=files, sources=sources)
            found = _map_files(partial(_declared, root=root), paths, workers=workers)
        finally:
            configuration._config = previous  # noqa: SLF001 -- mutmut exposes no way to restore its configuration.
    return dict(zip(paths, found, strict=True))


def _declared(path: str, *, root: Path) -> tuple[str, ...]:
    """Return a filter per top-level function and method ``path`` declares."""
    try:
        tree = ast.parse((root / path).read_bytes())
    except (OSError, SyntaxError):
        return ()
    names = [
        f"x_{top.name}"
        if isinstance(top, ast.FunctionDef | ast.AsyncFunctionDef)
        else f"x{CLASS_NAME_SEPARATOR}{top.name}{CLASS_NAME_SEPARATOR}{member.name}"
        for top in tree.body
        if isinstance(top, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for member in (top.body if isinstance(top, ast.ClassDef) else [top])
        if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    return tuple(
        get_mutant_name(Path(path), mutant_method_name=f"{name}__mutmut_*")
        for name in names
    )


def _mutable_paths(
    root: Path,
    *,
    files: list[str],
    sources: tuple[str, ...],
) -> list[str]:
    """Return the named ``.py`` sources the mutation settings allow, deduplicated."""
    return [
        path
        for path in dict.fromkeys(Path(named).as_posix() for named in files)
        if path.endswith(".py")
        and not _is_test(path)
        and not any(
            part in {"third_party", "typings", ".export"} for part in Path(path).parts
        )
        and any(Path(path).is_relative_to(Path(source)) for source in sources)
        and (root / path).is_file()
        and config().should_mutate(path)
    ]


def _map_files[T](
    work: Callable[[str], T],
    paths: list[str],
    *,
    workers: int,
) -> list[T]:
    """Return ``work`` over ``paths`` in order, forked over up to ``workers``."""
    pool_size = capped_workers(workers, units=len(paths))
    if pool_size < 2:
        return list(map(work, paths))
    # ``fork`` as in houselint: children inherit the loaded mutmut
    # configuration and the root cwd, which they read rather than reload.
    with ProcessPoolExecutor(max_workers=pool_size, mp_context=ForkContext()) as pool:
        return list(
            pool.map(work, paths, chunksize=max(1, len(paths) // pool_size // 8)),
        )


def _rank(
    name: str,
    *,
    seed: str,
    preferred: frozenset[str] = frozenset(),
) -> tuple[bool, bytes]:
    """Return ``name``'s sampling key: preferred first, then by seeded hash."""
    return name not in preferred, hashlib.sha256(f"{seed}\0{name}".encode()).digest()


def _function_lines(path: Path) -> list[tuple[str, range]]:
    """Return each function mutmut names in ``path``, with the lines it spans."""
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError):
        return []
    found: list[tuple[str, range]] = []
    for top in tree.body:
        members = top.body if isinstance(top, ast.ClassDef) else [top]
        for member in members:
            if not isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            name = (
                f"x{CLASS_NAME_SEPARATOR}{top.name}{CLASS_NAME_SEPARATOR}{member.name}"
                if isinstance(top, ast.ClassDef)
                else f"x_{member.name}"
            )
            first = min([member.lineno, *(d.lineno for d in member.decorator_list)])
            found.append((name, range(first, (member.end_lineno or member.lineno) + 1)))
    return found


def _file_filters(path: str, *, root: Path) -> tuple[str, ...]:
    """Return the filters for the mutable functions in root-relative ``path``."""
    return tuple(
        get_mutant_name(Path(path), mutant_method_name=f"{name}__mutmut_*")
        for name in _functions_in(root / path)
    )


# Mutmut generates every mutant of a mutated file and imports them all on each
# test run; a 7k-line module took 10s to render 7,154 mutants for 2 selected
# functions. A trailing comment keeps line numbers, which mutmut's coverage
# lookup and diffs depend on.
def _skip_unselected(path: Path, *, name: str, filters: tuple[str, ...]) -> None:
    """Append ``# pragma: no mutate block`` to each unselected def's header line."""
    module = get_mutant_name(Path(name), mutant_method_name="").removesuffix(".")
    selected = {
        mangled
        for pattern in filters
        if pattern.startswith(f"{module}.")
        for mangled in [pattern.removesuffix("__mutmut_*").removeprefix(f"{module}.")]
    }
    # Bytes, not text: text mode would translate a CRLF source to LF.
    source = path.read_bytes().decode()
    # ``splitlines`` also breaks at U+2028 and form feeds, which Python's
    # tokenizer does not count, and the line numbers below are the tokenizer's.
    lines = source.split("\n")
    # The pragma must trail the header's colon, which a multi-line signature
    # moves below the ``def`` and a comment can separate from the body.
    ends = sorted(
        token.end[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NEWLINE
    )
    for top in ast.parse(source).body:
        if isinstance(top, ast.FunctionDef | ast.AsyncFunctionDef):
            defs = [(f"x_{top.name}", top)]
        elif isinstance(top, ast.ClassDef):
            prefix = f"x{CLASS_NAME_SEPARATOR}{top.name}{CLASS_NAME_SEPARATOR}"
            defs = [
                (f"{prefix}{member.name}", member)
                for member in top.body
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
        else:
            continue
        for mangled, node in defs:
            if mangled in selected:
                continue
            index = ends[bisect.bisect_left(ends, node.lineno)] - 1
            text = lines[index].removesuffix("\r")
            joiner = " " if "#" in text else "  "
            lines[index] = (
                f"{text}{joiner}# pragma: no mutate block{lines[index][len(text) :]}"
            )
    path.write_bytes("\n".join(lines).encode())


def _is_test(path: str) -> bool:
    """Return whether ``path`` is a test module rather than a source one."""
    name = Path(path).name
    return (
        name == "conftest.py" or name.endswith("_test.py") or name.startswith("test_")
    )


# Mutmut names only top-level functions and direct methods. Closures belong
# to their enclosing function; inventing nested names aborts mutant selection.
def _functions_in(path: Path) -> tuple[str, ...]:
    """Return mutmut's mangled names for the mutable functions in ``path``."""
    try:
        source = path.read_bytes().decode()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return ()
    mutable = _mutable_function_names(path, source=source)
    kernels = _kernel_bodies(tree)
    found: set[str] = set()
    for top in tree.body:
        if isinstance(top, ast.FunctionDef | ast.AsyncFunctionDef):
            if (
                f"x_{top.name}" in mutable
                and not _is_declaration(top)
                and top.name not in kernels
            ):
                found.add(f"x_{top.name}")
        elif isinstance(top, ast.ClassDef):
            prefix = f"x{CLASS_NAME_SEPARATOR}{top.name}{CLASS_NAME_SEPARATOR}"
            found.update(
                f"{prefix}{member.name}"
                for member in top.body
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
                and f"{prefix}{member.name}" in mutable
                and not _is_declaration(member)
            )
    return tuple(sorted(found))


# Triton compiles a kernel from its source text, so a trampoline cannot stand in
# for it: mutmut can neither switch a kernel's mutants nor leave it runnable.
# Kernels and the device helpers bound into them are written against
# ``triton.language`` types, which nothing else annotates with.
def _kernel_bodies(tree: ast.Module) -> set[str]:
    """Return the top-level functions with a ``language.*``-annotated parameter."""
    return {
        top.name
        for top in tree.body
        if isinstance(top, ast.FunctionDef)
        and any(
            argument.annotation is not None
            and re.search(
                r"\blanguage\.(?:tensor|tuple|constexpr)\b",
                ast.unparse(argument.annotation),
            )
            for argument in (*top.args.posonlyargs, *top.args.args)
        )
    }


# A stub's only mutants are its parameter defaults, which no call can observe.
def _is_declaration(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether ``function`` is a body-less stub (docstring, ``...``, ``pass``)."""
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return all(
        isinstance(statement, ast.Pass)
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is Ellipsis
        )
        for statement in body
    )


def _mutable_function_names(path: Path, *, source: str) -> set[str]:
    """Enumerate mutmut's generating functions without rendering mutant source."""
    module, mutations, _, _ = create_mutations(filename=str(path), code=source)
    functions = {mutation.contained_by_top_level_function for mutation in mutations}
    names: set[str] = set()
    for statement in module.body:
        if isinstance(statement, FunctionDef) and statement in functions:
            names.add(_mutant_function_name(path, name=statement.name.value))
        elif isinstance(statement, ClassDef):
            names.update(
                _mutant_function_name(
                    path,
                    name=member.name.value,
                    class_name=statement.name.value,
                )
                for member in statement.body.children
                if isinstance(member, FunctionDef) and member in functions
            )
    return names


def _mutant_function_name(
    path: Path,
    *,
    name: str,
    class_name: str = "",
) -> str:
    """Validate names against mutmut's mangling and runtime lookup rules."""
    try:
        mangled = mangle_function_name(name=name, class_name=class_name)
    except AssertionError:
        mangled = None
    if mangled is None or normalize("NFKC", mangled) != mangled:
        qualified = f"{class_name}.{name}" if class_name else name
        raise ValueError(f"{path}: mutmut does not support identifier {qualified!r}")
    return mangled
