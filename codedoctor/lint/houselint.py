# ruff: noqa: T201 -- CLI output is this command's user-facing product.
"""Lint Python files and notebooks against the house rules.

Paths work as in ruff: none means the current directory, a directory means
every .py, .pyi, and .ipynb file under it outside hidden directories and
.gitignore patterns, and a named file is linted as given. Flag defaults come
from [tool.houselint] in the nearest pyproject.toml; flags override them, and
--exclude and --set add to them.

Examples:
  houselint                                 lint the working tree
  houselint --fix ''                        report only, rewrite nothing
  houselint --select digits pkg/lib         one rule, one directory
  houselint --set digits.min-digits=6       change a rule setting
  houselint --exclude '*_test.py = globals nested-def'

Exit status: 0 clean, 1 findings remain, 2 usage error.

"""

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import Field, dataclass, fields, replace
from functools import partial
from itertools import compress
from multiprocessing.context import ForkContext
from pathlib import Path
from typing import (
    ClassVar,
    Final,
    NoReturn,
    Protocol,
    cast,
    get_origin,
    override,
    runtime_checkable,
)

import argparse
import ast
import re
import textwrap

from treekle import from_plain

from codedoctor.lib import pyproject_flags
from codedoctor.lib.files import walk
from codedoctor.lib.worker_count import capped_workers, default_workers, worker_count
from codedoctor.lint import house_ignore, houselint_rules
from codedoctor.lint.houselint_types import Finding, Unit


_DESCRIPTION: Final = (__doc__ or "").strip()

# The suffixes a directory argument contributes and the kind each reads as.
_KINDS: Final = {".py": "python", ".pyi": "pyi", ".ipynb": "jupyter"}


@dataclass(frozen=True, kw_only=True, slots=True)
class Exclusion:
    """One ``--exclude`` entry: paths matching ``globs`` skip ``rules``.

    Ruff's ``per-file-ignores`` shape: a path pattern on the left, the rules
    it silences on the right. An entry naming no rules excludes the path from
    every rule and never opens the file, which is ruff's ``exclude``.

    Attributes:
      globs: Compiled shell globs (``glob_regex``). A pattern with no ``/``
        matches a basename anywhere, as ``.gitignore`` reads one; ``**``
        crosses directories.
      rules: Names silenced on a match; empty means all of them.

    """

    globs: tuple[re.Pattern[str], ...]

    rules: frozenset[str]

    def matches(self, relative: str, /) -> bool:
        """Whether the repo-relative path matches any of ``globs``."""
        return any(glob.match(relative) for glob in self.globs)


class FixError(RuntimeError):
    """A rule's ``fix`` returned source that no longer parses: a houselint bug."""


@runtime_checkable
class Rule(Protocol):
    """A configurable houselint rule. Every member is required.

    Settings are the implementing dataclass's own fields, so they are typed,
    defaulted, and documented in one place. The four ClassVars below are
    metadata rather than fields precisely so ``dataclasses.fields`` yields
    only the settable knobs.
    """

    name: ClassVar[str]
    """Selector in ``--select``/``--fix``. Unique across the registry."""

    kinds: ClassVar[frozenset[str]]
    """File kinds read, from ``{"python", "pyi", "jupyter"}`` -- pre-commit's
    ``identify`` tags for ``.py``, ``.pyi``, and ``.ipynb``. A file of another
    kind is skipped before any view is built, so a notebook never reaches
    ``ast.parse``. A stub mirrors another package's signatures, so only a rule
    that cannot rename or restructure them declares ``pyi``."""

    fixable: ClassVar[bool]
    """Whether ``fix`` can rewrite. False makes ``--fix <name>`` an error
    instead of a silent no-op, which no inspection of ``fix`` could tell
    you."""

    __dataclass_fields__: ClassVar[dict[str, Field[object]]]
    """Every implementation is a dataclass, and this is what says so.

    ``dataclasses.fields`` and ``replace`` are typed against typeshed's
    ``DataclassInstance``, which a plain Protocol does not satisfy however
    many dataclasses implement it; declaring the attribute typeshed uses as
    its structural marker makes the protocol satisfy it directly. It also
    makes the requirement real: a non-dataclass Rule would have no settable
    settings, so ``--set <name>.x=1`` could never work.

    Spelled ``dict``, as typeshed does, not ``Mapping``: the attribute is
    matched invariantly, so a wider container makes every dataclass merely
    assignable, not a subtype. ``Field[object]`` stands in for typeshed's
    ``Field[Any]``, which the house bans.
    """

    def check(self, unit: Unit, /) -> Sequence[Finding]:
        """Report every violation in ``unit``. Must not write.

        Args:
          unit: Unit.

        Returns:
          result: The Sequence[Finding].

        """
        ...

    def fix(self, unit: Unit, /) -> str | None:
        """Return rewritten text, or None when nothing changes.

        Must be idempotent -- feeding its own output back returns None.
        ``run`` relies on that to be a single pass rather than a fixpoint
        loop, and ``houselint_test`` asserts it per rule. Output that no
        longer parses raises ``FixError`` and is never written.

        Args:
          unit: The file, with its kind, source, and lazily parsed views.

        Returns:
          rewritten: Fixed source, or None if no changes.

        """
        ...


def default_rules() -> tuple[Rule, ...]:
    """Every shipped rule.

    A function rather than a module constant so nothing at import time holds
    mutable state, and so a caller (or a test) can build its own registry
    without monkeypatching an attribute.

    The order is not a fix-application order: ``run`` applies fixes in
    ``active`` order, which ``houselint_test`` pins with two fixers of its
    own -- the guarantee lives there, where it can be checked.

    ``houselint_rules`` owns the registered rule implementations; importing
    it at module load time is safe because shared types live below both modules.

    Returns:
      rules: All registered rules in their testing order.

    """
    # pragma: no mutate start -- a cast is a no-op at runtime.
    return cast(
        tuple[Rule, ...],
        (
            houselint_rules.DataclassRule(),
            houselint_rules.StaticMethodRule(),
            houselint_rules.ConfigGlobalsRule(),
            houselint_rules.DigitsRule(),
            houselint_rules.EmDashRule(),
            houselint_rules.PytestFooterRule(),
            houselint_rules.PytestNamingRule(),
            houselint_rules.XdgLiteralRule(),
            houselint_rules.UserdirsNamespaceRule(),
            houselint_rules.UserdirsWrapperRule(),
            houselint_rules.CacheKwargRule(),
            houselint_rules.LazyImportBindingRule(),
            houselint_rules.TmpLiteralRule(),
            houselint_rules.TempfilePrefixRule(),
            houselint_rules.CwdFinalRule(),
            houselint_rules.ImportAliasRule(),
            houselint_rules.ImportHoistRule(),
            houselint_rules.DottedImportRule(),
            houselint_rules.NestedDefRule(),
            houselint_rules.GetLoggerNameRule(),
            houselint_rules.NestedTryRule(),
            houselint_rules.NegativeBoolParamRule(),
            houselint_rules.FluentNameRule(),
            houselint_rules.AllReexportRule(),
            houselint_rules.InitFacadeRule(),
            houselint_rules.DefOrderRule(),
            houselint_rules.ModuleSideEffectsRule(),
            houselint_rules.UnderscoreParamRule(),
            houselint_rules.NegatedCompareRule(),
            houselint_rules.NegatedTernaryRule(),
            houselint_rules.SuperOnlyRule(),
            houselint_rules.TensorValueGuardRule(),
            houselint_rules.MkdirModeRule(),
            houselint_rules.ReturnsNamedRule(),
            houselint_rules.ArgsSectionRule(),
            houselint_rules.ReturnsSectionRule(),
            houselint_rules.PrivateOnelinerRule(),
            houselint_rules.AttrBlankLineRule(),
            houselint_rules.AttrDocFormRule(),
            houselint_rules.CommentSentenceRule(),
            houselint_rules.CliShapeRule(),
            houselint_rules.SuppressionFormRule(),
            houselint_rules.FmtDirectiveRule(),
            houselint_rules.AnnotationImportRule(),
            houselint_rules.FinalInlineRule(),
            houselint_rules.SingleUseHelperRule(),
            houselint_rules.ForwarderHelperRule(),
            houselint_rules.LazyImportCostRule(),
            houselint_rules.AssertRule(),
            houselint_rules.TreekleTorchLoadRule(),
            houselint_rules.LoggingFstringRule(),
        ),
    )
    # pragma: no mutate end


def run(
    paths: Sequence[Path],
    *,
    active: Sequence[Rule],
    fixing: frozenset[str],
    exclude: Sequence[Exclusion] = (),
    root: Path = Path(),
    workers: int = 1,
) -> tuple[tuple[Finding, ...], tuple[Path, ...]]:
    """Fix, then check, each path.

    Fixes run first, in ``active`` order, and the Unit is rebuilt from the
    rewritten text before the next rule sees it. Otherwise a checker reads
    pre-fix source and reports a violation this same invocation just
    repaired -- which could not happen while each hook was its own process.

    Args:
      paths: Files to inspect, already expanded from any directories. A
        suffix naming no kind is skipped silently.
      active: Rules to run, already configured from the command line.
      fixing: Names whose ``fix`` may write. A subset of ``active`` names.
      exclude: Path globs and the rules each silences. An entry naming no
        rules skips the path for every rule without opening it. Entries
        naming rules are what let rules that differ only in what they skip
        share one process, one read, and one parse.
      root: Directory the globs are relative to. They match the path as
        seen from here, so ``pkg/lib/x.py`` behaves the same whether the
        caller passed a relative or an absolute path.
      workers: Cap on processes to spread the paths over. The work is
        pure-Python AST walking with nothing shared between files, so it
        scales on processes and not on threads. The pool is sized to the
        path count, so a one-file commit never pays for a pool it cannot
        use.

    Returns:
      findings: Unfixed violations, sorted by path then line.
      written: Paths whose bytes changed on disk, in ``paths`` order.

    """
    lint = partial(
        _lint_path,
        active=tuple(active),
        fixing=fixing,
        exclude=tuple(exclude),
        root=root,
    )
    # A worker costs ~6 ms to start plus a 40 ms import of the rules, so one
    # is worth starting only past ~10 files. Measured on the CLI, best of 3:
    # 5 files 0.16s serial vs 0.30s at 24 workers; 50 files 0.92s vs 0.50s.
    pool_size = capped_workers(workers, units=len(paths), per_worker=10)
    if pool_size <= 1:
        results = list(map(lint, paths))
    else:
        # ``fork``, not the default ``forkserver``: the server is exec'd
        # with the parent's ``sys.argv`` on its own command line, and an
        # ``--all-files`` run passes ~4000 paths -- past Linux's 128 KB
        # single-argument cap, so the server died at exec and the parent saw
        # ``BrokenPipeError``. Forking is safe here: this process has no
        # threads, and the children inherit the imported rules for free.
        with ProcessPoolExecutor(
            max_workers=pool_size,
            mp_context=ForkContext(),
        ) as pool:
            # Small chunks so a directory of large files does not pin one
            # worker while the rest sit idle; the path list is not sorted by
            # cost and a contiguous split by count is not a split by work.
            results = list(
                pool.map(lint, paths, chunksize=max(1, len(paths) // (pool_size * 8))),
            )
    findings = sorted(finding for found, _ in results for finding in found)
    return tuple(findings), tuple(compress(paths, (wrote for _, wrote in results)))


def read_unit(path: Path, /, *, reporter: str = "houselint") -> Unit | Finding | None:
    """Build a Unit carrying every view its kind admits, or say why it could not.

    Takes no view selection. A ``.py`` or ``.pyi`` file gets all three views;
    a notebook gets none, because it has no single module or token stream and
    no rule reads its raw JSON as lines. What a rule reads is therefore never
    declared anywhere, so nothing can drift from it.

    Args:
      path: File to read.
      reporter: Name to put on a Finding this function raises itself. The
        dispatcher is not a rule, but every Finding has to say what spoke.

    Returns:
      result: A Unit; a Finding when the bytes are not valid UTF-8; None
        when the suffix names no kind, which is a silent skip.

    """
    kind = _KINDS.get(path.suffix)
    if kind is None:
        return None
    try:
        raw = path.read_bytes()
    except OSError as err:
        # A path can vanish between staging and running, and pre-commit
        # passes what it selected earlier. A traceback here names no file
        # and abandons every path after it in the list.
        return Finding(
            path=path,
            line=1,
            rule=reporter,
            message=f"cannot read: {err.strerror or err}",
        )
    try:
        source = raw.decode()
    except UnicodeDecodeError as err:
        # Reported, never skipped. Decoding used to happen per-view with
        # ``errors="replace"`` while ``_tokenize`` silently returned no
        # tokens, which the digits rule read as "already grouped" -- the
        # file passed the hook without ever having been inspected.
        return Finding(
            path=path,
            line=1,
            rule=reporter,
            message=f"cannot decode as UTF-8: {err.reason} at byte {err.start}",
        )
    return Unit(path=path, source=source, kind=kind)


def main() -> int:
    """Fix and check the given files; exit 1 if any finding remains.

    Returns:
      exit_code: 0 if no findings, 1 otherwise.

    """
    rules = default_rules()
    # Help text is presentation, and a None for either False flag reads the
    # same; test_an_abbreviated_flag_is_rejected pins allow_abbrev itself.
    # pragma: no mutate start
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION,
        epilog=textwrap.fill(
            "rules (* can fix): "
            + ", ".join(f"{rule.name}{'*' * rule.fixable}" for rule in rules),
            width=79,
            subsequent_indent="  ",
            break_on_hyphens=False,
        ),
        formatter_class=_HelpFormatter,
        # Without this, argparse resolves any unambiguous PREFIX, so the
        # typo ``--digits-min-digit`` silently sets ``--digits-min-digits``
        # and a misspelled flag configures the rule instead of failing.
        allow_abbrev=False,
    )
    # pragma: no mutate end
    _add_arguments(parser)
    namespace, root = pyproject_flags.parse_args(parser, tool="houselint")
    # pragma: no mutate start -- a cast is a no-op at runtime.
    flags = cast(_Flags, namespace)
    # pragma: no mutate end
    try:
        active, fixing = select_rules(
            rules,
            select=flags.select,
            fix=flags.fix,
            settings=flags.settings,
        )
        exclude = tuple(
            parse_exclusion(spec, known={rule.name for rule in rules})
            for spec in flags.exclude
        )
        paths = walk.expand(flags.paths, suffixes=_KINDS)
    except ValueError as err:
        # ``parser.error`` prints usage and exits 2. Raising through would
        # print a traceback naming this file, which tells a hook author
        # nothing about which flag they got wrong.
        parser.error(str(err))
    findings, written = run(
        paths,
        active=active,
        fixing=fixing,
        exclude=exclude,
        root=root,
        workers=flags.workers or default_workers(),
    )
    for path in written:
        # The hooks this replaced each announced their rewrites. A silent
        # autofix leaves the author with a dirty tree and no reason for it.
        print(f"rewrote {path}")
    for finding in findings:
        print(finding)
    return 1 if findings else 0


def select_rules(
    rules: Sequence[Rule],
    /,
    *,
    select: str,
    fix: str,
    settings: Sequence[str],
) -> tuple[tuple[Rule, ...], frozenset[str]]:
    """Resolve the three list flags into configured rules and fix names.

    Every rejection names the segment that failed rather than the whole
    argument, because the caller's next move differs entirely between "no
    such rule", "no such field", and "that value is the wrong type".

    Args:
      rules: The registry, in fix-application order.
      select: Whitespace-separated rule names to run; ``ALL`` names every
        rule.
      fix: Whitespace-separated subset of ``select`` allowed to rewrite;
        ``ALL`` names every fixable rule in ``select``.
      settings: ``rule.field=value`` assignments, one per element.

    Returns:
      active: The selected rules, in registry order, carrying their
        settings.
      fixing: Names whose ``fix`` may write.

    Raises:
      ValueError: A name matches no rule, a fix target was not selected or
        cannot fix, or a setting names no field / will not coerce.

    """
    by_name = {rule.name: rule for rule in rules}
    chosen = _names(select)
    if "ALL" in chosen:
        chosen = set(by_name)
    unknown = sorted(chosen - by_name.keys())
    if unknown:
        raise ValueError(
            f"--select: unknown rule(s) {unknown}; known: {sorted(by_name)}",
        )
    if not chosen:
        raise ValueError(f"--select is required; known rules: {sorted(by_name)}")
    fixing = _names(fix)
    if "ALL" in fixing:
        fixing = {name for name in chosen if by_name[name].fixable}
    unselected = sorted(fixing - chosen)
    if unselected:
        raise ValueError(f"--fix: {unselected} not in --select")
    unfixable = sorted(name for name in fixing if not by_name[name].fixable)
    if unfixable:
        raise ValueError(f"--fix: {unfixable} cannot fix; no autofix for that rule")
    overrides = _settings_by_rule(settings, selected=chosen, known=set(by_name))
    active = tuple(
        replace(rule, **_coerced(rule, overrides.get(rule.name, {})))
        for rule in rules
        if rule.name in chosen
    )
    return active, frozenset(fixing)


def parse_exclusion(spec: str, /, *, known: set[str]) -> Exclusion:
    """Parse one ``--exclude`` value, ``GLOB... [= RULE...]``.

    The ``=`` splits globs from rules, so a glob may hold neither ``=`` nor
    whitespace; none in this tree does. ``glob_regex`` gives the globs
    ``pathlib`` semantics -- ``*`` stops at ``/``, ``**`` does not -- and a
    glob without a ``/`` is anchored at every directory, so ``conftest.py``
    names each one.

    Args:
      spec: The flag value.
      known: Registered rule names, for rejecting a typo on the right side.

    Returns:
      exclusion: The compiled entry.

    Raises:
      ValueError: No glob, or a rule the registry does not know.

    """
    globs_part, sep, rules_part = spec.partition("=")
    globs = _names(globs_part)
    if not globs:
        raise ValueError(f"--exclude {spec!r}: expected GLOBS [= RULES]")
    rules = _names(rules_part)
    if sep and not rules:
        raise ValueError(f"--exclude {spec!r}: '=' with no rule after it")
    unknown = sorted(rules - known)
    if unknown:
        raise ValueError(f"--exclude {spec!r}: unknown rule(s) {unknown}")
    return Exclusion(
        globs=tuple(
            glob_regex(pattern if "/" in pattern else f"**/{pattern}")
            for pattern in sorted(globs)
        ),
        rules=frozenset(rules),
    )


# Not ``glob.translate``: it arrived in 3.13, and the exported packages validate
# on 3.12 (AGENTS.md Python Versions). Unlike it, ``[...]`` here is literal text.
def glob_regex(pattern: str, /) -> re.Pattern[str]:
    """Compile a ``pathlib``-style glob: ``*`` stops at ``/``, ``**`` does not.

    Args:
      pattern: The glob, with ``/`` separators.

    Returns:
      regex: Matches the whole path.

    """
    # ``.*/`` rather than ``(?:[^/]+/)*``: a path outside the config's root
    # stays absolute, and its leading ``/`` is an empty first segment.
    wildcards = {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}
    return re.compile(
        "".join(
            wildcards.get(piece[0]) or re.escape(piece[0])
            for piece in re.finditer(r"\*\*/|\*\*|\*|\?|[^*?]+", pattern)
        )
        + r"\Z",
    )


def _names(value: str) -> set[str]:
    """Split a whitespace-separated list; a rule name or glob holds no space."""
    return set(value.split())


def _settings_by_rule(
    settings: Sequence[str],
    /,
    *,
    selected: set[str],
    known: set[str],
) -> dict[str, dict[str, str]]:
    """Group ``rule.field=value`` assignments under their rule."""
    grouped: dict[str, dict[str, str]] = {}
    for assignment in settings:
        key, sep, value = assignment.partition("=")
        if not sep:
            raise ValueError(f"--set {assignment!r}: expected RULE.FIELD=VALUE")
        rule_name, dot, field_name = key.strip().partition(".")
        if not dot:
            raise ValueError(f"--set {key!r}: expected RULE.FIELD, with a dot")
        if rule_name not in known:
            raise ValueError(f"--set {key!r}: unknown rule {rule_name!r}")
        if rule_name not in selected:
            raise ValueError(f"--set {key!r}: {rule_name!r} is not in --select")
        grouped.setdefault(rule_name, {})[field_name.replace("-", "_")] = value
    return grouped


def _coerced(rule: Rule, overrides: Mapping[str, str]) -> dict[str, object]:
    """Coerce one rule's raw setting strings to their declared field types."""
    by_field = {f.name: f for f in fields(rule)}
    unknown = sorted(overrides.keys() - by_field.keys())
    if unknown:
        raise ValueError(
            f"--set: {rule.name} has no field(s) {unknown}; valid: {sorted(by_field)}",
        )
    coerced: dict[str, object] = {}
    for name, raw in overrides.items():
        annotation = by_field[name].type
        # A container value is one token holding several items, so it is
        # split HERE, against a field already known to be a container --
        # never against a value whose type is still unknown.
        given: object = (
            (raw.split())
            if (get_origin(annotation) in (tuple, list, set, frozenset))
            else raw
        )
        try:
            coerced[name] = from_plain(given, annotation, strict=False)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"--set {rule.name}.{name}: cannot read {raw!r} as "
                f"{_type_name(annotation)} ({err})",
            ) from err
    return coerced


def _relative(path: Path, root: Path) -> str:
    """Return ``path`` as seen from ``root``, or unchanged if it is elsewhere."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _type_name(annotation: object) -> str:
    """Render a field's declared type for ``--help``."""
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


# ``--select`` and ``--fix`` both take a whitespace-separated list of RULE NAMES, and a
# rule name cannot contain whitespace, so that separator is never ambiguous -- unlike a
# settings value, which can contain anything. Settings therefore never share a token
# with anything else: one ``--set`` carries exactly one ``rule.field=value``, and only a
# value whose field is declared a container is split, on whitespace, after the fact.
#
# The key is a path, so it routes without a per-rule or per-field flag. An earlier shape
# gave every field its own ``--<rule>-<field>``, which made the CLI surface grow with
# the registry and stuttered (``--digits-min-digits``); the one before that packed
# settings into the selector's own value (``--check-digits=fix,min-digits=5``), split on
# ``,`` before parsing JSON so a list was unexpressible, and decided whether its value
# was a spec or a path with ``Path(spec).exists()`` -- so a file named ``fix`` in the
# working directory changed what the command meant.
#
# There is no ``--deselect``: with nothing on by default, disabling a rule is not naming
# it.
def _add_arguments(parser: argparse.ArgumentParser, /) -> None:
    """Register the flags. Their number does not grow with the registry."""
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path()],
        help="files and directories to lint",
    )
    parser.add_argument(
        "--select",
        default="",
        metavar="RULES",
        help="space-separated rules to run, or ALL",
    )
    parser.add_argument(
        "--fix",
        default="",
        metavar="RULES",
        help="space-separated selected rules allowed to rewrite files, or ALL",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="settings",
        metavar="RULE.FIELD=VALUE",
        help="set one rule setting; repeatable",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOBS[=RULES]",
        help=(
            "skip files matching GLOBS for every rule, or only for RULES; "
            "repeatable; a glob without '/' matches a basename anywhere"
        ),
    )
    parser.add_argument(
        "--workers",
        type=worker_count,
        help="processes to lint with; empty or below 1 means 1",
    )


def _lint_path(
    path: Path,
    /,
    *,
    active: tuple[Rule, ...],
    fixing: frozenset[str],
    exclude: tuple[Exclusion, ...],
    root: Path,
) -> tuple[tuple[Finding, ...], bool]:
    """Fix, then check, one path; the unit of work a pool hands to a worker."""
    relative = _relative(path, root)
    silenced: set[str] = set()
    for entry in exclude:
        if not entry.matches(relative):
            continue
        if not entry.rules:
            return (), False
        silenced |= entry.rules
    applies = tuple(rule for rule in active if rule.name not in silenced)
    if not applies:
        return (), False
    built = read_unit(path)
    if built is None:
        return (), False
    if isinstance(built, Finding):
        return (built,), False
    unit = built
    findings: list[Finding] = []
    # Check results on the CURRENT unit, reused by the reporting pass below;
    # a rewrite invalidates them all, since any rule may see the new text.
    checked: dict[str, Sequence[Finding]] = {}
    for rule in applies:
        if rule.name not in fixing or unit.kind not in rule.kinds:
            continue
        # A fixer never rewrites what its own check passes (the rules tests
        # assert it per fixture), and most files are clean, so the check
        # gates the fix. Measured: fix cost 10-50x check on clean files. A
        # finding a house-ignore marker suppresses does not open the gate:
        # the rewrite would leave that marker pointing at nothing.
        checked[rule.name] = rule.check(unit)
        if not house_ignore.apply(unit, checked[rule.name], active=frozenset()):
            continue
        rewritten = rule.fix(unit)
        if rewritten is None:
            continue
        candidate = Unit(path=path, source=rewritten, kind=unit.kind)
        # Candidate first: later rules read its tree anyway, and the original
        # is parsed only to excuse a file that never parsed.
        if candidate.tree is None and unit.tree is not None:
            _raise_fix_error(rule.name, path, rewritten)
        checked.clear()
        try:
            path.write_bytes(rewritten.encode())
        except OSError as err:
            # Recorded, and the remaining paths still run. A raised
            # PermissionError here abandoned every file after this one
            # with no record of which had already been rewritten.
            findings.append(
                Finding(
                    path=path,
                    line=1,
                    rule="houselint",
                    message=f"cannot write: {err.strerror or err}",
                ),
            )
            break
        # The candidate is built from what the fixer RETURNED, not by
        # re-reading: the bytes are in hand, and the parse is already paid.
        unit = candidate
    # A marker on a line of its own covers the FILE -- for a file whose SHAPE
    # violates a rule, as a jupytext percent notebook is module scope by
    # construction. Deliberately rare: an ordinary module fixes its findings or
    # marks each statement. It lives in the source, not in a config, so the
    # ``.ipynb`` twin carries it too and jupytext round-trips it.
    findings.extend(house_ignore.marker_findings(unit))
    ran = [rule for rule in applies if unit.kind in rule.kinds]
    findings.extend(
        house_ignore.apply(
            unit,
            (
                finding
                for rule in ran
                for finding in (
                    checked[rule.name] if rule.name in checked else rule.check(unit)
                )
            ),
            active=frozenset(rule.name for rule in ran),
        ),
    )
    # A rewrite replaces ``unit``, and only a successful write does.
    return tuple(findings), unit is not built


# Raised, not reported: a finding blames the user's file for the linter's bug, and
# writing the output would leave a file that will not import. Nothing has been
# written for this path yet, so the file on disk is still the user's.
def _raise_fix_error(rule: str, path: Path, rewritten: str) -> NoReturn:
    """Raise ``FixError`` chained to the ``SyntaxError`` in ``rewritten``."""
    try:
        ast.parse(rewritten, filename=str(path))
    except SyntaxError as err:
        raise FixError(
            f"{rule} fix produced unparseable source for {path}; "
            f"fix the rule in houselint_rules.py ({err.msg}, line {err.lineno})",
        ) from err
    raise AssertionError(rewritten)


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Append each flag's effective default, after pyproject.toml applied."""

    @override
    def _get_help_string(self, action: argparse.Action) -> str | None:
        default = cast(object, action.default)
        if default is argparse.SUPPRESS:
            return action.help
        if default is None:
            # The one unset default, `--workers`, is sized when the run starts.
            return f"{action.help} (default: the machine's count)"
        values = cast(list[object], default) if isinstance(default, list) else [default]
        # Table entries may span lines; `%` would be read as a format spec.
        shown = "; ".join(" ".join(str(value).split()) for value in values) or "none"
        return f"{action.help} (default: {shown.replace('%', '%%')})"


class _Flags(Protocol):
    select: str
    fix: str
    settings: list[str]
    exclude: list[str]
    paths: list[Path]
    workers: int | None
