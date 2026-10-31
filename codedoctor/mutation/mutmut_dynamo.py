"""Pytest plugin keeping mutmut's ``stats`` phase where it can be recorded.

Mutmut's ``stats`` phase records which test reaches which function by reading
and setting a ``ContextVar`` inside every trampoline. Dynamo cannot trace
``ContextVar.get``, so a ``fullgraph=True`` compile of any mutated function
raises ``Unsupported`` and the whole file records no results. While Dynamo
traces, the trampoline is told no mutant is active, so it calls the original
function and skips the bookkeeping; the eager calls a test makes still map it.
Mutant runs are unchanged: their trampolines trace.

Nor does the phase reach a Python child a test starts: there each trampoline
hit reads mutmut's config from the child's cwd, finds none, and fails the
whole file. The phase stays in this process alone; a mutant's name, or
``fail``, still reaches the child, which runs it as this process does.

``mutated.py`` loads this with ``-p`` in each snapshot; nothing else should.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import sys

from mutmut.mutation import trampoline


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import compiler
else:
    from wrapt import lazy_import

    compiler = lazy_import("torch", "compiler")


_mutant_under_test = trampoline.get_mutant_under_test
# Never imported here: torch costs ~1s per pytest start, and nothing compiles
# until the code under test imports it. Bound at the first lookup that finds
# it, so a test that later patches ``torch.compiler.is_compiling`` to exercise
# a compile-only branch cannot also hide that branch's trampolines from
# ``stats``.
_is_compiling: list[Callable[[], bool]] = []


# Never mutated: a trampoline wrapped around the lookup every trampoline calls
# recurses until the stats phase dies.
def pytest_configure() -> None:  # pragma: no mutate block
    """Route every trampoline through the Dynamo-traceable lookup; keep ``stats`` here."""
    trampoline.get_mutant_under_test = _traceable_mutant_under_test  # ty: ignore[invalid-assignment] -- Monkeypatching mutmut's module function is this plugin's purpose.
    # The lookup falls back to the process-local copy once the variable is gone.
    if os.environ.get("MUTANT_UNDER_TEST") == "stats":
        trampoline.set_mutant_under_test("stats")
        del os.environ["MUTANT_UNDER_TEST"]


def _traceable_mutant_under_test() -> str:  # pragma: no mutate block
    """Return the active mutant, hiding the ``stats`` phase from Dynamo."""
    name = _mutant_under_test()
    return "" if name == "stats" and _compiling() else name


def _compiling() -> bool:  # pragma: no mutate block
    """Return whether Dynamo is tracing; ``False`` while torch is unimported."""
    if not _is_compiling:
        if "torch" not in sys.modules:
            return False
        _is_compiling.append(compiler.is_compiling)
    return _is_compiling[0]()


# A process that imported torch first (a test module, a conftest) binds now.
_compiling()
