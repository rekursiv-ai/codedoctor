r"""Pytest plugin keeping parametrize ids stable across in-process sessions.

Mutmut runs ``pytest.main`` several times in one process: stats, the clean
run, forced fail. Pytest caches the ids it generates for a parametrize mark on
the mark itself, and the test module stays imported, so the next session takes
those cached ids as ``ids=[...]`` and escapes them again: ``b'\\x00'`` gains a
backslash per session. The node ids the stats run recorded then match nothing,
the clean run exits 4 ("not found"), and the file records no results. Each
session here drops the cache once per mark before pytest reads it, so ids are
generated fresh, exactly as in a session of their own.

``mutated.py`` loads this with ``-p`` in each snapshot; nothing else should.
"""

from __future__ import annotations

import pytest


_SEEN = pytest.StashKey[set[int]]()


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Forget ids an earlier session generated, once per mark per session.

    Args:
      metafunc: The test function being parametrized.

    """
    seen = metafunc.config.stash.setdefault(_SEEN, set())
    for mark in metafunc.definition.iter_markers("parametrize"):
        source = mark._param_ids_from  # noqa: SLF001 -- pytest exposes no way to reset its id cache.
        # Within one session the cache is pytest's own: a mark shared by a
        # class's methods must keep the ids its first use generated.
        if source is not None and id(source) not in seen:
            seen.add(id(source))
            object.__setattr__(source, "_param_ids_generated", None)
