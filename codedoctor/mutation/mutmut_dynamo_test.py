"""Tests for the Dynamo-traceable trampoline plugin.

Skipped where torch is absent: the plugin exists for projects that compile with
it, and one without it has nothing for Dynamo to trace.
"""

from __future__ import annotations

from importlib import util
from types import ModuleType
from typing import TYPE_CHECKING, cast

import os
import sys

from mutmut import configuration
from mutmut.mutation import trampoline
from mutmut.mutation.file_mutation import mutate_file_contents

import pytest

from codedoctor.mutation import mutmut_dynamo


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    import torch


pytestmark = pytest.mark.skipif(
    util.find_spec("torch") is None,
    reason="torch is not installed",
)


@pytest.fixture
def stats_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModuleType]:
    """Yield a mutated module whose trampolines record, as the stats phase runs it.

    A trampoline hit reads mutmut's config from the cwd, so the module's project is
    the cwd. Dynamo's caches are cleared after, so no compiled frame reaches the
    next test.
    """
    from torch import _dynamo  # noqa: PLC0415 -- optional; see the module docstring.

    source = tmp_path / "mod.py"
    source.write_text("def shift(x):\n    return x + 1\n")
    (tmp_path / "pyproject.toml").write_text('[tool.mutmut]\nsource_paths = ["."]\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        trampoline,
        "get_mutant_under_test",
        trampoline.get_mutant_under_test,
    )
    monkeypatch.setattr(trampoline, "_mutant_under_test", None)
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    configuration.reset_config()
    code = mutate_file_contents(str(source), source.read_text()).code
    module = ModuleType("pkg.mod")
    exec(compile(code, str(source), "exec"), module.__dict__)  # noqa: S102 -- mutmut's own output, as its snapshot imports it.
    yield module
    _dynamo.reset()
    configuration.reset_config()


def test_a_fullgraph_compile_traces_a_trampoline_while_stats_are_collected(
    stats_phase: ModuleType,
) -> None:
    from torch import _dynamo  # noqa: PLC0415 -- optional; see the module docstring.
    from torch._dynamo.exc import Unsupported  # noqa: PLC0415 -- optional; see the module docstring.

    import torch  # noqa: PLC0415 -- optional; see the module docstring.

    shift = cast("Callable[[torch.Tensor], torch.Tensor]", stats_phase.shift)

    # Without the plugin, mutmut's bookkeeping is what Dynamo cannot trace.
    with pytest.raises(Unsupported):
        torch.compile(shift, fullgraph=True, backend="eager")(torch.ones(2))
    _dynamo.reset()
    mutmut_dynamo.pytest_configure()

    assert torch.compile(shift, fullgraph=True, backend="eager")(
        torch.ones(2),
    ).tolist() == [2.0, 2.0]


def test_a_patched_is_compiling_does_not_hide_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch import compiler  # noqa: PLC0415 -- optional; see the module docstring.

    monkeypatch.setattr(
        trampoline,
        "get_mutant_under_test",
        trampoline.get_mutant_under_test,
    )
    monkeypatch.setattr(mutmut_dynamo, "_mutant_under_test", lambda: "stats")
    # Bound before the patch, as the plugin's import does once torch is loaded.
    monkeypatch.setattr(mutmut_dynamo, "_is_compiling", [compiler.is_compiling])
    monkeypatch.setattr(compiler, "is_compiling", lambda: True)
    mutmut_dynamo.pytest_configure()

    assert trampoline.get_mutant_under_test() == "stats"


def test_the_stats_phase_is_kept_from_a_child_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child records nothing, and finds no mutmut config from its own cwd."""
    monkeypatch.setattr(
        trampoline,
        "get_mutant_under_test",
        trampoline.get_mutant_under_test,
    )
    monkeypatch.setattr(trampoline, "_mutant_under_test", None)
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    mutmut_dynamo.pytest_configure()

    assert os.environ.get("MUTANT_UNDER_TEST") is None
    assert trampoline.get_mutant_under_test() == "stats"


def test_compiling_is_false_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mutmut_dynamo, "_is_compiling", [])
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert mutmut_dynamo._compiling() is False
    assert mutmut_dynamo._is_compiling == []
    assert "torch" not in sys.modules


def test_compiling_binds_torchs_own_check_at_the_first_lookup_that_finds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch import compiler  # noqa: PLC0415 -- optional; see the module docstring.

    monkeypatch.setattr(mutmut_dynamo, "_is_compiling", [])

    assert mutmut_dynamo._compiling() is False
    assert mutmut_dynamo._is_compiling == [compiler.is_compiling]


@pytest.mark.parametrize("name", ["", "fail", "loop.mod.x_f__mutmut_1"])
def test_a_child_still_runs_the_mutant_or_failure_under_test(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setattr(
        trampoline,
        "get_mutant_under_test",
        trampoline.get_mutant_under_test,
    )
    monkeypatch.setenv("MUTANT_UNDER_TEST", name)
    mutmut_dynamo.pytest_configure()

    assert os.environ.get("MUTANT_UNDER_TEST") == name


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
