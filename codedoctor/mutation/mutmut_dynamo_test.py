"""Tests for the Dynamo-traceable trampoline plugin.

Skipped where torch is absent: the plugin exists for projects that compile with
it, and one without it has nothing for Dynamo to trace.
"""

from __future__ import annotations

from importlib import util
from pathlib import Path
from typing import Final

import os
import subprocess
import sys

from mutmut.mutation import trampoline

import pytest

from codedoctor.mutation import mutmut_dynamo


_CWD: Final = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(
    util.find_spec("torch") is None,
    reason="torch is not installed",
)


@pytest.mark.cli_python_subprocess
def test_a_fullgraph_compile_traces_a_trampoline_while_stats_are_collected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mod.py"
    source.write_text("def shift(x):\n    return x + 1\n")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib.util, os, sys, torch\n"
        "from mutmut.mutation.file_mutation import mutate_file_contents\n"
        "if sys.argv[1] == 'plugin':\n"
        "    import codedoctor.mutation.mutmut_dynamo as plugin\n"
        "    plugin.pytest_configure()\n"
        "path = sys.argv[2]\n"
        "code = mutate_file_contents(path, open(path).read()).code\n"
        "module = type(sys)('pkg.mod')\n"
        "exec(compile(code, path, 'exec'), module.__dict__)\n"
        "os.environ['MUTANT_UNDER_TEST'] = 'stats'\n"
        "compiled = torch.compile(module.shift, fullgraph=True, backend='eager')\n"
        "try:\n"
        "    print(compiled(torch.ones(2)).tolist())\n"
        "except Exception as error:\n"
        "    print(type(error).__name__)\n",
    )

    def run(mode: str) -> str:
        return subprocess.run(  # noqa: S603 -- Fixed interpreter and probe.
            [sys.executable, str(probe), mode, str(source)],
            capture_output=True,
            check=True,
            cwd=_CWD.parents[2],
            text=True,
        ).stdout.strip()

    assert run("bare") == "Unsupported"
    assert run("plugin") == "[2.0, 2.0]"


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
