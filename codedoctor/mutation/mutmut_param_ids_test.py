"""Tests for the plugin keeping parametrize ids stable across sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import subprocess
import sys

import pytest


_CWD: Final = Path(__file__).resolve().parent

_MODULE = """\
import pytest


@pytest.mark.parametrize("v", [b"\\x00\\xff", "plain"], ids=repr)
def test_v(v):
    pass


@pytest.mark.parametrize("w", [b"\\x01"], ids=repr)
class TestShared:
    def test_a(self, w):
        pass

    def test_b(self, w):
        pass
"""

# Mutmut's sequence in one process: collect, then run the ids it recorded.
_PROBE = """\
import sys, pytest

class Ids:
    def __init__(self):
        self.ids = []

    def pytest_collection_modifyitems(self, items):
        self.ids = [item.nodeid for item in items]

plugins = sys.argv[1:]
first = Ids()
pytest.main(["-q", "--co", "-p", "no:cacheprovider", *plugins, "probe_test.py"], plugins=[first])
for _ in range(2):
    code = int(pytest.main(["-q", "-p", "no:cacheprovider", *plugins, *first.ids]))
    print("exit", code)
"""


@pytest.mark.cli_python_subprocess
@pytest.mark.parametrize(
    ("plugins", "exits"),
    [
        ([], ["exit 4", "exit 4"]),
        (["-p", "codedoctor.mutation.mutmut_param_ids"], ["exit 0", "exit 0"]),
    ],
    ids=["bare", "plugin"],
)
def test_ids_from_one_session_select_tests_in_the_next(
    tmp_path: Path,
    plugins: list[str],
    exits: list[str],
) -> None:
    """Without the plugin pytest re-escapes cached ids and selects nothing."""
    (tmp_path / "probe_test.py").write_text(_MODULE)
    (tmp_path / "probe.py").write_text(_PROBE)
    completed = subprocess.run(  # noqa: S603 -- Fixed interpreter and probe.
        [sys.executable, "probe.py", *plugins],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env={"PYTHONPATH": str(_CWD.parents[2]), "PATH": ""},
        text=True,
    )

    assert [
        line for line in completed.stdout.splitlines() if line.startswith("exit")
    ] == exits, completed.stdout + completed.stderr


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
