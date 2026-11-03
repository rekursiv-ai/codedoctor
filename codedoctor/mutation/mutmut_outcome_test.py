"""Tests for the plugin failing a mutation run that ran no test."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codedoctor.mutation import mutmut_outcome


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (
            "import pytest\n@pytest.mark.skip\ndef test_a():\n    pass\n",
            pytest.ExitCode.TESTS_FAILED,
        ),
        (
            (
                "import pytest\n@pytest.mark.skip\ndef test_a():\n    pass\n"
                "def test_b():\n    pass\n"
            ),
            pytest.ExitCode.OK,
        ),
        (
            "import pytest\ndef test_a():\n    pytest.skip()\n",
            pytest.ExitCode.TESTS_FAILED,
        ),
        ("def test_a():\n    assert False\n", pytest.ExitCode.TESTS_FAILED),
        ("def test_a():\n    pass\n", pytest.ExitCode.OK),
        ("", pytest.ExitCode.NO_TESTS_COLLECTED),
    ],
    ids=["all-skipped", "one-ran", "skipped-in-body", "failed", "passed", "none"],
)
def test_a_session_that_ran_no_test_fails(
    tmp_path: Path,
    body: str,
    code: pytest.ExitCode,
) -> None:
    probe = tmp_path / f"{tmp_path.name}_test.py"
    probe.write_text(body)
    args = ["-q", "-p", "no:cacheprovider", "-p", "no:asyncio", str(probe)]
    assert pytest.main(args, plugins=[mutmut_outcome]) == code


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
