"""Test adapter activation and unchanged native CLI dispatch."""

from __future__ import annotations

from typing import NoReturn

import runpy
import sys

from mutmut import __main__ as mutmut_main

import pytest

from codedoctor.mutation import mutmut_adapter


def test_dispatch_activates_before_native_cli_and_preserves_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    argv = ["mutmut_cli", "run", "--max-children", "1", "mod.x_f__mutmut_1"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(mutmut_adapter, "activate", lambda: events.append("activate"))

    def native_cli() -> NoReturn:
        events.append("native")
        assert sys.argv == argv
        raise SystemExit(7)

    monkeypatch.setattr(mutmut_main, "cli", native_cli)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("codedoctor.mutation.mutmut_cli", run_name="__main__")
    assert stopped.value.code == 7
    assert events == ["activate", "native"]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
