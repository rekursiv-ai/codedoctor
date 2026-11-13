#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Activate CodeDoctor's adapter before dispatching to native mutmut.
'''
# fmt: on

from __future__ import annotations

from mutmut.__main__ import cli

from codedoctor.mutation.mutmut_adapter import activate


if __name__ == "__main__":
    activate()
    cli()
# vim: ft=python
