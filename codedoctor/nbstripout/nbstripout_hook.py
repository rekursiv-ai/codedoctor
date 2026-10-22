"""Portable pre-commit entrypoint for notebook output stripping."""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    """Run the fastest installed nbstripout implementation.

    Returns:
      exit_code: Pass-through from nbstripout or 127 if neither binary found.

    """
    passthrough = sys.argv[1:]
    executable = shutil.which("nbstripout-fast") or shutil.which("nbstripout")
    if executable is None:
        sys.stderr.write("nbstripout or nbstripout-fast is required.\n")
        return 127
    completed = subprocess.run([executable, *passthrough], check=False)  # noqa: S603 -- The hook invokes the fixed nbstripout command on repository files.
    return completed.returncode
