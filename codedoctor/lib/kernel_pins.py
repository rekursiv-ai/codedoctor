"""The BLAS kernel pins a gate gives each pytest it starts.

They are the pins ``codedoctor.lib.testing.resources.cap_math_threads`` sets in a
test process. A pytest plugin loads NumPy before any conftest runs, so a pytest
subprocess needs them in its environment at launch to reach OpenBLAS.

A copy, not an import: ``resources`` is a pytest plugin, and the wheel ships no
module that imports pytest. ``kernel_pins_test.py`` keeps the two equal.
"""

from __future__ import annotations


def kernel_pins() -> dict[str, str]:
    """Return each pinned variable's name and value, for a child's environment."""
    return {"MKL_CBWR": "COMPATIBLE", "OPENBLAS_CORETYPE": "Haswell"}
