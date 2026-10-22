"""Tests for ``kernel_pins``: a gate's pytest children get the harness's pins."""

from codedoctor.lib.kernel_pins import kernel_pins
from codedoctor.lib.testing import resources


def test_the_pins_are_the_ones_the_test_harness_sets() -> None:
    assert kernel_pins() == resources.kernel_pins()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
