"""Tests for the one test-pairing rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedoctor.lib.sidecars import candidates, is_test, paired, related


def test_a_nested_source_pairs_beside_in_tests_and_in_the_root_mirror() -> None:
    assert candidates("pkg/sub/mod.py") == (
        Path("pkg/sub/mod_test.py"),
        Path("pkg/sub/test_mod.py"),
        Path("pkg/sub/tests/mod_test.py"),
        Path("pkg/sub/tests/test_mod.py"),
        Path("tests/sub/mod_test.py"),
        Path("tests/sub/test_mod.py"),
    )


@pytest.mark.parametrize("name", ["mod_test.py", "test_mod.py", "mod.py"])
def test_a_test_pairs_like_its_source(name: str) -> None:
    assert candidates(f"pkg/{name}") == candidates("pkg/mod.py")


def test_a_module_of_the_top_level_package_mirrors_into_the_tests_root() -> None:
    """``configgle/fig.py`` is tested by ``tests/fig_test.py`` in its export."""
    assert candidates("pkg/mod.py")[-2:] == (
        Path("tests/mod_test.py"),
        Path("tests/test_mod.py"),
    )


def test_a_top_level_module_has_no_mirror() -> None:
    assert candidates("mod.py") == (
        Path("mod_test.py"),
        Path("test_mod.py"),
        Path("tests/mod_test.py"),
        Path("tests/test_mod.py"),
    )


def test_a_root_test_pairs_without_mirroring_itself() -> None:
    assert candidates("tests/sub/mod_test.py") == (
        Path("tests/sub/mod_test.py"),
        Path("tests/sub/test_mod.py"),
        Path("tests/sub/tests/mod_test.py"),
        Path("tests/sub/tests/test_mod.py"),
    )


def test_a_source_pairs_with_its_sidecar_and_its_splits(tmp_path: Path) -> None:
    """``mod_pglite_test.py`` is mod's; ``mod_extra_test.py`` is ``mod_extra``'s."""
    for name in (
        "mod.py",
        "mod_extra.py",
        "mod_test.py",
        "mod_pglite_test.py",
        "tests/test_mod_cli.py",
        "mod_extra_test.py",
        "mod_extra_deep_test.py",
    ):
        (tmp_path / "pkg" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "pkg" / name).touch()
    (tmp_path / "pkg" / "mod_dir_test.py").mkdir()

    assert paired("pkg/mod.py", root=tmp_path) == [
        Path("pkg/mod_test.py"),
        Path("pkg/mod_pglite_test.py"),
        Path("pkg/tests/test_mod_cli.py"),
    ]
    assert paired("pkg/mod_test.py", root=tmp_path) == [Path("pkg/mod_test.py")]


def test_a_longer_modules_prefixed_tests_are_not_its_splits(tmp_path: Path) -> None:
    """``test_mod_extra*.py`` belongs to ``mod_extra``, not to ``mod``."""
    for name in (
        "mod.py",
        "mod_extra.py",
        "test_mod_extra.py",
        "test_mod_extra_more.py",
        "test_mod_cli.py",
    ):
        (tmp_path / "pkg").mkdir(exist_ok=True)
        (tmp_path / "pkg" / name).touch()

    assert paired("pkg/mod.py", root=tmp_path) == [Path("pkg/test_mod_cli.py")]


@pytest.mark.parametrize(
    ("changed", "split"),
    [
        ("mod_test.py", "mod_test_x_test.py"),
        ("test_mod.py", "test_mod_x_test.py"),
    ],
    ids=["suffix", "prefix"],
)
def test_a_test_pairs_with_its_sidecars_but_never_with_splits(
    tmp_path: Path,
    changed: str,
    split: str,
) -> None:
    """Read as a source, ``mod_test.py`` would claim ``mod_test_x_test.py``."""
    (tmp_path / "pkg").mkdir()
    for name in (changed, split):
        (tmp_path / "pkg" / name).touch()

    assert paired(f"pkg/{changed}", root=tmp_path) == [Path("pkg", changed)]


def test_without_a_root_pairs_from_the_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "mod_test.py").touch()
    monkeypatch.chdir(tmp_path)

    assert paired("mod.py") == [Path("mod_test.py")]


@pytest.mark.parametrize(
    "name",
    ["conftest.py", "test_mod.py", "mod_test.py", "mod_parity.py"],
)
def test_a_module_named_as_pytest_collects_it_is_a_test(name: str) -> None:
    assert is_test(f"pkg/{name}")


@pytest.mark.parametrize(
    ("body", "test"),
    [
        ("def test_x():\n    pass\n", True),
        ("async def test_x():\n    pass\n", True),
        ("class TestX:\n    pass\n", True),
        ("def helper():\n    def test_inner():\n        pass\n", False),
        ("X = 1\n", False),
        ("def (\n", False),
        ("__test__ = False\ndef test_main(test_file):\n    pass\n", False),
        ("__test__: bool = False\nclass TestHelper:\n    pass\n", False),
        ("__test__ = True\ndef test_x():\n    pass\n", True),
        ("other = False\ndef test_x():\n    pass\n", True),
        ("flags.__test__ = False\ndef test_x():\n    pass\n", True),
        ("__test__ = 0\ndef test_x():\n    pass\n", True),
    ],
    ids=[
        "function",
        "async",
        "class",
        "nested",
        "source",
        "unparsable",
        "helper-opt-out",
        "annotated-opt-out",
        "explicit-suite",
        "unrelated-flag",
        "attribute-flag",
        "non-boolean-flag",
    ],
)
def test_a_hand_run_suite_is_a_test_by_what_it_defines(
    tmp_path: Path,
    body: str,
    test: bool,
) -> None:
    """A suite run by hand defines tests under a name pytest never collects."""
    module = tmp_path / "mod_checks.py"
    module.write_text(body)

    assert is_test(module) is test


@pytest.mark.parametrize(
    "named",
    [
        "pkg/sub/mod_extra.py",
        "pkg/sub/mod_extra_test.py",
        "pkg/sub/test_mod_extra.py",
        "pkg/sub/tests/test_mod_extra_more.py",
        "tests/sub/mod_extra_test.py",
    ],
)
def test_related_finds_source_and_all_its_test_pairs(
    tmp_path: Path,
    named: str,
) -> None:
    files = [
        "pkg/sub/mod_extra.py",
        "pkg/sub/mod_extra_test.py",
        "pkg/sub/test_mod_extra.py",
        "pkg/sub/tests/test_mod_extra_more.py",
        "tests/sub/mod_extra_test.py",
    ]
    for name in [*files, "pkg/sub/mod.py", "pkg/sub/mod_test.py"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert related(named, root=tmp_path) == sorted(map(Path, files))


def test_related_keeps_an_unpaired_file(tmp_path: Path) -> None:
    (tmp_path / "orphan_test.py").touch()
    assert related("orphan_test.py", root=tmp_path) == [Path("orphan_test.py")]


def test_a_missing_module_is_judged_by_its_name_alone(tmp_path: Path) -> None:
    assert not is_test(tmp_path / "gone.py")


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
