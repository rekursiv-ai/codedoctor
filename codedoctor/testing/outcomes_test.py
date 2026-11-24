"""Tests for per-test completion in the coverage hand-off."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import coverage
import pytest

from codedoctor.testing import outcomes


if TYPE_CHECKING:
    from pathlib import Path


pytest_plugins = ("pytester",)


@pytest.mark.parametrize("distributed", [False, True])
def test_collection_records_the_final_selected_nodeids(
    pytester: pytest.Pytester,
    distributed: bool,
) -> None:
    destination = pytester.path / "events"
    recorder = outcomes.Outcomes(destination)
    # Collection loads every installed plugin; pytest-asyncio warns unless set.
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    items = pytester.getitems("def test_a(): pass\ndef test_b(): pass\n")
    nodeids = [item.nodeid for item in items]
    if distributed:
        recorder.pytest_xdist_node_collection_finished(nodeids)
    else:
        session = items[0].session
        session.items = items
        recorder.pytest_collection_finish(session)
    assert len(nodeids) == 2
    assert destination.read_text().splitlines() == [
        "selected",
        *(f"selected\t{nodeid}" for nodeid in nodeids),
    ]


@pytest.mark.parametrize("failed", [False, True])
def test_collection_errors_keep_their_file(tmp_path: Path, failed: bool) -> None:
    destination = tmp_path / "events"
    report = pytest.CollectReport(
        "a_test.py",
        "failed" if failed else "passed",
        None,
        [],
    )
    outcomes.Outcomes(destination).pytest_collectreport(report)
    assert destination.exists() is failed
    if failed:
        assert destination.read_text() == "failed\nfailed\ta_test.py\n"


@pytest.mark.parametrize("when", ["setup", "call", "teardown"])
@pytest.mark.parametrize("outcome", ["passed", "failed", "skipped"])
def test_every_phase_failure_counts_and_only_teardown_finishes(
    tmp_path: Path,
    when: Literal["setup", "call", "teardown"],
    outcome: Literal["passed", "failed", "skipped"],
) -> None:
    destination = tmp_path / "events"
    report = pytest.TestReport(
        "a_test.py::test_a",
        ("a_test.py", 0, "test_a"),
        {},
        outcome,
        None,
        when,
    )
    outcomes.Outcomes(destination).pytest_runtest_logreport(report)
    expected: list[str] = []
    if outcome == "failed":
        expected += ["failed", "failed\ta_test.py::test_a"]
    if when == "teardown":
        expected += ["finished", "finished\ta_test.py::test_a"]
    assert (
        destination.read_text().splitlines() if destination.exists() else []
    ) == expected


@pytest.mark.parametrize(
    ("events", "status", "invalid"),
    [
        (None, 0, {"a_test.py", "b_test.py"}),
        ("selected\n", 0, set[str]()),
        ("selected\n", 5, set[str]()),
        ("selected\n", 3, {"a_test.py", "b_test.py"}),
        ("selected\na", 4, {"a_test.py", "b_test.py"}),
        (
            (
                "selected\nselected\ta_test.py::test_a\nselected\tb_test.py::test_b\n"
                "finished\ta_test.py::test_a\nfailed\tb_test.py::test_b\n"
                "finished\tb_test.py::test_b\n"
            ),
            1,
            {"b_test.py"},
        ),
        (
            (
                "selected\nselected\ta_test.py::test_a\nselected\ta_test.py::test_b\n"
                "finished\ta_test.py::test_a\n"
            ),
            1,
            {"a_test.py"},
        ),
        (
            (
                "selected\nselected\ta_test.py::test_a\nfinished\ta_test.py::test_a\n"
                "failed\tb_test.py\n"
            ),
            2,
            {"b_test.py"},
        ),
        (
            "selected\nselected\ta_test.py::test_a\nfinished\ta_test.py::test_a\n",
            2,
            {"b_test.py"},
        ),
        (
            "selected\nselected\ta_test.py::test_a\nfinished\ta_test.py::test_a\n",
            3,
            {"b_test.py"},
        ),
        (
            (
                "selected\nselected\ta_test.py::TestA::test_a\n"
                "finished\ta_test.py::TestA::test_a\n"
            ),
            2,
            {"b_test.py"},
        ),
        (
            (
                "selected\nselected\ta_test.py::test_a[x\ty]\n"
                "finished\ta_test.py::test_a[x\ty]\n"
            ),
            0,
            set[str](),
        ),
        (
            "selected\nfailed\ta_test.py::test_a[x\ty]\n",
            1,
            {"a_test.py"},
        ),
    ],
)
def test_handoff_keeps_failures_and_incomplete_collections(
    tmp_path: Path,
    events: str | None,
    status: int,
    invalid: set[str],
) -> None:
    data = tmp_path / "coverage"
    recorded = coverage.CoverageData(basename=str(data))
    recorded.add_lines({str(tmp_path / "a.py"): [1]})
    recorded.write()
    recorded.close()
    log = tmp_path / "events"
    if events is not None:
        log.write_text(events)
    outcomes.save(
        data,
        events=log,
        tests=["a_test.py", "b_test.py"],
        status=status,
        sources={},
    )
    assert outcomes.invalid_tests(data) == invalid
    recorded.read()
    assert recorded.lines(str(tmp_path / "a.py")) == [1]
    recorded.close()


def test_plain_coverage_data_has_no_test_failures(tmp_path: Path) -> None:
    data = tmp_path / "coverage"
    recorded = coverage.CoverageData(basename=str(data))
    recorded.add_lines({"a.py": [1]})
    recorded.write()
    recorded.close()
    assert outcomes.invalid_tests(data) == set()
    assert outcomes.changed_sources(data) == set()


@pytest.mark.parametrize(
    ("edit", "changed"),
    [("kept", set[str]()), ("edited", {"a.py"}), ("deleted", {"a.py"})],
)
def test_a_source_changed_since_its_tests_ran_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit: Literal["kept", "edited", "deleted"],
    changed: set[str],
) -> None:
    """Its lines no longer match the data: an edit shifts every line below it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 1\n")
    data = tmp_path / "coverage"
    recorded = coverage.CoverageData(basename=str(data))
    recorded.add_lines({str(tmp_path / "a.py"): [1]})
    recorded.write()
    recorded.close()
    before = {name: outcomes.fingerprint(tmp_path / name) for name in ("a.py", "b.py")}
    outcomes.save(
        data,
        events=tmp_path / "events",
        tests=[],
        status=0,
        sources=before,
    )
    if edit == "edited":
        (tmp_path / "a.py").write_text("\nx = 1\n")
    elif edit == "deleted":
        (tmp_path / "a.py").unlink()

    assert outcomes.changed_sources(data) == changed


def test_a_fingerprint_is_the_contents_and_empty_for_no_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    (tmp_path / "c.py").write_text("x = 2\n")

    assert outcomes.fingerprint(tmp_path / "a.py") == outcomes.fingerprint(
        tmp_path / "b.py",
    )
    assert outcomes.fingerprint(tmp_path / "a.py") != outcomes.fingerprint(
        tmp_path / "c.py",
    )
    assert outcomes.fingerprint(tmp_path / "gone.py") == ""


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
