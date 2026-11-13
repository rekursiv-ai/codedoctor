"""Record failed or unfinished own tests alongside a gate's coverage data."""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, cast

import sqlite3

from _pytest.config import hookimpl


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pytest


class Outcomes:
    """Record collection and completion events in the pytest controller only."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Record the final serial collection."""
        self._write("selected", [item.nodeid for item in session.items])

    @hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, ids: Sequence[str]) -> None:
        """Record each worker's collection before scheduling."""
        self._write("selected", ids)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        """Keep collection errors attributable to their test module."""
        if report.failed:
            self._write("failed", [report.nodeid])

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Record failures in any phase and completion after teardown."""
        if report.failed:
            self._write("failed", [report.nodeid])
        if report.when == "teardown":
            self._write("finished", [report.nodeid])

    def _write(self, event: str, nodeids: Sequence[str]) -> None:
        with self.destination.open("a") as handle:
            handle.write(f"{event}\n")
            handle.writelines(f"{event}\t{nodeid}\n" for nodeid in nodeids)


def save(data: Path, *, events: Path, tests: Sequence[str], status: int) -> None:
    """Store failed and unfinished test files inside the coverage SQLite hand-off.

    Args:
      data: Coverage database from the measured own-test run.
      events: Controller's collection and test-phase event log.
      tests: Selected own test files.
      status: Own pytest process exit status.

    """
    rows = events.read_text().splitlines() if events.is_file() else []
    selected = {row.partition("\t")[2] for row in rows if row.startswith("selected\t")}
    finished = {row.partition("\t")[2] for row in rows if row.startswith("finished\t")}
    failed = {row.partition("\t")[2] for row in rows if row.startswith("failed\t")}
    invalid = {nodeid.partition("::")[0] for nodeid in (selected - finished) | failed}
    if "selected" not in rows:
        invalid.update(tests)
    elif status not in {0, 1, 5}:
        completed_files = {nodeid.partition("::")[0] for nodeid in finished}
        invalid.update(set(tests) - completed_files)
    with closing(sqlite3.connect(data)) as connection, connection:
        connection.execute(
            "CREATE TABLE codedoctor_invalid_tests (path TEXT PRIMARY KEY)",  # pragma: no mutate[case-insensitive]
        )
        connection.executemany(
            "INSERT INTO codedoctor_invalid_tests VALUES (?)",  # pragma: no mutate[case-insensitive]
            [(path,) for path in sorted(invalid)],
        )


def invalid_tests(data: Path) -> set[str]:
    """Read failed or unfinished tests, if the gate annotated this coverage database.

    Args:
      data: Coverage database, optionally annotated by ``save``.

    Returns:
      tests: Paths whose measurements are incomplete or whose tests failed.

    """
    with closing(sqlite3.connect(data)) as connection:
        exists = cast(
            tuple[int] | None,
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'codedoctor_invalid_tests'",  # pragma: no mutate -- SQLite SQL keywords are case-insensitive.
            ).fetchone(),
        )
        if not exists:
            return set()
        rows = cast(
            list[tuple[str]],
            connection.execute(
                "SELECT path FROM codedoctor_invalid_tests",  # pragma: no mutate[case-insensitive]
            ).fetchall(),
        )
        return {row[0] for row in rows}
