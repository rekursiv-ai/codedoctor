from __future__ import annotations

from pathlib import Path

import fcntl
import os
import subprocess
import sys

import pytest

from codedoctor.lib.files import tempclaim


def test_a_claim_is_removed_with_its_lock_on_exit(tmp_path: Path) -> None:
    with tempclaim.claimed_directory(tmp_path, prefix="run-") as directory:
        (directory / "file").touch()
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            directory.name,
            f"{directory.name}.lock",
        ]

    assert list(tmp_path.iterdir()) == []


def test_a_claim_sweeps_a_dead_owners_directory_and_keeps_a_live_one(
    tmp_path: Path,
) -> None:
    dead, live = tmp_path / "run-dead", tmp_path / "run-live"
    for directory in (dead, live):
        (directory / "popen-gw0").mkdir(parents=True)
        directory.with_suffix(".lock").touch()
    held = os.open(live.with_suffix(".lock"), os.O_RDONLY)
    try:
        fcntl.flock(held, fcntl.LOCK_EX)
        with tempclaim.claimed_directory(tmp_path, prefix="run-") as directory:
            remaining = {path.name for path in tmp_path.iterdir()}
    finally:
        os.close(held)

    assert remaining == {
        "run-live",
        "run-live.lock",
        directory.name,
        f"{directory.name}.lock",
    }


def test_sweep_removes_only_old_unlocked_entries(tmp_path: Path) -> None:
    for name in ("run-old", "run-young"):
        (tmp_path / name).mkdir()
    (tmp_path / "run-old.pending").touch()
    os.utime(tmp_path / "run-old", (0, 0))
    os.utime(tmp_path / "run-old.pending", (0, 0))

    tempclaim.sweep(tmp_path, prefix="run-", max_age_sec=60)

    assert [path.name for path in tmp_path.iterdir()] == ["run-young"]


def test_sweep_ignores_entries_without_the_prefix(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    other.with_suffix(".lock").touch()
    os.utime(other, (0, 0))

    tempclaim.sweep(tmp_path, prefix="run-", max_age_sec=0)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "other",
        "other.lock",
    ]


def test_sweep_skips_claims_owned_by_another_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign = tmp_path / "run-foreign"
    foreign.mkdir()
    monkeypatch.setattr(os, "getuid", lambda: foreign.stat().st_uid + 1)

    tempclaim.sweep(tmp_path, prefix="run-", max_age_sec=0)

    assert foreign.is_dir()


def test_reap_ignores_a_lock_removed_before_open(tmp_path: Path) -> None:
    missing = tmp_path / "gone.lock"
    tempclaim._reap(missing)
    assert not missing.exists()


@pytest.mark.cli_python_subprocess
def test_a_killed_owners_claim_is_swept_by_the_next_claim(tmp_path: Path) -> None:
    owner = subprocess.Popen(  # noqa: S603 -- fixed Python and module entry point.
        [
            sys.executable,
            "-c",
            (
                "import sys, pathlib\n"
                "from codedoctor.lib.files import tempclaim\n"
                "with tempclaim.claimed_directory(pathlib.Path(sys.argv[1]), prefix='run-')"
                " as d:\n"
                "    (d / 'file').touch()\n"
                "    print(d, flush=True)\n"
                "    sys.stdin.read()\n"
            ),
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert owner.stdout is not None
    orphan = Path(owner.stdout.readline().strip())
    owner.kill()
    owner.communicate(timeout=30)
    assert orphan.is_dir()

    with tempclaim.claimed_directory(tmp_path, prefix="run-") as directory:
        remaining = sorted(path.name for path in tmp_path.iterdir())

    assert remaining == [directory.name, f"{directory.name}.lock"]


def test_remove_tree_deletes_directories_a_test_made_read_only(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "run" / "staged" / "agent"
    staged.mkdir(parents=True)
    (staged / "file").touch()
    staged.chmod(0o500)
    staged.parent.chmod(0o500)

    tempclaim.remove_tree(tmp_path / "run")

    assert not (tmp_path / "run").exists()


def test_dotted_claims_keep_distinct_locks_and_reap_the_exact_directory(
    tmp_path: Path,
) -> None:
    with (
        tempclaim.claimed_directory(tmp_path, prefix="a.b-") as first,
        tempclaim.claimed_directory(tmp_path, prefix="a.b-") as second,
    ):
        assert first.is_dir()
        assert second.is_dir()
        assert (tmp_path / f"{first.name}.lock").is_file()
        assert (tmp_path / f"{second.name}.lock").is_file()
    assert list(tmp_path.iterdir()) == []
    dead = tmp_path / "a.b-dead"
    dead.mkdir()
    (tmp_path / f"{dead.name}.lock").touch()
    unrelated = tmp_path / "a"
    unrelated.mkdir()
    tempclaim.sweep(tmp_path, prefix="a.b-")
    assert not dead.exists()
    assert unrelated.is_dir()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
