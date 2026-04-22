import os
from pathlib import Path

from digital_drn.app.find_events_cli import main


def _write_event_file(path: Path, stamp: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub\n")
    time_value = float(stamp)
    os.utime(path, (time_value, time_value))


def test_find_events_cli_lists_newest_first(capsys, tmp_path: Path):
    oldest = tmp_path / "a" / "events.out.tfevents.old"
    newest = tmp_path / "b" / "events.out.tfevents.new"
    _write_event_file(oldest, 100)
    _write_event_file(newest, 200)

    exit_code = main(["--output-root", str(tmp_path), "--limit", "2"])
    assert exit_code == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert str(newest) in lines[0]
    assert str(oldest) in lines[1]


def test_find_events_cli_filters_by_contains(capsys, tmp_path: Path):
    matching = tmp_path / "cifar10_model" / "events.out.tfevents.match"
    excluded = tmp_path / "mnist_model" / "events.out.tfevents.other"
    _write_event_file(matching, 100)
    _write_event_file(excluded, 200)

    exit_code = main(
        [
            "--output-root",
            str(tmp_path),
            "--contains",
            "cifar10",
        ]
    )
    assert exit_code == 0

    output = capsys.readouterr().out
    assert str(matching) in output
    assert str(excluded) not in output


def test_find_events_cli_reports_no_matches(capsys, tmp_path: Path):
    exit_code = main(["--output-root", str(tmp_path)])
    assert exit_code == 1
    assert f"no event files found under {tmp_path.resolve()}" in capsys.readouterr().out
