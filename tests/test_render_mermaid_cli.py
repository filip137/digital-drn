from pathlib import Path

import pytest

from digital_drn.app.render_mermaid_cli import main


def test_render_mermaid_cli_dry_run_for_mmd_prints_svg_and_png_commands(capsys, tmp_path: Path):
    source = tmp_path / "diagram.mmd"
    source.write_text("flowchart LR\nA-->B\n", encoding="utf-8")

    exit_code = main(["--input", str(source), "--dry-run"])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert f"+ mmdc -i {source.resolve()} -o {(tmp_path / 'diagram.svg').resolve()}" in output
    assert f"+ mmdc -i {source.resolve()} -o {(tmp_path / 'diagram.png').resolve()}" in output
    assert f"render-mermaid: input={source.resolve()}" in output


def test_render_mermaid_cli_dry_run_extracts_mermaid_block_from_markdown(capsys, tmp_path: Path):
    source = tmp_path / "diagram.md"
    source.write_text(
        "# Demo\n\n```mermaid\nflowchart LR\nA-->B\n```\n",
        encoding="utf-8",
    )

    exit_code = main(["--input", str(source), "--dry-run"])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "+ mmdc -i " in output
    assert str((tmp_path / "diagram.svg").resolve()) in output
    assert str((tmp_path / "diagram.png").resolve()) in output


def test_render_mermaid_cli_rejects_markdown_without_mermaid_block(tmp_path: Path):
    source = tmp_path / "diagram.md"
    source.write_text("# Demo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No ```mermaid fenced block found"):
        main(["--input", str(source), "--dry-run"])
