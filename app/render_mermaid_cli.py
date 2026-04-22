from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def _extract_mermaid_from_markdown(markdown_path: Path) -> str:
    text = markdown_path.read_text(encoding="utf-8")
    start_token = "```mermaid"
    start = text.find(start_token)
    if start < 0:
        raise ValueError(f"No ```mermaid fenced block found in {markdown_path}.")
    content_start = start + len(start_token)
    end = text.find("```", content_start)
    if end < 0:
        raise ValueError(f"Unterminated ```mermaid fenced block in {markdown_path}.")
    return text[content_start:end].strip() + "\n"


def _derive_output_paths(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    parent = input_path.parent
    return parent / f"{stem}.svg", parent / f"{stem}.png"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-render-mermaid",
        description="Render a Mermaid .mmd file or Markdown Mermaid block to SVG and PNG using mmdc.",
    )
    parser.add_argument("--input", required=True, help="Path to a .mmd or .md Mermaid source file.")
    parser.add_argument("--output-svg", default=None, help="Optional output SVG path.")
    parser.add_argument("--output-png", default=None, help="Optional output PNG path.")
    parser.add_argument("--theme", default="default", help="Mermaid theme passed to mmdc.")
    parser.add_argument(
        "--background-color",
        default="transparent",
        help="Background color passed to mmdc.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Scale factor passed to mmdc for PNG output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the mmdc commands without running them.",
    )
    return parser


def _render_one(
    *,
    source_path: Path,
    output_path: Path,
    theme: str,
    background_color: str,
    scale: float,
    dry_run: bool,
) -> int:
    command = [
        "mmdc",
        "-i",
        str(source_path),
        "-o",
        str(output_path),
        "-t",
        theme,
        "-b",
        background_color,
    ]
    if output_path.suffix.lower() == ".png":
        command.extend(["-s", str(scale)])

    if dry_run:
        print("+ " + " ".join(command))
        return 0

    subprocess.run(command, check=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Mermaid input file does not exist: {input_path}")

    output_svg = (
        Path(args.output_svg).expanduser().resolve()
        if args.output_svg is not None
        else _derive_output_paths(input_path)[0]
    )
    output_png = (
        Path(args.output_png).expanduser().resolve()
        if args.output_png is not None
        else _derive_output_paths(input_path)[1]
    )

    if not args.dry_run and shutil.which("mmdc") is None:
        raise FileNotFoundError(
            "Mermaid CLI `mmdc` was not found on PATH. Install it with `npm install -g @mermaid-js/mermaid-cli`."
        )

    if input_path.suffix.lower() == ".mmd":
        source_path = input_path
        temp_dir_cm = tempfile.TemporaryDirectory(prefix="digital_drn_mermaid_")
    elif input_path.suffix.lower() == ".md":
        mermaid_source = _extract_mermaid_from_markdown(input_path)
        temp_dir_cm = tempfile.TemporaryDirectory(prefix="digital_drn_mermaid_")
        temp_dir = Path(temp_dir_cm.__enter__())
        source_path = temp_dir / f"{input_path.stem}.mmd"
        source_path.write_text(mermaid_source, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported Mermaid input type: {input_path.suffix}. Use .mmd or .md.")

    try:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        _render_one(
            source_path=source_path,
            output_path=output_svg,
            theme=args.theme,
            background_color=args.background_color,
            scale=args.scale,
            dry_run=args.dry_run,
        )
        _render_one(
            source_path=source_path,
            output_path=output_png,
            theme=args.theme,
            background_color=args.background_color,
            scale=args.scale,
            dry_run=args.dry_run,
        )
    finally:
        temp_dir_cm.__exit__(None, None, None)

    print(f"render-mermaid: input={input_path}")
    print(f"render-mermaid: svg={output_svg}")
    print(f"render-mermaid: png={output_png}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
