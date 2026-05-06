#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_JSON = REPO_ROOT / "literature" / "papers.json"
CACHE_ROOT = REPO_ROOT / "literature" / "_pdf_cache"
MANIFEST_PATH = CACHE_ROOT / "manifest.sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_papers(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return data


def _download(url: str, destination: Path, *, timeout: float) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "digital-drn-literature-fetcher/1.0"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _manifest_line(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).as_posix()
    return f"{_sha256(path)}  {relative}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download cached PDFs for the literature packet.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without writing files.")
    parser.add_argument("--force", action="store_true", help="Redownload PDFs even when the cache file exists.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Network timeout in seconds per paper.")
    args = parser.parse_args(argv)

    papers = _load_papers(PAPERS_JSON)
    cached_paths: list[Path] = []
    failures = 0

    for paper in papers:
        slug = paper.get("slug", "<missing slug>")
        title = paper.get("title", slug)
        pdf_url = paper.get("pdf_url")
        official_url = paper.get("official_url") or paper.get("abs_url")
        pdf_path_raw = paper.get("pdf_path")
        if not pdf_path_raw:
            print(f"ERROR missing pdf_path for {slug}", file=sys.stderr)
            failures += 1
            continue

        destination = REPO_ROOT / pdf_path_raw
        if not pdf_url:
            print(f"SKIP no direct PDF: {title}")
            if official_url:
                print(f"  official: {official_url}")
            continue

        if destination.exists() and not args.force:
            print(f"OK cached: {destination.relative_to(REPO_ROOT)}")
            cached_paths.append(destination)
            continue

        if args.dry_run:
            action = "REDOWNLOAD" if destination.exists() else "DOWNLOAD"
            print(f"{action}: {title}")
            print(f"  from: {pdf_url}")
            print(f"  to:   {destination.relative_to(REPO_ROOT)}")
            continue

        try:
            print(f"Downloading: {title}")
            _download(pdf_url, destination, timeout=args.timeout)
            cached_paths.append(destination)
        except (OSError, urllib.error.URLError) as exc:
            print(f"ERROR failed to download {slug}: {exc}", file=sys.stderr)
            failures += 1

    if not args.dry_run:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_lines = [_manifest_line(path) for path in sorted(cached_paths)]
        MANIFEST_PATH.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
        print(f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
