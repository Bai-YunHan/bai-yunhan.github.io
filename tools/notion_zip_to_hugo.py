#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import unicodedata
import urllib.parse
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}
ASSET_EXTS = IMAGE_EXTS | {".pdf"}

def slugify(name: str) -> str:
    # conservative slugify
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_-]+", "-", name).strip("-")
    return name or "page"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def write_text(p: Path, s: str) -> None:
    ensure_dir(p.parent)
    p.write_text(s, encoding="utf-8")

def has_front_matter(md: str) -> bool:
    return md.lstrip().startswith("---\n")

def add_front_matter(md: str, title: str) -> str:
    if has_front_matter(md):
        return md
    fm = (
        "---\n"
        f'title: "{title.replace(chr(34), r"\"")}"\n'
        "draft: false\n"
        "math: true\n"
        "---\n\n"
    )
    return fm + md

def convert_callout_html_to_blockquote(md: str) -> str:
    # Notion exports callouts as HTML sometimes. :contentReference[oaicite:4]{index=4}
    # This is a simple best-effort conversion.
    md = re.sub(r"</?aside[^>]*>", "", md, flags=re.IGNORECASE)
    md = re.sub(r"<br\s*/?>", "\n", md, flags=re.IGNORECASE)
    # Wrap remaining lines that came from the aside-ish blocks
    # If your export has richer HTML, consider a real HTML->MD converter.
    return md

def rewrite_image_links(md: str, assets_map: dict) -> str:
    # Markdown image: ![alt](path)
    # We use a non-greedy match for the URL and handle potential nested parentheses 
    # by matching until the last closing parenthesis in a simple way, 
    # or just use a more restricted match if we assume no spaces in URLs (but Notion has them).
    # A better way is to match until the end of line or next markdown structure.
    # For now, let's use a regex that handles common Notion paths with parentheses.
    def repl(m):
        alt = m.group(1)
        url = m.group(2)
        
        # Unquote URL (e.g., %20 -> space)
        clean = urllib.parse.unquote(url.strip())

        # Also try basename
        base = os.path.basename(clean)
        if base in assets_map:
            return f"![{alt}]({assets_map[base]})"

        # Try exact match
        if clean in assets_map:
            return f"![{alt}]({assets_map[clean]})"

        return m.group(0)

    # Use a regex that allows nested parentheses by matching until the last ')' on the line
    return re.sub(r"!\[([^\]]*)\]\((.+)\)", repl, md)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notion_export_dir", required=True)
    ap.add_argument("--hugo_content_dir", required=True)
    ap.add_argument("--hugo_static_dir", required=True)
    ap.add_argument("--section", default="docs")  # content/<section>/
    args = ap.parse_args()

    notion_dir = Path(args.notion_export_dir)
    content_root = Path(args.hugo_content_dir) / args.section
    static_root = Path(args.hugo_static_dir) / "notion-assets"

    ensure_dir(content_root)
    ensure_dir(static_root)

    # Build an asset map from export folder: relative file -> /notion-assets/<file>
    # Copy assets into static/notion-assets/
    assets_map = {}
    for p in notion_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in ASSET_EXTS:
            dest_name = p.name
            dest = static_root / dest_name

            # Avoid collisions by suffixing
            if dest.exists():
                stem, ext = p.stem, p.suffix
                i = 2
                while (static_root / f"{stem}-{i}{ext}").exists():
                    i += 1
                dest = static_root / f"{stem}-{i}{ext}"

            shutil.copy2(p, dest)
            assets_map[p.name] = f"/notion-assets/{dest.name}"

    # Convert markdown pages
    md_files = list(notion_dir.rglob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found under {notion_dir}")

    for md_path in md_files:
        md = read_text(md_path)

        # Title: use filename (strip Notion suffixes like " PageName 123abc.md" if present)
        raw_title = md_path.stem
        # remove trailing Notion id-like tokens
        raw_title = re.sub(r"\s+[0-9a-fA-F]{8,}$", "", raw_title).strip()

        md = convert_callout_html_to_blockquote(md)
        md = rewrite_image_links(md, assets_map)
        md = add_front_matter(md, raw_title)

        out_slug = slugify(raw_title)
        out_path = content_root / out_slug / "index.md"
        write_text(out_path, md)

    print(f"Imported {len(md_files)} markdown file(s) into {content_root}")
    print(f"Copied assets into {static_root} (referenced as /notion-assets/...)")

if __name__ == "__main__":
    main()
