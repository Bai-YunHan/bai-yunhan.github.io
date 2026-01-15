#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-}"
HUGO_ROOT="${2:-.}"

echo "ZIP_PATH: $ZIP_PATH"
echo "HUGO_ROOT: $HUGO_ROOT"

if [[ -z "$ZIP_PATH" ]]; then
  echo "Usage: $0 <notion_export.zip> [hugo_root]"
  exit 1
fi

WORKDIR="$(mktemp -d)"
echo "WORKDIR: $WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> Unzipping Notion export..."
unzip -q "$ZIP_PATH" -d "$WORKDIR/notion"

echo "==> Converting to Hugo content..."
python3 "$HUGO_ROOT/tools/notion_zip_to_hugo.py" \
  --notion_export_dir "$WORKDIR/notion" \
  --hugo_content_dir "$HUGO_ROOT/content" \
  --hugo_static_dir "$HUGO_ROOT/static" \
  --section "posts"

echo "==> Done. Now run: hugo server"
