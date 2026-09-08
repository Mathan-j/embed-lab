#!/usr/bin/env bash
# Assemble the Hugging Face Space directory from this repo.
#
# The Space is a separate git repo hosted on huggingface.co, so it needs its own
# self-contained tree: the backend's app/ and data/ at the root, plus the
# Dockerfile and the HF-frontmatter README.
#
# Deliberately NOT copied: data/models (92 MB of weights -- the Dockerfile bakes
# them in from the Hub at build time) and data/images (the Imagenette corpus,
# which the backend does not use yet).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO_ROOT/.space-build}"

rm -rf "$OUT"
mkdir -p "$OUT"

cp "$REPO_ROOT/deploy/hf-space/Dockerfile"        "$OUT/Dockerfile"
cp "$REPO_ROOT/deploy/hf-space/README.md"         "$OUT/README.md"
cp "$REPO_ROOT/deploy/hf-space/requirements.txt"  "$OUT/requirements.txt"

cp -r "$REPO_ROOT/backend/app"  "$OUT/app"
mkdir -p "$OUT/data"
cp "$REPO_ROOT/backend/data/vocabulary.py" "$OUT/data/"
# data/ needs to be a package for `from data.vocabulary import ...` to resolve
[ -f "$REPO_ROOT/backend/data/__init__.py" ] && cp "$REPO_ROOT/backend/data/__init__.py" "$OUT/data/" || true

# Strip anything that must never reach a public Space or would bloat the image.
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -name "*.pyc" -delete 2>/dev/null || true

echo "Space tree assembled at: $OUT"
echo
echo "  files:  $(find "$OUT" -type f | wc -l)"
echo "  size:   $(du -sh "$OUT" | cut -f1)"
echo
echo "Guard: no model weights or image corpus should appear below."
find "$OUT" -type f -size +1M -exec ls -lh {} \; 2>/dev/null | awk '{print "  LARGE: "$5" "$9}' || true
echo
echo "Next:"
echo "  hf auth login                     # once, in your own terminal"
echo "  hf repo create <user>/embed-lab-api --repo-type space --space_sdk docker"
echo "  cd $OUT && git init && git remote add origin https://huggingface.co/spaces/<user>/embed-lab-api"
echo "  git add -A && git commit -m 'Embed Lab API' && git push -u origin main"
