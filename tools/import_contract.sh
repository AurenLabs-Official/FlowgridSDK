#!/usr/bin/env bash
set -euo pipefail
SRC="${1:?usage: import_contract.sh <source.json> <dest-relative-to-repo>}"
DEST="${2:?}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/$DEST"

RAW="$(cat "$SRC")"
# Basic redaction; always inspect before committing.
REDACTED="$(printf '%s' "$RAW" | sed \
  -E 's/sk-ant-api[0-9]{2}-[A-Za-z0-9_-]+/REDACTED/g' \
  -E 's/sk-[A-Za-z0-9]{10,}/REDACTED/g' \
  -E 's/api-key:[[:space:]]+[^[:space:]]+/api-key: REDACTED/g')"

printf '%s' "$REDACTED" | python3 -c "import json,sys; json.load(sys.stdin)"

mkdir -p "$(dirname "$OUT")"
printf '%s\n' "$REDACTED" >"$OUT"
echo "Wrote $OUT — review for secrets before git add."
