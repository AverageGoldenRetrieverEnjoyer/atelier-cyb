#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-gdrive:tpot-backup}"
REMOTE_SUBPATH="${REMOTE_SUBPATH:-logs}"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/}"
RCLONE_OPTS=(
  --checksum
  --progress
  --transfers=8
  --checkers=16
  --drive-chunk-size=64M
  --low-level-retries=10
  --retries=3
  --contimeout=60s
  --timeout=300s
  --ignore-existing
  --verbose
  --max-age=1d
  --min-age=5m
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [subpath]

Sync data from rclone remote (default: \$REMOTE/$REMOTE_SUBPATH) into data/.
Override defaults with environment variables:
  REMOTE=gdrive:tpot-backup
  REMOTE_SUBPATH=logs
  LOCAL_ROOT=/path/to/pourtpot/data/
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGET_SUBPATH="${1:-$REMOTE_SUBPATH}"

mkdir -p "$LOCAL_ROOT/$TARGET_SUBPATH"

echo "=== Syncing from ${REMOTE}/${TARGET_SUBPATH} to ${LOCAL_ROOT}/${TARGET_SUBPATH} ==="
rclone copy \
  "${REMOTE}/${TARGET_SUBPATH}" \
  "${LOCAL_ROOT}/${TARGET_SUBPATH}" \
  "${RCLONE_OPTS[@]}"

echo ">>> Done"