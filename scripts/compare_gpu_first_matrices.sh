#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: compare_gpu_first_matrices.sh <before_dir> <after_dir> [top_n]" >&2
  exit 2
fi

BEFORE_DIR=$1
AFTER_DIR=$2
TOP_N=${3:-10}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for BEFORE_JSON in "$BEFORE_DIR"/*.json; do
  NAME=$(basename "$BEFORE_JSON")
  AFTER_JSON="$AFTER_DIR/$NAME"
  if [[ ! -f "$AFTER_JSON" ]]; then
    echo "skip $NAME: missing in after dir" >&2
    continue
  fi
  echo "=== $NAME ==="
  python "$SCRIPT_DIR/compare_packed_profile_hotspots.py" "$BEFORE_JSON" "$AFTER_JSON" "$TOP_N"
  echo
  echo
 done
