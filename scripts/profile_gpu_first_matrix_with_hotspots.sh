#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/jeremy/models/bonsai-1.7b}
ARTIFACT_DIR=${2:-/home/jeremy/jengine/.artifacts/jengine-packed-model}
PROMPT=${3:-hello}
TOKENS=${4:-1}
OUT_DIR=${5:-/home/jeremy/jengine/.artifacts/gpu-first-matrix}
TOP_N=${6:-12}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

MATRIX_OUTPUT=$(
  "$SCRIPT_DIR/profile_gpu_first_matrix.sh" "$ROOT" "$ARTIFACT_DIR" "$PROMPT" "$TOKENS" "$OUT_DIR"
)
printf '%s
' "$MATRIX_OUTPUT"
RESULTS_DIR=$(printf '%s
' "$MATRIX_OUTPUT" | awk -F= '/^results_dir=/{print $2}' | tail -n 1)
if [[ -z "$RESULTS_DIR" || ! -d "$RESULTS_DIR" ]]; then
  echo "failed to determine results_dir from profile_gpu_first_matrix.sh output" >&2
  exit 1
fi

SUMMARY_PATH="$RESULTS_DIR/hotspots.txt"
: > "$SUMMARY_PATH"
for JSON_PATH in "$RESULTS_DIR"/*.json; do
  {
    echo "=== $(basename "$JSON_PATH") ==="
    python "$SCRIPT_DIR/summarize_packed_profile_hotspots.py" "$JSON_PATH" "$TOP_N"
    echo
  } | tee -a "$SUMMARY_PATH"
done

echo "hotspots_path=$SUMMARY_PATH"
