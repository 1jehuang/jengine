#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/fixtures/release_benchmark.env"

OUT_DIR=${1:-"$ROOT_DIR/.artifacts/release-baselines"}
mkdir -p "$OUT_DIR"

export TMPDIR=${TMPDIR:-"$ROOT_DIR/.tmp"}
mkdir -p "$TMPDIR"

STAMP=$(date -Is | tr ':' '-')
REPORT="$OUT_DIR/$STAMP.md"
COMMIT=$(git -C "$ROOT_DIR" rev-parse --short HEAD)
HOST=$(hostname)

run_and_capture() {
  local label=$1
  shift
  {
    echo "## Benchmark: $label"
    echo
    echo "- Date: $(date -Is)"
    echo "- Commit: $COMMIT"
    echo "- Host: $HOST"
    echo "- Command:"
    echo '```bash'
    printf '%q ' "$@"
    echo
    echo '```'
    echo '- Result:'
    echo '```text'
    "$@"
    echo '```'
    echo
  } >> "$REPORT"
}

{
  echo "# Jengine release baseline capture"
  echo
  echo "- Date: $(date -Is)"
  echo "- Commit: $COMMIT"
  echo "- Host: $HOST"
  echo "- Fixture file: fixtures/release_benchmark.env"
  echo
} > "$REPORT"

(cd "$ROOT_DIR" && cargo build --release \
  --bin jengine \
  --bin run_reference \
  --bin bench_real_text \
  --bin compare_reference_projection \
  --bin vulkan_dense_matvec \
  --bin vulkan_packed_matvec \
  --bin bench_hybrid_qproj)

run_and_capture \
  "cpu_reference_one_token" \
  "$ROOT_DIR/target/release/run_reference" \
  "$MODEL_ROOT" "$PROMPT_SHORT" 1

run_and_capture \
  "cpu_reference_short_bench" \
  "$ROOT_DIR/target/release/bench_real_text" \
  "$MODEL_ROOT" "$PROMPT_SHORT" "$CPU_BENCH_NEW_TOKENS" "$CPU_BENCH_ITERATIONS"

run_and_capture \
  "runtime_qproj_dense_vs_packed" \
  "$ROOT_DIR/target/release/compare_reference_projection" \
  "$MODEL_ROOT" "$QPROJ_LAYER" "$TOKEN_ID"

run_and_capture \
  "vulkan_dense_qproj" \
  "$ROOT_DIR/target/release/vulkan_dense_matvec" \
  "$WEIGHTS_PATH" "$QPROJ_TENSOR" "$QPROJ_ROWS" "$QPROJ_COLS"

run_and_capture \
  "vulkan_packed_qproj" \
  "$ROOT_DIR/target/release/vulkan_packed_matvec" \
  "$WEIGHTS_PATH" "$QPROJ_TENSOR" "$QPROJ_ROWS" "$QPROJ_COLS"

run_and_capture \
  "hybrid_qproj_decode" \
  "$ROOT_DIR/target/release/bench_hybrid_qproj" \
  "$MODEL_ROOT" "$PROMPT_SHORT" "$HYBRID_NEW_TOKENS" "$QPROJ_LAYER"

echo "Wrote $REPORT"
