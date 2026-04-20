#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/fixtures/release_benchmark.env"

export TMPDIR=${TMPDIR:-"$ROOT_DIR/.tmp"}
mkdir -p "$TMPDIR"

STAMP=$(date -Is | tr ':' '-')
OUT_DIR=${1:-"$ROOT_DIR/.artifacts/local-profile/$STAMP"}
mkdir -p "$OUT_DIR"

cd "$ROOT_DIR"

cargo build --release \
  --bin jengine \
  --bin compare_packed_matvec \
  --bin run_reference \
  --bin bench_real_text \
  --bin vulkan_dense_matvec \
  --bin vulkan_packed_matvec

run_capture() {
  local name=$1
  shift
  {
    echo "# $name"
    echo
    echo '```bash'
    printf '%q ' "$@"
    echo
    echo '```'
    echo '```text'
    if command -v /usr/bin/time >/dev/null 2>&1; then
      /usr/bin/time -v "$@"
    else
      "$@"
    fi
    echo '```'
    echo
  } > "$OUT_DIR/$name.md"
}

run_capture inspect_release \
  ./target/release/jengine inspect "$MODEL_ROOT" "$PROMPT_SHORT" 1 --format plain

run_capture profile_release \
  ./target/release/jengine profile "$MODEL_ROOT" "$PROMPT_SHORT" 1 --format kv

run_capture compare_packed_matvec_release \
  ./target/release/compare_packed_matvec "$WEIGHTS_PATH" "$QPROJ_TENSOR" "$QPROJ_ROWS" "$QPROJ_COLS"

./scripts/capture_release_baselines.sh "$OUT_DIR/release-baselines" > "$OUT_DIR/capture_release_baselines.stdout"

echo "local_profile_out_dir=$OUT_DIR"
