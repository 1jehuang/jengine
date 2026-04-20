#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/fixtures/release_benchmark.env"

export TMPDIR=${TMPDIR:-"$ROOT_DIR/.tmp"}
mkdir -p "$TMPDIR"

STAMP=$(date -Is | tr ':' '-')
OUT_DIR=${1:-"$ROOT_DIR/.artifacts/regression/$STAMP"}
mkdir -p "$OUT_DIR"

cd "$ROOT_DIR"

./scripts/quick_ci.sh
cargo build --release --bin jengine

./target/release/jengine validate "$MODEL_ROOT" "$PROMPT_SHORT" 1 --format kv \
  > "$OUT_DIR/validate.kv"

./target/release/jengine bench "$MODEL_ROOT" "$PROMPT_SHORT" 1 1 \
  --format kv \
  --markdown "$OUT_DIR/bench.md" \
  --kv "$OUT_DIR/bench.kv" \
  --csv "$OUT_DIR/bench.csv" \
  > "$OUT_DIR/bench.stdout"

./target/release/jengine inspect "$MODEL_ROOT" "$PROMPT_SHORT" 1 --format plain \
  > "$OUT_DIR/inspect.txt"

echo "regression_out_dir=$OUT_DIR"
