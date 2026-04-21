#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/jeremy/models/bonsai-1.7b}
ARTIFACT_DIR=${2:-/home/jeremy/jengine/.artifacts/jengine-packed-model}
PROMPT=${3:-hello}
TOKENS=${4:-1}
OUT_DIR=${5:-/home/jeremy/jengine/.artifacts/gpu-first-matrix}

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$OUT_DIR/$TS"
mkdir -p "$RUN_DIR"

COMMON_ENV=(
  JENGINE_NO_HEARTBEAT=1
  JENGINE_PREWARM_PACKED=1
  JENGINE_PACKED_ATTENTION_FULL=1
  JENGINE_PACKED_MLP_FULL=1
)

run_profile() {
  local name=$1
  shift
  local json_path="$RUN_DIR/${name}.json"
  local txt_path="$RUN_DIR/${name}.txt"
  echo "[run] $name"
  env "${COMMON_ENV[@]}" "$@" \
    cargo run --quiet --bin profile_packed_decode -- \
    "$ROOT" "$ARTIFACT_DIR" "$PROMPT" "$TOKENS" combined "$json_path" \
    > "$txt_path"
}

summarize() {
  python - <<'PY' "$RUN_DIR"
import json, os, glob, sys
run_dir = sys.argv[1]
for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
    name = os.path.basename(path).rsplit('.', 1)[0]
    p = json.load(open(path))
    m = p['packed_metrics']
    print(f"{name}: total_ms={m['total_ms']:.3f} gpu_ms={m['gpu_ms']:.3f} download_ms={m['download_ms']:.3f} compile_ms={m['compile_ms']:.3f} non_offloaded_dense_ms={m['non_offloaded_dense_ms']:.3f} dispatch_count={m['dispatch_count']}")
PY
}

# Strong packed baseline
run_profile strong_packed env

# Full last-layer gpu-first path
run_profile gpu_full_last_layer env JENGINE_GPU_FULL_LAST_LAYER=1

# Broader gpu-first attention + swiglu path
run_profile gpu_attention_swiglu_block env JENGINE_GPU_ATTENTION_BLOCK=1 JENGINE_GPU_SWIGLU_BLOCK=1

summarize | tee "$RUN_DIR/summary.txt"
echo "results_dir=$RUN_DIR"
