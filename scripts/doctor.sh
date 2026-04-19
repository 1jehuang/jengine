#!/usr/bin/env bash
set -euo pipefail

check() {
  local name="$1"
  local cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "[ok]   $name: $(command -v "$cmd")"
  else
    echo "[miss] $name: $cmd"
  fi
}

echo "Jengine doctor"
check "cargo" cargo
check "rustc" rustc
check "cargo-clippy" cargo-clippy
check "rustfmt" rustfmt
check "vulkaninfo" vulkaninfo
check "glslc" glslc
check "spirv-val" spirv-val
check "perf" perf
check "git" git
