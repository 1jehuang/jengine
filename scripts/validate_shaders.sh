#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

shader_files=(shaders/*.comp shaders/*.vert shaders/*.frag)
if [ ${#shader_files[@]} -eq 0 ]; then
  echo "No shader files found under shaders/"
  exit 0
fi

for shader in "${shader_files[@]}"; do
  out="${shader}.spv"
  echo "Compiling $shader -> $out"
  glslc "$shader" -o "$out"
  spirv-val "$out"
  rm -f "$out"
done
