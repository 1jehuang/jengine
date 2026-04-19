#!/usr/bin/env bash
set -euo pipefail

if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v cargo run --release -- "$@"
else
  cargo run --release -- "$@"
fi
