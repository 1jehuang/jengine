#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export TMPDIR=${TMPDIR:-"$ROOT_DIR/.tmp"}
mkdir -p "$TMPDIR"

cd "$ROOT_DIR"

cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets
cargo bench --workspace --no-run

echo "quick_ci=ok"
