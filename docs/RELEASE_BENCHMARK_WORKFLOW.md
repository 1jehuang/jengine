# Release benchmark workflow

This document defines the stable fixture set and pinned commands used for release-mode benchmark comparisons.

## Fixture source

Use:

- `fixtures/release_benchmark.env`

Default pinned values:

- model root: `/home/jeremy/models/bonsai-1.7b`
- weights path: `/home/jeremy/models/bonsai-1.7b/model.safetensors`
- short prompt: `hello`
- tensor: `model.layers.0.self_attn.q_proj.weight`
- tensor shape: `2048 x 2048`
- layer: `0`
- token id: `42`

## Pinned release commands

Build once:

```bash
cargo build --release \
  --bin jengine \
  --bin run_reference \
  --bin bench_real_text \
  --bin compare_reference_projection \
  --bin vulkan_dense_matvec \
  --bin vulkan_packed_matvec \
  --bin bench_hybrid_qproj
```

### CPU one-token reference

```bash
./target/release/run_reference /home/jeremy/models/bonsai-1.7b hello 1
```

### CPU short text benchmark

```bash
./target/release/bench_real_text /home/jeremy/models/bonsai-1.7b hello 3 1
```

### Runtime-integrated packed q_proj comparison

```bash
./target/release/compare_reference_projection /home/jeremy/models/bonsai-1.7b 0 42
```

### Vulkan dense q_proj baseline

```bash
./target/release/vulkan_dense_matvec \
  /home/jeremy/models/bonsai-1.7b/model.safetensors \
  model.layers.0.self_attn.q_proj.weight \
  2048 2048
```

### Vulkan packed q_proj baseline

```bash
./target/release/vulkan_packed_matvec \
  /home/jeremy/models/bonsai-1.7b/model.safetensors \
  model.layers.0.self_attn.q_proj.weight \
  2048 2048
```

### Hybrid q_proj decode baseline

```bash
./target/release/bench_hybrid_qproj /home/jeremy/models/bonsai-1.7b hello 1 0
```

## One-command capture

Use:

```bash
./scripts/capture_release_baselines.sh
```

This script defaults `TMPDIR` to a repo-local `.tmp/` directory so benchmark capture stays stable even when the system temp area is quota-limited.

This writes a markdown report under:

```text
.artifacts/release-baselines/
```

## Comparison rules

When comparing two runs, hold these constant:

- same git commit or explicitly recorded commit delta
- same fixture env file
- same release binaries
- same prompt and token counts
- same tensor, layer, and token id
- same machine and power state when possible

## Minimum metrics to record

For CPU decode runs:

- total ms
- qkv ms
- attention ms
- mlp ms
- logits ms
- generated tok/s
- memory report

For projection and GPU runs:

- compile ms
- execution ms
- upload ms when applicable
- download ms when applicable
- max abs diff
- mean abs diff when available
