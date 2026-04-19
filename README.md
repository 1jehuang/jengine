# Jengine

A Rust inference runtime for compact LLMs.

## Current status

Jengine can already run the real **PrismML Ternary-Bonsai-1.7B** model on this machine.

Working today:

- real Bonsai 1.7B weight loading
- real text-prompt CPU generation
- token-id generation path
- prompt analysis and tokenizer diagnostics
- KV-cache and working-set memory estimates
- strict ternary-g128 repacking
- packed tensor serialization
- Vulkan dense FP16 matvec baseline
- Vulkan packed ternary matvec baseline
- hybrid decode experiment with GPU-packed `q_proj`
- benchmark markdown report generation

## Immediate goal

Get the smallest PrismML Bonsai model, `Ternary-Bonsai-1.7B`, running locally with:

- a correct CPU reference path first
- validation against model metadata and weights
- measurable benchmarks at every layer
- profiling and live analysis built in from the start

## Architecture

Jengine is built in layers:

1. application / CLI
2. inference runtime
3. CPU reference backend
4. GPU backend
5. Vulkan interface

We got correctness first on CPU, then repacked and accelerated selected paths.

## Measured snapshot

Current measured real end-to-end CPU throughput is roughly:

- **~0.66 tok/s** for a representative one-token prompt run
- **~0.88 tok/s** for a short multi-token benchmark run

Current measured real Vulkan microbench results on `layer0 q_proj` `2048x2048`:

- dense FP16 matvec: **2.519 ms**
- packed ternary matvec: **1.711 ms**

See also:

- `docs/BASELINES.md`
- `docs/BENCHMARK_CAPTURE_FORMAT.md`
- `docs/VULKAN_PLAN.md`

## CLI

The top-level `jengine` binary now exposes subcommands:

```bash
cargo run --release -- inspect [root] [prompt] [max_new_tokens]
cargo run --release -- run [root] [prompt] [max_new_tokens]
cargo run --release -- bench [root] [prompt] [max_new_tokens] [iterations] [--markdown path]
cargo run --release -- profile [root] [prompt] [max_new_tokens]
cargo run --release -- validate [root] [prompt] [max_new_tokens]
cargo run --release -- pack [weights_path] [tensor_name] [rows] [cols] [out_path]
```

Default model root:

```text
/home/jeremy/models/bonsai-1.7b
```

### Examples

Inspect model state, tokenizer, weights, and Vulkan availability:

```bash
cargo run --release -- inspect /home/jeremy/models/bonsai-1.7b hello 1
```

Run one-token text generation:

```bash
cargo run --release -- run /home/jeremy/models/bonsai-1.7b hello 1
```

Run a short benchmark and emit a markdown report:

```bash
cargo run --release -- bench /home/jeremy/models/bonsai-1.7b hello 3 2 --markdown /tmp/jengine-bench.md
```

Profile one decode and print per-phase shares:

```bash
cargo run --release -- profile /home/jeremy/models/bonsai-1.7b hello 1
```

Validate assets, tokenizer, weights, and a smoke decode:

```bash
cargo run --release -- validate /home/jeremy/models/bonsai-1.7b hello 1
```

Analyze and optionally serialize a packed tensor:

```bash
cargo run --release -- pack \
  /home/jeremy/models/bonsai-1.7b/model.safetensors \
  model.layers.0.self_attn.q_proj.weight \
  2048 2048 /tmp/jengine-qproj.jtpk
```

## Quality gates

Every milestone is expected to keep these green:

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo bench --workspace --no-run`

Helper scripts:

- `scripts/check.sh`
- `scripts/bench.sh`
- `scripts/profile_cpu.sh`
- `scripts/validate_shaders.sh`

## Repo docs

- `docs/PLAN.md` - execution plan and milestones
- `docs/QUALITY_GATES.md` - required checks for every code chunk
- `docs/BONSAI_1_7B.md` - concrete model notes and constraints
- `docs/BASELINES.md` - measured CPU and runtime baseline numbers
- `docs/BENCHMARK_CAPTURE_FORMAT.md` - stable benchmark capture format
- `docs/VULKAN_PLAN.md` - Vulkan backend plan based on measured hotspots
