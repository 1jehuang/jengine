# Jengine project status

## Current state

Jengine can now:

- load the real Bonsai 1.7B model
- run a CPU reference decode path on real prompts
- repack ternary-compatible tensors into a packed ternary-g128 format
- run Vulkan dense and packed q_proj matvec baselines
- cache packed q_proj data across runs
- cache Vulkan q_proj execution resources across runs
- run and measure a real cached hybrid q_proj decode path

## Major milestones completed

### CPU bring-up
- model config parsing
- safetensors inventory scanning
- tokenizer loading with fallback path
- CPU reference decode path
- prompt regression harness

### Repacking
- ternary packability analysis
- packed tensor format
- strict repacker and correctness checks
- packed CPU reference matvec

### GPU bring-up
- Vulkan device discovery
- Vulkan dense FP16 matvec baseline
- Vulkan packed ternary matvec baseline
- hybrid q_proj decode path
- cached hybrid q_proj runtime path

### Benchmarking and reporting
- release benchmark fixture file
- reproducible release benchmark workflow
- markdown benchmark report generation
- baseline docs
- explicit performance target doc

## Latest measured highlights

### Cached q_proj warm hybrid vs dense
From the latest real one-token run:

- dense total: `9822.266 ms`
- cached q_proj warm total: `5475.410 ms`
- speedup: about `1.79x`

### Cached warm GPU-side q_proj costs
- upload: `22.279 ms`
- compute: `15.424 ms`
- download: `0.238 ms`

## Current bottlenecks

1. MLP is still a major remaining cost center
2. Only q_proj is currently offloaded in the hybrid runtime path
3. The rest of attention-side projections still run on CPU
4. Upload and compute costs are now visible after removing repack and compile costs

## Best next step

The most valuable next milestone is:

- extend cached packed GPU offload to `k_proj`, `v_proj`, and `o_proj`
- benchmark projection mixes like `q`, `qkv`, and `qkvo`
- then expand into MLP projections

## Success bar for next milestone

Reach at least:

- **<= 4.0 s** total for a one-token cached attention-only hybrid run
- **>= 0.25 tok/s** generated throughput
