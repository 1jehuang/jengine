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

### Packed benchmark harnesses now capture cleanly
From the latest real packed-artifact release runs:

- all-layer attention `qkv` sweep: `2038.344 ms`
- all-layer MLP `gu` sweep: `1827.961 ms`
- all-layer combined `qkv + gu` sweep: `3866.305 ms`
- one-token packed combined step, post-pairing: `10938.667 ms` with `84` dispatches
- one-token packed combined full-span prefill, post-pairing: `9900.304 ms` with `84` dispatches
- short-context packed `mlp`: `14441.214 ms`, about `0.069 tok/s`
- short-context packed `combined`: `11873.378 ms`, about `0.084 tok/s`

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

1. Host-side orchestration and per-dispatch overhead still dominate the packed decode path, even after reducing the combined path from `140` to `84` dispatches
2. MLP is still a major remaining cost center on the dense side
3. Compile cost is still large for first-pass all-layer sweeps, though paired dispatches help the combined path more than compile avoidance alone
4. Raw GPU upload, compute, and download time are still much smaller than total packed wall time

## Best next step

The most valuable next milestone is:

- further batch work in the packed decode path so each dispatch covers more useful projection work
- reuse compiled pipelines and runner state more aggressively across end-to-end decode steps
- reduce host-side launch overhead enough to move the combined short-context path meaningfully above the current `0.084 tok/s`
- separately stabilize `attention`-only short-context capture so it can be tracked alongside `mlp` and `combined`

## Success bar for next milestone

Reach at least:

- **<= 6.0 s** total for a one-token packed combined step on the current packed-artifact path
- **<= 6.0 s** total for a one-token packed combined full-span prefill benchmark
- short-context packed `combined` throughput clearly above the current `0.084 tok/s`
