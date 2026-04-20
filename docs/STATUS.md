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

### Decode-wide packed attribution now exists
From the latest real combined packed decode attribution run:

- total: `12762.262 ms`
- non-offloaded dense: `11309.553 ms`
- orchestration: `528.955 ms`
- weight upload: `82.525 ms`
- activation upload: `3.417 ms`
- gpu: `714.748 ms`
- download: `29.558 ms`
- end-to-end effective bandwidth: `0.042 GB/s`
- stream-window effective bandwidth: `0.642 GB/s`
- percent of `137 GB/s` hardware ceiling in stream-window terms: about `0.468%`
- resident per-projection GPU runners are now reusing weights and compiled state across decode steps within a session, evidenced by `gpu_cache_hits=166` across `168` dispatches and only one step's worth of weight uploads

### Packed benchmark harnesses now capture cleanly
From the latest real packed-artifact release runs:

- all-layer attention `qkv` sweep: `2038.344 ms`
- all-layer MLP `gu` sweep: `1827.961 ms`
- all-layer combined `qkv + gu` sweep: `3866.305 ms`
- one-token packed combined step, post-pairing: `10938.667 ms` with `84` dispatches
- one-token packed combined step, post-name-cache: `10077.689 ms` with `84` dispatches
- one-token packed combined full-span prefill, post-pairing: `9900.304 ms` with `84` dispatches
- short-context packed `mlp`: `14441.214 ms`, about `0.069 tok/s`
- short-context packed `combined`: `11873.378 ms`, about `0.084 tok/s`

### `o_proj` hybrid experiment did not beat `qkv+gu`

Real one-token cached hybrid comparisons on layers `0`, `14`, and `27` all kept `qkv+gu` ahead of `qkvo+gu`:

- layer `0`: `qkv+gu` `1310.133 ms`, `qkvo+gu` `1338.556 ms`
- layer `14`: `qkv+gu` `1386.601 ms`, `qkvo+gu` `1393.474 ms`
- layer `27`: `qkv+gu` `1393.079 ms`, `qkvo+gu` `1395.680 ms`

So `o_proj` offload is not the next obvious decode-side win on this stack. The next dense-hotspot focus should shift toward `down_proj` and logits.

### `down_proj` hybrid experiment also failed to produce a clear win

Real one-token cached hybrid comparisons on layers `0`, `14`, and `27` showed `qkv+gud` mostly trailing `qkv+gu` and only edging it out by noise-level margin once:

- layer `0`: `qkv+gu` `1313.025 ms`, `qkv+gud` `1319.966 ms`
- layer `14`: `qkv+gu` `1348.146 ms`, `qkv+gud` `1359.994 ms`
- layer `27`: `qkv+gu` `1438.766 ms`, `qkv+gud` `1436.803 ms`

So `down_proj` also does not look like the next clean decode-side win in the current hybrid form. That leaves logits and larger packed-first execution changes as the more promising next dense-side directions.

### `down_proj` hybrid experiment also failed to produce a clear win

Real one-token cached hybrid comparisons on layers `0`, `14`, and `27` showed `qkv+gud` mostly trailing `qkv+gu` and only edging it out by noise-level margin once:

- layer `0`: `qkv+gu` `1313.025 ms`, `qkv+gud` `1319.966 ms`
- layer `14`: `qkv+gu` `1348.146 ms`, `qkv+gud` `1359.994 ms`
- layer `27`: `qkv+gu` `1438.766 ms`, `qkv+gud` `1436.803 ms`

So `down_proj` also does not look like the next clean decode-side win in the current hybrid form. That leaves logits and larger packed-first execution changes as the more promising next dense-side directions.

### `logits` hybrid experiment is mixed but still promising

Three-sample one-token cached hybrid medians on layers `0`, `14`, and `27` showed:

- layer `0`: `qkv+gu` `1378.146 ms`, `qkv+gu+logits` `1388.570 ms`
- layer `14`: `qkv+gu` `1451.116 ms`, `qkv+gu+logits` `1623.055 ms`
- layer `27`: `qkv+gu` `1466.605 ms`, `qkv+gu+logits` `1344.904 ms`

So logits offload is not a universal win yet, but unlike `o_proj` and `down_proj` it does show a meaningful gain on the late sampled layer. That makes logits a better next dense-hotspot candidate than the other two projection tails, especially if we can reduce the extra upload and download cost around the large vocab output.

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

1. The dominant cost in the first decode-wide attribution sample is still non-offloaded dense work, not GPU bandwidth saturation
2. Host-side orchestration and per-dispatch overhead still matter, but they are now smaller after resident-runner reuse and a tensor-name caching pass that reduced the rebuilt combined one-token step from `10938.667 ms` to `10077.689 ms`
3. Raw GPU upload, compute, and download time are still much smaller than total packed wall time, and measured decode-wide bandwidth is far below the hardware ceiling
4. The `o_proj` hybrid experiment did not beat `qkv+gu` on layers `0`, `14`, or `27`, so broadening attention-side offload blindly is not the next win
5. The follow-up `down_proj` hybrid experiment also failed to produce a clear end-to-end gain, so it is not the best next dense-side bet either
6. The logits hybrid experiment is mixed, but it is the first remaining dense-hotspot test that showed a meaningful win on at least one sampled late layer, so it is worth deeper follow-up
7. The next meaningful wins now come from reducing dense-side work and synchronization overhead, not from merely making runner reuse exist at all

## Best next step

The most valuable next milestone is:

- further batch work in the packed decode path so each dispatch covers more useful projection work
- reuse compiled pipelines and runner state more aggressively across end-to-end decode steps
- reduce host-side launch overhead enough to move the combined short-context path meaningfully above the current `0.084 tok/s`
- follow up on logits offload specifically, since it showed a real late-layer win even though it is not stable enough for broad rollout yet
- keep pushing toward broader packed-first execution changes so small packed wins are not erased by dense-side bounce-back and output transfer overhead
- separately stabilize `attention`-only short-context capture so it can be tracked alongside `mlp` and `combined`

## Success bar for next milestone

Reach at least:

- **<= 6.0 s** total for a one-token packed combined step on the current packed-artifact path
- **<= 6.0 s** total for a one-token packed combined full-span prefill benchmark
- short-context packed `combined` throughput clearly above the current `0.084 tok/s`
