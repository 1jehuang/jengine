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

### `logits` hybrid experiment became the first clear dense-hotspot win after removing full-vector download

After switching the hybrid logits path to scan argmax directly from the mapped GPU output buffer instead of materializing a full logits `Vec<f32>`, three-sample one-token medians on layers `0`, `14`, and `27` showed:

- layer `0`: `qkv+gu` `1397.266 ms`, `qkv+gu+logits` `1377.805 ms`
- layer `14`: `qkv+gu` `1447.585 ms`, `qkv+gu+logits` `1372.383 ms`
- layer `27`: `qkv+gu` `1459.991 ms`, `qkv+gu+logits` `1387.104 ms`

So logits is now the first remaining dense-hotspot experiment with median wins across all three sampled layers. That makes output-side handling the strongest next dense-side lead, and it also confirms that avoiding unnecessary output downloads matters on this stack.

### Chunked capture now works around the `rc=143` kill window

By chaining `bench_packed_prefill_chunk` across four 7-layer spans and only including logits on the last chunk, we can now reconstruct full packed one-token upper bounds without each individual chunk getting killed by the harness.

Current chunked reconstructed samples:

- combined:
  - total: `12212.554 ms`
  - compile: `1712.426 ms`
  - upload: `22.052 ms`
  - gpu: `657.716 ms`
  - download: `35.070 ms`
  - non-offloaded dense: `8987.582 ms`
  - orchestration: `797.694 ms`
  - dispatches: `57`
- attention-only:
  - total: `17779.632 ms`
  - compile: `464.922 ms`
  - upload: `9.696 ms`
  - gpu: `108.760 ms`
  - download: `22.253 ms`
  - non-offloaded dense: `16865.714 ms`
  - orchestration: `308.283 ms`
  - dispatches: `29`

Important caveat:

- these are **upper bounds**, not clean apples-to-apples replacements for the one-process packed step benchmark
- each chunk runs in a fresh process, so cross-chunk runner reuse and weight residency are lost
- that means compile and upload costs are overstated relative to the intended warm in-process path

Still, this is useful because it confirms the latest packed path structure and dispatch counts while giving us a kill-window-safe capture path until the direct full-step benchmark can complete cleanly.

The key conclusion from the new attention-only sample is that attention-side offload alone is **not** beating the current combined packed path here.

### Xe2-relevant Vulkan capabilities are present on this Intel iGPU

A new local hardware probe now confirms that the Lunar Lake iGPU exposes the key Vulkan features we would want for a more Xe2-friendly kernel path:

- subgroup size: `32`
- subgroup size control: `true`
- compute full subgroups: `true`
- min subgroup size: `16`
- max subgroup size: `32`
- integer dot product: `true`
- shader float16: `true`
- shader int8: `true`
- `VK_KHR_cooperative_matrix`: `true`
- `VK_KHR_shader_integer_dot_product`: `true`
- `VK_EXT_subgroup_size_control`: `true`

That does **not** prove a better kernel will be faster yet, but it does mean the hardware and driver are exposing the right building blocks. So an Xe2/XMX-oriented shader path is now a credible engineering direction rather than a speculative one.

### First larger Xe2-oriented shader rewrite is a real microbenchmark win

The first meaningful subgroup-oriented rewrite now exists as `JENGINE_PACKED_SHADER_VARIANT=xe2_subgroup_row`, which assigns one workgroup per row and uses subgroup reduction across the packed-column loop.

Repeated real `q_proj` tensor samples on the 2048x2048 tensor gave:

- default packed shader median GPU time: `1.249 ms`
- subgroup-row Xe2 shader median GPU time: `0.556 ms`

That is about a **`2.25x`** GPU-time speedup on this real packed matvec microbenchmark.

So the Xe2 investigation now has a concrete result:

- simple local-size retuning alone was not enough
- but a more subgroup-oriented kernel structure **can** materially improve packed throughput on this Intel Arc iGPU
- the next question is how much of that microbenchmark win survives once it is threaded through the broader packed runtime

### Subgroup-row kernel win only partially survives in the combined packed upper bound

Using the chunked kill-window-safe combined capture path with `JENGINE_PACKED_SHADER_VARIANT=xe2_subgroup_row` produced this reconstructed upper bound:

- total: `11885.064 ms`
- compile: `1364.736 ms`
- upload: `20.824 ms`
- gpu: `250.854 ms`
- download: `30.269 ms`
- non-offloaded dense: `9445.204 ms`
- orchestration: `773.164 ms`
- dispatches: `57`

Compared with the earlier default-shader chunked combined upper bound:

- total improved from `12212.554 ms` to `11885.064 ms`
- gpu time improved from `657.716 ms` to `250.854 ms`

So the kernel-level win is real and it does survive upward, but only as about a **`2.7%`** total upper-bound improvement in this current broader packed path because dense-side work still dominates.

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
6. After removing full logits-vector download from the hybrid logits path, `qkv+gu+logits` became the first remaining dense-hotspot experiment with median wins across all three sampled layers
7. Chunked packed capture now works around the `rc=143` kill window and confirms the latest path shapes, with chunked upper bounds of `12212.554 ms` for combined at `57` dispatches and `17779.632 ms` for attention-only at `29` dispatches
8. The new chunked attention-only capture shows that attention-side offload alone is still losing badly to the current combined packed path here
9. The Intel Lunar Lake Vulkan stack does expose cooperative matrix, integer dot, subgroup size control, and float16/int8 features, so an Xe2-oriented kernel path is plausible and worth active investigation
10. A simple subgroup-aligned `32`-thread local-size tweak was effectively a wash, but a larger subgroup-row rewrite improved the real 2048x2048 packed `q_proj` microbenchmark from `1.249 ms` to `0.556 ms` median GPU time, about `2.25x` faster
11. Carrying that subgroup-row shader into the chunked combined packed path reduced the reconstructed upper bound from `12212.554 ms` to `11885.064 ms`, only about `2.7%` total, which confirms that dense-side work is still the main limiter
12. The next meaningful wins now come from carrying that kind of kernel-level win through more of the broader runtime while still reducing dense-side work and synchronization overhead

## Best next step

The most valuable next milestone is:

- further batch work in the packed decode path so each dispatch covers more useful projection work
- reuse compiled pipelines and runner state more aggressively across end-to-end decode steps
- reduce host-side launch overhead enough to move the combined short-context path meaningfully above the current `0.084 tok/s`
- follow up on logits offload first, because removing the full logits-vector copy turned it into the first dense-hotspot variant with median wins across all three sampled layers
- use the new chunked packed capture path as a temporary measurement workaround while the direct long combined benchmarks are still being killed with `rc=143`
- keep pushing toward broader packed-first execution changes so small packed wins are not erased by dense-side bounce-back and output transfer overhead
- separately stabilize `attention`-only short-context capture so it can be tracked alongside `mlp` and `combined`

## Success bar for next milestone

Reach at least:

- **<= 6.0 s** total for a one-token packed combined step on the current packed-artifact path
- **<= 6.0 s** total for a one-token packed combined full-span prefill benchmark
- short-context packed `combined` throughput clearly above the current `0.084 tok/s`
