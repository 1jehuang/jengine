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

- earlier samples:
  - layer `0`: `qkv+gu` `1310.133 ms`, `qkvo+gu` `1338.556 ms`
  - layer `14`: `qkv+gu` `1386.601 ms`, `qkvo+gu` `1393.474 ms`
  - layer `27`: `qkv+gu` `1393.079 ms`, `qkvo+gu` `1395.680 ms`
- after the subgroup-row packed kernel improvement:
  - layer `0`: `qkv+gu` `1341.268 ms`, `qkvo+gu` `1373.316 ms`
  - layer `14`: `qkv+gu` `1387.395 ms`, `qkvo+gu` `1408.045 ms`
  - layer `27`: `qkv+gu` `1438.597 ms`, `qkvo+gu` `1475.124 ms`

So `o_proj` offload is still not the next decode-side win here, even after the stronger packed kernel landed.

### `down_proj` hybrid experiment also failed to produce a clear win

Real one-token cached hybrid comparisons on layers `0`, `14`, and `27` showed `qkv+gud` trailing `qkv+gu` again after the subgroup-row kernel improvement:

- layer `0`: `qkv+gu` `1341.268 ms`, `qkv+gud` `1341.461 ms`
- layer `14`: `qkv+gu` `1387.395 ms`, `qkv+gud` `1455.990 ms`
- layer `27`: `qkv+gu` `1438.597 ms`, `qkv+gud` `1469.006 ms`

So `down_proj` also still does not look like the next clean decode-side win in the broader hybrid path. That keeps logits and broader packed-first execution as the stronger remaining dense-side directions.

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

### Direct packed CPU fallback plus the rebuilt packed-first path dropped the combined upper bound sharply

With the latest packed-first runtime path, subgroup-row shader, and direct packed CPU fallback for both embedding lookup and matvec, the rebuilt release chunked combined upper bound is now in the mid-`4.5 s` range.

A representative recent sample is:

- total: `4557.668 ms`
- embed: `36.293 ms`
- norm: `119.368 ms`
- qkv: `601.657 ms`
- attention: `654.818 ms`
- mlp: `2872.509 ms`
  - `mlp_swiglu`: `0.395 ms`
  - `mlp_down`: `1913.680 ms`
  - `mlp_residual`: `0.088 ms`
- logits: `272.840 ms`
- compile: `872.561 ms`
- upload: `19.742 ms`
- gpu: `70.335 ms`
- download: `24.491 ms`
- non-offloaded dense: `2724.712 ms`
- orchestration: `845.815 ms`
- dispatches: `57`

That is a major improvement over the older chunked combined upper bounds. The most important new conclusion is that the **MLP stage is now clearly the largest remaining stage-level bucket, and within it `down_proj` is by far the dominant subcomponent**.

### Warm one-process chunked capture shows what cache reuse buys us

A new one-process chunked runner (`bench_packed_chunked_upper_bound`) now lets us execute the same 7-layer chunk plan twice inside one process so model-level cache reuse can actually show up in the numbers.

For the `combined` variant on this machine:

- iteration 1 aggregate:
  - total: `5869.436 ms`
  - compile: `1825.886 ms`
  - weight upload: `20.538 ms`
  - gpu: `74.905 ms`
  - download: `24.657 ms`
  - non-offloaded dense: `3076.297 ms`
  - orchestration: `847.140 ms`
  - dispatches: `57`
- iteration 2 aggregate, warm in-process:
  - total: `2819.341 ms`
  - compile: `0.000 ms`
  - weight upload: `0.000 ms`
  - gpu: `73.787 ms`
  - download: `32.924 ms`
  - non-offloaded dense: `2710.366 ms`
  - orchestration: `2.227 ms`
  - dispatches: `57`
  - `pack_cache_hits=57`
  - `gpu_cache_hits=57`

That warm in-process combined upper bound is about **`0.355 tok/s`** for one-token decode, which is the first time this packed path has clearly moved beyond the old sub-`0.25 tok/s` territory in a measurement that preserves cache reuse.

It also reinforces the same next target: once compile and weight upload are removed, the remaining dominant bucket is still the MLP tail, especially `down_proj`.

### Naive full-MLP offload is not the right next fix

An env-gated experiment (`JENGINE_PACKED_MLP_FULL=1`) that offloads `down_proj` inside the broader packed path regressed badly:

- combined chunked upper bound: `10010.763 ms`
- MLP-only chunked upper bound: `9835.970 ms`

The stage breakdown in those runs still showed the MLP stage as the largest bucket, but the total regressions confirm that simply turning `down_proj` into another standalone packed projection is **not** the right next solution.

That means the next MLP-side win likely needs a more structural approach than naive full offload, such as better packed-first staging around the MLP tail or a more fused design.

### `down_proj` still has large upside if integrated structurally

A direct real-tensor microbenchmark on `model.layers.0.mlp.down_proj.weight` with shape `2048 x 6144` now shows:

- dense CPU: `123.933 ms`
- packed CPU: `133.083 ms`
- packed GPU kernel: `0.869 ms`

So the kernel math itself is **not** the limiting issue for `down_proj`. The huge gap between the packed GPU kernel time and the current broader packed-path regressions means the real problem is integration overhead and staging, not raw GPU arithmetic throughput.

That is why naive full-MLP offload regressed even though `down_proj` is still the dominant MLP subcomponent: the broader path is not yet structured well enough to cash in the kernel-side upside.

### Packed-first generation is now the default control path for packed-artifact models

The packed-artifact `generate_greedy` / `generate_from_token_ids` path no longer falls back through the dense-style reference loop. It now routes into the packed decode path automatically when a packed model artifact is loaded.

That means the packed-first runtime is no longer just an opt-in helper or benchmark path. It is now the default generation control flow for packed-artifact models in the runtime.

### Direct packed CPU fallback also shaved the chunked combined upper bound further

After switching packed-model CPU fallback matvec and embedding lookup to use direct packed decode instead of unpacking whole tensors first, the chunked combined upper bound improved again:

- prior subgroup-row combined upper bound: `11885.064 ms`
- current combined upper bound: `11458.751 ms`

The new reconstructed combined sample is:

- total: `11458.751 ms`
- compile: `1358.502 ms`
- upload: `21.392 ms`
- gpu: `224.086 ms`
- download: `29.585 ms`
- non-offloaded dense: `9064.034 ms`
- orchestration: `761.142 ms`
- dispatches: `57`

So this CPU-side packed-first cleanup bought another roughly **`3.6%`** improvement over the earlier subgroup-row carry-up sample, and about **`6.2%`** over the older default-shader chunked combined upper bound.

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
10. A simple subgroup-aligned `32`-thread local-size tweak was effectively a wash, but a larger subgroup-row rewrite improved the real 2048x2048 packed `q_proj` microbenchmark from `1.249 ms` to `0.556 ms` median GPU time, about `2.25x` faster, and this Lunar Lake machine now auto-selects that faster path by default
11. Carrying that subgroup-row shader into the chunked combined packed path reduced the reconstructed upper bound from `12212.554 ms` to `11885.064 ms`, only about `2.7%` total, which confirms that dense-side work is still the main limiter
12. Packed-artifact `generate_greedy` now routes into the packed decode path automatically, so the packed-first runtime is no longer just benchmark-only infrastructure
13. With the rebuilt release chunked capture path, the latest combined upper bound is now in the mid-`4.5 s` range, and the stage breakdown shows `mlp_ms=2872.509` as the largest remaining stage-level bucket
14. The one-process chunked runner shows a warm in-process combined upper bound of `2819.341 ms` with `pack_cache_hits=57` and `gpu_cache_hits=57`, which is about `0.355 tok/s` for one-token decode and confirms that cache reuse now matters a lot
15. The new MLP sub-breakdown shows `mlp_down_ms=1913.680`, which means `down_proj` handling is the dominant subcomponent inside the MLP tail even though naive full offload regresses
16. A direct real-tensor microbenchmark still shows `down_proj` packed GPU kernel time at only `0.869 ms` versus `123.933 ms` dense CPU and `133.083 ms` packed CPU, which means the problem is structural integration overhead rather than raw GPU math throughput
17. A naive full-MLP packed experiment (`JENGINE_PACKED_MLP_FULL=1`) regressed badly, so the next MLP-side fix is not simply turning `down_proj` into another standalone packed projection
18. That means the next meaningful wins now come from carrying the packed-first and kernel-level improvements further into the MLP tail with a more structural redesign around `down_proj` while still reducing remaining dense-side work and synchronization overhead

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
