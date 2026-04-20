# Packed decode bandwidth notes

## Scope

This note records the first-order bandwidth estimates we can already derive from the real packed Vulkan kernel measurements on this machine.

It is **not** yet a full decode-wide bandwidth profile. That still depends on stable end-to-end packed decode benchmark capture.

## Measured packed Vulkan q_proj microbenchmark

Real release-mode packed Vulkan result on:

- tensor: `model.layers.0.self_attn.q_proj.weight`
- shape: `2048 x 2048`

Observed samples:

- earlier healthy packed run:
  - compile: `99.474 ms`
  - upload: `0.013 ms`
  - gpu: `1.466 ms`
  - download: `0.047 ms`
- later cached-shader runs:
  - first: `compile 260.477 ms`, `gpu 1.351 ms`
  - second: `compile 187.654 ms`, `gpu 1.406 ms`

The compile numbers are startup overhead. The bandwidth estimate below uses the **GPU execution** window.

## Static packed weight footprint for q_proj

From the packed tensor work, the real `q_proj` tensor is about:

- packed file size: about `1,179,814 bytes`

That is close to the in-memory packed payload used by the GPU path.

## First-order effective weight bandwidth estimate

If we treat one kernel run as streaming roughly one packed tensor payload through the GPU in:

- about `1.35 ms` to `1.47 ms`

then the corresponding first-order effective bandwidth is roughly:

- `1,179,814 / 0.001466 ≈ 0.80 GB/s`
- `1,179,814 / 0.001351 ≈ 0.87 GB/s`

So a simple estimate is:

- **about `0.8 to 0.9 GB/s` effective packed weight bandwidth**

## Host-visible transfer estimate

The same packed Vulkan runs show:

- upload: about `0.013 ms`
- download: about `0.044` to `0.049 ms`

The host-visible input payload is small because the packed path uploads a packed activation vector, not the whole weight tensor each run.

For `2048` inputs:

- packed activation upload is about `1024 * 4 = 4096 bytes`
- output download is about `2048 * 4 = 8192 bytes`

That implies rough host-visible transfer rates of:

- upload: `4096 / 0.000013 ≈ 315 MB/s`
- download: `8192 / 0.000047 ≈ 174 MB/s`

These numbers are much smaller than system memory bandwidth and confirm that current per-dispatch host-visible transfer volume is not the main bottleneck for this microkernel.

## Interpretation

Current takeaway:

- packed kernel GPU execution itself is healthy
- host-visible per-dispatch transfer volume is small
- compile/setup cost is still large relative to compute
- the remaining big unknown is decode-wide orchestration cost across many projections and layers

## Next profiling step

When the full packed decode benchmark capture is stable, compute the same estimates for:

- total upload bytes per token
- total download bytes per token
- dispatch count per token
- total packed GPU time per token

That will let us estimate effective decode-wide packed bandwidth instead of just kernel-local bandwidth.
