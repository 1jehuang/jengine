# Jengine baseline measurements

For reproducible release-mode capture, use:

- `fixtures/release_benchmark.env`
- `docs/RELEASE_BENCHMARK_WORKFLOW.md`
- `scripts/capture_release_baselines.sh`

## Latest verified release-style measurements

These values are the latest verified measurements gathered on this machine from successful real-model runs and tensor microbenchmarks.

- CPU one-token text prompt run: about **1513.965 ms** total, about **0.66 tok/s**
- CPU short text benchmark: about **3418.009 ms** for 3 generated tokens, about **0.88 tok/s**
- CPU packed q_proj matvec optimization: about **1.73x** faster than the prior packed reference path on the real 2048x2048 tensor
- Vulkan dense q_proj matvec: about **2.519 ms** GPU execution
- Vulkan packed q_proj matvec: about **1.711 ms** GPU execution
- Hybrid q_proj decode: works correctly, but still slower end to end because pack and pipeline setup are not cached yet
- Cached hybrid q_proj decode: warm-cache run reduces total time further and removes pack/compile cost entirely

## Real Bonsai 1.7B CPU runs

### One-token text prompt run

Command shape:

```bash
cargo run --release --bin run_reference -- /home/jeremy/models/bonsai-1.7b hello 1
```

Observed samples:

- earlier sample:
  - `prompt_tokens=1`
  - `generated_tokens=1`
  - `total_ms=1513.965`
  - `embed_ms=0.490`
  - `norm_ms=56.923`
  - `qkv_ms=248.663`
  - `attention_ms=111.360`
  - `mlp_ms=849.401`
  - `logits_ms=246.633`
- current post-index-cache sample:
  - `prompt_tokens=1`
  - `generated_tokens=1`
  - `total_ms=4690.908`
  - `embed_ms=18.935`
  - `norm_ms=97.711`
  - `qkv_ms=1093.304`
  - `attention_ms=277.480`
  - `mlp_ms=2497.849`
  - `logits_ms=702.435`

### One-token token-id run

Command shape:

```bash
./target/release/run_reference_ids /home/jeremy/models/bonsai-1.7b 42 1
```

Observed sample:

- `prompt_tokens=1`
- `generated_tokens=1`
- `total_ms=2527.042`
- `embed_ms=0.557`
- `norm_ms=53.924`
- `qkv_ms=520.173`
- `attention_ms=201.257`
- `mlp_ms=1440.222`
- `logits_ms=308.975`

### Multi-token text benchmark

Command shape:

```bash
cargo run --release --bin bench_real_text -- /home/jeremy/models/bonsai-1.7b hello 3 2
```

Observed samples:

- iteration 1 total: `3627.780 ms`
- iteration 2 total: `3208.238 ms`
- average total: `3418.009 ms`
- average qkv: `595.950 ms`
- average attention: `255.213 ms`
- average mlp: `1942.555 ms`

## Repack baselines

These tensor-level measurements use the pinned release tensor fixture from `fixtures/release_benchmark.env`.

### Real q_proj tensor pack

Command shape:

```bash
cargo run --release --bin pack_real_tensor -- \
  /home/jeremy/models/bonsai-1.7b/model.safetensors \
  model.layers.0.self_attn.q_proj.weight \
  2048 2048
```

Observed sample:

- elements: `4,194,304`
- groups: `32,768`
- perfect groups: `32,768`
- estimated packed bytes: `1,114,112`
- reduction: `7.529x`
- strict pack time: `68.949 ms`

## Packed matvec baselines

These projection-level measurements use the pinned release tensor fixture from `fixtures/release_benchmark.env`.

### Real q_proj dense vs packed

Command shape:

```bash
cargo run --release --bin compare_packed_matvec -- \
  /home/jeremy/models/bonsai-1.7b/model.safetensors \
  model.layers.0.self_attn.q_proj.weight \
  2048 2048
```

Observed samples:

- earlier sample:
  - dense: `8.391 ms`
  - packed reference: `34.038 ms`
  - packed vs dense speedup: `0.247x`
  - max abs diff: `0.000000`
- current post-grouped-packed sample:
  - dense: `3.691 ms`
  - packed reference: `34.492 ms`
  - packed optimized: `19.933 ms`
  - packed optimized vs packed reference speedup: `1.730x`
  - packed optimized vs dense speedup: `0.185x`
  - max abs diff vs dense for packed reference: `0.000000`
  - max abs diff vs dense for packed optimized: `0.000001`

### Runtime-integrated q_proj dense vs packed

Command shape:

```bash
cargo run --release --bin compare_reference_projection -- /home/jeremy/models/bonsai-1.7b 0 42
```

Observed sample:

- dense: `2.388 ms`
- pack: `58.412 ms`
- packed projection: `33.896 ms`
- max abs diff: `0.000000`
- mean abs diff: `0.000000`

## Hybrid decode baselines

### Real q_proj hybrid decode comparison

Command shape:

```bash
TMPDIR=$PWD/.tmp ./target/release/bench_hybrid_qproj \
  /home/jeremy/models/bonsai-1.7b hello 1 0 .artifacts/hybrid_cached.txt
```

Observed sample:

- dense total: `9822.266 ms`
- hybrid uncached total: `5920.685 ms`
  - pack: `56.546 ms`
  - gpu compile: `447.967 ms`
  - gpu upload: `0.048 ms`
  - gpu compute: `2.524 ms`
  - gpu download: `0.147 ms`
- hybrid cached first total: `6076.817 ms`
  - pack cache hit: `false`
  - gpu cache hit: `false`
  - gpu compile: `228.059 ms`
- hybrid cached warm total: `5475.410 ms`
  - pack cache hit: `true`
  - gpu cache hit: `true`
  - gpu upload: `22.279 ms`
  - gpu compute: `15.424 ms`
  - gpu download: `0.238 ms`

## Attention projection mix baselines

### Real one-layer attention projection mix comparison

Command shape:

```bash
TMPDIR=$PWD/.tmp cargo run --release --bin bench_attention_projection_mix -- \
  /home/jeremy/models/bonsai-1.7b 0 42 .artifacts/attention_mix.txt
```

Observed sample:

- `q` variant:
  - total: `21.431 ms`
  - pack: `69.883 ms`
  - compile: `209.190 ms`
  - upload: `0.024 ms`
  - gpu: `1.296 ms`
  - download: `0.067 ms`
  - max abs diff: `0.000000`
- `qkv` variant:
  - total: `8.299 ms`
  - pack: `63.558 ms`
  - compile: `266.649 ms`
  - upload: `0.060 ms`
  - gpu: `3.402 ms`
  - download: `0.087 ms`
  - max abs diff: `0.001472`
- `qkvo` variant:
  - total: `216.190 ms`
  - pack: `68.946 ms`
  - compile: `131.283 ms`
  - upload: `0.150 ms`
  - gpu: `6.019 ms`
  - download: `0.434 ms`
  - max abs diff: `0.001204`

Current interpretation:

- `qkv` is currently the best attention-side projection mix among these three tested variants
- `qkvo` is currently much slower, so `o_proj` offload needs more investigation before rolling it into a broader decode path

## MLP projection mix baselines

### Real one-layer MLP projection mix comparison

Command shape:

```bash
TMPDIR=$PWD/.tmp cargo run --release --bin bench_mlp_projection_mix -- \
  /home/jeremy/models/bonsai-1.7b 0 42 .artifacts/mlp_mix.txt
```

Observed sample:

- `gu` variant:
  - total: `31.034 ms`
  - pack: `379.373 ms`
  - compile: `301.464 ms`
  - upload: `0.063 ms`
  - gpu: `5.883 ms`
  - download: `0.468 ms`
  - max abs diff: `0.000280`
- `gud` variant:
  - total: `469.963 ms`
  - pack: `211.331 ms`
  - compile: `211.125 ms`
  - upload: `0.219 ms`
  - gpu: `10.374 ms`
  - download: `0.321 ms`
  - max abs diff: `0.000515`

Current interpretation:

- `gate_proj + up_proj` currently looks like the promising first MLP offload slice
- adding `down_proj` in the current form is much slower and needs separate investigation before broader rollout

## Broader hybrid decode baselines

### Real cached `qkv + gu` decode comparison

Command shape:

```bash
TMPDIR=$PWD/.tmp cargo run --release --bin bench_hybrid_qkv_gu -- \
  /home/jeremy/models/bonsai-1.7b hello 1 0 .artifacts/hybrid_qkv_gu.txt
```

Observed sample:

- dense total: `4410.178 ms`
- hybrid `qkv + gu` total: `4613.321 ms`
- pack: `548.142 ms`
- compile: `1110.788 ms`
- upload: `11.568 ms`
- gpu: `32.298 ms`
- download: `1.129 ms`

Current interpretation:

- this broader `qkv + gu` mix is **not yet faster than dense**
- the current bottleneck is no longer just whether individual projection slices work
- the current bottleneck is now the aggregate cost of:
  - pack/setup work
  - cached runner initialization footprint across more projections
  - transfer / execution overhead once more projections participate

### Real cached `qkv + gu` vs `qkvo + gu` layer samples

Command shape:

```bash
JENGINE_NO_HEARTBEAT=1 ./target/release/bench_hybrid_qkv_gu \
  /home/jeremy/models/bonsai-1.7b hello 1 <layer> <variant>
```

Observed samples:

- layer `0`
  - `qkv+gu`: `1310.133 ms`
  - `qkvo+gu`: `1338.556 ms`
- layer `14`
  - `qkv+gu`: `1386.601 ms`
  - `qkvo+gu`: `1393.474 ms`
- layer `27`
  - `qkv+gu`: `1393.079 ms`
  - `qkvo+gu`: `1395.680 ms`

Current interpretation:

- `qkv+gu` remains slightly faster than `qkvo+gu` on early, middle, and late sampled layers
- `o_proj` offload is therefore not the next obvious decode-side win on this stack
- the next dense-hotspot experiments should shift toward `down_proj` and logits

### Real cached `qkv + gu` vs `qkv + gud` layer samples

Command shape:

```bash
JENGINE_NO_HEARTBEAT=1 ./target/release/bench_hybrid_qkv_gu \
  /home/jeremy/models/bonsai-1.7b hello 1 <layer> <variant>
```

Observed samples:

- layer `0`
  - `qkv+gu`: `1313.025 ms`
  - `qkv+gud`: `1319.966 ms`
- layer `14`
  - `qkv+gu`: `1348.146 ms`
  - `qkv+gud`: `1359.994 ms`
- layer `27`
  - `qkv+gu`: `1438.766 ms`
  - `qkv+gud`: `1436.803 ms`

Current interpretation:

- `qkv+gud` does not produce a clear end-to-end win over `qkv+gu`
- the tiny layer `27` edge is within noise compared with the losses on layers `0` and `14`
- `down_proj` therefore does not look like the next clean hybrid decode-side win either
- the next dense-hotspot experiments should bias toward logits and larger packed-first execution changes

## Packed layer sweep baselines

### Real all-layer packed projection sweep

Command shape:

```bash
env JENGINE_NO_HEARTBEAT=1 ./target/release/bench_packed_layer_sweep \
  /home/jeremy/models/bonsai-1.7b .artifacts/jengine-packed-model 42
```

Observed sample:

- attention `qkv` over all 28 layers:
  - total: `2038.344 ms`
  - pack: `0.000 ms`
  - compile: `1350.200 ms`
  - upload: `15.630 ms`
  - gpu: `114.352 ms`
  - download: `4.856 ms`
  - max abs diff: `0.094574`
- MLP `gu` over all 28 layers:
  - total: `1827.961 ms`
  - pack: `0.000 ms`
  - compile: `751.991 ms`
  - upload: `32.522 ms`
  - gpu: `165.853 ms`
  - download: `8.977 ms`
  - max abs diff: `0.168579`
- combined `qkv + gu` over all 28 layers:
  - total: `3866.305 ms`
  - pack: `0.000 ms`
  - compile: `2102.191 ms`
  - upload: `48.151 ms`
  - gpu: `280.205 ms`
  - download: `13.833 ms`

Current interpretation:

- compile cost still dominates the first full-layer sweep, even with persistent packed artifacts
- raw GPU execution plus transfer time is much smaller than total wall time in this first-pass sweep
- the next performance win should come from stronger pipeline reuse across benchmark phases and decode steps

## Packed step-id baselines

### Real one-token token-id packed step comparison

Command shape:

```bash
env JENGINE_NO_HEARTBEAT=1 ./target/release/bench_packed_step_ids \
  /home/jeremy/models/bonsai-1.7b .artifacts/jengine-packed-model 42 <variant> .artifacts/step_ids_<variant>.txt
```

Observed samples:

- dense one-token step:
  - total: `1599.766 ms`
  - embed: `0.008 ms`
  - norm: `8.013 ms`
  - qkv: `383.907 ms`
  - attention: `80.507 ms`
  - mlp: `924.377 ms`
  - logits: `202.766 ms`
- packed attention `qkv` step:
  - total: `17771.496 ms`
  - compile: `122.139 ms`
  - upload: `16.057 ms`
  - gpu: `109.337 ms`
  - download: `3.969 ms`
  - gpu cache hits: `82`
  - dispatches: `84`
  - upload bytes: `66404352`
  - download bytes: `458752`
- packed MLP `gu` step:
  - total: `11816.556 ms`
  - compile: `52.621 ms`
  - upload: `35.406 ms`
  - gpu: `178.836 ms`
  - download: `8.309 ms`
  - gpu cache hits: `55`
  - dispatches: `56`
  - upload bytes: `198410240`
  - download bytes: `1376256`
- packed combined `qkv + gu` step:
  - earlier sample:
    - total: `10359.618 ms`
    - compile: `72.471 ms`
    - upload: `58.044 ms`
    - gpu: `282.698 ms`
    - download: `18.571 ms`
    - gpu cache hits: `137`
    - dispatches: `140`
    - upload bytes: `264814592`
    - download bytes: `1835008`
  - post-pairing sample:
    - total: `10938.667 ms`
    - compile: `30.393 ms`
    - upload: `46.625 ms`
    - gpu: `320.308 ms`
    - download: `24.915 ms`
    - gpu cache hits: `82`
    - dispatches: `84`
    - upload bytes: `264585216`
    - download bytes: `1835008`
  - post-name-cache sample:
    - total: `10077.689 ms`
    - compile: `59.880 ms`
    - upload: `42.627 ms`
    - gpu: `350.575 ms`
    - download: `12.680 ms`
    - gpu cache hits: `82`
    - dispatches: `84`
    - upload bytes: `264585216`
    - download bytes: `1835008`

Current interpretation:

- lower-level packed projection execution is working, but end-to-end one-token packed step latency is still much worse than dense
- the packed combined step is materially better than packed attention-only or packed MLP-only in this benchmark, but still far from competitive overall
- host-side orchestration and per-dispatch overhead now look like the main bottleneck in the packed step path

## Packed prefill chunk baselines

### Real full-span packed prefill chunk comparison

Command shape:

```bash
env JENGINE_NO_HEARTBEAT=1 ./target/release/bench_packed_prefill_chunk \
  /home/jeremy/models/bonsai-1.7b .artifacts/jengine-packed-model 42 0 28 <variant> - - \
  .artifacts/prefill_<variant>_full.txt 1
```

Observed samples:

- packed attention `qkv` full-span prefill:
  - total: `18730.666 ms`
  - compile: `29.374 ms`
  - upload: `16.616 ms`
  - gpu: `107.570 ms`
  - download: `5.592 ms`
  - gpu cache hits: `82`
  - dispatches: `84`
- packed MLP `gu` full-span prefill:
  - total: `11987.256 ms`
  - compile: `45.577 ms`
  - upload: `36.469 ms`
  - gpu: `181.333 ms`
  - download: `8.122 ms`
  - gpu cache hits: `55`
  - dispatches: `56`
- packed combined `qkv + gu` full-span prefill:
  - earlier sample:
    - total: `10258.111 ms`
    - compile: `84.826 ms`
    - upload: `57.067 ms`
    - gpu: `283.962 ms`
    - download: `16.414 ms`
    - gpu cache hits: `137`
    - dispatches: `140`
  - post-pairing sample:
    - total: `9900.304 ms`
    - compile: `50.981 ms`
    - upload: `43.633 ms`
    - gpu: `322.363 ms`
    - download: `18.288 ms`
    - gpu cache hits: `82`
    - dispatches: `84`

Current interpretation:

- chunk-level prefill results closely track the step-id benchmark and confirm the same ranking: combined is best among the packed variants tested here
- compile time is no longer the dominant cost in the chunk benchmark once runner shapes are reused within a single run
- paired dispatching cuts the combined path from `140` to `84` dispatches and improves the full-span prefill sample, but total wall time is still dominated by non-kernel overhead

## Packed model artifact baselines

### Real Bonsai packed artifact creation

Command shape:

```bash
TMPDIR=$PWD/.tmp cargo run --release --bin pack_model_artifact -- \
  /home/jeremy/models/bonsai-1.7b .artifacts/jengine-packed-model
```

Observed sample:

- entry count: `197`
- packed total bytes: `483755614`
- source file bytes: `3440091640`
- reduction: `7.111x`

### Real packed artifact validation

Command shape:

```bash
TMPDIR=$PWD/.tmp cargo run --release --bin validate_packed_model_artifact -- \
  /home/jeremy/models/bonsai-1.7b .artifacts/jengine-packed-model
```

Observed sample:

- checked entries: `197`
- max abs diff: `0.000977`
- mean abs diff: `0.000000`

### Real packed artifact manifest load benchmark

Command shape:

```bash
TMPDIR=$PWD/.tmp cargo run --release --bin bench_packed_model_artifact -- \
  .artifacts/jengine-packed-model
```

Observed sample:

- manifest load: `0.179 ms`
- entry count: `197`
- packed total bytes: `483755614`
- reduction: `7.111x`

## Key current conclusion

The model, repacker, and runtime integration are correct enough to measure real behavior.

Current highest CPU hotspot remains the MLP path, followed by QKV projections and logits. The current packed CPU kernel is correctness-first and not yet performance-competitive with dense FP16.

For apples-to-apples future comparisons, prefer the release workflow in `docs/RELEASE_BENCHMARK_WORKFLOW.md` and capture reports with `scripts/capture_release_baselines.sh`.

Recent important improvement: memory-mapped weight loading reduced real-model startup overhead enough for the cached hybrid q_proj benchmark path to become observable in this harness.

Recent important caution: the first broader `qkv + gu` hybrid decode benchmark is slightly slower than dense, so the next optimization pass should focus on reducing aggregate multi-projection overhead before adding more decode-path complexity.

Recent important packed-runtime result: paired packed projection dispatches now reduce the combined path from `140` to `84` dispatches on the one-token step and full-span prefill benchmarks, with the full-span prefill sample improving from `10258.111 ms` to `9900.304 ms`.

Recent important artifact result: the real Bonsai packed artifact flow now achieves about **7.11x size reduction** against source safetensors while keeping aggregate validation error very low (`max_abs_diff` about `0.000977`).

Recent important dense-CPU note: a `WeightStore` tensor-index cache is now in place, so future CPU baselines should be compared against the newer post-index-cache runs rather than the oldest early bring-up numbers.
