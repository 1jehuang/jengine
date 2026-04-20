# Persistent GPU-first packed decode plan

## Goal

Define the redesign needed to move Bonsai 1.7B packed decode much closer to the hardware limit on Intel Lunar Lake, with `200 tok/s` treated as a north-star goal rather than a forecast.

## Current measured situation

### Prompt-inclusive direct packed benchmark

Strong current configuration:

- `JENGINE_PREWARM_PACKED=1`
- `JENGINE_PACKED_ATTENTION_FULL=1`
- `JENGINE_PACKED_MLP_FULL=1`

Observed direct benchmark:

- iteration 1: `352.139 ms`, `2.840 tok/s`
- iteration 2: `278.126 ms`, `3.595 tok/s`
- average: `315.133 ms`, `3.218 tok/s`

### Explicit one-token packed step

Same strong configuration on an explicit one-token step:

- total: `184.738 ms`
- throughput: about `5.41 tok/s`
- `gpu_ms=129.879`
- `download_ms=36.109`
- `non_offloaded_dense_ms=17.041`
- `orchestration_ms=1.658`
- `dispatch_count=113`

## Current architecture

Today the runtime is still structurally a CPU-driven reference decode loop with packed GPU subroutines inserted into it.

### Current flow

```text
CPU decode loop
  -> resolve layer tensors
  -> launch packed GPU projections for selected ops
  -> download outputs into CPU-visible memory
  -> run dense CPU glue and residual work
  -> repeat for next layer
  -> produce logits and select token
```

### What is already good

- packed GPU kernels are fast enough to matter
- compiled runners and weight uploads are reused much better than before
- prewarm exists and is useful
- profiling and machine-readable tracing now exist
- packed-first generation entrypoints now exist

### What is still structurally bad

- too many dispatches per token
- too much GPU-to-CPU output movement
- hidden and residual state still bounce through CPU-visible paths
- `o_proj` and `down_proj` are still integrated as isolated projection decisions rather than part of a GPU-first block flow
- decode is still controlled by a reference-style CPU loop rather than by a persistent GPU decode engine

## Why the current architecture cannot approach the physical limit

The hardware-limit thought experiment assumes:

- packed weights stay resident
- hidden state stays on GPU
- almost no per-token setup cost
- almost no downloads except the final chosen token
- very few GPU submissions
- minimal CPU work per token

The current runtime does not satisfy those assumptions.

Even after recent wins, the explicit one-token step still has:

- `113` dispatches
- `36 ms` download time
- about `130 ms` GPU time

That is already enough to show that the current architecture shape is many times away from a `5 ms/token` goal.

## Target architecture

The redesign target is a persistent GPU-first decode engine.

### Target flow

```text
Persistent GPU decode session
  -> resident packed weights, descriptors, pipelines, command buffers
  -> resident hidden state, residual state, scratch buffers, KV/cache
  -> GPU-first per-layer/block decode execution
  -> GPU-native logits and token selection
  -> CPU receives only token id and minimal metadata
```

### Core design principles

1. Keep state on GPU
2. Minimize host-visible downloads
3. Minimize dispatch count
4. Replace per-op CPU control flow with persistent decode-session execution
5. Integrate `o_proj` and `down_proj` into GPU-first layer/block execution
6. Return only final token selection to the CPU

## Token-time budget for the north-star goal

If the north-star goal is `200 tok/s`, the per-token budget is:

- total token time: `<= 5.0 ms`

A reasonable stretch budget split would be:

- GPU compute: `<= 3.0 ms`
- all transfers: `<= 1.0 ms`
- CPU orchestration and synchronization: `<= 0.5 ms`
- residual slack: `<= 0.5 ms`

The current explicit one-token packed step is far from this:

- total: `184.738 ms`
- GPU time: `129.879 ms`
- download: `36.109 ms`

So the required change is architectural, not incremental.

## Redesign milestones

### Milestone 1: persistent decode session

Build a long-lived decode session that keeps resident:

- packed weights
- compiled pipelines
- descriptor sets
- command buffers
- scratch buffers
- hidden state buffers
- KV/cache buffers

The CPU should submit a tiny per-token request instead of rebuilding decode state each step.

### Milestone 1a: shared Vulkan context prerequisite

A direct attempt to make packed runners share a single Vulkan context exposed an important implementation blocker:

- the current packed runner implementation was originally built around one isolated Vulkan instance/device/queue per runner
- resident outputs can stay on-GPU inside one runner, but cannot yet be safely chained across runners
- a first all-at-once shared-context prototype currently triggers Vulkan-side instability in synthetic packed decode tests

So the safe next step is to stage this refactor more carefully:

1. extract an explicit reusable packed Vulkan context type
2. make runner construction optionally accept that context without changing decode behavior
3. validate multiple runners on one shared device with synthetic micro-tests
4. only then wire the shared context into decode-session caches

This shared-context refactor is a prerequisite for true GPU-resident hidden-state flow across projections.

### Milestone 2: GPU-resident hidden and residual flow

Keep the following on GPU across the one-token path:

- embedding output
- hidden state
- residual state
- norm outputs
- attention outputs
- MLP outputs

Only the final token id should need to return to the CPU.

### Milestone 3: GPU-native logits and token selection

Move logits projection and token selection fully onto the GPU.

Return only:

- token id
- optional score metadata

This removes the need for large logits downloads and reduces CPU-side decode glue.

### Milestone 4: dispatch collapse

Reduce dispatch count from the current `113` per token toward:

- low teens at minimum
- single digits as the real target

This requires much larger fused submissions, ideally at a layer or block level rather than per-projection.

### Milestone 5: GPU-first `o_proj` and `down_proj` integration

Do not treat these as isolated packed projection toggles.

Instead:

- integrate `o_proj` into the attention block flow
- integrate `down_proj` into the MLP tail flow
- keep their inputs and outputs on-device
- avoid CPU-visible glue between adjacent steps

### Milestone 6: GPU-first one-token prototype

Build an explicit one-token decode path that bypasses the current CPU reference-style loop as much as possible.

Measure it directly against the current explicit one-token packed step.

This is the first milestone that should tell us how much architectural headroom is really available.

## Ordered implementation priority

1. Persistent GPU decode session
2. GPU-resident hidden/residual state
3. GPU-native logits/token selection
4. Dispatch collapse
5. `o_proj` / `down_proj` integration redesign
6. GPU-first one-token prototype
7. Benchmark checkpoints and reassessment

## Benchmark checkpoints

After each milestone, re-run:

- explicit one-token packed step
- prompt-inclusive direct packed benchmark
- attribution and trace capture

Track checkpoints at:

- `10 tok/s`
- `25 tok/s`
- `50 tok/s`
- `100 tok/s`
- `200 tok/s`

## Reassessment rule

After the GPU-first prototype and at least the first major dispatch-collapse pass land, reassess:

1. the new practical architecture ceiling on this Intel Lunar Lake iGPU
2. how much of the remaining gap is due to avoidable architecture overhead
3. how much of the remaining gap is due to more fundamental model or hardware limits

## Current conclusion

The current system has crossed the point where raw packed kernel performance is no longer the main unknown.

The next meaningful question is no longer:

> can packed ternary GPU kernels be fast enough?

The next meaningful question is:

> how much of decode can be turned into a persistent GPU-first execution model with almost no CPU-visible intermediate traffic?

That is the only path that has a credible chance of moving substantially closer to the physical limit.
