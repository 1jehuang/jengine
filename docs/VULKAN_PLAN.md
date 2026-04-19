# Vulkan backend plan

## Why Vulkan next

The measured CPU hotspots already point to where acceleration will matter most.

## Current measured hotspots

From existing Jengine microbenchmarks and live profiles:

- CPU `matvec_512x1024`: roughly `287 to 402 us`
- CPU block attention for seq32: roughly `600 to 616 us`
- CPU block profile for seq16: roughly `346 to 361 us`
- live block profile at seq16:
  - rope: about `0.041 ms`
  - qk norm: about `0.033 ms`
  - attention: about `0.344 ms`
  - swiglu: about `0.120 ms`

## Implication

The first Vulkan wins should target:

1. packed linear algebra
2. attention score/value mixing
3. decode-path memory traffic

## Backend phases

### Phase 1: microkernels

Implement isolated compute kernels for:

- FP16 dense matvec baseline
- ternary packed matvec
- qk score kernel
- attention value accumulation kernel

Validation:

- compare against CPU outputs on small deterministic fixtures
- benchmark per-kernel throughput and latency

### Phase 2: decode-focused execution

Implement decode-path kernels optimized for batch=1 / small batch:

- q/k/v projection path
- grouped-query attention
- output projection
- MLP projections

Validation:

- one-token decode output must match CPU reference within tolerance
- stage timings must be emitted per kernel

### Phase 3: prefill path

Implement longer-sequence kernels for prompt ingestion.

## API boundary

Jengine owns:

- tensor layout
- scheduler
- packed formats
- kernel dispatch order
- profiling / tracing

Vulkan owns:

- device abstraction
- command buffers
- memory objects
- compute dispatch
- synchronization primitives

## Kernel priority order

1. dense FP16 matvec reference Vulkan kernel
2. ternary packed matvec kernel
3. attention logits kernel
4. attention value accumulation kernel
5. fused MLP path where justified by profile data

## Benchmark plan

Every Vulkan kernel must have:

- CPU reference comparison
- microbenchmark
- end-to-end contribution measurement
- profiler-friendly tracing hooks

## Success criteria for first Vulkan milestone

- one kernel validated against CPU numerics
- one kernel benchmarked on the local Intel Lunar Lake GPU
- no regressions in runtime observability
