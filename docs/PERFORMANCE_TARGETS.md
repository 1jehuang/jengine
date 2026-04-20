# Jengine performance targets

This document defines the current measured baseline and the next concrete performance goals for Bonsai 1.7B on this machine.

## Measurement basis

Use the pinned release workflow from:

- `docs/RELEASE_BENCHMARK_WORKFLOW.md`
- `fixtures/release_benchmark.env`
- `scripts/capture_release_baselines.sh`

For hybrid decode experiments that may run longer in this harness, use the staged benchmark binaries with progress output and artifact capture.

## Current measured reference points

### Dense reference

Real one-token cached-benchmark comparison run:

- prompt: `hello`
- generated tokens: `1`
- dense total: `9822.266 ms`
- implied generated tok/s: about `0.102`

### Cached q_proj warm hybrid

Same run:

- total: `5475.410 ms`
- implied generated tok/s: about `0.183`
- pack cache hit: `true`
- gpu cache hit: `true`

### Improvement from current dense to cached q_proj warm

- latency ratio: about `1.79x` better
- tok/s ratio: about `1.79x` better

## Near-term targets

### Target A: attention-only expansion milestone

After extending cached packed offload to `q_proj`, `k_proj`, `v_proj`, and `o_proj`:

- target one-token total: **<= 4.0 s**
- target generated throughput: **>= 0.25 tok/s**

Why this target:
- it is materially better than the current cached-q-only result
- it should be reachable without touching MLP yet
- it gives a concrete success bar for the next GPU milestone

### Target B: attention + MLP projection milestone

After extending packed GPU offload into MLP projections:

- target one-token total: **<= 2.0 s**
- target generated throughput: **>= 0.50 tok/s**

Why this target:
- current hotspot ranking is still MLP-heavy
- meaningful tok/s improvement probably requires MLP participation

### Stretch target

For a substantially accelerated Bonsai 1.7B local path on this hardware:

- target one-token total: **<= 1.0 s**
- target generated throughput: **>= 1.0 tok/s**

This is a stretch target, not the immediate next milestone.

## Tracking rules

After every major optimization pass, record:

- command used
- git commit
- dense total ms
- hybrid total ms
- pack ms
- gpu compile ms
- gpu upload ms
- gpu compute ms
- gpu download ms
- generated tok/s
- whether caches were warm

## Current bottleneck view

Based on the latest real runs:

1. CPU-side model work outside the single q_proj offload still dominates total latency
2. MLP remains a major remaining target
3. GPU setup costs are now largely removed for warm q_proj runs
4. upload + actual execution are the next visible GPU-side costs

## Current conclusion

The project has crossed the point where caching matters in real measured decode time.

The next meaningful progress bar is no longer “make caching work”.
It is:

> **reach at least 0.25 tok/s with cached attention-side projection offload beyond q_proj.**
