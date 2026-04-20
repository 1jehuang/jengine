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

Real one-token token-id packed-step comparison run:

- prompt ids: `42`
- generated tokens: `0`
- dense total: `1599.766 ms`

### Packed combined step

Same benchmark family:

- total: `10359.618 ms`
- compile: `72.471 ms`
- upload: `58.044 ms`
- gpu: `282.698 ms`
- download: `18.571 ms`
- dispatch count: `140`

### Packed combined full-span prefill

Same benchmark family:

- total: `10258.111 ms`
- compile: `84.826 ms`
- upload: `57.067 ms`
- gpu: `283.962 ms`
- download: `16.414 ms`
- dispatch count: `140`

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

### Target A: packed decode orchestration reduction milestone

On the current packed-artifact path:

- target one-token packed combined step total: **<= 6.0 s**
- target one-token packed combined full-span prefill total: **<= 6.0 s**
- target dispatch count for the combined path: **materially below `140`**

Why this target:
- the latest benchmarks show raw GPU time is not the primary limiter
- host-side orchestration and dispatch overhead now dominate
- reducing that overhead is the clearest next way to make the packed path competitive

### Target B: packed short-context throughput milestone

After reducing orchestration overhead and re-running the short-context tok/s harness:

- target short-context combined packed throughput: **>= 1.0 tok/s**

Why this target:
- it matches the near-term packed short-context target document
- it is conservative enough to be a first meaningful packed-runtime win on this hardware

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

Based on the latest real packed-artifact runs:

1. Host-side orchestration and per-dispatch overhead dominate total packed step and prefill time
2. MLP remains a major dense-side hotspot, but raw GPU compute for packed projections is not the main limiter yet
3. Compile cost still matters for first-pass all-layer sweeps, though runner reuse reduces it within a run
4. The next meaningful improvement should come from fewer dispatches, stronger batching, and better cross-step reuse

## Current conclusion

The project has crossed the point where the packed runtime can be measured cleanly at layer-sweep, token-id step, and full-span prefill levels.

The next meaningful progress bar is no longer “make the packed benchmark harness run”.
It is:

> **cut host-side overhead and dispatch count enough to bring the packed combined step closer to dense latency, then re-measure short-context packed tok/s.**
