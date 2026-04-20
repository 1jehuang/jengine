# Packed short-context performance targets

## Scope

These are the near-term practical targets for packed decode on this machine, using the current Jengine packed runtime path.

## Current situation

The packed runtime architecture is now in place, and the benchmark harnesses can now capture `mlp` and `combined` short-context runs reliably after adding staged progress output.

A paired-dispatch optimization also reduced the combined packed decode path from `140` GPU dispatches per one-token step or prefill span down to `84`, and later work brought the current warm direct path down to much more usable territory.

The old `1 tok/s` near-term target has now been cleared on the direct warm path, so the next benchmark target should move into stable multi-token-per-second territory.

## Latest measured short-context samples

Command shape:

```bash
./target/release/bench_packed_toks \
  /home/jeremy/models/bonsai-1.7b .artifacts/jengine-packed-model hello 1 2 combined
```

Observed samples:

- earlier `combined` sample:
  - total: `11873.378 ms`
  - throughput: `0.084 tok/s`
- warm full-MLP sample:
  - iteration 2: `1969.099 ms`, `0.508 tok/s`
- prewarmed full-attention + full-MLP sample:
  - iteration 1: `622.446 ms`, `1.607 tok/s`
  - iteration 2: `289.158 ms`, `3.458 tok/s`
  - average: `455.802 ms`, `2.532 tok/s`

So the benchmark is no longer merely crawling. The direct warm path is now in genuine multi-token-per-second territory when the warm-path structure is preserved.

## Near-term target

### Functional target

- the packed short-context benchmark harness must run successfully for:
  - `combined`
- historical `attention`-only / `mlp`-only captures are now less interesting than the stronger direct warm combined path, so the next practical focus should stay on the direct combined benchmark and its warm-path structure

### Performance target

For short-context local packed decode on this hardware, target next:

- **at least `4.0 tok/s`** on the direct warm `combined` packed path
- and improve the first measured prewarmed iteration further above the current `1.607 tok/s`

This is the new near-term target after clearing the older `1.0 tok/s` milestone.

## Why this target

- the packed direct path now already clears the old `1.0 tok/s` milestone and, with broader decode prewarm, both measured iterations stay above `2.2 tok/s`
- the next useful benchmark question is therefore not whether packed decode can break `1 tok/s`, but whether it can sustain and raise multi-token-per-second behavior on the direct path
- the main remaining gap is now in pushing the stable prewarmed direct path above the current `~2.4 tok/s` band rather than proving viability at all

## Practical current-architecture ceiling

Using the strongest current direct warm sample:

- `JENGINE_PREWARM_PACKED=1`
- `JENGINE_PACKED_ATTENTION_FULL=1`
- `JENGINE_PACKED_MLP_FULL=1`
- warm second pass total: `289.158 ms`
- warm second pass throughput: `3.458 tok/s`

We can estimate two useful upper bounds for the **current architecture style**:

1. **GPU + transfer floor**
   - using `gpu_ms + download_ms + upload_ms`
   - `173.172 + 51.888 + 0.070 = 225.130 ms`
   - about **`4.44 tok/s`**

2. **GPU-only floor**
   - using only `gpu_ms`
   - `173.172 ms`
   - about **`5.77 tok/s`**

Interpretation:

- on the current architecture, even if we remove the remaining direct dense CPU work perfectly, we are still in the **single-digit tok/s** regime
- that is dramatically better than where the project started, but it is also dramatically below the earlier speculative `200 tok/s` idea
- so the real remaining work is not small polishing. Reaching anything much larger than this current single-digit band would require another substantial architectural shift beyond the present packed decode design

After the first reproducible direct warm combined result at or above `4.0 tok/s`, update this document with:

- refreshed direct first-iteration prewarmed tok/s
- refreshed direct warm second-pass tok/s
- a new practical target beyond `4 tok/s`
- and, once stable enough, an updated practical ceiling estimate for this Intel Arc iGPU

