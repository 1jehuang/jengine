# Packed short-context performance targets

## Scope

These are the near-term practical targets for packed decode on this machine, using the current Jengine packed runtime path.

## Current situation

The packed runtime architecture is now in place, and the benchmark harnesses can now capture `mlp` and `combined` short-context runs reliably after adding staged progress output.

A paired-dispatch optimization also reduced the combined packed decode path from `140` GPU dispatches per one-token step or prefill span down to `84`.

The `attention`-only short-context run is still slow enough that this harness may terminate it before the final summary prints, so its tok/s number is not yet stable here.

## Latest measured short-context samples

Command shape:

```bash
./target/release/bench_packed_toks \
  /home/jeremy/models/bonsai-1.7b .artifacts/jengine-packed-model hello 1 1 <variant>
```

Observed samples:

- `mlp`
  - total: `14441.214 ms`
  - throughput: `0.069 tok/s`
  - dispatches: `56`
- `combined`
  - total: `11873.378 ms`
  - throughput: `0.084 tok/s`
  - dispatches: `168`
- `attention`
  - run still terminates in this harness before the final summary is emitted
  - staged progress confirms the decode work is running, but the result is not yet stable enough to record as a baseline here

So the immediate target should remain modest and concrete.

## Near-term target

### Functional target

- the packed short-context benchmark harness must run successfully for:
  - `mlp`
  - `combined`
- the `attention`-only variant still needs a more stable capture path in this harness

### Performance target

For short-context local packed decode on this hardware, target:

- **at least `1.0 tok/s`** for the `combined` packed path on a short prompt once stable benchmark harvesting is working

This is a deliberately conservative near-term target. It is not the final ceiling target.

## Why this target

- the current measured `combined` short-context run is only about `0.084 tok/s`, so there is still a large gap to close
- a conservative `1.0 tok/s` target still represents a clear and meaningful packed-runtime win on this hardware
- the paired-dispatch reduction already removed a chunk of obvious overhead, so the next improvements should come from further orchestration and batching work rather than just adding more projections

## Next target after first clean win

After the first reproducible short-context win at or above `1.0 tok/s`, update this document with:

- a stable measured `attention` tok/s capture
- refreshed `mlp` tok/s
- refreshed `combined` tok/s
- the next target, likely in the multi-token-per-second range above `1.0 tok/s`
