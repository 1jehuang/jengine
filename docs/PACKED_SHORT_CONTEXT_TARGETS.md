# Packed short-context performance targets

## Scope

These are the near-term practical targets for packed decode on this machine, using the current Jengine packed runtime path.

## Current situation

The packed runtime architecture is now in place, but stable end-to-end benchmark harvesting is still difficult in this harness for the longer packed decode runs.

So the immediate target should be modest and concrete.

## Near-term target

### Functional target

- the packed short-context benchmark harness must run successfully for:
  - `attention`
  - `mlp`
  - `combined`

### Performance target

For short-context local packed decode on this hardware, target:

- **at least `1.0 tok/s`** for the `combined` packed path on a short prompt once stable benchmark harvesting is working

This is a deliberately conservative near-term target. It is not the final ceiling target.

## Why this target

- it should clearly beat the current worst-case prototype behavior
- it is still below the long-term practical target range discussed earlier
- it is realistic enough to use as a first milestone for the packed runtime phase

## Next target after first clean win

Once the first clean packed short-context run is captured and reproducible, update this document with:

- measured `attention` tok/s
- measured `mlp` tok/s
- measured `combined` tok/s
- the next target, likely in the multi-token-per-second range above `1.0 tok/s`
