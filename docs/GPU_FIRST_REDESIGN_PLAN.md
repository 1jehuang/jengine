# GPU-First Decode Redesign Plan

## Trusted performance anchor

Use commit `8d387d5` as the trusted throughput anchor for comparisons.

Winner config:

- `JENGINE_PREWARM_PACKED=1`
- `JENGINE_PACKED_ATTENTION_FULL=1`
- `JENGINE_PACKED_MLP_FULL=1`
- `JENGINE_GPU_EMBEDDING=1`
- `JENGINE_GPU_FULL_LAST_LAYER=1`

Trusted measurements:

- 32-token, 5-run avg: **14.355 tok/s**
- 64-token, 3-run avg: **12.007 tok/s**

## Why we are shifting away from seam-level experiments

Recent local experiments that stayed test-clean but still regressed the trusted winner path:

- grouped fused-argmax submit
- pure group-wise fused-logits traversal
- regular group-wise packed-matvec shader heuristic
- GPU final-norm tail bridge
- narrow GPU `o_proj` + vector-add bridge
- later-layer raw-f32 host-hidden QKV-to-host
- last-layer-only raw-f32 QKV-to-host
- suffix-only expansion of `GPU_FULL_LAST_LAYER`
- raw-f32 XE2 shader swap
- hidden-sized output scratch reuse
- broad `GPU_MLP_ENTRY`
- broad `GPU_SWIGLU_BLOCK`

The consistent pattern is that isolated or seam-level wins do not transfer to the decode-wide winner path.

## Active redesign milestones

### 1. Control-plane extraction
Already started.

Goals:
- move env-policy and request/session selection out of `reference.rs`
- make decode plan, engine, and request first-class runtime concepts

Current landed slices:
- `PackedDecodePlan`
- `GpuDecodeEngine`
- `PackedDecodeRequest`

### 2. Resident/state extraction
Already started.

Goals:
- move decode-local state out of `reference.rs`
- make resident-state plumbing explicit

Current landed slices:
- `GpuKvBinding`
- `ResidentHiddenState`
- decode metrics state module
- decode scratch state module
- projection state module
- session state module
- model state module
- output/report state module

### 3. Tail/token-return unification
Already started.

Goals:
- one canonical GPU tail return contract
- reduce last-layer special-case sprawl before broader block expansion

Current landed slice:
- unified GPU tail return interface for token/logits paths

### 4. Persistent GPU-first decode redesign
Not yet implemented.

Planned next decomposition:
1. move more decode setup/control through `GpuDecodeEngine`
2. make resident decode state the steady-state model, not fallback tuples and host-visible vectors
3. introduce a dedicated GPU-first token driver path
4. only then revisit broader block/suffix execution changes

## Immediate coding priorities

1. Continue `t57` state extraction until the remaining decode-local helper/control state is no longer defined in `reference.rs`
2. Continue `t56` so `GpuDecodeEngine` owns more than session creation and prewarm
3. Avoid new seam-level performance toggles unless they are exceptionally compelling and decode-wide, not microbench-only

## Decision rule

If the next few local winner-path experiments still lose, do not add more bridge toggles.
Instead, proceed directly with the broader persistent GPU-first decode redesign.
