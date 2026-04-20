# Packed runtime architecture

## Goal

Run Bonsai primarily from packed ternary artifacts, with dense source tensors used only where they are still the best practical choice.

## Current layered design

```text
CLI / benchmark drivers
    ↓
ReferenceModel runtime
    ↓
Packed-first tensor resolution
    ↓
PackedGpuSession
    ↓
CachedGpuPackedMatvecRunner
    ↓
ash / Vulkan
    ↓
Intel Vulkan driver + xe stack
```

## Dataflow

### 1. Model load

`ReferenceModel` can now be created in two modes:

- dense-source mode
- packed-artifact mode

Packed-artifact mode loads:

- source model assets
- config and tokenizer
- packed model manifest
- packed tensor metadata

### 2. Tensor resolution

The runtime resolves tensors in this order:

1. packed artifact store, if present
2. dense source weights fallback

This applies to:

- vector loads
- embedding lookup
- matrix-vector products

### 3. Packed projection execution

For packed projection execution, the flow is:

1. `PackedGpuSession::run_projection(...)`
2. packed cache lookup by tensor name
3. GPU runner lookup by tensor name
4. shader dispatch
5. session-level accounting update

## Current packed choices

### Packed on GPU in decode

Current beneficial packed subsets:

- attention side:
  - `q_proj`
  - `k_proj`
  - `v_proj`
- MLP side:
  - `gate_proj`
  - `up_proj`

### Dense for now

Current dense choices:

- `o_proj`
- `down_proj`
- norm weights
- logits projection

These remain dense because they either:

- measured worse in current offload experiments, or
- still need separate policy decisions

## Ownership model

### PackedModelStore

Owns:

- packed manifest
- packed tensor metadata by name
- cached packed tensor files
- cached unpacked vectors

### ReferenceModel

Owns:

- source assets
- config
- tokenizer
- dense source `WeightStore`
- optional `PackedModelStore`
- CPU-side packed caches
- GPU-side runner caches

### PackedGpuSession

Owns per-session execution accounting:

- pack duration
- compile duration
- upload duration
- gpu duration
- download duration
- pack cache hits
- gpu cache hits
- dispatch count
- upload bytes
- download bytes

The session does **not** own the long-lived packed caches themselves. Those remain on `ReferenceModel` and are reused across sessions.

## Why this structure

This split keeps responsibilities clear:

- `PackedModelStore` answers: where do packed tensors come from?
- `ReferenceModel` answers: how does decode work overall?
- `PackedGpuSession` answers: how do we execute and measure packed GPU projections during one run?
- `CachedGpuPackedMatvecRunner` answers: how is one projection dispatched on Vulkan?

## Near-term next step

The next performance step is to use this architecture to collect stable real benchmarks for:

- packed attention decode
- packed MLP decode
- packed combined decode

Then use those numbers to decide:

- whether logits should stay dense
- whether `o_proj` and `down_proj` are worth revisiting
- where startup and bandwidth optimizations matter most
