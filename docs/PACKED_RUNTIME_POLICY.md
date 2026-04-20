# Packed runtime policy decisions

## LM head handling

### Observed model fact

For Bonsai 1.7B:

- `tie_word_embeddings = true`

So the logits projection is tied to:

- `model.embed_tokens.weight`

There is no separate untied LM-head tensor to optimize independently.

### Current runtime choice

Keep the logits path **dense for now**.

### Why

- the tied embedding matrix is large, but it also serves embedding lookup
- the current packed decode work is focused on the dominant transformer projection weights first
- logits projection happens once per decode step and still needs a clean policy for shared representation with embeddings
- this path should not be packed until we have stable end-to-end packed decode benchmarks for the main transformer body

### Current recommendation

Short term:

- keep tied embeddings dense for both embedding lookup and logits

Later, revisit only after:

- packed attention and packed MLP decode are benchmarked end to end
- packed decode correctness is stable
- we have a clear picture of whether logits is a top remaining bottleneck

## Activation and norm precision

### Current runtime behavior

The runtime currently uses `f32` for:

- activations
- RMSNorm computation
- rotary-transformed query and key activations
- attention softmax and score accumulation
- KV cache working representation

Dense source weights are expanded into `f32` for CPU-side math.

### Current runtime choice

Keep norms and activation math in **`f32` for now**.

### Why

- this is still the reference and validation-heavy phase of the project
- keeping norms and activations in `f32` reduces numerical ambiguity while packed execution is being brought up
- correctness comparisons against dense reference are easier to interpret in the current form
- the packed decode path still has larger system bottlenecks than activation precision alone

### Current recommendation

Short term:

- keep norm and activation math in `f32`
- keep packed execution focused on large projection weights

Later investigate whether some of these can safely move to a cheaper precision for:

- reduced bandwidth
- smaller KV footprint
- faster CPU fallback behavior

but only after the end-to-end packed decode benchmark path is stable.

## Summary

Current packed runtime policy is:

- packed on GPU:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `gate_proj`
  - `up_proj`
- dense for now:
  - tied embedding / logits matrix
  - `o_proj`
  - `down_proj`
  - norm weights
  - activation and norm math in `f32`
