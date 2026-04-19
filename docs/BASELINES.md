# Jengine baseline measurements

## Real Bonsai 1.7B CPU runs

### One-token text prompt run

Command shape:

```bash
cargo run --release --bin run_reference -- /home/jeremy/models/bonsai-1.7b hello 1
```

Observed sample:

- `prompt_tokens=1`
- `generated_tokens=1`
- `total_ms=1513.965`
- `embed_ms=0.490`
- `norm_ms=56.923`
- `qkv_ms=248.663`
- `attention_ms=111.360`
- `mlp_ms=849.401`
- `logits_ms=246.633`

### One-token token-id run

Command shape:

```bash
./target/release/run_reference_ids /home/jeremy/models/bonsai-1.7b 42 1
```

Observed sample:

- `prompt_tokens=1`
- `generated_tokens=1`
- `total_ms=2527.042`
- `embed_ms=0.557`
- `norm_ms=53.924`
- `qkv_ms=520.173`
- `attention_ms=201.257`
- `mlp_ms=1440.222`
- `logits_ms=308.975`

### Multi-token text benchmark

Command shape:

```bash
cargo run --release --bin bench_real_text -- /home/jeremy/models/bonsai-1.7b hello 3 2
```

Observed samples:

- iteration 1 total: `3627.780 ms`
- iteration 2 total: `3208.238 ms`
- average total: `3418.009 ms`
- average qkv: `595.950 ms`
- average attention: `255.213 ms`
- average mlp: `1942.555 ms`

## Repack baselines

### Real q_proj tensor pack

Command shape:

```bash
cargo run --release --bin pack_real_tensor -- \
  /home/jeremy/models/bonsai-1.7b/model.safetensors \
  model.layers.0.self_attn.q_proj.weight \
  2048 2048
```

Observed sample:

- elements: `4,194,304`
- groups: `32,768`
- perfect groups: `32,768`
- estimated packed bytes: `1,114,112`
- reduction: `7.529x`
- strict pack time: `68.949 ms`

## Packed matvec baselines

### Real q_proj dense vs packed

Command shape:

```bash
cargo run --release --bin compare_packed_matvec -- \
  /home/jeremy/models/bonsai-1.7b/model.safetensors \
  model.layers.0.self_attn.q_proj.weight \
  2048 2048
```

Observed sample:

- dense: `8.391 ms`
- packed reference: `34.038 ms`
- speedup: `0.247x`
- max abs diff: `0.000000`

### Runtime-integrated q_proj dense vs packed

Command shape:

```bash
cargo run --release --bin compare_reference_projection -- /home/jeremy/models/bonsai-1.7b 0 42
```

Observed sample:

- dense: `2.388 ms`
- pack: `58.412 ms`
- packed projection: `33.896 ms`
- max abs diff: `0.000000`
- mean abs diff: `0.000000`

## Key current conclusion

The model, repacker, and runtime integration are correct enough to measure real behavior.

Current highest CPU hotspot remains the MLP path, followed by QKV projections and logits. The current packed CPU kernel is correctness-first and not yet performance-competitive with dense FP16.
