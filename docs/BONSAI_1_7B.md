# Bonsai 1.7B notes

Source model:

- `prism-ml/Ternary-Bonsai-1.7B-unpacked`

## Architecture facts

- base architecture: `Qwen3ForCausalLM`
- vocab size: `151669`
- max position embeddings: `32768`
- hidden size: `2048`
- intermediate size: `6144`
- layers: `28`
- attention heads: `16`
- key/value heads: `8`
- head dim: `128`
- activation: `silu`
- norm epsilon: `1e-6`
- rope theta: `1_000_000`
- rope scaling: YaRN, factor `4.0`, original max positions `8192`
- tied embeddings: true
- attention bias: false

## Runtime implications

- grouped-query attention is required
- YaRN rope scaling is required
- tied embeddings should reduce duplicated logic
- KV cache design must handle 8 KV heads and 16 attention heads

## Today's implementation target

Use the unpacked FP16 model as the correctness oracle.

That means:

- correctness first
- dense CPU execution first
- ternary repacking only after the reference path works
