# Tokenizer fallback and model file expectations

Jengine prefers a standard Hugging Face `tokenizer.json`, but it also supports a fallback path for the current Bonsai 1.7B release.

## Primary tokenizer path

If `tokenizer.json` loads successfully through the Rust `tokenizers` crate, Jengine uses it directly.

Expected file:

- `tokenizer.json`

## Fallback tokenizer path

If `tokenizer.json` is present but not accepted by the current `tokenizers` crate version, Jengine falls back to reconstructing a Qwen-style tokenizer from the sidecar assets.

Expected fallback files:

- `vocab.json`
- `merges.txt`
- `tokenizer_config.json`

The fallback loader reconstructs:

- BPE vocabulary and merges
- NFC normalization
- Qwen2 regex splitting
- byte-level pretokenization
- byte-level decoding
- added special tokens from `tokenizer_config.json`

## Expected model root layout

Jengine currently expects a model root that contains at least:

- `config.json`
- `model.safetensors`
- tokenizer assets, either:
  - `tokenizer.json`, or
  - `vocab.json`, `merges.txt`, and `tokenizer_config.json`

Optional but useful:

- `generation_config.json`

## Supported workflows today

### Inspect

```bash
cargo run --release -- inspect /path/to/model hello 1
```

This reports:

- weight availability
- tokenizer diagnostics
- prompt analysis
- memory ownership estimates
- Vulkan availability

### Run

```bash
cargo run --release -- run /path/to/model hello 1
```

This performs a real text-prompt decode using the tokenizer path that successfully loaded.

### Validate

```bash
cargo run --release -- validate /path/to/model hello 1
```

This checks:

- assets
- weights
- tokenizer
- a smoke decode

## Current limitations

- the fallback path is tuned for the current Qwen-style Bonsai release
- the tokenizer fallback is focused on correctness and usability, not tokenizer loading speed
- if both `tokenizer.json` and fallback sidecar assets are missing, text-prompt workflows will fail
- token-id workflows can still be used even if text tokenization is unavailable
