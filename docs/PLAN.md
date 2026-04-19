# Jengine execution plan

## Objective

Get `prism-ml/Ternary-Bonsai-1.7B-unpacked` working today in Jengine.

"Working" for today means:

- load model config and tokenizer assets
- load FP16 safetensors weights
- run a correct CPU reference forward path
- generate at least one token from a prompt
- expose timing and memory metrics
- have reproducible checks, benchmarks, and profiling hooks

## Non-goals for today

- custom ternary packed runtime end-to-end
- Vulkan acceleration end-to-end
- maximum throughput
- long-context optimization

## Reality-based approach

The fastest route to a real working model today is:

1. implement a dense CPU reference path over the unpacked FP16 release
2. validate correctness and observability
3. use that as the oracle for the later ternary repacker and Vulkan backend

## Architecture layers

1. **App / CLI**
   - commands: doctor, inspect, run, bench, pack, profile
2. **Inference runtime**
   - model loading, scheduler, decode loop, sampler, KV cache
3. **CPU backend**
   - correctness-first tensor ops and transformer execution
4. **GPU backend**
   - custom execution backend for accelerated kernels
5. **Vulkan**
   - low-level compute API used by the GPU backend

## Today's milestone plan

### Phase 0: development gates and instrumentation
- add docs, check scripts, benchmark harness, profiling entry points
- success criteria:
  - `scripts/check.sh` passes
  - `scripts/doctor.sh` reports environment status
  - `cargo bench --no-run` succeeds

### Phase 1: asset and config ingestion
- parse config.json, generation config, tokenizer config
- inspect tensor inventory from safetensors
- success criteria:
  - `jengine inspect` or equivalent test prints model facts
  - memory estimate for weights and KV cache is computed

### Phase 2: tokenizer and prompt path
- load tokenizer assets
- encode/decode test strings
- apply chat template if needed
- success criteria:
  - roundtrip encode/decode tests pass
  - prompt preparation is deterministic

### Phase 3: CPU reference backend
- embeddings
- RMSNorm
- RoPE with YaRN scaling
- GQA attention with KV cache
- SwiGLU MLP
- LM head
- greedy sampler first, then temperature/top-k/top-p
- success criteria:
  - per-stage unit tests pass
  - end-to-end forward for a tiny prompt produces logits
  - one-token generation works

### Phase 4: observability and correctness hardening
- per-layer timings
- memory accounting
- decode loop metrics
- structured tracing for prefill/decode steps
- success criteria:
  - benchmarkable subcommands exist
  - profiling scripts can measure end-to-end runs
  - live logs identify slowest stages

### Phase 5: ternary verification and repacker
- detect ternary-g128 structure in unpacked weights
- implement repack format
- verify lossless reconstruction to source values
- success criteria:
  - reconstruction error is zero or within proven representation tolerance
  - packed size estimate matches expectation

### Phase 6: Vulkan backend
- start with isolated microkernels
- ternary linear kernel first
- reference against CPU output on small fixtures
- success criteria:
  - microbenchmarks compare CPU vs GPU kernels
  - correctness tests pass on representative tensors

## Benchmarks required throughout

Every meaningful module must ship with at least one of:

- a microbenchmark
- an end-to-end benchmark
- a memory/timing inspector

Minimum benchmark inventory:

- config parsing
- safetensors scan
- tokenizer encode/decode
- RMSNorm
- RoPE
- attention
- MLP block
- one-layer decode
- full-model one-token decode

## Static analysis and verification policy

For every good chunk of code:

1. `cargo fmt --check`
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
3. `cargo test --workspace --all-targets`
4. `cargo bench --workspace --no-run`
5. targeted runtime benchmark or smoke run

## Commit policy

Commit after each coherent milestone, especially when one of these becomes true:

- new subsystem compiles and is tested
- new benchmark lands
- new profiler or live analysis hook lands
- a milestone document or interface definition becomes stable
