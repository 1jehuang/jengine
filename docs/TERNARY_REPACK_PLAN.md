# Ternary repack plan

## Goal

Convert the unpacked FP16 Bonsai weights into a runtime-native ternary-g128 format suitable for CPU and Vulkan execution.

## Observed source format facts

From the unpacked Bonsai 1.7B release:

- source tensor dtype: FP16
- total tensor count: 310
- architecture shape matches Qwen3-1.7B geometry
- unpacked weight file size is roughly 3.44 GB at FP16

## Target packed format

For tensors that are ternary-eligible:

- values constrained to `{-1, 0, +1}`
- storage: 2 bits per value in packed groups
- one FP16 scale per group of 128 weights
- layout optimized for matvec / matmul traversal

### Proposed record layout

Per tensor:

- tensor name
- shape
- tensor class
- packability flag
- group size, initially `128`
- packed ternary bytes
- FP16 group scales
- optional pretransposed / tiled variants for backend-specific kernels

## Expected storage cost

### Effective packed cost

For group size 128:

- ternary codes: `2 bits * 128 = 256 bits = 32 bytes`
- scale: `1 FP16 = 2 bytes`
- total per 128 weights: `34 bytes`
- effective bits/weight: `34 * 8 / 128 = 2.125 bits/weight`

### Relative to FP16

- FP16 uses `16 bits/weight`
- packed ternary-g128 uses `2.125 bits/weight`
- reduction ratio: about `7.53x`

This aligns with the general Bonsai claim that the packed release is much smaller than the unpacked FP16 export.

## Eligibility assumptions

Initially assume packable:

- embeddings
- q/k/v/o projection weights
- gate/up/down MLP weights
- lm head if untied or materialized

Keep uncompressed for now:

- RMSNorm weights
- metadata
- tokenizer assets

## Correctness checks required

For each packed tensor:

1. verify that every source FP16 value is representable as `scale * ternary_value` under the intended grouping
2. reconstruct the tensor from packed form
3. compare reconstructed values to source values
4. record max absolute error and non-representable groups if any

## Benchmark plan

- pack throughput in MB/s
- unpack throughput in MB/s
- packed size vs unpacked size
- CPU packed matvec vs FP16 matvec
- Vulkan packed matvec vs CPU packed matvec

## Immediate next implementation steps

1. add a local size-analysis command using the actual safetensors header
2. classify tensors by packability
3. implement one tensor pack/unpack prototype
4. benchmark the prototype before expanding to full-model conversion
