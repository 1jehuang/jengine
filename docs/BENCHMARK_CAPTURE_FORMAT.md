# Benchmark capture format

When recording benchmark results for Jengine, include the following fields whenever possible.

## Required fields

- date/time
- git commit hash
- command run
- model name / tensor name
- hardware summary
- result metrics

## Recommended markdown template

```markdown
## Benchmark: <name>

- Date: <timestamp>
- Commit: <git sha>
- Machine: <cpu/gpu/ram>
- Command:
  ```bash
  <command>
  ```
- Result:
  - metric_a: <value>
  - metric_b: <value>
  - metric_c: <value>
- Notes:
  - <observations>
```

## For correctness comparisons

Also include:

- max abs diff
- mean abs diff
- tensor shape or prompt size
- tolerance threshold

## For profiling runs

Also include:

- total time
- qkv time
- attention time
- mlp time
- logits time
- memory usage if available
