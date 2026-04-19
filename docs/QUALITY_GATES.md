# Quality gates

## Rule

Every meaningful code chunk must be:

- statically analyzed
- unit tested or smoke tested
- benchmarkable
- profileable
- observable at runtime

## Required checks

Run these after each coherent change set:

```bash
scripts/check.sh
```

This currently runs:

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo bench --workspace --no-run`

## Benchmark requirements

Each subsystem must expose at least one measurable surface:

- parser: parse benchmark
- loader: file scan benchmark
- kernel: microbenchmark
- runtime path: end-to-end prompt benchmark

## Profiling requirements

At minimum each runtime path must support:

- wall-clock timing
- stage timings
- memory estimate output
- ad hoc trace logging

Preferred tools:

- `scripts/profile_cpu.sh`
- `/usr/bin/time -v`
- `perf stat` when available
- structured logging via `tracing` once runtime code lands

## Live analysis expectations

When a feature lands, it should be possible to answer:

- where time is spent
- where memory is spent
- what input sizes trigger regressions
- whether correctness changed compared to the baseline

## Commit policy

Do not let large unverified changes accumulate. Commit small, verified milestones.
