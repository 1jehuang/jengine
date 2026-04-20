# Automation workflows

Jengine now includes three higher-level automation scripts.

## Quick CI-style check

```bash
./scripts/quick_ci.sh
```

What it does:

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo bench --workspace --no-run`

This is the fastest repo-level workflow intended for CI-style smoke coverage.

## One-command regression workflow

```bash
./scripts/run_regression.sh
```

Optional output directory override:

```bash
./scripts/run_regression.sh /tmp/jengine-regression
```

What it does:

- runs `./scripts/quick_ci.sh`
- builds the release `jengine` binary
- runs `jengine validate`
- runs `jengine bench`
- emits benchmark reports in:
  - markdown
  - key-value
  - CSV
- captures a plain-text `inspect` snapshot

Default output root:

- `.artifacts/regression/<timestamp>/`

## Longer local release profiling workflow

```bash
./scripts/profile_release_local.sh
```

Optional output directory override:

```bash
./scripts/profile_release_local.sh /tmp/jengine-local-profile
```

What it does:

- builds selected release binaries
- captures release `inspect` output
- captures release `profile` output
- runs the real `compare_packed_matvec` release benchmark
- runs `./scripts/capture_release_baselines.sh`
- records command lines and command output in markdown files
- uses `/usr/bin/time -v` when available

Default output root:

- `.artifacts/local-profile/<timestamp>/`

## Fixture source

The regression and local profiling workflows use:

- `fixtures/release_benchmark.env`

That keeps model root, prompt, tensor, and layer choices pinned across runs.

## Temp directory behavior

These scripts default `TMPDIR` to a repo-local `.tmp/` directory so they are less likely to fail on quota-limited system temp paths.
