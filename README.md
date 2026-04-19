# Jengine

A Rust inference runtime for compact LLMs.

## Immediate goal

Get the smallest PrismML Bonsai model, `Ternary-Bonsai-1.7B`, running locally today with:

- a correct CPU reference path first
- validation against known model metadata
- measurable benchmarks at every layer
- profiling and live analysis built in from the start

## Strategy

Jengine will be built in layers:

1. application / CLI
2. inference runtime
3. CPU reference backend
4. GPU backend
5. Vulkan interface

We will get correctness first on CPU, then repack and accelerate.

## Docs

- `docs/PLAN.md` - execution plan and milestones
- `docs/QUALITY_GATES.md` - required checks for every code chunk
- `docs/BONSAI_1_7B.md` - concrete model notes and constraints

## Status

Planning and tooling scaffold is in place.
