#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict


def load_profile(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def fmt(ms: float) -> str:
    return f"{ms:.3f}"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: summarize_packed_profile_hotspots.py <profile.json> [top_n]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
    profile = load_profile(path)
    metrics = profile.get("packed_metrics", {})
    traces = profile.get("dispatch_trace", [])
    if not traces:
        print(f"no dispatch_trace found in {path}", file=sys.stderr)
        return 1

    by_stage_path = defaultdict(lambda: defaultdict(float))
    by_stage = defaultdict(lambda: defaultdict(float))
    component_totals = defaultdict(float)

    for t in traces:
        stage = t.get("stage", "<unknown>")
        p = t.get("path", "<unknown>")
        cpu = float(t.get("cpu_ms", 0.0))
        gpu = float(t.get("gpu_ms", 0.0))
        download = float(t.get("download_ms", 0.0))
        compile = float(t.get("compile_ms", 0.0))
        total = cpu + gpu + download + compile
        by_stage_path[(stage, p)]["total_ms"] += total
        by_stage_path[(stage, p)]["cpu_ms"] += cpu
        by_stage_path[(stage, p)]["gpu_ms"] += gpu
        by_stage_path[(stage, p)]["download_ms"] += download
        by_stage_path[(stage, p)]["compile_ms"] += compile
        by_stage[stage]["total_ms"] += total
        by_stage[stage]["cpu_ms"] += cpu
        by_stage[stage]["gpu_ms"] += gpu
        by_stage[stage]["download_ms"] += download
        by_stage[stage]["compile_ms"] += compile
        component_totals["cpu_ms"] += cpu
        component_totals["gpu_ms"] += gpu
        component_totals["download_ms"] += download
        component_totals["compile_ms"] += compile

    print(f"profile={os.path.abspath(path)}")
    if metrics:
        print(
            "metrics "
            + " ".join(
                f"{k}={fmt(float(v))}"
                for k, v in [
                    ("total_ms", metrics.get("total_ms", 0.0)),
                    ("gpu_ms", metrics.get("gpu_ms", 0.0)),
                    ("download_ms", metrics.get("download_ms", 0.0)),
                    ("compile_ms", metrics.get("compile_ms", 0.0)),
                    ("non_offloaded_dense_ms", metrics.get("non_offloaded_dense_ms", 0.0)),
                ]
            )
            + f" dispatch_count={metrics.get('dispatch_count', 0)}"
        )

    print("\ncomponent_totals")
    for key in ("cpu_ms", "gpu_ms", "download_ms", "compile_ms"):
        print(f"  {key:12} {fmt(component_totals[key])}")

    print(f"\ntop_stage_path_hotspots top_n={top_n}")
    rows = sorted(by_stage_path.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)[:top_n]
    for (stage, p), vals in rows:
        print(
            f"  {stage:28} {p:18} total={fmt(vals['total_ms'])} cpu={fmt(vals['cpu_ms'])} gpu={fmt(vals['gpu_ms'])} download={fmt(vals['download_ms'])} compile={fmt(vals['compile_ms'])}"
        )

    print(f"\ntop_stage_hotspots top_n={top_n}")
    rows = sorted(by_stage.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)[:top_n]
    for stage, vals in rows:
        print(
            f"  {stage:28} total={fmt(vals['total_ms'])} cpu={fmt(vals['cpu_ms'])} gpu={fmt(vals['gpu_ms'])} download={fmt(vals['download_ms'])} compile={fmt(vals['compile_ms'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
