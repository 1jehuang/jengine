#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict


def load(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def fmt(v: float) -> str:
    return f"{v:.3f}"


def aggregate(profile):
    metrics = profile.get('packed_metrics', {})
    traces = profile.get('dispatch_trace', [])
    by_stage_path = defaultdict(float)
    for t in traces:
        stage = t.get('stage', '<unknown>')
        path = t.get('path', '<unknown>')
        total = float(t.get('cpu_ms', 0.0)) + float(t.get('gpu_ms', 0.0)) + float(t.get('download_ms', 0.0)) + float(t.get('compile_ms', 0.0))
        by_stage_path[(stage, path)] += total
    return metrics, by_stage_path


def main() -> int:
    if len(sys.argv) < 3:
        print('usage: compare_packed_profile_hotspots.py <before.json> <after.json> [top_n]', file=sys.stderr)
        return 2
    before_path, after_path = sys.argv[1], sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
    before = load(before_path)
    after = load(after_path)
    bm, bmap = aggregate(before)
    am, amap = aggregate(after)

    print(f'before={os.path.abspath(before_path)}')
    print(f'after={os.path.abspath(after_path)}')
    print('\nmetric_deltas')
    for key in ('total_ms', 'gpu_ms', 'download_ms', 'compile_ms', 'non_offloaded_dense_ms'):
        b = float(bm.get(key, 0.0))
        a = float(am.get(key, 0.0))
        d = a - b
        print(f'  {key:24} before={fmt(b)} after={fmt(a)} delta={fmt(d)}')
    print(f"  {'dispatch_count':24} before={bm.get('dispatch_count', 0)} after={am.get('dispatch_count', 0)} delta={int(am.get('dispatch_count', 0)) - int(bm.get('dispatch_count', 0))}")

    delta_rows = []
    keys = set(bmap) | set(amap)
    for key in keys:
        b = bmap.get(key, 0.0)
        a = amap.get(key, 0.0)
        delta_rows.append((a - b, key, b, a))

    print(f'\nregressions top_n={top_n}')
    regressions = [row for row in sorted(delta_rows, key=lambda r: r[0], reverse=True) if row[0] > 0][:top_n]
    for delta, (stage, path), b, a in regressions:
        print(f'  {stage:28} {path:18} before={fmt(b)} after={fmt(a)} delta=+{fmt(delta)}')

    print(f'\nimprovements top_n={top_n}')
    improvements = [row for row in sorted(delta_rows, key=lambda r: r[0]) if row[0] < 0][:top_n]
    for delta, (stage, path), b, a in improvements:
        print(f'  {stage:28} {path:18} before={fmt(b)} after={fmt(a)} delta={fmt(delta)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
