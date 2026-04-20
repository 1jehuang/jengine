#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

FLOAT_KEYS = [
    'total_ms',
    'embed_ms',
    'norm_ms',
    'qkv_ms',
    'attention_ms',
    'mlp_ms',
    'logits_ms',
    'pack_ms',
    'compile_ms',
    'weight_upload_ms',
    'activation_upload_ms',
    'upload_ms',
    'gpu_ms',
    'download_ms',
    'non_offloaded_dense_ms',
    'orchestration_ms',
]
INT_KEYS = [
    'pack_cache_hits',
    'gpu_cache_hits',
    'dispatch_count',
    'weight_upload_bytes',
    'activation_upload_bytes',
    'upload_bytes',
    'download_bytes',
    'streamed_bytes',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run chunked packed upper-bound capture')
    parser.add_argument('--root', default='/home/jeremy/models/bonsai-1.7b')
    parser.add_argument('--artifact-dir', default='/home/jeremy/jengine/.artifacts/jengine-packed-model')
    parser.add_argument('--token-id', type=int, default=42)
    parser.add_argument('--variant', choices=['attention', 'mlp', 'combined'], default='combined')
    parser.add_argument('--chunk-size', type=int, default=7)
    parser.add_argument('--bin-path', default='./target/release/bench_packed_prefill_chunk')
    return parser.parse_args()


def read_num_hidden_layers(root: str) -> int:
    config = json.loads(pathlib.Path(root, 'config.json').read_text())
    return int(config['num_hidden_layers'])


def parse_summary(summary: str):
    out = {}
    for key in FLOAT_KEYS:
        match = re.search(rf'{key}=([0-9.]+)', summary)
        if match:
            out[key] = float(match.group(1))
    for key in INT_KEYS:
        match = re.search(rf'{key}=([0-9]+)', summary)
        if match:
            out[key] = int(match.group(1))
    return out


def main():
    args = parse_args()
    layers = read_num_hidden_layers(args.root)
    env = os.environ.copy()
    env.setdefault('JENGINE_NO_HEARTBEAT', '1')

    agg = {key: 0.0 for key in FLOAT_KEYS}
    agg.update({key: 0 for key in INT_KEYS})

    with tempfile.TemporaryDirectory(prefix='jengine-chunked-') as tmpdir:
        tmp = pathlib.Path(tmpdir)
        hidden_in = '-'
        for start in range(0, layers, args.chunk_size):
            end = min(start + args.chunk_size, layers)
            hidden_out = tmp / f'hidden_{start}_{end}.json'
            summary_out = tmp / f'summary_{start}_{end}.txt'
            include_logits = '1' if end == layers else '0'
            cmd = [
                args.bin_path,
                args.root,
                args.artifact_dir,
                str(args.token_id),
                str(start),
                str(end),
                args.variant,
                hidden_in,
                str(hidden_out) if end < layers else '-',
                str(summary_out),
                include_logits,
            ]
            print('running', ' '.join(cmd), file=sys.stderr)
            subprocess.run(cmd, check=True, env=env)
            summary = summary_out.read_text().strip()
            print(summary)
            values = parse_summary(summary)
            for key, value in values.items():
                agg[key] += value
            hidden_in = str(hidden_out)

    print('\nchunked_aggregate')
    for key in FLOAT_KEYS:
        print(f'{key}={agg[key]:.3f}')
    for key in INT_KEYS:
        print(f'{key}={agg[key]}')


if __name__ == '__main__':
    main()
