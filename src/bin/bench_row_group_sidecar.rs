use jengine::runtime::repack::{
    build_row_group_bitplane_sidecar, build_row_group_pair_sidecar, matvec_packed_ternary,
    matvec_row_group_bitplane_sidecar, matvec_row_group_pair_sidecar, pack_ternary_g128,
};
use jengine::runtime::weights::WeightStore;
use std::time::Instant;

fn time_best_of<F>(iters: usize, mut f: F) -> (Vec<f32>, std::time::Duration)
where
    F: FnMut() -> Vec<f32>,
{
    let mut best_output = Vec::new();
    let mut best_duration = std::time::Duration::MAX;
    for _ in 0..iters {
        let started = Instant::now();
        let output = f();
        let elapsed = started.elapsed();
        if elapsed < best_duration {
            best_duration = elapsed;
            best_output = output;
        }
    }
    (best_output, best_duration)
}

fn main() {
    let weights_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let tensor_name = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "model.layers.0.self_attn.q_proj.weight".to_string());
    let rows = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2048);
    let cols = std::env::args()
        .nth(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2048);
    let iters = std::env::args()
        .nth(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);

    let store = WeightStore::load_from_file(&weights_path).expect("weights should load");
    let values = store
        .load_vector_f32(&tensor_name)
        .expect("tensor should load");
    let input = (0..cols)
        .map(|i| (i % 7) as f32 * 0.1 - 0.3)
        .collect::<Vec<_>>();

    let pack_started = Instant::now();
    let (packed, pack_report) =
        pack_ternary_g128(&values, vec![rows, cols], 1e-3).expect("tensor should pack");
    let pack_elapsed = pack_started.elapsed();

    let pair_sidecar_started = Instant::now();
    let pair_sidecar = build_row_group_pair_sidecar(&packed).expect("pair sidecar should build");
    let pair_sidecar_build_elapsed = pair_sidecar_started.elapsed();

    let bitplane_sidecar_started = Instant::now();
    let bitplane_sidecar =
        build_row_group_bitplane_sidecar(&packed).expect("bitplane sidecar should build");
    let bitplane_sidecar_build_elapsed = bitplane_sidecar_started.elapsed();

    let (packed_output, packed_best) =
        time_best_of(iters, || matvec_packed_ternary(&packed, &input).expect("packed matvec"));
    let (pair_sidecar_output, pair_sidecar_best) = time_best_of(iters, || {
        matvec_row_group_pair_sidecar(&pair_sidecar, &input).expect("pair sidecar matvec")
    });
    let (bitplane_sidecar_output, bitplane_sidecar_best) = time_best_of(iters, || {
        matvec_row_group_bitplane_sidecar(&bitplane_sidecar, &input)
            .expect("bitplane sidecar matvec")
    });

    let pair_max_abs_diff = packed_output
        .iter()
        .zip(pair_sidecar_output.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let pair_mean_abs_diff = if packed_output.is_empty() {
        0.0
    } else {
        packed_output
            .iter()
            .zip(pair_sidecar_output.iter())
            .map(|(left, right)| (left - right).abs())
            .sum::<f32>()
            / packed_output.len() as f32
    };
    let bitplane_max_abs_diff = packed_output
        .iter()
        .zip(bitplane_sidecar_output.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let bitplane_mean_abs_diff = if packed_output.is_empty() {
        0.0
    } else {
        packed_output
            .iter()
            .zip(bitplane_sidecar_output.iter())
            .map(|(left, right)| (left - right).abs())
            .sum::<f32>()
            / packed_output.len() as f32
    };

    println!(
        "tensor={} rows={} cols={} pack_ms={:.3} pair_sidecar_build_ms={:.3} bitplane_sidecar_build_ms={:.3} packed_best_ms={:.3} pair_sidecar_best_ms={:.3} bitplane_sidecar_best_ms={:.3} pair_speedup_x={:.3} bitplane_speedup_x={:.3} pair_max_abs_diff={:.6} pair_mean_abs_diff={:.6} bitplane_max_abs_diff={:.6} bitplane_mean_abs_diff={:.6} pair_code_bytes={} bitplane_mask_bytes={} packed_code_bytes={} scale_count={} reduction_x={:.3}",
        tensor_name,
        rows,
        cols,
        pack_elapsed.as_secs_f64() * 1_000.0,
        pair_sidecar_build_elapsed.as_secs_f64() * 1_000.0,
        bitplane_sidecar_build_elapsed.as_secs_f64() * 1_000.0,
        packed_best.as_secs_f64() * 1_000.0,
        pair_sidecar_best.as_secs_f64() * 1_000.0,
        bitplane_sidecar_best.as_secs_f64() * 1_000.0,
        packed_best.as_secs_f64() / pair_sidecar_best.as_secs_f64(),
        packed_best.as_secs_f64() / bitplane_sidecar_best.as_secs_f64(),
        pair_max_abs_diff,
        pair_mean_abs_diff,
        bitplane_max_abs_diff,
        bitplane_mean_abs_diff,
        pair_sidecar.pair_codes.len(),
        (bitplane_sidecar.positive_masks.len() + bitplane_sidecar.negative_masks.len())
            * std::mem::size_of::<u32>(),
        packed.packed_codes.len(),
        pair_sidecar.scales.len(),
        pack_report.reduction_ratio_vs_fp16,
    );
}
