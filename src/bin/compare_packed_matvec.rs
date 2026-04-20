use jengine::runtime::repack::{
    matvec_packed_ternary, matvec_packed_ternary_reference, pack_ternary_g128,
};
use jengine::runtime::weights::WeightStore;
use std::time::Instant;

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

    let store = WeightStore::load_from_file(&weights_path).expect("weights should load");
    let values = store
        .load_vector_f32(&tensor_name)
        .expect("tensor should load");
    let input = (0..cols)
        .map(|i| (i % 7) as f32 * 0.1 - 0.3)
        .collect::<Vec<_>>();

    let started = Instant::now();
    let dense = store
        .matvec_f16(&tensor_name, &input)
        .expect("dense matvec should work");
    let dense_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let (packed, _) =
        pack_ternary_g128(&values, vec![rows, cols], 1e-3).expect("tensor should pack");
    let started = Instant::now();
    let packed_reference_out =
        matvec_packed_ternary_reference(&packed, &input).expect("reference packed matvec");
    let packed_reference_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let started = Instant::now();
    let packed_out = matvec_packed_ternary(&packed, &input).expect("packed matvec should work");
    let packed_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let max_abs_diff_reference = dense
        .iter()
        .zip(packed_reference_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let max_abs_diff = dense
        .iter()
        .zip(packed_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "dense_ms={:.3} packed_reference_ms={:.3} packed_ms={:.3} packed_vs_reference_speedup_x={:.3} packed_vs_dense_speedup_x={:.3} max_abs_diff_reference={:.6} max_abs_diff={:.6}",
        dense_ms,
        packed_reference_ms,
        packed_ms,
        packed_reference_ms / packed_ms,
        dense_ms / packed_ms,
        max_abs_diff_reference,
        max_abs_diff,
    );
}
