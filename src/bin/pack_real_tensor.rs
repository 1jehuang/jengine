use jengine::runtime::repack::{analyze_ternary_packability, pack_ternary_g128};
use jengine::runtime::weights::WeightStore;
use std::time::Instant;

fn main() {
    let path = std::env::args()
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

    let store = WeightStore::load_from_file(&path).expect("weight file should load");
    let values = store
        .load_vector_f32(&tensor_name)
        .expect("tensor should load as f32 vector");
    let report = analyze_ternary_packability(&values);
    println!("analyze {}", report.summarize());

    let started_at = Instant::now();
    match pack_ternary_g128(&values, vec![rows, cols], 1e-3) {
        Ok((packed, strict_report)) => {
            println!(
                "strict_pack_ok packed_codes={} scales={} elapsed_ms={:.3} {}",
                packed.packed_codes.len(),
                packed.scales.len(),
                started_at.elapsed().as_secs_f64() * 1_000.0,
                strict_report.summarize(),
            );
        }
        Err(error) => {
            println!(
                "strict_pack_failed elapsed_ms={:.3} error={}",
                started_at.elapsed().as_secs_f64() * 1_000.0,
                error,
            );
        }
    }
}
