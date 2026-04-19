use jengine::gpu::packed_matvec::run_packed_ternary_matvec;
use jengine::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};
use jengine::runtime::weights::WeightStore;

fn codes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut word = 0u32;
            for (i, byte) in chunk.iter().enumerate() {
                word |= (*byte as u32) << (i * 8);
            }
            word
        })
        .collect()
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

    let store = WeightStore::load_from_file(&weights_path).expect("weights should load");
    let values = store
        .load_vector_f32(&tensor_name)
        .expect("tensor should load");
    let input = (0..cols)
        .map(|i| (i % 7) as f32 * 0.1 - 0.3)
        .collect::<Vec<_>>();
    let dense = store
        .matvec_f16(&tensor_name, &input)
        .expect("dense baseline should run");
    let (packed, _) =
        pack_ternary_g128(&values, vec![rows, cols], 1e-3).expect("tensor should pack");
    let cpu_packed = matvec_packed_ternary(&packed, &input).expect("cpu packed should run");
    let code_words = codes_to_words(&packed.packed_codes);
    let report = run_packed_ternary_matvec(
        &code_words,
        &packed.scales,
        packed.group_size,
        rows,
        cols,
        &input,
        &cpu_packed,
    )
    .expect("gpu packed should run");
    let max_abs_diff_dense = dense
        .iter()
        .zip(cpu_packed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "cpu_dense_vs_cpu_packed_max_abs_diff={:.6}",
        max_abs_diff_dense
    );
    println!("{}", report.summarize());
}
