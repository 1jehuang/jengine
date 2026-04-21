use half::f16;
use jengine::cpu::primitives::swiglu;
use jengine::gpu::packed_pair_swiglu_pack::run_packed_pair_swiglu_pack_with_output;
use jengine::runtime::repack::{PackedTernaryTensor, matvec_packed_ternary, pack_ternary_g128};
use jengine::runtime::weights::WeightStore;

fn packed_codes_to_words(bytes: &[u8]) -> Vec<u32> {
    let mut words = Vec::with_capacity(bytes.len().div_ceil(4));
    for chunk in bytes.chunks(4) {
        let mut word = 0u32;
        for (idx, byte) in chunk.iter().enumerate() {
            word |= (*byte as u32) << (idx * 8);
        }
        words.push(word);
    }
    words
}

fn pack_f16_pairs(values: &[f32]) -> Vec<u32> {
    values
        .chunks(2)
        .map(|chunk| {
            let a = f16::from_f32(chunk[0]).to_bits() as u32;
            let b = chunk
                .get(1)
                .map(|v| f16::from_f32(*v).to_bits() as u32)
                .unwrap_or(0);
            a | (b << 16)
        })
        .collect()
}

fn main() {
    let weights_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let gate_name = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "model.layers.0.mlp.gate_proj.weight".to_string());
    let up_name = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "model.layers.0.mlp.up_proj.weight".to_string());

    let store = WeightStore::load_from_file(&weights_path).expect("weights should load");
    let gate_values = store.load_vector_f32(&gate_name).expect("gate should load");
    let up_values = store.load_vector_f32(&up_name).expect("up should load");
    let cols = 2048usize;
    let rows = gate_values.len() / cols;

    let (gate_packed, _) = pack_ternary_g128(&gate_values, vec![rows, cols], 1e-3).expect("pack gate");
    let (up_packed, _) = pack_ternary_g128(&up_values, vec![rows, cols], 1e-3).expect("pack up");
    let mut code_words = packed_codes_to_words(&gate_packed.packed_codes);
    code_words.extend_from_slice(&packed_codes_to_words(&up_packed.packed_codes));
    let mut scales = gate_packed.scales.clone();
    scales.extend_from_slice(&up_packed.scales);

    let input = (0..cols)
        .map(|i| (i % 17) as f32 * 0.03125 - 0.25)
        .collect::<Vec<_>>();
    let gate_tensor = PackedTernaryTensor {
        shape: vec![rows, cols],
        original_len: gate_values.len(),
        group_size: gate_packed.group_size,
        packed_codes: gate_packed.packed_codes.clone(),
        scales: gate_packed.scales.clone(),
    };
    let up_tensor = PackedTernaryTensor {
        shape: vec![rows, cols],
        original_len: up_values.len(),
        group_size: up_packed.group_size,
        packed_codes: up_packed.packed_codes.clone(),
        scales: up_packed.scales.clone(),
    };
    let gate = matvec_packed_ternary(&gate_tensor, &input).expect("packed gate matvec");
    let up = matvec_packed_ternary(&up_tensor, &input).expect("packed up matvec");
    let reference = pack_f16_pairs(&swiglu(&gate, &up));

    let (_output, report) = run_packed_pair_swiglu_pack_with_output(
        &code_words,
        &scales,
        gate_packed.group_size,
        rows,
        cols,
        &input,
        Some(&reference),
    )
    .expect("gpu fused pair swiglu pack should succeed");

    println!(
        "rows={} cols={} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} mismatched_words={}",
        report.rows,
        report.cols,
        report.compile_duration.as_secs_f64() * 1_000.0,
        report.upload_duration.as_secs_f64() * 1_000.0,
        report.gpu_duration.as_secs_f64() * 1_000.0,
        report.download_duration.as_secs_f64() * 1_000.0,
        report.mismatched_words,
    );
}
