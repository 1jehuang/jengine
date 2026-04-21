use jengine::gpu::packed_matvec::{CachedGpuPackedMatvecRunner, SharedGpuPackedContext};
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

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0usize;
    let mut best_value = values[0];
    for (index, value) in values.iter().enumerate().skip(1) {
        if *value > best_value {
            best_value = *value;
            best_index = index;
        }
    }
    best_index
}

fn main() {
    let weights_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let tensor_name = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "model.embed_tokens.weight".to_string());
    let rows = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(151669);
    let cols = std::env::args()
        .nth(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2048);

    let store = WeightStore::load_from_file(&weights_path).expect("weights should load");
    let values = store
        .load_vector_f32(&tensor_name)
        .expect("tensor should load");
    let input = (0..cols)
        .map(|i| (i % 17) as f32 * 0.07 - 0.4)
        .collect::<Vec<_>>();

    let dense = store
        .matvec_f16(&tensor_name, &input)
        .expect("dense baseline should run");
    let dense_argmax = argmax(&dense);

    let (packed, _) =
        pack_ternary_g128(&values, vec![rows, cols], 1e-3).expect("tensor should pack");
    let packed_cpu = matvec_packed_ternary(&packed, &input).expect("cpu packed should run");
    let packed_cpu_argmax = argmax(&packed_cpu);

    let context = SharedGpuPackedContext::new().expect("gpu context should build");
    let code_words = codes_to_words(&packed.packed_codes);
    let (mut runner, compile_duration) = CachedGpuPackedMatvecRunner::new_with_context(
        context,
        &code_words,
        &packed.scales,
        packed.group_size,
        rows,
        cols,
    )
    .expect("gpu packed runner should build");
    let (gpu_argmax, report) = runner
        .run_with_argmax(&input)
        .expect("gpu packed argmax should run");

    let cpu_packed_diff = dense
        .iter()
        .zip(packed_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "tensor={tensor_name} rows={rows} cols={cols} compile_ms={:.3} gpu_ms={:.3} upload_ms={:.3} download_ms={:.3}",
        compile_duration.as_secs_f64() * 1_000.0,
        report.gpu_duration.as_secs_f64() * 1_000.0,
        report.upload_duration.as_secs_f64() * 1_000.0,
        report.download_duration.as_secs_f64() * 1_000.0,
    );
    println!(
        "dense_argmax={} packed_cpu_argmax={} gpu_packed_argmax={} cpu_dense_vs_cpu_packed_max_abs_diff={:.6}",
        dense_argmax,
        packed_cpu_argmax,
        gpu_argmax,
        cpu_packed_diff,
    );
    println!(
        "gpu_matches_packed_cpu={} gpu_matches_dense={}",
        gpu_argmax == packed_cpu_argmax,
        gpu_argmax == dense_argmax,
    );
}
