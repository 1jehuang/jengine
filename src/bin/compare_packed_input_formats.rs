use jengine::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, SharedGpuPackedContext,
    run_packed_ternary_matvec_raw_f32_with_output, run_packed_ternary_matvec_with_output,
};
use jengine::runtime::repack::pack_ternary_g128;
use jengine::runtime::weights::WeightStore;

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
        .expect("dense matvec should work");

    let (packed, _) =
        pack_ternary_g128(&values, vec![rows, cols], 1e-3).expect("tensor should pack");
    let code_words = packed
        .packed_codes
        .chunks(4)
        .map(|chunk| {
            let mut word = 0u32;
            for (shift, byte) in chunk.iter().enumerate() {
                word |= (*byte as u32) << (shift * 8);
            }
            word
        })
        .collect::<Vec<_>>();

    let (_half_out, half_report) = run_packed_ternary_matvec_with_output(
        &code_words,
        &packed.scales,
        packed.group_size,
        rows,
        cols,
        &input,
        Some(&dense),
    )
    .expect("packed half-pair GPU run should succeed");

    let (_raw_out, raw_report) = run_packed_ternary_matvec_raw_f32_with_output(
        &code_words,
        &packed.scales,
        packed.group_size,
        rows,
        cols,
        &input,
        Some(&dense),
    )
    .expect("packed raw-f32 GPU run should succeed");

    let dense_chained = store
        .matvec_f16(&tensor_name, &dense)
        .expect("dense chained matvec should work");
    let shared_context = SharedGpuPackedContext::new().expect("shared context should initialize");
    let (mut source_runner, _) = CachedGpuPackedMatvecRunner::new_with_context(
        shared_context.clone(),
        &code_words,
        &packed.scales,
        packed.group_size,
        rows,
        cols,
    )
    .expect("source runner should initialize");
    let (mut chained_runner, _) = CachedGpuPackedMatvecRunner::new_uninitialized_raw_f32_input_with_context(
        shared_context,
        code_words.len(),
        packed.scales.len(),
        packed.group_size,
        rows,
        cols,
    )
    .expect("chained runner should initialize");
    chained_runner
        .update_weights(&code_words, &packed.scales)
        .expect("chained runner weights should upload");
    let (_source_out, source_report) = source_runner
        .run_with_output(&input, Some(&dense))
        .expect("source runner should execute");
    let (_chained_out, chained_report) = chained_runner
        .run_with_output_from_runner(&source_runner, Some(&dense_chained))
        .expect("chained runner should execute from resident output");

    println!(
        "half_pair={{ {} }} raw_f32={{ {} }} source_half_pair={{ {} }} chained_raw_f32={{ {} }}",
        half_report.summarize(),
        raw_report.summarize(),
        source_report.summarize(),
        chained_report.summarize()
    );
}
