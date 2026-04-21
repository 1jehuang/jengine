use jengine::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, PackedRunnerInputMode, SharedGpuPackedContext,
};
use jengine::gpu::weighted_rms_norm::CachedGpuWeightedRmsNormRunner;
use jengine::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hidden = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(512);
    let intermediate = hidden * 3;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0031).sin() * 0.5 + ((i as f32) * 0.0017).cos() * 0.25)
        .collect();
    let norm_weight = vec![1.0f32; hidden];
    let normalized = weighted_rms_norm(&input, &norm_weight, epsilon);

    let gate: Vec<f32> = (0..intermediate * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 17, 31))
        .collect();
    let up: Vec<f32> = (0..intermediate * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 29, 11))
        .collect();
    let mut concatenated = gate.clone();
    concatenated.extend_from_slice(&up);

    let (packed, _) = pack_ternary_g128(&concatenated, vec![intermediate * 2, hidden], 0.0)?;
    let code_words = packed_codes_to_words(&packed.packed_codes);

    let context = SharedGpuPackedContext::new()?;
    let (mut norm_runner, _) =
        CachedGpuWeightedRmsNormRunner::new_with_context(context.clone(), hidden, epsilon)?;
    let (mut pair_runner, compile_duration) =
        CachedGpuPackedMatvecRunner::new_with_context_and_input_mode(
            context,
            &code_words,
            &packed.scales,
            packed.group_size,
            intermediate * 2,
            hidden,
            PackedRunnerInputMode::RawF32,
        )?;

    let _ = norm_runner.run_resident(&input, &norm_weight)?;
    let mut report = pair_runner.run_resident_from_f32_buffer(
        norm_runner.shared_context(),
        norm_runner.output_buffer_handle(),
        hidden,
        norm_runner.output_buffer_size(),
    )?;
    let (output, download_duration) = pair_runner.read_output()?;
    report.compile_duration = compile_duration;
    report.download_duration = download_duration;

    let reference = matvec_packed_ternary(&packed, &normalized)?;
    let (max_abs_diff, mean_abs_diff) = compare_outputs(&reference, &output);

    println!(
        "hidden={} intermediate={} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
        hidden,
        intermediate,
        report.compile_duration.as_secs_f64() * 1_000.0,
        report.upload_duration.as_secs_f64() * 1_000.0,
        report.gpu_duration.as_secs_f64() * 1_000.0,
        report.download_duration.as_secs_f64() * 1_000.0,
        max_abs_diff,
        mean_abs_diff,
    );

    Ok(())
}

fn weighted_rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let sumsq: f32 = input.iter().map(|v| v * v).sum();
    let scale = (sumsq / input.len() as f32 + epsilon).sqrt().recip();
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * scale * w)
        .collect()
}

fn ternary_value(row: usize, col: usize, a: usize, b: usize) -> f32 {
    match (row.wrapping_mul(a) + col.wrapping_mul(b)) % 3 {
        0 => -1.0,
        1 => 0.0,
        _ => 1.0,
    }
}

fn packed_codes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect()
}

fn compare_outputs(reference: &[f32], output: &[f32]) -> (f32, f32) {
    let (max_abs_diff, sum_abs_diff) = reference.iter().zip(output.iter()).fold(
        (0.0f32, 0.0f32),
        |(max_diff, sum), (expected, actual)| {
            let diff = (expected - actual).abs();
            (max_diff.max(diff), sum + diff)
        },
    );
    (max_abs_diff, sum_abs_diff / reference.len().max(1) as f32)
}
