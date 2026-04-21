use half::f16;
use jengine::gpu::pack_f16_pairs::CachedGpuPackF16PairsRunner;
use jengine::gpu::packed_matvec::{CachedGpuPackedMatvecRunner, SharedGpuPackedContext};
use jengine::gpu::vector_add::CachedGpuVectorAddRunner;
use jengine::gpu::weighted_rms_norm::CachedGpuWeightedRmsNormRunner;
use jengine::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hidden = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(512);
    let vocab = std::env::args()
        .nth(2)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(2048);
    let epsilon = 1e-5f32;

    let down: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0041).sin() * 0.4)
        .collect();
    let residual: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0023).cos() * 0.3)
        .collect();
    let final_norm_weight = vec![1.0f32; hidden];

    let logits_dense: Vec<f32> = (0..vocab * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 7, 23))
        .collect();
    let (logits_packed, _) = pack_ternary_g128(&logits_dense, vec![vocab, hidden], 0.0)?;
    let logits_code_words = packed_codes_to_words(&logits_packed.packed_codes);

    let hidden_ref: Vec<f32> = down
        .iter()
        .zip(residual.iter())
        .map(|(a, b)| a + b)
        .collect();
    let final_norm_ref = quantize_f16(&weighted_rms_norm(&hidden_ref, &final_norm_weight, epsilon));
    let logits_ref = matvec_packed_ternary(&logits_packed, &final_norm_ref)?;
    let expected_argmax = logits_ref
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let zeros = vec![0.0f32; hidden];
    let context = SharedGpuPackedContext::new()?;
    let (mut down_seed, down_seed_compile) =
        CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)?;
    let (mut residual_seed, residual_seed_compile) =
        CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)?;
    let (mut add_runner, add_compile) =
        CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)?;
    let (mut norm_runner, norm_compile) =
        CachedGpuWeightedRmsNormRunner::new_with_context(context.clone(), hidden, epsilon)?;
    let (mut pack_runner, pack_compile) =
        CachedGpuPackF16PairsRunner::new_with_context(context.clone(), hidden)?;
    let (mut logits_runner, logits_compile) = CachedGpuPackedMatvecRunner::new_with_context(
        context,
        &logits_code_words,
        &logits_packed.scales,
        logits_packed.group_size,
        vocab,
        hidden,
    )?;

    let down_seed_report = down_seed.run_with_output(&down, &zeros, None)?.1;
    let residual_seed_report = residual_seed.run_with_output(&residual, &zeros, None)?.1;
    let add_report = add_runner.run_resident_from_buffers(
        down_seed.shared_context(),
        down_seed.output_buffer_handle(),
        hidden,
        down_seed.output_buffer_size(),
        residual_seed.output_buffer_handle(),
        hidden,
        residual_seed.output_buffer_size(),
    )?;
    let norm_report = norm_runner.run_resident_from_f32_buffer(
        add_runner.shared_context(),
        add_runner.output_buffer_handle(),
        hidden,
        add_runner.output_buffer_size(),
        &final_norm_weight,
    )?;
    let pack_report = pack_runner.run_resident_from_f32_buffer(
        norm_runner.shared_context(),
        norm_runner.output_buffer_handle(),
        hidden,
        norm_runner.output_buffer_size(),
    )?;
    let mut logits_report = logits_runner.run_resident_from_packed_buffer(
        pack_runner.shared_context(),
        pack_runner.output_buffer_handle(),
        pack_runner.packed_len(),
        pack_runner.output_buffer_size(),
    )?;
    let (logits_output, logits_download_duration) = logits_runner.read_output()?;
    logits_report.download_duration = logits_download_duration;
    let actual_argmax = logits_output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let (max_abs_diff, mean_abs_diff) = compare_outputs(&logits_ref, &logits_output);

    println!(
        "hidden={} vocab={} down_seed_compile_ms={:.3} residual_seed_compile_ms={:.3} add_compile_ms={:.3} norm_compile_ms={:.3} pack_compile_ms={:.3} logits_compile_ms={:.3} add_gpu_ms={:.3} norm_gpu_ms={:.3} pack_gpu_ms={:.3} logits_gpu_ms={:.3} logits_download_ms={:.3} expected_argmax={} actual_argmax={} max_abs_diff={:.6} mean_abs_diff={:.6}",
        hidden,
        vocab,
        down_seed_compile.as_secs_f64() * 1_000.0,
        residual_seed_compile.as_secs_f64() * 1_000.0,
        add_compile.as_secs_f64() * 1_000.0,
        norm_compile.as_secs_f64() * 1_000.0,
        pack_compile.as_secs_f64() * 1_000.0,
        logits_compile.as_secs_f64() * 1_000.0,
        add_report.gpu_duration.as_secs_f64() * 1_000.0,
        norm_report.gpu_duration.as_secs_f64() * 1_000.0,
        pack_report.gpu_duration.as_secs_f64() * 1_000.0,
        logits_report.gpu_duration.as_secs_f64() * 1_000.0,
        logits_report.download_duration.as_secs_f64() * 1_000.0,
        expected_argmax,
        actual_argmax,
        max_abs_diff,
        mean_abs_diff,
    );

    let _ = down_seed_report;
    let _ = residual_seed_report;
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

fn quantize_f16(values: &[f32]) -> Vec<f32> {
    values.iter().map(|v| f16::from_f32(*v).to_f32()).collect()
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
