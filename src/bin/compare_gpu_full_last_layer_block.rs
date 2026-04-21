use half::f16;
use jengine::gpu::pack_f16_pairs::CachedGpuPackF16PairsRunner;
use jengine::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, PackedRunnerInputMode, SharedGpuPackedContext,
};
use jengine::gpu::swiglu_pack_f16_pairs::CachedGpuSwigluPackF16PairsRunner;
use jengine::gpu::vector_add::CachedGpuVectorAddRunner;
use jengine::gpu::weighted_rms_norm::CachedGpuWeightedRmsNormRunner;
use jengine::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hidden = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(512);
    let intermediate = std::env::args()
        .nth(2)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(hidden * 3);
    let vocab = std::env::args()
        .nth(3)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(2048);
    let epsilon = 1e-5f32;

    let post_attention_residual: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0021).sin() * 0.45 + ((i as f32) * 0.0013).cos() * 0.2)
        .collect();
    let mlp_residual: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0017).sin() * 0.35 - ((i as f32) * 0.0029).cos() * 0.15)
        .collect();
    let post_norm_weight: Vec<f32> = (0..hidden).map(|i| 0.75 + (i % 17) as f32 * 0.01).collect();
    let final_norm_weight: Vec<f32> = (0..hidden).map(|i| 0.9 + (i % 13) as f32 * 0.008).collect();

    let gate: Vec<f32> = (0..intermediate * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 17, 31))
        .collect();
    let up: Vec<f32> = (0..intermediate * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 29, 11))
        .collect();
    let down: Vec<f32> = (0..hidden * intermediate)
        .map(|idx| ternary_value(idx / intermediate, idx % intermediate, 13, 19))
        .collect();
    let logits: Vec<f32> = (0..vocab * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 7, 23))
        .collect();

    let mut pair_concat = gate.clone();
    pair_concat.extend_from_slice(&up);

    let (pair_packed, _) = pack_ternary_g128(&pair_concat, vec![intermediate * 2, hidden], 0.0)?;
    let (down_packed, _) = pack_ternary_g128(&down, vec![hidden, intermediate], 0.0)?;
    let (logits_packed, _) = pack_ternary_g128(&logits, vec![vocab, hidden], 0.0)?;

    let post_norm_ref = weighted_rms_norm(&post_attention_residual, &post_norm_weight, epsilon);
    let pair_ref = matvec_packed_ternary(&pair_packed, &post_norm_ref)?;
    let swiglu_ref = quantize_f16(&swiglu(
        &pair_ref[..intermediate],
        &pair_ref[intermediate..],
    ));
    let down_ref = matvec_packed_ternary(&down_packed, &swiglu_ref)?;
    let hidden_ref: Vec<f32> = down_ref
        .iter()
        .zip(mlp_residual.iter())
        .map(|(a, b)| a + b)
        .collect();
    let final_norm_ref = quantize_f16(&weighted_rms_norm(&hidden_ref, &final_norm_weight, epsilon));
    let logits_ref = matvec_packed_ternary(&logits_packed, &final_norm_ref)?;
    let expected_argmax = argmax(&logits_ref);

    let pair_code_words = packed_codes_to_words(&pair_packed.packed_codes);
    let down_code_words = packed_codes_to_words(&down_packed.packed_codes);
    let logits_code_words = packed_codes_to_words(&logits_packed.packed_codes);

    let zeros = vec![0.0f32; hidden];
    let context = SharedGpuPackedContext::new()?;
    let (mut post_norm_runner, post_norm_compile) =
        CachedGpuWeightedRmsNormRunner::new_with_context(context.clone(), hidden, epsilon)?;
    let (mut pair_runner, pair_compile) =
        CachedGpuPackedMatvecRunner::new_with_context_and_input_mode(
            context.clone(),
            &pair_code_words,
            &pair_packed.scales,
            pair_packed.group_size,
            intermediate * 2,
            hidden,
            PackedRunnerInputMode::RawF32,
        )?;
    let (mut swiglu_pack_runner, swiglu_pack_compile) =
        CachedGpuSwigluPackF16PairsRunner::new_with_context(context.clone(), intermediate)?;
    let (mut down_runner, down_compile) = CachedGpuPackedMatvecRunner::new_with_context(
        context.clone(),
        &down_code_words,
        &down_packed.scales,
        down_packed.group_size,
        hidden,
        intermediate,
    )?;
    let (mut residual_seed_runner, residual_seed_compile) =
        CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)?;
    let (mut add_runner, add_compile) =
        CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)?;
    let (mut final_norm_runner, final_norm_compile) =
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

    let post_norm_report =
        post_norm_runner.run_resident(&post_attention_residual, &post_norm_weight)?;
    let pair_report = pair_runner.run_resident_from_f32_buffer(
        post_norm_runner.shared_context(),
        post_norm_runner.output_buffer_handle(),
        hidden,
        post_norm_runner.output_buffer_size(),
    )?;
    let swiglu_pack_report = swiglu_pack_runner.run_with_output_from_buffer(
        pair_runner.shared_context(),
        pair_runner.output_buffer_handle(),
        intermediate * 2,
        pair_runner.output_buffer_size(),
    )?;
    let down_report = down_runner.run_resident_from_packed_buffer(
        swiglu_pack_runner.shared_context(),
        swiglu_pack_runner.output_buffer_handle(),
        swiglu_pack_runner.packed_len(),
        swiglu_pack_runner.output_buffer_size(),
    )?;
    let _ = residual_seed_runner.run_with_output(&mlp_residual, &zeros, None)?;
    let add_report = add_runner.run_resident_from_buffers(
        down_runner.shared_context(),
        down_runner.output_buffer_handle(),
        hidden,
        down_runner.output_buffer_size(),
        residual_seed_runner.output_buffer_handle(),
        hidden,
        residual_seed_runner.output_buffer_size(),
    )?;
    let final_norm_report = final_norm_runner.run_resident_from_f32_buffer(
        add_runner.shared_context(),
        add_runner.output_buffer_handle(),
        hidden,
        add_runner.output_buffer_size(),
        &final_norm_weight,
    )?;
    let pack_report = pack_runner.run_resident_from_f32_buffer(
        final_norm_runner.shared_context(),
        final_norm_runner.output_buffer_handle(),
        hidden,
        final_norm_runner.output_buffer_size(),
    )?;
    let mut logits_report = logits_runner.run_resident_from_packed_buffer(
        pack_runner.shared_context(),
        pack_runner.output_buffer_handle(),
        pack_runner.packed_len(),
        pack_runner.output_buffer_size(),
    )?;
    let (logits_output, logits_download_duration) = logits_runner.read_output()?;
    logits_report.download_duration = logits_download_duration;
    let actual_argmax = argmax(&logits_output);
    let (max_abs_diff, mean_abs_diff) = compare_outputs(&logits_ref, &logits_output);

    println!(
        "hidden={} intermediate={} vocab={} post_norm_compile_ms={:.3} pair_compile_ms={:.3} swiglu_pack_compile_ms={:.3} down_compile_ms={:.3} residual_seed_compile_ms={:.3} add_compile_ms={:.3} final_norm_compile_ms={:.3} pack_compile_ms={:.3} logits_compile_ms={:.3} post_norm_gpu_ms={:.3} pair_gpu_ms={:.3} swiglu_pack_gpu_ms={:.3} down_gpu_ms={:.3} add_gpu_ms={:.3} final_norm_gpu_ms={:.3} pack_gpu_ms={:.3} logits_gpu_ms={:.3} logits_download_ms={:.3} expected_argmax={} actual_argmax={} max_abs_diff={:.6} mean_abs_diff={:.6}",
        hidden,
        intermediate,
        vocab,
        post_norm_compile.as_secs_f64() * 1_000.0,
        pair_compile.as_secs_f64() * 1_000.0,
        swiglu_pack_compile.as_secs_f64() * 1_000.0,
        down_compile.as_secs_f64() * 1_000.0,
        residual_seed_compile.as_secs_f64() * 1_000.0,
        add_compile.as_secs_f64() * 1_000.0,
        final_norm_compile.as_secs_f64() * 1_000.0,
        pack_compile.as_secs_f64() * 1_000.0,
        logits_compile.as_secs_f64() * 1_000.0,
        post_norm_report.gpu_duration.as_secs_f64() * 1_000.0,
        pair_report.gpu_duration.as_secs_f64() * 1_000.0,
        swiglu_pack_report.gpu_duration.as_secs_f64() * 1_000.0,
        down_report.gpu_duration.as_secs_f64() * 1_000.0,
        add_report.gpu_duration.as_secs_f64() * 1_000.0,
        final_norm_report.gpu_duration.as_secs_f64() * 1_000.0,
        pack_report.gpu_duration.as_secs_f64() * 1_000.0,
        logits_report.gpu_duration.as_secs_f64() * 1_000.0,
        logits_report.download_duration.as_secs_f64() * 1_000.0,
        expected_argmax,
        actual_argmax,
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

fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| (g * (1.0 / (1.0 + (-g).exp()))) * u)
        .collect()
}

fn quantize_f16(values: &[f32]) -> Vec<f32> {
    values.iter().map(|v| f16::from_f32(*v).to_f32()).collect()
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

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
