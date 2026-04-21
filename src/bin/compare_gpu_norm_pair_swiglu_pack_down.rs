use half::f16;
use jengine::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, PackedRunnerInputMode, SharedGpuPackedContext,
};
use jengine::gpu::swiglu_pack_f16_pairs::CachedGpuSwigluPackF16PairsRunner;
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
    let down: Vec<f32> = (0..hidden * intermediate)
        .map(|idx| ternary_value(idx / intermediate, idx % intermediate, 13, 19))
        .collect();

    let mut concatenated = gate.clone();
    concatenated.extend_from_slice(&up);

    let (pair_packed, _) = pack_ternary_g128(&concatenated, vec![intermediate * 2, hidden], 0.0)?;
    let (down_packed, _) = pack_ternary_g128(&down, vec![hidden, intermediate], 0.0)?;
    let pair_code_words = packed_codes_to_words(&pair_packed.packed_codes);
    let down_code_words = packed_codes_to_words(&down_packed.packed_codes);

    let pair_ref = matvec_packed_ternary(&pair_packed, &normalized)?;
    let swiglu_ref = quantize_f16(&swiglu(
        &pair_ref[..intermediate],
        &pair_ref[intermediate..],
    ));
    let down_ref = matvec_packed_ternary(&down_packed, &swiglu_ref)?;

    let context = SharedGpuPackedContext::new()?;
    let (mut norm_runner, norm_compile_duration) =
        CachedGpuWeightedRmsNormRunner::new_with_context(context.clone(), hidden, epsilon)?;
    let (mut pair_runner, pair_compile_duration) =
        CachedGpuPackedMatvecRunner::new_with_context_and_input_mode(
            context.clone(),
            &pair_code_words,
            &pair_packed.scales,
            pair_packed.group_size,
            intermediate * 2,
            hidden,
            PackedRunnerInputMode::RawF32,
        )?;
    let (mut swiglu_pack_runner, swiglu_compile_duration) =
        CachedGpuSwigluPackF16PairsRunner::new_with_context(context.clone(), intermediate)?;
    let (mut down_runner, down_compile_duration) = CachedGpuPackedMatvecRunner::new_with_context(
        context,
        &down_code_words,
        &down_packed.scales,
        down_packed.group_size,
        hidden,
        intermediate,
    )?;

    let norm_report = norm_runner.run_resident(&input, &norm_weight)?;
    let pair_report = pair_runner.run_resident_from_f32_buffer(
        norm_runner.shared_context(),
        norm_runner.output_buffer_handle(),
        hidden,
        norm_runner.output_buffer_size(),
    )?;
    let swiglu_report = swiglu_pack_runner.run_with_output_from_buffer(
        pair_runner.shared_context(),
        pair_runner.output_buffer_handle(),
        intermediate * 2,
        pair_runner.output_buffer_size(),
    )?;
    let mut down_report = down_runner.run_resident_from_packed_buffer(
        swiglu_pack_runner.shared_context(),
        swiglu_pack_runner.output_buffer_handle(),
        swiglu_pack_runner.packed_len(),
        swiglu_pack_runner.output_buffer_size(),
    )?;
    let (down_output, down_download_duration) = down_runner.read_output()?;
    down_report.download_duration = down_download_duration;

    let (max_abs_diff, mean_abs_diff) = compare_outputs(&down_ref, &down_output);
    println!(
        "hidden={} intermediate={} norm_compile_ms={:.3} norm_upload_ms={:.3} norm_gpu_ms={:.3} pair_compile_ms={:.3} pair_upload_ms={:.3} pair_gpu_ms={:.3} swiglu_compile_ms={:.3} swiglu_upload_ms={:.3} swiglu_gpu_ms={:.3} down_compile_ms={:.3} down_upload_ms={:.3} down_gpu_ms={:.3} down_download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
        hidden,
        intermediate,
        norm_compile_duration.as_secs_f64() * 1_000.0,
        norm_report.upload_duration.as_secs_f64() * 1_000.0,
        norm_report.gpu_duration.as_secs_f64() * 1_000.0,
        pair_compile_duration.as_secs_f64() * 1_000.0,
        pair_report.upload_duration.as_secs_f64() * 1_000.0,
        pair_report.gpu_duration.as_secs_f64() * 1_000.0,
        swiglu_compile_duration.as_secs_f64() * 1_000.0,
        swiglu_report.upload_duration.as_secs_f64() * 1_000.0,
        swiglu_report.gpu_duration.as_secs_f64() * 1_000.0,
        down_compile_duration.as_secs_f64() * 1_000.0,
        down_report.upload_duration.as_secs_f64() * 1_000.0,
        down_report.gpu_duration.as_secs_f64() * 1_000.0,
        down_report.download_duration.as_secs_f64() * 1_000.0,
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

fn quantize_f16(values: &[f32]) -> Vec<f32> {
    values.iter().map(|v| f16::from_f32(*v).to_f32()).collect()
}
