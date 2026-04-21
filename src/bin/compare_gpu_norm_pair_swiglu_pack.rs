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
    let mut concatenated = gate.clone();
    concatenated.extend_from_slice(&up);

    let (packed, _) = pack_ternary_g128(&concatenated, vec![intermediate * 2, hidden], 0.0)?;
    let code_words = packed_codes_to_words(&packed.packed_codes);

    let gate_ref = matvec_packed_ternary(&packed, &normalized)?;
    let reference = pack_f16_pairs(&swiglu(
        &gate_ref[..intermediate],
        &gate_ref[intermediate..],
    ));

    let context = SharedGpuPackedContext::new()?;
    let (mut norm_runner, norm_compile_duration) =
        CachedGpuWeightedRmsNormRunner::new_with_context(context.clone(), hidden, epsilon)?;
    let (mut pair_runner, pair_compile_duration) =
        CachedGpuPackedMatvecRunner::new_with_context_and_input_mode(
            context.clone(),
            &code_words,
            &packed.scales,
            packed.group_size,
            intermediate * 2,
            hidden,
            PackedRunnerInputMode::RawF32,
        )?;
    let (mut swiglu_pack_runner, swiglu_compile_duration) =
        CachedGpuSwigluPackF16PairsRunner::new_with_context(context, intermediate)?;

    let norm_report = norm_runner.run_resident(&input, &norm_weight)?;
    let pair_report = pair_runner.run_resident_from_f32_buffer(
        norm_runner.shared_context(),
        norm_runner.output_buffer_handle(),
        hidden,
        norm_runner.output_buffer_size(),
    )?;
    let mut swiglu_report = swiglu_pack_runner.run_with_output_from_buffer(
        pair_runner.shared_context(),
        pair_runner.output_buffer_handle(),
        intermediate * 2,
        pair_runner.output_buffer_size(),
    )?;
    let (output, download_duration) = swiglu_pack_runner.read_output()?;
    swiglu_report.download_duration = download_duration;
    let mismatched_words = reference
        .iter()
        .zip(output.iter())
        .filter(|(a, b)| a != b)
        .count();

    println!(
        "hidden={} intermediate={} norm_compile_ms={:.3} norm_upload_ms={:.3} norm_gpu_ms={:.3} pair_compile_ms={:.3} pair_upload_ms={:.3} pair_gpu_ms={:.3} swiglu_compile_ms={:.3} swiglu_upload_ms={:.3} swiglu_gpu_ms={:.3} mismatched_words={}",
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
        mismatched_words,
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
