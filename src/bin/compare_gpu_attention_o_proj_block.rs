use jengine::gpu::attention_single_query::CachedGpuAttentionSingleQueryRunner;
use jengine::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, PackedRunnerInputMode, SharedGpuPackedContext,
};
use jengine::gpu::vector_add::CachedGpuVectorAddRunner;
use jengine::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seq_len = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(64);
    let num_query_heads = std::env::args()
        .nth(2)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    let num_key_value_heads = std::env::args()
        .nth(3)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(4);
    let head_dim = std::env::args()
        .nth(4)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(64);

    let hidden = num_query_heads * head_dim;
    let query_len = hidden;
    let kv_len = seq_len * num_key_value_heads * head_dim;

    let query: Vec<f32> = (0..query_len)
        .map(|i| ((i as f32) * 0.0021).sin() * 0.5 + ((i as f32) * 0.0017).cos() * 0.25)
        .collect();
    let keys: Vec<f32> = (0..kv_len)
        .map(|i| ((i as f32) * 0.0013).sin() * 0.45 - ((i as f32) * 0.0031).cos() * 0.15)
        .collect();
    let values: Vec<f32> = (0..kv_len)
        .map(|i| ((i as f32) * 0.0019).cos() * 0.35 + ((i as f32) * 0.0027).sin() * 0.1)
        .collect();
    let residual: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0011).sin() * 0.4 - ((i as f32) * 0.0023).cos() * 0.2)
        .collect();

    let o_proj_dense: Vec<f32> = (0..hidden * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 11, 17))
        .collect();
    let (o_proj_packed, _) = pack_ternary_g128(&o_proj_dense, vec![hidden, hidden], 0.0)?;
    let o_proj_code_words = packed_codes_to_words(&o_proj_packed.packed_codes);

    let attn_ref = attention_single_query_cpu(
        &query,
        &keys,
        &values,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
    );
    let o_proj_ref = matvec_packed_ternary(&o_proj_packed, &attn_ref)?;
    let hidden_ref: Vec<f32> = o_proj_ref
        .iter()
        .zip(residual.iter())
        .map(|(a, b)| a + b)
        .collect();

    let zeros = vec![0.0f32; hidden];
    let context = SharedGpuPackedContext::new()?;
    let (mut attn_runner, attn_compile) = CachedGpuAttentionSingleQueryRunner::new_with_context(
        context.clone(),
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
    )?;
    let (mut o_proj_runner, o_proj_compile) =
        CachedGpuPackedMatvecRunner::new_with_context_and_input_mode(
            context.clone(),
            &o_proj_code_words,
            &o_proj_packed.scales,
            o_proj_packed.group_size,
            hidden,
            hidden,
            PackedRunnerInputMode::RawF32,
        )?;
    let (mut residual_seed_runner, residual_seed_compile) =
        CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)?;
    let (mut add_runner, add_compile) =
        CachedGpuVectorAddRunner::new_with_context(context, hidden)?;

    let attn_report = attn_runner.run_with_output(&query, &keys, &values, None)?.1;
    let o_proj_report = o_proj_runner.run_resident_from_f32_buffer(
        attn_runner.shared_context(),
        attn_runner.output_buffer_handle(),
        hidden,
        attn_runner.output_buffer_size(),
    )?;
    let _ = residual_seed_runner.run_with_output(&residual, &zeros, None)?;
    let add_report = add_runner.run_resident_from_buffers(
        o_proj_runner.shared_context(),
        o_proj_runner.output_buffer_handle(),
        hidden,
        o_proj_runner.output_buffer_size(),
        residual_seed_runner.output_buffer_handle(),
        hidden,
        residual_seed_runner.output_buffer_size(),
    )?;
    let (hidden_output, download_duration) = add_runner.read_output()?;

    let (max_abs_diff, mean_abs_diff) = compare_outputs(&hidden_ref, &hidden_output);
    println!(
        "seq_len={} q_heads={} kv_heads={} head_dim={} attn_compile_ms={:.3} oproj_compile_ms={:.3} residual_seed_compile_ms={:.3} add_compile_ms={:.3} attn_gpu_ms={:.3} oproj_gpu_ms={:.3} add_gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        attn_compile.as_secs_f64() * 1_000.0,
        o_proj_compile.as_secs_f64() * 1_000.0,
        residual_seed_compile.as_secs_f64() * 1_000.0,
        add_compile.as_secs_f64() * 1_000.0,
        attn_report.gpu_duration.as_secs_f64() * 1_000.0,
        o_proj_report.gpu_duration.as_secs_f64() * 1_000.0,
        add_report.gpu_duration.as_secs_f64() * 1_000.0,
        download_duration.as_secs_f64() * 1_000.0,
        max_abs_diff,
        mean_abs_diff,
    );

    Ok(())
}

fn attention_single_query_cpu(
    query: &[f32],
    cached_keys: &[f32],
    cached_values: &[f32],
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_rep = num_query_heads / num_key_value_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0; num_query_heads * head_dim];

    for head in 0..num_query_heads {
        let kv_head = head / n_rep;
        let q_offset = head * head_dim;
        let q_slice = &query[q_offset..q_offset + head_dim];
        let mut logits = vec![0.0; seq_len];
        for (position, logit) in logits.iter_mut().enumerate().take(seq_len) {
            let k_offset = (position * num_key_value_heads + kv_head) * head_dim;
            let k_slice = &cached_keys[k_offset..k_offset + head_dim];
            *logit = q_slice
                .iter()
                .zip(k_slice)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                * scale;
        }
        let probs = softmax(&logits);
        for (position, prob) in probs.iter().enumerate() {
            let v_offset = (position * num_key_value_heads + kv_head) * head_dim;
            let v_slice = &cached_values[v_offset..v_offset + head_dim];
            for i in 0..head_dim {
                output[q_offset + i] += prob * v_slice[i];
            }
        }
    }

    output
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| (v - max_value).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
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
