use jengine::gpu::attention_block::CachedGpuAttentionBlockRunner;
use jengine::gpu::full_last_layer_block::PackedLinearSpec;
use jengine::gpu::mlp_block::CachedGpuMlpBlockRunner;
use jengine::gpu::packed_matvec::SharedGpuPackedContext;
use jengine::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seq_len = 64usize;
    let num_query_heads = 8usize;
    let num_key_value_heads = 4usize;
    let head_dim = 64usize;
    let hidden = num_query_heads * head_dim;
    let intermediate = hidden * 3;
    let epsilon = 1e-5f32;

    let query: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0021).sin() * 0.5 + ((i as f32) * 0.0017).cos() * 0.25)
        .collect();
    let kv_len = seq_len * num_key_value_heads * head_dim;
    let keys: Vec<f32> = (0..kv_len)
        .map(|i| ((i as f32) * 0.0013).sin() * 0.45 - ((i as f32) * 0.0031).cos() * 0.15)
        .collect();
    let values: Vec<f32> = (0..kv_len)
        .map(|i| ((i as f32) * 0.0019).cos() * 0.35 + ((i as f32) * 0.0027).sin() * 0.1)
        .collect();
    let attention_residual: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0011).sin() * 0.4 - ((i as f32) * 0.0023).cos() * 0.2)
        .collect();
    let post_norm_weight: Vec<f32> = (0..hidden).map(|i| 0.75 + (i % 17) as f32 * 0.01).collect();

    let o_proj_dense: Vec<f32> = (0..hidden * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 11, 17))
        .collect();
    let gate: Vec<f32> = (0..intermediate * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 17, 31))
        .collect();
    let up: Vec<f32> = (0..intermediate * hidden)
        .map(|idx| ternary_value(idx / hidden, idx % hidden, 29, 11))
        .collect();
    let down: Vec<f32> = (0..hidden * intermediate)
        .map(|idx| ternary_value(idx / intermediate, idx % intermediate, 13, 19))
        .collect();

    let (o_proj_packed, _) = pack_ternary_g128(&o_proj_dense, vec![hidden, hidden], 0.0)?;
    let mut pair_concat = gate.clone();
    pair_concat.extend_from_slice(&up);
    let (pair_packed, _) = pack_ternary_g128(&pair_concat, vec![intermediate * 2, hidden], 0.0)?;
    let (down_packed, _) = pack_ternary_g128(&down, vec![hidden, intermediate], 0.0)?;

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
    let post_attention_hidden: Vec<f32> = o_proj_ref
        .iter()
        .zip(attention_residual.iter())
        .map(|(a, b)| a + b)
        .collect();
    let expected_mlp_hidden = mlp_reference(
        &post_attention_hidden,
        &post_norm_weight,
        &pair_packed,
        &down_packed,
        epsilon,
    )?;

    let o_proj_spec = to_spec(&o_proj_packed);
    let pair_spec = to_spec(&pair_packed);
    let down_spec = to_spec(&down_packed);
    let context = SharedGpuPackedContext::new()?;
    let mut attention_block = CachedGpuAttentionBlockRunner::new_with_context(
        context.clone(),
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        &o_proj_spec,
    )?;
    let _ = attention_block.run_resident(&query, &keys, &values, &attention_residual)?;
    let (attention_output, _) = attention_block.read_output()?;
    let (attention_max_abs_diff, attention_mean_abs_diff) =
        compare_outputs(&post_attention_hidden, &attention_output);

    let mut mlp_block = CachedGpuMlpBlockRunner::new_with_context(
        context,
        hidden,
        intermediate,
        epsilon,
        &pair_spec,
        &down_spec,
    )?;
    let _ = mlp_block.run_with_host_residual(&attention_output, &post_norm_weight)?;
    let (host_output, _) = mlp_block.read_output()?;
    let (host_max_abs_diff, host_mean_abs_diff) =
        compare_outputs(&expected_mlp_hidden, &host_output);
    let _ = mlp_block.run_from_resident_residual(
        attention_block.shared_context(),
        attention_block.output_buffer_handle(),
        attention_block.hidden(),
        attention_block.output_buffer_size(),
        &post_norm_weight,
    )?;
    let (resident_output, _) = mlp_block.read_output()?;
    let (resident_max_abs_diff, resident_mean_abs_diff) =
        compare_outputs(&expected_mlp_hidden, &resident_output);

    println!(
        "attention_max_abs_diff={:.6} attention_mean_abs_diff={:.6} host_mlp_max_abs_diff={:.6} host_mlp_mean_abs_diff={:.6} resident_mlp_max_abs_diff={:.6} resident_mlp_mean_abs_diff={:.6}",
        attention_max_abs_diff,
        attention_mean_abs_diff,
        host_max_abs_diff,
        host_mean_abs_diff,
        resident_max_abs_diff,
        resident_mean_abs_diff,
    );

    Ok(())
}

fn to_spec(packed: &jengine::runtime::repack::PackedTernaryTensor) -> PackedLinearSpec {
    PackedLinearSpec {
        code_words: packed_codes_to_words(&packed.packed_codes),
        scales: packed.scales.clone(),
        group_size: packed.group_size,
        rows: packed.shape[0],
        cols: packed.shape[1],
    }
}

fn mlp_reference(
    post_attention_hidden: &[f32],
    post_norm_weight: &[f32],
    pair_packed: &jengine::runtime::repack::PackedTernaryTensor,
    down_packed: &jengine::runtime::repack::PackedTernaryTensor,
    epsilon: f32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let intermediate = pair_packed.shape[0] / 2;
    let post_norm = weighted_rms_norm(post_attention_hidden, post_norm_weight, epsilon);
    let pair = matvec_packed_ternary(pair_packed, &post_norm)?;
    let swiglu = quantize_f16(&swiglu(&pair[..intermediate], &pair[intermediate..]));
    let down = matvec_packed_ternary(down_packed, &swiglu)?;
    Ok(down
        .iter()
        .zip(post_attention_hidden.iter())
        .map(|(a, b)| a + b)
        .collect())
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
    values
        .iter()
        .map(|v| half::f16::from_f32(*v).to_f32())
        .collect()
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

fn softmax(values: &[f32]) -> Vec<f32> {
    let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| (v - max_value).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
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

fn ternary_value(row: usize, col: usize, a: usize, b: usize) -> f32 {
    match (row.wrapping_mul(a) + col.wrapping_mul(b)) % 3 {
        0 => -1.0,
        1 => 0.0,
        _ => 1.0,
    }
}
