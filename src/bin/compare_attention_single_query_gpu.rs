use jengine::gpu::attention_single_query::run_attention_single_query_with_output;

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

    let query_len = num_query_heads * head_dim;
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

    let reference = attention_single_query_cpu(
        &query,
        &keys,
        &values,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
    );
    let (_output, report) = run_attention_single_query_with_output(
        &query,
        &keys,
        &values,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        Some(&reference),
    )?;

    println!(
        "seq_len={} q_heads={} kv_heads={} head_dim={} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        report.compile_duration.as_secs_f64() * 1_000.0,
        report.upload_duration.as_secs_f64() * 1_000.0,
        report.gpu_duration.as_secs_f64() * 1_000.0,
        report.download_duration.as_secs_f64() * 1_000.0,
        report.max_abs_diff,
        report.mean_abs_diff,
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
