use crate::model::config::BonsaiModelConfig;
use std::f32::consts::PI;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub struct YarnRope {
    pub inv_freq: Vec<f32>,
    pub attention_factor: f32,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionDims {
    pub seq_len: usize,
    pub num_query_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionContext {
    pub output: Vec<f32>,
    pub attn_weights: Vec<f32>,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockProfile {
    pub rope_duration: Duration,
    pub qk_norm_duration: Duration,
    pub attention_duration: Duration,
    pub swiglu_duration: Duration,
    pub scratch_bytes: usize,
}

impl BlockProfile {
    pub fn summarize(&self) -> String {
        format!(
            "rope_ms={:.3} qk_norm_ms={:.3} attention_ms={:.3} swiglu_ms={:.3} scratch_bytes={}",
            self.rope_duration.as_secs_f64() * 1_000.0,
            self.qk_norm_duration.as_secs_f64() * 1_000.0,
            self.attention_duration.as_secs_f64() * 1_000.0,
            self.swiglu_duration.as_secs_f64() * 1_000.0,
            self.scratch_bytes,
        )
    }
}

pub fn rms_norm_in_place(values: &mut [f32], epsilon: f32) {
    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let scale = 1.0 / (mean_square + epsilon).sqrt();
    for value in values {
        *value *= scale;
    }
}

pub fn qk_head_rms_norm_in_place(
    query: &mut [f32],
    key: &mut [f32],
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    epsilon: f32,
) {
    assert_eq!(query.len(), seq_len * num_query_heads * head_dim);
    assert_eq!(key.len(), seq_len * num_key_value_heads * head_dim);

    for chunk in query.chunks_exact_mut(head_dim) {
        rms_norm_in_place(chunk, epsilon);
    }
    for chunk in key.chunks_exact_mut(head_dim) {
        rms_norm_in_place(chunk, epsilon);
    }
}

pub fn build_yarn_rope(config: &BonsaiModelConfig) -> YarnRope {
    let dim = config.head_dim;
    let base = config.rope_theta as f32;
    let factor = config.rope_scaling.factor as f32;
    let original_max_position_embeddings =
        config.rope_scaling.original_max_position_embeddings as f32;

    let get_mscale = |scale: f32, mscale: f32| {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * mscale * scale.ln() + 1.0
        }
    };

    let attention_factor = get_mscale(factor, 1.0);
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;

    let find_correction_dim = |num_rotations: f32| {
        (dim as f32 * (original_max_position_embeddings / (num_rotations * 2.0 * PI)).ln())
            / (2.0 * base.ln())
    };
    let mut low = find_correction_dim(beta_fast).floor();
    let mut high = find_correction_dim(beta_slow).ceil();
    low = low.max(0.0);
    high = high.min((dim - 1) as f32);

    let half_dim = dim / 2;
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let exponent = (2 * i) as f32 / dim as f32;
        let pos_freq = base.powf(exponent);
        let inv_freq_extrapolation = 1.0 / pos_freq;
        let inv_freq_interpolation = 1.0 / (factor * pos_freq);
        let ramp = if (high - low).abs() < f32::EPSILON {
            1.0
        } else {
            (((i as f32) - low) / (high - low)).clamp(0.0, 1.0)
        };
        let extrapolation_factor = 1.0 - ramp;
        let value = inv_freq_interpolation * (1.0 - extrapolation_factor)
            + inv_freq_extrapolation * extrapolation_factor;
        inv_freq.push(value);
    }

    YarnRope {
        inv_freq,
        attention_factor,
        head_dim: dim,
    }
}

pub fn rope_cos_sin(rope: &YarnRope, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(positions.len() * rope.head_dim);
    let mut sin = Vec::with_capacity(positions.len() * rope.head_dim);

    for &position in positions {
        for &inv_freq in &rope.inv_freq {
            let freq = inv_freq * position as f32;
            cos.push(freq.cos() * rope.attention_factor);
            sin.push(freq.sin() * rope.attention_factor);
        }
        let tail_start = cos.len() - rope.inv_freq.len();
        let cos_tail = cos[tail_start..].to_vec();
        let sin_tail = sin[tail_start..].to_vec();
        cos.extend_from_slice(&cos_tail);
        sin.extend_from_slice(&sin_tail);
    }

    (cos, sin)
}

fn rotate_half(x: &[f32]) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = Vec::with_capacity(x.len());
    out.extend(x[half..].iter().map(|value| -*value));
    out.extend_from_slice(&x[..half]);
    out
}

pub fn apply_rotary_pos_emb_in_place(
    query: &mut [f32],
    key: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    dims: AttentionDims,
) {
    assert_eq!(cos.len(), dims.seq_len * dims.head_dim);
    assert_eq!(sin.len(), dims.seq_len * dims.head_dim);
    for position in 0..dims.seq_len {
        let rope_offset = position * dims.head_dim;
        let cos_slice = &cos[rope_offset..rope_offset + dims.head_dim];
        let sin_slice = &sin[rope_offset..rope_offset + dims.head_dim];

        for head in 0..dims.num_query_heads {
            let offset = (position * dims.num_query_heads + head) * dims.head_dim;
            let rotated = rotate_half(&query[offset..offset + dims.head_dim]);
            for i in 0..dims.head_dim {
                query[offset + i] = query[offset + i] * cos_slice[i] + rotated[i] * sin_slice[i];
            }
        }

        for head in 0..dims.num_key_value_heads {
            let offset = (position * dims.num_key_value_heads + head) * dims.head_dim;
            let rotated = rotate_half(&key[offset..offset + dims.head_dim]);
            for i in 0..dims.head_dim {
                key[offset + i] = key[offset + i] * cos_slice[i] + rotated[i] * sin_slice[i];
            }
        }
    }
}

pub fn repeat_kv(
    hidden_states: &[f32],
    seq_len: usize,
    num_key_value_heads: usize,
    n_rep: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(
        hidden_states.len(),
        seq_len * num_key_value_heads * head_dim
    );
    if n_rep == 1 {
        return hidden_states.to_vec();
    }

    let mut out = Vec::with_capacity(seq_len * num_key_value_heads * n_rep * head_dim);
    for position in 0..seq_len {
        for kv_head in 0..num_key_value_heads {
            let offset = (position * num_key_value_heads + kv_head) * head_dim;
            let slice = &hidden_states[offset..offset + head_dim];
            for _ in 0..n_rep {
                out.extend_from_slice(slice);
            }
        }
    }
    out
}

pub fn causal_gqa_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> AttentionContext {
    assert_eq!(query.len(), seq_len * num_query_heads * head_dim);
    assert_eq!(key.len(), seq_len * num_key_value_heads * head_dim);
    assert_eq!(value.len(), seq_len * num_key_value_heads * head_dim);

    let n_rep = num_query_heads / num_key_value_heads;
    let repeated_key = repeat_kv(key, seq_len, num_key_value_heads, n_rep, head_dim);
    let repeated_value = repeat_kv(value, seq_len, num_key_value_heads, n_rep, head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0; seq_len * num_query_heads * head_dim];
    let mut attn_weights = vec![0.0; seq_len * num_query_heads * seq_len];

    for position in 0..seq_len {
        for head in 0..num_query_heads {
            let q_offset = (position * num_query_heads + head) * head_dim;
            let q_slice = &query[q_offset..q_offset + head_dim];

            let mut logits = vec![f32::NEG_INFINITY; seq_len];
            for (key_pos, logit) in logits.iter_mut().enumerate().take(position + 1) {
                let k_offset = (key_pos * num_query_heads + head) * head_dim;
                let k_slice = &repeated_key[k_offset..k_offset + head_dim];
                *logit = crate::cpu::primitives::dot(q_slice, k_slice) * scale;
            }

            let valid_logits = &logits[..=position];
            let probs = crate::cpu::primitives::softmax(valid_logits);
            for (key_pos, prob) in probs.iter().enumerate() {
                attn_weights[(position * num_query_heads + head) * seq_len + key_pos] = *prob;
                let v_offset = (key_pos * num_query_heads + head) * head_dim;
                let v_slice = &repeated_value[v_offset..v_offset + head_dim];
                let out_offset = (position * num_query_heads + head) * head_dim;
                for i in 0..head_dim {
                    output[out_offset + i] += prob * v_slice[i];
                }
            }
        }
    }

    AttentionContext {
        output,
        attn_weights,
        seq_len,
        num_heads: num_query_heads,
        head_dim,
    }
}

pub fn profile_block_components(config: &BonsaiModelConfig, seq_len: usize) -> BlockProfile {
    let num_query_heads = config.num_attention_heads;
    let num_key_value_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let query_len = seq_len * num_query_heads * head_dim;
    let key_len = seq_len * num_key_value_heads * head_dim;

    let mut query: Vec<f32> = (0..query_len)
        .map(|index| ((index % 19) as f32 - 9.0) * 0.05)
        .collect();
    let mut key: Vec<f32> = (0..key_len)
        .map(|index| ((index % 23) as f32 - 11.0) * 0.04)
        .collect();
    let value: Vec<f32> = (0..key_len)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.03)
        .collect();
    let positions: Vec<usize> = (0..seq_len).collect();

    let started_at = Instant::now();
    let rope = build_yarn_rope(config);
    let (cos, sin) = rope_cos_sin(&rope, &positions);
    apply_rotary_pos_emb_in_place(
        &mut query,
        &mut key,
        &cos,
        &sin,
        AttentionDims {
            seq_len,
            num_query_heads,
            num_key_value_heads,
            head_dim,
        },
    );
    let rope_duration = started_at.elapsed();

    let started_at = Instant::now();
    qk_head_rms_norm_in_place(
        &mut query,
        &mut key,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        config.rms_norm_eps as f32,
    );
    let qk_norm_duration = started_at.elapsed();

    let started_at = Instant::now();
    let attention = causal_gqa_attention(
        &query,
        &key,
        &value,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
    );
    let attention_duration = started_at.elapsed();

    let gate = attention.output.clone();
    let up = attention.output.clone();
    let started_at = Instant::now();
    let _mlp = crate::cpu::primitives::swiglu(&gate, &up);
    let swiglu_duration = started_at.elapsed();

    let scratch_bytes =
        (query.len() + key.len() + value.len() + cos.len() + sin.len() + attention.output.len())
            * std::mem::size_of::<f32>();

    BlockProfile {
        rope_duration,
        qk_norm_duration,
        attention_duration,
        swiglu_duration,
        scratch_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionDims, apply_rotary_pos_emb_in_place, build_yarn_rope, causal_gqa_attention,
        profile_block_components, qk_head_rms_norm_in_place, repeat_kv, rope_cos_sin,
    };
    use crate::model::config::BonsaiModelConfig;

    const MODEL_CONFIG_JSON: &str = include_str!("../../fixtures/bonsai_1_7b_config.json");

    fn config() -> BonsaiModelConfig {
        BonsaiModelConfig::from_json_str(MODEL_CONFIG_JSON).expect("config should parse")
    }

    #[test]
    fn builds_yarn_rope_and_cos_sin_tables() {
        let rope = build_yarn_rope(&config());
        let (cos, sin) = rope_cos_sin(&rope, &[0, 1, 2]);

        assert_eq!(rope.inv_freq.len(), config().head_dim / 2);
        assert!(rope.attention_factor >= 1.0);
        assert_eq!(cos.len(), 3 * config().head_dim);
        assert_eq!(sin.len(), 3 * config().head_dim);
    }

    #[test]
    fn repeats_kv_heads_for_grouped_query_attention() {
        let hidden_states = vec![1.0, 2.0, 3.0, 4.0];
        let repeated = repeat_kv(&hidden_states, 2, 2, 2, 1);
        assert_eq!(repeated, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn applies_rotary_embeddings_and_qk_norm() {
        let mut query = vec![1.0, 0.0, 0.0, 1.0];
        let mut key = vec![0.0, 1.0, 1.0, 0.0];
        let cos = vec![1.0, 1.0, 0.0, 0.0];
        let sin = vec![0.0, 0.0, 1.0, 1.0];

        apply_rotary_pos_emb_in_place(
            &mut query,
            &mut key,
            &cos,
            &sin,
            AttentionDims {
                seq_len: 2,
                num_query_heads: 1,
                num_key_value_heads: 1,
                head_dim: 2,
            },
        );
        qk_head_rms_norm_in_place(&mut query, &mut key, 2, 1, 1, 2, 1e-6);

        assert_eq!(query.len(), 4);
        assert_eq!(key.len(), 4);
        let q0_norm = (query[0] * query[0] + query[1] * query[1]) / 2.0;
        assert!((q0_norm - 1.0).abs() < 1e-3);
    }

    #[test]
    fn computes_causal_grouped_attention() {
        let query = vec![1.0, 0.0, 0.0, 1.0];
        let key = vec![1.0, 0.0, 0.0, 1.0];
        let value = vec![1.0, 2.0, 3.0, 4.0];
        let attention = causal_gqa_attention(&query, &key, &value, 2, 1, 1, 2);

        assert_eq!(attention.output.len(), 4);
        assert_eq!(attention.attn_weights.len(), 4);
        assert!((attention.attn_weights[0] - 1.0).abs() < 1e-6);
        assert_eq!(attention.output[0], 1.0);
        assert_eq!(attention.output[1], 2.0);
    }

    #[test]
    fn profiles_block_components_with_live_metrics() {
        let profile = profile_block_components(&config(), 8);
        let summary = profile.summarize();

        assert!(profile.attention_duration > Duration::ZERO);
        assert!(profile.scratch_bytes > 0);
        assert!(summary.contains("attention_ms="));
        assert!(summary.contains("scratch_bytes="));
    }

    use std::time::Duration;
}
