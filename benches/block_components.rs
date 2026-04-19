use criterion::{Criterion, black_box, criterion_group, criterion_main};
use jengine::cpu::block::{
    AttentionDims, apply_rotary_pos_emb_in_place, build_yarn_rope, causal_gqa_attention,
    profile_block_components, qk_head_rms_norm_in_place, rope_cos_sin,
};
use jengine::model::config::BonsaiModelConfig;

const MODEL_CONFIG_JSON: &str = include_str!("../fixtures/bonsai_1_7b_config.json");

fn config() -> BonsaiModelConfig {
    BonsaiModelConfig::from_json_str(MODEL_CONFIG_JSON).expect("config should parse")
}

fn bench_rope(c: &mut Criterion) {
    let config = config();
    let rope = build_yarn_rope(&config);
    let positions: Vec<usize> = (0..128).collect();
    let query = vec![0.1f32; 128 * config.num_attention_heads * config.head_dim];
    let key = vec![0.2f32; 128 * config.num_key_value_heads * config.head_dim];
    let (cos, sin) = rope_cos_sin(&rope, &positions);

    c.bench_function("block/rope_apply_seq128", |b| {
        b.iter(|| {
            let mut q = query.clone();
            let mut k = key.clone();
            apply_rotary_pos_emb_in_place(
                black_box(&mut q),
                black_box(&mut k),
                black_box(&cos),
                black_box(&sin),
                AttentionDims {
                    seq_len: 128,
                    num_query_heads: config.num_attention_heads,
                    num_key_value_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                },
            )
        })
    });

    c.bench_function("block/qk_head_rms_norm_seq128", |b| {
        b.iter(|| {
            let mut q = query.clone();
            let mut k = key.clone();
            qk_head_rms_norm_in_place(
                black_box(&mut q),
                black_box(&mut k),
                128,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                black_box(config.rms_norm_eps as f32),
            )
        })
    });
}

fn bench_attention(c: &mut Criterion) {
    let config = config();
    let seq_len = 32;
    let query = vec![0.1f32; seq_len * config.num_attention_heads * config.head_dim];
    let key = vec![0.2f32; seq_len * config.num_key_value_heads * config.head_dim];
    let value = vec![0.3f32; seq_len * config.num_key_value_heads * config.head_dim];

    c.bench_function("block/gqa_attention_seq32", |b| {
        b.iter(|| {
            causal_gqa_attention(
                black_box(&query),
                black_box(&key),
                black_box(&value),
                seq_len,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )
        })
    });
}

fn bench_profile(c: &mut Criterion) {
    let config = config();
    c.bench_function("block/profile_seq16", |b| {
        b.iter(|| profile_block_components(black_box(&config), black_box(16)))
    });
}

criterion_group!(block_benches, bench_rope, bench_attention, bench_profile);
criterion_main!(block_benches);
