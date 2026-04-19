use criterion::{Criterion, black_box, criterion_group, criterion_main};
use jengine::model::config::{BonsaiModelConfig, GenerationConfig, TokenizerConfig};

const MODEL_CONFIG_JSON: &str = include_str!("../fixtures/bonsai_1_7b_config.json");
const GENERATION_CONFIG_JSON: &str = include_str!("../fixtures/bonsai_1_7b_generation_config.json");
const TOKENIZER_CONFIG_JSON: &str = include_str!("../fixtures/bonsai_1_7b_tokenizer_config.json");

fn bench_model_config_parse(c: &mut Criterion) {
    c.bench_function("config_parse/model", |b| {
        b.iter(|| {
            BonsaiModelConfig::from_json_str(black_box(MODEL_CONFIG_JSON))
                .expect("model config parse should succeed")
        })
    });
}

fn bench_generation_config_parse(c: &mut Criterion) {
    c.bench_function("config_parse/generation", |b| {
        b.iter(|| {
            GenerationConfig::from_json_str(black_box(GENERATION_CONFIG_JSON))
                .expect("generation config parse should succeed")
        })
    });
}

fn bench_tokenizer_config_parse(c: &mut Criterion) {
    c.bench_function("config_parse/tokenizer", |b| {
        b.iter(|| {
            TokenizerConfig::from_json_str(black_box(TOKENIZER_CONFIG_JSON))
                .expect("tokenizer config parse should succeed")
        })
    });
}

fn bench_model_inspection(c: &mut Criterion) {
    let config = BonsaiModelConfig::from_json_str(MODEL_CONFIG_JSON)
        .expect("model config parse should succeed");

    c.bench_function("config_parse/inspection", |b| {
        b.iter(|| black_box(config.inspect()))
    });
}

criterion_group!(
    config_benches,
    bench_model_config_parse,
    bench_generation_config_parse,
    bench_tokenizer_config_parse,
    bench_model_inspection
);
criterion_main!(config_benches);
