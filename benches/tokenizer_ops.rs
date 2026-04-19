use criterion::{Criterion, black_box, criterion_group, criterion_main};
use jengine::model::tokenizer::TokenizerRuntime;
use std::collections::HashMap;
use tempfile::NamedTempFile;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

fn make_tokenizer_fixture() -> NamedTempFile {
    let vocab = HashMap::from([
        ("[UNK]".to_string(), 0),
        ("hello".to_string(), 1),
        ("world".to_string(), 2),
        ("from".to_string(), 3),
        ("jengine".to_string(), 4),
    ]);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .expect("word-level model should build");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Whitespace);

    let temp = NamedTempFile::new().expect("temp file should be created");
    tokenizer
        .save(temp.path(), false)
        .expect("tokenizer should be written");
    temp
}

fn bench_tokenizer_load(c: &mut Criterion) {
    let fixture = make_tokenizer_fixture();
    c.bench_function("tokenizer/load", |b| {
        b.iter(|| TokenizerRuntime::load_from_file(fixture.path()).expect("load should succeed"))
    });
}

fn bench_tokenizer_encode_decode(c: &mut Criterion) {
    let fixture = make_tokenizer_fixture();
    let runtime = TokenizerRuntime::load_from_file(fixture.path()).expect("load should succeed");
    let prompt = "hello world from jengine";

    c.bench_function("tokenizer/encode", |b| {
        b.iter(|| {
            runtime
                .encode(black_box(prompt))
                .expect("encode should succeed")
        })
    });

    let ids = runtime.encode(prompt).expect("encode should succeed");
    c.bench_function("tokenizer/decode", |b| {
        b.iter(|| {
            runtime
                .decode(black_box(&ids))
                .expect("decode should succeed")
        })
    });

    c.bench_function("tokenizer/analyze_prompt", |b| {
        b.iter(|| {
            runtime
                .analyze_prompt(black_box(prompt))
                .expect("analysis should succeed")
        })
    });
}

criterion_group!(
    tokenizer_benches,
    bench_tokenizer_load,
    bench_tokenizer_encode_decode
);
criterion_main!(tokenizer_benches);
