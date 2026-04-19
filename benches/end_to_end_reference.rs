use criterion::{Criterion, black_box, criterion_group, criterion_main};
use half::f16;
use jengine::runtime::reference::ReferenceModel;
use safetensors::tensor::{Dtype, TensorView, serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use tempfile::{TempDir, tempdir};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

fn encode(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_le_bytes())
        .collect()
}

fn write_synthetic_model(root: &std::path::Path) {
    fs::write(
        root.join("config.json"),
        r#"{"vocab_size":4,"max_position_embeddings":32,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000.0,"rope_scaling":{"rope_type":"yarn","factor":1.0,"original_max_position_embeddings":32},"attention_bias":false,"tie_word_embeddings":true,"architectures":["Qwen3ForCausalLM"],"pad_token_id":0,"eos_token_id":3,"model_type":"qwen3"}"#,
    ).unwrap();
    fs::write(
        root.join("generation_config.json"),
        r#"{"eos_token_id":3,"pad_token_id":0,"begin_suppress_tokens":[],"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"do_sample":false}"#,
    ).unwrap();
    fs::write(
        root.join("tokenizer_config.json"),
        r#"{"add_bos_token":false,"add_prefix_space":false,"added_tokens_decoder":{},"additional_special_tokens":[],"bos_token":null,"clean_up_tokenization_spaces":false,"eos_token":"tok3","model_max_length":32,"pad_token":"tok0","split_special_tokens":false,"tokenizer_class":"WordLevel","unk_token":"[UNK]"}"#,
    ).unwrap();
    let vocab = HashMap::from([
        ("[UNK]".to_string(), 0),
        ("tok1".to_string(), 1),
        ("tok2".to_string(), 2),
        ("tok3".to_string(), 3),
    ]);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Whitespace);
    tokenizer.save(root.join("tokenizer.json"), false).unwrap();

    let embed = encode(&[
        2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
    ]);
    let ones4 = encode(&[1.0, 1.0, 1.0, 1.0]);
    let ones2 = encode(&[1.0, 1.0]);
    let zeros_4x4 = encode(&[0.0; 16]);
    let zeros_2x4 = encode(&[0.0; 8]);
    let zeros_8x4 = encode(&[0.0; 32]);
    let zeros_4x8 = encode(&[0.0; 32]);
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        TensorView::new(Dtype::F16, vec![4, 4], &embed).unwrap(),
    );
    tensors.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        TensorView::new(Dtype::F16, vec![4], &ones4).unwrap(),
    );
    tensors.insert(
        "model.layers.0.post_attention_layernorm.weight".to_string(),
        TensorView::new(Dtype::F16, vec![4], &ones4).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_norm.weight".to_string(),
        TensorView::new(Dtype::F16, vec![2], &ones2).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_norm.weight".to_string(),
        TensorView::new(Dtype::F16, vec![2], &ones2).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        TensorView::new(Dtype::F16, vec![4, 4], &zeros_4x4).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        TensorView::new(Dtype::F16, vec![2, 4], &zeros_2x4).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        TensorView::new(Dtype::F16, vec![2, 4], &zeros_2x4).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        TensorView::new(Dtype::F16, vec![4, 4], &zeros_4x4).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        TensorView::new(Dtype::F16, vec![8, 4], &zeros_8x4).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        TensorView::new(Dtype::F16, vec![8, 4], &zeros_8x4).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        TensorView::new(Dtype::F16, vec![4, 8], &zeros_4x8).unwrap(),
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        TensorView::new(Dtype::F16, vec![4], &ones4).unwrap(),
    );
    fs::write(
        root.join("model.safetensors"),
        serialize(tensors, &None).unwrap(),
    )
    .unwrap();
}

fn build_fixture() -> TempDir {
    let dir = tempdir().unwrap();
    write_synthetic_model(dir.path());
    dir
}

fn bench_reference_decode(c: &mut Criterion) {
    let fixture = build_fixture();
    let model = ReferenceModel::load_from_root(fixture.path()).unwrap();
    c.bench_function("reference/one_token_decode_synthetic", |b| {
        b.iter(|| {
            model
                .generate_greedy(black_box("tok2"), black_box(1))
                .unwrap()
        })
    });
}

criterion_group!(reference_benches, bench_reference_decode);
criterion_main!(reference_benches);
