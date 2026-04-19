use jengine::model::config::BonsaiModelConfig;
use jengine::runtime::weights::WeightStore;

const MODEL_CONFIG_JSON: &str = include_str!("../../fixtures/bonsai_1_7b_config.json");

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let config = BonsaiModelConfig::from_json_str(MODEL_CONFIG_JSON).expect("config should parse");
    let report =
        validate_expected_layout(&path, &config).expect("layout validation should succeed");
    println!("{}", report);
}

fn expected_tensor_names(config: &BonsaiModelConfig) -> Vec<String> {
    let mut names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
    ];
    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer_idx}");
        names.push(format!("{prefix}.input_layernorm.weight"));
        names.push(format!("{prefix}.post_attention_layernorm.weight"));
        names.push(format!("{prefix}.self_attn.q_norm.weight"));
        names.push(format!("{prefix}.self_attn.k_norm.weight"));
        names.push(format!("{prefix}.self_attn.q_proj.weight"));
        names.push(format!("{prefix}.self_attn.k_proj.weight"));
        names.push(format!("{prefix}.self_attn.v_proj.weight"));
        names.push(format!("{prefix}.self_attn.o_proj.weight"));
        names.push(format!("{prefix}.mlp.gate_proj.weight"));
        names.push(format!("{prefix}.mlp.up_proj.weight"));
        names.push(format!("{prefix}.mlp.down_proj.weight"));
    }
    names
}

fn validate_expected_layout(
    path: &str,
    config: &BonsaiModelConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    let progress = WeightStore::download_progress(path)?;
    let bytes = std::fs::read(path)?;
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("header length")) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + header_len])?;
    let names = header
        .keys()
        .filter(|name| *name != "__metadata__")
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let expected = expected_tensor_names(config);
    let missing = expected
        .iter()
        .filter(|name| !names.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    Ok(format!(
        "expected_tensors={} header_tensors={} missing_expected={} available_tensors={} next_missing_tensor={}",
        expected.len(),
        names.len(),
        missing.len(),
        progress.available_tensors,
        progress
            .next_missing_tensor
            .unwrap_or_else(|| "<none>".to_string())
    ))
}
