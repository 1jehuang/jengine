use jengine::cpu::block::profile_block_components;
use jengine::model::config::BonsaiModelConfig;

const MODEL_CONFIG_JSON: &str = include_str!("../../fixtures/bonsai_1_7b_config.json");

fn main() {
    let seq_len = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);

    let config = BonsaiModelConfig::from_json_str(MODEL_CONFIG_JSON).expect("config should parse");
    let profile = profile_block_components(&config, seq_len);
    println!("seq_len={seq_len} {}", profile.summarize());
}
