use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct BonsaiModelConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub architectures: Vec<String>,
    pub pad_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub model_type: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GenerationConfig {
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub begin_suppress_tokens: Vec<usize>,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub min_p: f64,
    pub presence_penalty: f64,
    pub repetition_penalty: f64,
    pub do_sample: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct AddedToken {
    pub content: String,
    pub lstrip: bool,
    pub normalized: bool,
    pub rstrip: bool,
    pub single_word: bool,
    pub special: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TokenizerConfig {
    pub add_bos_token: bool,
    pub add_prefix_space: bool,
    pub added_tokens_decoder: BTreeMap<String, AddedToken>,
    pub additional_special_tokens: Vec<String>,
    pub bos_token: Option<String>,
    pub clean_up_tokenization_spaces: bool,
    pub eos_token: String,
    pub model_max_length: usize,
    pub pad_token: String,
    pub split_special_tokens: bool,
    pub tokenizer_class: String,
    pub unk_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Architecture-derived inspection data.
///
/// Parameter and byte totals here are estimates from config-level geometry only.
/// The loader stage will compute authoritative tensor counts from safetensors files.
pub struct ModelInspection {
    pub architecture: String,
    pub estimated_parameter_count: usize,
    pub estimated_fp16_bytes: usize,
    pub kv_cache_bytes_per_token: usize,
    pub vocab_size: usize,
    pub layers: usize,
}

impl BonsaiModelConfig {
    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn estimated_parameter_count(&self) -> usize {
        let embedding = self.vocab_size * self.hidden_size;
        let q_proj = self.hidden_size * self.hidden_size;
        let kv_proj = self.hidden_size * (self.num_key_value_heads * self.head_dim);
        let o_proj = self.hidden_size * self.hidden_size;
        let gate_proj = self.hidden_size * self.intermediate_size;
        let up_proj = self.hidden_size * self.intermediate_size;
        let down_proj = self.intermediate_size * self.hidden_size;
        let norms = self.hidden_size * 2;

        let per_layer =
            q_proj + kv_proj + kv_proj + o_proj + gate_proj + up_proj + down_proj + norms;
        let final_norm = self.hidden_size;
        let lm_head = if self.tie_word_embeddings {
            0
        } else {
            self.hidden_size * self.vocab_size
        };

        embedding + (per_layer * self.num_hidden_layers) + final_norm + lm_head
    }

    pub fn approx_fp16_bytes(&self) -> usize {
        self.estimated_parameter_count() * 2
    }

    pub fn kv_cache_bytes(
        &self,
        sequence_length: usize,
        batch_size: usize,
        element_bytes: usize,
    ) -> usize {
        2 * self.num_hidden_layers
            * batch_size
            * sequence_length
            * self.num_key_value_heads
            * self.head_dim
            * element_bytes
    }

    pub fn kv_cache_bytes_per_token(&self, batch_size: usize, element_bytes: usize) -> usize {
        self.kv_cache_bytes(1, batch_size, element_bytes)
    }

    pub fn inspect(&self) -> ModelInspection {
        ModelInspection {
            architecture: self
                .architectures
                .first()
                .cloned()
                .unwrap_or_else(|| self.model_type.clone()),
            estimated_parameter_count: self.estimated_parameter_count(),
            estimated_fp16_bytes: self.approx_fp16_bytes(),
            kv_cache_bytes_per_token: self.kv_cache_bytes_per_token(1, 2),
            vocab_size: self.vocab_size,
            layers: self.num_hidden_layers,
        }
    }
}

impl GenerationConfig {
    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}

impl TokenizerConfig {
    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn special_token_count(&self) -> usize {
        self.added_tokens_decoder
            .values()
            .filter(|token| token.special)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::{BonsaiModelConfig, GenerationConfig, TokenizerConfig};

    const MODEL_CONFIG_JSON: &str = include_str!("../../fixtures/bonsai_1_7b_config.json");
    const GENERATION_CONFIG_JSON: &str =
        include_str!("../../fixtures/bonsai_1_7b_generation_config.json");
    const TOKENIZER_CONFIG_JSON: &str =
        include_str!("../../fixtures/bonsai_1_7b_tokenizer_config.json");

    #[test]
    fn parses_bonsai_model_config() {
        let config =
            BonsaiModelConfig::from_json_str(MODEL_CONFIG_JSON).expect("config should parse");

        assert_eq!(config.architectures, ["Qwen3ForCausalLM"]);
        assert_eq!(config.hidden_size, 2_048);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.rope_scaling.rope_type, "yarn");
    }

    #[test]
    fn computes_expected_parameter_and_memory_estimates() {
        let config =
            BonsaiModelConfig::from_json_str(MODEL_CONFIG_JSON).expect("config should parse");
        let inspection = config.inspect();

        assert_eq!(inspection.estimated_parameter_count, 1_720_020_992);
        assert_eq!(inspection.estimated_fp16_bytes, 3_440_041_984);
        assert_eq!(inspection.kv_cache_bytes_per_token, 114_688);
    }

    #[test]
    fn parses_generation_config() {
        let config = GenerationConfig::from_json_str(GENERATION_CONFIG_JSON)
            .expect("generation config should parse");

        assert_eq!(config.top_k, 20);
        assert_eq!(config.eos_token_id, 151_645);
        assert!(config.do_sample);
    }

    #[test]
    fn parses_tokenizer_config() {
        let config = TokenizerConfig::from_json_str(TOKENIZER_CONFIG_JSON)
            .expect("tokenizer config should parse");

        assert_eq!(config.tokenizer_class, "Qwen2Tokenizer");
        assert_eq!(config.model_max_length, 131_072);
        assert_eq!(config.special_token_count(), 4);
    }
}
