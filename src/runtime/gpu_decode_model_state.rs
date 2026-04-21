#[derive(Debug, Clone)]
pub(crate) struct HybridQProjCache {
    pub layer_idx: usize,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub code_words: Vec<u32>,
    pub scales: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct LayerTensorNames {
    pub input_layernorm_weight: String,
    pub post_attention_layernorm_weight: String,
    pub q_norm_weight: String,
    pub k_norm_weight: String,
    pub q_proj_weight: String,
    pub k_proj_weight: String,
    pub v_proj_weight: String,
    pub o_proj_weight: String,
    pub gate_proj_weight: String,
    pub up_proj_weight: String,
    pub down_proj_weight: String,
}
