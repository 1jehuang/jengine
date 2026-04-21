#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedDecodePlan {
    pub use_attention_qkv: bool,
    pub use_mlp_gu: bool,
    pub argmax_only: bool,
    pub use_attention_full: bool,
    pub use_mlp_full: bool,
    pub gpu_final_norm: bool,
    pub gpu_swiglu_block: bool,
    pub gpu_attention_block: bool,
    pub gpu_embedding: bool,
    pub gpu_mlp_entry: bool,
    pub gpu_full_last_layer: bool,
    pub gpu_tail: bool,
    pub gpu_first_session: bool,
}

impl PackedDecodePlan {
    pub fn from_env(use_attention_qkv: bool, use_mlp_gu: bool, argmax_only: bool) -> Self {
        let use_attention_full = use_attention_qkv && std::env::var_os("JENGINE_PACKED_ATTENTION_FULL").is_some();
        let use_mlp_full = use_mlp_gu && std::env::var_os("JENGINE_PACKED_MLP_FULL").is_some();
        let gpu_final_norm = std::env::var_os("JENGINE_GPU_FINAL_NORM").is_some();
        let gpu_swiglu_block = std::env::var_os("JENGINE_GPU_SWIGLU_BLOCK").is_some();
        let gpu_attention_block = std::env::var_os("JENGINE_GPU_ATTENTION_BLOCK").is_some();
        let gpu_embedding = std::env::var_os("JENGINE_GPU_EMBEDDING").is_some();
        let gpu_mlp_entry = std::env::var_os("JENGINE_GPU_MLP_ENTRY").is_some();
        let gpu_full_last_layer = std::env::var_os("JENGINE_GPU_FULL_LAST_LAYER").is_some();
        let gpu_tail = std::env::var_os("JENGINE_GPU_TAIL").is_some();
        let gpu_first_session = gpu_attention_block
            || gpu_embedding
            || gpu_swiglu_block
            || gpu_full_last_layer
            || gpu_tail;
        Self {
            use_attention_qkv,
            use_mlp_gu,
            argmax_only,
            use_attention_full,
            use_mlp_full,
            gpu_final_norm,
            gpu_swiglu_block,
            gpu_attention_block,
            gpu_embedding,
            gpu_mlp_entry,
            gpu_full_last_layer,
            gpu_tail,
            gpu_first_session,
        }
    }
}
