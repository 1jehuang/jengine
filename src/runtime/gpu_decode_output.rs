use crate::runtime::gpu_decode_metrics::{DecodeMetrics, PackedDecodeMetrics};

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PackedDispatchTrace {
    pub index: usize,
    pub operation: String,
    pub path: String,
    pub stage: String,
    pub tensor_name: String,
    pub rows: usize,
    pub cols: usize,
    pub pack_cache_hit: bool,
    pub gpu_cache_hit: bool,
    pub cpu_ms: f64,
    pub compile_ms: f64,
    pub weight_upload_ms: f64,
    pub activation_upload_ms: f64,
    pub gpu_ms: f64,
    pub download_ms: f64,
    pub weight_upload_bytes: usize,
    pub activation_upload_bytes: usize,
    pub download_bytes: usize,
}

impl PackedDispatchTrace {
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_packed(
        index: usize,
        tensor_name: &str,
        operation: &str,
        rows: usize,
        cols: usize,
        pack_cache_hit: bool,
        gpu_cache_hit: bool,
        compile_duration: std::time::Duration,
        weight_upload_duration: std::time::Duration,
        activation_upload_duration: std::time::Duration,
        gpu_duration: std::time::Duration,
        download_duration: std::time::Duration,
        weight_upload_bytes: usize,
        activation_upload_bytes: usize,
        download_bytes: usize,
    ) -> Self {
        let stage = if tensor_name == "model.embed_tokens.weight" {
            if operation == "argmax" {
                "logits_argmax".to_string()
            } else {
                "logits".to_string()
            }
        } else if tensor_name.contains("self_attn.o_proj") {
            "attention_oproj".to_string()
        } else if tensor_name.contains("self_attn")
            || tensor_name.contains("concat::model.layers") && tensor_name.contains("self_attn")
        {
            "attention_qkv".to_string()
        } else if tensor_name.contains("mlp.down_proj") {
            "mlp_down".to_string()
        } else if tensor_name.contains("mlp") {
            "mlp_gu".to_string()
        } else {
            "packed_dispatch".to_string()
        };
        Self {
            index,
            operation: operation.to_string(),
            path: "gpu_packed".to_string(),
            stage,
            tensor_name: tensor_name.to_string(),
            rows,
            cols,
            pack_cache_hit,
            gpu_cache_hit,
            cpu_ms: (compile_duration + weight_upload_duration + activation_upload_duration)
                .as_secs_f64()
                * 1_000.0,
            compile_ms: compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: weight_upload_duration.as_secs_f64() * 1_000.0,
            activation_upload_ms: activation_upload_duration.as_secs_f64() * 1_000.0,
            gpu_ms: gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: download_duration.as_secs_f64() * 1_000.0,
            weight_upload_bytes,
            activation_upload_bytes,
            download_bytes,
        }
    }

    pub fn dense_stage(
        index: usize,
        stage: &str,
        tensor_name: &str,
        cpu_duration: std::time::Duration,
    ) -> Self {
        Self {
            index,
            operation: "dense_stage".to_string(),
            path: "dense_cpu".to_string(),
            stage: stage.to_string(),
            tensor_name: tensor_name.to_string(),
            rows: 0,
            cols: 0,
            pack_cache_hit: false,
            gpu_cache_hit: false,
            cpu_ms: cpu_duration.as_secs_f64() * 1_000.0,
            compile_ms: 0.0,
            weight_upload_ms: 0.0,
            activation_upload_ms: 0.0,
            gpu_ms: 0.0,
            download_ms: 0.0,
            weight_upload_bytes: 0,
            activation_upload_bytes: 0,
            download_bytes: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resident_stage(
        index: usize,
        path: &str,
        stage: &str,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        gpu_cache_hit: bool,
        compile_duration: std::time::Duration,
        weight_upload_duration: std::time::Duration,
        activation_upload_duration: std::time::Duration,
        gpu_duration: std::time::Duration,
        weight_upload_bytes: usize,
        activation_upload_bytes: usize,
    ) -> Self {
        Self {
            index,
            operation: "resident".to_string(),
            path: path.to_string(),
            stage: stage.to_string(),
            tensor_name: tensor_name.to_string(),
            rows,
            cols,
            pack_cache_hit: false,
            gpu_cache_hit,
            cpu_ms: (compile_duration + weight_upload_duration + activation_upload_duration)
                .as_secs_f64()
                * 1_000.0,
            compile_ms: compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: weight_upload_duration.as_secs_f64() * 1_000.0,
            activation_upload_ms: activation_upload_duration.as_secs_f64() * 1_000.0,
            gpu_ms: gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: 0.0,
            weight_upload_bytes,
            activation_upload_bytes,
            download_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeResult {
    pub output_token_ids: Vec<usize>,
    pub output_text: String,
    pub metrics: DecodeMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedDecodeResult {
    pub output_token_ids: Vec<usize>,
    pub output_text: String,
    pub decode_metrics: DecodeMetrics,
    pub metrics: PackedDecodeMetrics,
    pub dispatch_trace: Vec<PackedDispatchTrace>,
}
