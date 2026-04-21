use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_duration: Duration,
    pub embedding_duration: Duration,
    pub norm_duration: Duration,
    pub qkv_duration: Duration,
    pub attention_duration: Duration,
    pub mlp_duration: Duration,
    pub logits_duration: Duration,
}

impl DecodeMetrics {
    pub fn generated_tokens_per_second(&self) -> f64 {
        let seconds = self.total_duration.as_secs_f64();
        if self.generated_tokens == 0 || seconds <= f64::EPSILON {
            0.0
        } else {
            self.generated_tokens as f64 / seconds
        }
    }

    pub fn total_sequence_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }

    pub fn summarize(&self) -> String {
        format!(
            "prompt_tokens={} generated_tokens={} total_ms={:.3} embed_ms={:.3} norm_ms={:.3} qkv_ms={:.3} attention_ms={:.3} mlp_ms={:.3} logits_ms={:.3}",
            self.prompt_tokens,
            self.generated_tokens,
            self.total_duration.as_secs_f64() * 1_000.0,
            self.embedding_duration.as_secs_f64() * 1_000.0,
            self.norm_duration.as_secs_f64() * 1_000.0,
            self.qkv_duration.as_secs_f64() * 1_000.0,
            self.attention_duration.as_secs_f64() * 1_000.0,
            self.mlp_duration.as_secs_f64() * 1_000.0,
            self.logits_duration.as_secs_f64() * 1_000.0,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedDecodeMetrics {
    pub enabled_projections: String,
    pub total_duration: Duration,
    pub embedding_duration: Duration,
    pub norm_duration: Duration,
    pub qkv_duration: Duration,
    pub attention_duration: Duration,
    pub attention_query_duration: Duration,
    pub attention_oproj_duration: Duration,
    pub attention_residual_duration: Duration,
    pub mlp_duration: Duration,
    pub mlp_swiglu_duration: Duration,
    pub mlp_down_duration: Duration,
    pub mlp_residual_duration: Duration,
    pub logits_duration: Duration,
    pub pack_duration: Duration,
    pub compile_duration: Duration,
    pub weight_upload_duration: Duration,
    pub activation_upload_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub non_offloaded_dense_duration: Duration,
    pub orchestration_duration: Duration,
    pub pack_cache_hits: usize,
    pub gpu_cache_hits: usize,
    pub dispatch_count: usize,
    pub weight_upload_bytes: usize,
    pub activation_upload_bytes: usize,
    pub upload_bytes: usize,
    pub download_bytes: usize,
    pub output_text: String,
}

impl PackedDecodeMetrics {
    pub fn total_streamed_bytes(&self) -> usize {
        self.weight_upload_bytes + self.activation_upload_bytes + self.download_bytes
    }

    pub fn effective_end_to_end_bandwidth_gbps(&self) -> f64 {
        let seconds = self.total_duration.as_secs_f64();
        if seconds <= f64::EPSILON {
            0.0
        } else {
            self.total_streamed_bytes() as f64 / seconds / 1_000_000_000.0
        }
    }

    pub fn effective_stream_window_bandwidth_gbps(&self) -> f64 {
        let seconds = (self.weight_upload_duration
            + self.activation_upload_duration
            + self.gpu_duration
            + self.download_duration)
            .as_secs_f64();
        if seconds <= f64::EPSILON {
            0.0
        } else {
            self.total_streamed_bytes() as f64 / seconds / 1_000_000_000.0
        }
    }

    pub fn summarize(&self) -> String {
        format!(
            "enabled={} total_ms={:.3} embed_ms={:.3} norm_ms={:.3} qkv_ms={:.3} attention_ms={:.3} attention_query_ms={:.3} attention_oproj_ms={:.3} attention_residual_ms={:.3} mlp_ms={:.3} mlp_swiglu_ms={:.3} mlp_down_ms={:.3} mlp_residual_ms={:.3} logits_ms={:.3} pack_ms={:.3} compile_ms={:.3} weight_upload_ms={:.3} activation_upload_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} non_offloaded_dense_ms={:.3} orchestration_ms={:.3} pack_cache_hits={} gpu_cache_hits={} dispatch_count={} weight_upload_bytes={} activation_upload_bytes={} upload_bytes={} download_bytes={} streamed_bytes={} e2e_gbps={:.3} stream_window_gbps={:.3} output={}",
            self.enabled_projections,
            self.total_duration.as_secs_f64() * 1_000.0,
            self.embedding_duration.as_secs_f64() * 1_000.0,
            self.norm_duration.as_secs_f64() * 1_000.0,
            self.qkv_duration.as_secs_f64() * 1_000.0,
            self.attention_duration.as_secs_f64() * 1_000.0,
            self.attention_query_duration.as_secs_f64() * 1_000.0,
            self.attention_oproj_duration.as_secs_f64() * 1_000.0,
            self.attention_residual_duration.as_secs_f64() * 1_000.0,
            self.mlp_duration.as_secs_f64() * 1_000.0,
            self.mlp_swiglu_duration.as_secs_f64() * 1_000.0,
            self.mlp_down_duration.as_secs_f64() * 1_000.0,
            self.mlp_residual_duration.as_secs_f64() * 1_000.0,
            self.logits_duration.as_secs_f64() * 1_000.0,
            self.pack_duration.as_secs_f64() * 1_000.0,
            self.compile_duration.as_secs_f64() * 1_000.0,
            self.weight_upload_duration.as_secs_f64() * 1_000.0,
            self.activation_upload_duration.as_secs_f64() * 1_000.0,
            self.upload_duration.as_secs_f64() * 1_000.0,
            self.gpu_duration.as_secs_f64() * 1_000.0,
            self.download_duration.as_secs_f64() * 1_000.0,
            self.non_offloaded_dense_duration.as_secs_f64() * 1_000.0,
            self.orchestration_duration.as_secs_f64() * 1_000.0,
            self.pack_cache_hits,
            self.gpu_cache_hits,
            self.dispatch_count,
            self.weight_upload_bytes,
            self.activation_upload_bytes,
            self.upload_bytes,
            self.download_bytes,
            self.total_streamed_bytes(),
            self.effective_end_to_end_bandwidth_gbps(),
            self.effective_stream_window_bandwidth_gbps(),
            self.output_text,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionProjectionMixMetrics {
    pub enabled_projections: String,
    pub total_duration: Duration,
    pub pack_duration: Duration,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

impl AttentionProjectionMixMetrics {
    pub fn summarize(&self) -> String {
        format!(
            "enabled={} total_ms={:.3} pack_ms={:.3} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
            self.enabled_projections,
            self.total_duration.as_secs_f64() * 1_000.0,
            self.pack_duration.as_secs_f64() * 1_000.0,
            self.compile_duration.as_secs_f64() * 1_000.0,
            self.upload_duration.as_secs_f64() * 1_000.0,
            self.gpu_duration.as_secs_f64() * 1_000.0,
            self.download_duration.as_secs_f64() * 1_000.0,
            self.max_abs_diff,
            self.mean_abs_diff,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlpProjectionMixMetrics {
    pub enabled_projections: String,
    pub total_duration: Duration,
    pub pack_duration: Duration,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

impl MlpProjectionMixMetrics {
    pub fn summarize(&self) -> String {
        format!(
            "enabled={} total_ms={:.3} pack_ms={:.3} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
            self.enabled_projections,
            self.total_duration.as_secs_f64() * 1_000.0,
            self.pack_duration.as_secs_f64() * 1_000.0,
            self.compile_duration.as_secs_f64() * 1_000.0,
            self.upload_duration.as_secs_f64() * 1_000.0,
            self.gpu_duration.as_secs_f64() * 1_000.0,
            self.download_duration.as_secs_f64() * 1_000.0,
            self.max_abs_diff,
            self.mean_abs_diff,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HybridProjectionDecodeMetrics {
    pub enabled_projections: String,
    pub total_duration: Duration,
    pub pack_duration: Duration,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub output_text: String,
}

impl HybridProjectionDecodeMetrics {
    pub fn summarize(&self) -> String {
        format!(
            "enabled={} total_ms={:.3} pack_ms={:.3} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} output={}",
            self.enabled_projections,
            self.total_duration.as_secs_f64() * 1_000.0,
            self.pack_duration.as_secs_f64() * 1_000.0,
            self.compile_duration.as_secs_f64() * 1_000.0,
            self.upload_duration.as_secs_f64() * 1_000.0,
            self.gpu_duration.as_secs_f64() * 1_000.0,
            self.download_duration.as_secs_f64() * 1_000.0,
            self.output_text,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedDecodeValidationReport {
    pub enabled_projections: String,
    pub prompt_tokens: usize,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

impl PackedDecodeValidationReport {
    pub fn summarize(&self) -> String {
        format!(
            "enabled={} prompt_tokens={} max_abs_diff={:.6} mean_abs_diff={:.6}",
            self.enabled_projections, self.prompt_tokens, self.max_abs_diff, self.mean_abs_diff,
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PackedAttentionStageMetrics {
    pub query_duration: Duration,
    pub oproj_duration: Duration,
    pub residual_duration: Duration,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PackedMlpStageMetrics {
    pub swiglu_duration: Duration,
    pub down_duration: Duration,
    pub residual_duration: Duration,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PackedGpuSessionMetrics {
    pub pack_duration: Duration,
    pub compile_duration: Duration,
    pub weight_upload_duration: Duration,
    pub activation_upload_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub pack_cache_hits: usize,
    pub gpu_cache_hits: usize,
    pub dispatch_count: usize,
    pub weight_upload_bytes: usize,
    pub activation_upload_bytes: usize,
    pub upload_bytes: usize,
    pub download_bytes: usize,
}
