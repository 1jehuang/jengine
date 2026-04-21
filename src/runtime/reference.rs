use crate::cpu::block::{YarnRope, build_yarn_rope, rope_cos_sin};
use crate::cpu::primitives::{argmax, swiglu};
use crate::gpu::attention_block::{CachedGpuAttentionBlockRunner, GpuAttentionBlockReport};
use crate::gpu::embedding_lookup::{CachedGpuEmbeddingLookupRunner, GpuEmbeddingLookupReport};
use crate::gpu::full_last_layer_block::{
    CachedGpuFullLastLayerRunner, GpuFullLastLayerReport, PackedLinearSpec,
};
use crate::gpu::kv_cache::GpuKvCache;
use crate::gpu::mlp_block::{CachedGpuMlpBlockRunner, GpuMlpBlockReport};
use crate::gpu::pack_f16_pairs::CachedGpuPackF16PairsRunner;
use crate::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, PackedRunnerInputMode, SharedGpuPackedContext,
    run_packed_ternary_matvec_with_output,
};
use crate::gpu::qk_rope::{CachedGpuQkRopeRunner, GpuQkRopeReport};
use crate::gpu::resident_buffer::GpuResidentBuffer;
use crate::gpu::swiglu_combined::CachedGpuSwigluCombinedRunner;
use crate::gpu::swiglu_pack_f16_pairs::CachedGpuSwigluPackF16PairsRunner;
use crate::gpu::tail_block::{
    CachedGpuTailBlockRunner, GpuTailBlockLogitsReport, GpuTailBlockReport,
};
use crate::gpu::vector_add::CachedGpuVectorAddRunner;
use crate::gpu::weighted_rms_norm::{CachedGpuWeightedRmsNormRunner, GpuWeightedRmsNormReport};
use crate::model::config::{BonsaiModelConfig, GenerationConfig};
use crate::model::tokenizer::{PromptAnalysis, TokenizerDiagnostics, TokenizerRuntime};
use crate::runtime::assets::{AssetError, BonsaiAssetPaths};
use crate::runtime::packed_model::{PackedModelError, PackedModelStore};
use crate::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};
use crate::runtime::weights::{WeightError, WeightStore};
use ash::vk;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum ReferenceError {
    Asset(AssetError),
    Io(std::io::Error),
    Json(serde_json::Error),
    Tokenizer(crate::model::tokenizer::TokenizerLoadError),
    Weight(WeightError),
    PackedModel(PackedModelError),
    Decode(String),
}

impl std::fmt::Display for ReferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Asset(error) => write!(f, "asset error: {error}"),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
            Self::Tokenizer(error) => write!(f, "tokenizer error: {error}"),
            Self::Weight(error) => write!(f, "weight error: {error}"),
            Self::PackedModel(error) => write!(f, "packed model error: {error}"),
            Self::Decode(message) => write!(f, "decode error: {message}"),
        }
    }
}

impl std::error::Error for ReferenceError {}

impl From<AssetError> for ReferenceError {
    fn from(value: AssetError) -> Self {
        Self::Asset(value)
    }
}
impl From<std::io::Error> for ReferenceError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<serde_json::Error> for ReferenceError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
impl From<crate::model::tokenizer::TokenizerLoadError> for ReferenceError {
    fn from(value: crate::model::tokenizer::TokenizerLoadError) -> Self {
        Self::Tokenizer(value)
    }
}
impl From<WeightError> for ReferenceError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}
impl From<PackedModelError> for ReferenceError {
    fn from(value: PackedModelError) -> Self {
        Self::PackedModel(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionComparison {
    pub layer_idx: usize,
    pub token_id: usize,
    pub dense_duration: Duration,
    pub pack_duration: Duration,
    pub packed_duration: Duration,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HybridDecodeMetrics {
    pub total_duration: Duration,
    pub q_proj_pack_duration: Duration,
    pub q_proj_pack_cache_hit: bool,
    pub q_proj_gpu_compile_duration: Duration,
    pub q_proj_gpu_cache_hit: bool,
    pub q_proj_gpu_upload_duration: Duration,
    pub q_proj_gpu_duration: Duration,
    pub q_proj_gpu_download_duration: Duration,
    pub output_text: String,
}

impl HybridDecodeMetrics {
    pub fn summarize(&self) -> String {
        format!(
            "total_ms={:.3} qproj_pack_ms={:.3} qproj_pack_cache_hit={} qproj_gpu_compile_ms={:.3} qproj_gpu_cache_hit={} qproj_gpu_upload_ms={:.3} qproj_gpu_ms={:.3} qproj_gpu_download_ms={:.3} output={}",
            self.total_duration.as_secs_f64() * 1_000.0,
            self.q_proj_pack_duration.as_secs_f64() * 1_000.0,
            self.q_proj_pack_cache_hit,
            self.q_proj_gpu_compile_duration.as_secs_f64() * 1_000.0,
            self.q_proj_gpu_cache_hit,
            self.q_proj_gpu_upload_duration.as_secs_f64() * 1_000.0,
            self.q_proj_gpu_duration.as_secs_f64() * 1_000.0,
            self.q_proj_gpu_download_duration.as_secs_f64() * 1_000.0,
            self.output_text,
        )
    }
}
impl ProjectionComparison {
    pub fn summarize(&self) -> String {
        format!(
            "layer={} token_id={} dense_ms={:.3} pack_ms={:.3} packed_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
            self.layer_idx,
            self.token_id,
            self.dense_duration.as_secs_f64() * 1_000.0,
            self.pack_duration.as_secs_f64() * 1_000.0,
            self.packed_duration.as_secs_f64() * 1_000.0,
            self.max_abs_diff,
            self.mean_abs_diff,
        )
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReport {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_sequence_tokens: usize,
    pub estimated_model_fp16_bytes: usize,
    pub source_weight_bytes: usize,
    pub kv_cache_bytes_per_token_fp16: usize,
    pub kv_cache_bytes_per_token_runtime_f32: usize,
    pub kv_cache_total_bytes_fp16: usize,
    pub kv_cache_total_bytes_runtime_f32: usize,
    pub kv_cache_reserved_bytes_runtime_f32: usize,
    pub packed_cache_bytes: usize,
    pub gpu_cache_buffer_bytes: usize,
    pub activation_working_bytes: usize,
    pub staging_bytes: usize,
    pub estimated_runtime_working_set_bytes: usize,
}

impl MemoryReport {
    pub fn summarize(&self) -> String {
        format!(
            "prompt_tokens={} generated_tokens={} total_sequence_tokens={} model_fp16_bytes={} ({}) source_weight_bytes={} ({}) kv_per_token_fp16={} ({}) kv_per_token_runtime_f32={} ({}) kv_total_runtime_f32={} ({}) kv_reserved_runtime_f32={} ({}) packed_cache_bytes={} ({}) gpu_cache_buffer_bytes={} ({}) activation_working_bytes={} ({}) staging_bytes={} ({}) working_set_bytes={} ({})",
            self.prompt_tokens,
            self.generated_tokens,
            self.total_sequence_tokens,
            self.estimated_model_fp16_bytes,
            human_bytes(self.estimated_model_fp16_bytes),
            self.source_weight_bytes,
            human_bytes(self.source_weight_bytes),
            self.kv_cache_bytes_per_token_fp16,
            human_bytes(self.kv_cache_bytes_per_token_fp16),
            self.kv_cache_bytes_per_token_runtime_f32,
            human_bytes(self.kv_cache_bytes_per_token_runtime_f32),
            self.kv_cache_total_bytes_runtime_f32,
            human_bytes(self.kv_cache_total_bytes_runtime_f32),
            self.kv_cache_reserved_bytes_runtime_f32,
            human_bytes(self.kv_cache_reserved_bytes_runtime_f32),
            self.packed_cache_bytes,
            human_bytes(self.packed_cache_bytes),
            self.gpu_cache_buffer_bytes,
            human_bytes(self.gpu_cache_buffer_bytes),
            self.activation_working_bytes,
            human_bytes(self.activation_working_bytes),
            self.staging_bytes,
            human_bytes(self.staging_bytes),
            self.estimated_runtime_working_set_bytes,
            human_bytes(self.estimated_runtime_working_set_bytes),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeResult {
    pub output_token_ids: Vec<usize>,
    pub output_text: String,
    pub metrics: DecodeMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PackedDecodeStepResult {
    Logits(Vec<f32>),
    NextToken(usize),
}

#[derive(Debug, Clone)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerCache {
    fn with_capacity(tokens: usize, kv_width: usize) -> Self {
        let capacity = tokens * kv_width;
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    fn without_preallocated_cpu_kv() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct HybridQProjCache {
    layer_idx: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    code_words: Vec<u32>,
    scales: Vec<f32>,
}

#[derive(Debug, Clone)]
struct PackedProjectionCache {
    rows: usize,
    cols: usize,
    group_size: usize,
    code_words: Vec<u32>,
    scales: Vec<f32>,
}

#[derive(Debug, Clone)]
struct LayerTensorNames {
    input_layernorm_weight: String,
    post_attention_layernorm_weight: String,
    q_norm_weight: String,
    k_norm_weight: String,
    q_proj_weight: String,
    k_proj_weight: String,
    v_proj_weight: String,
    o_proj_weight: String,
    gate_proj_weight: String,
    up_proj_weight: String,
    down_proj_weight: String,
}

type CachedProjectionGpuRunner = Rc<RefCell<CachedGpuPackedMatvecRunner>>;
type CachedTailBlockGpuRunner = Rc<RefCell<CachedGpuTailBlockRunner>>;
type CachedFullLastLayerGpuRunner = Rc<RefCell<CachedGpuFullLastLayerRunner>>;
type CachedProjectionGpuCacheEntry = (CachedProjectionGpuRunner, Duration, Duration, bool);
type CachedWeightedRmsNormGpuRunner = Rc<RefCell<CachedGpuWeightedRmsNormRunner>>;
type CachedWeightedRmsNormGpuCacheEntry = (CachedWeightedRmsNormGpuRunner, Duration, bool);
type CachedPackF16PairsGpuRunner = Rc<RefCell<CachedGpuPackF16PairsRunner>>;
type CachedPackF16PairsGpuCacheEntry = (CachedPackF16PairsGpuRunner, Duration, bool);
type CachedVectorAddGpuRunner = Rc<RefCell<CachedGpuVectorAddRunner>>;
type CachedVectorAddGpuCacheEntry = (CachedVectorAddGpuRunner, Duration, bool);
type CachedSwigluCombinedGpuRunner = Rc<RefCell<CachedGpuSwigluCombinedRunner>>;
type CachedSwigluCombinedGpuCacheEntry = (CachedSwigluCombinedGpuRunner, Duration, bool);
#[allow(dead_code)]
type CachedSwigluPackF16PairsGpuRunner = Rc<RefCell<CachedGpuSwigluPackF16PairsRunner>>;
#[allow(dead_code)]
type CachedSwigluPackF16PairsGpuCacheEntry = (CachedSwigluPackF16PairsGpuRunner, Duration, bool);
type ProjectionTripletOutputs = (Vec<f32>, Vec<f32>, Vec<f32>);

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

#[derive(Debug, Clone, PartialEq)]
pub struct PackedDecodeResult {
    pub output_token_ids: Vec<usize>,
    pub output_text: String,
    pub decode_metrics: DecodeMetrics,
    pub metrics: PackedDecodeMetrics,
    pub dispatch_trace: Vec<PackedDispatchTrace>,
}

pub struct PersistentPackedDecodeSession<'a> {
    model: &'a ReferenceModel,
    cache: Vec<LayerCache>,
    gpu_session: PackedGpuSession<'a>,
    gpu_first_session: GpuFirstRunnerCache<'a>,
    metrics: DecodeMetrics,
    attention_stage_metrics: PackedAttentionStageMetrics,
    mlp_stage_metrics: PackedMlpStageMetrics,
    non_offloaded_dense_duration: Duration,
    next_position: usize,
    use_attention_qkv: bool,
    use_mlp_gu: bool,
    use_attention_full: bool,
    use_mlp_full: bool,
    argmax_only: bool,
}

impl<'a> PersistentPackedDecodeSession<'a> {
    fn new_with_cpu_kv_preallocation(
        model: &'a ReferenceModel,
        expected_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        argmax_only: bool,
        preallocate_cpu_kv: bool,
    ) -> Self {
        Self {
            model,
            cache: model.allocate_layer_cache_vec(expected_tokens, preallocate_cpu_kv),
            gpu_session: PackedGpuSession::new(model),
            gpu_first_session: GpuFirstRunnerCache::new(model, expected_tokens),
            metrics: DecodeMetrics {
                prompt_tokens: 0,
                generated_tokens: 0,
                total_duration: Duration::ZERO,
                embedding_duration: Duration::ZERO,
                norm_duration: Duration::ZERO,
                qkv_duration: Duration::ZERO,
                attention_duration: Duration::ZERO,
                mlp_duration: Duration::ZERO,
                logits_duration: Duration::ZERO,
            },
            attention_stage_metrics: PackedAttentionStageMetrics::default(),
            mlp_stage_metrics: PackedMlpStageMetrics::default(),
            non_offloaded_dense_duration: Duration::ZERO,
            next_position: 0,
            use_attention_qkv,
            use_mlp_gu,
            use_attention_full: use_attention_qkv && ReferenceModel::packed_use_attention_full(),
            use_mlp_full: use_mlp_gu && ReferenceModel::packed_use_mlp_full(),
            argmax_only,
        }
    }

    fn new(
        model: &'a ReferenceModel,
        expected_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        argmax_only: bool,
    ) -> Self {
        Self::new_with_cpu_kv_preallocation(
            model,
            expected_tokens,
            use_attention_qkv,
            use_mlp_gu,
            argmax_only,
            true,
        )
    }

    pub fn push_prompt_token(
        &mut self,
        token_id: usize,
    ) -> Result<PackedDecodeStepResult, ReferenceError> {
        self.metrics.prompt_tokens += 1;
        self.step_token(token_id)
    }

    pub fn push_generated_token(
        &mut self,
        token_id: usize,
    ) -> Result<PackedDecodeStepResult, ReferenceError> {
        self.metrics.generated_tokens += 1;
        self.step_token(token_id)
    }

    fn step_token(&mut self, token_id: usize) -> Result<PackedDecodeStepResult, ReferenceError> {
        let result = self.model.forward_step_packed_decode(
            token_id,
            self.next_position,
            &mut self.cache,
            &mut self.metrics,
            &mut self.attention_stage_metrics,
            &mut self.mlp_stage_metrics,
            &mut self.non_offloaded_dense_duration,
            &mut self.gpu_session,
            &mut self.gpu_first_session,
            self.use_attention_qkv,
            self.use_mlp_gu,
            self.use_attention_full,
            self.use_mlp_full,
            self.argmax_only,
        )?;
        self.next_position += 1;
        Ok(result)
    }

    pub fn dispatch_trace(&self) -> &[PackedDispatchTrace] {
        &self.gpu_session.dispatch_trace
    }

    pub fn finish_metrics(
        self,
        enabled_projections: String,
        total_duration: Duration,
        output_text: String,
    ) -> PackedDecodeMetrics {
        ReferenceModel::finish_packed_decode_metrics(
            enabled_projections,
            total_duration,
            &self.metrics,
            &self.attention_stage_metrics,
            &self.mlp_stage_metrics,
            self.non_offloaded_dense_duration,
            &self.gpu_session,
            output_text,
        )
    }

    pub fn finish_result(
        self,
        enabled_projections: String,
        total_duration: Duration,
        output_token_ids: Vec<usize>,
        output_text: String,
    ) -> PackedDecodeResult {
        let PersistentPackedDecodeSession {
            metrics: decode_metrics,
            attention_stage_metrics,
            mlp_stage_metrics,
            non_offloaded_dense_duration,
            gpu_session,
            ..
        } = self;
        let metrics = ReferenceModel::finish_packed_decode_metrics(
            enabled_projections,
            total_duration,
            &decode_metrics,
            &attention_stage_metrics,
            &mlp_stage_metrics,
            non_offloaded_dense_duration,
            &gpu_session,
            output_text.clone(),
        );
        let dispatch_trace = gpu_session.dispatch_trace;
        PackedDecodeResult {
            output_token_ids,
            output_text,
            decode_metrics,
            metrics,
            dispatch_trace,
        }
    }
}

pub struct GpuFirstPackedDecodeSession<'a> {
    inner: PersistentPackedDecodeSession<'a>,
}

impl<'a> GpuFirstPackedDecodeSession<'a> {
    fn new(
        model: &'a ReferenceModel,
        expected_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        argmax_only: bool,
    ) -> Self {
        let mut inner = PersistentPackedDecodeSession::new_with_cpu_kv_preallocation(
            model,
            expected_tokens,
            use_attention_qkv,
            use_mlp_gu,
            argmax_only,
            false,
        );
        inner.use_attention_full = use_attention_qkv
            && (ReferenceModel::packed_use_attention_full()
                || ReferenceModel::packed_use_gpu_attention_block()
                || ReferenceModel::packed_use_gpu_full_last_layer());
        inner.use_mlp_full = use_mlp_gu
            && (ReferenceModel::packed_use_mlp_full()
                || ReferenceModel::packed_use_gpu_swiglu_block()
                || ReferenceModel::packed_use_gpu_full_last_layer()
                || ReferenceModel::packed_use_gpu_mlp_entry());
        Self { inner }
    }

    pub fn push_prompt_token(
        &mut self,
        token_id: usize,
    ) -> Result<PackedDecodeStepResult, ReferenceError> {
        self.inner.push_prompt_token(token_id)
    }

    pub fn push_generated_token(
        &mut self,
        token_id: usize,
    ) -> Result<PackedDecodeStepResult, ReferenceError> {
        self.inner.push_generated_token(token_id)
    }

    pub fn next_position(&self) -> usize {
        self.inner.next_position
    }

    pub fn finish_metrics(
        self,
        enabled_projections: String,
        total_duration: Duration,
        output_text: String,
    ) -> PackedDecodeMetrics {
        self.inner
            .finish_metrics(enabled_projections, total_duration, output_text)
    }

    pub fn finish_result(
        self,
        enabled_projections: String,
        total_duration: Duration,
        output_token_ids: Vec<usize>,
        output_text: String,
    ) -> PackedDecodeResult {
        self.inner.finish_result(
            enabled_projections,
            total_duration,
            output_token_ids,
            output_text,
        )
    }
}

pub enum PackedDecodeSession<'a> {
    Legacy(PersistentPackedDecodeSession<'a>),
    GpuFirst(GpuFirstPackedDecodeSession<'a>),
}

impl<'a> PackedDecodeSession<'a> {
    pub fn push_prompt_token(
        &mut self,
        token_id: usize,
    ) -> Result<PackedDecodeStepResult, ReferenceError> {
        match self {
            Self::Legacy(session) => session.push_prompt_token(token_id),
            Self::GpuFirst(session) => session.push_prompt_token(token_id),
        }
    }

    pub fn push_generated_token(
        &mut self,
        token_id: usize,
    ) -> Result<PackedDecodeStepResult, ReferenceError> {
        match self {
            Self::Legacy(session) => session.push_generated_token(token_id),
            Self::GpuFirst(session) => session.push_generated_token(token_id),
        }
    }

    pub fn next_position(&self) -> usize {
        match self {
            Self::Legacy(session) => session.next_position,
            Self::GpuFirst(session) => session.next_position(),
        }
    }

    pub fn is_gpu_first(&self) -> bool {
        matches!(self, Self::GpuFirst(_))
    }

    pub fn finish_metrics(
        self,
        enabled_projections: String,
        total_duration: Duration,
        output_text: String,
    ) -> PackedDecodeMetrics {
        match self {
            Self::Legacy(session) => {
                session.finish_metrics(enabled_projections, total_duration, output_text)
            }
            Self::GpuFirst(session) => {
                session.finish_metrics(enabled_projections, total_duration, output_text)
            }
        }
    }

    pub fn finish_result(
        self,
        enabled_projections: String,
        total_duration: Duration,
        output_token_ids: Vec<usize>,
        output_text: String,
    ) -> PackedDecodeResult {
        match self {
            Self::Legacy(session) => session.finish_result(
                enabled_projections,
                total_duration,
                output_token_ids,
                output_text,
            ),
            Self::GpuFirst(session) => session.finish_result(
                enabled_projections,
                total_duration,
                output_token_ids,
                output_text,
            ),
        }
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
struct PackedAttentionStageMetrics {
    query_duration: Duration,
    oproj_duration: Duration,
    residual_duration: Duration,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PackedMlpStageMetrics {
    swiglu_duration: Duration,
    down_duration: Duration,
    residual_duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentHiddenState {
    Mlp { layer_idx: usize },
}

struct GpuFirstRunnerCache<'a> {
    model: &'a ReferenceModel,
    kv_capacity_tokens: usize,
    shared_context: Option<Arc<SharedGpuPackedContext>>,
    embedding_lookup_runner: Option<CachedGpuEmbeddingLookupRunner>,
    input_norm_runner: Option<CachedGpuWeightedRmsNormRunner>,
    qk_rope_runner: Option<CachedGpuQkRopeRunner>,
    raw_f32_projection_runners: HashMap<String, CachedGpuPackedMatvecRunner>,
    gpu_kv_caches: HashMap<usize, GpuKvCache>,
    attention_blocks: HashMap<(usize, usize), CachedGpuAttentionBlockRunner>,
    mlp_blocks: HashMap<usize, CachedGpuMlpBlockRunner>,
    full_last_layer_block: Option<CachedFullLastLayerGpuRunner>,
    tail_block: Option<CachedTailBlockGpuRunner>,
}

impl<'a> GpuFirstRunnerCache<'a> {
    fn new(model: &'a ReferenceModel, expected_tokens: usize) -> Self {
        Self {
            model,
            kv_capacity_tokens: expected_tokens,
            shared_context: None,
            embedding_lookup_runner: None,
            input_norm_runner: None,
            qk_rope_runner: None,
            raw_f32_projection_runners: HashMap::new(),
            gpu_kv_caches: HashMap::new(),
            attention_blocks: HashMap::new(),
            mlp_blocks: HashMap::new(),
            full_last_layer_block: None,
            tail_block: None,
        }
    }

    fn get_or_create_shared_context(
        &mut self,
    ) -> Result<Arc<SharedGpuPackedContext>, ReferenceError> {
        if let Some(context) = self.shared_context.as_ref().cloned() {
            return Ok(context);
        }
        let context = self.model.get_or_create_packed_gpu_context()?;
        self.shared_context = Some(context.clone());
        Ok(context)
    }

    fn ensure_gpu_kv_cache(&mut self, layer_idx: usize) -> Result<&mut GpuKvCache, ReferenceError> {
        if !self.gpu_kv_caches.contains_key(&layer_idx) {
            let context = self.get_or_create_shared_context()?;
            let kv_width = self.model.config.num_key_value_heads * self.model.config.head_dim;
            let cache = GpuKvCache::new_with_context(context, self.kv_capacity_tokens, kv_width)
                .map_err(|error| {
                    ReferenceError::Decode(format!(
                        "gpu-first kv cache init failed for layer {layer_idx}: {error}"
                    ))
                })?;
            self.gpu_kv_caches.insert(layer_idx, cache);
        }
        Ok(self
            .gpu_kv_caches
            .get_mut(&layer_idx)
            .expect("gpu kv cache should exist after creation"))
    }

    fn append_gpu_kv(
        &mut self,
        layer_idx: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), ReferenceError> {
        self.ensure_gpu_kv_cache(layer_idx)?
            .append(key, value)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu-first kv append failed for layer {layer_idx}: {error}"
                ))
            })
    }

    fn append_gpu_kv_key_tensor_and_value_host(
        &mut self,
        layer_idx: usize,
        key: &GpuResidentBuffer,
        value: &[f32],
    ) -> Result<(), ReferenceError> {
        self.ensure_gpu_kv_cache(layer_idx)?
            .append_key_from_tensor_and_value_host(key, value)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu-first resident key kv append failed for layer {layer_idx}: {error}"
                ))
            })
    }

    fn append_gpu_kv_tensors(
        &mut self,
        layer_idx: usize,
        key: &GpuResidentBuffer,
        value: &GpuResidentBuffer,
    ) -> Result<(), ReferenceError> {
        self.ensure_gpu_kv_cache(layer_idx)?
            .append_from_tensors(key, value)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu-first resident kv append failed for layer {layer_idx}: {error}"
                ))
            })
    }

    fn gpu_kv_state(
        &self,
        layer_idx: usize,
    ) -> Option<(vk::Buffer, usize, u64, vk::Buffer, usize, u64)> {
        self.gpu_kv_caches.get(&layer_idx).map(|cache| {
            let kv_len = cache.len_tokens() * cache.kv_width();
            (
                cache.key_buffer_handle(),
                kv_len,
                cache.key_buffer_size(),
                cache.value_buffer_handle(),
                kv_len,
                cache.value_buffer_size(),
            )
        })
    }

    fn ensure_embedding_lookup_runner(
        &mut self,
    ) -> Result<(Duration, &mut CachedGpuEmbeddingLookupRunner), ReferenceError> {
        let compile_duration = if self.embedding_lookup_runner.is_some() {
            Duration::ZERO
        } else {
            let (vocab, hidden, words) = self
                .model
                .weights
                .embedding_lookup_u32_words("model.embed_tokens.weight")
                .map_err(ReferenceError::Weight)?;
            let context = self.get_or_create_shared_context()?;
            let (runner, compile_duration) =
                CachedGpuEmbeddingLookupRunner::new_with_context(context, &words, vocab, hidden)
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            self.embedding_lookup_runner = Some(runner);
            compile_duration
        };
        Ok((
            compile_duration,
            self.embedding_lookup_runner
                .as_mut()
                .expect("embedding lookup runner should exist after creation"),
        ))
    }

    fn ensure_input_norm_runner(
        &mut self,
    ) -> Result<(Duration, &mut CachedGpuWeightedRmsNormRunner), ReferenceError> {
        let compile_duration = if self.input_norm_runner.is_some() {
            Duration::ZERO
        } else {
            let context = self.get_or_create_shared_context()?;
            let (runner, compile_duration) = CachedGpuWeightedRmsNormRunner::new_with_context(
                context,
                self.model.config.hidden_size,
                self.model.config.rms_norm_eps as f32,
            )
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            self.input_norm_runner = Some(runner);
            compile_duration
        };
        Ok((
            compile_duration,
            self.input_norm_runner
                .as_mut()
                .expect("input norm runner should exist after creation"),
        ))
    }

    fn ensure_qk_rope_runner(
        &mut self,
    ) -> Result<(Duration, &mut CachedGpuQkRopeRunner), ReferenceError> {
        let compile_duration = if self.qk_rope_runner.is_some() {
            Duration::ZERO
        } else {
            let context = self.get_or_create_shared_context()?;
            let (runner, compile_duration) = CachedGpuQkRopeRunner::new_with_context(
                context,
                self.model.config.num_attention_heads,
                self.model.config.num_key_value_heads,
                self.model.config.head_dim,
            )
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            self.qk_rope_runner = Some(runner);
            compile_duration
        };
        Ok((
            compile_duration,
            self.qk_rope_runner
                .as_mut()
                .expect("qk rope runner should exist after creation"),
        ))
    }

    fn gpu_kv_tensors(&self, layer_idx: usize) -> Option<(GpuResidentBuffer, GpuResidentBuffer)> {
        self.gpu_kv_caches.get(&layer_idx).map(|cache| {
            (
                GpuResidentBuffer::new(
                    self.shared_context
                        .as_ref()
                        .expect("shared context should exist when gpu kv cache exists")
                        .clone(),
                    cache.key_buffer_handle(),
                    cache.len_tokens() * cache.kv_width(),
                    cache.key_buffer_size(),
                ),
                GpuResidentBuffer::new(
                    self.shared_context
                        .as_ref()
                        .expect("shared context should exist when gpu kv cache exists")
                        .clone(),
                    cache.value_buffer_handle(),
                    cache.len_tokens() * cache.kv_width(),
                    cache.value_buffer_size(),
                ),
            )
        })
    }

    fn run_qk_rope_query_to_host_key_resident(
        &mut self,
        q: &[f32],
        k: &[f32],
        q_weight: &[f32],
        k_weight: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<((Vec<f32>, GpuResidentBuffer), GpuQkRopeReport, Duration), ReferenceError> {
        let (compile_duration, runner) = self.ensure_qk_rope_runner()?;
        let mut report = runner
            .run_resident(q, k, q_weight, k_weight, cos, sin)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (q_out, download_duration) = runner
            .read_query_output()
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        report.download_duration = download_duration;
        let key_out = runner.key_resident_output();
        Ok(((q_out, key_out), report, compile_duration))
    }

    fn ensure_raw_f32_projection_runner(
        &mut self,
        cache_key: &str,
        packed: &PackedProjectionCache,
    ) -> Result<(Duration, &mut CachedGpuPackedMatvecRunner), ReferenceError> {
        let compile_duration = if self.raw_f32_projection_runners.contains_key(cache_key) {
            Duration::ZERO
        } else {
            let context = self.get_or_create_shared_context()?;
            let (runner, compile_duration) =
                CachedGpuPackedMatvecRunner::new_with_context_and_input_mode(
                    context,
                    &packed.code_words,
                    &packed.scales,
                    packed.group_size,
                    packed.rows,
                    packed.cols,
                    PackedRunnerInputMode::RawF32,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            self.raw_f32_projection_runners
                .insert(cache_key.to_string(), runner);
            compile_duration
        };
        Ok((
            compile_duration,
            self.raw_f32_projection_runners
                .get_mut(cache_key)
                .expect("raw-f32 projection runner should exist after creation"),
        ))
    }

    fn ensure_raw_f32_single_projection_runner(
        &mut self,
        cache_key: &str,
        tensor_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(Duration, &mut CachedGpuPackedMatvecRunner), ReferenceError> {
        let (packed, _, _) = self.model.get_or_create_projection_cache(tensor_name, rows, cols)?;
        self.ensure_raw_f32_projection_runner(cache_key, &packed)
    }

    fn run_qk_rope_resident_query_and_key(
        &mut self,
        q: &GpuResidentBuffer,
        k: &GpuResidentBuffer,
        q_weight: &[f32],
        k_weight: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(GpuResidentBuffer, GpuResidentBuffer, GpuQkRopeReport, Duration), ReferenceError>
    {
        let (compile_duration, runner) = self.ensure_qk_rope_runner()?;
        let report = runner
            .run_resident_from_tensors(q, k, q_weight, k_weight, cos, sin)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let query_out = runner.query_resident_output();
        let key_out = runner.key_resident_output();
        Ok((query_out, key_out, report, compile_duration))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_first_layer_embedding_norm_qkv_to_host(
        &mut self,
        layer_idx: usize,
        token_id: usize,
        input_norm_weight: &[f32],
        q_proj_name: &str,
        k_proj_name: &str,
        v_proj_name: &str,
        kv_rows: usize,
    ) -> Result<
        (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            GpuEmbeddingLookupReport,
            GpuWeightedRmsNormReport,
            crate::gpu::packed_matvec::GpuPackedMatvecReport,
            Duration,
        ),
        ReferenceError,
    > {
        let hidden_size = self.model.config.hidden_size;
        let (embedding_compile_duration, embedding_runner) =
            self.ensure_embedding_lookup_runner()?;
        let embedding_report = embedding_runner
            .run_resident(token_id)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let hidden = vec![0.0; hidden_size];
        let embedding_output = embedding_runner.resident_output();

        let cache_key = format!(
            "gpu_first::layer::{layer_idx}::qkv_triplet::{q_proj_name}||{k_proj_name}||{v_proj_name}"
        );
        let (packed, _, _) = self.model.get_or_create_projection_triplet_cache(
            &cache_key,
            q_proj_name,
            hidden_size,
            k_proj_name,
            kv_rows,
            v_proj_name,
            kv_rows,
            hidden_size,
        )?;

        let (norm_compile_duration, norm_context, norm_report) = {
            let (norm_compile_duration, input_norm_runner) = self.ensure_input_norm_runner()?;
            let norm_report = input_norm_runner
                .run_resident_from_tensor(&embedding_output, input_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            (
                norm_compile_duration,
                input_norm_runner.resident_output(),
                norm_report,
            )
        };

        let (qkv_compile_duration, qkv_runner) =
            self.ensure_raw_f32_projection_runner(&cache_key, &packed)?;
        let qkv_report = qkv_runner
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (combined, qkv_download_duration) = qkv_runner
            .read_output()
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let qkv_report = crate::gpu::packed_matvec::GpuPackedMatvecReport {
            download_duration: qkv_download_duration,
            ..qkv_report
        };
        let (q, tail) = combined.split_at(self.model.config.hidden_size);
        let (k, v) = tail.split_at(kv_rows);

        Ok((
            hidden,
            q.to_vec(),
            k.to_vec(),
            v.to_vec(),
            embedding_report,
            norm_report,
            qkv_report,
            embedding_compile_duration + norm_compile_duration + qkv_compile_duration,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_first_layer_embedding_norm_qkv_tensors(
        &mut self,
        layer_idx: usize,
        token_id: usize,
        input_norm_weight: &[f32],
        q_proj_name: &str,
        k_proj_name: &str,
        v_proj_name: &str,
        kv_rows: usize,
    ) -> Result<
        (
            Vec<f32>,
            GpuResidentBuffer,
            GpuResidentBuffer,
            GpuResidentBuffer,
            GpuResidentBuffer,
            GpuEmbeddingLookupReport,
            GpuWeightedRmsNormReport,
            crate::gpu::packed_matvec::GpuPackedMatvecReport,
            Duration,
        ),
        ReferenceError,
    > {
        let hidden_size = self.model.config.hidden_size;
        let (embedding_compile_duration, embedding_runner) =
            self.ensure_embedding_lookup_runner()?;
        let (hidden, embedding_report) = embedding_runner
            .run_with_output(token_id)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let embedding_output = embedding_runner.resident_output();

        let (norm_compile_duration, norm_context, norm_report) = {
            let (norm_compile_duration, input_norm_runner) = self.ensure_input_norm_runner()?;
            let norm_report = input_norm_runner
                .run_resident_from_tensor(&embedding_output, input_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            (
                norm_compile_duration,
                input_norm_runner.resident_output(),
                norm_report,
            )
        };

        let q_key = format!("gpu_first::layer::{layer_idx}::q_proj::{q_proj_name}");
        let k_key = format!("gpu_first::layer::{layer_idx}::k_proj::{k_proj_name}");
        let v_key = format!("gpu_first::layer::{layer_idx}::v_proj::{v_proj_name}");
        let (q_compile_duration, q_runner) = self.ensure_raw_f32_single_projection_runner(
            &q_key,
            q_proj_name,
            hidden_size,
            hidden_size,
        )?;
        let q_report = q_runner
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let q_tensor = GpuResidentBuffer::new(
            q_runner.shared_context().clone(),
            q_runner.output_buffer_handle(),
            hidden_size,
            q_runner.output_buffer_size(),
        );

        let (k_compile_duration, k_runner) = self.ensure_raw_f32_single_projection_runner(
            &k_key,
            k_proj_name,
            kv_rows,
            hidden_size,
        )?;
        let k_report = k_runner
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let k_tensor = GpuResidentBuffer::new(
            k_runner.shared_context().clone(),
            k_runner.output_buffer_handle(),
            kv_rows,
            k_runner.output_buffer_size(),
        );

        let (v_compile_duration, v_runner) = self.ensure_raw_f32_single_projection_runner(
            &v_key,
            v_proj_name,
            kv_rows,
            hidden_size,
        )?;
        let v_report = v_runner
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let v_tensor = GpuResidentBuffer::new(
            v_runner.shared_context().clone(),
            v_runner.output_buffer_handle(),
            kv_rows,
            v_runner.output_buffer_size(),
        );

        let qkv_report = crate::gpu::packed_matvec::GpuPackedMatvecReport {
            rows: hidden_size + 2 * kv_rows,
            cols: hidden_size,
            compile_duration: Duration::ZERO,
            upload_duration: q_report.upload_duration
                + k_report.upload_duration
                + v_report.upload_duration,
            gpu_duration: q_report.gpu_duration + k_report.gpu_duration + v_report.gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        };

        Ok((
            hidden,
            embedding_output,
            q_tensor,
            k_tensor,
            v_tensor,
            embedding_report,
            norm_report,
            qkv_report,
            embedding_compile_duration
                + norm_compile_duration
                + q_compile_duration
                + k_compile_duration
                + v_compile_duration,
        ))
    }

    fn run_embedding_and_input_norm_to_host(
        &mut self,
        token_id: usize,
        input_norm_weight: &[f32],
    ) -> Result<
        (
            Vec<f32>,
            Vec<f32>,
            GpuEmbeddingLookupReport,
            GpuWeightedRmsNormReport,
            Duration,
        ),
        ReferenceError,
    > {
        let (embedding_compile_duration, embedding_runner) =
            self.ensure_embedding_lookup_runner()?;
        let (hidden, embedding_report) = embedding_runner
            .run_with_output(token_id)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let embedding_context = embedding_runner.shared_context().clone();
        let embedding_buffer = embedding_runner.output_buffer_handle();
        let embedding_buffer_size = embedding_runner.output_buffer_size();
        let hidden_len = embedding_runner.hidden();

        let (norm_compile_duration, input_norm_runner) = self.ensure_input_norm_runner()?;
        let norm_report = input_norm_runner
            .run_resident_from_f32_buffer(
                &embedding_context,
                embedding_buffer,
                hidden_len,
                embedding_buffer_size,
                input_norm_weight,
            )
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (hidden_states, norm_download_duration) = input_norm_runner
            .read_output()
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let norm_report = GpuWeightedRmsNormReport {
            download_duration: norm_download_duration,
            ..norm_report
        };
        Ok((
            hidden,
            hidden_states,
            embedding_report,
            norm_report,
            embedding_compile_duration + norm_compile_duration,
        ))
    }

    fn resident_hidden_output(
        &mut self,
        state: ResidentHiddenState,
    ) -> Result<GpuResidentBuffer, ReferenceError> {
        match state {
            ResidentHiddenState::Mlp { layer_idx } => {
                let runner = self.mlp_blocks.get_mut(&layer_idx).ok_or_else(|| {
                    ReferenceError::Decode(format!(
                        "gpu-first mlp block output missing for layer {layer_idx}"
                    ))
                })?;
                Ok(runner.resident_output())
            }
        }
    }

    fn read_hidden_output(
        &mut self,
        state: ResidentHiddenState,
    ) -> Result<(Vec<f32>, Duration), ReferenceError> {
        match state {
            ResidentHiddenState::Mlp { layer_idx } => self
                .mlp_blocks
                .get_mut(&layer_idx)
                .ok_or_else(|| {
                    ReferenceError::Decode(format!(
                        "gpu-first mlp block output missing for layer {layer_idx}"
                    ))
                })?
                .read_output()
                .map_err(|error| ReferenceError::Decode(error.to_string())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer_input_norm_qkv_from_hidden_resident(
        &mut self,
        state: ResidentHiddenState,
        layer_idx: usize,
        input_norm_weight: &[f32],
        q_proj_name: &str,
        k_proj_name: &str,
        v_proj_name: &str,
        kv_rows: usize,
    ) -> Result<
        (
            GpuResidentBuffer,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            GpuWeightedRmsNormReport,
            crate::gpu::packed_matvec::GpuPackedMatvecReport,
            Duration,
        ),
        ReferenceError,
    > {
        let hidden_size = self.model.config.hidden_size;
        let hidden_resident = self.resident_hidden_output(state)?;
        let cache_key = format!(
            "gpu_first::layer::{layer_idx}::qkv_triplet::{q_proj_name}||{k_proj_name}||{v_proj_name}"
        );
        let (packed, _, _) = self.model.get_or_create_projection_triplet_cache(
            &cache_key,
            q_proj_name,
            hidden_size,
            k_proj_name,
            kv_rows,
            v_proj_name,
            kv_rows,
            hidden_size,
        )?;

        let (norm_compile_duration, norm_context, norm_report) = {
            let (norm_compile_duration, input_norm_runner) = self.ensure_input_norm_runner()?;
            let norm_report = input_norm_runner
                .run_resident_from_tensor(&hidden_resident, input_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            (
                norm_compile_duration,
                input_norm_runner.resident_output(),
                norm_report,
            )
        };

        let (qkv_compile_duration, qkv_runner) =
            self.ensure_raw_f32_projection_runner(&cache_key, &packed)?;
        let qkv_report = qkv_runner
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (combined, qkv_download_duration) = qkv_runner
            .read_output()
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let qkv_report = crate::gpu::packed_matvec::GpuPackedMatvecReport {
            download_duration: qkv_download_duration,
            ..qkv_report
        };
        let (q, tail) = combined.split_at(self.model.config.hidden_size);
        let (k, v) = tail.split_at(kv_rows);

        Ok((
            hidden_resident,
            q.to_vec(),
            k.to_vec(),
            v.to_vec(),
            norm_report,
            qkv_report,
            norm_compile_duration + qkv_compile_duration,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer_input_norm_qkv_tensors_from_hidden_resident(
        &mut self,
        state: ResidentHiddenState,
        layer_idx: usize,
        input_norm_weight: &[f32],
        q_proj_name: &str,
        k_proj_name: &str,
        v_proj_name: &str,
        kv_rows: usize,
    ) -> Result<
        (
            GpuResidentBuffer,
            GpuResidentBuffer,
            GpuResidentBuffer,
            GpuResidentBuffer,
            GpuWeightedRmsNormReport,
            crate::gpu::packed_matvec::GpuPackedMatvecReport,
            Duration,
        ),
        ReferenceError,
    > {
        let hidden_size = self.model.config.hidden_size;
        let hidden_resident = self.resident_hidden_output(state)?;

        let (norm_compile_duration, norm_context, norm_report) = {
            let (norm_compile_duration, input_norm_runner) = self.ensure_input_norm_runner()?;
            let norm_report = input_norm_runner
                .run_resident_from_tensor(&hidden_resident, input_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            (
                norm_compile_duration,
                input_norm_runner.resident_output(),
                norm_report,
            )
        };

        let q_key = format!("gpu_first::layer::{layer_idx}::q_proj::{q_proj_name}");
        let k_key = format!("gpu_first::layer::{layer_idx}::k_proj::{k_proj_name}");
        let v_key = format!("gpu_first::layer::{layer_idx}::v_proj::{v_proj_name}");
        let (q_compile_duration, q_runner) = self.ensure_raw_f32_single_projection_runner(
            &q_key,
            q_proj_name,
            hidden_size,
            hidden_size,
        )?;
        let q_report = q_runner
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let q_tensor = GpuResidentBuffer::new(
            q_runner.shared_context().clone(),
            q_runner.output_buffer_handle(),
            hidden_size,
            q_runner.output_buffer_size(),
        );

        let (k_compile_duration, k_runner) = self.ensure_raw_f32_single_projection_runner(
            &k_key,
            k_proj_name,
            kv_rows,
            hidden_size,
        )?;
        let k_report = k_runner
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let k_tensor = GpuResidentBuffer::new(
            k_runner.shared_context().clone(),
            k_runner.output_buffer_handle(),
            kv_rows,
            k_runner.output_buffer_size(),
        );

        let (v_compile_duration, v_runner) = self.ensure_raw_f32_single_projection_runner(
            &v_key,
            v_proj_name,
            kv_rows,
            hidden_size,
        )?;
        let v_report = v_runner
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let v_tensor = GpuResidentBuffer::new(
            v_runner.shared_context().clone(),
            v_runner.output_buffer_handle(),
            kv_rows,
            v_runner.output_buffer_size(),
        );

        let qkv_report = crate::gpu::packed_matvec::GpuPackedMatvecReport {
            rows: hidden_size + 2 * kv_rows,
            cols: hidden_size,
            compile_duration: Duration::ZERO,
            upload_duration: q_report.upload_duration + k_report.upload_duration + v_report.upload_duration,
            gpu_duration: q_report.gpu_duration + k_report.gpu_duration + v_report.gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        };

        Ok((
            hidden_resident,
            q_tensor,
            k_tensor,
            v_tensor,
            norm_report,
            qkv_report,
            norm_compile_duration + q_compile_duration + k_compile_duration + v_compile_duration,
        ))
    }

    fn packed_linear_spec(
        &self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<PackedLinearSpec, ReferenceError> {
        let (packed, _, _) = self
            .model
            .get_or_create_projection_cache(tensor_name, rows, cols)?;
        Ok(PackedLinearSpec {
            code_words: packed.code_words.clone(),
            scales: packed.scales.clone(),
            group_size: packed.group_size,
            rows: packed.rows,
            cols: packed.cols,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn packed_pair_linear_spec(
        &self,
        cache_key: &str,
        first_name: &str,
        first_rows: usize,
        second_name: &str,
        second_rows: usize,
        cols: usize,
    ) -> Result<PackedLinearSpec, ReferenceError> {
        let (packed, _, _) = self.model.get_or_create_projection_pair_cache(
            cache_key,
            first_name,
            first_rows,
            second_name,
            second_rows,
            cols,
        )?;
        Ok(PackedLinearSpec {
            code_words: packed.code_words.clone(),
            scales: packed.scales.clone(),
            group_size: packed.group_size,
            rows: packed.rows,
            cols: packed.cols,
        })
    }

    fn ensure_attention_block(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<(Duration, &mut CachedGpuAttentionBlockRunner), ReferenceError> {
        let cache_key = (layer_idx, seq_len);
        let compile_duration = if self.attention_blocks.contains_key(&cache_key) {
            Duration::ZERO
        } else {
            let layer_tensors = &self.model.layer_tensors[layer_idx];
            let o_proj_spec = self.packed_linear_spec(
                &layer_tensors.o_proj_weight,
                self.model.config.hidden_size,
                self.model.config.hidden_size,
            )?;
            let context = self.get_or_create_shared_context()?;
            let runner = CachedGpuAttentionBlockRunner::new_with_context(
                context,
                seq_len,
                self.model.config.num_attention_heads,
                self.model.config.num_key_value_heads,
                self.model.config.head_dim,
                &o_proj_spec,
            )
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            let compile_duration = runner.compile_duration();
            self.attention_blocks.insert(cache_key, runner);
            compile_duration
        };
        Ok((
            compile_duration,
            self.attention_blocks
                .get_mut(&cache_key)
                .expect("attention block runner should exist after creation"),
        ))
    }

    fn ensure_mlp_block(
        &mut self,
        layer_idx: usize,
    ) -> Result<(Duration, &mut CachedGpuMlpBlockRunner), ReferenceError> {
        let compile_duration = if self.mlp_blocks.contains_key(&layer_idx) {
            Duration::ZERO
        } else {
            let layer_tensors = &self.model.layer_tensors[layer_idx];
            let pair_cache_key = format!(
                "gpu_first::layer::{layer_idx}::mlp_pair::{}+{}",
                layer_tensors.gate_proj_weight, layer_tensors.up_proj_weight
            );
            let pair_spec = self.packed_pair_linear_spec(
                &pair_cache_key,
                &layer_tensors.gate_proj_weight,
                self.model.config.intermediate_size,
                &layer_tensors.up_proj_weight,
                self.model.config.intermediate_size,
                self.model.config.hidden_size,
            )?;
            let down_spec = self.packed_linear_spec(
                &layer_tensors.down_proj_weight,
                self.model.config.hidden_size,
                self.model.config.intermediate_size,
            )?;
            let context = self.get_or_create_shared_context()?;
            let runner = CachedGpuMlpBlockRunner::new_with_context(
                context,
                self.model.config.hidden_size,
                self.model.config.intermediate_size,
                self.model.config.rms_norm_eps as f32,
                &pair_spec,
                &down_spec,
            )
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            let compile_duration = runner.compile_duration();
            self.mlp_blocks.insert(layer_idx, runner);
            compile_duration
        };
        Ok((
            compile_duration,
            self.mlp_blocks
                .get_mut(&layer_idx)
                .expect("mlp block runner should exist after creation"),
        ))
    }

    fn ensure_tail_block(
        &mut self,
    ) -> Result<(Duration, CachedTailBlockGpuRunner), ReferenceError> {
        if let Some(runner) = self.tail_block.as_ref().cloned() {
            return Ok((Duration::ZERO, runner));
        }
        let (runner, compile_duration, _gpu_cache_hit) = self.model.get_or_create_tail_block_gpu()?;
        self.tail_block = Some(runner.clone());
        Ok((compile_duration, runner))
    }

    fn ensure_full_last_layer_block(
        &mut self,
    ) -> Result<(Duration, CachedFullLastLayerGpuRunner), ReferenceError> {
        if let Some(runner) = self.full_last_layer_block.as_ref().cloned() {
            return Ok((Duration::ZERO, runner));
        }
        let (runner, compile_duration, _gpu_cache_hit) =
            self.model.get_or_create_full_last_layer_block_gpu()?;
        self.full_last_layer_block = Some(runner.clone());
        Ok((compile_duration, runner))
    }

    fn prewarm_decode_path(
        &mut self,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<(), ReferenceError> {
        let _ = self.get_or_create_shared_context()?;

        if ReferenceModel::packed_use_gpu_embedding()
            || ReferenceModel::packed_use_gpu_attention_block()
            || ReferenceModel::packed_use_gpu_tail()
            || ReferenceModel::packed_use_gpu_full_last_layer()
            || ReferenceModel::packed_use_gpu_swiglu_block()
        {
            let _ = self.ensure_embedding_lookup_runner()?;
            let _ = self.ensure_input_norm_runner()?;
        }

        if use_attention_qkv
            && (ReferenceModel::packed_use_gpu_attention_block()
                || ReferenceModel::packed_use_gpu_tail()
                || ReferenceModel::packed_use_gpu_full_last_layer())
        {
            let _ = self.ensure_qk_rope_runner()?;
            for layer_idx in 0..self.model.config.num_hidden_layers {
                let _ = self.ensure_gpu_kv_cache(layer_idx)?;
                let _ = self.ensure_attention_block(layer_idx, 1)?;
            }
        }

        if use_mlp_gu
            && (ReferenceModel::packed_use_gpu_swiglu_block()
                || ReferenceModel::packed_use_gpu_tail()
                || ReferenceModel::packed_use_gpu_full_last_layer())
        {
            for layer_idx in 0..self.model.config.num_hidden_layers {
                let _ = self.ensure_mlp_block(layer_idx)?;
            }
        }

        if ReferenceModel::packed_use_gpu_tail() {
            let _ = self.ensure_tail_block()?;
        }

        if ReferenceModel::packed_use_gpu_full_last_layer() {
            let last_layer_idx = self.model.config.num_hidden_layers - 1;
            let _ = self.ensure_attention_block(last_layer_idx, 1)?;
            let _ = self.ensure_full_last_layer_block()?;
        }

        Ok(())
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn run_attention_mlp_layer_to_host(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        q: Option<&[f32]>,
        query_resident: Option<&GpuResidentBuffer>,
        keys: &[f32],
        values: &[f32],
        residual: &[f32],
        post_norm_weight: &[f32],
    ) -> Result<
        (
            Vec<f32>,
            GpuAttentionBlockReport,
            GpuMlpBlockReport,
            Duration,
            Duration,
        ),
        ReferenceError,
    > {
        let kv_state = self.gpu_kv_state(layer_idx);
        let kv_tensors = self.gpu_kv_tensors(layer_idx);
        let (attention_compile_duration, attention_block) =
            self.ensure_attention_block(layer_idx, seq_len)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            attention_block
                .run_with_resident_query_and_kv(query_resident, key_tensor, value_tensor, residual)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else if let Some((
            key_buffer,
            key_len,
            key_buffer_size,
            value_buffer,
            value_len,
            value_buffer_size,
        )) = kv_state
        {
            attention_block
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    key_buffer,
                    key_len,
                    key_buffer_size,
                    value_buffer,
                    value_len,
                    value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let attention_output = attention_block.resident_output();

        let (mlp_compile_duration, mlp_block) = self.ensure_mlp_block(layer_idx)?;
        let mlp_report = mlp_block
            .run_from_resident_tensor(&attention_output, post_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (hidden, download_duration) = mlp_block
            .read_output()
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;

        Ok((
            hidden,
            attention_report,
            mlp_report,
            attention_compile_duration + mlp_compile_duration,
            download_duration,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention_mlp_layer_resident(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        q: Option<&[f32]>,
        query_resident: Option<&GpuResidentBuffer>,
        residual_resident: Option<&GpuResidentBuffer>,
        keys: &[f32],
        values: &[f32],
        residual: &[f32],
        post_norm_weight: &[f32],
    ) -> Result<(GpuAttentionBlockReport, GpuMlpBlockReport, Duration), ReferenceError> {
        let kv_state = self.gpu_kv_state(layer_idx);
        let kv_tensors = self.gpu_kv_tensors(layer_idx);
        let (attention_compile_duration, attention_block) =
            self.ensure_attention_block(layer_idx, seq_len)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            if let Some(residual_resident) = residual_resident {
                attention_block
                    .run_with_resident_query_kv_and_residual(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual_resident,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            } else {
                attention_block
                    .run_with_resident_query_and_kv(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            }
        } else if let Some((
            key_buffer,
            key_len,
            key_buffer_size,
            value_buffer,
            value_len,
            value_buffer_size,
        )) = kv_state
        {
            attention_block
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    key_buffer,
                    key_len,
                    key_buffer_size,
                    value_buffer,
                    value_len,
                    value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let attention_output = attention_block.resident_output();

        let (mlp_compile_duration, mlp_block) = self.ensure_mlp_block(layer_idx)?;
        let mlp_report = mlp_block
            .run_from_resident_tensor(&attention_output, post_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;

        Ok((
            attention_report,
            mlp_report,
            attention_compile_duration + mlp_compile_duration,
        ))
    }

    fn run_attention_layer_to_host(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        q: Option<&[f32]>,
        query_resident: Option<&GpuResidentBuffer>,
        keys: &[f32],
        values: &[f32],
        residual: &[f32],
    ) -> Result<(Vec<f32>, GpuAttentionBlockReport, Duration, Duration), ReferenceError> {
        let kv_state = self.gpu_kv_state(layer_idx);
        let kv_tensors = self.gpu_kv_tensors(layer_idx);
        let (attention_compile_duration, attention_block) =
            self.ensure_attention_block(layer_idx, seq_len)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            attention_block
                .run_with_resident_query_and_kv(query_resident, key_tensor, value_tensor, residual)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else if let Some((
            key_buffer,
            key_len,
            key_buffer_size,
            value_buffer,
            value_len,
            value_buffer_size,
        )) = kv_state
        {
            attention_block
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    key_buffer,
                    key_len,
                    key_buffer_size,
                    value_buffer,
                    value_len,
                    value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let (hidden, download_duration) = attention_block
            .read_output()
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((
            hidden,
            attention_report,
            attention_compile_duration,
            download_duration,
        ))
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn run_attention_layer_resident(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        q: Option<&[f32]>,
        query_resident: Option<&GpuResidentBuffer>,
        keys: &[f32],
        values: &[f32],
        residual: &[f32],
    ) -> Result<(GpuAttentionBlockReport, Duration), ReferenceError> {
        let kv_state = self.gpu_kv_state(layer_idx);
        let kv_tensors = self.gpu_kv_tensors(layer_idx);
        let (attention_compile_duration, attention_block) =
            self.ensure_attention_block(layer_idx, seq_len)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            attention_block
                .run_with_resident_query_and_kv(query_resident, key_tensor, value_tensor, residual)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else if let Some((
            key_buffer,
            key_len,
            key_buffer_size,
            value_buffer,
            value_len,
            value_buffer_size,
        )) = kv_state
        {
            attention_block
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    key_buffer,
                    key_len,
                    key_buffer_size,
                    value_buffer,
                    value_len,
                    value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        Ok((attention_report, attention_compile_duration))
    }

    fn run_attention_layer_to_tail_argmax(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        q: Option<&[f32]>,
        query_resident: Option<&GpuResidentBuffer>,
        residual_resident: Option<&GpuResidentBuffer>,
        keys: &[f32],
        values: &[f32],
        residual: &[f32],
        final_norm_weight: &[f32],
    ) -> Result<(usize, GpuAttentionBlockReport, GpuTailBlockReport, Duration), ReferenceError>
    {
        let kv_state = self.gpu_kv_state(layer_idx);
        let kv_tensors = self.gpu_kv_tensors(layer_idx);
        let (attention_compile_duration, attention_block) =
            self.ensure_attention_block(layer_idx, seq_len)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            if let Some(residual_resident) = residual_resident {
                attention_block
                    .run_with_resident_query_kv_and_residual(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual_resident,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            } else {
                attention_block
                    .run_with_resident_query_and_kv(query_resident, key_tensor, value_tensor, residual)
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            }
        } else if let Some((
            key_buffer,
            key_len,
            key_buffer_size,
            value_buffer,
            value_len,
            value_buffer_size,
        )) = kv_state
        {
            attention_block
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    key_buffer,
                    key_len,
                    key_buffer_size,
                    value_buffer,
                    value_len,
                    value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let attention_output = attention_block.resident_output();
        let (tail_compile_duration, tail_block) = self.ensure_tail_block()?;
        let tail_report = tail_block
            .borrow_mut()
            .run_argmax_from_resident_tensor(&attention_output, final_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((
            tail_report.argmax_index,
            attention_report,
            tail_report,
            attention_compile_duration + tail_compile_duration,
        ))
    }

    #[allow(dead_code)]
    fn run_mlp_layer_to_host(
        &mut self,
        layer_idx: usize,
        residual: &[f32],
        post_norm_weight: &[f32],
    ) -> Result<(Vec<f32>, GpuMlpBlockReport, Duration, Duration), ReferenceError> {
        let (mlp_compile_duration, mlp_block) = self.ensure_mlp_block(layer_idx)?;
        let mlp_report = mlp_block
            .run_with_host_residual(residual, post_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (hidden, download_duration) = mlp_block
            .read_output()
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((hidden, mlp_report, mlp_compile_duration, download_duration))
    }

    fn run_mlp_layer_resident(
        &mut self,
        layer_idx: usize,
        residual: &[f32],
        post_norm_weight: &[f32],
    ) -> Result<(GpuMlpBlockReport, Duration), ReferenceError> {
        let (mlp_compile_duration, mlp_block) = self.ensure_mlp_block(layer_idx)?;
        let mlp_report = mlp_block
            .run_with_host_residual(residual, post_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((mlp_report, mlp_compile_duration))
    }

    fn run_mlp_layer_to_tail_argmax(
        &mut self,
        layer_idx: usize,
        residual: &[f32],
        post_norm_weight: &[f32],
        final_norm_weight: &[f32],
    ) -> Result<(usize, GpuMlpBlockReport, GpuTailBlockReport, Duration), ReferenceError> {
        let (mlp_compile_duration, mlp_block) = self.ensure_mlp_block(layer_idx)?;
        let mlp_report = mlp_block
            .run_with_host_residual(residual, post_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let mlp_output = mlp_block.resident_output();
        let (tail_compile_duration, tail_block) = self.ensure_tail_block()?;
        let tail_report = tail_block
            .borrow_mut()
            .run_argmax_from_resident_tensor(&mlp_output, final_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((
            tail_report.argmax_index,
            mlp_report,
            tail_report,
            mlp_compile_duration + tail_compile_duration,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention_to_full_last_layer_argmax(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        q: Option<&[f32]>,
        query_resident: Option<&GpuResidentBuffer>,
        residual_resident: Option<&GpuResidentBuffer>,
        keys: &[f32],
        values: &[f32],
        residual: &[f32],
        post_norm_weight: &[f32],
        final_norm_weight: &[f32],
    ) -> Result<
        (
            usize,
            GpuAttentionBlockReport,
            GpuFullLastLayerReport,
            Duration,
        ),
        ReferenceError,
    > {
        let kv_state = self.gpu_kv_state(layer_idx);
        let kv_tensors = self.gpu_kv_tensors(layer_idx);
        let (attention_compile_duration, attention_block) =
            self.ensure_attention_block(layer_idx, seq_len)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            if let Some(residual_resident) = residual_resident {
                attention_block
                    .run_with_resident_query_kv_and_residual(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual_resident,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            } else {
                attention_block
                    .run_with_resident_query_and_kv(query_resident, key_tensor, value_tensor, residual)
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            }
        } else if let Some((
            key_buffer,
            key_len,
            key_buffer_size,
            value_buffer,
            value_len,
            value_buffer_size,
        )) = kv_state
        {
            attention_block
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    key_buffer,
                    key_len,
                    key_buffer_size,
                    value_buffer,
                    value_len,
                    value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let attention_output = attention_block.resident_output();

        let (full_last_layer_compile_duration, full_last_layer_block) =
            self.ensure_full_last_layer_block()?;
        let full_last_layer_report = full_last_layer_block
            .borrow_mut()
            .run_argmax_from_resident_tensor(&attention_output, post_norm_weight, final_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((
            full_last_layer_report.argmax_index,
            attention_report,
            full_last_layer_report,
            attention_compile_duration + full_last_layer_compile_duration,
        ))
    }

    fn run_tail_argmax_from_resident_hidden(
        &mut self,
        source: &GpuResidentBuffer,
        final_norm_weight: &[f32],
    ) -> Result<(usize, GpuTailBlockReport, Duration), ReferenceError> {
        let (compile_duration, tail_block) = self.ensure_tail_block()?;
        let report = tail_block
            .borrow_mut()
            .run_argmax_from_resident_tensor(source, final_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((report.argmax_index, report, compile_duration))
    }

    fn run_tail_argmax_from_host_hidden(
        &mut self,
        hidden_input: &[f32],
        final_norm_weight: &[f32],
    ) -> Result<(usize, GpuTailBlockReport, Duration), ReferenceError> {
        let (compile_duration, tail_block) = self.ensure_tail_block()?;
        let report = tail_block
            .borrow_mut()
            .run_argmax(hidden_input, final_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((report.argmax_index, report, compile_duration))
    }

    fn run_tail_logits_from_resident_hidden(
        &mut self,
        source: &GpuResidentBuffer,
        final_norm_weight: &[f32],
    ) -> Result<(Vec<f32>, GpuTailBlockLogitsReport, Duration), ReferenceError> {
        let (compile_duration, tail_block) = self.ensure_tail_block()?;
        let (logits, report) = tail_block
            .borrow_mut()
            .run_logits_from_resident_tensor(source, final_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((logits, report, compile_duration))
    }

    fn run_tail_logits_from_host_hidden(
        &mut self,
        hidden_input: &[f32],
        final_norm_weight: &[f32],
    ) -> Result<(Vec<f32>, GpuTailBlockLogitsReport, Duration), ReferenceError> {
        let (compile_duration, tail_block) = self.ensure_tail_block()?;
        let (logits, report) = tail_block
            .borrow_mut()
            .run_logits(hidden_input, final_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((logits, report, compile_duration))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PackedGpuSessionMetrics {
    pack_duration: Duration,
    compile_duration: Duration,
    weight_upload_duration: Duration,
    activation_upload_duration: Duration,
    upload_duration: Duration,
    gpu_duration: Duration,
    download_duration: Duration,
    pack_cache_hits: usize,
    gpu_cache_hits: usize,
    dispatch_count: usize,
    weight_upload_bytes: usize,
    activation_upload_bytes: usize,
    upload_bytes: usize,
    download_bytes: usize,
}

struct PreparedProjectionRunner {
    packed: Rc<PackedProjectionCache>,
    runner: CachedProjectionGpuRunner,
    compile_duration: Duration,
    weight_upload_duration: Duration,
    pack_cache_hit: bool,
    gpu_cache_hit: bool,
}

struct ResidentPackedProjection {
    tensor_name: String,
    operation: String,
    rows: usize,
    cols: usize,
    prepared: PreparedProjectionRunner,
    report: crate::gpu::packed_matvec::GpuPackedMatvecReport,
}

#[allow(dead_code)]
struct ResidentPackedPairProjection {
    tensor_name: String,
    first_rows: usize,
    second_rows: usize,
    cols: usize,
    activation_upload_bytes: usize,
    prepared: PreparedProjectionRunner,
    report: crate::gpu::packed_matvec::GpuPackedMatvecReport,
}

struct ResidentGpuFinalNorm {
    runner: CachedWeightedRmsNormGpuRunner,
    report: crate::gpu::weighted_rms_norm::GpuWeightedRmsNormReport,
    compile_duration: Duration,
    gpu_cache_hit: bool,
}

struct ResidentGpuVectorAdd {
    runner: CachedVectorAddGpuRunner,
    report: crate::gpu::vector_add::GpuVectorAddReport,
    compile_duration: Duration,
    gpu_cache_hit: bool,
}

#[allow(dead_code)]
enum ResidentPackedActivationKeepalive {
    PackF16(CachedPackF16PairsGpuRunner),
    SwigluPackF16(CachedSwigluPackF16PairsGpuRunner),
}

struct ResidentGpuPackedActivation {
    #[allow(dead_code)]
    keepalive: ResidentPackedActivationKeepalive,
    shared_context: Arc<SharedGpuPackedContext>,
    buffer: vk::Buffer,
    buffer_size: u64,
    packed_len: usize,
    logical_len: usize,
    upload_duration: Duration,
    gpu_duration: Duration,
    compile_duration: Duration,
    gpu_cache_hit: bool,
}

#[allow(dead_code)]
struct ResidentGpuSwigluCombined {
    runner: CachedSwigluCombinedGpuRunner,
    report: crate::gpu::swiglu_combined::GpuSwigluCombinedReport,
    compile_duration: Duration,
    gpu_cache_hit: bool,
}

struct PackedGpuSession<'a> {
    model: &'a ReferenceModel,
    metrics: PackedGpuSessionMetrics,
    dispatch_trace: Vec<PackedDispatchTrace>,
}

impl<'a> PackedGpuSession<'a> {
    fn new(model: &'a ReferenceModel) -> Self {
        Self {
            model,
            metrics: PackedGpuSessionMetrics::default(),
            dispatch_trace: Vec::new(),
        }
    }

    fn run_projection(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, ReferenceError> {
        let resident = self.run_projection_resident(tensor_name, rows, cols, input, "single")?;
        self.download_projection_output(resident)
    }

    fn run_projection_argmax(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<usize, ReferenceError> {
        let resident = self.run_projection_resident(tensor_name, rows, cols, input, "argmax")?;
        self.argmax_projection_output(resident)
    }

    fn run_projection_argmax_from_packed_activation(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        activation: ResidentGpuPackedActivation,
    ) -> Result<usize, ReferenceError> {
        let prepared = self.prepare_projection_runner(tensor_name, rows, cols)?;
        let mut report = prepared
            .runner
            .borrow_mut()
            .run_resident_from_packed_buffer(
                &activation.shared_context,
                activation.buffer,
                activation.packed_len,
                activation.buffer_size,
            )
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu packed projection chaining failed for {tensor_name}: {error}"
                ))
            })?;
        let (argmax_index, logits_download_duration) =
            prepared.runner.borrow().argmax_output().map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu projection argmax failed for {tensor_name}: {error}"
                ))
            })?;
        report.download_duration = logits_download_duration;

        self.metrics.compile_duration += activation.compile_duration;
        self.metrics.upload_duration += activation.upload_duration;
        self.metrics.gpu_duration += activation.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.gpu_cache_hits += usize::from(activation.gpu_cache_hit);
        self.metrics.activation_upload_bytes += activation.logical_len * std::mem::size_of::<f32>();
        self.metrics.upload_bytes += activation.logical_len * std::mem::size_of::<f32>();
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
            operation: "resident".to_string(),
            path: "gpu_pack".to_string(),
            stage: "pack_f16_pairs".to_string(),
            tensor_name: "pack_f16_pairs".to_string(),
            rows: cols.div_ceil(2),
            cols,
            pack_cache_hit: false,
            gpu_cache_hit: activation.gpu_cache_hit,
            cpu_ms: (activation.compile_duration + activation.upload_duration).as_secs_f64()
                * 1_000.0,
            compile_ms: activation.compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: 0.0,
            activation_upload_ms: activation.upload_duration.as_secs_f64() * 1_000.0,
            gpu_ms: activation.gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: 0.0,
            weight_upload_bytes: 0,
            activation_upload_bytes: activation.logical_len * std::mem::size_of::<f32>(),
            download_bytes: 0,
        });

        let weight_upload_bytes = usize::from(!prepared.gpu_cache_hit)
            * (prepared.packed.code_words.len() * std::mem::size_of::<u32>()
                + prepared.packed.scales.len() * std::mem::size_of::<f32>());
        let activation_upload_bytes = cols.div_ceil(2) * std::mem::size_of::<u32>();
        self.account_projection_report(
            &prepared.packed,
            cols,
            prepared.weight_upload_duration,
            prepared.gpu_cache_hit,
            &report,
            0,
        );
        self.push_dispatch_trace(
            tensor_name,
            "argmax",
            rows,
            cols,
            prepared.pack_cache_hit,
            prepared.gpu_cache_hit,
            prepared.compile_duration,
            prepared.weight_upload_duration,
            report.upload_duration,
            report.gpu_duration,
            report.download_duration,
            weight_upload_bytes,
            activation_upload_bytes,
            0,
        );
        Ok(argmax_index)
    }

    fn run_projection_resident(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        input: &[f32],
        operation: &str,
    ) -> Result<ResidentPackedProjection, ReferenceError> {
        let prepared = self.prepare_projection_runner(tensor_name, rows, cols)?;
        let report = prepared
            .runner
            .borrow_mut()
            .run_resident(input)
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu projection failed for {tensor_name}: {error}"))
            })?;
        Ok(ResidentPackedProjection {
            tensor_name: tensor_name.to_string(),
            operation: operation.to_string(),
            rows,
            cols,
            prepared,
            report,
        })
    }

    fn download_projection_output(
        &mut self,
        resident: ResidentPackedProjection,
    ) -> Result<Vec<f32>, ReferenceError> {
        let (output, download_duration) =
            resident
                .prepared
                .runner
                .borrow()
                .read_output()
                .map_err(|error| {
                    ReferenceError::Decode(format!(
                        "gpu projection output download failed for {}: {error}",
                        resident.tensor_name
                    ))
                })?;
        let mut report = resident.report;
        report.download_duration = download_duration;
        let weight_upload_bytes = usize::from(!resident.prepared.gpu_cache_hit)
            * (resident.prepared.packed.code_words.len() * std::mem::size_of::<u32>()
                + resident.prepared.packed.scales.len() * std::mem::size_of::<f32>());
        let activation_upload_bytes = resident.cols.div_ceil(2) * std::mem::size_of::<u32>();
        let download_bytes = resident.rows * std::mem::size_of::<f32>();
        self.account_projection_report(
            &resident.prepared.packed,
            resident.cols,
            resident.prepared.weight_upload_duration,
            resident.prepared.gpu_cache_hit,
            &report,
            download_bytes,
        );
        self.push_dispatch_trace(
            &resident.tensor_name,
            &resident.operation,
            resident.rows,
            resident.cols,
            resident.prepared.pack_cache_hit,
            resident.prepared.gpu_cache_hit,
            resident.prepared.compile_duration,
            resident.prepared.weight_upload_duration,
            report.upload_duration,
            report.gpu_duration,
            report.download_duration,
            weight_upload_bytes,
            activation_upload_bytes,
            download_bytes,
        );
        Ok(output)
    }

    fn argmax_projection_output(
        &mut self,
        resident: ResidentPackedProjection,
    ) -> Result<usize, ReferenceError> {
        let (argmax_index, download_duration) = resident
            .prepared
            .runner
            .borrow()
            .argmax_output()
            .map_err(|error| {
            ReferenceError::Decode(format!(
                "gpu projection argmax failed for {}: {error}",
                resident.tensor_name
            ))
        })?;
        let mut report = resident.report;
        report.download_duration = download_duration;
        let weight_upload_bytes = usize::from(!resident.prepared.gpu_cache_hit)
            * (resident.prepared.packed.code_words.len() * std::mem::size_of::<u32>()
                + resident.prepared.packed.scales.len() * std::mem::size_of::<f32>());
        let activation_upload_bytes = resident.cols.div_ceil(2) * std::mem::size_of::<u32>();
        self.account_projection_report(
            &resident.prepared.packed,
            resident.cols,
            resident.prepared.weight_upload_duration,
            resident.prepared.gpu_cache_hit,
            &report,
            0,
        );
        self.push_dispatch_trace(
            &resident.tensor_name,
            &resident.operation,
            resident.rows,
            resident.cols,
            resident.prepared.pack_cache_hit,
            resident.prepared.gpu_cache_hit,
            resident.prepared.compile_duration,
            resident.prepared.weight_upload_duration,
            report.upload_duration,
            report.gpu_duration,
            report.download_duration,
            weight_upload_bytes,
            activation_upload_bytes,
            0,
        );
        Ok(argmax_index)
    }

    fn prepare_projection_runner(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<PreparedProjectionRunner, ReferenceError> {
        let (packed, pack_duration, pack_cache_hit) =
            self.model
                .get_or_create_projection_cache(tensor_name, rows, cols)?;
        let (runner, compile_duration, weight_upload_duration, gpu_cache_hit) = self
            .model
            .get_or_create_projection_gpu(tensor_name, &packed)?;
        self.metrics.pack_duration += pack_duration;
        self.metrics.compile_duration += compile_duration;
        self.metrics.weight_upload_duration += weight_upload_duration;
        self.metrics.pack_cache_hits += usize::from(pack_cache_hit);
        self.metrics.gpu_cache_hits += usize::from(gpu_cache_hit);
        Ok(PreparedProjectionRunner {
            packed,
            runner,
            compile_duration,
            weight_upload_duration,
            pack_cache_hit,
            gpu_cache_hit,
        })
    }

    fn run_final_norm_resident(
        &mut self,
        input: &[f32],
        weight: &[f32],
    ) -> Result<ResidentGpuFinalNorm, ReferenceError> {
        let (runner, compile_duration, gpu_cache_hit) =
            self.model.get_or_create_final_norm_gpu()?;
        let report = runner
            .borrow_mut()
            .run_resident(input, weight)
            .map_err(|error| ReferenceError::Decode(format!("gpu final norm failed: {error}")))?;
        Ok(ResidentGpuFinalNorm {
            runner,
            report,
            compile_duration,
            gpu_cache_hit,
        })
    }

    fn run_pack_f16_pairs_resident(
        &mut self,
        final_norm: ResidentGpuFinalNorm,
    ) -> Result<ResidentGpuPackedActivation, ReferenceError> {
        let (runner, compile_duration, gpu_cache_hit) = self
            .model
            .get_or_create_pack_f16_pairs_gpu(final_norm.runner.borrow().len())?;
        let report = runner
            .borrow_mut()
            .run_resident_from_f32_buffer(
                final_norm.runner.borrow().shared_context(),
                final_norm.runner.borrow().output_buffer_handle(),
                final_norm.runner.borrow().len(),
                (final_norm.runner.borrow().len() * std::mem::size_of::<f32>()) as u64,
            )
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu pack f16 pairs failed: {error}"))
            })?;
        self.metrics.compile_duration += final_norm.compile_duration;
        self.metrics.weight_upload_duration += final_norm.report.upload_duration;
        self.metrics.upload_duration += final_norm.report.upload_duration;
        self.metrics.gpu_duration += final_norm.report.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.weight_upload_bytes +=
            final_norm.runner.borrow().len() * std::mem::size_of::<f32>();
        self.metrics.activation_upload_bytes +=
            final_norm.runner.borrow().len() * std::mem::size_of::<f32>();
        self.metrics.upload_bytes +=
            2 * final_norm.runner.borrow().len() * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(final_norm.gpu_cache_hit);
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
            operation: "resident".to_string(),
            path: "gpu_dense".to_string(),
            stage: "final_norm_gpu".to_string(),
            tensor_name: "model.norm.weight".to_string(),
            rows: final_norm.runner.borrow().len(),
            cols: final_norm.runner.borrow().len(),
            pack_cache_hit: false,
            gpu_cache_hit: final_norm.gpu_cache_hit,
            cpu_ms: (final_norm.compile_duration + final_norm.report.upload_duration).as_secs_f64()
                * 1_000.0,
            compile_ms: final_norm.compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: final_norm.report.upload_duration.as_secs_f64() * 1_000.0,
            activation_upload_ms: 0.0,
            gpu_ms: final_norm.report.gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: 0.0,
            weight_upload_bytes: final_norm.runner.borrow().len() * std::mem::size_of::<f32>(),
            activation_upload_bytes: final_norm.runner.borrow().len() * std::mem::size_of::<f32>(),
            download_bytes: 0,
        });
        Ok(ResidentGpuPackedActivation {
            keepalive: ResidentPackedActivationKeepalive::PackF16(runner.clone()),
            shared_context: runner.borrow().shared_context().clone(),
            buffer: runner.borrow().output_buffer_handle(),
            buffer_size: runner.borrow().output_buffer_size(),
            packed_len: runner.borrow().packed_len(),
            logical_len: final_norm.runner.borrow().len(),
            upload_duration: report.upload_duration,
            gpu_duration: report.gpu_duration,
            compile_duration,
            gpu_cache_hit,
        })
    }

    #[allow(dead_code)]
    fn run_swiglu_combined_resident(
        &mut self,
        pair: ResidentPackedPairProjection,
    ) -> Result<ResidentGpuSwigluCombined, ReferenceError> {
        let weight_upload_bytes = usize::from(!pair.prepared.gpu_cache_hit)
            * (pair.prepared.packed.code_words.len() * std::mem::size_of::<u32>()
                + pair.prepared.packed.scales.len() * std::mem::size_of::<f32>());
        self.account_projection_report(
            &pair.prepared.packed,
            pair.cols,
            pair.prepared.weight_upload_duration,
            pair.prepared.gpu_cache_hit,
            &pair.report,
            0,
        );
        self.push_dispatch_trace(
            &pair.tensor_name,
            "pair_resident",
            pair.first_rows + pair.second_rows,
            pair.cols,
            pair.prepared.pack_cache_hit,
            pair.prepared.gpu_cache_hit,
            pair.prepared.compile_duration,
            pair.prepared.weight_upload_duration,
            pair.report.upload_duration,
            pair.report.gpu_duration,
            pair.report.download_duration,
            weight_upload_bytes,
            pair.activation_upload_bytes,
            0,
        );

        let (runner, compile_duration, gpu_cache_hit) =
            self.model.get_or_create_swiglu_combined_gpu()?;
        let report = runner
            .borrow_mut()
            .run_resident_from_f32_buffer(
                pair.prepared.runner.borrow().shared_context(),
                pair.prepared.runner.borrow().output_buffer_handle(),
                pair.first_rows + pair.second_rows,
                pair.prepared.runner.borrow().output_buffer_size(),
            )
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu combined swiglu failed: {error}"))
            })?;
        self.metrics.compile_duration += compile_duration;
        self.metrics.upload_duration += report.upload_duration;
        self.metrics.gpu_duration += report.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.activation_upload_bytes +=
            (pair.first_rows + pair.second_rows) * std::mem::size_of::<f32>();
        self.metrics.upload_bytes +=
            (pair.first_rows + pair.second_rows) * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(gpu_cache_hit);
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
            operation: "resident".to_string(),
            path: "gpu_dense".to_string(),
            stage: "mlp_swiglu_gpu".to_string(),
            tensor_name: "swiglu_combined".to_string(),
            rows: pair.first_rows,
            cols: pair.first_rows + pair.second_rows,
            pack_cache_hit: false,
            gpu_cache_hit,
            cpu_ms: (compile_duration + report.upload_duration).as_secs_f64() * 1_000.0,
            compile_ms: compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: 0.0,
            activation_upload_ms: report.upload_duration.as_secs_f64() * 1_000.0,
            gpu_ms: report.gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: 0.0,
            weight_upload_bytes: 0,
            activation_upload_bytes: (pair.first_rows + pair.second_rows)
                * std::mem::size_of::<f32>(),
            download_bytes: 0,
        });
        Ok(ResidentGpuSwigluCombined {
            runner,
            report,
            compile_duration,
            gpu_cache_hit,
        })
    }

    fn run_swiglu_pack_f16_pairs_resident(
        &mut self,
        pair: ResidentPackedPairProjection,
    ) -> Result<ResidentGpuPackedActivation, ReferenceError> {
        let weight_upload_bytes = usize::from(!pair.prepared.gpu_cache_hit)
            * (pair.prepared.packed.code_words.len() * std::mem::size_of::<u32>()
                + pair.prepared.packed.scales.len() * std::mem::size_of::<f32>());
        self.account_projection_report(
            &pair.prepared.packed,
            pair.cols,
            pair.prepared.weight_upload_duration,
            pair.prepared.gpu_cache_hit,
            &pair.report,
            0,
        );
        self.push_dispatch_trace(
            &pair.tensor_name,
            "pair_resident",
            pair.first_rows + pair.second_rows,
            pair.cols,
            pair.prepared.pack_cache_hit,
            pair.prepared.gpu_cache_hit,
            pair.prepared.compile_duration,
            pair.prepared.weight_upload_duration,
            pair.report.upload_duration,
            pair.report.gpu_duration,
            pair.report.download_duration,
            weight_upload_bytes,
            pair.activation_upload_bytes,
            0,
        );

        let (runner, compile_duration, gpu_cache_hit) =
            self.model.get_or_create_swiglu_pack_f16_pairs_gpu()?;
        let report = runner
            .borrow_mut()
            .run_with_output_from_buffer(
                pair.prepared.runner.borrow().shared_context(),
                pair.prepared.runner.borrow().output_buffer_handle(),
                pair.first_rows + pair.second_rows,
                pair.prepared.runner.borrow().output_buffer_size(),
            )
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu fused swiglu pack failed: {error}"))
            })?;
        self.metrics.compile_duration += compile_duration;
        self.metrics.upload_duration += report.upload_duration;
        self.metrics.gpu_duration += report.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.activation_upload_bytes +=
            (pair.first_rows + pair.second_rows) * std::mem::size_of::<f32>();
        self.metrics.upload_bytes +=
            (pair.first_rows + pair.second_rows) * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(gpu_cache_hit);
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
            operation: "resident".to_string(),
            path: "gpu_pack".to_string(),
            stage: "mlp_swiglu_pack_gpu".to_string(),
            tensor_name: "swiglu_pack_f16_pairs".to_string(),
            rows: pair.first_rows.div_ceil(2),
            cols: pair.first_rows + pair.second_rows,
            pack_cache_hit: false,
            gpu_cache_hit,
            cpu_ms: (compile_duration + report.upload_duration).as_secs_f64() * 1_000.0,
            compile_ms: compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: 0.0,
            activation_upload_ms: report.upload_duration.as_secs_f64() * 1_000.0,
            gpu_ms: report.gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: 0.0,
            weight_upload_bytes: 0,
            activation_upload_bytes: (pair.first_rows + pair.second_rows)
                * std::mem::size_of::<f32>(),
            download_bytes: 0,
        });
        Ok(ResidentGpuPackedActivation {
            keepalive: ResidentPackedActivationKeepalive::SwigluPackF16(runner.clone()),
            shared_context: runner.borrow().shared_context().clone(),
            buffer: runner.borrow().output_buffer_handle(),
            buffer_size: runner.borrow().output_buffer_size(),
            packed_len: runner.borrow().packed_len(),
            logical_len: pair.first_rows,
            upload_duration: report.upload_duration,
            gpu_duration: report.gpu_duration,
            compile_duration,
            gpu_cache_hit,
        })
    }

    #[allow(dead_code)]
    fn run_pack_f16_pairs_from_swiglu(
        &mut self,
        swiglu: ResidentGpuSwigluCombined,
    ) -> Result<ResidentGpuPackedActivation, ReferenceError> {
        let (runner, compile_duration, gpu_cache_hit) = self
            .model
            .get_or_create_pack_f16_pairs_gpu(swiglu.runner.borrow().len())?;
        let report = runner
            .borrow_mut()
            .run_resident_from_f32_buffer(
                swiglu.runner.borrow().shared_context(),
                swiglu.runner.borrow().output_buffer_handle(),
                swiglu.runner.borrow().len(),
                swiglu.runner.borrow().output_buffer_size(),
            )
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu pack f16 pairs failed: {error}"))
            })?;
        self.metrics.compile_duration += compile_duration;
        self.metrics.upload_duration += report.upload_duration;
        self.metrics.gpu_duration += report.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.activation_upload_bytes +=
            swiglu.runner.borrow().len() * std::mem::size_of::<f32>();
        self.metrics.upload_bytes += swiglu.runner.borrow().len() * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(gpu_cache_hit);
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
            operation: "resident".to_string(),
            path: "gpu_pack".to_string(),
            stage: "pack_f16_pairs".to_string(),
            tensor_name: "pack_f16_pairs".to_string(),
            rows: swiglu.runner.borrow().len().div_ceil(2),
            cols: swiglu.runner.borrow().len(),
            pack_cache_hit: false,
            gpu_cache_hit,
            cpu_ms: (compile_duration + report.upload_duration).as_secs_f64() * 1_000.0,
            compile_ms: compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: 0.0,
            activation_upload_ms: report.upload_duration.as_secs_f64() * 1_000.0,
            gpu_ms: report.gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: 0.0,
            weight_upload_bytes: 0,
            activation_upload_bytes: swiglu.runner.borrow().len() * std::mem::size_of::<f32>(),
            download_bytes: 0,
        });
        Ok(ResidentGpuPackedActivation {
            keepalive: ResidentPackedActivationKeepalive::PackF16(runner.clone()),
            shared_context: runner.borrow().shared_context().clone(),
            buffer: runner.borrow().output_buffer_handle(),
            buffer_size: runner.borrow().output_buffer_size(),
            packed_len: runner.borrow().packed_len(),
            logical_len: swiglu.runner.borrow().len(),
            upload_duration: report.upload_duration,
            gpu_duration: report.gpu_duration,
            compile_duration,
            gpu_cache_hit,
        })
    }

    fn run_projection_resident_from_packed_activation(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        activation: ResidentGpuPackedActivation,
    ) -> Result<ResidentPackedProjection, ReferenceError> {
        let prepared = self.prepare_projection_runner(tensor_name, rows, cols)?;
        let report = prepared
            .runner
            .borrow_mut()
            .run_resident_from_packed_buffer(
                &activation.shared_context,
                activation.buffer,
                activation.packed_len,
                activation.buffer_size,
            )
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu packed projection chaining failed for {tensor_name}: {error}"
                ))
            })?;
        self.metrics.compile_duration += activation.compile_duration;
        self.metrics.upload_duration += activation.upload_duration;
        self.metrics.gpu_duration += activation.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.gpu_cache_hits += usize::from(activation.gpu_cache_hit);
        self.metrics.activation_upload_bytes += activation.logical_len * std::mem::size_of::<f32>();
        self.metrics.upload_bytes += activation.logical_len * std::mem::size_of::<f32>();
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
            operation: "resident".to_string(),
            path: "gpu_pack".to_string(),
            stage: "pack_f16_pairs".to_string(),
            tensor_name: "pack_f16_pairs".to_string(),
            rows: cols.div_ceil(2),
            cols,
            pack_cache_hit: false,
            gpu_cache_hit: activation.gpu_cache_hit,
            cpu_ms: (activation.compile_duration + activation.upload_duration).as_secs_f64()
                * 1_000.0,
            compile_ms: activation.compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: 0.0,
            activation_upload_ms: activation.upload_duration.as_secs_f64() * 1_000.0,
            gpu_ms: activation.gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: 0.0,
            weight_upload_bytes: 0,
            activation_upload_bytes: activation.logical_len * std::mem::size_of::<f32>(),
            download_bytes: 0,
        });
        Ok(ResidentPackedProjection {
            tensor_name: tensor_name.to_string(),
            operation: "single_resident".to_string(),
            rows,
            cols,
            prepared,
            report,
        })
    }

    fn run_vector_add_resident(
        &mut self,
        left: ResidentPackedProjection,
        right: &[f32],
    ) -> Result<ResidentGpuVectorAdd, ReferenceError> {
        let weight_upload_bytes = usize::from(!left.prepared.gpu_cache_hit)
            * (left.prepared.packed.code_words.len() * std::mem::size_of::<u32>()
                + left.prepared.packed.scales.len() * std::mem::size_of::<f32>());
        let activation_upload_bytes = left.cols.div_ceil(2) * std::mem::size_of::<u32>();
        self.account_projection_report(
            &left.prepared.packed,
            left.cols,
            left.prepared.weight_upload_duration,
            left.prepared.gpu_cache_hit,
            &left.report,
            0,
        );
        self.push_dispatch_trace(
            &left.tensor_name,
            &left.operation,
            left.rows,
            left.cols,
            left.prepared.pack_cache_hit,
            left.prepared.gpu_cache_hit,
            left.prepared.compile_duration,
            left.prepared.weight_upload_duration,
            left.report.upload_duration,
            left.report.gpu_duration,
            left.report.download_duration,
            weight_upload_bytes,
            activation_upload_bytes,
            0,
        );

        let (runner, compile_duration, gpu_cache_hit) =
            self.model.get_or_create_vector_add_gpu(left.rows)?;
        let report = runner
            .borrow_mut()
            .run_resident_from_left_buffer_and_host(
                left.prepared.runner.borrow().shared_context(),
                left.prepared.runner.borrow().output_buffer_handle(),
                left.rows,
                left.prepared.runner.borrow().output_buffer_size(),
                right,
            )
            .map_err(|error| ReferenceError::Decode(format!("gpu vector add failed: {error}")))?;
        Ok(ResidentGpuVectorAdd {
            runner,
            report,
            compile_duration,
            gpu_cache_hit,
        })
    }

    fn run_final_norm_resident_from_vector_add(
        &mut self,
        activation: ResidentGpuVectorAdd,
        weight: &[f32],
    ) -> Result<ResidentGpuFinalNorm, ReferenceError> {
        self.metrics.compile_duration += activation.compile_duration;
        self.metrics.upload_duration += activation.report.upload_duration;
        self.metrics.gpu_duration += activation.report.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.activation_upload_bytes +=
            2 * activation.runner.borrow().len() * std::mem::size_of::<f32>();
        self.metrics.upload_bytes +=
            2 * activation.runner.borrow().len() * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(activation.gpu_cache_hit);
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
            operation: "resident".to_string(),
            path: "gpu_dense".to_string(),
            stage: "vector_add_gpu".to_string(),
            tensor_name: "vector_add".to_string(),
            rows: activation.runner.borrow().len(),
            cols: activation.runner.borrow().len(),
            pack_cache_hit: false,
            gpu_cache_hit: activation.gpu_cache_hit,
            cpu_ms: (activation.compile_duration + activation.report.upload_duration).as_secs_f64()
                * 1_000.0,
            compile_ms: activation.compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: 0.0,
            activation_upload_ms: activation.report.upload_duration.as_secs_f64() * 1_000.0,
            gpu_ms: activation.report.gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: 0.0,
            weight_upload_bytes: 0,
            activation_upload_bytes: 2
                * activation.runner.borrow().len()
                * std::mem::size_of::<f32>(),
            download_bytes: 0,
        });

        let (runner, compile_duration, gpu_cache_hit) =
            self.model.get_or_create_final_norm_gpu()?;
        let report = runner
            .borrow_mut()
            .run_resident_from_f32_buffer(
                activation.runner.borrow().shared_context(),
                activation.runner.borrow().output_buffer_handle(),
                activation.runner.borrow().len(),
                activation.runner.borrow().output_buffer_size(),
                weight,
            )
            .map_err(|error| ReferenceError::Decode(format!("gpu final norm failed: {error}")))?;
        Ok(ResidentGpuFinalNorm {
            runner,
            report,
            compile_duration,
            gpu_cache_hit,
        })
    }

    fn run_projection_from_packed_activation(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        activation: ResidentGpuPackedActivation,
    ) -> Result<Vec<f32>, ReferenceError> {
        let prepared = self.prepare_projection_runner(tensor_name, rows, cols)?;
        let mut report = prepared
            .runner
            .borrow_mut()
            .run_resident_from_packed_buffer(
                &activation.shared_context,
                activation.buffer,
                activation.packed_len,
                activation.buffer_size,
            )
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu packed projection chaining failed for {tensor_name}: {error}"
                ))
            })?;
        let (output, download_duration) =
            prepared.runner.borrow().read_output().map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu projection output download failed for {tensor_name}: {error}"
                ))
            })?;
        report.download_duration = download_duration;
        let weight_upload_bytes = usize::from(!prepared.gpu_cache_hit)
            * (prepared.packed.code_words.len() * std::mem::size_of::<u32>()
                + prepared.packed.scales.len() * std::mem::size_of::<f32>());
        let activation_upload_bytes = cols.div_ceil(2) * std::mem::size_of::<u32>();
        self.account_projection_report(
            &prepared.packed,
            cols,
            prepared.weight_upload_duration,
            prepared.gpu_cache_hit,
            &report,
            rows * std::mem::size_of::<f32>(),
        );
        self.push_dispatch_trace(
            tensor_name,
            "single",
            rows,
            cols,
            prepared.pack_cache_hit,
            prepared.gpu_cache_hit,
            prepared.compile_duration,
            prepared.weight_upload_duration,
            report.upload_duration,
            report.gpu_duration,
            report.download_duration,
            weight_upload_bytes,
            activation_upload_bytes,
            rows * std::mem::size_of::<f32>(),
        );
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn push_dispatch_trace(
        &mut self,
        tensor_name: &str,
        operation: &str,
        rows: usize,
        cols: usize,
        pack_cache_hit: bool,
        gpu_cache_hit: bool,
        compile_duration: Duration,
        weight_upload_duration: Duration,
        activation_upload_duration: Duration,
        gpu_duration: Duration,
        download_duration: Duration,
        weight_upload_bytes: usize,
        activation_upload_bytes: usize,
        download_bytes: usize,
    ) {
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
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
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
        });
    }

    fn push_dense_stage_trace(&mut self, stage: &str, tensor_name: &str, cpu_duration: Duration) {
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
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
        });
    }

    fn account_projection_report(
        &mut self,
        packed: &PackedProjectionCache,
        cols: usize,
        weight_upload_duration: Duration,
        gpu_cache_hit: bool,
        report: &crate::gpu::packed_matvec::GpuPackedMatvecReport,
        download_bytes: usize,
    ) {
        self.metrics.activation_upload_duration += report.upload_duration;
        self.metrics.upload_duration += weight_upload_duration + report.upload_duration;
        self.metrics.gpu_duration += report.gpu_duration;
        self.metrics.download_duration += report.download_duration;
        self.metrics.dispatch_count += 1;
        let weight_bytes = packed.code_words.len() * std::mem::size_of::<u32>()
            + packed.scales.len() * std::mem::size_of::<f32>();
        let activation_upload_bytes = cols.div_ceil(2) * std::mem::size_of::<u32>();
        if !gpu_cache_hit {
            self.metrics.weight_upload_bytes += weight_bytes;
        }
        self.metrics.activation_upload_bytes += activation_upload_bytes;
        self.metrics.upload_bytes +=
            usize::from(!gpu_cache_hit) * weight_bytes + activation_upload_bytes;
        self.metrics.download_bytes += download_bytes;
    }

    fn run_projection_pair(
        &mut self,
        first_name: &str,
        first_rows: usize,
        second_name: &str,
        second_rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), ReferenceError> {
        let pair_key = format!("concat::{first_name}||{second_name}");
        let (packed, pack_duration, pack_cache_hit) =
            self.model.get_or_create_projection_pair_cache(
                &pair_key,
                first_name,
                first_rows,
                second_name,
                second_rows,
                cols,
            )?;
        let (runner, compile_duration, weight_upload_duration, gpu_cache_hit) = self
            .model
            .get_or_create_projection_gpu(&pair_key, &packed)?;
        self.metrics.pack_duration += pack_duration;
        self.metrics.compile_duration += compile_duration;
        self.metrics.weight_upload_duration += weight_upload_duration;
        self.metrics.pack_cache_hits += usize::from(pack_cache_hit);
        self.metrics.gpu_cache_hits += usize::from(gpu_cache_hit);

        let (combined, report) =
            runner
                .borrow_mut()
                .run_with_output(input, None)
                .map_err(|error| {
                    ReferenceError::Decode(format!(
                        "gpu projection failed for {first_name}+{second_name}: {error}"
                    ))
                })?;
        self.metrics.activation_upload_duration += report.upload_duration;
        self.metrics.upload_duration += weight_upload_duration + report.upload_duration;
        self.metrics.gpu_duration += report.gpu_duration;
        self.metrics.download_duration += report.download_duration;
        self.metrics.dispatch_count += 1;
        let weight_bytes = packed.code_words.len() * std::mem::size_of::<u32>()
            + packed.scales.len() * std::mem::size_of::<f32>();
        let activation_upload_bytes = cols.div_ceil(2) * std::mem::size_of::<u32>();
        if !gpu_cache_hit {
            self.metrics.weight_upload_bytes += weight_bytes;
        }
        self.metrics.activation_upload_bytes += activation_upload_bytes;
        self.metrics.upload_bytes +=
            usize::from(!gpu_cache_hit) * weight_bytes + activation_upload_bytes;
        self.metrics.download_bytes += (first_rows + second_rows) * std::mem::size_of::<f32>();

        let (first, second) = combined.split_at(first_rows);
        let weight_upload_bytes = usize::from(!gpu_cache_hit)
            * (packed.code_words.len() * std::mem::size_of::<u32>()
                + packed.scales.len() * std::mem::size_of::<f32>());
        let activation_upload_bytes = cols.div_ceil(2) * std::mem::size_of::<u32>();
        let download_bytes = (first_rows + second_rows) * std::mem::size_of::<f32>();
        self.push_dispatch_trace(
            &pair_key,
            "pair",
            first_rows + second_rows,
            cols,
            pack_cache_hit,
            gpu_cache_hit,
            compile_duration,
            weight_upload_duration,
            report.upload_duration,
            report.gpu_duration,
            report.download_duration,
            weight_upload_bytes,
            activation_upload_bytes,
            download_bytes,
        );
        Ok((first.to_vec(), second.to_vec()))
    }

    #[allow(dead_code)]
    fn run_projection_pair_resident(
        &mut self,
        first_name: &str,
        first_rows: usize,
        second_name: &str,
        second_rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<ResidentPackedPairProjection, ReferenceError> {
        let pair_key = format!("concat::{first_name}||{second_name}");
        let (packed, pack_duration, pack_cache_hit) =
            self.model.get_or_create_projection_pair_cache(
                &pair_key,
                first_name,
                first_rows,
                second_name,
                second_rows,
                cols,
            )?;
        let (runner, compile_duration, weight_upload_duration, gpu_cache_hit) = self
            .model
            .get_or_create_projection_gpu(&pair_key, &packed)?;
        self.metrics.pack_duration += pack_duration;
        self.metrics.compile_duration += compile_duration;
        self.metrics.weight_upload_duration += weight_upload_duration;
        self.metrics.pack_cache_hits += usize::from(pack_cache_hit);
        self.metrics.gpu_cache_hits += usize::from(gpu_cache_hit);
        let report = runner.borrow_mut().run_resident(input).map_err(|error| {
            ReferenceError::Decode(format!(
                "gpu projection failed for {first_name}+{second_name}: {error}"
            ))
        })?;
        Ok(ResidentPackedPairProjection {
            tensor_name: pair_key,
            first_rows,
            second_rows,
            cols,
            activation_upload_bytes: cols.div_ceil(2) * std::mem::size_of::<u32>(),
            prepared: PreparedProjectionRunner {
                packed,
                runner,
                compile_duration,
                weight_upload_duration,
                pack_cache_hit,
                gpu_cache_hit,
            },
            report,
        })
    }

    fn run_projection_pair_resident_from_final_norm(
        &mut self,
        first_name: &str,
        first_rows: usize,
        second_name: &str,
        second_rows: usize,
        cols: usize,
        final_norm: ResidentGpuFinalNorm,
    ) -> Result<ResidentPackedPairProjection, ReferenceError> {
        let len = final_norm.runner.borrow().len();
        self.metrics.compile_duration += final_norm.compile_duration;
        self.metrics.upload_duration += final_norm.report.upload_duration;
        self.metrics.gpu_duration += final_norm.report.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.activation_upload_bytes += len * std::mem::size_of::<f32>();
        self.metrics.weight_upload_bytes += len * std::mem::size_of::<f32>();
        self.metrics.upload_bytes += 2 * len * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(final_norm.gpu_cache_hit);
        self.dispatch_trace.push(PackedDispatchTrace {
            index: self.dispatch_trace.len() + 1,
            operation: "resident".to_string(),
            path: "gpu_dense".to_string(),
            stage: "post_attention_norm_gpu".to_string(),
            tensor_name: "post_attention_layernorm.weight".to_string(),
            rows: len,
            cols: len,
            pack_cache_hit: false,
            gpu_cache_hit: final_norm.gpu_cache_hit,
            cpu_ms: (final_norm.compile_duration + final_norm.report.upload_duration).as_secs_f64()
                * 1_000.0,
            compile_ms: final_norm.compile_duration.as_secs_f64() * 1_000.0,
            weight_upload_ms: final_norm.report.upload_duration.as_secs_f64() * 1_000.0,
            activation_upload_ms: 0.0,
            gpu_ms: final_norm.report.gpu_duration.as_secs_f64() * 1_000.0,
            download_ms: 0.0,
            weight_upload_bytes: len * std::mem::size_of::<f32>(),
            activation_upload_bytes: len * std::mem::size_of::<f32>(),
            download_bytes: 0,
        });

        let pair_key = format!("concat::{first_name}||{second_name}");
        let (packed, pack_duration, pack_cache_hit) =
            self.model.get_or_create_projection_pair_cache(
                &pair_key,
                first_name,
                first_rows,
                second_name,
                second_rows,
                cols,
            )?;
        let (runner, compile_duration, weight_upload_duration, gpu_cache_hit) = self
            .model
            .get_or_create_projection_gpu_raw_f32(&pair_key, &packed)?;
        self.metrics.pack_duration += pack_duration;
        self.metrics.compile_duration += compile_duration;
        self.metrics.weight_upload_duration += weight_upload_duration;
        self.metrics.pack_cache_hits += usize::from(pack_cache_hit);
        self.metrics.gpu_cache_hits += usize::from(gpu_cache_hit);
        let report = runner
            .borrow_mut()
            .run_resident_from_f32_buffer(
                final_norm.runner.borrow().shared_context(),
                final_norm.runner.borrow().output_buffer_handle(),
                len,
                final_norm.runner.borrow().output_buffer_size(),
            )
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu raw-f32 pair projection failed for {first_name}+{second_name}: {error}"
                ))
            })?;
        Ok(ResidentPackedPairProjection {
            tensor_name: pair_key,
            first_rows,
            second_rows,
            cols,
            activation_upload_bytes: cols * std::mem::size_of::<f32>(),
            prepared: PreparedProjectionRunner {
                packed,
                runner,
                compile_duration,
                weight_upload_duration,
                pack_cache_hit,
                gpu_cache_hit,
            },
            report,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn run_projection_triplet(
        &mut self,
        first_name: &str,
        first_rows: usize,
        second_name: &str,
        second_rows: usize,
        third_name: &str,
        third_rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<ProjectionTripletOutputs, ReferenceError> {
        let triplet_key = format!("concat::{first_name}||{second_name}||{third_name}");
        let (packed, pack_duration, pack_cache_hit) =
            self.model.get_or_create_projection_triplet_cache(
                &triplet_key,
                first_name,
                first_rows,
                second_name,
                second_rows,
                third_name,
                third_rows,
                cols,
            )?;
        let (runner, compile_duration, weight_upload_duration, gpu_cache_hit) = self
            .model
            .get_or_create_projection_gpu(&triplet_key, &packed)?;
        self.metrics.pack_duration += pack_duration;
        self.metrics.compile_duration += compile_duration;
        self.metrics.weight_upload_duration += weight_upload_duration;
        self.metrics.pack_cache_hits += usize::from(pack_cache_hit);
        self.metrics.gpu_cache_hits += usize::from(gpu_cache_hit);

        let (combined, report) =
            runner
                .borrow_mut()
                .run_with_output(input, None)
                .map_err(|error| {
                    ReferenceError::Decode(format!(
                        "gpu projection failed for {first_name}+{second_name}+{third_name}: {error}"
                    ))
                })?;
        self.metrics.activation_upload_duration += report.upload_duration;
        self.metrics.upload_duration += weight_upload_duration + report.upload_duration;
        self.metrics.gpu_duration += report.gpu_duration;
        self.metrics.download_duration += report.download_duration;
        self.metrics.dispatch_count += 1;
        let weight_bytes = packed.code_words.len() * std::mem::size_of::<u32>()
            + packed.scales.len() * std::mem::size_of::<f32>();
        let activation_upload_bytes = cols.div_ceil(2) * std::mem::size_of::<u32>();
        if !gpu_cache_hit {
            self.metrics.weight_upload_bytes += weight_bytes;
        }
        self.metrics.activation_upload_bytes += activation_upload_bytes;
        self.metrics.upload_bytes +=
            usize::from(!gpu_cache_hit) * weight_bytes + activation_upload_bytes;
        self.metrics.download_bytes +=
            (first_rows + second_rows + third_rows) * std::mem::size_of::<f32>();

        let (first, tail) = combined.split_at(first_rows);
        let (second, third) = tail.split_at(second_rows);
        let weight_upload_bytes = usize::from(!gpu_cache_hit)
            * (packed.code_words.len() * std::mem::size_of::<u32>()
                + packed.scales.len() * std::mem::size_of::<f32>());
        let activation_upload_bytes = cols.div_ceil(2) * std::mem::size_of::<u32>();
        let download_bytes = (first_rows + second_rows + third_rows) * std::mem::size_of::<f32>();
        self.push_dispatch_trace(
            &triplet_key,
            "triplet",
            first_rows + second_rows + third_rows,
            cols,
            pack_cache_hit,
            gpu_cache_hit,
            compile_duration,
            weight_upload_duration,
            report.upload_duration,
            report.gpu_duration,
            report.download_duration,
            weight_upload_bytes,
            activation_upload_bytes,
            download_bytes,
        );
        Ok((first.to_vec(), second.to_vec(), third.to_vec()))
    }
}

pub struct ReferenceModel {
    pub assets: BonsaiAssetPaths,
    pub config: BonsaiModelConfig,
    pub generation_config: GenerationConfig,
    pub tokenizer: Option<TokenizerRuntime>,
    pub weights: WeightStore,
    packed_model: Option<PackedModelStore>,
    rope: YarnRope,
    layer_tensors: Vec<LayerTensorNames>,
    cached_hybrid_qproj: RefCell<HashMap<usize, Rc<HybridQProjCache>>>,
    cached_hybrid_qproj_gpu: RefCell<HashMap<usize, Rc<RefCell<CachedGpuPackedMatvecRunner>>>>,
    cached_projection_packed: RefCell<HashMap<String, Rc<PackedProjectionCache>>>,
    cached_projection_gpu: RefCell<HashMap<String, CachedProjectionGpuRunner>>,
    cached_projection_gpu_raw_f32: RefCell<HashMap<String, CachedProjectionGpuRunner>>,
    cached_final_norm_gpu: RefCell<Option<CachedWeightedRmsNormGpuRunner>>,
    cached_pack_f16_pairs_gpu: RefCell<HashMap<usize, CachedPackF16PairsGpuRunner>>,
    cached_vector_add_gpu: RefCell<HashMap<usize, CachedVectorAddGpuRunner>>,
    cached_swiglu_combined_gpu: RefCell<Option<CachedSwigluCombinedGpuRunner>>,
    cached_swiglu_pack_f16_pairs_gpu: RefCell<Option<CachedSwigluPackF16PairsGpuRunner>>,
    cached_tail_block_gpu: RefCell<Option<CachedTailBlockGpuRunner>>,
    cached_full_last_layer_block_gpu: RefCell<Option<CachedFullLastLayerGpuRunner>>,
    packed_gpu_context: RefCell<Option<Arc<SharedGpuPackedContext>>>,
}

impl ReferenceModel {
    fn packed_enabled_label(use_attention_qkv: bool, use_mlp_gu: bool) -> String {
        let use_attention_full = use_attention_qkv && Self::packed_use_attention_full();
        let use_mlp_full = use_mlp_gu && Self::packed_use_mlp_full();
        let mut enabled = String::new();
        if use_attention_qkv {
            enabled.push_str("qkv");
            if use_attention_full {
                enabled.push_str("+o");
            }
        }
        if use_mlp_gu {
            if !enabled.is_empty() {
                enabled.push('+');
            }
            enabled.push_str("gu");
            if use_mlp_full {
                enabled.push_str("+d");
            }
        }
        if enabled.is_empty() {
            enabled.push_str("dense");
        }
        enabled
    }

    fn packed_use_mlp_full() -> bool {
        std::env::var_os("JENGINE_PACKED_MLP_FULL").is_some()
    }

    fn packed_use_attention_full() -> bool {
        std::env::var_os("JENGINE_PACKED_ATTENTION_FULL").is_some()
    }

    fn packed_use_gpu_final_norm() -> bool {
        std::env::var_os("JENGINE_GPU_FINAL_NORM").is_some()
    }

    fn packed_use_gpu_swiglu_block() -> bool {
        std::env::var_os("JENGINE_GPU_SWIGLU_BLOCK").is_some()
    }

    fn packed_use_gpu_attention_block() -> bool {
        std::env::var_os("JENGINE_GPU_ATTENTION_BLOCK").is_some()
    }

    fn packed_use_gpu_embedding() -> bool {
        std::env::var_os("JENGINE_GPU_EMBEDDING").is_some()
    }

    fn packed_use_gpu_first_session() -> bool {
        Self::packed_use_gpu_attention_block()
            || Self::packed_use_gpu_embedding()
            || Self::packed_use_gpu_swiglu_block()
            || Self::packed_use_gpu_full_last_layer()
            || Self::packed_use_gpu_tail()
    }

    fn packed_use_gpu_mlp_entry() -> bool {
        std::env::var_os("JENGINE_GPU_MLP_ENTRY").is_some()
    }

    fn packed_use_gpu_full_last_layer() -> bool {
        std::env::var_os("JENGINE_GPU_FULL_LAST_LAYER").is_some()
    }

    fn packed_use_gpu_tail() -> bool {
        std::env::var_os("JENGINE_GPU_TAIL").is_some()
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_packed_decode_metrics(
        enabled_projections: String,
        total_duration: Duration,
        decode_metrics: &DecodeMetrics,
        attention_stage_metrics: &PackedAttentionStageMetrics,
        mlp_stage_metrics: &PackedMlpStageMetrics,
        non_offloaded_dense_duration: Duration,
        session: &PackedGpuSession<'_>,
        output_text: String,
    ) -> PackedDecodeMetrics {
        let orchestration_duration = total_duration.saturating_sub(
            session.metrics.pack_duration
                + session.metrics.compile_duration
                + session.metrics.weight_upload_duration
                + session.metrics.activation_upload_duration
                + session.metrics.gpu_duration
                + session.metrics.download_duration
                + non_offloaded_dense_duration,
        );
        PackedDecodeMetrics {
            enabled_projections,
            total_duration,
            embedding_duration: decode_metrics.embedding_duration,
            norm_duration: decode_metrics.norm_duration,
            qkv_duration: decode_metrics.qkv_duration,
            attention_duration: decode_metrics.attention_duration,
            attention_query_duration: attention_stage_metrics.query_duration,
            attention_oproj_duration: attention_stage_metrics.oproj_duration,
            attention_residual_duration: attention_stage_metrics.residual_duration,
            mlp_duration: decode_metrics.mlp_duration,
            mlp_swiglu_duration: mlp_stage_metrics.swiglu_duration,
            mlp_down_duration: mlp_stage_metrics.down_duration,
            mlp_residual_duration: mlp_stage_metrics.residual_duration,
            logits_duration: decode_metrics.logits_duration,
            pack_duration: session.metrics.pack_duration,
            compile_duration: session.metrics.compile_duration,
            weight_upload_duration: session.metrics.weight_upload_duration,
            activation_upload_duration: session.metrics.activation_upload_duration,
            upload_duration: session.metrics.upload_duration,
            gpu_duration: session.metrics.gpu_duration,
            download_duration: session.metrics.download_duration,
            non_offloaded_dense_duration,
            orchestration_duration,
            pack_cache_hits: session.metrics.pack_cache_hits,
            gpu_cache_hits: session.metrics.gpu_cache_hits,
            dispatch_count: session.metrics.dispatch_count,
            weight_upload_bytes: session.metrics.weight_upload_bytes,
            activation_upload_bytes: session.metrics.activation_upload_bytes,
            upload_bytes: session.metrics.upload_bytes,
            download_bytes: session.metrics.download_bytes,
            output_text,
        }
    }

    pub fn load_core_from_root(root: impl AsRef<Path>) -> Result<Self, ReferenceError> {
        let assets = BonsaiAssetPaths::from_root(root)?;
        let config = serde_json::from_str::<BonsaiModelConfig>(&std::fs::read_to_string(
            &assets.config_json,
        )?)?;
        let generation_config = serde_json::from_str::<GenerationConfig>(
            &std::fs::read_to_string(&assets.generation_config_json)?,
        )?;
        let weights = WeightStore::load_from_assets(&assets)?;
        let rope = build_yarn_rope(&config);
        let layer_tensors = (0..config.num_hidden_layers)
            .map(|layer_idx| {
                let prefix = format!("model.layers.{layer_idx}");
                LayerTensorNames {
                    input_layernorm_weight: format!("{prefix}.input_layernorm.weight"),
                    post_attention_layernorm_weight: format!(
                        "{prefix}.post_attention_layernorm.weight"
                    ),
                    q_norm_weight: format!("{prefix}.self_attn.q_norm.weight"),
                    k_norm_weight: format!("{prefix}.self_attn.k_norm.weight"),
                    q_proj_weight: format!("{prefix}.self_attn.q_proj.weight"),
                    k_proj_weight: format!("{prefix}.self_attn.k_proj.weight"),
                    v_proj_weight: format!("{prefix}.self_attn.v_proj.weight"),
                    o_proj_weight: format!("{prefix}.self_attn.o_proj.weight"),
                    gate_proj_weight: format!("{prefix}.mlp.gate_proj.weight"),
                    up_proj_weight: format!("{prefix}.mlp.up_proj.weight"),
                    down_proj_weight: format!("{prefix}.mlp.down_proj.weight"),
                }
            })
            .collect();
        Ok(Self {
            assets,
            config,
            generation_config,
            tokenizer: None,
            weights,
            packed_model: None,
            rope,
            layer_tensors,
            cached_hybrid_qproj: RefCell::new(HashMap::new()),
            cached_hybrid_qproj_gpu: RefCell::new(HashMap::new()),
            cached_projection_packed: RefCell::new(HashMap::new()),
            cached_projection_gpu: RefCell::new(HashMap::new()),
            cached_projection_gpu_raw_f32: RefCell::new(HashMap::new()),
            cached_final_norm_gpu: RefCell::new(None),
            cached_pack_f16_pairs_gpu: RefCell::new(HashMap::new()),
            cached_vector_add_gpu: RefCell::new(HashMap::new()),
            cached_swiglu_combined_gpu: RefCell::new(None),
            cached_swiglu_pack_f16_pairs_gpu: RefCell::new(None),
            cached_tail_block_gpu: RefCell::new(None),
            cached_full_last_layer_block_gpu: RefCell::new(None),
            packed_gpu_context: RefCell::new(None),
        })
    }

    pub fn load_from_root(root: impl AsRef<Path>) -> Result<Self, ReferenceError> {
        let mut model = Self::load_core_from_root(root)?;
        model.tokenizer = Some(TokenizerRuntime::load_from_file(
            &model.assets.tokenizer_json,
        )?);
        Ok(model)
    }

    pub fn load_core_from_root_with_packed_artifact(
        root: impl AsRef<Path>,
        artifact_dir: impl AsRef<Path>,
    ) -> Result<Self, ReferenceError> {
        let mut model = Self::load_core_from_root(root)?;
        model.packed_model = Some(PackedModelStore::load_from_artifact_dir(
            artifact_dir,
            &model.config,
        )?);
        Ok(model)
    }

    pub fn load_from_root_with_packed_artifact(
        root: impl AsRef<Path>,
        artifact_dir: impl AsRef<Path>,
    ) -> Result<Self, ReferenceError> {
        let mut model = Self::load_core_from_root_with_packed_artifact(root, artifact_dir)?;
        model.tokenizer = Some(TokenizerRuntime::load_from_file(
            &model.assets.tokenizer_json,
        )?);
        Ok(model)
    }

    pub fn prompt_analysis(&self, prompt: &str) -> Result<PromptAnalysis, ReferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ReferenceError::Decode("tokenizer is not loaded".to_string()))?;
        Ok(tokenizer.analyze_prompt(prompt)?)
    }

    pub fn tokenizer_diagnostics(&self) -> Option<TokenizerDiagnostics> {
        self.tokenizer.as_ref().map(TokenizerRuntime::diagnostics)
    }

    pub fn memory_report(&self, prompt_tokens: usize, generated_tokens: usize) -> MemoryReport {
        let total_sequence_tokens = prompt_tokens + generated_tokens;
        let estimated_model_fp16_bytes = self.config.approx_fp16_bytes();
        let source_weight_bytes = self
            .assets
            .safetensors_path
            .metadata()
            .map(|metadata| metadata.len() as usize)
            .unwrap_or(0);
        let kv_cache_bytes_per_token_fp16 = self.config.kv_cache_bytes_per_token(1, 2);
        let kv_cache_bytes_per_token_runtime_f32 = self.config.kv_cache_bytes_per_token(1, 4);
        let kv_cache_total_bytes_fp16 = self.config.kv_cache_bytes(total_sequence_tokens, 1, 2);
        let kv_cache_total_bytes_runtime_f32 =
            self.config.kv_cache_bytes(total_sequence_tokens, 1, 4);
        let packed_cache_bytes = self.packed_cache_bytes();
        let gpu_cache_buffer_bytes = self.gpu_cache_buffer_bytes();
        let activation_working_bytes = self.activation_working_bytes();
        let staging_bytes = gpu_cache_buffer_bytes;
        MemoryReport {
            prompt_tokens,
            generated_tokens,
            total_sequence_tokens,
            estimated_model_fp16_bytes,
            source_weight_bytes,
            kv_cache_bytes_per_token_fp16,
            kv_cache_bytes_per_token_runtime_f32,
            kv_cache_total_bytes_fp16,
            kv_cache_total_bytes_runtime_f32,
            kv_cache_reserved_bytes_runtime_f32: kv_cache_total_bytes_runtime_f32,
            packed_cache_bytes,
            gpu_cache_buffer_bytes,
            activation_working_bytes,
            staging_bytes,
            estimated_runtime_working_set_bytes: estimated_model_fp16_bytes
                + kv_cache_total_bytes_runtime_f32
                + packed_cache_bytes
                + gpu_cache_buffer_bytes
                + activation_working_bytes,
        }
    }

    pub fn prewarm_packed_decode_caches(
        &self,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        use_attention_full: bool,
        use_mlp_full: bool,
    ) -> Result<(), ReferenceError> {
        if self.packed_model.is_none() {
            return Ok(());
        }
        let kv_rows = self.config.num_key_value_heads * self.config.head_dim;
        for layer_tensors in &self.layer_tensors {
            let _ = self.load_vector_f32_resolved(&layer_tensors.input_layernorm_weight)?;
            let _ =
                self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
            let _ = self.load_vector_f32_resolved(&layer_tensors.q_norm_weight)?;
            let _ = self.load_vector_f32_resolved(&layer_tensors.k_norm_weight)?;
            if use_attention_qkv {
                let triplet_key = format!(
                    "concat::{}||{}||{}",
                    layer_tensors.q_proj_weight,
                    layer_tensors.k_proj_weight,
                    layer_tensors.v_proj_weight
                );
                let (packed, _, _) = self.get_or_create_projection_triplet_cache(
                    &triplet_key,
                    &layer_tensors.q_proj_weight,
                    self.config.hidden_size,
                    &layer_tensors.k_proj_weight,
                    kv_rows,
                    &layer_tensors.v_proj_weight,
                    kv_rows,
                    self.config.hidden_size,
                )?;
                let _ = self.get_or_create_projection_gpu(&triplet_key, &packed)?;
                if use_attention_full {
                    let (packed, _, _) = self.get_or_create_projection_cache(
                        &layer_tensors.o_proj_weight,
                        self.config.hidden_size,
                        self.config.hidden_size,
                    )?;
                    let _ =
                        self.get_or_create_projection_gpu(&layer_tensors.o_proj_weight, &packed)?;
                }
            }
            if use_mlp_gu {
                let pair_key = format!(
                    "concat::{}||{}",
                    layer_tensors.gate_proj_weight, layer_tensors.up_proj_weight
                );
                let (packed, _, _) = self.get_or_create_projection_pair_cache(
                    &pair_key,
                    &layer_tensors.gate_proj_weight,
                    self.config.intermediate_size,
                    &layer_tensors.up_proj_weight,
                    self.config.intermediate_size,
                    self.config.hidden_size,
                )?;
                let _ = self.get_or_create_projection_gpu(&pair_key, &packed)?;
                if Self::packed_use_gpu_mlp_entry() || Self::packed_use_gpu_full_last_layer() {
                    let _ = self.get_or_create_projection_gpu_raw_f32(&pair_key, &packed)?;
                }
                if use_mlp_full {
                    let (packed, _, _) = self.get_or_create_projection_cache(
                        &layer_tensors.down_proj_weight,
                        self.config.hidden_size,
                        self.config.intermediate_size,
                    )?;
                    let _ = self
                        .get_or_create_projection_gpu(&layer_tensors.down_proj_weight, &packed)?;
                    if Self::packed_use_gpu_swiglu_block() {
                        let _ = self.get_or_create_swiglu_pack_f16_pairs_gpu()?;
                    }
                }
            }
        }
        let _ = self.load_vector_f32_resolved("model.norm.weight")?;
        if use_mlp_gu {
            if Self::packed_use_gpu_mlp_entry() || Self::packed_use_gpu_full_last_layer() {
                let pair_key = format!(
                    "concat::{}||{}",
                    self.layer_tensors[0].gate_proj_weight, self.layer_tensors[0].up_proj_weight
                );
                let (packed, _, _) = self.get_or_create_projection_pair_cache(
                    &pair_key,
                    &self.layer_tensors[0].gate_proj_weight,
                    self.config.intermediate_size,
                    &self.layer_tensors[0].up_proj_weight,
                    self.config.intermediate_size,
                    self.config.hidden_size,
                )?;
                let _ = self.get_or_create_projection_gpu_raw_f32(&pair_key, &packed)?;
            }
            if Self::packed_use_gpu_swiglu_block() || Self::packed_use_gpu_full_last_layer() {
                let _ = self.get_or_create_swiglu_pack_f16_pairs_gpu()?;
                let _ = self.get_or_create_pack_f16_pairs_gpu(self.config.intermediate_size)?;
            }
        }
        if Self::packed_use_gpu_final_norm() || Self::packed_use_gpu_full_last_layer() {
            let _ = self.get_or_create_final_norm_gpu()?;
            let _ = self.get_or_create_pack_f16_pairs_gpu(self.config.hidden_size)?;
            if Self::packed_use_gpu_tail() || Self::packed_use_gpu_full_last_layer() {
                let _ = self.get_or_create_vector_add_gpu(self.config.hidden_size)?;
            }
        }
        let (_logits_packed, _, _) = self.get_or_create_projection_cache(
            "model.embed_tokens.weight",
            self.config.vocab_size,
            self.config.hidden_size,
        )?;
        if Self::packed_use_gpu_first_session() {
            let mut gpu_first_cache = GpuFirstRunnerCache::new(self, 1);
            gpu_first_cache.prewarm_decode_path(use_attention_qkv, use_mlp_gu)?;
        }
        Ok(())
    }

    pub fn begin_packed_decode_session(
        &self,
        expected_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        argmax_only: bool,
    ) -> PackedDecodeSession<'_> {
        if Self::packed_use_gpu_first_session() {
            PackedDecodeSession::GpuFirst(GpuFirstPackedDecodeSession::new(
                self,
                expected_tokens,
                use_attention_qkv,
                use_mlp_gu,
                argmax_only,
            ))
        } else {
            PackedDecodeSession::Legacy(PersistentPackedDecodeSession::new(
                self,
                expected_tokens,
                use_attention_qkv,
                use_mlp_gu,
                argmax_only,
            ))
        }
    }

    fn packed_cache_bytes(&self) -> usize {
        let mut total = self
            .packed_model
            .as_ref()
            .map(|packed| packed.unpacked_cache_bytes())
            .unwrap_or(0);
        let generic = self.cached_projection_packed.borrow();
        for cache in generic.values() {
            total += cache.code_words.len() * std::mem::size_of::<u32>();
            total += cache.scales.len() * std::mem::size_of::<f32>();
        }
        let qproj = self.cached_hybrid_qproj.borrow();
        for (layer_idx, cache) in qproj.iter() {
            let q_name = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
            if !generic.contains_key(&q_name) {
                total += cache.code_words.len() * std::mem::size_of::<u32>();
                total += cache.scales.len() * std::mem::size_of::<f32>();
            }
        }
        total
    }

    fn gpu_cache_buffer_bytes(&self) -> usize {
        let mut total = 0usize;
        let generic = self.cached_projection_gpu.borrow();
        for runner in generic.values() {
            total += runner.borrow().buffer_bytes();
        }
        let qproj = self.cached_hybrid_qproj_gpu.borrow();
        for (layer_idx, runner) in qproj.iter() {
            let q_name = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
            if !generic.contains_key(&q_name) {
                total += runner.borrow().buffer_bytes();
            }
        }
        total
    }

    fn load_vector_f32_resolved(&self, name: &str) -> Result<Vec<f32>, ReferenceError> {
        if let Some(packed) = &self.packed_model
            && let Some(values) = packed.load_vector_f32(name)?
        {
            return Ok(values);
        }
        Ok(self.weights.load_vector_f32(name)?)
    }

    fn embedding_lookup_resolved(
        &self,
        name: &str,
        token_id: usize,
    ) -> Result<Vec<f32>, ReferenceError> {
        if let Some(packed) = &self.packed_model
            && let Some(values) = packed.embedding_lookup(name, token_id)?
        {
            return Ok(values);
        }
        Ok(self.weights.embedding_lookup(name, token_id)?)
    }

    fn matvec_f16_resolved(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, ReferenceError> {
        if let Some(packed) = &self.packed_model
            && let Some(values) = packed.matvec_f32(name, input)?
        {
            return Ok(values);
        }
        Ok(self.weights.matvec_f16(name, input)?)
    }

    fn packed_codes_to_words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks(4)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, byte) in chunk.iter().enumerate() {
                    word |= (*byte as u32) << (i * 8);
                }
                word
            })
            .collect()
    }

    fn load_packed_projection_cache_from_artifact(
        &self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Option<PackedProjectionCache>, ReferenceError> {
        let Some(packed_model) = &self.packed_model else {
            return Ok(None);
        };
        let Some(packed) = packed_model.load_packed_tensor_file(tensor_name)? else {
            return Ok(None);
        };
        if packed.tensor.shape != vec![rows, cols] {
            return Err(ReferenceError::Decode(format!(
                "packed artifact shape mismatch for {tensor_name}: {:?} vs [{rows}, {cols}]",
                packed.tensor.shape
            )));
        }
        Ok(Some(PackedProjectionCache {
            rows,
            cols,
            group_size: packed.tensor.group_size,
            code_words: Self::packed_codes_to_words(&packed.tensor.packed_codes),
            scales: packed.tensor.scales,
        }))
    }

    fn load_hybrid_qproj_cache_from_artifact(
        &self,
        layer_idx: usize,
    ) -> Result<Option<HybridQProjCache>, ReferenceError> {
        let tensor_name = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
        let Some(packed) = self.load_packed_projection_cache_from_artifact(
            &tensor_name,
            self.config.hidden_size,
            self.config.hidden_size,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(HybridQProjCache {
            layer_idx,
            rows: packed.rows,
            cols: packed.cols,
            group_size: packed.group_size,
            code_words: packed.code_words,
            scales: packed.scales,
        }))
    }

    fn activation_working_bytes(&self) -> usize {
        let hidden = self.config.hidden_size * std::mem::size_of::<f32>();
        let kv =
            self.config.num_key_value_heads * self.config.head_dim * std::mem::size_of::<f32>();
        let intermediate = self.config.intermediate_size * std::mem::size_of::<f32>();
        let logits = self.config.vocab_size * std::mem::size_of::<f32>();
        hidden
            + hidden
            + hidden
            + kv
            + kv
            + hidden
            + intermediate
            + intermediate
            + intermediate
            + logits
    }

    fn allocate_layer_cache_vec(
        &self,
        expected_tokens: usize,
        preallocate_cpu_kv: bool,
    ) -> Vec<LayerCache> {
        let kv_width = self.config.num_key_value_heads * self.config.head_dim;
        (0..self.config.num_hidden_layers)
            .map(|_| {
                if preallocate_cpu_kv {
                    LayerCache::with_capacity(expected_tokens, kv_width)
                } else {
                    LayerCache::without_preallocated_cpu_kv()
                }
            })
            .collect()
    }

    pub fn generate_greedy(
        &self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<DecodeResult, ReferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ReferenceError::Decode("tokenizer is not loaded".to_string()))?;
        let prompt_ids = tokenizer
            .encode(prompt)?
            .into_iter()
            .map(|id| id as usize)
            .collect::<Vec<_>>();
        if prompt_ids.is_empty() {
            return Err(ReferenceError::Decode(
                "prompt encoded to zero tokens".to_string(),
            ));
        }
        self.generate_from_token_ids(&prompt_ids, max_new_tokens)
    }

    pub fn generate_from_token_ids(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<DecodeResult, ReferenceError> {
        if prompt_ids.is_empty() {
            return Err(ReferenceError::Decode(
                "prompt_ids cannot be empty".to_string(),
            ));
        }

        if self.packed_model.is_some() {
            let packed =
                self.generate_packed_from_token_ids(prompt_ids, max_new_tokens, true, true)?;
            return Ok(DecodeResult {
                output_token_ids: packed.output_token_ids,
                output_text: packed.output_text,
                metrics: packed.decode_metrics,
            });
        }

        let started_at = Instant::now();
        let mut metrics = DecodeMetrics {
            prompt_tokens: prompt_ids.len(),
            generated_tokens: 0,
            total_duration: Duration::ZERO,
            embedding_duration: Duration::ZERO,
            norm_duration: Duration::ZERO,
            qkv_duration: Duration::ZERO,
            attention_duration: Duration::ZERO,
            mlp_duration: Duration::ZERO,
            logits_duration: Duration::ZERO,
        };
        let mut cache = self.allocate_layer_cache_vec(prompt_ids.len() + max_new_tokens, true);

        let mut last_logits = Vec::new();
        for (position, &token_id) in prompt_ids.iter().enumerate() {
            last_logits = self.forward_step(token_id, position, &mut cache, &mut metrics)?;
        }

        let mut output_ids = prompt_ids.to_vec();
        for generation_index in 0..max_new_tokens {
            let next_token = argmax(&last_logits).ok_or_else(|| {
                ReferenceError::Decode("argmax failed on empty logits".to_string())
            })?;
            output_ids.push(next_token);
            metrics.generated_tokens += 1;
            last_logits = self.forward_step(
                next_token,
                prompt_ids.len() + generation_index,
                &mut cache,
                &mut metrics,
            )?;
        }

        metrics.total_duration = started_at.elapsed();
        let output_text = match &self.tokenizer {
            Some(tokenizer) => {
                tokenizer.decode(&output_ids.iter().map(|id| *id as u32).collect::<Vec<_>>())?
            }
            None => output_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        };

        Ok(DecodeResult {
            output_token_ids: output_ids,
            output_text,
            metrics,
        })
    }

    pub fn compare_qproj_dense_vs_packed(
        &self,
        layer_idx: usize,
        token_id: usize,
    ) -> Result<ProjectionComparison, ReferenceError> {
        let hidden_states = self.layer_input_hidden(token_id, layer_idx)?;
        let tensor_name = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");

        let started_at = Instant::now();
        let dense = self.weights.matvec_f16(&tensor_name, &hidden_states)?;
        let dense_duration = started_at.elapsed();

        let values = self.weights.load_vector_f32(&tensor_name)?;
        let started_at = Instant::now();
        let (packed, _) = pack_ternary_g128(
            &values,
            vec![self.config.hidden_size, self.config.hidden_size],
            1e-3,
        )
        .map_err(|error| ReferenceError::Decode(format!("pack failed: {error}")))?;
        let pack_duration = started_at.elapsed();

        let started_at = Instant::now();
        let packed_out = matvec_packed_ternary(&packed, &hidden_states)
            .map_err(|error| ReferenceError::Decode(format!("packed matvec failed: {error}")))?;
        let packed_duration = started_at.elapsed();

        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        for (left, right) in dense.iter().zip(packed_out.iter()) {
            let diff = (left - right).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff;
        }
        let mean_abs_diff = sum_abs_diff / dense.len() as f32;

        Ok(ProjectionComparison {
            layer_idx,
            token_id,
            dense_duration,
            pack_duration,
            packed_duration,
            max_abs_diff,
            mean_abs_diff,
        })
    }

    fn layer_input_hidden(
        &self,
        token_id: usize,
        layer_idx: usize,
    ) -> Result<Vec<f32>, ReferenceError> {
        let hidden = self.embedding_lookup_resolved("model.embed_tokens.weight", token_id)?;
        let input_norm_weight = self.load_vector_f32_resolved(&format!(
            "model.layers.{layer_idx}.input_layernorm.weight"
        ))?;
        Ok(weighted_rms_norm(
            &hidden,
            &input_norm_weight,
            self.config.rms_norm_eps as f32,
        ))
    }

    pub fn benchmark_hybrid_qproj_decode(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        layer_idx: usize,
    ) -> Result<HybridDecodeMetrics, ReferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ReferenceError::Decode("tokenizer is not loaded".to_string()))?;
        let prompt_ids = self.encode_prompt_ids(tokenizer, prompt)?;
        let (hybrid, q_proj_pack_duration) = self.build_hybrid_qproj_cache(layer_idx)?;
        self.run_hybrid_qproj_decode(
            tokenizer,
            &prompt_ids,
            max_new_tokens,
            &hybrid,
            q_proj_pack_duration,
            false,
            Duration::ZERO,
            false,
            None,
        )
    }

    pub fn benchmark_cached_hybrid_qproj_decode(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        layer_idx: usize,
    ) -> Result<HybridDecodeMetrics, ReferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ReferenceError::Decode("tokenizer is not loaded".to_string()))?;
        let prompt_ids = self.encode_prompt_ids(tokenizer, prompt)?;
        let (hybrid, q_proj_pack_duration, q_proj_pack_cache_hit) =
            self.get_or_create_hybrid_qproj_cache(layer_idx)?;
        let (gpu_runner, q_proj_gpu_compile_duration, q_proj_gpu_cache_hit) =
            self.get_or_create_hybrid_qproj_gpu(layer_idx, &hybrid)?;
        self.run_hybrid_qproj_decode(
            tokenizer,
            &prompt_ids,
            max_new_tokens,
            &hybrid,
            q_proj_pack_duration,
            q_proj_pack_cache_hit,
            q_proj_gpu_compile_duration,
            q_proj_gpu_cache_hit,
            Some(gpu_runner),
        )
    }

    fn encode_prompt_ids(
        &self,
        tokenizer: &TokenizerRuntime,
        prompt: &str,
    ) -> Result<Vec<usize>, ReferenceError> {
        let prompt_ids = tokenizer
            .encode(prompt)?
            .into_iter()
            .map(|id| id as usize)
            .collect::<Vec<_>>();
        if prompt_ids.is_empty() {
            return Err(ReferenceError::Decode(
                "prompt encoded to zero tokens".to_string(),
            ));
        }
        Ok(prompt_ids)
    }

    fn build_hybrid_qproj_cache(
        &self,
        layer_idx: usize,
    ) -> Result<(HybridQProjCache, Duration), ReferenceError> {
        if let Some(hybrid) = self.load_hybrid_qproj_cache_from_artifact(layer_idx)? {
            return Ok((hybrid, Duration::ZERO));
        }
        let tensor_name = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
        let values = self.weights.load_vector_f32(&tensor_name)?;
        let pack_started = Instant::now();
        let (packed, _) = pack_ternary_g128(
            &values,
            vec![self.config.hidden_size, self.config.hidden_size],
            1e-3,
        )
        .map_err(|error| ReferenceError::Decode(format!("pack failed: {error}")))?;
        let q_proj_pack_duration = pack_started.elapsed();
        let code_words = Self::packed_codes_to_words(&packed.packed_codes);
        Ok((
            HybridQProjCache {
                layer_idx,
                rows: self.config.hidden_size,
                cols: self.config.hidden_size,
                group_size: packed.group_size,
                code_words,
                scales: packed.scales,
            },
            q_proj_pack_duration,
        ))
    }

    fn get_or_create_hybrid_qproj_cache(
        &self,
        layer_idx: usize,
    ) -> Result<(Rc<HybridQProjCache>, Duration, bool), ReferenceError> {
        if let Some(cached) = self.cached_hybrid_qproj.borrow().get(&layer_idx).cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let (hybrid, duration) = self.build_hybrid_qproj_cache(layer_idx)?;
        let hybrid = Rc::new(hybrid);
        self.cached_hybrid_qproj
            .borrow_mut()
            .insert(layer_idx, hybrid.clone());
        Ok((hybrid, duration, false))
    }

    fn get_or_create_hybrid_qproj_gpu(
        &self,
        layer_idx: usize,
        hybrid: &HybridQProjCache,
    ) -> Result<(Rc<RefCell<CachedGpuPackedMatvecRunner>>, Duration, bool), ReferenceError> {
        if let Some(cached) = self
            .cached_hybrid_qproj_gpu
            .borrow()
            .get(&layer_idx)
            .cloned()
        {
            return Ok((cached, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (runner, duration) = CachedGpuPackedMatvecRunner::new_with_context(
            context,
            &hybrid.code_words,
            &hybrid.scales,
            hybrid.group_size,
            hybrid.rows,
            hybrid.cols,
        )
        .map_err(|error| {
            ReferenceError::Decode(format!("gpu packed q_proj init failed: {error}"))
        })?;
        let runner = Rc::new(RefCell::new(runner));
        self.cached_hybrid_qproj_gpu
            .borrow_mut()
            .insert(layer_idx, runner.clone());
        Ok((runner, duration, false))
    }

    fn build_projection_cache(
        &self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(PackedProjectionCache, Duration), ReferenceError> {
        if let Some(packed) =
            self.load_packed_projection_cache_from_artifact(tensor_name, rows, cols)?
        {
            return Ok((packed, Duration::ZERO));
        }
        let values = self.weights.load_vector_f32(tensor_name)?;
        let pack_started = Instant::now();
        let (packed, _) = pack_ternary_g128(&values, vec![rows, cols], 1e-3).map_err(|error| {
            ReferenceError::Decode(format!("pack failed for {tensor_name}: {error}"))
        })?;
        let pack_duration = pack_started.elapsed();
        let code_words = Self::packed_codes_to_words(&packed.packed_codes);
        Ok((
            PackedProjectionCache {
                rows,
                cols,
                group_size: packed.group_size,
                code_words,
                scales: packed.scales,
            },
            pack_duration,
        ))
    }

    fn get_or_create_projection_cache(
        &self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(Rc<PackedProjectionCache>, Duration, bool), ReferenceError> {
        if let Some(cached) = self
            .cached_projection_packed
            .borrow()
            .get(tensor_name)
            .cloned()
        {
            return Ok((cached, Duration::ZERO, true));
        }
        let (packed, duration) = self.build_projection_cache(tensor_name, rows, cols)?;
        let packed = Rc::new(packed);
        self.cached_projection_packed
            .borrow_mut()
            .insert(tensor_name.to_string(), packed.clone());
        Ok((packed, duration, false))
    }

    fn get_or_create_projection_pair_cache(
        &self,
        cache_key: &str,
        first_name: &str,
        first_rows: usize,
        second_name: &str,
        second_rows: usize,
        cols: usize,
    ) -> Result<(Rc<PackedProjectionCache>, Duration, bool), ReferenceError> {
        if let Some(cached) = self
            .cached_projection_packed
            .borrow()
            .get(cache_key)
            .cloned()
        {
            return Ok((cached, Duration::ZERO, true));
        }
        let (first, first_duration, _) =
            self.get_or_create_projection_cache(first_name, first_rows, cols)?;
        let (second, second_duration, _) =
            self.get_or_create_projection_cache(second_name, second_rows, cols)?;
        if first.cols != second.cols {
            return Err(ReferenceError::Decode(format!(
                "cannot concatenate packed projections with different column counts: {first_name}={} vs {second_name}={}",
                first.cols, second.cols
            )));
        }
        if first.group_size != second.group_size {
            return Err(ReferenceError::Decode(format!(
                "cannot concatenate packed projections with different group sizes: {first_name}={} vs {second_name}={}",
                first.group_size, second.group_size
            )));
        }

        let mut code_words = Vec::with_capacity(first.code_words.len() + second.code_words.len());
        code_words.extend_from_slice(&first.code_words);
        code_words.extend_from_slice(&second.code_words);
        let mut scales = Vec::with_capacity(first.scales.len() + second.scales.len());
        scales.extend_from_slice(&first.scales);
        scales.extend_from_slice(&second.scales);

        let packed = Rc::new(PackedProjectionCache {
            rows: first.rows + second.rows,
            cols: first.cols,
            group_size: first.group_size,
            code_words,
            scales,
        });
        self.cached_projection_packed
            .borrow_mut()
            .insert(cache_key.to_string(), packed.clone());
        Ok((packed, first_duration + second_duration, false))
    }

    #[allow(clippy::too_many_arguments)]
    fn get_or_create_projection_triplet_cache(
        &self,
        cache_key: &str,
        first_name: &str,
        first_rows: usize,
        second_name: &str,
        second_rows: usize,
        third_name: &str,
        third_rows: usize,
        cols: usize,
    ) -> Result<(Rc<PackedProjectionCache>, Duration, bool), ReferenceError> {
        if let Some(cached) = self
            .cached_projection_packed
            .borrow()
            .get(cache_key)
            .cloned()
        {
            return Ok((cached, Duration::ZERO, true));
        }
        let (first, first_duration, _) =
            self.get_or_create_projection_cache(first_name, first_rows, cols)?;
        let (second, second_duration, _) =
            self.get_or_create_projection_cache(second_name, second_rows, cols)?;
        let (third, third_duration, _) =
            self.get_or_create_projection_cache(third_name, third_rows, cols)?;

        if first.cols != second.cols || first.cols != third.cols {
            return Err(ReferenceError::Decode(format!(
                "cannot concatenate packed projections with different column counts: {first_name}={}, {second_name}={}, {third_name}={}",
                first.cols, second.cols, third.cols
            )));
        }
        if first.group_size != second.group_size || first.group_size != third.group_size {
            return Err(ReferenceError::Decode(format!(
                "cannot concatenate packed projections with different group sizes: {first_name}={}, {second_name}={}, {third_name}={}",
                first.group_size, second.group_size, third.group_size
            )));
        }

        let mut code_words = Vec::with_capacity(
            first.code_words.len() + second.code_words.len() + third.code_words.len(),
        );
        code_words.extend_from_slice(&first.code_words);
        code_words.extend_from_slice(&second.code_words);
        code_words.extend_from_slice(&third.code_words);
        let mut scales =
            Vec::with_capacity(first.scales.len() + second.scales.len() + third.scales.len());
        scales.extend_from_slice(&first.scales);
        scales.extend_from_slice(&second.scales);
        scales.extend_from_slice(&third.scales);

        let packed = Rc::new(PackedProjectionCache {
            rows: first.rows + second.rows + third.rows,
            cols: first.cols,
            group_size: first.group_size,
            code_words,
            scales,
        });
        self.cached_projection_packed
            .borrow_mut()
            .insert(cache_key.to_string(), packed.clone());
        Ok((
            packed,
            first_duration + second_duration + third_duration,
            false,
        ))
    }

    fn get_or_create_projection_gpu(
        &self,
        tensor_name: &str,
        packed: &PackedProjectionCache,
    ) -> Result<CachedProjectionGpuCacheEntry, ReferenceError> {
        if let Some(cached) = self
            .cached_projection_gpu
            .borrow()
            .get(tensor_name)
            .cloned()
        {
            return Ok((cached, Duration::ZERO, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (mut runner, duration) = CachedGpuPackedMatvecRunner::new_uninitialized_with_context(
            context,
            packed.code_words.len(),
            packed.scales.len(),
            packed.group_size,
            packed.rows,
            packed.cols,
        )
        .map_err(|error| {
            ReferenceError::Decode(format!("gpu packed init failed for {tensor_name}: {error}"))
        })?;
        let weight_upload_duration = runner
            .update_weights(&packed.code_words, &packed.scales)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu weight upload failed for {tensor_name}: {error}"
                ))
            })?;
        let runner = Rc::new(RefCell::new(runner));
        self.cached_projection_gpu
            .borrow_mut()
            .insert(tensor_name.to_string(), runner.clone());
        Ok((runner, duration, weight_upload_duration, false))
    }

    fn get_or_create_projection_gpu_raw_f32(
        &self,
        tensor_name: &str,
        packed: &PackedProjectionCache,
    ) -> Result<CachedProjectionGpuCacheEntry, ReferenceError> {
        let cache_key = format!("rawf32::{tensor_name}");
        if let Some(cached) = self
            .cached_projection_gpu_raw_f32
            .borrow()
            .get(&cache_key)
            .cloned()
        {
            return Ok((cached, Duration::ZERO, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (mut runner, duration) = CachedGpuPackedMatvecRunner::new_with_context_and_input_mode(
            context,
            &packed.code_words,
            &packed.scales,
            packed.group_size,
            packed.rows,
            packed.cols,
            PackedRunnerInputMode::RawF32,
        )
        .map_err(|error| {
            ReferenceError::Decode(format!(
                "gpu raw-f32 init failed for {tensor_name}: {error}"
            ))
        })?;
        let weight_upload_duration = runner
            .update_weights(&packed.code_words, &packed.scales)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu raw-f32 weight upload failed for {tensor_name}: {error}"
                ))
            })?;
        let runner = Rc::new(RefCell::new(runner));
        self.cached_projection_gpu_raw_f32
            .borrow_mut()
            .insert(cache_key, runner.clone());
        Ok((runner, duration, weight_upload_duration, false))
    }

    fn get_or_create_final_norm_gpu(
        &self,
    ) -> Result<CachedWeightedRmsNormGpuCacheEntry, ReferenceError> {
        if let Some(cached) = self.cached_final_norm_gpu.borrow().as_ref().cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (runner, duration) = CachedGpuWeightedRmsNormRunner::new_with_context(
            context,
            self.config.hidden_size,
            self.config.rms_norm_eps as f32,
        )
        .map_err(|error| ReferenceError::Decode(format!("gpu final norm init failed: {error}")))?;
        let runner = Rc::new(RefCell::new(runner));
        *self.cached_final_norm_gpu.borrow_mut() = Some(runner.clone());
        Ok((runner, duration, false))
    }

    fn get_or_create_pack_f16_pairs_gpu(
        &self,
        len: usize,
    ) -> Result<CachedPackF16PairsGpuCacheEntry, ReferenceError> {
        if let Some(cached) = self.cached_pack_f16_pairs_gpu.borrow().get(&len).cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (runner, duration) = CachedGpuPackF16PairsRunner::new_with_context(context, len)
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu pack f16 pairs init failed: {error}"))
            })?;
        let runner = Rc::new(RefCell::new(runner));
        self.cached_pack_f16_pairs_gpu
            .borrow_mut()
            .insert(len, runner.clone());
        Ok((runner, duration, false))
    }

    fn get_or_create_vector_add_gpu(
        &self,
        len: usize,
    ) -> Result<CachedVectorAddGpuCacheEntry, ReferenceError> {
        if let Some(cached) = self.cached_vector_add_gpu.borrow().get(&len).cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (runner, duration) =
            CachedGpuVectorAddRunner::new_with_context(context, len).map_err(|error| {
                ReferenceError::Decode(format!("gpu vector add init failed: {error}"))
            })?;
        let runner = Rc::new(RefCell::new(runner));
        self.cached_vector_add_gpu
            .borrow_mut()
            .insert(len, runner.clone());
        Ok((runner, duration, false))
    }

    fn get_or_create_swiglu_combined_gpu(
        &self,
    ) -> Result<CachedSwigluCombinedGpuCacheEntry, ReferenceError> {
        if let Some(cached) = self.cached_swiglu_combined_gpu.borrow().as_ref().cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (runner, duration) =
            CachedGpuSwigluCombinedRunner::new_with_context(context, self.config.intermediate_size)
                .map_err(|error| {
                    ReferenceError::Decode(format!("gpu combined swiglu init failed: {error}"))
                })?;
        let runner = Rc::new(RefCell::new(runner));
        *self.cached_swiglu_combined_gpu.borrow_mut() = Some(runner.clone());
        Ok((runner, duration, false))
    }

    fn get_or_create_swiglu_pack_f16_pairs_gpu(
        &self,
    ) -> Result<CachedSwigluPackF16PairsGpuCacheEntry, ReferenceError> {
        if let Some(cached) = self
            .cached_swiglu_pack_f16_pairs_gpu
            .borrow()
            .as_ref()
            .cloned()
        {
            return Ok((cached, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (runner, duration) = CachedGpuSwigluPackF16PairsRunner::new_with_context(
            context,
            self.config.intermediate_size,
        )
        .map_err(|error| {
            ReferenceError::Decode(format!("gpu fused swiglu pack init failed: {error}"))
        })?;
        let runner = Rc::new(RefCell::new(runner));
        *self.cached_swiglu_pack_f16_pairs_gpu.borrow_mut() = Some(runner.clone());
        Ok((runner, duration, false))
    }

    fn get_or_create_packed_gpu_context(
        &self,
    ) -> Result<Arc<SharedGpuPackedContext>, ReferenceError> {
        if let Some(context) = self.packed_gpu_context.borrow().as_ref().cloned() {
            return Ok(context);
        }
        let context = SharedGpuPackedContext::new()
            .map_err(|error| ReferenceError::Decode(format!("gpu context init failed: {error}")))?;
        *self.packed_gpu_context.borrow_mut() = Some(context.clone());
        Ok(context)
    }

    fn get_or_create_tail_block_gpu(
        &self,
    ) -> Result<(CachedTailBlockGpuRunner, Duration, bool), ReferenceError> {
        if let Some(cached) = self.cached_tail_block_gpu.borrow().as_ref().cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let (logits_packed, _pack_duration, _pack_cache_hit) = self.get_or_create_projection_cache(
            "model.embed_tokens.weight",
            self.config.vocab_size,
            self.config.hidden_size,
        )?;
        let logits_spec = PackedLinearSpec {
            code_words: logits_packed.code_words.clone(),
            scales: logits_packed.scales.clone(),
            group_size: logits_packed.group_size,
            rows: logits_packed.rows,
            cols: logits_packed.cols,
        };
        let context = self.get_or_create_packed_gpu_context()?;
        let runner = CachedGpuTailBlockRunner::new_with_context(
            context,
            self.config.hidden_size,
            self.config.vocab_size,
            self.config.rms_norm_eps as f32,
            &logits_spec,
        )
        .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let duration = runner.compile_duration();
        let runner = Rc::new(RefCell::new(runner));
        *self.cached_tail_block_gpu.borrow_mut() = Some(runner.clone());
        Ok((runner, duration, false))
    }

    fn get_or_create_full_last_layer_block_gpu(
        &self,
    ) -> Result<(CachedFullLastLayerGpuRunner, Duration, bool), ReferenceError> {
        if let Some(cached) = self
            .cached_full_last_layer_block_gpu
            .borrow()
            .as_ref()
            .cloned()
        {
            return Ok((cached, Duration::ZERO, true));
        }
        let layer_idx = self.config.num_hidden_layers - 1;
        let layer_tensors = &self.layer_tensors[layer_idx];
        let pair_cache_key = format!(
            "gpu_first::last_layer::mlp_pair::{}+{}",
            layer_tensors.gate_proj_weight, layer_tensors.up_proj_weight
        );
        let (pair_packed, _pair_pack_duration, _pair_pack_cache_hit) =
            self.get_or_create_projection_pair_cache(
                &pair_cache_key,
                &layer_tensors.gate_proj_weight,
                self.config.intermediate_size,
                &layer_tensors.up_proj_weight,
                self.config.intermediate_size,
                self.config.hidden_size,
            )?;
        let pair_spec = PackedLinearSpec {
            code_words: pair_packed.code_words.clone(),
            scales: pair_packed.scales.clone(),
            group_size: pair_packed.group_size,
            rows: pair_packed.rows,
            cols: pair_packed.cols,
        };
        let (down_packed, _down_pack_duration, _down_pack_cache_hit) = self
            .get_or_create_projection_cache(
                &layer_tensors.down_proj_weight,
                self.config.hidden_size,
                self.config.intermediate_size,
            )?;
        let down_spec = PackedLinearSpec {
            code_words: down_packed.code_words.clone(),
            scales: down_packed.scales.clone(),
            group_size: down_packed.group_size,
            rows: down_packed.rows,
            cols: down_packed.cols,
        };
        let (logits_packed, _logits_pack_duration, _logits_pack_cache_hit) =
            self.get_or_create_projection_cache(
                "model.embed_tokens.weight",
                self.config.vocab_size,
                self.config.hidden_size,
            )?;
        let logits_spec = PackedLinearSpec {
            code_words: logits_packed.code_words.clone(),
            scales: logits_packed.scales.clone(),
            group_size: logits_packed.group_size,
            rows: logits_packed.rows,
            cols: logits_packed.cols,
        };
        let context = self.get_or_create_packed_gpu_context()?;
        let runner = CachedGpuFullLastLayerRunner::new_with_context(
            context,
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.vocab_size,
            self.config.rms_norm_eps as f32,
            &pair_spec,
            &down_spec,
            &logits_spec,
        )
        .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let duration = runner.compile_duration();
        let runner = Rc::new(RefCell::new(runner));
        *self.cached_full_last_layer_block_gpu.borrow_mut() = Some(runner.clone());
        Ok((runner, duration, false))
    }

    pub fn benchmark_attention_projection_mix(
        &self,
        layer_idx: usize,
        token_id: usize,
        use_q: bool,
        use_k: bool,
        use_v: bool,
        use_o: bool,
    ) -> Result<AttentionProjectionMixMetrics, ReferenceError> {
        let prefix = format!("model.layers.{layer_idx}");
        let hidden_states = self.layer_input_hidden(token_id, layer_idx)?;
        let q_rows = self.config.hidden_size;
        let kv_rows = self.config.num_key_value_heads * self.config.head_dim;
        let cols = self.config.hidden_size;

        let mut session = PackedGpuSession::new(self);

        let q_name = format!("{prefix}.self_attn.q_proj.weight");
        let k_name = format!("{prefix}.self_attn.k_proj.weight");
        let v_name = format!("{prefix}.self_attn.v_proj.weight");
        let o_name = format!("{prefix}.self_attn.o_proj.weight");

        let total_started = Instant::now();
        let mut q = if use_q {
            session.run_projection(&q_name, q_rows, cols, &hidden_states)?
        } else {
            self.weights.matvec_f16(&q_name, &hidden_states)?
        };
        let mut k = if use_k {
            session.run_projection(&k_name, kv_rows, cols, &hidden_states)?
        } else {
            self.weights.matvec_f16(&k_name, &hidden_states)?
        };
        let v = if use_v {
            session.run_projection(&v_name, kv_rows, cols, &hidden_states)?
        } else {
            self.weights.matvec_f16(&v_name, &hidden_states)?
        };

        let q_norm_weight = self
            .weights
            .load_vector_f32(&format!("{prefix}.self_attn.q_norm.weight"))?;
        let k_norm_weight = self
            .weights
            .load_vector_f32(&format!("{prefix}.self_attn.k_norm.weight"))?;
        let (cos, sin) = rope_cos_sin(&self.rope, &[0]);
        apply_head_rms_norm_weighted(
            &mut q,
            self.config.num_attention_heads,
            self.config.head_dim,
            &q_norm_weight,
            self.config.rms_norm_eps as f32,
        );
        apply_head_rms_norm_weighted(
            &mut k,
            self.config.num_key_value_heads,
            self.config.head_dim,
            &k_norm_weight,
            self.config.rms_norm_eps as f32,
        );
        apply_rotary_single(
            &mut q,
            &mut k,
            &cos,
            &sin,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        );

        let attn = attention_single_query(
            &q,
            &k,
            &v,
            1,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        );

        let mixed_o = if use_o {
            session.run_projection(&o_name, q_rows, q_rows, &attn)?
        } else {
            self.weights.matvec_f16(&o_name, &attn)?
        };
        let total_duration = total_started.elapsed();

        let dense_q = self.weights.matvec_f16(&q_name, &hidden_states)?;
        let mut dense_k = self.weights.matvec_f16(&k_name, &hidden_states)?;
        let dense_v = self.weights.matvec_f16(&v_name, &hidden_states)?;
        let mut dense_q_normed = dense_q.clone();
        apply_head_rms_norm_weighted(
            &mut dense_q_normed,
            self.config.num_attention_heads,
            self.config.head_dim,
            &q_norm_weight,
            self.config.rms_norm_eps as f32,
        );
        apply_head_rms_norm_weighted(
            &mut dense_k,
            self.config.num_key_value_heads,
            self.config.head_dim,
            &k_norm_weight,
            self.config.rms_norm_eps as f32,
        );
        apply_rotary_single(
            &mut dense_q_normed,
            &mut dense_k,
            &cos,
            &sin,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        );
        let dense_attn = attention_single_query(
            &dense_q_normed,
            &dense_k,
            &dense_v,
            1,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        );
        let dense_o = self.weights.matvec_f16(&o_name, &dense_attn)?;
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        for (left, right) in mixed_o.iter().zip(dense_o.iter()) {
            let diff = (left - right).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff;
        }
        let mean_abs_diff = sum_abs_diff / mixed_o.len() as f32;

        let enabled_projections = [("q", use_q), ("k", use_k), ("v", use_v), ("o", use_o)]
            .into_iter()
            .filter_map(|(name, enabled)| enabled.then_some(name))
            .collect::<Vec<_>>()
            .join("");

        Ok(AttentionProjectionMixMetrics {
            enabled_projections,
            total_duration,
            pack_duration: session.metrics.pack_duration,
            compile_duration: session.metrics.compile_duration,
            upload_duration: session.metrics.upload_duration,
            gpu_duration: session.metrics.gpu_duration,
            download_duration: session.metrics.download_duration,
            max_abs_diff,
            mean_abs_diff,
        })
    }

    pub fn benchmark_mlp_projection_mix(
        &self,
        layer_idx: usize,
        token_id: usize,
        use_gate: bool,
        use_up: bool,
        use_down: bool,
    ) -> Result<MlpProjectionMixMetrics, ReferenceError> {
        let prefix = format!("model.layers.{layer_idx}");
        let hidden = self
            .weights
            .embedding_lookup("model.embed_tokens.weight", token_id)?;
        let input_norm_weight = self
            .weights
            .load_vector_f32(&format!("{prefix}.input_layernorm.weight"))?;
        let mut hidden_states =
            weighted_rms_norm(&hidden, &input_norm_weight, self.config.rms_norm_eps as f32);

        let q_name = format!("{prefix}.self_attn.q_proj.weight");
        let k_name = format!("{prefix}.self_attn.k_proj.weight");
        let v_name = format!("{prefix}.self_attn.v_proj.weight");
        let o_name = format!("{prefix}.self_attn.o_proj.weight");
        let q_norm_weight = self
            .weights
            .load_vector_f32(&format!("{prefix}.self_attn.q_norm.weight"))?;
        let k_norm_weight = self
            .weights
            .load_vector_f32(&format!("{prefix}.self_attn.k_norm.weight"))?;
        let (cos, sin) = rope_cos_sin(&self.rope, &[0]);

        let mut q = self.weights.matvec_f16(&q_name, &hidden_states)?;
        let mut k = self.weights.matvec_f16(&k_name, &hidden_states)?;
        let v = self.weights.matvec_f16(&v_name, &hidden_states)?;
        apply_head_rms_norm_weighted(
            &mut q,
            self.config.num_attention_heads,
            self.config.head_dim,
            &q_norm_weight,
            self.config.rms_norm_eps as f32,
        );
        apply_head_rms_norm_weighted(
            &mut k,
            self.config.num_key_value_heads,
            self.config.head_dim,
            &k_norm_weight,
            self.config.rms_norm_eps as f32,
        );
        apply_rotary_single(
            &mut q,
            &mut k,
            &cos,
            &sin,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        );
        let attn = attention_single_query(
            &q,
            &k,
            &v,
            1,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        );
        let attn_output = self.weights.matvec_f16(&o_name, &attn)?;
        let hidden_after_attention = hidden
            .iter()
            .zip(attn_output.iter())
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        let post_norm_weight = self
            .weights
            .load_vector_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;
        hidden_states = weighted_rms_norm(
            &hidden_after_attention,
            &post_norm_weight,
            self.config.rms_norm_eps as f32,
        );

        let gate_name = format!("{prefix}.mlp.gate_proj.weight");
        let up_name = format!("{prefix}.mlp.up_proj.weight");
        let down_name = format!("{prefix}.mlp.down_proj.weight");
        let hidden_rows = self.config.hidden_size;
        let intermediate_rows = self.config.intermediate_size;

        let mut session = PackedGpuSession::new(self);

        let total_started = Instant::now();
        let gate = if use_gate {
            session.run_projection(&gate_name, intermediate_rows, hidden_rows, &hidden_states)?
        } else {
            self.weights.matvec_f16(&gate_name, &hidden_states)?
        };
        let up = if use_up {
            session.run_projection(&up_name, intermediate_rows, hidden_rows, &hidden_states)?
        } else {
            self.weights.matvec_f16(&up_name, &hidden_states)?
        };
        let mlp = swiglu(&gate, &up);

        let mixed_down = if use_down {
            session.run_projection(&down_name, hidden_rows, intermediate_rows, &mlp)?
        } else {
            self.weights.matvec_f16(&down_name, &mlp)?
        };
        let total_duration = total_started.elapsed();

        let dense_gate = self.weights.matvec_f16(&gate_name, &hidden_states)?;
        let dense_up = self.weights.matvec_f16(&up_name, &hidden_states)?;
        let dense_mlp = swiglu(&dense_gate, &dense_up);
        let dense_down = self.weights.matvec_f16(&down_name, &dense_mlp)?;
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        for (left, right) in mixed_down.iter().zip(dense_down.iter()) {
            let diff = (left - right).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff;
        }
        let mean_abs_diff = sum_abs_diff / mixed_down.len() as f32;
        let enabled_projections = [("g", use_gate), ("u", use_up), ("d", use_down)]
            .into_iter()
            .filter_map(|(name, enabled)| enabled.then_some(name))
            .collect::<Vec<_>>()
            .join("");

        Ok(MlpProjectionMixMetrics {
            enabled_projections,
            total_duration,
            pack_duration: session.metrics.pack_duration,
            compile_duration: session.metrics.compile_duration,
            upload_duration: session.metrics.upload_duration,
            gpu_duration: session.metrics.gpu_duration,
            download_duration: session.metrics.download_duration,
            max_abs_diff,
            mean_abs_diff,
        })
    }

    pub fn benchmark_cached_hybrid_qkv_gu_decode(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        layer_idx: usize,
    ) -> Result<HybridProjectionDecodeMetrics, ReferenceError> {
        self.benchmark_cached_hybrid_attention_mlp_decode(
            prompt,
            max_new_tokens,
            layer_idx,
            false,
            false,
            false,
        )
    }

    pub fn benchmark_cached_hybrid_qkvo_gu_decode(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        layer_idx: usize,
    ) -> Result<HybridProjectionDecodeMetrics, ReferenceError> {
        self.benchmark_cached_hybrid_attention_mlp_decode(
            prompt,
            max_new_tokens,
            layer_idx,
            true,
            false,
            false,
        )
    }

    pub fn benchmark_cached_hybrid_qkv_gud_decode(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        layer_idx: usize,
    ) -> Result<HybridProjectionDecodeMetrics, ReferenceError> {
        self.benchmark_cached_hybrid_attention_mlp_decode(
            prompt,
            max_new_tokens,
            layer_idx,
            false,
            true,
            false,
        )
    }

    pub fn benchmark_cached_hybrid_qkv_gu_logits_decode(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        layer_idx: usize,
    ) -> Result<HybridProjectionDecodeMetrics, ReferenceError> {
        self.benchmark_cached_hybrid_attention_mlp_decode(
            prompt,
            max_new_tokens,
            layer_idx,
            false,
            false,
            true,
        )
    }

    fn benchmark_cached_hybrid_attention_mlp_decode(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        layer_idx: usize,
        use_o_proj: bool,
        use_down_proj: bool,
        use_logits_proj: bool,
    ) -> Result<HybridProjectionDecodeMetrics, ReferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ReferenceError::Decode("tokenizer is not loaded".to_string()))?;
        let prompt_ids = self.encode_prompt_ids(tokenizer, prompt)?;
        let prefix = format!("model.layers.{layer_idx}");

        let q_name = format!("{prefix}.self_attn.q_proj.weight");
        let k_name = format!("{prefix}.self_attn.k_proj.weight");
        let v_name = format!("{prefix}.self_attn.v_proj.weight");
        let o_name = format!("{prefix}.self_attn.o_proj.weight");
        let gate_name = format!("{prefix}.mlp.gate_proj.weight");
        let up_name = format!("{prefix}.mlp.up_proj.weight");
        let down_name = format!("{prefix}.mlp.down_proj.weight");
        let logits_name = "model.embed_tokens.weight";

        let hidden_rows = self.config.hidden_size;
        let kv_rows = self.config.num_key_value_heads * self.config.head_dim;
        let intermediate_rows = self.config.intermediate_size;

        let mut pack_duration = Duration::ZERO;
        let mut compile_duration = Duration::ZERO;

        let (q_packed, pack, _) =
            self.get_or_create_projection_cache(&q_name, hidden_rows, hidden_rows)?;
        let (q_gpu, compile, weight_upload, _) =
            self.get_or_create_projection_gpu(&q_name, &q_packed)?;
        pack_duration += pack;
        compile_duration += compile;
        let mut upload_duration = weight_upload;

        let (k_packed, pack, _) =
            self.get_or_create_projection_cache(&k_name, kv_rows, hidden_rows)?;
        let (k_gpu, compile, weight_upload, _) =
            self.get_or_create_projection_gpu(&k_name, &k_packed)?;
        pack_duration += pack;
        compile_duration += compile;
        upload_duration += weight_upload;

        let (v_packed, pack, _) =
            self.get_or_create_projection_cache(&v_name, kv_rows, hidden_rows)?;
        let (v_gpu, compile, weight_upload, _) =
            self.get_or_create_projection_gpu(&v_name, &v_packed)?;
        pack_duration += pack;
        compile_duration += compile;
        upload_duration += weight_upload;

        let o_gpu = if use_o_proj {
            let (o_packed, pack, _) =
                self.get_or_create_projection_cache(&o_name, hidden_rows, hidden_rows)?;
            let (o_gpu, compile, weight_upload, _) =
                self.get_or_create_projection_gpu(&o_name, &o_packed)?;
            pack_duration += pack;
            compile_duration += compile;
            upload_duration += weight_upload;
            Some(o_gpu)
        } else {
            None
        };

        let (gate_packed, pack, _) =
            self.get_or_create_projection_cache(&gate_name, intermediate_rows, hidden_rows)?;
        let (gate_gpu, compile, weight_upload, _) =
            self.get_or_create_projection_gpu(&gate_name, &gate_packed)?;
        pack_duration += pack;
        compile_duration += compile;
        upload_duration += weight_upload;

        let (up_packed, pack, _) =
            self.get_or_create_projection_cache(&up_name, intermediate_rows, hidden_rows)?;
        let (up_gpu, compile, weight_upload, _) =
            self.get_or_create_projection_gpu(&up_name, &up_packed)?;
        pack_duration += pack;
        compile_duration += compile;
        upload_duration += weight_upload;

        let down_gpu = if use_down_proj {
            let (down_packed, pack, _) =
                self.get_or_create_projection_cache(&down_name, hidden_rows, intermediate_rows)?;
            let (down_gpu, compile, weight_upload, _) =
                self.get_or_create_projection_gpu(&down_name, &down_packed)?;
            pack_duration += pack;
            compile_duration += compile;
            upload_duration += weight_upload;
            Some(down_gpu)
        } else {
            None
        };

        let logits_gpu = if use_logits_proj {
            let (logits_packed, pack, _) = self.get_or_create_projection_cache(
                logits_name,
                self.config.vocab_size,
                hidden_rows,
            )?;
            let (logits_gpu, compile, weight_upload, _) =
                self.get_or_create_projection_gpu(logits_name, &logits_packed)?;
            pack_duration += pack;
            compile_duration += compile;
            upload_duration += weight_upload;
            Some(logits_gpu)
        } else {
            None
        };

        let total_started = Instant::now();
        let mut metrics = DecodeMetrics {
            prompt_tokens: prompt_ids.len(),
            generated_tokens: 0,
            total_duration: Duration::ZERO,
            embedding_duration: Duration::ZERO,
            norm_duration: Duration::ZERO,
            qkv_duration: Duration::ZERO,
            attention_duration: Duration::ZERO,
            mlp_duration: Duration::ZERO,
            logits_duration: Duration::ZERO,
        };
        let mut cache = self.allocate_layer_cache_vec(prompt_ids.len() + max_new_tokens, true);
        let mut upload = upload_duration;
        let mut gpu = Duration::ZERO;
        let mut download = Duration::ZERO;
        let mut last_next_token = None;

        for (position, &token_id) in prompt_ids.iter().enumerate() {
            let (next_token, u, g, d) = self.forward_step_hybrid_qkv_gu(
                token_id,
                position,
                &mut cache,
                &mut metrics,
                layer_idx,
                &q_gpu,
                &k_gpu,
                &v_gpu,
                o_gpu.as_ref(),
                &gate_gpu,
                &up_gpu,
                down_gpu.as_ref(),
                logits_gpu.as_ref(),
            )?;
            last_next_token = Some(next_token);
            upload += u;
            gpu += g;
            download += d;
        }
        let mut output_ids = prompt_ids.to_vec();
        for generation_index in 0..max_new_tokens {
            let next_token = last_next_token.ok_or_else(|| {
                ReferenceError::Decode("argmax failed on empty logits".to_string())
            })?;
            output_ids.push(next_token);
            metrics.generated_tokens += 1;
            let (predicted_next_token, u, g, d) = self.forward_step_hybrid_qkv_gu(
                next_token,
                prompt_ids.len() + generation_index,
                &mut cache,
                &mut metrics,
                layer_idx,
                &q_gpu,
                &k_gpu,
                &v_gpu,
                o_gpu.as_ref(),
                &gate_gpu,
                &up_gpu,
                down_gpu.as_ref(),
                logits_gpu.as_ref(),
            )?;
            last_next_token = Some(predicted_next_token);
            upload += u;
            gpu += g;
            download += d;
        }
        let output_text =
            tokenizer.decode(&output_ids.iter().map(|id| *id as u32).collect::<Vec<_>>())?;

        Ok(HybridProjectionDecodeMetrics {
            enabled_projections: {
                let mut label = if use_o_proj {
                    "qkvo".to_string()
                } else {
                    "qkv".to_string()
                };
                label.push_str("+gu");
                if use_down_proj {
                    label.push('d');
                }
                if use_logits_proj {
                    label.push_str("+logits");
                }
                label
            },
            total_duration: total_started.elapsed(),
            pack_duration,
            compile_duration,
            upload_duration: upload,
            gpu_duration: gpu,
            download_duration: download,
            output_text,
        })
    }

    pub fn benchmark_dense_step_from_token_ids(
        &self,
        prompt_ids: &[usize],
    ) -> Result<DecodeMetrics, ReferenceError> {
        if prompt_ids.is_empty() {
            return Err(ReferenceError::Decode(
                "prompt_ids cannot be empty".to_string(),
            ));
        }

        let total_started = Instant::now();
        let mut metrics = DecodeMetrics {
            prompt_tokens: prompt_ids.len(),
            generated_tokens: 0,
            total_duration: Duration::ZERO,
            embedding_duration: Duration::ZERO,
            norm_duration: Duration::ZERO,
            qkv_duration: Duration::ZERO,
            attention_duration: Duration::ZERO,
            mlp_duration: Duration::ZERO,
            logits_duration: Duration::ZERO,
        };
        let mut cache = self.allocate_layer_cache_vec(prompt_ids.len(), true);
        for (position, &token_id) in prompt_ids.iter().enumerate() {
            let _ = self.forward_step(token_id, position, &mut cache, &mut metrics)?;
        }
        metrics.total_duration = total_started.elapsed();
        Ok(metrics)
    }

    pub fn benchmark_packed_step_from_token_ids(
        &self,
        prompt_ids: &[usize],
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<PackedDecodeMetrics, ReferenceError> {
        if prompt_ids.is_empty() {
            return Err(ReferenceError::Decode(
                "prompt_ids cannot be empty".to_string(),
            ));
        }

        let total_started = Instant::now();
        let mut session =
            self.begin_packed_decode_session(prompt_ids.len(), use_attention_qkv, use_mlp_gu, true);
        for (position, &token_id) in prompt_ids.iter().enumerate() {
            debug_assert_eq!(position, session.next_position());
            let _ = session.push_prompt_token(token_id)?;
        }

        Ok(session.finish_metrics(
            Self::packed_enabled_label(use_attention_qkv, use_mlp_gu),
            total_started.elapsed(),
            String::new(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn benchmark_packed_prefill_chunk(
        &self,
        token_id: Option<usize>,
        hidden_in: Option<&[f32]>,
        start_layer: usize,
        end_layer: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        include_logits: bool,
    ) -> Result<(Vec<f32>, PackedDecodeMetrics), ReferenceError> {
        if start_layer >= end_layer || end_layer > self.config.num_hidden_layers {
            return Err(ReferenceError::Decode(format!(
                "invalid layer range [{start_layer}, {end_layer}) for {} layers",
                self.config.num_hidden_layers
            )));
        }

        let total_started = Instant::now();
        let mut metrics = DecodeMetrics {
            prompt_tokens: 1,
            generated_tokens: 0,
            total_duration: Duration::ZERO,
            embedding_duration: Duration::ZERO,
            norm_duration: Duration::ZERO,
            qkv_duration: Duration::ZERO,
            attention_duration: Duration::ZERO,
            mlp_duration: Duration::ZERO,
            logits_duration: Duration::ZERO,
        };
        let mut session = PackedGpuSession::new(self);
        let mut non_offloaded_dense_duration = Duration::ZERO;
        let mut attention_stage_metrics = PackedAttentionStageMetrics::default();
        let mut mlp_stage_metrics = PackedMlpStageMetrics::default();
        let mut hidden = if let Some(hidden_in) = hidden_in {
            hidden_in.to_vec()
        } else {
            let token_id = token_id.ok_or_else(|| {
                ReferenceError::Decode(
                    "token_id is required when hidden_in is not provided".to_string(),
                )
            })?;
            let started_at = Instant::now();
            let hidden = self.embedding_lookup_resolved("model.embed_tokens.weight", token_id)?;
            let elapsed = started_at.elapsed();
            metrics.embedding_duration += elapsed;
            non_offloaded_dense_duration += elapsed;
            hidden
        };

        let (cos, sin) = rope_cos_sin(&self.rope, &[0]);

        for layer_idx in start_layer..end_layer {
            let layer_tensors = &self.layer_tensors[layer_idx];

            let started_at = Instant::now();
            let input_norm_weight =
                self.load_vector_f32_resolved(&layer_tensors.input_layernorm_weight)?;
            let mut hidden_states =
                weighted_rms_norm(&hidden, &input_norm_weight, self.config.rms_norm_eps as f32);
            let elapsed = started_at.elapsed();
            metrics.norm_duration += elapsed;
            non_offloaded_dense_duration += elapsed;
            let residual = hidden.clone();

            let started_at = Instant::now();
            let kv_rows = self.config.num_key_value_heads * self.config.head_dim;
            let (mut q, mut k, v) = if use_attention_qkv {
                session.run_projection_triplet(
                    &layer_tensors.q_proj_weight,
                    self.config.hidden_size,
                    &layer_tensors.k_proj_weight,
                    kv_rows,
                    &layer_tensors.v_proj_weight,
                    kv_rows,
                    self.config.hidden_size,
                    &hidden_states,
                )?
            } else {
                (
                    self.matvec_f16_resolved(&layer_tensors.q_proj_weight, &hidden_states)?,
                    self.matvec_f16_resolved(&layer_tensors.k_proj_weight, &hidden_states)?,
                    self.matvec_f16_resolved(&layer_tensors.v_proj_weight, &hidden_states)?,
                )
            };
            let elapsed = started_at.elapsed();
            metrics.qkv_duration += elapsed;
            if !use_attention_qkv {
                non_offloaded_dense_duration += elapsed;
            }

            let started_at = Instant::now();
            let q_norm_weight = self.load_vector_f32_resolved(&layer_tensors.q_norm_weight)?;
            let k_norm_weight = self.load_vector_f32_resolved(&layer_tensors.k_norm_weight)?;
            apply_head_rms_norm_weighted(
                &mut q,
                self.config.num_attention_heads,
                self.config.head_dim,
                &q_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            apply_head_rms_norm_weighted(
                &mut k,
                self.config.num_key_value_heads,
                self.config.head_dim,
                &k_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            apply_rotary_single(
                &mut q,
                &mut k,
                &cos,
                &sin,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            );
            let elapsed = started_at.elapsed();
            metrics.norm_duration += elapsed;
            non_offloaded_dense_duration += elapsed;

            let started_at = Instant::now();
            let attn_started_at = Instant::now();
            let attn = attention_single_query(
                &q,
                &k,
                &v,
                1,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            );
            let attn_elapsed = attn_started_at.elapsed();
            attention_stage_metrics.query_duration += attn_elapsed;
            session.push_dense_stage_trace(
                "attention_core",
                "attention_single_query",
                attn_elapsed,
            );
            let use_attention_full = use_attention_qkv && Self::packed_use_attention_full();
            let oproj_started_at = Instant::now();
            let attn_output = if use_attention_full {
                session.run_projection(
                    &layer_tensors.o_proj_weight,
                    self.config.hidden_size,
                    self.config.hidden_size,
                    &attn,
                )?
            } else {
                self.matvec_f16_resolved(&layer_tensors.o_proj_weight, &attn)?
            };
            let oproj_elapsed = oproj_started_at.elapsed();
            attention_stage_metrics.oproj_duration += oproj_elapsed;
            let residual_started_at = Instant::now();
            hidden = residual
                .iter()
                .zip(attn_output.iter())
                .map(|(left, right)| left + right)
                .collect();
            let residual_elapsed = residual_started_at.elapsed();
            attention_stage_metrics.residual_duration += residual_elapsed;
            session.push_dense_stage_trace(
                "attention_residual",
                "attention_residual_add",
                residual_elapsed,
            );
            let elapsed = started_at.elapsed();
            metrics.attention_duration += elapsed;
            if use_attention_qkv {
                if use_attention_full {
                    non_offloaded_dense_duration += attn_elapsed + residual_elapsed;
                } else {
                    non_offloaded_dense_duration += elapsed;
                }
            } else {
                non_offloaded_dense_duration += elapsed;
            }

            let residual = hidden.clone();
            let started_at = Instant::now();
            let post_norm_weight =
                self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
            hidden_states = weighted_rms_norm(
                &residual,
                &post_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            let elapsed = started_at.elapsed();
            metrics.norm_duration += elapsed;
            non_offloaded_dense_duration += elapsed;
            session.push_dense_stage_trace(
                "post_attention_norm",
                &layer_tensors.post_attention_layernorm_weight,
                elapsed,
            );

            let started_at = Instant::now();
            let (gate, up, dense_tail_started_at) = if use_mlp_gu {
                session
                    .run_projection_pair(
                        &layer_tensors.gate_proj_weight,
                        self.config.intermediate_size,
                        &layer_tensors.up_proj_weight,
                        self.config.intermediate_size,
                        self.config.hidden_size,
                        &hidden_states,
                    )
                    .map(|(gate, up)| (gate, up, Instant::now()))?
            } else {
                let dense_started_at = Instant::now();
                (
                    self.matvec_f16_resolved(&layer_tensors.gate_proj_weight, &hidden_states)?,
                    self.matvec_f16_resolved(&layer_tensors.up_proj_weight, &hidden_states)?,
                    dense_started_at,
                )
            };
            let use_mlp_full = use_mlp_gu && Self::packed_use_mlp_full();
            let swiglu_started_at = Instant::now();
            let mlp = swiglu(&gate, &up);
            let swiglu_elapsed = swiglu_started_at.elapsed();
            mlp_stage_metrics.swiglu_duration += swiglu_elapsed;
            session.push_dense_stage_trace("mlp_swiglu", "swiglu", swiglu_elapsed);
            let down_started_at = Instant::now();
            let down = if use_mlp_full {
                session.run_projection(
                    &layer_tensors.down_proj_weight,
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    &mlp,
                )?
            } else {
                self.matvec_f16_resolved(&layer_tensors.down_proj_weight, &mlp)?
            };
            let down_elapsed = down_started_at.elapsed();
            mlp_stage_metrics.down_duration += down_elapsed;
            let residual_started_at = Instant::now();
            hidden = residual
                .iter()
                .zip(down.iter())
                .map(|(left, right)| left + right)
                .collect();
            let residual_elapsed = residual_started_at.elapsed();
            mlp_stage_metrics.residual_duration += residual_elapsed;
            session.push_dense_stage_trace("mlp_residual", "mlp_residual_add", residual_elapsed);
            let elapsed = started_at.elapsed();
            metrics.mlp_duration += elapsed;
            if use_mlp_gu {
                if use_mlp_full {
                    non_offloaded_dense_duration += swiglu_elapsed + residual_elapsed;
                } else {
                    non_offloaded_dense_duration += dense_tail_started_at.elapsed();
                }
            } else {
                non_offloaded_dense_duration += elapsed;
            }
        }

        if include_logits {
            let started_at = Instant::now();
            let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
            hidden =
                weighted_rms_norm(&hidden, &final_norm_weight, self.config.rms_norm_eps as f32);
            let elapsed = started_at.elapsed();
            metrics.norm_duration += elapsed;
            non_offloaded_dense_duration += elapsed;

            let started_at = Instant::now();
            let _ = session.run_projection_argmax(
                "model.embed_tokens.weight",
                self.config.vocab_size,
                self.config.hidden_size,
                &hidden,
            )?;
            metrics.logits_duration += started_at.elapsed();
        }

        Ok((
            hidden,
            Self::finish_packed_decode_metrics(
                Self::packed_enabled_label(use_attention_qkv, use_mlp_gu),
                total_started.elapsed(),
                &metrics,
                &attention_stage_metrics,
                &mlp_stage_metrics,
                non_offloaded_dense_duration,
                &session,
                String::new(),
            ),
        ))
    }

    pub fn benchmark_packed_decode(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<PackedDecodeMetrics, ReferenceError> {
        Ok(self
            .generate_packed_greedy(prompt, max_new_tokens, use_attention_qkv, use_mlp_gu)?
            .metrics)
    }

    pub fn generate_packed_greedy(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<PackedDecodeResult, ReferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ReferenceError::Decode("tokenizer is not loaded".to_string()))?;
        let prompt_ids = self.encode_prompt_ids(tokenizer, prompt)?;
        self.generate_packed_from_token_ids(
            &prompt_ids,
            max_new_tokens,
            use_attention_qkv,
            use_mlp_gu,
        )
    }

    pub fn generate_packed_from_token_ids(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<PackedDecodeResult, ReferenceError> {
        if prompt_ids.is_empty() {
            return Err(ReferenceError::Decode(
                "prompt_ids cannot be empty".to_string(),
            ));
        }
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ReferenceError::Decode("tokenizer is not loaded".to_string()))?;
        let total_started = Instant::now();
        let mut session = self.begin_packed_decode_session(
            prompt_ids.len() + max_new_tokens,
            use_attention_qkv,
            use_mlp_gu,
            true,
        );
        let mut last_next_token = None;

        for (position, &token_id) in prompt_ids.iter().enumerate() {
            debug_assert_eq!(position, session.next_position());
            let step = session.push_prompt_token(token_id)?;
            last_next_token = Some(match step {
                PackedDecodeStepResult::NextToken(token_id) => token_id,
                PackedDecodeStepResult::Logits(_) => {
                    unreachable!("argmax-only packed decode should not return logits")
                }
            });
        }

        let mut output_ids = prompt_ids.to_vec();
        for generation_index in 0..max_new_tokens {
            let next_token = last_next_token.ok_or_else(|| {
                ReferenceError::Decode("argmax failed on empty logits".to_string())
            })?;
            output_ids.push(next_token);
            debug_assert_eq!(prompt_ids.len() + generation_index, session.next_position());
            let step = session.push_generated_token(next_token)?;
            last_next_token = Some(match step {
                PackedDecodeStepResult::NextToken(token_id) => token_id,
                PackedDecodeStepResult::Logits(_) => {
                    unreachable!("argmax-only packed decode should not return logits")
                }
            });
        }

        let output_text =
            tokenizer.decode(&output_ids.iter().map(|id| *id as u32).collect::<Vec<_>>())?;

        Ok(session.finish_result(
            Self::packed_enabled_label(use_attention_qkv, use_mlp_gu),
            total_started.elapsed(),
            output_ids,
            output_text,
        ))
    }

    pub fn prefill_logits_for_variant(
        &self,
        prompt: &str,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<Vec<f32>, ReferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ReferenceError::Decode("tokenizer is not loaded".to_string()))?;
        let prompt_ids = self.encode_prompt_ids(tokenizer, prompt)?;
        let mut last_logits = Vec::new();
        let mut session = self.begin_packed_decode_session(
            prompt_ids.len(),
            use_attention_qkv,
            use_mlp_gu,
            false,
        );
        for &token_id in &prompt_ids {
            last_logits = match session.push_prompt_token(token_id)? {
                PackedDecodeStepResult::Logits(logits) => logits,
                PackedDecodeStepResult::NextToken(_) => {
                    unreachable!("full-logits prefill path should not return argmax-only output")
                }
            };
        }
        Ok(last_logits)
    }

    pub fn compare_prefill_logits_against(
        &self,
        dense_reference: &ReferenceModel,
        prompt: &str,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<PackedDecodeValidationReport, ReferenceError> {
        let dense_logits = dense_reference.prefill_logits_for_variant(prompt, false, false)?;
        let packed_logits =
            self.prefill_logits_for_variant(prompt, use_attention_qkv, use_mlp_gu)?;
        if dense_logits.len() != packed_logits.len() {
            return Err(ReferenceError::Decode(format!(
                "logit length mismatch: dense {} vs packed {}",
                dense_logits.len(),
                packed_logits.len()
            )));
        }
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        for (left, right) in dense_logits.iter().zip(packed_logits.iter()) {
            let diff = (left - right).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff;
        }
        let mean_abs_diff = if dense_logits.is_empty() {
            0.0
        } else {
            sum_abs_diff / dense_logits.len() as f32
        };
        let prompt_tokens = dense_reference.prompt_analysis(prompt)?.token_count;
        let mut enabled = String::new();
        if use_attention_qkv {
            enabled.push_str("qkv");
        }
        if use_mlp_gu {
            if !enabled.is_empty() {
                enabled.push('+');
            }
            enabled.push_str("gu");
        }
        if enabled.is_empty() {
            enabled.push_str("dense");
        }
        Ok(PackedDecodeValidationReport {
            enabled_projections: enabled,
            prompt_tokens,
            max_abs_diff,
            mean_abs_diff,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_hybrid_qproj_decode(
        &self,
        tokenizer: &TokenizerRuntime,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        hybrid: &HybridQProjCache,
        q_proj_pack_duration: Duration,
        q_proj_pack_cache_hit: bool,
        q_proj_gpu_compile_duration: Duration,
        q_proj_gpu_cache_hit: bool,
        gpu_runner: Option<Rc<RefCell<CachedGpuPackedMatvecRunner>>>,
    ) -> Result<HybridDecodeMetrics, ReferenceError> {
        let total_started = Instant::now();
        let mut metrics = DecodeMetrics {
            prompt_tokens: prompt_ids.len(),
            generated_tokens: 0,
            total_duration: Duration::ZERO,
            embedding_duration: Duration::ZERO,
            norm_duration: Duration::ZERO,
            qkv_duration: Duration::ZERO,
            attention_duration: Duration::ZERO,
            mlp_duration: Duration::ZERO,
            logits_duration: Duration::ZERO,
        };
        let mut cache = self.allocate_layer_cache_vec(prompt_ids.len() + max_new_tokens, true);
        let mut compile = q_proj_gpu_compile_duration;
        let mut upload = Duration::ZERO;
        let mut gpu = Duration::ZERO;
        let mut download = Duration::ZERO;
        let mut last_logits = Vec::new();

        for (position, &token_id) in prompt_ids.iter().enumerate() {
            let (logits, c, u, g, d) = self.forward_step_hybrid_qproj(
                token_id,
                position,
                &mut cache,
                &mut metrics,
                hybrid,
                gpu_runner.as_ref(),
            )?;
            last_logits = logits;
            compile += c;
            upload += u;
            gpu += g;
            download += d;
        }
        let mut output_ids = prompt_ids.to_vec();
        for generation_index in 0..max_new_tokens {
            let next_token = argmax(&last_logits).ok_or_else(|| {
                ReferenceError::Decode("argmax failed on empty logits".to_string())
            })?;
            output_ids.push(next_token);
            metrics.generated_tokens += 1;
            let (logits, c, u, g, d) = self.forward_step_hybrid_qproj(
                next_token,
                prompt_ids.len() + generation_index,
                &mut cache,
                &mut metrics,
                hybrid,
                gpu_runner.as_ref(),
            )?;
            last_logits = logits;
            compile += c;
            upload += u;
            gpu += g;
            download += d;
        }
        let output_text =
            tokenizer.decode(&output_ids.iter().map(|id| *id as u32).collect::<Vec<_>>())?;

        Ok(HybridDecodeMetrics {
            total_duration: total_started.elapsed(),
            q_proj_pack_duration,
            q_proj_pack_cache_hit,
            q_proj_gpu_compile_duration: compile,
            q_proj_gpu_cache_hit,
            q_proj_gpu_upload_duration: upload,
            q_proj_gpu_duration: gpu,
            q_proj_gpu_download_duration: download,
            output_text,
        })
    }

    fn forward_step(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut [LayerCache],
        metrics: &mut DecodeMetrics,
    ) -> Result<Vec<f32>, ReferenceError> {
        let started_at = Instant::now();
        let mut hidden = self.embedding_lookup_resolved("model.embed_tokens.weight", token_id)?;
        metrics.embedding_duration += started_at.elapsed();

        let (cos, sin) = rope_cos_sin(&self.rope, &[position]);

        for (layer_idx, layer_cache) in cache
            .iter_mut()
            .enumerate()
            .take(self.config.num_hidden_layers)
        {
            let layer_tensors = &self.layer_tensors[layer_idx];

            let started_at = Instant::now();
            let input_norm_weight =
                self.load_vector_f32_resolved(&layer_tensors.input_layernorm_weight)?;
            let mut hidden_states =
                weighted_rms_norm(&hidden, &input_norm_weight, self.config.rms_norm_eps as f32);
            metrics.norm_duration += started_at.elapsed();
            let residual = hidden.clone();

            let started_at = Instant::now();
            let mut q = self.matvec_f16_resolved(&layer_tensors.q_proj_weight, &hidden_states)?;
            let mut k = self.matvec_f16_resolved(&layer_tensors.k_proj_weight, &hidden_states)?;
            let v = self.matvec_f16_resolved(&layer_tensors.v_proj_weight, &hidden_states)?;
            metrics.qkv_duration += started_at.elapsed();

            let started_at = Instant::now();
            let q_norm_weight = self.load_vector_f32_resolved(&layer_tensors.q_norm_weight)?;
            let k_norm_weight = self.load_vector_f32_resolved(&layer_tensors.k_norm_weight)?;
            apply_head_rms_norm_weighted(
                &mut q,
                self.config.num_attention_heads,
                self.config.head_dim,
                &q_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            apply_head_rms_norm_weighted(
                &mut k,
                self.config.num_key_value_heads,
                self.config.head_dim,
                &k_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            apply_rotary_single(
                &mut q,
                &mut k,
                &cos,
                &sin,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            );
            metrics.norm_duration += started_at.elapsed();

            let started_at = Instant::now();
            layer_cache.keys.extend_from_slice(&k);
            layer_cache.values.extend_from_slice(&v);
            let attn = attention_single_query(
                &q,
                &layer_cache.keys,
                &layer_cache.values,
                position + 1,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            );
            let attn_output = self.matvec_f16_resolved(&layer_tensors.o_proj_weight, &attn)?;
            hidden = residual
                .iter()
                .zip(attn_output.iter())
                .map(|(left, right)| left + right)
                .collect();
            metrics.attention_duration += started_at.elapsed();

            let residual = hidden.clone();
            let started_at = Instant::now();
            let post_norm_weight =
                self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
            hidden_states = weighted_rms_norm(
                &residual,
                &post_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            metrics.norm_duration += started_at.elapsed();

            let started_at = Instant::now();
            let gate = self.matvec_f16_resolved(&layer_tensors.gate_proj_weight, &hidden_states)?;
            let up = self.matvec_f16_resolved(&layer_tensors.up_proj_weight, &hidden_states)?;
            let mlp = swiglu(&gate, &up);
            let down = self.matvec_f16_resolved(&layer_tensors.down_proj_weight, &mlp)?;
            hidden = residual
                .iter()
                .zip(down.iter())
                .map(|(left, right)| left + right)
                .collect();
            metrics.mlp_duration += started_at.elapsed();
        }

        let started_at = Instant::now();
        let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
        let hidden =
            weighted_rms_norm(&hidden, &final_norm_weight, self.config.rms_norm_eps as f32);
        metrics.norm_duration += started_at.elapsed();

        let started_at = Instant::now();
        let logits = self.matvec_f16_resolved("model.embed_tokens.weight", &hidden)?;
        metrics.logits_duration += started_at.elapsed();
        Ok(logits)
    }

    fn forward_step_hybrid_qproj(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut [LayerCache],
        metrics: &mut DecodeMetrics,
        hybrid: &HybridQProjCache,
        gpu_runner: Option<&Rc<RefCell<CachedGpuPackedMatvecRunner>>>,
    ) -> Result<(Vec<f32>, Duration, Duration, Duration, Duration), ReferenceError> {
        let started_at = Instant::now();
        let mut hidden = self
            .weights
            .embedding_lookup("model.embed_tokens.weight", token_id)?;
        metrics.embedding_duration += started_at.elapsed();

        let (cos, sin) = rope_cos_sin(&self.rope, &[position]);
        let mut compile = Duration::ZERO;
        let mut upload = Duration::ZERO;
        let mut gpu = Duration::ZERO;
        let mut download = Duration::ZERO;

        for (layer_idx, layer_cache) in cache
            .iter_mut()
            .enumerate()
            .take(self.config.num_hidden_layers)
        {
            let prefix = format!("model.layers.{layer_idx}");

            let started_at = Instant::now();
            let input_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.input_layernorm.weight"))?;
            let mut hidden_states =
                weighted_rms_norm(&hidden, &input_norm_weight, self.config.rms_norm_eps as f32);
            metrics.norm_duration += started_at.elapsed();
            let residual = hidden;

            let started_at = Instant::now();
            let mut q = if layer_idx == hybrid.layer_idx {
                let (out, report) = match gpu_runner {
                    Some(runner) => runner
                        .borrow_mut()
                        .run_with_output(&hidden_states, None)
                        .map_err(|error| {
                            ReferenceError::Decode(format!(
                                "cached gpu packed q_proj failed: {error}"
                            ))
                        })?,
                    None => run_packed_ternary_matvec_with_output(
                        &hybrid.code_words,
                        &hybrid.scales,
                        hybrid.group_size,
                        hybrid.rows,
                        hybrid.cols,
                        &hidden_states,
                        None,
                    )
                    .map_err(|error| {
                        ReferenceError::Decode(format!("gpu packed q_proj failed: {error}"))
                    })?,
                };
                compile += report.compile_duration;
                upload += report.upload_duration;
                gpu += report.gpu_duration;
                download += report.download_duration;
                out
            } else {
                self.weights
                    .matvec_f16(&format!("{prefix}.self_attn.q_proj.weight"), &hidden_states)?
            };
            let mut k = self
                .weights
                .matvec_f16(&format!("{prefix}.self_attn.k_proj.weight"), &hidden_states)?;
            let v = self
                .weights
                .matvec_f16(&format!("{prefix}.self_attn.v_proj.weight"), &hidden_states)?;
            metrics.qkv_duration += started_at.elapsed();

            let started_at = Instant::now();
            let q_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.self_attn.q_norm.weight"))?;
            let k_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.self_attn.k_norm.weight"))?;
            apply_head_rms_norm_weighted(
                &mut q,
                self.config.num_attention_heads,
                self.config.head_dim,
                &q_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            apply_head_rms_norm_weighted(
                &mut k,
                self.config.num_key_value_heads,
                self.config.head_dim,
                &k_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            apply_rotary_single(
                &mut q,
                &mut k,
                &cos,
                &sin,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            );
            metrics.norm_duration += started_at.elapsed();

            let started_at = Instant::now();
            layer_cache.keys.extend_from_slice(&k);
            layer_cache.values.extend_from_slice(&v);
            let attn = attention_single_query(
                &q,
                &layer_cache.keys,
                &layer_cache.values,
                position + 1,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            );
            let attn_output = self
                .weights
                .matvec_f16(&format!("{prefix}.self_attn.o_proj.weight"), &attn)?;
            hidden = residual
                .iter()
                .zip(attn_output.iter())
                .map(|(left, right)| left + right)
                .collect();
            metrics.attention_duration += started_at.elapsed();

            let residual = hidden;
            let started_at = Instant::now();
            let post_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;
            hidden_states = weighted_rms_norm(
                &residual,
                &post_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            metrics.norm_duration += started_at.elapsed();

            let started_at = Instant::now();
            let gate = self
                .weights
                .matvec_f16(&format!("{prefix}.mlp.gate_proj.weight"), &hidden_states)?;
            let up = self
                .weights
                .matvec_f16(&format!("{prefix}.mlp.up_proj.weight"), &hidden_states)?;
            let mlp = swiglu(&gate, &up);
            let down = self
                .weights
                .matvec_f16(&format!("{prefix}.mlp.down_proj.weight"), &mlp)?;
            hidden = residual
                .iter()
                .zip(down.iter())
                .map(|(left, right)| left + right)
                .collect();
            metrics.mlp_duration += started_at.elapsed();
        }

        let started_at = Instant::now();
        let final_norm_weight = self.weights.load_vector_f32("model.norm.weight")?;
        let hidden =
            weighted_rms_norm(&hidden, &final_norm_weight, self.config.rms_norm_eps as f32);
        metrics.norm_duration += started_at.elapsed();

        let started_at = Instant::now();
        let logits = self
            .weights
            .matvec_f16("model.embed_tokens.weight", &hidden)?;
        metrics.logits_duration += started_at.elapsed();
        Ok((logits, compile, upload, gpu, download))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step_packed_decode(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut [LayerCache],
        metrics: &mut DecodeMetrics,
        attention_stage_metrics: &mut PackedAttentionStageMetrics,
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
        non_offloaded_dense_duration: &mut Duration,
        session: &mut PackedGpuSession<'_>,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        use_attention_full: bool,
        use_mlp_full: bool,
        argmax_only: bool,
    ) -> Result<PackedDecodeStepResult, ReferenceError> {
        let started_at = Instant::now();
        let mut hidden = if Self::packed_use_gpu_embedding() && Self::packed_use_gpu_first_session()
        {
            Vec::new()
        } else {
            self.embedding_lookup_resolved("model.embed_tokens.weight", token_id)?
        };
        let elapsed = started_at.elapsed();
        if !hidden.is_empty() {
            metrics.embedding_duration += elapsed;
            *non_offloaded_dense_duration += elapsed;
        }

        let (cos, sin) = rope_cos_sin(&self.rope, &[position]);
        let mut final_hidden_gpu: Option<ResidentGpuVectorAdd> = None;
        let mut resident_hidden_state: Option<ResidentHiddenState> = None;

        for (layer_idx, layer_cache) in cache
            .iter_mut()
            .enumerate()
            .take(self.config.num_hidden_layers)
        {
            let layer_tensors = &self.layer_tensors[layer_idx];
            let use_gpu_attention_block =
                use_attention_qkv && use_attention_full && Self::packed_use_gpu_attention_block();
            let use_gpu_attention_mlp_block = use_gpu_attention_block
                && use_mlp_gu
                && use_mlp_full
                && Self::packed_use_gpu_swiglu_block();

            let input_norm_weight =
                self.load_vector_f32_resolved(&layer_tensors.input_layernorm_weight)?;
            let resident_hidden_tensor_qkv =
                if layer_idx > 0
                    && use_attention_qkv
                    && resident_hidden_state.is_some()
                    && use_gpu_attention_block
                {
                    Some(gpu_first_session.run_layer_input_norm_qkv_tensors_from_hidden_resident(
                        resident_hidden_state.expect("resident hidden state should exist"),
                        layer_idx,
                        &input_norm_weight,
                        &layer_tensors.q_proj_weight,
                        &layer_tensors.k_proj_weight,
                        &layer_tensors.v_proj_weight,
                        self.config.num_key_value_heads * self.config.head_dim,
                    )?)
                } else {
                    None
                };
            let resident_hidden_entry_qkv = if layer_idx > 0
                && use_attention_qkv
                && resident_hidden_state.is_some()
                && !use_gpu_attention_block
            {
                Some(gpu_first_session.run_layer_input_norm_qkv_from_hidden_resident(
                    resident_hidden_state.expect("resident hidden state should exist"),
                    layer_idx,
                    &input_norm_weight,
                    &layer_tensors.q_proj_weight,
                    &layer_tensors.k_proj_weight,
                    &layer_tensors.v_proj_weight,
                    self.config.num_key_value_heads * self.config.head_dim,
                )?)
            } else {
                None
            };
            let first_layer_gpu_entry_tensor_qkv = if resident_hidden_tensor_qkv.is_none()
                && resident_hidden_entry_qkv.is_none()
                && layer_idx == 0
                && Self::packed_use_gpu_embedding()
                && Self::packed_use_gpu_first_session()
                && use_attention_qkv
                && use_gpu_attention_block
            {
                Some(
                    gpu_first_session.run_first_layer_embedding_norm_qkv_tensors(
                        layer_idx,
                        token_id,
                        &input_norm_weight,
                        &layer_tensors.q_proj_weight,
                        &layer_tensors.k_proj_weight,
                        &layer_tensors.v_proj_weight,
                        self.config.num_key_value_heads * self.config.head_dim,
                    )?,
                )
            } else {
                None
            };
            let first_layer_gpu_entry_qkv = if first_layer_gpu_entry_tensor_qkv.is_none()
                && resident_hidden_tensor_qkv.is_none()
                && resident_hidden_entry_qkv.is_none()
                && layer_idx == 0
                && Self::packed_use_gpu_embedding()
                && Self::packed_use_gpu_first_session()
                && use_attention_qkv
            {
                Some(
                    gpu_first_session.run_first_layer_embedding_norm_qkv_to_host(
                        layer_idx,
                        token_id,
                        &input_norm_weight,
                        &layer_tensors.q_proj_weight,
                        &layer_tensors.k_proj_weight,
                        &layer_tensors.v_proj_weight,
                        self.config.num_key_value_heads * self.config.head_dim,
                    )?,
                )
            } else {
                None
            };
            let mut hidden_states = if let Some((
                gpu_hidden,
                _hidden_resident,
                _q,
                _k,
                _v,
                embedding_report,
                norm_report,
                qkv_report,
                compile_duration,
            )) = first_layer_gpu_entry_tensor_qkv.as_ref()
            {
                hidden = gpu_hidden.clone();
                session.metrics.compile_duration += *compile_duration;
                session.metrics.activation_upload_duration += embedding_report.upload_duration
                    + norm_report.upload_duration
                    + qkv_report.upload_duration;
                session.metrics.upload_duration += embedding_report.upload_duration
                    + norm_report.upload_duration
                    + qkv_report.upload_duration;
                session.metrics.gpu_duration += embedding_report.gpu_duration
                    + norm_report.gpu_duration
                    + qkv_report.gpu_duration;
                session.metrics.download_duration += embedding_report.download_duration
                    + norm_report.download_duration
                    + qkv_report.download_duration;
                session.metrics.activation_upload_bytes +=
                    std::mem::size_of::<u32>() + self.config.hidden_size * std::mem::size_of::<f32>();
                session.metrics.upload_bytes +=
                    std::mem::size_of::<u32>() + self.config.hidden_size * std::mem::size_of::<f32>();
                metrics.embedding_duration += embedding_report.upload_duration
                    + embedding_report.gpu_duration
                    + embedding_report.download_duration;
                metrics.norm_duration += norm_report.upload_duration
                    + norm_report.gpu_duration
                    + norm_report.download_duration;
                metrics.qkv_duration += qkv_report.upload_duration
                    + qkv_report.gpu_duration
                    + qkv_report.download_duration;
                vec![0.0; self.config.hidden_size]
            } else if let Some((
                gpu_hidden,
                _q,
                _k,
                _v,
                embedding_report,
                norm_report,
                qkv_report,
                compile_duration,
            )) = first_layer_gpu_entry_qkv.as_ref()
            {
                hidden = gpu_hidden.clone();
                session.metrics.compile_duration += *compile_duration;
                session.metrics.activation_upload_duration += embedding_report.upload_duration
                    + norm_report.upload_duration
                    + qkv_report.upload_duration;
                session.metrics.upload_duration += embedding_report.upload_duration
                    + norm_report.upload_duration
                    + qkv_report.upload_duration;
                session.metrics.gpu_duration += embedding_report.gpu_duration
                    + norm_report.gpu_duration
                    + qkv_report.gpu_duration;
                session.metrics.download_duration += embedding_report.download_duration
                    + norm_report.download_duration
                    + qkv_report.download_duration;
                session.metrics.activation_upload_bytes += std::mem::size_of::<u32>()
                    + self.config.hidden_size * std::mem::size_of::<f32>();
                session.metrics.upload_bytes += std::mem::size_of::<u32>()
                    + self.config.hidden_size * std::mem::size_of::<f32>();
                session.metrics.download_bytes +=
                    self.config.hidden_size * std::mem::size_of::<f32>() * 2
                        + (self.config.hidden_size
                            + 2 * (self.config.num_key_value_heads * self.config.head_dim))
                            * std::mem::size_of::<f32>();
                metrics.embedding_duration += embedding_report.upload_duration
                    + embedding_report.gpu_duration
                    + embedding_report.download_duration;
                metrics.norm_duration += norm_report.upload_duration
                    + norm_report.gpu_duration
                    + norm_report.download_duration;
                metrics.qkv_duration += qkv_report.upload_duration
                    + qkv_report.gpu_duration
                    + qkv_report.download_duration;
                vec![0.0; self.config.hidden_size]
            } else if let Some((
                _hidden_resident,
                _q,
                _k,
                _v,
                norm_report,
                qkv_report,
                compile_duration,
            )) = resident_hidden_tensor_qkv.as_ref()
            {
                session.metrics.compile_duration += *compile_duration;
                session.metrics.activation_upload_duration +=
                    norm_report.upload_duration + qkv_report.upload_duration;
                session.metrics.upload_duration +=
                    norm_report.upload_duration + qkv_report.upload_duration;
                session.metrics.gpu_duration += norm_report.gpu_duration + qkv_report.gpu_duration;
                session.metrics.download_duration +=
                    norm_report.download_duration + qkv_report.download_duration;
                session.metrics.activation_upload_bytes +=
                    self.config.hidden_size * std::mem::size_of::<f32>();
                session.metrics.upload_bytes +=
                    self.config.hidden_size * std::mem::size_of::<f32>();
                metrics.norm_duration += norm_report.upload_duration
                    + norm_report.gpu_duration
                    + norm_report.download_duration;
                metrics.qkv_duration += qkv_report.upload_duration
                    + qkv_report.gpu_duration
                    + qkv_report.download_duration;
                vec![0.0; self.config.hidden_size]
            } else if let Some((
                _hidden_resident,
                _q,
                _k,
                _v,
                norm_report,
                qkv_report,
                compile_duration,
            )) = resident_hidden_entry_qkv.as_ref()
            {
                session.metrics.compile_duration += *compile_duration;
                session.metrics.activation_upload_duration +=
                    norm_report.upload_duration + qkv_report.upload_duration;
                session.metrics.upload_duration +=
                    norm_report.upload_duration + qkv_report.upload_duration;
                session.metrics.gpu_duration += norm_report.gpu_duration + qkv_report.gpu_duration;
                session.metrics.download_duration +=
                    norm_report.download_duration + qkv_report.download_duration;
                session.metrics.activation_upload_bytes +=
                    self.config.hidden_size * std::mem::size_of::<f32>();
                session.metrics.upload_bytes +=
                    self.config.hidden_size * std::mem::size_of::<f32>();
                session.metrics.download_bytes += (self.config.hidden_size
                        + 2 * (self.config.num_key_value_heads * self.config.head_dim))
                        * std::mem::size_of::<f32>();
                metrics.norm_duration += norm_report.upload_duration
                    + norm_report.gpu_duration
                    + norm_report.download_duration;
                metrics.qkv_duration += qkv_report.upload_duration
                    + qkv_report.gpu_duration
                    + qkv_report.download_duration;
                let (materialized_hidden, hidden_download_duration) = gpu_first_session
                    .read_hidden_output(
                        resident_hidden_state.expect("resident hidden state should exist"),
                    )?;
                hidden = materialized_hidden;
                session.metrics.download_duration += hidden_download_duration;
                session.metrics.download_bytes +=
                    self.config.hidden_size * std::mem::size_of::<f32>();
                vec![0.0; self.config.hidden_size]
            } else if layer_idx == 0
                && Self::packed_use_gpu_embedding()
                && Self::packed_use_gpu_first_session()
            {
                let started_at = Instant::now();
                let (
                    gpu_hidden,
                    gpu_hidden_states,
                    embedding_report,
                    norm_report,
                    compile_duration,
                ) = gpu_first_session
                    .run_embedding_and_input_norm_to_host(token_id, &input_norm_weight)?;
                hidden = gpu_hidden;
                session.metrics.compile_duration += compile_duration;
                session.metrics.activation_upload_duration +=
                    embedding_report.upload_duration + norm_report.upload_duration;
                session.metrics.upload_duration +=
                    embedding_report.upload_duration + norm_report.upload_duration;
                session.metrics.gpu_duration +=
                    embedding_report.gpu_duration + norm_report.gpu_duration;
                session.metrics.download_duration +=
                    embedding_report.download_duration + norm_report.download_duration;
                session.metrics.activation_upload_bytes += std::mem::size_of::<u32>()
                    + self.config.hidden_size * std::mem::size_of::<f32>();
                session.metrics.upload_bytes += std::mem::size_of::<u32>()
                    + self.config.hidden_size * std::mem::size_of::<f32>();
                session.metrics.download_bytes +=
                    self.config.hidden_size * std::mem::size_of::<f32>() * 2;
                metrics.embedding_duration += started_at.elapsed();
                metrics.norm_duration += norm_report.upload_duration
                    + norm_report.gpu_duration
                    + norm_report.download_duration;
                gpu_hidden_states
            } else {
                let started_at = Instant::now();
                let hidden_states =
                    weighted_rms_norm(&hidden, &input_norm_weight, self.config.rms_norm_eps as f32);
                let elapsed = started_at.elapsed();
                metrics.norm_duration += elapsed;
                *non_offloaded_dense_duration += elapsed;
                session.push_dense_stage_trace(
                    "input_norm",
                    &layer_tensors.input_layernorm_weight,
                    elapsed,
                );
                hidden_states
            };
            let residual = hidden.clone();
            let residual_resident = if let Some((_, hidden_resident, ..)) =
                first_layer_gpu_entry_tensor_qkv.as_ref()
            {
                Some(hidden_resident.clone())
            } else {
                resident_hidden_tensor_qkv
                    .as_ref()
                    .map(|(hidden_resident, ..)| hidden_resident.clone())
            };

            let started_at = Instant::now();
            let kv_rows = self.config.num_key_value_heads * self.config.head_dim;
            let (mut q, mut k, v) = if first_layer_gpu_entry_tensor_qkv.is_some() {
                (
                    vec![0.0; self.config.hidden_size],
                    vec![0.0; kv_rows],
                    vec![0.0; kv_rows],
                )
            } else if let Some((_, q, k, v, ..)) = first_layer_gpu_entry_qkv {
                (q, k, v)
            } else if let Some((_, q, k, v, ..)) = resident_hidden_entry_qkv {
                (q, k, v)
            } else if use_attention_qkv {
                session.run_projection_triplet(
                    &layer_tensors.q_proj_weight,
                    self.config.hidden_size,
                    &layer_tensors.k_proj_weight,
                    kv_rows,
                    &layer_tensors.v_proj_weight,
                    kv_rows,
                    self.config.hidden_size,
                    &hidden_states,
                )?
            } else {
                (
                    self.matvec_f16_resolved(&layer_tensors.q_proj_weight, &hidden_states)?,
                    self.matvec_f16_resolved(&layer_tensors.k_proj_weight, &hidden_states)?,
                    self.matvec_f16_resolved(&layer_tensors.v_proj_weight, &hidden_states)?,
                )
            };
            let elapsed = started_at.elapsed();
            metrics.qkv_duration += elapsed;
            if !use_attention_qkv {
                *non_offloaded_dense_duration += elapsed;
            }

            let started_at = Instant::now();
            let q_norm_weight = self.load_vector_f32_resolved(&layer_tensors.q_norm_weight)?;
            let k_norm_weight = self.load_vector_f32_resolved(&layer_tensors.k_norm_weight)?;
            let mut q_resident: Option<GpuResidentBuffer> = None;
            let mut kv_appended_on_gpu = false;
            if use_gpu_attention_block {
                if let Some((_, _, q_tensor, k_tensor, v_tensor, ..)) =
                    first_layer_gpu_entry_tensor_qkv.as_ref()
                {
                    let (query_out, key_out, qk_rope_report, compile_duration) = gpu_first_session
                        .run_qk_rope_resident_query_and_key(
                            q_tensor,
                            k_tensor,
                            &q_norm_weight,
                            &k_norm_weight,
                            &cos,
                            &sin,
                        )?;
                    q_resident = Some(query_out);
                    gpu_first_session.append_gpu_kv_tensors(layer_idx, &key_out, v_tensor)?;
                    kv_appended_on_gpu = true;
                    metrics.norm_duration += qk_rope_report.upload_duration
                        + qk_rope_report.gpu_duration
                        + qk_rope_report.download_duration;
                    session.metrics.compile_duration += compile_duration;
                    session.metrics.activation_upload_duration += qk_rope_report.upload_duration;
                    session.metrics.upload_duration += qk_rope_report.upload_duration;
                    session.metrics.gpu_duration += qk_rope_report.gpu_duration;
                    session.metrics.download_duration += qk_rope_report.download_duration;
                    session.metrics.activation_upload_bytes +=
                        (q_norm_weight.len() + k_norm_weight.len() + cos.len() + sin.len())
                            * std::mem::size_of::<f32>();
                    session.metrics.upload_bytes +=
                        (q_norm_weight.len() + k_norm_weight.len() + cos.len() + sin.len())
                            * std::mem::size_of::<f32>();
                } else if let Some((_, q_tensor, k_tensor, v_tensor, ..)) = resident_hidden_tensor_qkv.as_ref() {
                    let (query_out, key_out, qk_rope_report, compile_duration) = gpu_first_session
                        .run_qk_rope_resident_query_and_key(
                            q_tensor,
                            k_tensor,
                            &q_norm_weight,
                            &k_norm_weight,
                            &cos,
                            &sin,
                        )?;
                    q_resident = Some(query_out);
                    gpu_first_session.append_gpu_kv_tensors(layer_idx, &key_out, v_tensor)?;
                    kv_appended_on_gpu = true;
                    metrics.norm_duration += qk_rope_report.upload_duration
                        + qk_rope_report.gpu_duration
                        + qk_rope_report.download_duration;
                    session.metrics.compile_duration += compile_duration;
                    session.metrics.activation_upload_duration += qk_rope_report.upload_duration;
                    session.metrics.upload_duration += qk_rope_report.upload_duration;
                    session.metrics.gpu_duration += qk_rope_report.gpu_duration;
                    session.metrics.download_duration += qk_rope_report.download_duration;
                    session.metrics.activation_upload_bytes +=
                        (q_norm_weight.len() + k_norm_weight.len() + cos.len() + sin.len())
                            * std::mem::size_of::<f32>();
                    session.metrics.upload_bytes +=
                        (q_norm_weight.len() + k_norm_weight.len() + cos.len() + sin.len())
                            * std::mem::size_of::<f32>();
                } else {
                    let ((q_out, key_out), qk_rope_report, compile_duration) = gpu_first_session
                        .run_qk_rope_query_to_host_key_resident(
                            &q,
                            &k,
                            &q_norm_weight,
                            &k_norm_weight,
                            &cos,
                            &sin,
                        )?;
                    q = q_out;
                    q_resident = Some(
                        gpu_first_session
                            .qk_rope_runner
                            .as_ref()
                            .expect("qk rope runner should exist after execution")
                            .query_resident_output(),
                    );
                    gpu_first_session
                        .append_gpu_kv_key_tensor_and_value_host(layer_idx, &key_out, &v)?;
                    kv_appended_on_gpu = true;
                    metrics.norm_duration += qk_rope_report.upload_duration
                        + qk_rope_report.gpu_duration
                        + qk_rope_report.download_duration;
                    session.metrics.compile_duration += compile_duration;
                    session.metrics.activation_upload_duration += qk_rope_report.upload_duration;
                    session.metrics.upload_duration += qk_rope_report.upload_duration;
                    session.metrics.gpu_duration += qk_rope_report.gpu_duration;
                    session.metrics.download_duration += qk_rope_report.download_duration;
                    session.metrics.activation_upload_bytes += (q.len()
                        + k.len()
                        + q_norm_weight.len()
                        + k_norm_weight.len()
                        + cos.len()
                        + sin.len())
                        * std::mem::size_of::<f32>();
                    session.metrics.upload_bytes += (q.len()
                        + k.len()
                        + q_norm_weight.len()
                        + k_norm_weight.len()
                        + cos.len()
                        + sin.len())
                        * std::mem::size_of::<f32>();
                    session.metrics.download_bytes += q.len() * std::mem::size_of::<f32>();
                }
            } else {
                apply_head_rms_norm_weighted(
                    &mut q,
                    self.config.num_attention_heads,
                    self.config.head_dim,
                    &q_norm_weight,
                    self.config.rms_norm_eps as f32,
                );
                apply_head_rms_norm_weighted(
                    &mut k,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    &k_norm_weight,
                    self.config.rms_norm_eps as f32,
                );
                apply_rotary_single(
                    &mut q,
                    &mut k,
                    &cos,
                    &sin,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                );
                let elapsed = started_at.elapsed();
                metrics.norm_duration += elapsed;
                *non_offloaded_dense_duration += elapsed;
                session.push_dense_stage_trace(
                    "qk_norm_rope",
                    &layer_tensors.q_norm_weight,
                    elapsed,
                );
            }

            if use_gpu_attention_block && !kv_appended_on_gpu {
                gpu_first_session.append_gpu_kv(layer_idx, &k, &v)?;
            }

            let use_gpu_attention_only = use_gpu_attention_block && !use_gpu_attention_mlp_block;
            if use_gpu_attention_only {
                let use_gpu_attention_tail = argmax_only
                    && Self::packed_use_gpu_tail()
                    && layer_idx + 1 == self.config.num_hidden_layers;
                if use_gpu_attention_tail {
                    let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
                    let (next_token, attention_report, tail_report, compile_duration) =
                        gpu_first_session.run_attention_layer_to_tail_argmax(
                            layer_idx,
                            position + 1,
                            Some(&q),
                            q_resident.as_ref(),
                            residual_resident.as_ref(),
                            &layer_cache.keys,
                            &layer_cache.values,
                            &residual,
                            &final_norm_weight,
                        )?;
                    attention_stage_metrics.query_duration +=
                        attention_report.attention_gpu_duration;
                    attention_stage_metrics.oproj_duration += attention_report.oproj_gpu_duration;
                    attention_stage_metrics.residual_duration +=
                        attention_report.residual_add_gpu_duration;
                    metrics.attention_duration += attention_report.attention_gpu_duration
                        + attention_report.oproj_gpu_duration
                        + attention_report.residual_add_gpu_duration;
                    metrics.norm_duration += tail_report.final_norm_gpu_duration;
                    metrics.logits_duration += tail_report.pack_gpu_duration
                        + tail_report.logits_gpu_duration
                        + tail_report.logits_download_duration;
                    session.metrics.compile_duration += compile_duration;
                    session.metrics.gpu_duration += attention_report.attention_gpu_duration
                        + attention_report.oproj_gpu_duration
                        + attention_report.residual_add_gpu_duration
                        + tail_report.final_norm_gpu_duration
                        + tail_report.pack_gpu_duration
                        + tail_report.logits_gpu_duration;
                    session.metrics.download_duration += tail_report.logits_download_duration;
                    return Ok(PackedDecodeStepResult::NextToken(next_token));
                } else {
                    let (next_hidden, attention_report, compile_duration, download_duration) =
                        gpu_first_session.run_attention_layer_to_host(
                            layer_idx,
                            position + 1,
                            Some(&q),
                            q_resident.as_ref(),
                            &layer_cache.keys,
                            &layer_cache.values,
                            &residual,
                        )?;
                    hidden = next_hidden;
                    attention_stage_metrics.query_duration +=
                        attention_report.attention_gpu_duration;
                    attention_stage_metrics.oproj_duration += attention_report.oproj_gpu_duration;
                    attention_stage_metrics.residual_duration +=
                        attention_report.residual_add_gpu_duration;
                    metrics.attention_duration += attention_report.attention_gpu_duration
                        + attention_report.oproj_gpu_duration
                        + attention_report.residual_add_gpu_duration;
                    session.metrics.compile_duration += compile_duration;
                    session.metrics.gpu_duration += attention_report.attention_gpu_duration
                        + attention_report.oproj_gpu_duration
                        + attention_report.residual_add_gpu_duration;
                    session.metrics.download_duration += download_duration;
                }
            }

            if !use_gpu_attention_block {
                layer_cache.keys.extend_from_slice(&k);
                layer_cache.values.extend_from_slice(&v);
            }

            let use_gpu_full_last_layer_block = use_gpu_attention_mlp_block
                && argmax_only
                && Self::packed_use_gpu_full_last_layer()
                && layer_idx + 1 == self.config.num_hidden_layers;
            if use_gpu_full_last_layer_block {
                let post_norm_weight =
                    self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
                let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
                let (next_token, attention_report, full_last_layer_report, compile_duration) =
                    gpu_first_session.run_attention_to_full_last_layer_argmax(
                        layer_idx,
                        position + 1,
                        Some(&q),
                        q_resident.as_ref(),
                        residual_resident.as_ref(),
                        &layer_cache.keys,
                        &layer_cache.values,
                        &residual,
                        &post_norm_weight,
                        &final_norm_weight,
                    )?;
                attention_stage_metrics.query_duration += attention_report.attention_gpu_duration;
                attention_stage_metrics.oproj_duration += attention_report.oproj_gpu_duration;
                attention_stage_metrics.residual_duration +=
                    attention_report.residual_add_gpu_duration;
                metrics.attention_duration += attention_report.attention_gpu_duration
                    + attention_report.oproj_gpu_duration
                    + attention_report.residual_add_gpu_duration;
                metrics.norm_duration += full_last_layer_report.post_norm_gpu_duration
                    + full_last_layer_report.final_norm_gpu_duration;
                mlp_stage_metrics.swiglu_duration +=
                    full_last_layer_report.swiglu_pack_gpu_duration;
                mlp_stage_metrics.down_duration += full_last_layer_report.down_gpu_duration;
                mlp_stage_metrics.residual_duration +=
                    full_last_layer_report.residual_add_gpu_duration;
                metrics.mlp_duration += full_last_layer_report.post_norm_gpu_duration
                    + full_last_layer_report.pair_gpu_duration
                    + full_last_layer_report.swiglu_pack_gpu_duration
                    + full_last_layer_report.down_gpu_duration
                    + full_last_layer_report.residual_add_gpu_duration;
                metrics.logits_duration += full_last_layer_report.pack_gpu_duration
                    + full_last_layer_report.logits_gpu_duration
                    + full_last_layer_report.logits_download_duration;
                session.metrics.compile_duration += compile_duration;
                session.metrics.gpu_duration += attention_report.attention_gpu_duration
                    + attention_report.oproj_gpu_duration
                    + attention_report.residual_add_gpu_duration
                    + full_last_layer_report.post_norm_gpu_duration
                    + full_last_layer_report.pair_gpu_duration
                    + full_last_layer_report.swiglu_pack_gpu_duration
                    + full_last_layer_report.down_gpu_duration
                    + full_last_layer_report.residual_add_gpu_duration
                    + full_last_layer_report.final_norm_gpu_duration
                    + full_last_layer_report.pack_gpu_duration
                    + full_last_layer_report.logits_gpu_duration;
                session.metrics.download_duration +=
                    full_last_layer_report.logits_download_duration;
                return Ok(PackedDecodeStepResult::NextToken(next_token));
            }
            if use_gpu_attention_mlp_block {
                let post_norm_weight =
                    self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
                let (attention_report, mlp_report, compile_duration) = gpu_first_session
                    .run_attention_mlp_layer_resident(
                        layer_idx,
                        position + 1,
                        Some(&q),
                        q_resident.as_ref(),
                        residual_resident.as_ref(),
                        &layer_cache.keys,
                        &layer_cache.values,
                        &residual,
                        &post_norm_weight,
                    )?;
                resident_hidden_state = Some(ResidentHiddenState::Mlp { layer_idx });
                attention_stage_metrics.query_duration += attention_report.attention_gpu_duration;
                attention_stage_metrics.oproj_duration += attention_report.oproj_gpu_duration;
                attention_stage_metrics.residual_duration +=
                    attention_report.residual_add_gpu_duration;
                metrics.attention_duration += attention_report.attention_gpu_duration
                    + attention_report.oproj_gpu_duration
                    + attention_report.residual_add_gpu_duration;
                metrics.norm_duration += mlp_report.post_norm_gpu_duration;
                mlp_stage_metrics.swiglu_duration += mlp_report.swiglu_pack_gpu_duration;
                mlp_stage_metrics.down_duration += mlp_report.down_gpu_duration;
                mlp_stage_metrics.residual_duration += mlp_report.residual_add_gpu_duration;
                metrics.mlp_duration += mlp_report.post_norm_gpu_duration
                    + mlp_report.pair_gpu_duration
                    + mlp_report.swiglu_pack_gpu_duration
                    + mlp_report.down_gpu_duration
                    + mlp_report.residual_add_gpu_duration;
                session.metrics.compile_duration += compile_duration;
                session.metrics.gpu_duration += attention_report.attention_gpu_duration
                    + attention_report.oproj_gpu_duration
                    + attention_report.residual_add_gpu_duration
                    + mlp_report.post_norm_gpu_duration
                    + mlp_report.pair_gpu_duration
                    + mlp_report.swiglu_pack_gpu_duration
                    + mlp_report.down_gpu_duration
                    + mlp_report.residual_add_gpu_duration;
                continue;
            }

            if !use_gpu_attention_only {
                let started_at = Instant::now();
                let attn_started_at = Instant::now();
                let attn = attention_single_query(
                    &q,
                    &layer_cache.keys,
                    &layer_cache.values,
                    position + 1,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                );
                let attn_elapsed = attn_started_at.elapsed();
                attention_stage_metrics.query_duration += attn_elapsed;
                session.push_dense_stage_trace(
                    "attention_core",
                    "attention_single_query",
                    attn_elapsed,
                );
                let oproj_started_at = Instant::now();
                let attn_output = if use_attention_full {
                    session.run_projection(
                        &layer_tensors.o_proj_weight,
                        self.config.hidden_size,
                        self.config.hidden_size,
                        &attn,
                    )?
                } else {
                    self.matvec_f16_resolved(&layer_tensors.o_proj_weight, &attn)?
                };
                let oproj_elapsed = oproj_started_at.elapsed();
                attention_stage_metrics.oproj_duration += oproj_elapsed;
                let residual_started_at = Instant::now();
                hidden = residual
                    .iter()
                    .zip(attn_output.iter())
                    .map(|(left, right)| left + right)
                    .collect();
                let residual_elapsed = residual_started_at.elapsed();
                attention_stage_metrics.residual_duration += residual_elapsed;
                session.push_dense_stage_trace(
                    "attention_residual",
                    "attention_residual_add",
                    residual_elapsed,
                );
                let elapsed = started_at.elapsed();
                metrics.attention_duration += elapsed;
                if use_attention_qkv {
                    if use_attention_full {
                        *non_offloaded_dense_duration += attn_elapsed + residual_elapsed;
                    } else {
                        *non_offloaded_dense_duration += elapsed;
                    }
                } else {
                    *non_offloaded_dense_duration += elapsed;
                }
            }

            let residual = hidden.clone();
            let use_gpu_full_last_layer = use_mlp_gu
                && use_mlp_full
                && argmax_only
                && Self::packed_use_gpu_full_last_layer()
                && layer_idx + 1 == self.config.num_hidden_layers;
            if use_gpu_full_last_layer {
                let post_norm_weight =
                    self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
                let post_norm_started = Instant::now();
                let post_norm = session.run_final_norm_resident(&residual, &post_norm_weight)?;
                let pair = session.run_projection_pair_resident_from_final_norm(
                    &layer_tensors.gate_proj_weight,
                    self.config.intermediate_size,
                    &layer_tensors.up_proj_weight,
                    self.config.intermediate_size,
                    self.config.hidden_size,
                    post_norm,
                )?;
                let post_norm_elapsed = post_norm_started.elapsed();
                metrics.norm_duration += post_norm_elapsed;

                let mlp_started = Instant::now();
                let swiglu_started = Instant::now();
                let packed_activation = session.run_swiglu_pack_f16_pairs_resident(pair)?;
                let swiglu_elapsed = swiglu_started.elapsed();
                mlp_stage_metrics.swiglu_duration += swiglu_elapsed;

                let down_started = Instant::now();
                let down = session.run_projection_resident_from_packed_activation(
                    &layer_tensors.down_proj_weight,
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    packed_activation,
                )?;
                let down_elapsed = down_started.elapsed();
                mlp_stage_metrics.down_duration += down_elapsed;

                let residual_started = Instant::now();
                let hidden_gpu = session.run_vector_add_resident(down, &residual)?;
                let residual_elapsed = residual_started.elapsed();
                mlp_stage_metrics.residual_duration += residual_elapsed;
                metrics.mlp_duration += mlp_started.elapsed();

                let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
                let final_norm_started = Instant::now();
                let final_norm = session
                    .run_final_norm_resident_from_vector_add(hidden_gpu, &final_norm_weight)?;
                metrics.norm_duration += final_norm_started.elapsed();

                let logits_started = Instant::now();
                let packed_activation = session.run_pack_f16_pairs_resident(final_norm)?;
                let next_token = session.run_projection_argmax_from_packed_activation(
                    "model.embed_tokens.weight",
                    self.config.vocab_size,
                    self.config.hidden_size,
                    packed_activation,
                )?;
                metrics.logits_duration += logits_started.elapsed();
                return Ok(PackedDecodeStepResult::NextToken(next_token));
            }
            let post_norm_weight =
                self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
            let use_gpu_mlp_entry = use_mlp_gu && use_mlp_full && Self::packed_use_gpu_mlp_entry();
            let use_gpu_mlp_only = use_mlp_gu
                && use_mlp_full
                && Self::packed_use_gpu_swiglu_block()
                && !use_gpu_attention_mlp_block;
            let use_gpu_mlp_tail = use_gpu_mlp_only
                && argmax_only
                && Self::packed_use_gpu_tail()
                && layer_idx + 1 == self.config.num_hidden_layers;
            if use_gpu_mlp_only {
                if use_gpu_mlp_tail {
                    let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
                    let (next_token, mlp_report, tail_report, compile_duration) = gpu_first_session
                        .run_mlp_layer_to_tail_argmax(
                            layer_idx,
                            &residual,
                            &post_norm_weight,
                            &final_norm_weight,
                        )?;
                    metrics.norm_duration +=
                        mlp_report.post_norm_gpu_duration + tail_report.final_norm_gpu_duration;
                    mlp_stage_metrics.swiglu_duration += mlp_report.swiglu_pack_gpu_duration;
                    mlp_stage_metrics.down_duration += mlp_report.down_gpu_duration;
                    mlp_stage_metrics.residual_duration += mlp_report.residual_add_gpu_duration;
                    metrics.mlp_duration += mlp_report.post_norm_gpu_duration
                        + mlp_report.pair_gpu_duration
                        + mlp_report.swiglu_pack_gpu_duration
                        + mlp_report.down_gpu_duration
                        + mlp_report.residual_add_gpu_duration;
                    metrics.logits_duration += tail_report.pack_gpu_duration
                        + tail_report.logits_gpu_duration
                        + tail_report.logits_download_duration;
                    session.metrics.compile_duration += compile_duration;
                    session.metrics.gpu_duration += mlp_report.post_norm_gpu_duration
                        + mlp_report.pair_gpu_duration
                        + mlp_report.swiglu_pack_gpu_duration
                        + mlp_report.down_gpu_duration
                        + mlp_report.residual_add_gpu_duration
                        + tail_report.final_norm_gpu_duration
                        + tail_report.pack_gpu_duration
                        + tail_report.logits_gpu_duration;
                    session.metrics.download_duration += tail_report.logits_download_duration;
                    return Ok(PackedDecodeStepResult::NextToken(next_token));
                } else {
                    let (mlp_report, compile_duration) = gpu_first_session.run_mlp_layer_resident(
                        layer_idx,
                        &residual,
                        &post_norm_weight,
                    )?;
                    resident_hidden_state = Some(ResidentHiddenState::Mlp { layer_idx });
                    metrics.norm_duration += mlp_report.post_norm_gpu_duration;
                    mlp_stage_metrics.swiglu_duration += mlp_report.swiglu_pack_gpu_duration;
                    mlp_stage_metrics.down_duration += mlp_report.down_gpu_duration;
                    mlp_stage_metrics.residual_duration += mlp_report.residual_add_gpu_duration;
                    metrics.mlp_duration += mlp_report.post_norm_gpu_duration
                        + mlp_report.pair_gpu_duration
                        + mlp_report.swiglu_pack_gpu_duration
                        + mlp_report.down_gpu_duration
                        + mlp_report.residual_add_gpu_duration;
                    session.metrics.compile_duration += compile_duration;
                    session.metrics.gpu_duration += mlp_report.post_norm_gpu_duration
                        + mlp_report.pair_gpu_duration
                        + mlp_report.swiglu_pack_gpu_duration
                        + mlp_report.down_gpu_duration
                        + mlp_report.residual_add_gpu_duration;
                    continue;
                }
            }
            let mut pair_from_gpu_norm: Option<ResidentPackedPairProjection> = None;
            if use_gpu_mlp_entry {
                let started_at = Instant::now();
                let final_norm = session.run_final_norm_resident(&residual, &post_norm_weight)?;
                let pair = session.run_projection_pair_resident_from_final_norm(
                    &layer_tensors.gate_proj_weight,
                    self.config.intermediate_size,
                    &layer_tensors.up_proj_weight,
                    self.config.intermediate_size,
                    self.config.hidden_size,
                    final_norm,
                )?;
                let elapsed = started_at.elapsed();
                metrics.norm_duration += elapsed;
                pair_from_gpu_norm = Some(pair);
            } else {
                let started_at = Instant::now();
                hidden_states = weighted_rms_norm(
                    &residual,
                    &post_norm_weight,
                    self.config.rms_norm_eps as f32,
                );
                let elapsed = started_at.elapsed();
                metrics.norm_duration += elapsed;
                *non_offloaded_dense_duration += elapsed;
                session.push_dense_stage_trace(
                    "post_attention_norm",
                    &layer_tensors.post_attention_layernorm_weight,
                    elapsed,
                );
            }

            let started_at = Instant::now();
            let use_gpu_swiglu_block =
                use_mlp_gu && use_mlp_full && Self::packed_use_gpu_swiglu_block();
            let use_gpu_tail = use_gpu_swiglu_block
                && Self::packed_use_gpu_tail()
                && Self::packed_use_gpu_final_norm()
                && argmax_only
                && layer_idx + 1 == self.config.num_hidden_layers;
            let (
                down,
                gpu_hidden_after_mlp,
                swiglu_elapsed,
                down_elapsed,
                gpu_residual_elapsed,
                dense_tail_started_at,
            ) = if use_gpu_swiglu_block {
                let pair = if let Some(pair) = pair_from_gpu_norm.take() {
                    pair
                } else {
                    session.run_projection_pair_resident(
                        &layer_tensors.gate_proj_weight,
                        self.config.intermediate_size,
                        &layer_tensors.up_proj_weight,
                        self.config.intermediate_size,
                        self.config.hidden_size,
                        &hidden_states,
                    )?
                };
                let swiglu_started_at = Instant::now();
                let packed_activation = session.run_swiglu_pack_f16_pairs_resident(pair)?;
                let swiglu_elapsed = swiglu_started_at.elapsed();
                mlp_stage_metrics.swiglu_duration += swiglu_elapsed;
                let down_started_at = Instant::now();
                if use_gpu_tail {
                    let down = session.run_projection_resident_from_packed_activation(
                        &layer_tensors.down_proj_weight,
                        self.config.hidden_size,
                        self.config.intermediate_size,
                        packed_activation,
                    )?;
                    let down_elapsed = down_started_at.elapsed();
                    let residual_started_at = Instant::now();
                    let hidden_gpu = session.run_vector_add_resident(down, &residual)?;
                    let residual_elapsed = residual_started_at.elapsed();
                    (
                        None,
                        Some(hidden_gpu),
                        swiglu_elapsed,
                        down_elapsed,
                        residual_elapsed,
                        Instant::now(),
                    )
                } else {
                    let down = session.run_projection_from_packed_activation(
                        &layer_tensors.down_proj_weight,
                        self.config.hidden_size,
                        self.config.intermediate_size,
                        packed_activation,
                    )?;
                    let down_elapsed = down_started_at.elapsed();
                    (
                        Some(down),
                        None,
                        swiglu_elapsed,
                        down_elapsed,
                        Duration::ZERO,
                        Instant::now(),
                    )
                }
            } else {
                let (gate, up, dense_tail_started_at) = if use_mlp_gu {
                    session
                        .run_projection_pair(
                            &layer_tensors.gate_proj_weight,
                            self.config.intermediate_size,
                            &layer_tensors.up_proj_weight,
                            self.config.intermediate_size,
                            self.config.hidden_size,
                            &hidden_states,
                        )
                        .map(|(gate, up)| (gate, up, Instant::now()))?
                } else {
                    let dense_started_at = Instant::now();
                    (
                        self.matvec_f16_resolved(&layer_tensors.gate_proj_weight, &hidden_states)?,
                        self.matvec_f16_resolved(&layer_tensors.up_proj_weight, &hidden_states)?,
                        dense_started_at,
                    )
                };
                let swiglu_started_at = Instant::now();
                let mlp = swiglu(&gate, &up);
                let swiglu_elapsed = swiglu_started_at.elapsed();
                mlp_stage_metrics.swiglu_duration += swiglu_elapsed;
                session.push_dense_stage_trace("mlp_swiglu", "swiglu", swiglu_elapsed);
                let down_started_at = Instant::now();
                let down = if use_mlp_full {
                    session.run_projection(
                        &layer_tensors.down_proj_weight,
                        self.config.hidden_size,
                        self.config.intermediate_size,
                        &mlp,
                    )?
                } else {
                    self.matvec_f16_resolved(&layer_tensors.down_proj_weight, &mlp)?
                };
                let down_elapsed = down_started_at.elapsed();
                (
                    Some(down),
                    None,
                    swiglu_elapsed,
                    down_elapsed,
                    Duration::ZERO,
                    dense_tail_started_at,
                )
            };
            mlp_stage_metrics.down_duration += down_elapsed;
            let residual_elapsed = if let Some(hidden_gpu) = gpu_hidden_after_mlp {
                hidden = residual;
                final_hidden_gpu = Some(hidden_gpu);
                resident_hidden_state = None;
                gpu_residual_elapsed
            } else {
                resident_hidden_state = None;
                let residual_started_at = Instant::now();
                hidden = residual
                    .iter()
                    .zip(
                        down.as_ref()
                            .expect("down must exist for cpu residual path")
                            .iter(),
                    )
                    .map(|(left, right)| left + right)
                    .collect();
                let residual_elapsed = residual_started_at.elapsed();
                session.push_dense_stage_trace(
                    "mlp_residual",
                    "mlp_residual_add",
                    residual_elapsed,
                );
                residual_elapsed
            };
            mlp_stage_metrics.residual_duration += residual_elapsed;
            let elapsed = started_at.elapsed();
            metrics.mlp_duration += elapsed;
            if use_mlp_gu {
                if use_mlp_full {
                    if use_gpu_swiglu_block {
                        if !use_gpu_tail {
                            *non_offloaded_dense_duration += residual_elapsed;
                        }
                    } else {
                        *non_offloaded_dense_duration += swiglu_elapsed + residual_elapsed;
                    }
                } else {
                    *non_offloaded_dense_duration += dense_tail_started_at.elapsed();
                }
            } else {
                *non_offloaded_dense_duration += elapsed;
            }
        }

        let result = if argmax_only {
            let norm_started = Instant::now();
            let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
            let next_token = if let Some(hidden_gpu) = final_hidden_gpu {
                let hidden_gpu_output = GpuResidentBuffer::new(
                    hidden_gpu.runner.borrow().shared_context().clone(),
                    hidden_gpu.runner.borrow().output_buffer_handle(),
                    hidden_gpu.runner.borrow().len(),
                    hidden_gpu.runner.borrow().output_buffer_size(),
                );
                let (next_token, tail_report, compile_duration) = gpu_first_session
                    .run_tail_argmax_from_resident_hidden(&hidden_gpu_output, &final_norm_weight)?;
                metrics.norm_duration += norm_started.elapsed();
                metrics.logits_duration += tail_report.pack_gpu_duration
                    + tail_report.logits_gpu_duration
                    + tail_report.logits_download_duration;
                session.metrics.compile_duration += hidden_gpu.compile_duration + compile_duration;
                session.metrics.gpu_duration += hidden_gpu.report.gpu_duration
                    + tail_report.final_norm_gpu_duration
                    + tail_report.pack_gpu_duration
                    + tail_report.logits_gpu_duration;
                session.metrics.download_duration += tail_report.logits_download_duration;
                next_token
            } else if Self::packed_use_gpu_tail() {
                let (next_token, tail_report, compile_duration) = gpu_first_session
                    .run_tail_argmax_from_host_hidden(&hidden, &final_norm_weight)?;
                metrics.norm_duration += norm_started.elapsed();
                metrics.logits_duration += tail_report.pack_gpu_duration
                    + tail_report.logits_gpu_duration
                    + tail_report.logits_download_duration;
                session.metrics.compile_duration += compile_duration;
                session.metrics.gpu_duration += tail_report.final_norm_gpu_duration
                    + tail_report.pack_gpu_duration
                    + tail_report.logits_gpu_duration;
                session.metrics.download_duration += tail_report.logits_download_duration;
                next_token
            } else if Self::packed_use_gpu_final_norm() {
                let final_norm = session.run_final_norm_resident(&hidden, &final_norm_weight)?;
                metrics.norm_duration += norm_started.elapsed();
                let logits_started = Instant::now();
                let packed_activation = session.run_pack_f16_pairs_resident(final_norm)?;
                let next_token = session.run_projection_argmax_from_packed_activation(
                    "model.embed_tokens.weight",
                    self.config.vocab_size,
                    self.config.hidden_size,
                    packed_activation,
                )?;
                metrics.logits_duration += logits_started.elapsed();
                next_token
            } else {
                let hidden =
                    weighted_rms_norm(&hidden, &final_norm_weight, self.config.rms_norm_eps as f32);
                let norm_elapsed = norm_started.elapsed();
                metrics.norm_duration += norm_elapsed;
                *non_offloaded_dense_duration += norm_elapsed;
                session.push_dense_stage_trace("final_norm", "model.norm.weight", norm_elapsed);
                let logits_started = Instant::now();
                let next_token = session.run_projection_argmax(
                    "model.embed_tokens.weight",
                    self.config.vocab_size,
                    self.config.hidden_size,
                    &hidden,
                )?;
                metrics.logits_duration += logits_started.elapsed();
                next_token
            };
            PackedDecodeStepResult::NextToken(next_token)
        } else {
            let norm_started = Instant::now();
            let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
            let logits = if let Some(hidden_gpu) = final_hidden_gpu {
                let hidden_gpu_output = GpuResidentBuffer::new(
                    hidden_gpu.runner.borrow().shared_context().clone(),
                    hidden_gpu.runner.borrow().output_buffer_handle(),
                    hidden_gpu.runner.borrow().len(),
                    hidden_gpu.runner.borrow().output_buffer_size(),
                );
                let (logits, tail_report, compile_duration) = gpu_first_session
                    .run_tail_logits_from_resident_hidden(&hidden_gpu_output, &final_norm_weight)?;
                metrics.norm_duration += norm_started.elapsed();
                metrics.logits_duration += tail_report.pack_gpu_duration
                    + tail_report.logits_gpu_duration
                    + tail_report.logits_download_duration;
                session.metrics.compile_duration += hidden_gpu.compile_duration + compile_duration;
                session.metrics.gpu_duration += hidden_gpu.report.gpu_duration
                    + tail_report.final_norm_gpu_duration
                    + tail_report.pack_gpu_duration
                    + tail_report.logits_gpu_duration;
                session.metrics.download_duration += tail_report.logits_download_duration;
                logits
            } else if Self::packed_use_gpu_tail() {
                let (logits, tail_report, compile_duration) =
                    gpu_first_session.run_tail_logits_from_host_hidden(&hidden, &final_norm_weight)?;
                metrics.norm_duration += norm_started.elapsed();
                metrics.logits_duration += tail_report.pack_gpu_duration
                    + tail_report.logits_gpu_duration
                    + tail_report.logits_download_duration;
                session.metrics.compile_duration += compile_duration;
                session.metrics.gpu_duration += tail_report.final_norm_gpu_duration
                    + tail_report.pack_gpu_duration
                    + tail_report.logits_gpu_duration;
                session.metrics.download_duration += tail_report.logits_download_duration;
                logits
            } else {
                let hidden =
                    weighted_rms_norm(&hidden, &final_norm_weight, self.config.rms_norm_eps as f32);
                let norm_elapsed = norm_started.elapsed();
                metrics.norm_duration += norm_elapsed;
                *non_offloaded_dense_duration += norm_elapsed;
                session.push_dense_stage_trace("final_norm", "model.norm.weight", norm_elapsed);
                let logits_started = Instant::now();
                let logits = self.matvec_f16_resolved("model.embed_tokens.weight", &hidden)?;
                let logits_elapsed = logits_started.elapsed();
                metrics.logits_duration += logits_elapsed;
                *non_offloaded_dense_duration += logits_elapsed;
                session.push_dense_stage_trace(
                    "logits_dense",
                    "model.embed_tokens.weight",
                    logits_elapsed,
                );
                logits
            };
            PackedDecodeStepResult::Logits(logits)
        };
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step_hybrid_qkv_gu(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut [LayerCache],
        metrics: &mut DecodeMetrics,
        hybrid_layer_idx: usize,
        q_gpu: &Rc<RefCell<CachedGpuPackedMatvecRunner>>,
        k_gpu: &Rc<RefCell<CachedGpuPackedMatvecRunner>>,
        v_gpu: &Rc<RefCell<CachedGpuPackedMatvecRunner>>,
        o_gpu: Option<&Rc<RefCell<CachedGpuPackedMatvecRunner>>>,
        gate_gpu: &Rc<RefCell<CachedGpuPackedMatvecRunner>>,
        up_gpu: &Rc<RefCell<CachedGpuPackedMatvecRunner>>,
        down_gpu: Option<&Rc<RefCell<CachedGpuPackedMatvecRunner>>>,
        logits_gpu: Option<&Rc<RefCell<CachedGpuPackedMatvecRunner>>>,
    ) -> Result<(usize, Duration, Duration, Duration), ReferenceError> {
        let started_at = Instant::now();
        let mut hidden = self
            .weights
            .embedding_lookup("model.embed_tokens.weight", token_id)?;
        metrics.embedding_duration += started_at.elapsed();

        let (cos, sin) = rope_cos_sin(&self.rope, &[position]);
        let mut upload = Duration::ZERO;
        let mut gpu = Duration::ZERO;
        let mut download = Duration::ZERO;

        for (layer_idx, layer_cache) in cache
            .iter_mut()
            .enumerate()
            .take(self.config.num_hidden_layers)
        {
            let prefix = format!("model.layers.{layer_idx}");

            let started_at = Instant::now();
            let input_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.input_layernorm.weight"))?;
            let mut hidden_states =
                weighted_rms_norm(&hidden, &input_norm_weight, self.config.rms_norm_eps as f32);
            metrics.norm_duration += started_at.elapsed();
            let residual = hidden;

            let started_at = Instant::now();
            let mut q = if layer_idx == hybrid_layer_idx {
                let (out, report) = q_gpu
                    .borrow_mut()
                    .run_with_output(&hidden_states, None)
                    .map_err(|error| {
                        ReferenceError::Decode(format!("gpu q_proj failed: {error}"))
                    })?;
                upload += report.upload_duration;
                gpu += report.gpu_duration;
                download += report.download_duration;
                out
            } else {
                self.weights
                    .matvec_f16(&format!("{prefix}.self_attn.q_proj.weight"), &hidden_states)?
            };
            let mut k = if layer_idx == hybrid_layer_idx {
                let (out, report) = k_gpu
                    .borrow_mut()
                    .run_with_output(&hidden_states, None)
                    .map_err(|error| {
                        ReferenceError::Decode(format!("gpu k_proj failed: {error}"))
                    })?;
                upload += report.upload_duration;
                gpu += report.gpu_duration;
                download += report.download_duration;
                out
            } else {
                self.weights
                    .matvec_f16(&format!("{prefix}.self_attn.k_proj.weight"), &hidden_states)?
            };
            let v = if layer_idx == hybrid_layer_idx {
                let (out, report) = v_gpu
                    .borrow_mut()
                    .run_with_output(&hidden_states, None)
                    .map_err(|error| {
                        ReferenceError::Decode(format!("gpu v_proj failed: {error}"))
                    })?;
                upload += report.upload_duration;
                gpu += report.gpu_duration;
                download += report.download_duration;
                out
            } else {
                self.weights
                    .matvec_f16(&format!("{prefix}.self_attn.v_proj.weight"), &hidden_states)?
            };
            metrics.qkv_duration += started_at.elapsed();

            let started_at = Instant::now();
            let q_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.self_attn.q_norm.weight"))?;
            let k_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.self_attn.k_norm.weight"))?;
            apply_head_rms_norm_weighted(
                &mut q,
                self.config.num_attention_heads,
                self.config.head_dim,
                &q_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            apply_head_rms_norm_weighted(
                &mut k,
                self.config.num_key_value_heads,
                self.config.head_dim,
                &k_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            apply_rotary_single(
                &mut q,
                &mut k,
                &cos,
                &sin,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            );
            metrics.norm_duration += started_at.elapsed();

            let started_at = Instant::now();
            layer_cache.keys.extend_from_slice(&k);
            layer_cache.values.extend_from_slice(&v);
            let attn = attention_single_query(
                &q,
                &layer_cache.keys,
                &layer_cache.values,
                position + 1,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            );
            let attn_output = if layer_idx == hybrid_layer_idx {
                if let Some(o_gpu) = o_gpu {
                    let (out, report) =
                        o_gpu
                            .borrow_mut()
                            .run_with_output(&attn, None)
                            .map_err(|error| {
                                ReferenceError::Decode(format!("gpu o_proj failed: {error}"))
                            })?;
                    upload += report.upload_duration;
                    gpu += report.gpu_duration;
                    download += report.download_duration;
                    out
                } else {
                    self.weights
                        .matvec_f16(&format!("{prefix}.self_attn.o_proj.weight"), &attn)?
                }
            } else {
                self.weights
                    .matvec_f16(&format!("{prefix}.self_attn.o_proj.weight"), &attn)?
            };
            hidden = residual
                .iter()
                .zip(attn_output.iter())
                .map(|(left, right)| left + right)
                .collect();
            metrics.attention_duration += started_at.elapsed();

            let residual = hidden;
            let started_at = Instant::now();
            let post_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;
            hidden_states = weighted_rms_norm(
                &residual,
                &post_norm_weight,
                self.config.rms_norm_eps as f32,
            );
            metrics.norm_duration += started_at.elapsed();

            let started_at = Instant::now();
            let gate = if layer_idx == hybrid_layer_idx {
                let (out, report) = gate_gpu
                    .borrow_mut()
                    .run_with_output(&hidden_states, None)
                    .map_err(|error| {
                        ReferenceError::Decode(format!("gpu gate_proj failed: {error}"))
                    })?;
                upload += report.upload_duration;
                gpu += report.gpu_duration;
                download += report.download_duration;
                out
            } else {
                self.weights
                    .matvec_f16(&format!("{prefix}.mlp.gate_proj.weight"), &hidden_states)?
            };
            let up = if layer_idx == hybrid_layer_idx {
                let (out, report) = up_gpu
                    .borrow_mut()
                    .run_with_output(&hidden_states, None)
                    .map_err(|error| {
                        ReferenceError::Decode(format!("gpu up_proj failed: {error}"))
                    })?;
                upload += report.upload_duration;
                gpu += report.gpu_duration;
                download += report.download_duration;
                out
            } else {
                self.weights
                    .matvec_f16(&format!("{prefix}.mlp.up_proj.weight"), &hidden_states)?
            };
            let mlp = swiglu(&gate, &up);
            let down = if layer_idx == hybrid_layer_idx {
                if let Some(down_gpu) = down_gpu {
                    let (out, report) =
                        down_gpu
                            .borrow_mut()
                            .run_with_output(&mlp, None)
                            .map_err(|error| {
                                ReferenceError::Decode(format!("gpu down_proj failed: {error}"))
                            })?;
                    upload += report.upload_duration;
                    gpu += report.gpu_duration;
                    download += report.download_duration;
                    out
                } else {
                    self.weights
                        .matvec_f16(&format!("{prefix}.mlp.down_proj.weight"), &mlp)?
                }
            } else {
                self.weights
                    .matvec_f16(&format!("{prefix}.mlp.down_proj.weight"), &mlp)?
            };
            hidden = residual
                .iter()
                .zip(down.iter())
                .map(|(left, right)| left + right)
                .collect();
            metrics.mlp_duration += started_at.elapsed();
        }

        let started_at = Instant::now();
        let final_norm_weight = self.weights.load_vector_f32("model.norm.weight")?;
        let hidden =
            weighted_rms_norm(&hidden, &final_norm_weight, self.config.rms_norm_eps as f32);
        metrics.norm_duration += started_at.elapsed();

        let started_at = Instant::now();
        let next_token = if let Some(logits_gpu) = logits_gpu {
            let (argmax_index, report) = logits_gpu
                .borrow_mut()
                .run_with_argmax(&hidden)
                .map_err(|error| ReferenceError::Decode(format!("gpu logits failed: {error}")))?;
            upload += report.upload_duration;
            gpu += report.gpu_duration;
            download += report.download_duration;
            argmax_index
        } else {
            let logits = self
                .weights
                .matvec_f16("model.embed_tokens.weight", &hidden)?;
            argmax(&logits).ok_or_else(|| {
                ReferenceError::Decode("argmax failed on empty logits".to_string())
            })?
        };
        metrics.logits_duration += started_at.elapsed();
        Ok((next_token, upload, gpu, download))
    }
}

fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

fn weighted_rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    assert_eq!(
        input.len(),
        weight.len(),
        "weighted_rms_norm dimensions must match"
    );
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (mean_square + epsilon).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(value, gamma)| value * scale * gamma)
        .collect()
}

fn apply_head_rms_norm_weighted(
    values: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    weight: &[f32],
    epsilon: f32,
) {
    assert_eq!(values.len(), num_heads * head_dim);
    assert_eq!(weight.len(), head_dim);
    for head in 0..num_heads {
        let offset = head * head_dim;
        let slice = &mut values[offset..offset + head_dim];
        let mean_square = slice.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
        let scale = 1.0 / (mean_square + epsilon).sqrt();
        for i in 0..head_dim {
            slice[i] *= scale * weight[i];
        }
    }
}

fn rotate_half(x: &[f32]) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = Vec::with_capacity(x.len());
    out.extend(x[half..].iter().map(|value| -*value));
    out.extend_from_slice(&x[..half]);
    out
}

fn apply_rotary_single(
    query: &mut [f32],
    key: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) {
    assert_eq!(cos.len(), head_dim);
    assert_eq!(sin.len(), head_dim);
    for head in 0..num_query_heads {
        let offset = head * head_dim;
        let rotated = rotate_half(&query[offset..offset + head_dim]);
        for i in 0..head_dim {
            query[offset + i] = query[offset + i] * cos[i] + rotated[i] * sin[i];
        }
    }
    for head in 0..num_key_value_heads {
        let offset = head * head_dim;
        let rotated = rotate_half(&key[offset..offset + head_dim]);
        for i in 0..head_dim {
            key[offset + i] = key[offset + i] * cos[i] + rotated[i] * sin[i];
        }
    }
}

fn attention_single_query(
    query: &[f32],
    cached_keys: &[f32],
    cached_values: &[f32],
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_rep = num_query_heads / num_key_value_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0; num_query_heads * head_dim];

    for head in 0..num_query_heads {
        let kv_head = head / n_rep;
        let q_offset = head * head_dim;
        let q_slice = &query[q_offset..q_offset + head_dim];
        let mut logits = vec![0.0; seq_len];
        for (position, logit) in logits.iter_mut().enumerate().take(seq_len) {
            let k_offset = (position * num_key_value_heads + kv_head) * head_dim;
            let k_slice = &cached_keys[k_offset..k_offset + head_dim];
            *logit = q_slice
                .iter()
                .zip(k_slice)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                * scale;
        }
        let probs = crate::cpu::primitives::softmax(&logits);
        for (position, prob) in probs.iter().enumerate() {
            let v_offset = (position * num_key_value_heads + kv_head) * head_dim;
            let v_slice = &cached_values[v_offset..v_offset + head_dim];
            for i in 0..head_dim {
                output[q_offset + i] += prob * v_slice[i];
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::{PackedDecodeSession, PackedDecodeStepResult, ReferenceModel};
    use crate::runtime::packed_model::write_packed_model_artifact;
    use half::f16;
    use safetensors::tensor::{Dtype, TensorView, serialize};
    use std::collections::BTreeMap;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;
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
            r#"{
                "vocab_size": 4,
                "max_position_embeddings": 32,
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "hidden_act": "silu",
                "rms_norm_eps": 0.000001,
                "rope_theta": 10000.0,
                "rope_scaling": {"rope_type": "yarn", "factor": 1.0, "original_max_position_embeddings": 32},
                "attention_bias": false,
                "tie_word_embeddings": true,
                "architectures": ["Qwen3ForCausalLM"],
                "pad_token_id": 0,
                "eos_token_id": 3,
                "model_type": "qwen3"
            }"#,
        )
        .expect("config should be written");
        fs::write(
            root.join("generation_config.json"),
            r#"{"eos_token_id":3,"pad_token_id":0,"begin_suppress_tokens":[],"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"do_sample":false}"#,
        )
        .expect("generation config should be written");
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{"add_bos_token":false,"add_prefix_space":false,"added_tokens_decoder":{},"additional_special_tokens":[],"bos_token":null,"clean_up_tokenization_spaces":false,"eos_token":"tok3","model_max_length":32,"pad_token":"tok0","split_special_tokens":false,"tokenizer_class":"WordLevel","unk_token":"[UNK]"}"#,
        )
        .expect("tokenizer config should be written");

        let vocab = std::collections::HashMap::from([
            ("[UNK]".to_string(), 0),
            ("tok1".to_string(), 1),
            ("tok2".to_string(), 2),
            ("tok3".to_string(), 3),
        ]);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel should build");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Whitespace);
        tokenizer
            .save(root.join("tokenizer.json"), false)
            .expect("tokenizer should save");

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
        let bytes = serialize(tensors, &None).expect("serialization should succeed");
        fs::write(root.join("model.safetensors"), bytes).expect("model should be written");
    }

    fn write_two_layer_synthetic_model(root: &std::path::Path) {
        fs::write(
            root.join("config.json"),
            r#"{
                "vocab_size": 4,
                "max_position_embeddings": 32,
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "hidden_act": "silu",
                "rms_norm_eps": 0.000001,
                "rope_theta": 10000.0,
                "rope_scaling": {"rope_type": "yarn", "factor": 1.0, "original_max_position_embeddings": 32},
                "attention_bias": false,
                "tie_word_embeddings": true,
                "architectures": ["Qwen3ForCausalLM"],
                "pad_token_id": 0,
                "eos_token_id": 3,
                "model_type": "qwen3"
            }"#,
        )
        .expect("config should be written");
        fs::write(
            root.join("generation_config.json"),
            r#"{"eos_token_id":3,"pad_token_id":0,"begin_suppress_tokens":[],"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"do_sample":false}"#,
        )
        .expect("generation config should be written");
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{"add_bos_token":false,"add_prefix_space":false,"added_tokens_decoder":{},"additional_special_tokens":[],"bos_token":null,"clean_up_tokenization_spaces":false,"eos_token":"tok3","model_max_length":32,"pad_token":"tok0","split_special_tokens":false,"tokenizer_class":"WordLevel","unk_token":"[UNK]"}"#,
        )
        .expect("tokenizer config should be written");

        let vocab = std::collections::HashMap::from([
            ("[UNK]".to_string(), 0),
            ("tok1".to_string(), 1),
            ("tok2".to_string(), 2),
            ("tok3".to_string(), 3),
        ]);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel should build");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Whitespace);
        tokenizer
            .save(root.join("tokenizer.json"), false)
            .expect("tokenizer should save");

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
        for layer_idx in 0..2 {
            let prefix = format!("model.layers.{layer_idx}");
            tensors.insert(
                format!("{prefix}.input_layernorm.weight"),
                TensorView::new(Dtype::F16, vec![4], &ones4).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                TensorView::new(Dtype::F16, vec![4], &ones4).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.self_attn.q_norm.weight"),
                TensorView::new(Dtype::F16, vec![2], &ones2).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.self_attn.k_norm.weight"),
                TensorView::new(Dtype::F16, vec![2], &ones2).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                TensorView::new(Dtype::F16, vec![4, 4], &zeros_4x4).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                TensorView::new(Dtype::F16, vec![2, 4], &zeros_2x4).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                TensorView::new(Dtype::F16, vec![2, 4], &zeros_2x4).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                TensorView::new(Dtype::F16, vec![4, 4], &zeros_4x4).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                TensorView::new(Dtype::F16, vec![8, 4], &zeros_8x4).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                TensorView::new(Dtype::F16, vec![8, 4], &zeros_8x4).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                TensorView::new(Dtype::F16, vec![4, 8], &zeros_4x8).unwrap(),
            );
        }
        tensors.insert(
            "model.norm.weight".to_string(),
            TensorView::new(Dtype::F16, vec![4], &ones4).unwrap(),
        );
        let bytes = serialize(tensors, &None).expect("serialization should succeed");
        fs::write(root.join("model.safetensors"), bytes).expect("model should be written");
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn set_env(key: &str, value: &str) {
        unsafe { std::env::set_var(key, value) }
    }

    fn remove_env(key: &str) {
        unsafe { std::env::remove_var(key) }
    }

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    struct ScopedEnvVars(Vec<(&'static str, Option<String>)>);

    impl ScopedEnvVars {
        fn set(pairs: &[(&'static str, &'static str)]) -> Self {
            let previous = pairs
                .iter()
                .map(|(key, value)| {
                    let old = std::env::var(key).ok();
                    set_env(key, value);
                    (*key, old)
                })
                .collect();
            Self(previous)
        }
    }

    impl Drop for ScopedEnvVars {
        fn drop(&mut self) {
            for (key, value) in self.0.drain(..) {
                match value {
                    Some(old) => set_env(key, &old),
                    None => remove_env(key),
                }
            }
        }
    }

    #[test]
    fn runs_end_to_end_greedy_decode_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let model = ReferenceModel::load_from_root(dir.path()).expect("model should load");
        let result = model
            .generate_greedy("tok2", 1)
            .expect("decode should succeed");

        assert_eq!(result.output_token_ids, vec![2, 2]);
        assert_eq!(result.output_text, "tok2 tok2");
        assert_eq!(result.metrics.generated_tokens, 1);
        assert!(result.metrics.logits_duration > Duration::ZERO);
        assert!(result.metrics.summarize().contains("generated_tokens=1"));
    }

    #[test]
    fn runs_end_to_end_packed_gpu_decode_from_packed_artifact_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let result = model
            .generate_packed_greedy("tok2", 1, true, true)
            .expect("packed gpu decode should succeed");

        assert_eq!(result.output_token_ids, vec![2, 2]);
        assert_eq!(result.output_text, "tok2 tok2");
        assert_eq!(result.metrics.output_text, "tok2 tok2");
        assert_eq!(
            result.metrics.dispatch_count,
            (model.config.num_hidden_layers * 2 + 1) * 2
        );
        assert!(!result.dispatch_trace.is_empty());
        assert!(
            result
                .dispatch_trace
                .iter()
                .any(|trace| trace.path == "gpu_packed" && trace.stage == "attention_qkv")
        );
        assert!(
            result
                .dispatch_trace
                .iter()
                .any(|trace| trace.path == "gpu_packed" && trace.stage == "logits_argmax")
        );
        assert!(
            result
                .dispatch_trace
                .iter()
                .any(|trace| trace.path == "dense_cpu" && trace.stage == "attention_core")
        );
        assert!(
            result
                .dispatch_trace
                .iter()
                .any(|trace| trace.path == "dense_cpu" && trace.stage == "mlp_swiglu")
        );
        let summary = result.metrics.summarize();
        assert!(summary.contains("attention_query_ms="));
        assert!(summary.contains("attention_oproj_ms="));
        assert!(summary.contains("mlp_down_ms="));
        assert!(result.metrics.gpu_duration > Duration::ZERO);
    }

    #[test]
    fn runs_end_to_end_gpu_block_decode_from_packed_artifact_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");

        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_PACKED_MLP_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
            ("JENGINE_GPU_SWIGLU_BLOCK", "1"),
            ("JENGINE_GPU_FULL_LAST_LAYER", "1"),
        ]);

        let result = model
            .generate_packed_greedy("tok2", 1, true, true)
            .expect("gpu block packed decode should succeed");

        assert_eq!(result.output_token_ids, vec![2, 2]);
        assert_eq!(result.output_text, "tok2 tok2");
        assert!(result.metrics.attention_duration > Duration::ZERO);
        assert!(result.metrics.mlp_duration > Duration::ZERO);
        assert!(result.metrics.logits_duration > Duration::ZERO);
        assert!(!result.dispatch_trace.is_empty());
    }

    #[test]
    fn reuses_persistent_packed_decode_session_across_generation_loop() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let predicted = match session
            .push_prompt_token(2)
            .expect("prompt token should decode")
        {
            PackedDecodeStepResult::NextToken(token_id) => token_id,
            PackedDecodeStepResult::Logits(_) => unreachable!("argmax-only session should predict"),
        };
        assert_eq!(predicted, 2);
        let predicted = match session
            .push_generated_token(predicted)
            .expect("generated token should decode")
        {
            PackedDecodeStepResult::NextToken(token_id) => token_id,
            PackedDecodeStepResult::Logits(_) => unreachable!("argmax-only session should predict"),
        };
        assert_eq!(predicted, 2);

        let result = session.finish_result(
            ReferenceModel::packed_enabled_label(true, true),
            Duration::ZERO,
            vec![2, 2],
            "tok2 tok2".to_string(),
        );
        assert_eq!(result.output_token_ids, vec![2, 2]);
        assert_eq!(result.decode_metrics.prompt_tokens, 1);
        assert_eq!(result.decode_metrics.generated_tokens, 1);
        assert!(!result.dispatch_trace.is_empty());
    }

    #[test]
    fn selects_gpu_first_decode_session_when_gpu_first_flags_are_enabled() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();

        let legacy = model.begin_packed_decode_session(2, true, true, true);
        assert!(!legacy.is_gpu_first());

        let _env = ScopedEnvVars::set(&[("JENGINE_GPU_ATTENTION_BLOCK", "1")]);
        let gpu_first = model.begin_packed_decode_session(2, true, true, true);
        assert!(gpu_first.is_gpu_first());
    }

    #[test]
    fn populates_gpu_kv_cache_when_gpu_first_attention_is_enabled() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu-first prompt token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when attention flag is enabled");
        };
        let kv_cache = session
            .inner
            .gpu_first_session
            .gpu_kv_caches
            .get(&0)
            .expect("layer 0 gpu kv cache should exist");
        assert_eq!(kv_cache.len_tokens(), 1);
        assert_eq!(
            kv_cache
                .snapshot_keys()
                .expect("kv keys should read back")
                .len(),
            model.config.num_key_value_heads * model.config.head_dim
        );
        assert_eq!(
            kv_cache
                .snapshot_values()
                .expect("kv values should read back")
                .len(),
            model.config.num_key_value_heads * model.config.head_dim
        );
    }

    #[test]
    fn avoids_cpu_kv_cache_on_gpu_first_attention_mlp_path() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_PACKED_MLP_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
            ("JENGINE_GPU_SWIGLU_BLOCK", "1"),
            ("JENGINE_GPU_FULL_LAST_LAYER", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu-first prompt token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when block flags are enabled");
        };
        assert!(session.inner.cache[0].keys.is_empty());
        assert!(session.inner.cache[0].values.is_empty());
        assert_eq!(session.inner.cache[0].keys.capacity(), 0);
        assert_eq!(session.inner.cache[0].values.capacity(), 0);
        let kv_cache = session
            .inner
            .gpu_first_session
            .gpu_kv_caches
            .get(&0)
            .expect("layer 0 gpu kv cache should exist");
        assert_eq!(kv_cache.len_tokens(), 1);
    }

    #[test]
    fn avoids_cpu_kv_cache_on_gpu_attention_only_path() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu attention token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu attention is enabled");
        };
        assert!(session.inner.cache[0].keys.is_empty());
        assert!(session.inner.cache[0].values.is_empty());
        assert_eq!(session.inner.cache[0].keys.capacity(), 0);
        assert_eq!(session.inner.cache[0].values.capacity(), 0);
        let kv_cache = session
            .inner
            .gpu_first_session
            .gpu_kv_caches
            .get(&0)
            .expect("layer 0 gpu kv cache should exist");
        assert_eq!(kv_cache.len_tokens(), 1);
    }

    #[test]
    fn gpu_attention_block_path_does_not_require_packed_attention_full_flag() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[("JENGINE_GPU_ATTENTION_BLOCK", "1")]);

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu attention token should decode without packed attention full flag");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu attention is enabled");
        };
        assert!(session.inner.cache[0].keys.is_empty());
        assert!(session.inner.cache[0].values.is_empty());
        assert!(session
            .inner
            .gpu_first_session
            .attention_blocks
            .contains_key(&(0, 1)));
    }

    #[test]
    fn creates_gpu_embedding_runner_and_matches_reference_embedding_row() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[("JENGINE_GPU_EMBEDDING", "1")]);

        let reference_hidden = model
            .embedding_lookup_resolved("model.embed_tokens.weight", 2)
            .expect("reference embedding should load");

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu embedding prompt token should decode");

        let PackedDecodeSession::GpuFirst(mut session) = session else {
            panic!("gpu-first session should be selected when gpu embedding is enabled");
        };
        let runner = session
            .inner
            .gpu_first_session
            .embedding_lookup_runner
            .as_mut()
            .expect("embedding runner should be created");
        let (gpu_hidden, _) = runner
            .run_with_output(2)
            .expect("gpu embedding lookup should succeed");
        assert_eq!(gpu_hidden, reference_hidden);
    }

    #[test]
    fn creates_first_layer_raw_f32_qkv_runner_when_gpu_embedding_and_packed_qkv_are_enabled() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_GPU_EMBEDDING", "1"),
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu embedding + packed qkv token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu embedding is enabled");
        };
        let expected_key = format!(
            "gpu_first::layer::0::qkv_triplet::{}||{}||{}",
            model.layer_tensors[0].q_proj_weight,
            model.layer_tensors[0].k_proj_weight,
            model.layer_tensors[0].v_proj_weight,
        );
        assert!(
            session
                .inner
                .gpu_first_session
                .raw_f32_projection_runners
                .contains_key(&expected_key)
        );
    }

    #[test]
    fn uses_first_layer_resident_qkv_tensors_when_gpu_embedding_and_gpu_attention_are_enabled() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_GPU_EMBEDDING", "1"),
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, true, false, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu embedding + gpu attention token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu embedding and attention are enabled");
        };
        let expected_q_key = format!(
            "gpu_first::layer::0::q_proj::{}",
            model.layer_tensors[0].q_proj_weight,
        );
        let expected_k_key = format!(
            "gpu_first::layer::0::k_proj::{}",
            model.layer_tensors[0].k_proj_weight,
        );
        let expected_v_key = format!(
            "gpu_first::layer::0::v_proj::{}",
            model.layer_tensors[0].v_proj_weight,
        );
        let legacy_triplet_key = format!(
            "gpu_first::layer::0::qkv_triplet::{}||{}||{}",
            model.layer_tensors[0].q_proj_weight,
            model.layer_tensors[0].k_proj_weight,
            model.layer_tensors[0].v_proj_weight,
        );
        assert!(session
            .inner
            .gpu_first_session
            .raw_f32_projection_runners
            .contains_key(&expected_q_key));
        assert!(session
            .inner
            .gpu_first_session
            .raw_f32_projection_runners
            .contains_key(&expected_k_key));
        assert!(session
            .inner
            .gpu_first_session
            .raw_f32_projection_runners
            .contains_key(&expected_v_key));
        assert!(!session
            .inner
            .gpu_first_session
            .raw_f32_projection_runners
            .contains_key(&legacy_triplet_key));
        assert!(session.inner.gpu_first_session.qk_rope_runner.is_some());
        assert!(session.inner.cache[0].keys.is_empty());
        assert!(session.inner.cache[0].values.is_empty());
        assert_eq!(
            session
                .inner
                .gpu_first_session
                .gpu_kv_caches
                .get(&0)
                .expect("layer 0 gpu kv cache should exist")
                .len_tokens(),
            1
        );
    }

    #[test]
    fn creates_second_layer_raw_f32_qkv_runner_from_resident_gpu_hidden_path() {
        let dir = tempdir().expect("tempdir should be created");
        write_two_layer_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_PACKED_MLP_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
            ("JENGINE_GPU_SWIGLU_BLOCK", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("two-layer gpu resident token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu block flags are enabled");
        };
        let expected_q_key = format!(
            "gpu_first::layer::1::q_proj::{}",
            model.layer_tensors[1].q_proj_weight,
        );
        let expected_k_key = format!(
            "gpu_first::layer::1::k_proj::{}",
            model.layer_tensors[1].k_proj_weight,
        );
        let expected_v_key = format!(
            "gpu_first::layer::1::v_proj::{}",
            model.layer_tensors[1].v_proj_weight,
        );
        assert!(
            session
                .inner
                .gpu_first_session
                .raw_f32_projection_runners
                .contains_key(&expected_q_key)
        );
        assert!(
            session
                .inner
                .gpu_first_session
                .raw_f32_projection_runners
                .contains_key(&expected_k_key)
        );
        assert!(
            session
                .inner
                .gpu_first_session
                .raw_f32_projection_runners
                .contains_key(&expected_v_key)
        );
        assert!(session.inner.cache[0].keys.is_empty());
        assert!(session.inner.cache[0].values.is_empty());
        assert!(session.inner.cache[1].keys.is_empty());
        assert!(session.inner.cache[1].values.is_empty());
        assert_eq!(
            session
                .inner
                .gpu_first_session
                .gpu_kv_caches
                .get(&0)
                .expect("layer 0 gpu kv cache should exist")
                .len_tokens(),
            1
        );
        assert_eq!(
            session
                .inner
                .gpu_first_session
                .gpu_kv_caches
                .get(&1)
                .expect("layer 1 gpu kv cache should exist")
                .len_tokens(),
            1
        );
    }

    #[test]
    fn runs_gpu_mlp_block_without_gpu_attention_block() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_MLP_FULL", "1"),
            ("JENGINE_GPU_SWIGLU_BLOCK", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, false, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu mlp token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu mlp is enabled");
        };
        assert!(session.inner.gpu_first_session.mlp_blocks.contains_key(&0));
    }

    #[test]
    fn gpu_mlp_block_path_does_not_require_packed_mlp_full_flag() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[("JENGINE_GPU_SWIGLU_BLOCK", "1")]);

        let mut session = model.begin_packed_decode_session(2, false, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu mlp token should decode without packed mlp full flag");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu mlp is enabled");
        };
        assert!(session.inner.gpu_first_session.mlp_blocks.contains_key(&0));
    }

    #[test]
    fn runs_gpu_tail_block_without_resident_hidden_pipeline() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[("JENGINE_GPU_TAIL", "1")]);

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu tail token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu tail is enabled");
        };
        assert!(session.inner.gpu_first_session.tail_block.is_some());
    }

    #[test]
    fn runs_gpu_attention_tail_path_without_gpu_mlp_block() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
            ("JENGINE_GPU_TAIL", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu attention tail token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu attention tail is enabled");
        };
        assert!(
            session
                .inner
                .gpu_first_session
                .attention_blocks
                .contains_key(&(0, 1))
        );
        assert!(session.inner.gpu_first_session.tail_block.is_some());
    }

    #[test]
    fn runs_gpu_mlp_tail_path_without_gpu_attention_block() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_MLP_FULL", "1"),
            ("JENGINE_GPU_SWIGLU_BLOCK", "1"),
            ("JENGINE_GPU_TAIL", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, false, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu mlp tail token should decode");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu mlp tail is enabled");
        };
        assert!(session.inner.gpu_first_session.mlp_blocks.contains_key(&0));
        assert!(session.inner.gpu_first_session.tail_block.is_some());
    }

    #[test]
    fn runs_gpu_full_last_layer_from_resident_hidden_path() {
        let dir = tempdir().expect("tempdir should be created");
        write_two_layer_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_PACKED_MLP_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
            ("JENGINE_GPU_SWIGLU_BLOCK", "1"),
            ("JENGINE_GPU_FULL_LAST_LAYER", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(2, true, true, true);
        let _ = session
            .push_prompt_token(2)
            .expect("gpu full-last-layer token should decode from resident hidden path");

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu full last layer is enabled");
        };
        assert!(session.inner.gpu_first_session.full_last_layer_block.is_some());
        assert!(session.inner.cache[0].keys.is_empty());
        assert!(session.inner.cache[0].values.is_empty());
        assert!(session.inner.cache[1].keys.is_empty());
        assert!(session.inner.cache[1].values.is_empty());
        assert_eq!(
            session
                .inner
                .gpu_first_session
                .gpu_kv_caches
                .get(&0)
                .expect("layer 0 gpu kv cache should exist")
                .len_tokens(),
            1
        );
        assert_eq!(
            session
                .inner
                .gpu_first_session
                .gpu_kv_caches
                .get(&1)
                .expect("layer 1 gpu kv cache should exist")
                .len_tokens(),
            1
        );
    }

    #[test]
    fn keeps_cpu_kv_empty_across_persistent_gpu_first_generation_loop() {
        let dir = tempdir().expect("tempdir should be created");
        write_two_layer_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_PACKED_MLP_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
            ("JENGINE_GPU_SWIGLU_BLOCK", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(3, true, true, true);
        let predicted = match session
            .push_prompt_token(2)
            .expect("gpu-first prompt token should decode")
        {
            PackedDecodeStepResult::NextToken(token_id) => token_id,
            PackedDecodeStepResult::Logits(_) => unreachable!("argmax-only session should predict"),
        };
        let predicted = match session
            .push_generated_token(predicted)
            .expect("gpu-first generated token should decode")
        {
            PackedDecodeStepResult::NextToken(token_id) => token_id,
            PackedDecodeStepResult::Logits(_) => unreachable!("argmax-only session should predict"),
        };
        assert_eq!(predicted, 2);

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu block flags are enabled");
        };
        for (layer_idx, layer_cache) in session.inner.cache.iter().enumerate().take(2) {
            assert!(
                layer_cache.keys.is_empty(),
                "layer {layer_idx} cpu key cache should stay empty"
            );
            assert!(
                layer_cache.values.is_empty(),
                "layer {layer_idx} cpu value cache should stay empty"
            );
            assert_eq!(
                layer_cache.keys.capacity(),
                0,
                "layer {layer_idx} cpu key cache should not be preallocated"
            );
            assert_eq!(
                layer_cache.values.capacity(),
                0,
                "layer {layer_idx} cpu value cache should not be preallocated"
            );
            assert_eq!(
                session
                    .inner
                    .gpu_first_session
                    .gpu_kv_caches
                    .get(&layer_idx)
                    .expect("layer gpu kv cache should exist")
                    .len_tokens(),
                2,
                "layer {layer_idx} gpu kv cache should contain both decoded tokens"
            );
        }
    }

    #[test]
    fn keeps_cpu_kv_empty_across_persistent_gpu_attention_only_loop() {
        let dir = tempdir().expect("tempdir should be created");
        write_two_layer_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[
            ("JENGINE_PACKED_ATTENTION_FULL", "1"),
            ("JENGINE_GPU_ATTENTION_BLOCK", "1"),
        ]);

        let mut session = model.begin_packed_decode_session(3, true, false, true);
        let predicted = match session
            .push_prompt_token(2)
            .expect("gpu attention prompt token should decode")
        {
            PackedDecodeStepResult::NextToken(token_id) => token_id,
            PackedDecodeStepResult::Logits(_) => unreachable!("argmax-only session should predict"),
        };
        let predicted = match session
            .push_generated_token(predicted)
            .expect("gpu attention generated token should decode")
        {
            PackedDecodeStepResult::NextToken(token_id) => token_id,
            PackedDecodeStepResult::Logits(_) => unreachable!("argmax-only session should predict"),
        };
        assert_eq!(predicted, 2);

        let PackedDecodeSession::GpuFirst(session) = session else {
            panic!("gpu-first session should be selected when gpu attention is enabled");
        };
        for (layer_idx, layer_cache) in session.inner.cache.iter().enumerate().take(2) {
            assert!(
                layer_cache.keys.is_empty(),
                "layer {layer_idx} cpu key cache should stay empty"
            );
            assert!(
                layer_cache.values.is_empty(),
                "layer {layer_idx} cpu value cache should stay empty"
            );
            assert_eq!(
                layer_cache.keys.capacity(),
                0,
                "layer {layer_idx} cpu key cache should not be preallocated"
            );
            assert_eq!(
                layer_cache.values.capacity(),
                0,
                "layer {layer_idx} cpu value cache should not be preallocated"
            );
            assert_eq!(
                session
                    .inner
                    .gpu_first_session
                    .gpu_kv_caches
                    .get(&layer_idx)
                    .expect("layer gpu kv cache should exist")
                    .len_tokens(),
                2,
                "layer {layer_idx} gpu kv cache should contain both decoded tokens"
            );
        }
    }

    #[test]
    fn benchmarks_hybrid_qproj_decode_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let model = ReferenceModel::load_from_root(dir.path()).expect("model should load");
        let result = model
            .benchmark_hybrid_qproj_decode("tok2", 1, 0)
            .expect("hybrid decode should succeed");
        assert!(result.total_duration > Duration::ZERO);
        assert!(result.summarize().contains("qproj_gpu_ms="));
        assert!(!result.q_proj_pack_cache_hit);
        assert!(!result.q_proj_gpu_cache_hit);
    }

    #[test]
    fn reuses_cached_hybrid_qproj_resources_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let model = ReferenceModel::load_from_root(dir.path()).expect("model should load");
        let first = model
            .benchmark_cached_hybrid_qproj_decode("tok2", 1, 0)
            .expect("cached hybrid decode should succeed");
        let second = model
            .benchmark_cached_hybrid_qproj_decode("tok2", 1, 0)
            .expect("cached warm hybrid decode should succeed");
        assert!(!first.q_proj_pack_cache_hit);
        assert!(!first.q_proj_gpu_cache_hit);
        assert!(second.q_proj_pack_cache_hit);
        assert!(second.q_proj_gpu_cache_hit);
        assert!(second.summarize().contains("qproj_pack_cache_hit=true"));
    }

    #[test]
    fn reports_memory_estimates_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let model = ReferenceModel::load_from_root(dir.path()).expect("model should load");
        let report = model.memory_report(3, 2);
        assert_eq!(report.total_sequence_tokens, 5);
        assert!(report.kv_cache_total_bytes_runtime_f32 > 0);
        assert!(report.summarize().contains("working_set_bytes="));
    }

    #[test]
    fn runs_end_to_end_greedy_decode_from_packed_artifact_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let _guard = lock_env();
        let result = model
            .generate_greedy("tok2", 1)
            .expect("packed-first decode should succeed");

        assert_eq!(result.output_token_ids, vec![2, 2]);
        assert_eq!(result.output_text, "tok2 tok2");
        let report = model.memory_report(1, 1);
        assert!(report.packed_cache_bytes > 0);
    }

    #[test]
    fn uses_persistent_packed_artifact_for_hybrid_qproj_cache_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let model =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed-first model should load");
        let metrics = model
            .benchmark_cached_hybrid_qproj_decode("tok2", 1, 0)
            .expect("cached hybrid decode should succeed");

        assert_eq!(metrics.q_proj_pack_duration, Duration::ZERO);
        assert!(!metrics.q_proj_pack_cache_hit);
    }

    #[test]
    fn compares_prefill_logits_between_dense_and_packed_variants_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let dense = ReferenceModel::load_from_root(dir.path()).expect("dense model should load");
        let packed =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed model should load");
        let _guard = lock_env();

        let attention = packed
            .compare_prefill_logits_against(&dense, "tok2", true, false)
            .expect("attention validation should succeed");
        let mlp = packed
            .compare_prefill_logits_against(&dense, "tok2", false, true)
            .expect("mlp validation should succeed");
        let combined = packed
            .compare_prefill_logits_against(&dense, "tok2", true, true)
            .expect("combined validation should succeed");

        assert_eq!(attention.max_abs_diff, 0.0);
        assert_eq!(mlp.max_abs_diff, 0.0);
        assert_eq!(combined.max_abs_diff, 0.0);
    }

    #[test]
    fn compares_prefill_logits_with_gpu_tail_against_dense_reference() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let dense = ReferenceModel::load_from_root(dir.path()).expect("dense model should load");
        let packed =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed model should load");
        let _guard = lock_env();
        let _env = ScopedEnvVars::set(&[("JENGINE_GPU_TAIL", "1")]);

        let report = packed
            .compare_prefill_logits_against(&dense, "tok2", true, true)
            .expect("gpu tail logits validation should succeed");

        assert_eq!(report.max_abs_diff, 0.0);
    }

    #[test]
    fn reduces_packed_dispatch_count_with_projection_pairing_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let packed =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed model should load");
        let _guard = lock_env();
        let layers = packed.config.num_hidden_layers;

        let (_, attention) = packed
            .benchmark_packed_prefill_chunk(Some(2), None, 0, layers, true, false, false)
            .expect("attention prefill chunk should succeed");
        let (_, mlp) = packed
            .benchmark_packed_prefill_chunk(Some(2), None, 0, layers, false, true, false)
            .expect("mlp prefill chunk should succeed");
        let (_, combined) = packed
            .benchmark_packed_prefill_chunk(Some(2), None, 0, layers, true, true, false)
            .expect("combined prefill chunk should succeed");

        assert_eq!(attention.dispatch_count, layers);
        assert_eq!(mlp.dispatch_count, layers);
        assert_eq!(combined.dispatch_count, layers * 2);
    }

    #[test]
    fn reuses_resident_projection_runners_without_repeat_weight_uploads_on_synthetic_model() {
        let dir = tempdir().expect("tempdir should be created");
        write_synthetic_model(dir.path());
        let artifact_dir = tempdir().expect("artifact dir should be created");
        write_packed_model_artifact(dir.path(), artifact_dir.path())
            .expect("packed artifact should be written");
        let packed =
            ReferenceModel::load_from_root_with_packed_artifact(dir.path(), artifact_dir.path())
                .expect("packed model should load");
        let _guard = lock_env();
        let prompt_ids = [2usize];
        let expected_dispatches = packed.config.num_hidden_layers * 2 + 1;
        let kv_rows = packed.config.num_key_value_heads * packed.config.head_dim;
        let expected_download_bytes = packed.config.num_hidden_layers
            * (packed.config.hidden_size + (2 * kv_rows) + (2 * packed.config.intermediate_size))
            * std::mem::size_of::<f32>();

        let first = packed
            .benchmark_packed_step_from_token_ids(&prompt_ids, true, true)
            .expect("first packed step should succeed");
        let second = packed
            .benchmark_packed_step_from_token_ids(&prompt_ids, true, true)
            .expect("second packed step should succeed");

        assert_eq!(first.dispatch_count, expected_dispatches);
        assert_eq!(second.dispatch_count, expected_dispatches);
        assert_eq!(first.download_bytes, expected_download_bytes);
        assert_eq!(second.download_bytes, expected_download_bytes);
        assert!(first.weight_upload_bytes > 0);
        assert!(first.weight_upload_duration > Duration::ZERO);
        assert!(first.compile_duration > Duration::ZERO);
        assert_eq!(second.weight_upload_bytes, 0);
        assert_eq!(second.weight_upload_duration, Duration::ZERO);
        assert_eq!(second.compile_duration, Duration::ZERO);
        assert_eq!(second.gpu_cache_hits, expected_dispatches);
        assert_eq!(second.pack_cache_hits, expected_dispatches);
    }

    use std::time::Duration;
}
