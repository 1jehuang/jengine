use crate::cpu::block::{YarnRope, build_yarn_rope, rope_cos_sin};
use crate::cpu::primitives::{argmax, swiglu};
use crate::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, run_packed_ternary_matvec_with_output,
};
use crate::model::config::{BonsaiModelConfig, GenerationConfig};
use crate::model::tokenizer::{PromptAnalysis, TokenizerDiagnostics, TokenizerRuntime};
use crate::runtime::assets::{AssetError, BonsaiAssetPaths};
use crate::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};
use crate::runtime::weights::{WeightError, WeightStore};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum ReferenceError {
    Asset(AssetError),
    Io(std::io::Error),
    Json(serde_json::Error),
    Tokenizer(crate::model::tokenizer::TokenizerLoadError),
    Weight(WeightError),
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
    pub kv_cache_bytes_per_token_fp16: usize,
    pub kv_cache_bytes_per_token_runtime_f32: usize,
    pub kv_cache_total_bytes_fp16: usize,
    pub kv_cache_total_bytes_runtime_f32: usize,
    pub estimated_runtime_working_set_bytes: usize,
}

impl MemoryReport {
    pub fn summarize(&self) -> String {
        format!(
            "prompt_tokens={} generated_tokens={} total_sequence_tokens={} model_fp16_bytes={} ({}) kv_per_token_fp16={} ({}) kv_per_token_runtime_f32={} ({}) kv_total_runtime_f32={} ({}) working_set_bytes={} ({})",
            self.prompt_tokens,
            self.generated_tokens,
            self.total_sequence_tokens,
            self.estimated_model_fp16_bytes,
            human_bytes(self.estimated_model_fp16_bytes),
            self.kv_cache_bytes_per_token_fp16,
            human_bytes(self.kv_cache_bytes_per_token_fp16),
            self.kv_cache_bytes_per_token_runtime_f32,
            human_bytes(self.kv_cache_bytes_per_token_runtime_f32),
            self.kv_cache_total_bytes_runtime_f32,
            human_bytes(self.kv_cache_total_bytes_runtime_f32),
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

#[derive(Debug, Clone)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
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

pub struct ReferenceModel {
    pub assets: BonsaiAssetPaths,
    pub config: BonsaiModelConfig,
    pub generation_config: GenerationConfig,
    pub tokenizer: Option<TokenizerRuntime>,
    pub weights: WeightStore,
    rope: YarnRope,
    cached_hybrid_qproj: RefCell<HashMap<usize, Rc<HybridQProjCache>>>,
    cached_hybrid_qproj_gpu: RefCell<HashMap<usize, Rc<RefCell<CachedGpuPackedMatvecRunner>>>>,
    cached_projection_packed: RefCell<HashMap<String, Rc<PackedProjectionCache>>>,
    cached_projection_gpu: RefCell<HashMap<String, Rc<RefCell<CachedGpuPackedMatvecRunner>>>>,
}

impl ReferenceModel {
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
        Ok(Self {
            assets,
            config,
            generation_config,
            tokenizer: None,
            weights,
            rope,
            cached_hybrid_qproj: RefCell::new(HashMap::new()),
            cached_hybrid_qproj_gpu: RefCell::new(HashMap::new()),
            cached_projection_packed: RefCell::new(HashMap::new()),
            cached_projection_gpu: RefCell::new(HashMap::new()),
        })
    }

    pub fn load_from_root(root: impl AsRef<Path>) -> Result<Self, ReferenceError> {
        let mut model = Self::load_core_from_root(root)?;
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
        let kv_cache_bytes_per_token_fp16 = self.config.kv_cache_bytes_per_token(1, 2);
        let kv_cache_bytes_per_token_runtime_f32 = self.config.kv_cache_bytes_per_token(1, 4);
        let kv_cache_total_bytes_fp16 = self.config.kv_cache_bytes(total_sequence_tokens, 1, 2);
        let kv_cache_total_bytes_runtime_f32 =
            self.config.kv_cache_bytes(total_sequence_tokens, 1, 4);
        MemoryReport {
            prompt_tokens,
            generated_tokens,
            total_sequence_tokens,
            estimated_model_fp16_bytes,
            kv_cache_bytes_per_token_fp16,
            kv_cache_bytes_per_token_runtime_f32,
            kv_cache_total_bytes_fp16,
            kv_cache_total_bytes_runtime_f32,
            estimated_runtime_working_set_bytes: estimated_model_fp16_bytes
                + kv_cache_total_bytes_runtime_f32,
        }
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
        let mut cache = (0..self.config.num_hidden_layers)
            .map(|_| LayerCache {
                keys: Vec::new(),
                values: Vec::new(),
            })
            .collect::<Vec<_>>();

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
        let hidden = self
            .weights
            .embedding_lookup("model.embed_tokens.weight", token_id)?;
        let input_norm_weight = self
            .weights
            .load_vector_f32(&format!("model.layers.{layer_idx}.input_layernorm.weight"))?;
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
        let code_words = packed
            .packed_codes
            .chunks(4)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, byte) in chunk.iter().enumerate() {
                    word |= (*byte as u32) << (i * 8);
                }
                word
            })
            .collect::<Vec<_>>();
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
        let (runner, duration) = CachedGpuPackedMatvecRunner::new(
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
        let values = self.weights.load_vector_f32(tensor_name)?;
        let pack_started = Instant::now();
        let (packed, _) = pack_ternary_g128(&values, vec![rows, cols], 1e-3).map_err(|error| {
            ReferenceError::Decode(format!("pack failed for {tensor_name}: {error}"))
        })?;
        let pack_duration = pack_started.elapsed();
        let code_words = packed
            .packed_codes
            .chunks(4)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, byte) in chunk.iter().enumerate() {
                    word |= (*byte as u32) << (i * 8);
                }
                word
            })
            .collect::<Vec<_>>();
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

    fn get_or_create_projection_gpu(
        &self,
        tensor_name: &str,
        packed: &PackedProjectionCache,
    ) -> Result<(Rc<RefCell<CachedGpuPackedMatvecRunner>>, Duration, bool), ReferenceError> {
        if let Some(cached) = self
            .cached_projection_gpu
            .borrow()
            .get(tensor_name)
            .cloned()
        {
            return Ok((cached, Duration::ZERO, true));
        }
        let (runner, duration) = CachedGpuPackedMatvecRunner::new(
            &packed.code_words,
            &packed.scales,
            packed.group_size,
            packed.rows,
            packed.cols,
        )
        .map_err(|error| {
            ReferenceError::Decode(format!("gpu packed init failed for {tensor_name}: {error}"))
        })?;
        let runner = Rc::new(RefCell::new(runner));
        self.cached_projection_gpu
            .borrow_mut()
            .insert(tensor_name.to_string(), runner.clone());
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

        let mut pack_duration = Duration::ZERO;
        let mut compile_duration = Duration::ZERO;
        let mut upload_duration = Duration::ZERO;
        let mut gpu_duration = Duration::ZERO;
        let mut download_duration = Duration::ZERO;

        let q_name = format!("{prefix}.self_attn.q_proj.weight");
        let k_name = format!("{prefix}.self_attn.k_proj.weight");
        let v_name = format!("{prefix}.self_attn.v_proj.weight");
        let o_name = format!("{prefix}.self_attn.o_proj.weight");

        let q_gpu = if use_q {
            let (packed, pack, _) = self.get_or_create_projection_cache(&q_name, q_rows, cols)?;
            let (runner, compile, _) = self.get_or_create_projection_gpu(&q_name, &packed)?;
            pack_duration += pack;
            compile_duration += compile;
            Some(runner)
        } else {
            None
        };
        let k_gpu = if use_k {
            let (packed, pack, _) = self.get_or_create_projection_cache(&k_name, kv_rows, cols)?;
            let (runner, compile, _) = self.get_or_create_projection_gpu(&k_name, &packed)?;
            pack_duration += pack;
            compile_duration += compile;
            Some(runner)
        } else {
            None
        };
        let v_gpu = if use_v {
            let (packed, pack, _) = self.get_or_create_projection_cache(&v_name, kv_rows, cols)?;
            let (runner, compile, _) = self.get_or_create_projection_gpu(&v_name, &packed)?;
            pack_duration += pack;
            compile_duration += compile;
            Some(runner)
        } else {
            None
        };

        let total_started = Instant::now();
        let mut q = if let Some(runner) = &q_gpu {
            let (out, report) = runner
                .borrow_mut()
                .run_with_output(&hidden_states, None)
                .map_err(|error| ReferenceError::Decode(format!("gpu q_proj failed: {error}")))?;
            upload_duration += report.upload_duration;
            gpu_duration += report.gpu_duration;
            download_duration += report.download_duration;
            out
        } else {
            self.weights.matvec_f16(&q_name, &hidden_states)?
        };
        let mut k = if let Some(runner) = &k_gpu {
            let (out, report) = runner
                .borrow_mut()
                .run_with_output(&hidden_states, None)
                .map_err(|error| ReferenceError::Decode(format!("gpu k_proj failed: {error}")))?;
            upload_duration += report.upload_duration;
            gpu_duration += report.gpu_duration;
            download_duration += report.download_duration;
            out
        } else {
            self.weights.matvec_f16(&k_name, &hidden_states)?
        };
        let v = if let Some(runner) = &v_gpu {
            let (out, report) = runner
                .borrow_mut()
                .run_with_output(&hidden_states, None)
                .map_err(|error| ReferenceError::Decode(format!("gpu v_proj failed: {error}")))?;
            upload_duration += report.upload_duration;
            gpu_duration += report.gpu_duration;
            download_duration += report.download_duration;
            out
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
            let (packed, pack, _) = self.get_or_create_projection_cache(&o_name, q_rows, q_rows)?;
            let (runner, compile, _) = self.get_or_create_projection_gpu(&o_name, &packed)?;
            pack_duration += pack;
            compile_duration += compile;
            let (out, report) = runner
                .borrow_mut()
                .run_with_output(&attn, None)
                .map_err(|error| ReferenceError::Decode(format!("gpu o_proj failed: {error}")))?;
            upload_duration += report.upload_duration;
            gpu_duration += report.gpu_duration;
            download_duration += report.download_duration;
            out
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
            pack_duration,
            compile_duration,
            upload_duration,
            gpu_duration,
            download_duration,
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

        let mut pack_duration = Duration::ZERO;
        let mut compile_duration = Duration::ZERO;
        let mut upload_duration = Duration::ZERO;
        let mut gpu_duration = Duration::ZERO;
        let mut download_duration = Duration::ZERO;

        let gate_gpu = if use_gate {
            let (packed, pack, _) =
                self.get_or_create_projection_cache(&gate_name, intermediate_rows, hidden_rows)?;
            let (runner, compile, _) = self.get_or_create_projection_gpu(&gate_name, &packed)?;
            pack_duration += pack;
            compile_duration += compile;
            Some(runner)
        } else {
            None
        };
        let up_gpu = if use_up {
            let (packed, pack, _) =
                self.get_or_create_projection_cache(&up_name, intermediate_rows, hidden_rows)?;
            let (runner, compile, _) = self.get_or_create_projection_gpu(&up_name, &packed)?;
            pack_duration += pack;
            compile_duration += compile;
            Some(runner)
        } else {
            None
        };

        let total_started = Instant::now();
        let gate = if let Some(runner) = &gate_gpu {
            let (out, report) = runner
                .borrow_mut()
                .run_with_output(&hidden_states, None)
                .map_err(|error| {
                    ReferenceError::Decode(format!("gpu gate_proj failed: {error}"))
                })?;
            upload_duration += report.upload_duration;
            gpu_duration += report.gpu_duration;
            download_duration += report.download_duration;
            out
        } else {
            self.weights.matvec_f16(&gate_name, &hidden_states)?
        };
        let up = if let Some(runner) = &up_gpu {
            let (out, report) = runner
                .borrow_mut()
                .run_with_output(&hidden_states, None)
                .map_err(|error| ReferenceError::Decode(format!("gpu up_proj failed: {error}")))?;
            upload_duration += report.upload_duration;
            gpu_duration += report.gpu_duration;
            download_duration += report.download_duration;
            out
        } else {
            self.weights.matvec_f16(&up_name, &hidden_states)?
        };
        let mlp = swiglu(&gate, &up);

        let mixed_down = if use_down {
            let (packed, pack, _) =
                self.get_or_create_projection_cache(&down_name, hidden_rows, intermediate_rows)?;
            let (runner, compile, _) = self.get_or_create_projection_gpu(&down_name, &packed)?;
            pack_duration += pack;
            compile_duration += compile;
            let (out, report) =
                runner
                    .borrow_mut()
                    .run_with_output(&mlp, None)
                    .map_err(|error| {
                        ReferenceError::Decode(format!("gpu down_proj failed: {error}"))
                    })?;
            upload_duration += report.upload_duration;
            gpu_duration += report.gpu_duration;
            download_duration += report.download_duration;
            out
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
            pack_duration,
            compile_duration,
            upload_duration,
            gpu_duration,
            download_duration,
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
        let mut cache = (0..self.config.num_hidden_layers)
            .map(|_| LayerCache {
                keys: Vec::new(),
                values: Vec::new(),
            })
            .collect::<Vec<_>>();
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
        let mut hidden = self
            .weights
            .embedding_lookup("model.embed_tokens.weight", token_id)?;
        metrics.embedding_duration += started_at.elapsed();

        let (cos, sin) = rope_cos_sin(&self.rope, &[position]);

        for (layer_idx, layer_cache) in cache
            .iter_mut()
            .enumerate()
            .take(self.config.num_hidden_layers)
        {
            let prefix = format!("model.layers.{layer_idx}");
            let residual = hidden.clone();

            let started_at = Instant::now();
            let input_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.input_layernorm.weight"))?;
            let mut hidden_states =
                weighted_rms_norm(&hidden, &input_norm_weight, self.config.rms_norm_eps as f32);
            metrics.norm_duration += started_at.elapsed();

            let started_at = Instant::now();
            let mut q = self
                .weights
                .matvec_f16(&format!("{prefix}.self_attn.q_proj.weight"), &hidden_states)?;
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

            let residual = hidden.clone();
            let started_at = Instant::now();
            let post_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;
            hidden_states =
                weighted_rms_norm(&hidden, &post_norm_weight, self.config.rms_norm_eps as f32);
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
            let residual = hidden.clone();

            let started_at = Instant::now();
            let input_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.input_layernorm.weight"))?;
            let mut hidden_states =
                weighted_rms_norm(&hidden, &input_norm_weight, self.config.rms_norm_eps as f32);
            metrics.norm_duration += started_at.elapsed();

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

            let residual = hidden.clone();
            let started_at = Instant::now();
            let post_norm_weight = self
                .weights
                .load_vector_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;
            hidden_states =
                weighted_rms_norm(&hidden, &post_norm_weight, self.config.rms_norm_eps as f32);
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
    use super::ReferenceModel;
    use half::f16;
    use safetensors::tensor::{Dtype, TensorView, serialize};
    use std::collections::BTreeMap;
    use std::fs;
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

    use std::time::Duration;
}
