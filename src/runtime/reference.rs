use crate::cpu::block::{YarnRope, build_yarn_rope, rope_cos_sin};
use crate::cpu::primitives::{argmax, swiglu, swiglu_into};
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
use crate::gpu::tail_block::{CachedGpuTailBlockRunner, GpuTailBlockReport};
use crate::gpu::vector_add::CachedGpuVectorAddRunner;
use crate::gpu::weighted_rms_norm::{CachedGpuWeightedRmsNormRunner, GpuWeightedRmsNormReport};
use crate::model::config::{BonsaiModelConfig, GenerationConfig};
use crate::model::tokenizer::{PromptAnalysis, TokenizerDiagnostics, TokenizerRuntime};
use crate::runtime::assets::BonsaiAssetPaths;
use crate::runtime::decode_plan::PackedDecodePlan;
use crate::runtime::decode_report::{MemoryReport, build_memory_report};
use crate::runtime::gpu_decode_env::{
    packed_enabled_label, packed_use_attention_full, packed_use_gpu_attention_block,
    packed_use_gpu_embedding, packed_use_gpu_final_norm, packed_use_gpu_first_session,
    packed_use_gpu_full_last_layer, packed_use_gpu_mlp_entry, packed_use_gpu_swiglu_block,
    packed_use_gpu_tail, packed_use_mlp_full,
};
use crate::runtime::gpu_decode_metrics::{
    AttentionProjectionMixMetrics, DecodeMetrics, HybridDecodeMetrics,
    HybridProjectionDecodeMetrics, MlpProjectionMixMetrics, PackedAttentionStageMetrics,
    PackedDecodeMetrics, PackedGpuSessionMetrics,
    PackedMlpStageMetrics, ProjectionComparison, account_projection_report,
    finish_packed_decode_metrics,
};
use crate::runtime::gpu_decode_model_state::{HybridQProjCache, LayerTensorNames};
use crate::runtime::gpu_decode_output::{DecodeResult, PackedDispatchTrace};
use crate::runtime::gpu_decode_projection_state::{
    PackedProjectionCache, PreparedProjectionRunner, ResidentGpuFinalNorm,
    ResidentGpuPackedActivation, ResidentGpuPackedActivationKeepalive, ResidentGpuSwigluCombined,
    ResidentGpuVectorAdd, ResidentPackedPairProjection, ResidentPackedProjection,
};
use crate::runtime::gpu_decode_scratch::PackedDecodeScratch;
use crate::runtime::gpu_decode_session_state::{
    LayerCache, PackedDecodeStepResult, allocate_layer_cache_vec,
};
use crate::runtime::gpu_decode_state::{
    GpuKvBinding, GpuTailResult, GpuTailStepReport, PackedResidentDecodeState,
    ResidentHiddenState,
};
use crate::runtime::packed_model::PackedModelStore;
use crate::runtime::reference_error::ReferenceError;
use crate::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};
use crate::runtime::weights::WeightStore;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

type CachedProjectionGpuRunner = Rc<RefCell<CachedGpuPackedMatvecRunner>>;
type CachedEmbeddingLookupGpuRunner = Rc<RefCell<CachedGpuEmbeddingLookupRunner>>;
type CachedQkRopeGpuRunner = Rc<RefCell<CachedGpuQkRopeRunner>>;
type CachedTailBlockGpuRunner = Rc<RefCell<CachedGpuTailBlockRunner>>;
type CachedFullLastLayerGpuRunner = Rc<RefCell<CachedGpuFullLastLayerRunner>>;
type CachedAttentionBlockGpuRunner = Rc<RefCell<CachedGpuAttentionBlockRunner>>;
type CachedMlpBlockGpuRunner = Rc<RefCell<CachedGpuMlpBlockRunner>>;
type CachedProjectionGpuCacheEntry = (CachedProjectionGpuRunner, Duration, Duration, bool);
type CachedEmbeddingLookupGpuCacheEntry = (CachedEmbeddingLookupGpuRunner, Duration, bool);
type CachedQkRopeGpuCacheEntry = (CachedQkRopeGpuRunner, Duration, bool);
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
type ResidentTensorQkvEntry = (
    GpuResidentBuffer,
    GpuResidentBuffer,
    GpuResidentBuffer,
    GpuResidentBuffer,
    GpuWeightedRmsNormReport,
    crate::gpu::packed_matvec::GpuPackedMatvecReport,
    Duration,
);
type ResidentHostQkvEntry = (
    GpuResidentBuffer,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    GpuWeightedRmsNormReport,
    crate::gpu::packed_matvec::GpuPackedMatvecReport,
    Duration,
);
type FirstLayerTensorQkvEntry = (
    Vec<f32>,
    GpuResidentBuffer,
    GpuResidentBuffer,
    GpuResidentBuffer,
    GpuResidentBuffer,
    GpuEmbeddingLookupReport,
    GpuWeightedRmsNormReport,
    crate::gpu::packed_matvec::GpuPackedMatvecReport,
    Duration,
);
type FirstLayerHostQkvEntry = (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    GpuEmbeddingLookupReport,
    GpuWeightedRmsNormReport,
    crate::gpu::packed_matvec::GpuPackedMatvecReport,
    Duration,
);
type GpuSwigluBlockExecution = (
    Option<Vec<f32>>,
    Option<ResidentGpuVectorAdd>,
    Duration,
    Duration,
    Duration,
    Instant,
);

pub(crate) struct GpuFirstRunnerCache<'a> {
    model: &'a ReferenceModel,
    kv_capacity_tokens: usize,
    shared_context: Option<Arc<SharedGpuPackedContext>>,
    embedding_lookup_runner: Option<CachedEmbeddingLookupGpuRunner>,
    input_norm_runner: Option<CachedWeightedRmsNormGpuRunner>,
    qk_rope_runner: Option<CachedQkRopeGpuRunner>,
    raw_f32_projection_runners: HashMap<String, CachedProjectionGpuRunner>,
    gpu_kv_caches: HashMap<usize, GpuKvCache>,
    attention_blocks: HashMap<usize, CachedAttentionBlockGpuRunner>,
    mlp_blocks: HashMap<usize, CachedMlpBlockGpuRunner>,
    full_last_layer_block: Option<CachedFullLastLayerGpuRunner>,
    tail_block: Option<CachedTailBlockGpuRunner>,
}

impl<'a> GpuFirstRunnerCache<'a> {
    pub(crate) fn new(model: &'a ReferenceModel, expected_tokens: usize) -> Self {
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

    fn gpu_kv_state(&self, layer_idx: usize) -> Option<GpuKvBinding> {
        self.gpu_kv_caches.get(&layer_idx).map(|cache| {
            let kv_len = cache.len_tokens() * cache.kv_width();
            GpuKvBinding {
                key_buffer: cache.key_buffer_handle(),
                key_len: kv_len,
                key_buffer_size: cache.key_buffer_size(),
                value_buffer: cache.value_buffer_handle(),
                value_len: kv_len,
                value_buffer_size: cache.value_buffer_size(),
            }
        })
    }

    pub(crate) fn has_qk_rope_runner(&self) -> bool {
        self.qk_rope_runner.is_some()
    }

    pub(crate) fn has_raw_f32_projection_runner(&self, key: &str) -> bool {
        self.raw_f32_projection_runners.contains_key(key)
    }

    pub(crate) fn gpu_kv_len_tokens(&self, layer_idx: usize) -> Option<usize> {
        self.gpu_kv_caches
            .get(&layer_idx)
            .map(|cache| cache.len_tokens())
    }

    pub(crate) fn gpu_kv_snapshot_lengths(
        &self,
        layer_idx: usize,
    ) -> Result<Option<(usize, usize)>, ReferenceError> {
        let Some(cache) = self.gpu_kv_caches.get(&layer_idx) else {
            return Ok(None);
        };
        let keys = cache.snapshot_keys().map_err(|error| {
            ReferenceError::Decode(format!(
                "gpu kv key snapshot failed for layer {layer_idx}: {error}"
            ))
        })?;
        let values = cache.snapshot_values().map_err(|error| {
            ReferenceError::Decode(format!(
                "gpu kv value snapshot failed for layer {layer_idx}: {error}"
            ))
        })?;
        Ok(Some((keys.len(), values.len())))
    }

    pub(crate) fn has_attention_block(&self, layer_idx: usize) -> bool {
        self.attention_blocks.contains_key(&layer_idx)
    }

    pub(crate) fn has_mlp_block(&self, layer_idx: usize) -> bool {
        self.mlp_blocks.contains_key(&layer_idx)
    }

    pub(crate) fn has_tail_block(&self) -> bool {
        self.tail_block.is_some()
    }

    pub(crate) fn has_full_last_layer_block(&self) -> bool {
        self.full_last_layer_block.is_some()
    }

    pub(crate) fn embedding_lookup_output(
        &mut self,
        token_id: usize,
    ) -> Result<Option<Vec<f32>>, ReferenceError> {
        let Some(runner) = self.embedding_lookup_runner.as_mut() else {
            return Ok(None);
        };
        let (output, _) = runner
            .borrow_mut()
            .run_with_output(token_id)
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu embedding lookup failed: {error}"))
            })?;
        Ok(Some(output))
    }

    fn ensure_embedding_lookup_runner(
        &mut self,
    ) -> Result<(Duration, CachedEmbeddingLookupGpuRunner), ReferenceError> {
        if let Some(runner) = self.embedding_lookup_runner.as_ref().cloned() {
            return Ok((Duration::ZERO, runner));
        }
        let (runner, compile_duration, _gpu_cache_hit) =
            self.model.get_or_create_embedding_lookup_gpu()?;
        self.embedding_lookup_runner = Some(runner.clone());
        Ok((compile_duration, runner))
    }

    fn ensure_input_norm_runner(
        &mut self,
    ) -> Result<(Duration, CachedWeightedRmsNormGpuRunner), ReferenceError> {
        if let Some(runner) = self.input_norm_runner.as_ref().cloned() {
            return Ok((Duration::ZERO, runner));
        }
        let (runner, compile_duration, _gpu_cache_hit) =
            self.model.get_or_create_input_norm_gpu()?;
        self.input_norm_runner = Some(runner.clone());
        Ok((compile_duration, runner))
    }

    fn ensure_qk_rope_runner(
        &mut self,
    ) -> Result<(Duration, CachedQkRopeGpuRunner), ReferenceError> {
        if let Some(runner) = self.qk_rope_runner.as_ref().cloned() {
            return Ok((Duration::ZERO, runner));
        }
        let (runner, compile_duration, _gpu_cache_hit) = self.model.get_or_create_qk_rope_gpu()?;
        self.qk_rope_runner = Some(runner.clone());
        Ok((compile_duration, runner))
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
            .borrow_mut()
            .run_resident(q, k, q_weight, k_weight, cos, sin)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (q_out, download_duration) = runner
            .borrow()
            .read_query_output()
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        report.download_duration = download_duration;
        let key_out = runner.borrow().key_resident_output();
        Ok(((q_out, key_out), report, compile_duration))
    }

    fn ensure_raw_f32_projection_runner(
        &mut self,
        cache_key: &str,
        packed: &PackedProjectionCache,
    ) -> Result<(Duration, CachedProjectionGpuRunner), ReferenceError> {
        if let Some(runner) = self.raw_f32_projection_runners.get(cache_key).cloned() {
            return Ok((Duration::ZERO, runner));
        }
        let (runner, compile_duration, _weight_upload_duration, _gpu_cache_hit) = self
            .model
            .get_or_create_projection_gpu_raw_f32(cache_key, packed)?;
        self.raw_f32_projection_runners
            .insert(cache_key.to_string(), runner.clone());
        Ok((compile_duration, runner))
    }

    fn ensure_raw_f32_single_projection_runner(
        &mut self,
        cache_key: &str,
        tensor_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(Duration, CachedProjectionGpuRunner), ReferenceError> {
        let (packed, _, _) = self
            .model
            .get_or_create_projection_cache(tensor_name, rows, cols)?;
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
    ) -> Result<
        (
            GpuResidentBuffer,
            GpuResidentBuffer,
            GpuQkRopeReport,
            Duration,
        ),
        ReferenceError,
    > {
        let (compile_duration, runner) = self.ensure_qk_rope_runner()?;
        let report = runner
            .borrow_mut()
            .run_resident_from_tensors(q, k, q_weight, k_weight, cos, sin)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let query_out = runner.borrow().query_resident_output();
        let key_out = runner.borrow().key_resident_output();
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
            .borrow_mut()
            .run_resident(token_id)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let hidden = vec![0.0; hidden_size];
        let embedding_output = embedding_runner.borrow().resident_output();

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
                .borrow_mut()
                .run_resident_from_tensor(&embedding_output, input_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            (
                norm_compile_duration,
                input_norm_runner.borrow().resident_output(),
                norm_report,
            )
        };

        let (qkv_compile_duration, qkv_runner) =
            self.ensure_raw_f32_projection_runner(&cache_key, &packed)?;
        let qkv_report = qkv_runner
            .borrow_mut()
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (combined, qkv_download_duration) = qkv_runner
            .borrow()
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
            .borrow_mut()
            .run_with_output(token_id)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let embedding_output = embedding_runner.borrow().resident_output();

        let (norm_compile_duration, norm_context, norm_report) = {
            let (norm_compile_duration, input_norm_runner) = self.ensure_input_norm_runner()?;
            let norm_report = input_norm_runner
                .borrow_mut()
                .run_resident_from_tensor(&embedding_output, input_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            (
                norm_compile_duration,
                input_norm_runner.borrow().resident_output(),
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
            .borrow_mut()
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let q_tensor = GpuResidentBuffer::new(
            q_runner.borrow().shared_context().clone(),
            q_runner.borrow().output_buffer_handle(),
            hidden_size,
            q_runner.borrow().output_buffer_size(),
        );

        let (k_compile_duration, k_runner) = self.ensure_raw_f32_single_projection_runner(
            &k_key,
            k_proj_name,
            kv_rows,
            hidden_size,
        )?;
        let k_report = k_runner
            .borrow_mut()
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let k_tensor = GpuResidentBuffer::new(
            k_runner.borrow().shared_context().clone(),
            k_runner.borrow().output_buffer_handle(),
            kv_rows,
            k_runner.borrow().output_buffer_size(),
        );

        let (v_compile_duration, v_runner) = self.ensure_raw_f32_single_projection_runner(
            &v_key,
            v_proj_name,
            kv_rows,
            hidden_size,
        )?;
        let v_report = v_runner
            .borrow_mut()
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let v_tensor = GpuResidentBuffer::new(
            v_runner.borrow().shared_context().clone(),
            v_runner.borrow().output_buffer_handle(),
            kv_rows,
            v_runner.borrow().output_buffer_size(),
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
            .borrow_mut()
            .run_with_output(token_id)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let embedding_context = embedding_runner.borrow().shared_context().clone();
        let embedding_buffer = embedding_runner.borrow().output_buffer_handle();
        let embedding_buffer_size = embedding_runner.borrow().output_buffer_size();
        let hidden_len = embedding_runner.borrow().hidden();

        let (norm_compile_duration, input_norm_runner) = self.ensure_input_norm_runner()?;
        let norm_report = input_norm_runner
            .borrow_mut()
            .run_resident_from_f32_buffer(
                &embedding_context,
                embedding_buffer,
                hidden_len,
                embedding_buffer_size,
                input_norm_weight,
            )
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (hidden_states, norm_download_duration) = input_norm_runner
            .borrow()
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
                let runner = self.mlp_blocks.get(&layer_idx).ok_or_else(|| {
                    ReferenceError::Decode(format!(
                        "gpu-first mlp block output missing for layer {layer_idx}"
                    ))
                })?;
                Ok(runner.borrow().resident_output())
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
                .get(&layer_idx)
                .ok_or_else(|| {
                    ReferenceError::Decode(format!(
                        "gpu-first mlp block output missing for layer {layer_idx}"
                    ))
                })?
                .borrow()
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
                .borrow_mut()
                .run_resident_from_tensor(&hidden_resident, input_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            (
                norm_compile_duration,
                input_norm_runner.borrow().resident_output(),
                norm_report,
            )
        };

        let (qkv_compile_duration, qkv_runner) =
            self.ensure_raw_f32_projection_runner(&cache_key, &packed)?;
        let qkv_report = qkv_runner
            .borrow_mut()
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (combined, qkv_download_duration) = qkv_runner
            .borrow()
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
                .borrow_mut()
                .run_resident_from_tensor(&hidden_resident, input_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            (
                norm_compile_duration,
                input_norm_runner.borrow().resident_output(),
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
            .borrow_mut()
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let q_tensor = GpuResidentBuffer::new(
            q_runner.borrow().shared_context().clone(),
            q_runner.borrow().output_buffer_handle(),
            hidden_size,
            q_runner.borrow().output_buffer_size(),
        );

        let (k_compile_duration, k_runner) = self.ensure_raw_f32_single_projection_runner(
            &k_key,
            k_proj_name,
            kv_rows,
            hidden_size,
        )?;
        let k_report = k_runner
            .borrow_mut()
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let k_tensor = GpuResidentBuffer::new(
            k_runner.borrow().shared_context().clone(),
            k_runner.borrow().output_buffer_handle(),
            kv_rows,
            k_runner.borrow().output_buffer_size(),
        );

        let (v_compile_duration, v_runner) = self.ensure_raw_f32_single_projection_runner(
            &v_key,
            v_proj_name,
            kv_rows,
            hidden_size,
        )?;
        let v_report = v_runner
            .borrow_mut()
            .run_resident_from_f32_tensor(&norm_context)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let v_tensor = GpuResidentBuffer::new(
            v_runner.borrow().shared_context().clone(),
            v_runner.borrow().output_buffer_handle(),
            kv_rows,
            v_runner.borrow().output_buffer_size(),
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
    ) -> Result<(Duration, CachedAttentionBlockGpuRunner), ReferenceError> {
        if let Some(runner) = self.attention_blocks.get(&layer_idx).cloned() {
            return Ok((Duration::ZERO, runner));
        }
        let (runner, compile_duration, _gpu_cache_hit) = self
            .model
            .get_or_create_attention_block_gpu(layer_idx, self.kv_capacity_tokens.max(1))?;
        self.attention_blocks.insert(layer_idx, runner.clone());
        Ok((compile_duration, runner))
    }

    fn ensure_mlp_block(
        &mut self,
        layer_idx: usize,
    ) -> Result<(Duration, CachedMlpBlockGpuRunner), ReferenceError> {
        if let Some(runner) = self.mlp_blocks.get(&layer_idx).cloned() {
            return Ok((Duration::ZERO, runner));
        }
        let (runner, compile_duration, _gpu_cache_hit) =
            self.model.get_or_create_mlp_block_gpu(layer_idx)?;
        self.mlp_blocks.insert(layer_idx, runner.clone());
        Ok((compile_duration, runner))
    }

    fn ensure_tail_block(
        &mut self,
    ) -> Result<(Duration, CachedTailBlockGpuRunner), ReferenceError> {
        if let Some(runner) = self.tail_block.as_ref().cloned() {
            return Ok((Duration::ZERO, runner));
        }
        let (runner, compile_duration, _gpu_cache_hit) =
            self.model.get_or_create_tail_block_gpu()?;
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

    fn prewarm_gpu_entry_runners(&mut self) -> Result<(), ReferenceError> {
        if packed_use_gpu_embedding()
            || packed_use_gpu_attention_block()
            || packed_use_gpu_tail()
            || packed_use_gpu_full_last_layer()
            || packed_use_gpu_swiglu_block()
        {
            let _ = self.ensure_embedding_lookup_runner()?;
            let _ = self.ensure_input_norm_runner()?;
        }
        Ok(())
    }

    fn prewarm_gpu_attention_runners(
        &mut self,
        use_attention_qkv: bool,
    ) -> Result<(), ReferenceError> {
        if use_attention_qkv
            && (packed_use_gpu_attention_block()
                || packed_use_gpu_tail()
                || packed_use_gpu_full_last_layer())
        {
            let _ = self.ensure_qk_rope_runner()?;
            for layer_idx in 0..self.model.config.num_hidden_layers {
                let _ = self.ensure_gpu_kv_cache(layer_idx)?;
                if packed_use_gpu_attention_block() {
                    for _seq_len in 1..=self.kv_capacity_tokens.max(1) {
                        let _ = self.ensure_attention_block(layer_idx)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn prewarm_gpu_mlp_runners(&mut self, use_mlp_gu: bool) -> Result<(), ReferenceError> {
        if use_mlp_gu
            && (packed_use_gpu_swiglu_block()
                || packed_use_gpu_tail()
                || packed_use_gpu_full_last_layer())
        {
            for layer_idx in 0..self.model.config.num_hidden_layers {
                let _ = self.ensure_mlp_block(layer_idx)?;
            }
        }
        Ok(())
    }

    fn prewarm_gpu_tail_runners(&mut self) -> Result<(), ReferenceError> {
        if packed_use_gpu_tail() {
            let _ = self.ensure_tail_block()?;
        }

        if packed_use_gpu_full_last_layer() {
            let last_layer_idx = self.model.config.num_hidden_layers - 1;
            for _seq_len in 1..=self.kv_capacity_tokens.max(1) {
                let _ = self.ensure_attention_block(last_layer_idx)?;
            }
            let _ = self.ensure_full_last_layer_block()?;
        }
        Ok(())
    }

    pub(crate) fn prewarm_decode_path(
        &mut self,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<(), ReferenceError> {
        let _ = self.get_or_create_shared_context()?;

        self.prewarm_gpu_entry_runners()?;
        self.prewarm_gpu_attention_runners(use_attention_qkv)?;
        self.prewarm_gpu_mlp_runners(use_mlp_gu)?;
        self.prewarm_gpu_tail_runners()?;

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
            self.ensure_attention_block(layer_idx)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            attention_block
                .borrow_mut()
                .run_with_resident_query_and_kv(query_resident, key_tensor, value_tensor, residual)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else if let Some(kv_state) = kv_state {
            attention_block
                .borrow_mut()
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    kv_state.key_buffer,
                    kv_state.key_len,
                    kv_state.key_buffer_size,
                    kv_state.value_buffer,
                    kv_state.value_len,
                    kv_state.value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .borrow_mut()
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let attention_output = attention_block.borrow().resident_output();

        let (mlp_compile_duration, mlp_block) = self.ensure_mlp_block(layer_idx)?;
        let mlp_report = mlp_block
            .borrow_mut()
            .run_from_resident_tensor(&attention_output, post_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (hidden, download_duration) = mlp_block
            .borrow()
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
            self.ensure_attention_block(layer_idx)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            if let Some(residual_resident) = residual_resident {
                attention_block
                    .borrow_mut()
                    .run_with_resident_query_kv_and_residual(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual_resident,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            } else {
                attention_block
                    .borrow_mut()
                    .run_with_resident_query_and_kv(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            }
        } else if let Some(kv_state) = kv_state {
            attention_block
                .borrow_mut()
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    kv_state.key_buffer,
                    kv_state.key_len,
                    kv_state.key_buffer_size,
                    kv_state.value_buffer,
                    kv_state.value_len,
                    kv_state.value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .borrow_mut()
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let attention_output = attention_block.borrow().resident_output();

        let (mlp_compile_duration, mlp_block) = self.ensure_mlp_block(layer_idx)?;
        let mlp_report = mlp_block
            .borrow_mut()
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
            self.ensure_attention_block(layer_idx)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            attention_block
                .borrow_mut()
                .run_with_resident_query_and_kv(query_resident, key_tensor, value_tensor, residual)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else if let Some(kv_state) = kv_state {
            attention_block
                .borrow_mut()
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    kv_state.key_buffer,
                    kv_state.key_len,
                    kv_state.key_buffer_size,
                    kv_state.value_buffer,
                    kv_state.value_len,
                    kv_state.value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .borrow_mut()
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let (hidden, download_duration) = attention_block
            .borrow()
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
            self.ensure_attention_block(layer_idx)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            attention_block
                .borrow_mut()
                .run_with_resident_query_and_kv(query_resident, key_tensor, value_tensor, residual)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else if let Some(kv_state) = kv_state {
            attention_block
                .borrow_mut()
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    kv_state.key_buffer,
                    kv_state.key_len,
                    kv_state.key_buffer_size,
                    kv_state.value_buffer,
                    kv_state.value_len,
                    kv_state.value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .borrow_mut()
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
            self.ensure_attention_block(layer_idx)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            if let Some(residual_resident) = residual_resident {
                attention_block
                    .borrow_mut()
                    .run_with_resident_query_kv_and_residual(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual_resident,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            } else {
                attention_block
                    .borrow_mut()
                    .run_with_resident_query_and_kv(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            }
        } else if let Some(kv_state) = kv_state {
            attention_block
                .borrow_mut()
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    kv_state.key_buffer,
                    kv_state.key_len,
                    kv_state.key_buffer_size,
                    kv_state.value_buffer,
                    kv_state.value_len,
                    kv_state.value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .borrow_mut()
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let attention_output = attention_block.borrow().resident_output();
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
            .borrow_mut()
            .run_with_host_residual(residual, post_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let (hidden, download_duration) = mlp_block
            .borrow()
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
            .borrow_mut()
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
            .borrow_mut()
            .run_with_host_residual(residual, post_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let mlp_output = mlp_block.borrow().resident_output();
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
            self.ensure_attention_block(layer_idx)?;
        let attention_report = if let Some(query_resident) = query_resident {
            let (key_tensor, value_tensor) = kv_tensors
                .as_ref()
                .expect("gpu kv tensors should exist when resident query is provided");
            if let Some(residual_resident) = residual_resident {
                attention_block
                    .borrow_mut()
                    .run_with_resident_query_kv_and_residual(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual_resident,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            } else {
                attention_block
                    .borrow_mut()
                    .run_with_resident_query_and_kv(
                        query_resident,
                        key_tensor,
                        value_tensor,
                        residual,
                    )
                    .map_err(|error| ReferenceError::Decode(error.to_string()))?
            }
        } else if let Some(kv_state) = kv_state {
            attention_block
                .borrow_mut()
                .run_with_resident_kv(
                    q.expect("host query should exist when resident query is not provided"),
                    kv_state.key_buffer,
                    kv_state.key_len,
                    kv_state.key_buffer_size,
                    kv_state.value_buffer,
                    kv_state.value_len,
                    kv_state.value_buffer_size,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        } else {
            attention_block
                .borrow_mut()
                .run_resident(
                    q.expect("host query should exist when resident query is not provided"),
                    keys,
                    values,
                    residual,
                )
                .map_err(|error| ReferenceError::Decode(error.to_string()))?
        };
        let attention_output = attention_block.borrow().resident_output();

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

    fn run_tail_from_resident_hidden(
        &mut self,
        source: &GpuResidentBuffer,
        final_norm_weight: &[f32],
        argmax_only: bool,
    ) -> Result<(GpuTailResult, GpuTailStepReport, Duration), ReferenceError> {
        let (compile_duration, tail_block) = self.ensure_tail_block()?;
        if argmax_only {
            let report = tail_block
                .borrow_mut()
                .run_argmax_from_resident_tensor(source, final_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            return Ok((
                GpuTailResult::NextToken(report.argmax_index),
                GpuTailStepReport {
                    final_norm_gpu_duration: report.final_norm_gpu_duration,
                    pack_gpu_duration: report.pack_gpu_duration,
                    logits_gpu_duration: report.logits_gpu_duration,
                    logits_download_duration: report.logits_download_duration,
                },
                compile_duration,
            ));
        }
        let (logits, report) = tail_block
            .borrow_mut()
            .run_logits_from_resident_tensor(source, final_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((
            GpuTailResult::Logits(logits),
            GpuTailStepReport {
                final_norm_gpu_duration: report.final_norm_gpu_duration,
                pack_gpu_duration: report.pack_gpu_duration,
                logits_gpu_duration: report.logits_gpu_duration,
                logits_download_duration: report.logits_download_duration,
            },
            compile_duration,
        ))
    }

    fn run_tail_from_host_hidden(
        &mut self,
        hidden_input: &[f32],
        final_norm_weight: &[f32],
        argmax_only: bool,
    ) -> Result<(GpuTailResult, GpuTailStepReport, Duration), ReferenceError> {
        let (compile_duration, tail_block) = self.ensure_tail_block()?;
        if argmax_only {
            let report = tail_block
                .borrow_mut()
                .run_argmax(hidden_input, final_norm_weight)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
            return Ok((
                GpuTailResult::NextToken(report.argmax_index),
                GpuTailStepReport {
                    final_norm_gpu_duration: report.final_norm_gpu_duration,
                    pack_gpu_duration: report.pack_gpu_duration,
                    logits_gpu_duration: report.logits_gpu_duration,
                    logits_download_duration: report.logits_download_duration,
                },
                compile_duration,
            ));
        }
        let (logits, report) = tail_block
            .borrow_mut()
            .run_logits(hidden_input, final_norm_weight)
            .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        Ok((
            GpuTailResult::Logits(logits),
            GpuTailStepReport {
                final_norm_gpu_duration: report.final_norm_gpu_duration,
                pack_gpu_duration: report.pack_gpu_duration,
                logits_gpu_duration: report.logits_gpu_duration,
                logits_download_duration: report.logits_download_duration,
            },
            compile_duration,
        ))
    }
}

pub(crate) struct PackedGpuSession<'a> {
    model: &'a ReferenceModel,
    pub(crate) metrics: PackedGpuSessionMetrics,
    pub(crate) dispatch_trace: Vec<PackedDispatchTrace>,
    scratch: PackedDecodeScratch,
}

struct PackedDecodeStageSelection {
    use_gpu_attention_block: bool,
    use_gpu_attention_mlp_block: bool,
    use_gpu_attention_only: bool,
    use_gpu_attention_tail: bool,
    use_gpu_full_last_layer_block: bool,
    use_gpu_full_last_layer: bool,
    use_gpu_mlp_entry: bool,
    use_gpu_mlp_only: bool,
    use_gpu_mlp_tail: bool,
    use_gpu_swiglu_block: bool,
    use_gpu_tail: bool,
}

enum PackedAttentionStageOutcome {
    Hidden(Vec<f32>),
    ResidentMlp,
    NextToken(usize),
}

enum PackedMlpStageOutcome {
    Continue,
    ResidentMlp,
    NextToken(usize),
}

enum PackedDecodeLayerStepOutcome {
    Continue,
    NextToken(usize),
}

impl<'a> PackedGpuSession<'a> {
    pub(crate) fn new(model: &'a ReferenceModel) -> Self {
        Self {
            model,
            metrics: PackedGpuSessionMetrics::default(),
            dispatch_trace: Vec::new(),
            scratch: PackedDecodeScratch::default(),
        }
    }

    fn take_qkv_scratch(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.scratch.take_qkv()
    }

    fn restore_qkv_scratch(&mut self, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>) {
        self.scratch.restore_qkv(q, k, v)
    }

    fn take_gate_up_scratch(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.scratch.take_gate_up()
    }

    fn restore_gate_up_scratch(&mut self, gate: Vec<f32>, up: Vec<f32>) {
        self.scratch.restore_gate_up(gate, up)
    }

    fn take_mlp_scratch(&mut self) -> Vec<f32> {
        self.scratch.take_mlp()
    }

    fn restore_mlp_scratch(&mut self, mlp: Vec<f32>) {
        self.scratch.restore_mlp(mlp)
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
        let prepared = self.prepare_projection_runner(tensor_name, rows, cols)?;
        let (argmax_index, report) = prepared
            .runner
            .borrow_mut()
            .run_with_argmax(input)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu projection argmax failed for {tensor_name}: {error}"
                ))
            })?;
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

    fn run_projection_argmax_from_packed_activation(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        activation: ResidentGpuPackedActivation,
    ) -> Result<usize, ReferenceError> {
        let prepared = self.prepare_projection_runner(tensor_name, rows, cols)?;
        let (argmax_index, report) = prepared
            .runner
            .borrow_mut()
            .run_with_argmax_from_packed_buffer(
                &activation.tensor.shared_context,
                activation.tensor.buffer,
                activation.tensor.len,
                activation.tensor.buffer_size,
            )
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu packed projection argmax failed for {tensor_name}: {error}"
                ))
            })?;

        self.metrics.compile_duration += activation.compile_duration;
        self.metrics.upload_duration += activation.upload_duration;
        self.metrics.gpu_duration += activation.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.gpu_cache_hits += usize::from(activation.gpu_cache_hit);
        self.metrics.activation_upload_bytes += activation.logical_len * std::mem::size_of::<f32>();
        self.metrics.upload_bytes += activation.logical_len * std::mem::size_of::<f32>();
        self.dispatch_trace
            .push(PackedDispatchTrace::resident_stage(
                self.dispatch_trace.len() + 1,
                "gpu_pack",
                "pack_f16_pairs",
                "pack_f16_pairs",
                cols.div_ceil(2),
                cols,
                activation.gpu_cache_hit,
                activation.compile_duration,
                Duration::ZERO,
                activation.upload_duration,
                activation.gpu_duration,
                0,
                activation.logical_len * std::mem::size_of::<f32>(),
            ));

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
        let tensor = GpuResidentBuffer::new(
            prepared.runner.borrow().shared_context().clone(),
            prepared.runner.borrow().output_buffer_handle(),
            rows,
            prepared.runner.borrow().output_buffer_size(),
        );
        Ok(ResidentPackedProjection {
            tensor_name: tensor_name.to_string(),
            operation: operation.to_string(),
            rows,
            cols,
            tensor,
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
        let len = runner.borrow().len();
        let tensor = runner.borrow().resident_output();
        Ok(ResidentGpuFinalNorm {
            tensor,
            runner,
            len,
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
            .get_or_create_pack_f16_pairs_gpu(final_norm.len)?;
        let report = runner
            .borrow_mut()
            .run_resident_from_f32_buffer(
                &final_norm.tensor.shared_context,
                final_norm.tensor.buffer,
                final_norm.len,
                final_norm.tensor.buffer_size,
            )
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu pack f16 pairs failed: {error}"))
            })?;
        self.metrics.compile_duration += final_norm.compile_duration;
        self.metrics.weight_upload_duration += final_norm.report.upload_duration;
        self.metrics.upload_duration += final_norm.report.upload_duration;
        self.metrics.gpu_duration += final_norm.report.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.weight_upload_bytes += final_norm.len * std::mem::size_of::<f32>();
        self.metrics.activation_upload_bytes += final_norm.len * std::mem::size_of::<f32>();
        self.metrics.upload_bytes += 2 * final_norm.len * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(final_norm.gpu_cache_hit);
        self.dispatch_trace
            .push(PackedDispatchTrace::resident_stage(
                self.dispatch_trace.len() + 1,
                "gpu_dense",
                "final_norm_gpu",
                "model.norm.weight",
                final_norm.len,
                final_norm.len,
                final_norm.gpu_cache_hit,
                final_norm.compile_duration,
                final_norm.report.upload_duration,
                Duration::ZERO,
                final_norm.report.gpu_duration,
                final_norm.len * std::mem::size_of::<f32>(),
                final_norm.len * std::mem::size_of::<f32>(),
            ));
        Ok(ResidentGpuPackedActivation {
            keepalive: ResidentGpuPackedActivationKeepalive::PackF16(runner.clone()),
            tensor: GpuResidentBuffer::new(
                runner.borrow().shared_context().clone(),
                runner.borrow().output_buffer_handle(),
                runner.borrow().packed_len(),
                runner.borrow().output_buffer_size(),
            ),
            logical_len: final_norm.len,
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
                &pair.tensor.shared_context,
                pair.tensor.buffer,
                pair.first_rows + pair.second_rows,
                pair.tensor.buffer_size,
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
        self.dispatch_trace
            .push(PackedDispatchTrace::resident_stage(
                self.dispatch_trace.len() + 1,
                "gpu_dense",
                "mlp_swiglu_gpu",
                "swiglu_combined",
                pair.first_rows,
                pair.first_rows + pair.second_rows,
                gpu_cache_hit,
                compile_duration,
                Duration::ZERO,
                report.upload_duration,
                report.gpu_duration,
                0,
                (pair.first_rows + pair.second_rows) * std::mem::size_of::<f32>(),
            ));
        let len = pair.first_rows;
        let tensor = GpuResidentBuffer::new(
            runner.borrow().shared_context().clone(),
            runner.borrow().output_buffer_handle(),
            len,
            runner.borrow().output_buffer_size(),
        );
        Ok(ResidentGpuSwigluCombined {
            tensor,
            len,
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
                &pair.tensor.shared_context,
                pair.tensor.buffer,
                pair.first_rows + pair.second_rows,
                pair.tensor.buffer_size,
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
        self.dispatch_trace
            .push(PackedDispatchTrace::resident_stage(
                self.dispatch_trace.len() + 1,
                "gpu_pack",
                "mlp_swiglu_pack_gpu",
                "swiglu_pack_f16_pairs",
                pair.first_rows.div_ceil(2),
                pair.first_rows + pair.second_rows,
                gpu_cache_hit,
                compile_duration,
                Duration::ZERO,
                report.upload_duration,
                report.gpu_duration,
                0,
                (pair.first_rows + pair.second_rows) * std::mem::size_of::<f32>(),
            ));
        Ok(ResidentGpuPackedActivation {
            keepalive: ResidentGpuPackedActivationKeepalive::SwigluPackF16(runner.clone()),
            tensor: GpuResidentBuffer::new(
                runner.borrow().shared_context().clone(),
                runner.borrow().output_buffer_handle(),
                runner.borrow().packed_len(),
                runner.borrow().output_buffer_size(),
            ),
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
                &swiglu.tensor.shared_context,
                swiglu.tensor.buffer,
                swiglu.len,
                swiglu.tensor.buffer_size,
            )
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu pack f16 pairs failed: {error}"))
            })?;
        self.metrics.compile_duration += compile_duration;
        self.metrics.upload_duration += report.upload_duration;
        self.metrics.gpu_duration += report.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.activation_upload_bytes += swiglu.len * std::mem::size_of::<f32>();
        self.metrics.upload_bytes += swiglu.len * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(gpu_cache_hit);
        self.dispatch_trace
            .push(PackedDispatchTrace::resident_stage(
                self.dispatch_trace.len() + 1,
                "gpu_pack",
                "pack_f16_pairs",
                "pack_f16_pairs",
                swiglu.len.div_ceil(2),
                swiglu.len,
                gpu_cache_hit,
                compile_duration,
                Duration::ZERO,
                report.upload_duration,
                report.gpu_duration,
                0,
                swiglu.len * std::mem::size_of::<f32>(),
            ));
        Ok(ResidentGpuPackedActivation {
            keepalive: ResidentGpuPackedActivationKeepalive::PackF16(runner.clone()),
            tensor: GpuResidentBuffer::new(
                runner.borrow().shared_context().clone(),
                runner.borrow().output_buffer_handle(),
                runner.borrow().packed_len(),
                runner.borrow().output_buffer_size(),
            ),
            logical_len: swiglu.len,
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
                &activation.tensor.shared_context,
                activation.tensor.buffer,
                activation.tensor.len,
                activation.tensor.buffer_size,
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
        self.dispatch_trace
            .push(PackedDispatchTrace::resident_stage(
                self.dispatch_trace.len() + 1,
                "gpu_pack",
                "pack_f16_pairs",
                "pack_f16_pairs",
                cols.div_ceil(2),
                cols,
                activation.gpu_cache_hit,
                activation.compile_duration,
                Duration::ZERO,
                activation.upload_duration,
                activation.gpu_duration,
                0,
                activation.logical_len * std::mem::size_of::<f32>(),
            ));
        let tensor = GpuResidentBuffer::new(
            prepared.runner.borrow().shared_context().clone(),
            prepared.runner.borrow().output_buffer_handle(),
            rows,
            prepared.runner.borrow().output_buffer_size(),
        );
        Ok(ResidentPackedProjection {
            tensor_name: tensor_name.to_string(),
            operation: "single_resident".to_string(),
            rows,
            cols,
            tensor,
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
                &left.tensor.shared_context,
                left.tensor.buffer,
                left.rows,
                left.tensor.buffer_size,
                right,
            )
            .map_err(|error| ReferenceError::Decode(format!("gpu vector add failed: {error}")))?;
        let len = runner.borrow().len();
        let tensor = runner.borrow().resident_output();
        Ok(ResidentGpuVectorAdd {
            tensor,
            runner,
            len,
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
        self.metrics.activation_upload_bytes += 2 * activation.len * std::mem::size_of::<f32>();
        self.metrics.upload_bytes += 2 * activation.len * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(activation.gpu_cache_hit);
        self.dispatch_trace
            .push(PackedDispatchTrace::resident_stage(
                self.dispatch_trace.len() + 1,
                "gpu_dense",
                "vector_add_gpu",
                "vector_add",
                activation.len,
                activation.len,
                activation.gpu_cache_hit,
                activation.compile_duration,
                Duration::ZERO,
                activation.report.upload_duration,
                activation.report.gpu_duration,
                0,
                2 * activation.len * std::mem::size_of::<f32>(),
            ));

        let (runner, compile_duration, gpu_cache_hit) =
            self.model.get_or_create_final_norm_gpu()?;
        let report = runner
            .borrow_mut()
            .run_resident_from_f32_buffer(
                &activation.tensor.shared_context,
                activation.tensor.buffer,
                activation.len,
                activation.tensor.buffer_size,
                weight,
            )
            .map_err(|error| ReferenceError::Decode(format!("gpu final norm failed: {error}")))?;
        let len = runner.borrow().len();
        let tensor = runner.borrow().resident_output();
        Ok(ResidentGpuFinalNorm {
            tensor,
            runner,
            len,
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
                &activation.tensor.shared_context,
                activation.tensor.buffer,
                activation.tensor.len,
                activation.tensor.buffer_size,
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
        self.dispatch_trace.push(PackedDispatchTrace::gpu_packed(
            self.dispatch_trace.len() + 1,
            tensor_name,
            operation,
            rows,
            cols,
            pack_cache_hit,
            gpu_cache_hit,
            compile_duration,
            weight_upload_duration,
            activation_upload_duration,
            gpu_duration,
            download_duration,
            weight_upload_bytes,
            activation_upload_bytes,
            download_bytes,
        ));
    }

    fn push_dense_stage_trace(&mut self, stage: &str, tensor_name: &str, cpu_duration: Duration) {
        self.dispatch_trace.push(PackedDispatchTrace::dense_stage(
            self.dispatch_trace.len() + 1,
            stage,
            tensor_name,
            cpu_duration,
        ));
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
        let packed_weight_bytes = packed.code_words.len() * std::mem::size_of::<u32>()
            + packed.scales.len() * std::mem::size_of::<f32>();
        account_projection_report(
            &mut self.metrics,
            packed_weight_bytes,
            cols,
            weight_upload_duration,
            gpu_cache_hit,
            report,
            download_bytes,
        );
    }

    fn run_projection_add_residual(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        input: &[f32],
        residual: &[f32],
    ) -> Result<Vec<f32>, ReferenceError> {
        if residual.len() != rows {
            return Err(ReferenceError::Decode(format!(
                "residual len {} must match rows {} for {tensor_name}",
                residual.len(),
                rows
            )));
        }
        let prepared = self.prepare_projection_runner(tensor_name, rows, cols)?;
        let report = prepared
            .runner
            .borrow_mut()
            .run_resident(input)
            .map_err(|error| {
                ReferenceError::Decode(format!("gpu projection failed for {tensor_name}: {error}"))
            })?;
        let download_started = Instant::now();
        let summed = {
            let runner_ref = prepared.runner.borrow();
            let output = runner_ref.output_slice().map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu projection output borrow failed for {tensor_name}: {error}"
                ))
            })?;
            residual
                .iter()
                .zip(output.iter())
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>()
        };
        let report = crate::gpu::packed_matvec::GpuPackedMatvecReport {
            download_duration: download_started.elapsed(),
            ..report
        };
        let weight_upload_bytes = usize::from(!prepared.gpu_cache_hit)
            * (prepared.packed.code_words.len() * std::mem::size_of::<u32>()
                + prepared.packed.scales.len() * std::mem::size_of::<f32>());
        let activation_upload_bytes = cols.div_ceil(2) * std::mem::size_of::<u32>();
        let download_bytes = rows * std::mem::size_of::<f32>();
        self.account_projection_report(
            &prepared.packed,
            cols,
            prepared.weight_upload_duration,
            prepared.gpu_cache_hit,
            &report,
            download_bytes,
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
            download_bytes,
        );
        Ok(summed)
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

        let report = runner.borrow_mut().run_resident(input).map_err(|error| {
            ReferenceError::Decode(format!(
                "gpu projection failed for {first_name}+{second_name}: {error}"
            ))
        })?;
        let ((first, second), download_duration) = runner
            .borrow()
            .read_output_pair(first_rows, second_rows)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu projection output download failed for {first_name}+{second_name}: {error}"
                ))
            })?;
        let report = crate::gpu::packed_matvec::GpuPackedMatvecReport {
            download_duration,
            ..report
        };
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
        Ok((first, second))
    }

    fn run_projection_pair_into(
        &mut self,
        first_name: &str,
        first_rows: usize,
        second_name: &str,
        second_rows: usize,
        cols: usize,
        input: &[f32],
        first: &mut Vec<f32>,
        second: &mut Vec<f32>,
    ) -> Result<(), ReferenceError> {
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
        let download_duration = runner
            .borrow()
            .read_output_pair_into(first_rows, second_rows, first, second)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu projection output download failed for {first_name}+{second_name}: {error}"
                ))
            })?;
        let report = crate::gpu::packed_matvec::GpuPackedMatvecReport {
            download_duration,
            ..report
        };
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
        Ok(())
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
        let tensor = GpuResidentBuffer::new(
            runner.borrow().shared_context().clone(),
            runner.borrow().output_buffer_handle(),
            first_rows + second_rows,
            runner.borrow().output_buffer_size(),
        );
        Ok(ResidentPackedPairProjection {
            tensor_name: pair_key,
            first_rows,
            second_rows,
            cols,
            tensor,
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
        let len = final_norm.len;
        self.metrics.compile_duration += final_norm.compile_duration;
        self.metrics.upload_duration += final_norm.report.upload_duration;
        self.metrics.gpu_duration += final_norm.report.gpu_duration;
        self.metrics.dispatch_count += 1;
        self.metrics.activation_upload_bytes += len * std::mem::size_of::<f32>();
        self.metrics.weight_upload_bytes += len * std::mem::size_of::<f32>();
        self.metrics.upload_bytes += 2 * len * std::mem::size_of::<f32>();
        self.metrics.gpu_cache_hits += usize::from(final_norm.gpu_cache_hit);
        self.dispatch_trace
            .push(PackedDispatchTrace::resident_stage(
                self.dispatch_trace.len() + 1,
                "gpu_dense",
                "post_attention_norm_gpu",
                "post_attention_layernorm.weight",
                len,
                len,
                final_norm.gpu_cache_hit,
                final_norm.compile_duration,
                final_norm.report.upload_duration,
                Duration::ZERO,
                final_norm.report.gpu_duration,
                len * std::mem::size_of::<f32>(),
                len * std::mem::size_of::<f32>(),
            ));

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
                &final_norm.tensor.shared_context,
                final_norm.tensor.buffer,
                len,
                final_norm.tensor.buffer_size,
            )
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu raw-f32 pair projection failed for {first_name}+{second_name}: {error}"
                ))
            })?;
        let tensor = GpuResidentBuffer::new(
            runner.borrow().shared_context().clone(),
            runner.borrow().output_buffer_handle(),
            first_rows + second_rows,
            runner.borrow().output_buffer_size(),
        );
        Ok(ResidentPackedPairProjection {
            tensor_name: pair_key,
            first_rows,
            second_rows,
            cols,
            tensor,
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

        let report = runner.borrow_mut().run_resident(input).map_err(|error| {
            ReferenceError::Decode(format!(
                "gpu projection failed for {first_name}+{second_name}+{third_name}: {error}"
            ))
        })?;
        let ((first, second, third), download_duration) = runner
            .borrow()
            .read_output_triplet(first_rows, second_rows, third_rows)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu projection output download failed for {first_name}+{second_name}+{third_name}: {error}"
                ))
            })?;
        let report = crate::gpu::packed_matvec::GpuPackedMatvecReport {
            download_duration,
            ..report
        };
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
        Ok((first, second, third))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_projection_triplet_into(
        &mut self,
        first_name: &str,
        first_rows: usize,
        second_name: &str,
        second_rows: usize,
        third_name: &str,
        third_rows: usize,
        cols: usize,
        input: &[f32],
        first: &mut Vec<f32>,
        second: &mut Vec<f32>,
        third: &mut Vec<f32>,
    ) -> Result<(), ReferenceError> {
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

        let report = runner.borrow_mut().run_resident(input).map_err(|error| {
            ReferenceError::Decode(format!(
                "gpu projection failed for {first_name}+{second_name}+{third_name}: {error}"
            ))
        })?;
        let download_duration = runner
            .borrow()
            .read_output_triplet_into(first_rows, second_rows, third_rows, first, second, third)
            .map_err(|error| {
                ReferenceError::Decode(format!(
                    "gpu projection output download failed for {first_name}+{second_name}+{third_name}: {error}"
                ))
            })?;
        let report = crate::gpu::packed_matvec::GpuPackedMatvecReport {
            download_duration,
            ..report
        };
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
        Ok(())
    }
}

pub struct ReferenceModel {
    pub assets: BonsaiAssetPaths,
    pub config: BonsaiModelConfig,
    pub generation_config: GenerationConfig,
    pub tokenizer: Option<TokenizerRuntime>,
    pub weights: WeightStore,
    pub(crate) packed_model: Option<PackedModelStore>,
    rope: YarnRope,
    layer_tensors: Vec<LayerTensorNames>,
    cached_hybrid_qproj: RefCell<HashMap<usize, Rc<HybridQProjCache>>>,
    cached_hybrid_qproj_gpu: RefCell<HashMap<usize, Rc<RefCell<CachedGpuPackedMatvecRunner>>>>,
    cached_projection_packed: RefCell<HashMap<String, Rc<PackedProjectionCache>>>,
    cached_projection_gpu: RefCell<HashMap<String, CachedProjectionGpuRunner>>,
    cached_projection_gpu_raw_f32: RefCell<HashMap<String, CachedProjectionGpuRunner>>,
    cached_embedding_lookup_gpu: RefCell<Option<CachedEmbeddingLookupGpuRunner>>,
    cached_input_norm_gpu: RefCell<Option<CachedWeightedRmsNormGpuRunner>>,
    cached_qk_rope_gpu: RefCell<Option<CachedQkRopeGpuRunner>>,
    cached_final_norm_gpu: RefCell<Option<CachedWeightedRmsNormGpuRunner>>,
    cached_pack_f16_pairs_gpu: RefCell<HashMap<usize, CachedPackF16PairsGpuRunner>>,
    cached_vector_add_gpu: RefCell<HashMap<usize, CachedVectorAddGpuRunner>>,
    cached_swiglu_combined_gpu: RefCell<Option<CachedSwigluCombinedGpuRunner>>,
    cached_swiglu_pack_f16_pairs_gpu: RefCell<Option<CachedSwigluPackF16PairsGpuRunner>>,
    cached_attention_block_gpu: RefCell<HashMap<(usize, usize), CachedAttentionBlockGpuRunner>>,
    cached_mlp_block_gpu: RefCell<HashMap<usize, CachedMlpBlockGpuRunner>>,
    cached_tail_block_gpu: RefCell<Option<CachedTailBlockGpuRunner>>,
    cached_full_last_layer_block_gpu: RefCell<Option<CachedFullLastLayerGpuRunner>>,
    packed_gpu_context: RefCell<Option<Arc<SharedGpuPackedContext>>>,
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
            cached_embedding_lookup_gpu: RefCell::new(None),
            cached_input_norm_gpu: RefCell::new(None),
            cached_qk_rope_gpu: RefCell::new(None),
            cached_final_norm_gpu: RefCell::new(None),
            cached_pack_f16_pairs_gpu: RefCell::new(HashMap::new()),
            cached_vector_add_gpu: RefCell::new(HashMap::new()),
            cached_swiglu_combined_gpu: RefCell::new(None),
            cached_swiglu_pack_f16_pairs_gpu: RefCell::new(None),
            cached_attention_block_gpu: RefCell::new(HashMap::new()),
            cached_mlp_block_gpu: RefCell::new(HashMap::new()),
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
        build_memory_report(
            prompt_tokens,
            generated_tokens,
            estimated_model_fp16_bytes,
            source_weight_bytes,
            kv_cache_bytes_per_token_fp16,
            kv_cache_bytes_per_token_runtime_f32,
            kv_cache_total_bytes_fp16,
            kv_cache_total_bytes_runtime_f32,
            packed_cache_bytes,
            gpu_cache_buffer_bytes,
            activation_working_bytes,
            staging_bytes,
        )
    }

    pub(crate) fn prewarm_layer_projection_caches(
        &self,
        plan: &PackedDecodePlan,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        use_attention_full: bool,
        use_mlp_full: bool,
        kv_rows: usize,
    ) -> Result<(), ReferenceError> {
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
                if plan.gpu_mlp_entry || plan.gpu_full_last_layer {
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
                    if plan.gpu_swiglu_block {
                        let _ = self.get_or_create_swiglu_pack_f16_pairs_gpu()?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn prewarm_tail_support_caches(
        &self,
        plan: &PackedDecodePlan,
        use_mlp_gu: bool,
    ) -> Result<(), ReferenceError> {
        let _ = self.load_vector_f32_resolved("model.norm.weight")?;
        if use_mlp_gu {
            if plan.gpu_mlp_entry || plan.gpu_full_last_layer {
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
            if plan.gpu_swiglu_block || plan.gpu_full_last_layer {
                let _ = self.get_or_create_swiglu_pack_f16_pairs_gpu()?;
                let _ = self.get_or_create_pack_f16_pairs_gpu(self.config.intermediate_size)?;
            }
        }
        if plan.gpu_final_norm || plan.gpu_full_last_layer {
            let _ = self.get_or_create_final_norm_gpu()?;
            let _ = self.get_or_create_pack_f16_pairs_gpu(self.config.hidden_size)?;
            if plan.gpu_tail || plan.gpu_full_last_layer {
                let _ = self.get_or_create_vector_add_gpu(self.config.hidden_size)?;
            }
        }
        let (_logits_packed, _, _) = self.get_or_create_projection_cache(
            "model.embed_tokens.weight",
            self.config.vocab_size,
            self.config.hidden_size,
        )?;
        Ok(())
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
        let mut cache = allocate_layer_cache_vec(
            self.config.num_hidden_layers,
            prompt_ids.len() + max_new_tokens,
            self.config.num_key_value_heads * self.config.head_dim,
            true,
        );

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

    #[allow(clippy::too_many_arguments)]
    fn try_resident_hidden_tensor_qkv_entry(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        resident_hidden_state: Option<ResidentHiddenState>,
        layer_idx: usize,
        use_attention_qkv: bool,
        use_gpu_attention_block: bool,
        input_norm_weight: &[f32],
        layer_tensors: &LayerTensorNames,
    ) -> Result<Option<ResidentTensorQkvEntry>, ReferenceError> {
        if layer_idx > 0
            && use_attention_qkv
            && resident_hidden_state.is_some()
            && use_gpu_attention_block
        {
            return Ok(Some(
                gpu_first_session.run_layer_input_norm_qkv_tensors_from_hidden_resident(
                    resident_hidden_state.expect("resident hidden state should exist"),
                    layer_idx,
                    input_norm_weight,
                    &layer_tensors.q_proj_weight,
                    &layer_tensors.k_proj_weight,
                    &layer_tensors.v_proj_weight,
                    self.config.num_key_value_heads * self.config.head_dim,
                )?,
            ));
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_resident_hidden_entry_qkv(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        resident_hidden_state: Option<ResidentHiddenState>,
        layer_idx: usize,
        use_attention_qkv: bool,
        use_gpu_attention_block: bool,
        input_norm_weight: &[f32],
        layer_tensors: &LayerTensorNames,
    ) -> Result<Option<ResidentHostQkvEntry>, ReferenceError> {
        if layer_idx > 0
            && use_attention_qkv
            && resident_hidden_state.is_some()
            && !use_gpu_attention_block
        {
            return Ok(Some(
                gpu_first_session.run_layer_input_norm_qkv_from_hidden_resident(
                    resident_hidden_state.expect("resident hidden state should exist"),
                    layer_idx,
                    input_norm_weight,
                    &layer_tensors.q_proj_weight,
                    &layer_tensors.k_proj_weight,
                    &layer_tensors.v_proj_weight,
                    self.config.num_key_value_heads * self.config.head_dim,
                )?,
            ));
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_first_layer_gpu_entry_tensor_qkv(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        resident_hidden_tensor_qkv: &Option<ResidentTensorQkvEntry>,
        resident_hidden_entry_qkv: &Option<ResidentHostQkvEntry>,
        layer_idx: usize,
        token_id: usize,
        use_attention_qkv: bool,
        use_gpu_attention_block: bool,
        input_norm_weight: &[f32],
        layer_tensors: &LayerTensorNames,
    ) -> Result<Option<FirstLayerTensorQkvEntry>, ReferenceError> {
        if resident_hidden_tensor_qkv.is_none()
            && resident_hidden_entry_qkv.is_none()
            && layer_idx == 0
            && packed_use_gpu_embedding()
            && packed_use_gpu_first_session()
            && use_attention_qkv
            && use_gpu_attention_block
        {
            return Ok(Some(
                gpu_first_session.run_first_layer_embedding_norm_qkv_tensors(
                    layer_idx,
                    token_id,
                    input_norm_weight,
                    &layer_tensors.q_proj_weight,
                    &layer_tensors.k_proj_weight,
                    &layer_tensors.v_proj_weight,
                    self.config.num_key_value_heads * self.config.head_dim,
                )?,
            ));
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_first_layer_gpu_entry_qkv(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        resident_hidden_tensor_qkv: &Option<ResidentTensorQkvEntry>,
        resident_hidden_entry_qkv: &Option<ResidentHostQkvEntry>,
        first_layer_gpu_entry_tensor_qkv: &Option<FirstLayerTensorQkvEntry>,
        layer_idx: usize,
        token_id: usize,
        use_attention_qkv: bool,
        input_norm_weight: &[f32],
        layer_tensors: &LayerTensorNames,
    ) -> Result<Option<FirstLayerHostQkvEntry>, ReferenceError> {
        if first_layer_gpu_entry_tensor_qkv.is_none()
            && resident_hidden_tensor_qkv.is_none()
            && resident_hidden_entry_qkv.is_none()
            && layer_idx == 0
            && packed_use_gpu_embedding()
            && packed_use_gpu_first_session()
            && use_attention_qkv
        {
            return Ok(Some(
                gpu_first_session.run_first_layer_embedding_norm_qkv_to_host(
                    layer_idx,
                    token_id,
                    input_norm_weight,
                    &layer_tensors.q_proj_weight,
                    &layer_tensors.k_proj_weight,
                    &layer_tensors.v_proj_weight,
                    self.config.num_key_value_heads * self.config.head_dim,
                )?,
            ));
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_resident_tensor_qkv_entry(
        &self,
        entry: &ResidentTensorQkvEntry,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Vec<f32> {
        let (_hidden_resident, _q, _k, _v, norm_report, qkv_report, compile_duration) = entry;
        session.metrics.compile_duration += *compile_duration;
        session.metrics.activation_upload_duration +=
            norm_report.upload_duration + qkv_report.upload_duration;
        session.metrics.upload_duration += norm_report.upload_duration + qkv_report.upload_duration;
        session.metrics.gpu_duration += norm_report.gpu_duration + qkv_report.gpu_duration;
        session.metrics.download_duration +=
            norm_report.download_duration + qkv_report.download_duration;
        session.metrics.activation_upload_bytes +=
            self.config.hidden_size * std::mem::size_of::<f32>();
        session.metrics.upload_bytes += self.config.hidden_size * std::mem::size_of::<f32>();
        metrics.norm_duration +=
            norm_report.upload_duration + norm_report.gpu_duration + norm_report.download_duration;
        metrics.qkv_duration +=
            qkv_report.upload_duration + qkv_report.gpu_duration + qkv_report.download_duration;
        vec![0.0; self.config.hidden_size]
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_resident_host_qkv_entry(
        &self,
        entry: &ResidentHostQkvEntry,
        hidden: &mut Vec<f32>,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
        resident_hidden_state: ResidentHiddenState,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
    ) -> Result<Vec<f32>, ReferenceError> {
        let (_hidden_resident, _q, _k, _v, norm_report, qkv_report, compile_duration) = entry;
        session.metrics.compile_duration += *compile_duration;
        session.metrics.activation_upload_duration +=
            norm_report.upload_duration + qkv_report.upload_duration;
        session.metrics.upload_duration += norm_report.upload_duration + qkv_report.upload_duration;
        session.metrics.gpu_duration += norm_report.gpu_duration + qkv_report.gpu_duration;
        session.metrics.download_duration +=
            norm_report.download_duration + qkv_report.download_duration;
        session.metrics.activation_upload_bytes +=
            self.config.hidden_size * std::mem::size_of::<f32>();
        session.metrics.upload_bytes += self.config.hidden_size * std::mem::size_of::<f32>();
        session.metrics.download_bytes += (self.config.hidden_size
            + 2 * (self.config.num_key_value_heads * self.config.head_dim))
            * std::mem::size_of::<f32>();
        metrics.norm_duration +=
            norm_report.upload_duration + norm_report.gpu_duration + norm_report.download_duration;
        metrics.qkv_duration +=
            qkv_report.upload_duration + qkv_report.gpu_duration + qkv_report.download_duration;
        let (materialized_hidden, hidden_download_duration) =
            gpu_first_session.read_hidden_output(resident_hidden_state)?;
        *hidden = materialized_hidden;
        session.metrics.download_duration += hidden_download_duration;
        session.metrics.download_bytes += self.config.hidden_size * std::mem::size_of::<f32>();
        Ok(vec![0.0; self.config.hidden_size])
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_first_layer_tensor_qkv_entry(
        &self,
        entry: &FirstLayerTensorQkvEntry,
        hidden: &mut Vec<f32>,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Vec<f32> {
        let (
            gpu_hidden,
            _hidden_resident,
            _q,
            _k,
            _v,
            embedding_report,
            norm_report,
            qkv_report,
            compile_duration,
        ) = entry;
        *hidden = gpu_hidden.clone();
        session.metrics.compile_duration += *compile_duration;
        session.metrics.activation_upload_duration += embedding_report.upload_duration
            + norm_report.upload_duration
            + qkv_report.upload_duration;
        session.metrics.upload_duration += embedding_report.upload_duration
            + norm_report.upload_duration
            + qkv_report.upload_duration;
        session.metrics.gpu_duration +=
            embedding_report.gpu_duration + norm_report.gpu_duration + qkv_report.gpu_duration;
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
        metrics.norm_duration +=
            norm_report.upload_duration + norm_report.gpu_duration + norm_report.download_duration;
        metrics.qkv_duration +=
            qkv_report.upload_duration + qkv_report.gpu_duration + qkv_report.download_duration;
        vec![0.0; self.config.hidden_size]
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_first_layer_host_qkv_entry(
        &self,
        entry: &FirstLayerHostQkvEntry,
        hidden: &mut Vec<f32>,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Vec<f32> {
        let (gpu_hidden, _q, _k, _v, embedding_report, norm_report, qkv_report, compile_duration) =
            entry;
        *hidden = gpu_hidden.clone();
        session.metrics.compile_duration += *compile_duration;
        session.metrics.activation_upload_duration += embedding_report.upload_duration
            + norm_report.upload_duration
            + qkv_report.upload_duration;
        session.metrics.upload_duration += embedding_report.upload_duration
            + norm_report.upload_duration
            + qkv_report.upload_duration;
        session.metrics.gpu_duration +=
            embedding_report.gpu_duration + norm_report.gpu_duration + qkv_report.gpu_duration;
        session.metrics.download_duration += embedding_report.download_duration
            + norm_report.download_duration
            + qkv_report.download_duration;
        session.metrics.activation_upload_bytes +=
            std::mem::size_of::<u32>() + self.config.hidden_size * std::mem::size_of::<f32>();
        session.metrics.upload_bytes +=
            std::mem::size_of::<u32>() + self.config.hidden_size * std::mem::size_of::<f32>();
        session.metrics.download_bytes += self.config.hidden_size * std::mem::size_of::<f32>() * 2
            + (self.config.hidden_size
                + 2 * (self.config.num_key_value_heads * self.config.head_dim))
                * std::mem::size_of::<f32>();
        metrics.embedding_duration += embedding_report.upload_duration
            + embedding_report.gpu_duration
            + embedding_report.download_duration;
        metrics.norm_duration +=
            norm_report.upload_duration + norm_report.gpu_duration + norm_report.download_duration;
        metrics.qkv_duration +=
            qkv_report.upload_duration + qkv_report.gpu_duration + qkv_report.download_duration;
        vec![0.0; self.config.hidden_size]
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_first_layer_embedding_norm_entry(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        token_id: usize,
        input_norm_weight: &[f32],
        hidden: &mut Vec<f32>,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<Vec<f32>, ReferenceError> {
        let started_at = Instant::now();
        let (gpu_hidden, gpu_hidden_states, embedding_report, norm_report, compile_duration) =
            gpu_first_session.run_embedding_and_input_norm_to_host(token_id, input_norm_weight)?;
        *hidden = gpu_hidden;
        session.metrics.compile_duration += compile_duration;
        session.metrics.activation_upload_duration +=
            embedding_report.upload_duration + norm_report.upload_duration;
        session.metrics.upload_duration +=
            embedding_report.upload_duration + norm_report.upload_duration;
        session.metrics.gpu_duration += embedding_report.gpu_duration + norm_report.gpu_duration;
        session.metrics.download_duration +=
            embedding_report.download_duration + norm_report.download_duration;
        session.metrics.activation_upload_bytes +=
            std::mem::size_of::<u32>() + self.config.hidden_size * std::mem::size_of::<f32>();
        session.metrics.upload_bytes +=
            std::mem::size_of::<u32>() + self.config.hidden_size * std::mem::size_of::<f32>();
        session.metrics.download_bytes += self.config.hidden_size * std::mem::size_of::<f32>() * 2;
        metrics.embedding_duration += started_at.elapsed();
        metrics.norm_duration +=
            norm_report.upload_duration + norm_report.gpu_duration + norm_report.download_duration;
        Ok(gpu_hidden_states)
    }

    fn residual_resident_source(
        &self,
        first_layer_gpu_entry_tensor_qkv: &Option<FirstLayerTensorQkvEntry>,
        resident_hidden_tensor_qkv: &Option<ResidentTensorQkvEntry>,
    ) -> Option<GpuResidentBuffer> {
        if let Some((_, hidden_resident, ..)) = first_layer_gpu_entry_tensor_qkv.as_ref() {
            Some(hidden_resident.clone())
        } else {
            resident_hidden_tensor_qkv
                .as_ref()
                .map(|(hidden_resident, ..)| hidden_resident.clone())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_qkv_host_vectors(
        &self,
        session: &mut PackedGpuSession<'_>,
        layer_tensors: &LayerTensorNames,
        hidden_states: &[f32],
        first_layer_gpu_entry_tensor_qkv: &Option<FirstLayerTensorQkvEntry>,
        first_layer_gpu_entry_qkv: Option<FirstLayerHostQkvEntry>,
        resident_hidden_entry_qkv: Option<ResidentHostQkvEntry>,
        use_attention_qkv: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, bool), ReferenceError> {
        let kv_rows = self.config.num_key_value_heads * self.config.head_dim;
        if first_layer_gpu_entry_tensor_qkv.is_some() {
            return Ok((
                vec![0.0; self.config.hidden_size],
                vec![0.0; kv_rows],
                vec![0.0; kv_rows],
                false,
            ));
        }
        if let Some((_, q, k, v, ..)) = first_layer_gpu_entry_qkv {
            return Ok((q, k, v, false));
        }
        if let Some((_, q, k, v, ..)) = resident_hidden_entry_qkv {
            return Ok((q, k, v, false));
        }
        if use_attention_qkv {
            let (mut q, mut k, mut v) = session.take_qkv_scratch();
            session.run_projection_triplet_into(
                &layer_tensors.q_proj_weight,
                self.config.hidden_size,
                &layer_tensors.k_proj_weight,
                kv_rows,
                &layer_tensors.v_proj_weight,
                kv_rows,
                self.config.hidden_size,
                hidden_states,
                &mut q,
                &mut k,
                &mut v,
            )?;
            return Ok((q, k, v, true));
        }
        Ok((
            self.matvec_f16_resolved(&layer_tensors.q_proj_weight, hidden_states)?,
            self.matvec_f16_resolved(&layer_tensors.k_proj_weight, hidden_states)?,
            self.matvec_f16_resolved(&layer_tensors.v_proj_weight, hidden_states)?,
            false,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_gpu_attention_query_and_kv(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        layer_idx: usize,
        q: &mut Vec<f32>,
        k: &[f32],
        v: &[f32],
        first_layer_gpu_entry_tensor_qkv: &Option<FirstLayerTensorQkvEntry>,
        resident_hidden_tensor_qkv: &Option<ResidentTensorQkvEntry>,
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        cos: &[f32],
        sin: &[f32],
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<(Option<GpuResidentBuffer>, bool), ReferenceError> {
        if let Some((_, _, q_tensor, k_tensor, v_tensor, ..)) =
            first_layer_gpu_entry_tensor_qkv.as_ref()
        {
            let (query_out, key_out, qk_rope_report, compile_duration) = gpu_first_session
                .run_qk_rope_resident_query_and_key(
                    q_tensor,
                    k_tensor,
                    q_norm_weight,
                    k_norm_weight,
                    cos,
                    sin,
                )?;
            gpu_first_session.append_gpu_kv_tensors(layer_idx, &key_out, v_tensor)?;
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
            return Ok((Some(query_out), true));
        }
        if let Some((_, q_tensor, k_tensor, v_tensor, ..)) = resident_hidden_tensor_qkv.as_ref() {
            let (query_out, key_out, qk_rope_report, compile_duration) = gpu_first_session
                .run_qk_rope_resident_query_and_key(
                    q_tensor,
                    k_tensor,
                    q_norm_weight,
                    k_norm_weight,
                    cos,
                    sin,
                )?;
            gpu_first_session.append_gpu_kv_tensors(layer_idx, &key_out, v_tensor)?;
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
            return Ok((Some(query_out), true));
        }
        let ((q_out, key_out), qk_rope_report, compile_duration) = gpu_first_session
            .run_qk_rope_query_to_host_key_resident(q, k, q_norm_weight, k_norm_weight, cos, sin)?;
        *q = q_out;
        let q_resident = gpu_first_session
            .qk_rope_runner
            .as_ref()
            .expect("qk rope runner should exist after execution")
            .borrow()
            .query_resident_output();
        gpu_first_session.append_gpu_kv_key_tensor_and_value_host(layer_idx, &key_out, v)?;
        metrics.norm_duration += qk_rope_report.upload_duration
            + qk_rope_report.gpu_duration
            + qk_rope_report.download_duration;
        session.metrics.compile_duration += compile_duration;
        session.metrics.activation_upload_duration += qk_rope_report.upload_duration;
        session.metrics.upload_duration += qk_rope_report.upload_duration;
        session.metrics.gpu_duration += qk_rope_report.gpu_duration;
        session.metrics.download_duration += qk_rope_report.download_duration;
        session.metrics.activation_upload_bytes +=
            (q.len() + k.len() + q_norm_weight.len() + k_norm_weight.len() + cos.len() + sin.len())
                * std::mem::size_of::<f32>();
        session.metrics.upload_bytes +=
            (q.len() + k.len() + q_norm_weight.len() + k_norm_weight.len() + cos.len() + sin.len())
                * std::mem::size_of::<f32>();
        session.metrics.download_bytes += q.len() * std::mem::size_of::<f32>();
        Ok((Some(q_resident), true))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_cpu_qk_norm_rope(
        &self,
        q: &mut Vec<f32>,
        k: &mut Vec<f32>,
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        cos: &[f32],
        sin: &[f32],
        layer_tensors: &LayerTensorNames,
        metrics: &mut DecodeMetrics,
        non_offloaded_dense_duration: &mut Duration,
        session: &mut PackedGpuSession<'_>,
        started_at: Instant,
    ) {
        apply_head_rms_norm_weighted(
            q,
            self.config.num_attention_heads,
            self.config.head_dim,
            q_norm_weight,
            self.config.rms_norm_eps as f32,
        );
        apply_head_rms_norm_weighted(
            k,
            self.config.num_key_value_heads,
            self.config.head_dim,
            k_norm_weight,
            self.config.rms_norm_eps as f32,
        );
        apply_rotary_single(
            q,
            k,
            cos,
            sin,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        );
        let elapsed = started_at.elapsed();
        metrics.norm_duration += elapsed;
        *non_offloaded_dense_duration += elapsed;
        session.push_dense_stage_trace("qk_norm_rope", &layer_tensors.q_norm_weight, elapsed);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gpu_attention_only_tail_argmax(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        layer_idx: usize,
        position: usize,
        q: &[f32],
        q_resident: Option<&GpuResidentBuffer>,
        residual_resident: Option<&GpuResidentBuffer>,
        layer_cache: &LayerCache,
        residual: &[f32],
        attention_stage_metrics: &mut PackedAttentionStageMetrics,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<usize, ReferenceError> {
        let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
        let (next_token, attention_report, tail_report, compile_duration) = gpu_first_session
            .run_attention_layer_to_tail_argmax(
                layer_idx,
                position + 1,
                Some(q),
                q_resident,
                residual_resident,
                layer_cache.keys(),
                layer_cache.values(),
                residual,
                &final_norm_weight,
            )?;
        attention_stage_metrics.query_duration += attention_report.attention_gpu_duration;
        attention_stage_metrics.oproj_duration += attention_report.oproj_gpu_duration;
        attention_stage_metrics.residual_duration += attention_report.residual_add_gpu_duration;
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
        Ok(next_token)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gpu_attention_only_to_host(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        layer_idx: usize,
        position: usize,
        q: &[f32],
        q_resident: Option<&GpuResidentBuffer>,
        layer_cache: &LayerCache,
        residual: &[f32],
        attention_stage_metrics: &mut PackedAttentionStageMetrics,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<Vec<f32>, ReferenceError> {
        let (next_hidden, attention_report, compile_duration, download_duration) =
            gpu_first_session.run_attention_layer_to_host(
                layer_idx,
                position + 1,
                Some(q),
                q_resident,
                layer_cache.keys(),
                layer_cache.values(),
                residual,
            )?;
        attention_stage_metrics.query_duration += attention_report.attention_gpu_duration;
        attention_stage_metrics.oproj_duration += attention_report.oproj_gpu_duration;
        attention_stage_metrics.residual_duration += attention_report.residual_add_gpu_duration;
        metrics.attention_duration += attention_report.attention_gpu_duration
            + attention_report.oproj_gpu_duration
            + attention_report.residual_add_gpu_duration;
        session.metrics.compile_duration += compile_duration;
        session.metrics.gpu_duration += attention_report.attention_gpu_duration
            + attention_report.oproj_gpu_duration
            + attention_report.residual_add_gpu_duration;
        session.metrics.download_duration += download_duration;
        Ok(next_hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gpu_full_last_layer_argmax(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        layer_idx: usize,
        position: usize,
        q: &[f32],
        q_resident: Option<&GpuResidentBuffer>,
        residual_resident: Option<&GpuResidentBuffer>,
        layer_cache: &LayerCache,
        residual: &[f32],
        layer_tensors: &LayerTensorNames,
        attention_stage_metrics: &mut PackedAttentionStageMetrics,
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<usize, ReferenceError> {
        let post_norm_weight =
            self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
        let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
        let (next_token, attention_report, full_last_layer_report, compile_duration) =
            gpu_first_session.run_attention_to_full_last_layer_argmax(
                layer_idx,
                position + 1,
                Some(q),
                q_resident,
                residual_resident,
                layer_cache.keys(),
                layer_cache.values(),
                residual,
                &post_norm_weight,
                &final_norm_weight,
            )?;
        attention_stage_metrics.query_duration += attention_report.attention_gpu_duration;
        attention_stage_metrics.oproj_duration += attention_report.oproj_gpu_duration;
        attention_stage_metrics.residual_duration += attention_report.residual_add_gpu_duration;
        metrics.attention_duration += attention_report.attention_gpu_duration
            + attention_report.oproj_gpu_duration
            + attention_report.residual_add_gpu_duration;
        metrics.norm_duration += full_last_layer_report.post_norm_gpu_duration
            + full_last_layer_report.final_norm_gpu_duration;
        mlp_stage_metrics.swiglu_duration += full_last_layer_report.swiglu_pack_gpu_duration;
        mlp_stage_metrics.down_duration += full_last_layer_report.down_gpu_duration;
        mlp_stage_metrics.residual_duration += full_last_layer_report.residual_add_gpu_duration;
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
        session.metrics.download_duration += full_last_layer_report.logits_download_duration;
        Ok(next_token)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gpu_attention_mlp_resident_handoff(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        layer_idx: usize,
        position: usize,
        q: &[f32],
        q_resident: Option<&GpuResidentBuffer>,
        residual_resident: Option<&GpuResidentBuffer>,
        layer_cache: &LayerCache,
        residual: &[f32],
        layer_tensors: &LayerTensorNames,
        attention_stage_metrics: &mut PackedAttentionStageMetrics,
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<(), ReferenceError> {
        let post_norm_weight =
            self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
        let (attention_report, mlp_report, compile_duration) = gpu_first_session
            .run_attention_mlp_layer_resident(
                layer_idx,
                position + 1,
                Some(q),
                q_resident,
                residual_resident,
                layer_cache.keys(),
                layer_cache.values(),
                residual,
                &post_norm_weight,
            )?;
        attention_stage_metrics.query_duration += attention_report.attention_gpu_duration;
        attention_stage_metrics.oproj_duration += attention_report.oproj_gpu_duration;
        attention_stage_metrics.residual_duration += attention_report.residual_add_gpu_duration;
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
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_dense_attention_fallback(
        &self,
        session: &mut PackedGpuSession<'_>,
        layer_tensors: &LayerTensorNames,
        q: &[f32],
        layer_cache: &LayerCache,
        position: usize,
        residual: &[f32],
        use_attention_full: bool,
        use_attention_qkv: bool,
        attention_stage_metrics: &mut PackedAttentionStageMetrics,
        metrics: &mut DecodeMetrics,
        non_offloaded_dense_duration: &mut Duration,
    ) -> Result<Vec<f32>, ReferenceError> {
        let started_at = Instant::now();
        let attn_started_at = Instant::now();
        let attn = attention_single_query(
            q,
            layer_cache.keys(),
            layer_cache.values(),
            position + 1,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        );
        let attn_elapsed = attn_started_at.elapsed();
        attention_stage_metrics.query_duration += attn_elapsed;
        session.push_dense_stage_trace("attention_core", "attention_single_query", attn_elapsed);
        let oproj_started_at = Instant::now();
        let hidden = if use_attention_full {
            session.run_projection_add_residual(
                &layer_tensors.o_proj_weight,
                self.config.hidden_size,
                self.config.hidden_size,
                &attn,
                residual,
            )?
        } else {
            let attn_output = self.matvec_f16_resolved(&layer_tensors.o_proj_weight, &attn)?;
            residual
                .iter()
                .zip(attn_output.iter())
                .map(|(left, right)| left + right)
                .collect()
        };
        let oproj_elapsed = oproj_started_at.elapsed();
        attention_stage_metrics.oproj_duration += oproj_elapsed;
        let residual_elapsed = Duration::ZERO;
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
        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gpu_mlp_only_tail_argmax(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        layer_idx: usize,
        residual: &[f32],
        post_norm_weight: &[f32],
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<usize, ReferenceError> {
        let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
        let (next_token, mlp_report, tail_report, compile_duration) = gpu_first_session
            .run_mlp_layer_to_tail_argmax(
                layer_idx,
                residual,
                post_norm_weight,
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
        Ok(next_token)
    }

    fn run_gpu_mlp_only_resident_handoff(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        layer_idx: usize,
        residual: &[f32],
        post_norm_weight: &[f32],
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
        metrics: &mut DecodeMetrics,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<(), ReferenceError> {
        let (mlp_report, compile_duration) =
            gpu_first_session.run_mlp_layer_resident(layer_idx, residual, post_norm_weight)?;
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
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_mlp_input_pair_or_hidden(
        &self,
        session: &mut PackedGpuSession<'_>,
        layer_tensors: &LayerTensorNames,
        residual: &[f32],
        post_norm_weight: &[f32],
        use_gpu_mlp_entry: bool,
        hidden_states: &mut Vec<f32>,
        metrics: &mut DecodeMetrics,
        non_offloaded_dense_duration: &mut Duration,
    ) -> Result<Option<ResidentPackedPairProjection>, ReferenceError> {
        if use_gpu_mlp_entry {
            let started_at = Instant::now();
            let final_norm = session.run_final_norm_resident(residual, post_norm_weight)?;
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
            return Ok(Some(pair));
        }
        let started_at = Instant::now();
        *hidden_states =
            weighted_rms_norm(residual, post_norm_weight, self.config.rms_norm_eps as f32);
        let elapsed = started_at.elapsed();
        metrics.norm_duration += elapsed;
        *non_offloaded_dense_duration += elapsed;
        session.push_dense_stage_trace(
            "post_attention_norm",
            &layer_tensors.post_attention_layernorm_weight,
            elapsed,
        );
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gpu_swiglu_block_execution(
        &self,
        session: &mut PackedGpuSession<'_>,
        layer_tensors: &LayerTensorNames,
        pair_from_gpu_norm: &mut Option<ResidentPackedPairProjection>,
        hidden_states: &[f32],
        residual: &[f32],
        use_gpu_tail: bool,
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
    ) -> Result<GpuSwigluBlockExecution, ReferenceError> {
        let pair = if let Some(pair) = pair_from_gpu_norm.take() {
            pair
        } else {
            session.run_projection_pair_resident(
                &layer_tensors.gate_proj_weight,
                self.config.intermediate_size,
                &layer_tensors.up_proj_weight,
                self.config.intermediate_size,
                self.config.hidden_size,
                hidden_states,
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
            let hidden_gpu = session.run_vector_add_resident(down, residual)?;
            let residual_elapsed = residual_started_at.elapsed();
            Ok((
                None,
                Some(hidden_gpu),
                swiglu_elapsed,
                down_elapsed,
                residual_elapsed,
                Instant::now(),
            ))
        } else {
            let down = session.run_projection_from_packed_activation(
                &layer_tensors.down_proj_weight,
                self.config.hidden_size,
                self.config.intermediate_size,
                packed_activation,
            )?;
            let down_elapsed = down_started_at.elapsed();
            Ok((
                Some(down),
                None,
                swiglu_elapsed,
                down_elapsed,
                Duration::ZERO,
                Instant::now(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_cpu_mlp_execution(
        &self,
        session: &mut PackedGpuSession<'_>,
        layer_tensors: &LayerTensorNames,
        hidden_states: &[f32],
        residual: &[f32],
        use_mlp_gu: bool,
        use_mlp_full: bool,
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
    ) -> Result<GpuSwigluBlockExecution, ReferenceError> {
        let mut reusable_gate_up_scratch = false;
        let (gate, up, dense_tail_started_at) = if use_mlp_gu {
            let (mut gate, mut up) = session.take_gate_up_scratch();
            session.run_projection_pair_into(
                &layer_tensors.gate_proj_weight,
                self.config.intermediate_size,
                &layer_tensors.up_proj_weight,
                self.config.intermediate_size,
                self.config.hidden_size,
                hidden_states,
                &mut gate,
                &mut up,
            )?;
            reusable_gate_up_scratch = true;
            (gate, up, Instant::now())
        } else {
            let dense_started_at = Instant::now();
            (
                self.matvec_f16_resolved(&layer_tensors.gate_proj_weight, hidden_states)?,
                self.matvec_f16_resolved(&layer_tensors.up_proj_weight, hidden_states)?,
                dense_started_at,
            )
        };
        let swiglu_started_at = Instant::now();
        let mut mlp = session.take_mlp_scratch();
        swiglu_into(&gate, &up, &mut mlp);
        if reusable_gate_up_scratch {
            session.restore_gate_up_scratch(gate, up);
        }
        let swiglu_elapsed = swiglu_started_at.elapsed();
        mlp_stage_metrics.swiglu_duration += swiglu_elapsed;
        session.push_dense_stage_trace("mlp_swiglu", "swiglu", swiglu_elapsed);
        let down_started_at = Instant::now();
        let down = if use_mlp_full {
            session.run_projection_add_residual(
                &layer_tensors.down_proj_weight,
                self.config.hidden_size,
                self.config.intermediate_size,
                &mlp,
                residual,
            )?
        } else {
            let down = self.matvec_f16_resolved(&layer_tensors.down_proj_weight, &mlp)?;
            residual
                .iter()
                .zip(down.iter())
                .map(|(left, right)| left + right)
                .collect()
        };
        session.restore_mlp_scratch(mlp);
        let down_elapsed = down_started_at.elapsed();
        Ok((
            Some(down),
            None,
            swiglu_elapsed,
            down_elapsed,
            Duration::ZERO,
            dense_tail_started_at,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_hidden_after_mlp(
        &self,
        session: &mut PackedGpuSession<'_>,
        down: Option<Vec<f32>>,
        gpu_hidden_after_mlp: Option<ResidentGpuVectorAdd>,
        residual: &[f32],
        hidden: &mut Vec<f32>,
        resident_decode_state: &mut PackedResidentDecodeState,
        use_mlp_full: bool,
        use_gpu_swiglu_block: bool,
        gpu_residual_elapsed: Duration,
    ) -> Duration {
        if let Some(hidden_gpu) = gpu_hidden_after_mlp {
            *hidden = residual.to_vec();
            resident_decode_state.final_hidden_gpu = Some(hidden_gpu);
            resident_decode_state.resident_hidden_state = None;
            return gpu_residual_elapsed;
        }
        resident_decode_state.resident_hidden_state = None;
        if use_mlp_full && !use_gpu_swiglu_block {
            *hidden = down.expect("down must exist for cpu residual path");
            return Duration::ZERO;
        }
        let residual_started_at = Instant::now();
        *hidden = residual
            .iter()
            .zip(
                down.as_ref()
                    .expect("down must exist for cpu residual path")
                    .iter(),
            )
            .map(|(left, right)| left + right)
            .collect();
        let residual_elapsed = residual_started_at.elapsed();
        session.push_dense_stage_trace("mlp_residual", "mlp_residual_add", residual_elapsed);
        residual_elapsed
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_argmax_tail_result(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        session: &mut PackedGpuSession<'_>,
        hidden: &[f32],
        final_hidden_gpu: Option<&ResidentGpuVectorAdd>,
        metrics: &mut DecodeMetrics,
        non_offloaded_dense_duration: &mut Duration,
    ) -> Result<usize, ReferenceError> {
        let norm_started = Instant::now();
        let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
        let next_token = if let Some(hidden_gpu) = final_hidden_gpu {
            let (tail_result, tail_report, compile_duration) = gpu_first_session
                .run_tail_from_resident_hidden(&hidden_gpu.tensor, &final_norm_weight, true)?;
            let next_token = match tail_result {
                GpuTailResult::NextToken(token_id) => token_id,
                GpuTailResult::Logits(_) => unreachable!("argmax-only tail should return a token"),
            };
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
        } else if packed_use_gpu_tail() {
            let (tail_result, tail_report, compile_duration) =
                gpu_first_session.run_tail_from_host_hidden(hidden, &final_norm_weight, true)?;
            let next_token = match tail_result {
                GpuTailResult::NextToken(token_id) => token_id,
                GpuTailResult::Logits(_) => unreachable!("argmax-only tail should return a token"),
            };
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
        } else if packed_use_gpu_final_norm() {
            let final_norm = session.run_final_norm_resident(hidden, &final_norm_weight)?;
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
                weighted_rms_norm(hidden, &final_norm_weight, self.config.rms_norm_eps as f32);
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
        Ok(next_token)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_packed_decode_step_result(
        &self,
        argmax_only: bool,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        session: &mut PackedGpuSession<'_>,
        hidden: &[f32],
        final_hidden_gpu: Option<&ResidentGpuVectorAdd>,
        metrics: &mut DecodeMetrics,
        non_offloaded_dense_duration: &mut Duration,
    ) -> Result<PackedDecodeStepResult, ReferenceError> {
        if argmax_only {
            let next_token = self.finish_argmax_tail_result(
                gpu_first_session,
                session,
                hidden,
                final_hidden_gpu,
                metrics,
                non_offloaded_dense_duration,
            )?;
            Ok(PackedDecodeStepResult::NextToken(next_token))
        } else {
            let logits = self.finish_logits_tail_result(
                gpu_first_session,
                session,
                hidden,
                final_hidden_gpu,
                metrics,
                non_offloaded_dense_duration,
            )?;
            Ok(PackedDecodeStepResult::Logits(logits))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_logits_tail_result(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        session: &mut PackedGpuSession<'_>,
        hidden: &[f32],
        final_hidden_gpu: Option<&ResidentGpuVectorAdd>,
        metrics: &mut DecodeMetrics,
        non_offloaded_dense_duration: &mut Duration,
    ) -> Result<Vec<f32>, ReferenceError> {
        let norm_started = Instant::now();
        let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
        let logits = if let Some(hidden_gpu) = final_hidden_gpu {
            let (tail_result, tail_report, compile_duration) = gpu_first_session
                .run_tail_from_resident_hidden(&hidden_gpu.tensor, &final_norm_weight, false)?;
            let logits = match tail_result {
                GpuTailResult::Logits(logits) => logits,
                GpuTailResult::NextToken(_) => unreachable!("logits tail should return logits"),
            };
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
        } else if packed_use_gpu_tail() {
            let (tail_result, tail_report, compile_duration) =
                gpu_first_session.run_tail_from_host_hidden(hidden, &final_norm_weight, false)?;
            let logits = match tail_result {
                GpuTailResult::Logits(logits) => logits,
                GpuTailResult::NextToken(_) => unreachable!("logits tail should return logits"),
            };
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
                weighted_rms_norm(hidden, &final_norm_weight, self.config.rms_norm_eps as f32);
            let norm_elapsed = norm_started.elapsed();
            metrics.norm_duration += norm_elapsed;
            *non_offloaded_dense_duration += norm_elapsed;
            session.push_dense_stage_trace("final_norm", "model.norm.weight", norm_elapsed);
            let logits_started = Instant::now();
            let logits = session.run_projection(
                "model.embed_tokens.weight",
                self.config.vocab_size,
                self.config.hidden_size,
                &hidden,
            )?;
            metrics.logits_duration += logits_started.elapsed();
            logits
        };
        Ok(logits)
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

    fn get_or_create_embedding_lookup_gpu(
        &self,
    ) -> Result<CachedEmbeddingLookupGpuCacheEntry, ReferenceError> {
        if let Some(cached) = self.cached_embedding_lookup_gpu.borrow().as_ref().cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let (vocab, hidden, words) = self
            .weights
            .embedding_lookup_u32_words("model.embed_tokens.weight")
            .map_err(ReferenceError::Weight)?;
        let context = self.get_or_create_packed_gpu_context()?;
        let (runner, duration) =
            CachedGpuEmbeddingLookupRunner::new_with_context(context, &words, vocab, hidden)
                .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let runner = Rc::new(RefCell::new(runner));
        *self.cached_embedding_lookup_gpu.borrow_mut() = Some(runner.clone());
        Ok((runner, duration, false))
    }

    fn get_or_create_input_norm_gpu(
        &self,
    ) -> Result<CachedWeightedRmsNormGpuCacheEntry, ReferenceError> {
        if let Some(cached) = self.cached_input_norm_gpu.borrow().as_ref().cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (runner, duration) = CachedGpuWeightedRmsNormRunner::new_with_context(
            context,
            self.config.hidden_size,
            self.config.rms_norm_eps as f32,
        )
        .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let runner = Rc::new(RefCell::new(runner));
        *self.cached_input_norm_gpu.borrow_mut() = Some(runner.clone());
        Ok((runner, duration, false))
    }

    fn get_or_create_qk_rope_gpu(&self) -> Result<CachedQkRopeGpuCacheEntry, ReferenceError> {
        if let Some(cached) = self.cached_qk_rope_gpu.borrow().as_ref().cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let context = self.get_or_create_packed_gpu_context()?;
        let (runner, duration) = CachedGpuQkRopeRunner::new_with_context(
            context,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let runner = Rc::new(RefCell::new(runner));
        *self.cached_qk_rope_gpu.borrow_mut() = Some(runner.clone());
        Ok((runner, duration, false))
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

    fn get_or_create_attention_block_gpu(
        &self,
        layer_idx: usize,
        seq_capacity: usize,
    ) -> Result<(CachedAttentionBlockGpuRunner, Duration, bool), ReferenceError> {
        let cache_key = (layer_idx, seq_capacity);
        if let Some(cached) = self
            .cached_attention_block_gpu
            .borrow()
            .get(&cache_key)
            .cloned()
        {
            return Ok((cached, Duration::ZERO, true));
        }
        let layer_tensors = &self.layer_tensors[layer_idx];
        let (o_proj_packed, _, _) = self.get_or_create_projection_cache(
            &layer_tensors.o_proj_weight,
            self.config.hidden_size,
            self.config.hidden_size,
        )?;
        let o_proj_spec = PackedLinearSpec {
            code_words: o_proj_packed.code_words.clone(),
            scales: o_proj_packed.scales.clone(),
            group_size: o_proj_packed.group_size,
            rows: o_proj_packed.rows,
            cols: o_proj_packed.cols,
        };
        let context = self.get_or_create_packed_gpu_context()?;
        let runner = CachedGpuAttentionBlockRunner::new_with_context(
            context,
            seq_capacity,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            &o_proj_spec,
        )
        .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let duration = runner.compile_duration();
        let runner = Rc::new(RefCell::new(runner));
        self.cached_attention_block_gpu
            .borrow_mut()
            .insert(cache_key, runner.clone());
        Ok((runner, duration, false))
    }

    fn get_or_create_mlp_block_gpu(
        &self,
        layer_idx: usize,
    ) -> Result<(CachedMlpBlockGpuRunner, Duration, bool), ReferenceError> {
        if let Some(cached) = self.cached_mlp_block_gpu.borrow().get(&layer_idx).cloned() {
            return Ok((cached, Duration::ZERO, true));
        }
        let layer_tensors = &self.layer_tensors[layer_idx];
        let pair_cache_key = format!(
            "gpu_first::layer::{layer_idx}::mlp_pair::{}+{}",
            layer_tensors.gate_proj_weight, layer_tensors.up_proj_weight
        );
        let (pair_packed, _, _) = self.get_or_create_projection_pair_cache(
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
        let (down_packed, _, _) = self.get_or_create_projection_cache(
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
        let context = self.get_or_create_packed_gpu_context()?;
        let runner = CachedGpuMlpBlockRunner::new_with_context(
            context,
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.rms_norm_eps as f32,
            &pair_spec,
            &down_spec,
        )
        .map_err(|error| ReferenceError::Decode(error.to_string()))?;
        let duration = runner.compile_duration();
        let runner = Rc::new(RefCell::new(runner));
        self.cached_mlp_block_gpu
            .borrow_mut()
            .insert(layer_idx, runner.clone());
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
        let (logits_packed, _pack_duration, _pack_cache_hit) = self
            .get_or_create_projection_cache(
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
        let (pair_packed, _pair_pack_duration, _pair_pack_cache_hit) = self
            .get_or_create_projection_pair_cache(
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
        let (logits_packed, _logits_pack_duration, _logits_pack_cache_hit) = self
            .get_or_create_projection_cache(
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
        let mut cache = allocate_layer_cache_vec(
            self.config.num_hidden_layers,
            prompt_ids.len() + max_new_tokens,
            self.config.num_key_value_heads * self.config.head_dim,
            true,
        );
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
        let mut cache = allocate_layer_cache_vec(
            self.config.num_hidden_layers,
            prompt_ids.len(),
            self.config.num_key_value_heads * self.config.head_dim,
            true,
        );
        for (position, &token_id) in prompt_ids.iter().enumerate() {
            let _ = self.forward_step(token_id, position, &mut cache, &mut metrics)?;
        }
        metrics.total_duration = total_started.elapsed();
        Ok(metrics)
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
            let use_attention_full = use_attention_qkv && packed_use_attention_full();
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
            let use_mlp_full = use_mlp_gu && packed_use_mlp_full();
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
            finish_packed_decode_metrics(
                packed_enabled_label(use_attention_qkv, use_mlp_gu),
                total_started.elapsed(),
                &metrics,
                &attention_stage_metrics,
                &mlp_stage_metrics,
                non_offloaded_dense_duration,
                &session.metrics,
                String::new(),
            ),
        ))
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
        let mut cache = allocate_layer_cache_vec(
            self.config.num_hidden_layers,
            prompt_ids.len() + max_new_tokens,
            self.config.num_key_value_heads * self.config.head_dim,
            true,
        );
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
            layer_cache.append(&k, &v);
            let attn = attention_single_query(
                &q,
                layer_cache.keys(),
                layer_cache.values(),
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
            layer_cache.append(&k, &v);
            let attn = attention_single_query(
                &q,
                layer_cache.keys(),
                layer_cache.values(),
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

    fn select_packed_decode_stages(
        &self,
        layer_idx: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        use_attention_full: bool,
        use_mlp_full: bool,
        argmax_only: bool,
    ) -> PackedDecodeStageSelection {
        let is_last_layer = layer_idx + 1 == self.config.num_hidden_layers;
        let use_gpu_attention_block =
            use_attention_qkv && use_attention_full && packed_use_gpu_attention_block();
        let use_gpu_attention_mlp_block =
            use_gpu_attention_block && use_mlp_gu && use_mlp_full && packed_use_gpu_swiglu_block();
        let use_gpu_attention_only = use_gpu_attention_block && !use_gpu_attention_mlp_block;
        let use_gpu_attention_tail =
            use_gpu_attention_only && argmax_only && packed_use_gpu_tail() && is_last_layer;
        let use_gpu_full_last_layer_block = use_gpu_attention_mlp_block
            && argmax_only
            && packed_use_gpu_full_last_layer()
            && is_last_layer;
        let use_gpu_full_last_layer = use_mlp_gu
            && use_mlp_full
            && argmax_only
            && packed_use_gpu_full_last_layer()
            && is_last_layer;
        let use_gpu_mlp_entry = use_mlp_gu && use_mlp_full && packed_use_gpu_mlp_entry();
        let use_gpu_mlp_only = use_mlp_gu
            && use_mlp_full
            && packed_use_gpu_swiglu_block()
            && !use_gpu_attention_mlp_block;
        let use_gpu_mlp_tail =
            use_gpu_mlp_only && argmax_only && packed_use_gpu_tail() && is_last_layer;
        let use_gpu_swiglu_block = use_mlp_gu && use_mlp_full && packed_use_gpu_swiglu_block();
        let use_gpu_tail = use_gpu_swiglu_block
            && packed_use_gpu_tail()
            && packed_use_gpu_final_norm()
            && argmax_only
            && is_last_layer;
        PackedDecodeStageSelection {
            use_gpu_attention_block,
            use_gpu_attention_mlp_block,
            use_gpu_attention_only,
            use_gpu_attention_tail,
            use_gpu_full_last_layer_block,
            use_gpu_full_last_layer,
            use_gpu_mlp_entry,
            use_gpu_mlp_only,
            use_gpu_mlp_tail,
            use_gpu_swiglu_block,
            use_gpu_tail,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_attention_stage_after_qkv(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        layer_idx: usize,
        position: usize,
        q: &[f32],
        q_resident: Option<&GpuResidentBuffer>,
        residual_resident: Option<&GpuResidentBuffer>,
        layer_cache: &mut LayerCache,
        residual: &[f32],
        layer_tensors: &LayerTensorNames,
        stage_selection: &PackedDecodeStageSelection,
        use_attention_full: bool,
        use_attention_qkv: bool,
        attention_stage_metrics: &mut PackedAttentionStageMetrics,
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
        metrics: &mut DecodeMetrics,
        non_offloaded_dense_duration: &mut Duration,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<PackedAttentionStageOutcome, ReferenceError> {
        if stage_selection.use_gpu_attention_only {
            if stage_selection.use_gpu_attention_tail {
                let next_token = self.run_gpu_attention_only_tail_argmax(
                    gpu_first_session,
                    layer_idx,
                    position,
                    q,
                    q_resident,
                    residual_resident,
                    layer_cache,
                    residual,
                    attention_stage_metrics,
                    metrics,
                    session,
                )?;
                return Ok(PackedAttentionStageOutcome::NextToken(next_token));
            }
            let hidden = self.run_gpu_attention_only_to_host(
                gpu_first_session,
                layer_idx,
                position,
                q,
                q_resident,
                layer_cache,
                residual,
                attention_stage_metrics,
                metrics,
                session,
            )?;
            return Ok(PackedAttentionStageOutcome::Hidden(hidden));
        }

        if stage_selection.use_gpu_full_last_layer_block {
            let next_token = self.run_gpu_full_last_layer_argmax(
                gpu_first_session,
                layer_idx,
                position,
                q,
                q_resident,
                residual_resident,
                layer_cache,
                residual,
                layer_tensors,
                attention_stage_metrics,
                mlp_stage_metrics,
                metrics,
                session,
            )?;
            return Ok(PackedAttentionStageOutcome::NextToken(next_token));
        }

        if stage_selection.use_gpu_attention_mlp_block {
            self.run_gpu_attention_mlp_resident_handoff(
                gpu_first_session,
                layer_idx,
                position,
                q,
                q_resident,
                residual_resident,
                layer_cache,
                residual,
                layer_tensors,
                attention_stage_metrics,
                mlp_stage_metrics,
                metrics,
                session,
            )?;
            return Ok(PackedAttentionStageOutcome::ResidentMlp);
        }

        let hidden = self.apply_dense_attention_fallback(
            session,
            layer_tensors,
            q,
            layer_cache,
            position,
            residual,
            use_attention_full,
            use_attention_qkv,
            attention_stage_metrics,
            metrics,
            non_offloaded_dense_duration,
        )?;
        Ok(PackedAttentionStageOutcome::Hidden(hidden))
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_mlp_stage_after_attention(
        &self,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        layer_idx: usize,
        layer_tensors: &LayerTensorNames,
        residual: &[f32],
        hidden_states: &mut Vec<f32>,
        hidden: &mut Vec<f32>,
        resident_decode_state: &mut PackedResidentDecodeState,
        stage_selection: &PackedDecodeStageSelection,
        use_mlp_gu: bool,
        use_mlp_full: bool,
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
        metrics: &mut DecodeMetrics,
        non_offloaded_dense_duration: &mut Duration,
        session: &mut PackedGpuSession<'_>,
    ) -> Result<PackedMlpStageOutcome, ReferenceError> {
        if stage_selection.use_gpu_full_last_layer {
            let post_norm_weight =
                self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
            let post_norm_started = Instant::now();
            let post_norm = session.run_final_norm_resident(residual, &post_norm_weight)?;
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
            let hidden_gpu = session.run_vector_add_resident(down, residual)?;
            let residual_elapsed = residual_started.elapsed();
            mlp_stage_metrics.residual_duration += residual_elapsed;
            metrics.mlp_duration += mlp_started.elapsed();

            let final_norm_weight = self.load_vector_f32_resolved("model.norm.weight")?;
            let final_norm_started = Instant::now();
            let final_norm =
                session.run_final_norm_resident_from_vector_add(hidden_gpu, &final_norm_weight)?;
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
            return Ok(PackedMlpStageOutcome::NextToken(next_token));
        }

        let post_norm_weight =
            self.load_vector_f32_resolved(&layer_tensors.post_attention_layernorm_weight)?;
        if stage_selection.use_gpu_mlp_only {
            if stage_selection.use_gpu_mlp_tail {
                let next_token = self.run_gpu_mlp_only_tail_argmax(
                    gpu_first_session,
                    layer_idx,
                    residual,
                    &post_norm_weight,
                    mlp_stage_metrics,
                    metrics,
                    session,
                )?;
                return Ok(PackedMlpStageOutcome::NextToken(next_token));
            }
            self.run_gpu_mlp_only_resident_handoff(
                gpu_first_session,
                layer_idx,
                residual,
                &post_norm_weight,
                mlp_stage_metrics,
                metrics,
                session,
            )?;
            return Ok(PackedMlpStageOutcome::ResidentMlp);
        }

        let mut pair_from_gpu_norm = self.prepare_mlp_input_pair_or_hidden(
            session,
            layer_tensors,
            residual,
            &post_norm_weight,
            stage_selection.use_gpu_mlp_entry,
            hidden_states,
            metrics,
            non_offloaded_dense_duration,
        )?;

        let started_at = Instant::now();
        let (
            down,
            gpu_hidden_after_mlp,
            swiglu_elapsed,
            down_elapsed,
            gpu_residual_elapsed,
            dense_tail_started_at,
        ) = if stage_selection.use_gpu_swiglu_block {
            self.run_gpu_swiglu_block_execution(
                session,
                layer_tensors,
                &mut pair_from_gpu_norm,
                hidden_states,
                residual,
                stage_selection.use_gpu_tail,
                mlp_stage_metrics,
            )?
        } else {
            self.run_cpu_mlp_execution(
                session,
                layer_tensors,
                hidden_states,
                residual,
                use_mlp_gu,
                use_mlp_full,
                mlp_stage_metrics,
            )?
        };
        mlp_stage_metrics.down_duration += down_elapsed;
        let residual_elapsed = self.resolve_hidden_after_mlp(
            session,
            down,
            gpu_hidden_after_mlp,
            residual,
            hidden,
            resident_decode_state,
            use_mlp_full,
            stage_selection.use_gpu_swiglu_block,
            gpu_residual_elapsed,
        );
        mlp_stage_metrics.residual_duration += residual_elapsed;
        let elapsed = started_at.elapsed();
        metrics.mlp_duration += elapsed;
        if use_mlp_gu {
            if use_mlp_full {
                if stage_selection.use_gpu_swiglu_block {
                    if !stage_selection.use_gpu_tail {
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
        Ok(PackedMlpStageOutcome::Continue)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_packed_decode_layer_step(
        &self,
        token_id: usize,
        position: usize,
        cos: &[f32],
        sin: &[f32],
        layer_idx: usize,
        layer_cache: &mut LayerCache,
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
        hidden: &mut Vec<f32>,
        resident_decode_state: &mut PackedResidentDecodeState,
    ) -> Result<PackedDecodeLayerStepOutcome, ReferenceError> {
        let layer_tensors = &self.layer_tensors[layer_idx];
        let stage_selection = self.select_packed_decode_stages(
            layer_idx,
            use_attention_qkv,
            use_mlp_gu,
            use_attention_full,
            use_mlp_full,
            argmax_only,
        );

        let input_norm_weight =
            self.load_vector_f32_resolved(&layer_tensors.input_layernorm_weight)?;
        let resident_hidden_tensor_qkv = self.try_resident_hidden_tensor_qkv_entry(
            gpu_first_session,
            resident_decode_state.resident_hidden_state,
            layer_idx,
            use_attention_qkv,
            stage_selection.use_gpu_attention_block,
            &input_norm_weight,
            layer_tensors,
        )?;
        let resident_hidden_entry_qkv = self.try_resident_hidden_entry_qkv(
            gpu_first_session,
            resident_decode_state.resident_hidden_state,
            layer_idx,
            use_attention_qkv,
            stage_selection.use_gpu_attention_block,
            &input_norm_weight,
            layer_tensors,
        )?;
        let first_layer_gpu_entry_tensor_qkv = self.try_first_layer_gpu_entry_tensor_qkv(
            gpu_first_session,
            &resident_hidden_tensor_qkv,
            &resident_hidden_entry_qkv,
            layer_idx,
            token_id,
            use_attention_qkv,
            stage_selection.use_gpu_attention_block,
            &input_norm_weight,
            layer_tensors,
        )?;
        let first_layer_gpu_entry_qkv = self.try_first_layer_gpu_entry_qkv(
            gpu_first_session,
            &resident_hidden_tensor_qkv,
            &resident_hidden_entry_qkv,
            &first_layer_gpu_entry_tensor_qkv,
            layer_idx,
            token_id,
            use_attention_qkv,
            &input_norm_weight,
            layer_tensors,
        )?;
        let mut hidden_states = if let Some(entry) = first_layer_gpu_entry_tensor_qkv.as_ref() {
            self.apply_first_layer_tensor_qkv_entry(entry, hidden, metrics, session)
        } else if let Some(entry) = first_layer_gpu_entry_qkv.as_ref() {
            self.apply_first_layer_host_qkv_entry(entry, hidden, metrics, session)
        } else if let Some(entry) = resident_hidden_tensor_qkv.as_ref() {
            self.apply_resident_tensor_qkv_entry(entry, metrics, session)
        } else if let Some(entry) = resident_hidden_entry_qkv.as_ref() {
            self.apply_resident_host_qkv_entry(
                entry,
                hidden,
                metrics,
                session,
                resident_decode_state
                    .resident_hidden_state
                    .expect("resident hidden state should exist"),
                gpu_first_session,
            )?
        } else if layer_idx == 0 && packed_use_gpu_embedding() && packed_use_gpu_first_session() {
            self.apply_first_layer_embedding_norm_entry(
                gpu_first_session,
                token_id,
                &input_norm_weight,
                hidden,
                metrics,
                session,
            )?
        } else {
            let started_at = Instant::now();
            let hidden_states =
                weighted_rms_norm(hidden, &input_norm_weight, self.config.rms_norm_eps as f32);
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
        let residual_resident = self.residual_resident_source(
            &first_layer_gpu_entry_tensor_qkv,
            &resident_hidden_tensor_qkv,
        );

        let started_at = Instant::now();
        let (mut q, mut k, v, reusable_qkv_scratch) = self.prepare_qkv_host_vectors(
            session,
            layer_tensors,
            &hidden_states,
            &first_layer_gpu_entry_tensor_qkv,
            first_layer_gpu_entry_qkv,
            resident_hidden_entry_qkv,
            use_attention_qkv,
        )?;
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
        if stage_selection.use_gpu_attention_block {
            (q_resident, kv_appended_on_gpu) = self.prepare_gpu_attention_query_and_kv(
                gpu_first_session,
                layer_idx,
                &mut q,
                &k,
                &v,
                &first_layer_gpu_entry_tensor_qkv,
                &resident_hidden_tensor_qkv,
                &q_norm_weight,
                &k_norm_weight,
                cos,
                sin,
                metrics,
                session,
            )?;
        } else {
            self.apply_cpu_qk_norm_rope(
                &mut q,
                &mut k,
                &q_norm_weight,
                &k_norm_weight,
                cos,
                sin,
                layer_tensors,
                metrics,
                non_offloaded_dense_duration,
                session,
                started_at,
            );
        }

        if stage_selection.use_gpu_attention_block && !kv_appended_on_gpu {
            gpu_first_session.append_gpu_kv(layer_idx, &k, &v)?;
        }

        if !stage_selection.use_gpu_attention_block {
            layer_cache.keys.extend_from_slice(&k);
            layer_cache.values.extend_from_slice(&v);
        }

        match self.handle_attention_stage_after_qkv(
            gpu_first_session,
            layer_idx,
            position,
            &q,
            q_resident.as_ref(),
            residual_resident.as_ref(),
            layer_cache,
            &residual,
            layer_tensors,
            &stage_selection,
            use_attention_full,
            use_attention_qkv,
            attention_stage_metrics,
            mlp_stage_metrics,
            metrics,
            non_offloaded_dense_duration,
            session,
        )? {
            PackedAttentionStageOutcome::NextToken(next_token) => {
                if reusable_qkv_scratch {
                    session.restore_qkv_scratch(q, k, v);
                }
                return Ok(PackedDecodeLayerStepOutcome::NextToken(next_token));
            }
            PackedAttentionStageOutcome::ResidentMlp => {
                resident_decode_state.resident_hidden_state =
                    Some(ResidentHiddenState::Mlp { layer_idx });
                if reusable_qkv_scratch {
                    session.restore_qkv_scratch(q, k, v);
                }
                return Ok(PackedDecodeLayerStepOutcome::Continue);
            }
            PackedAttentionStageOutcome::Hidden(next_hidden) => {
                *hidden = next_hidden;
            }
        }

        if reusable_qkv_scratch {
            session.restore_qkv_scratch(q, k, v);
        }

        let residual = hidden.clone();
        match self.handle_mlp_stage_after_attention(
            gpu_first_session,
            layer_idx,
            layer_tensors,
            &residual,
            &mut hidden_states,
            hidden,
            resident_decode_state,
            &stage_selection,
            use_mlp_gu,
            use_mlp_full,
            mlp_stage_metrics,
            metrics,
            non_offloaded_dense_duration,
            session,
        )? {
            PackedMlpStageOutcome::NextToken(next_token) => {
                return Ok(PackedDecodeLayerStepOutcome::NextToken(next_token));
            }
            PackedMlpStageOutcome::ResidentMlp => {
                resident_decode_state.resident_hidden_state =
                    Some(ResidentHiddenState::Mlp { layer_idx });
            }
            PackedMlpStageOutcome::Continue => {}
        }

        Ok(PackedDecodeLayerStepOutcome::Continue)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_step_packed_decode(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut [LayerCache],
        metrics: &mut DecodeMetrics,
        attention_stage_metrics: &mut PackedAttentionStageMetrics,
        mlp_stage_metrics: &mut PackedMlpStageMetrics,
        non_offloaded_dense_duration: &mut Duration,
        resident_decode_state: &mut PackedResidentDecodeState,
        session: &mut PackedGpuSession<'_>,
        gpu_first_session: &mut GpuFirstRunnerCache<'_>,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        use_attention_full: bool,
        use_mlp_full: bool,
        argmax_only: bool,
    ) -> Result<PackedDecodeStepResult, ReferenceError> {
        let started_at = Instant::now();
        let mut hidden = if packed_use_gpu_embedding() && packed_use_gpu_first_session() {
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
        *resident_decode_state = PackedResidentDecodeState::default();

        for (layer_idx, layer_cache) in cache
            .iter_mut()
            .enumerate()
            .take(self.config.num_hidden_layers)
        {
            match self.execute_packed_decode_layer_step(
                token_id,
                position,
                &cos,
                &sin,
                layer_idx,
                layer_cache,
                metrics,
                attention_stage_metrics,
                mlp_stage_metrics,
                non_offloaded_dense_duration,
                session,
                gpu_first_session,
                use_attention_qkv,
                use_mlp_gu,
                use_attention_full,
                use_mlp_full,
                argmax_only,
                &mut hidden,
                resident_decode_state,
            )? {
                PackedDecodeLayerStepOutcome::Continue => {}
                PackedDecodeLayerStepOutcome::NextToken(next_token) => {
                    return Ok(PackedDecodeStepResult::NextToken(next_token));
                }
            }
        }

        self.finish_packed_decode_step_result(
            argmax_only,
            gpu_first_session,
            session,
            &hidden,
            resident_decode_state.final_hidden_gpu.as_ref(),
            metrics,
            non_offloaded_dense_duration,
        )
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
            layer_cache.append(&k, &v);
            let attn = attention_single_query(
                &q,
                layer_cache.keys(),
                layer_cache.values(),
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
    use super::{PackedDecodeStepResult, ReferenceModel};
    use crate::runtime::gpu_decode_engine::PackedDecodeSession;
    use crate::runtime::gpu_decode_env::{ScopedEnvVars, lock_env, packed_enabled_label};
    use crate::runtime::packed_model::write_packed_model_artifact;
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
            packed_enabled_label(true, true),
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
        assert_eq!(session.gpu_kv_len_tokens(0), Some(1));
        assert_eq!(
            session
                .gpu_kv_snapshot_lengths(0)
                .expect("gpu kv snapshots should read back"),
            Some((
                model.config.num_key_value_heads * model.config.head_dim,
                model.config.num_key_value_heads * model.config.head_dim,
            ))
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
        assert_eq!(session.cpu_kv_is_empty(0), Some(true));
        assert_eq!(session.cpu_kv_capacities(0), Some((0, 0)));
        assert_eq!(session.gpu_kv_len_tokens(0), Some(1));
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
        assert_eq!(session.cpu_kv_is_empty(0), Some(true));
        assert_eq!(session.cpu_kv_capacities(0), Some((0, 0)));
        assert_eq!(session.gpu_kv_len_tokens(0), Some(1));
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
        assert_eq!(session.cpu_kv_is_empty(0), Some(true));
        assert!(session.has_attention_block(0));
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
        let gpu_hidden = session
            .embedding_lookup_output(2)
            .expect("gpu embedding lookup should succeed")
            .expect("embedding runner should be created");
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
        assert!(session.has_raw_f32_projection_runner(&expected_key));
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
            panic!(
                "gpu-first session should be selected when gpu embedding and attention are enabled"
            );
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
        assert!(session.has_raw_f32_projection_runner(&expected_q_key));
        assert!(session.has_raw_f32_projection_runner(&expected_k_key));
        assert!(session.has_raw_f32_projection_runner(&expected_v_key));
        assert!(!session.has_raw_f32_projection_runner(&legacy_triplet_key));
        assert!(session.has_qk_rope_runner());
        assert_eq!(session.cpu_kv_is_empty(0), Some(true));
        assert_eq!(
            session
                .gpu_kv_len_tokens(0)
                .expect("layer 0 gpu kv cache should exist"),
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
        assert!(session.has_raw_f32_projection_runner(&expected_q_key));
        assert!(session.has_raw_f32_projection_runner(&expected_k_key));
        assert!(session.has_raw_f32_projection_runner(&expected_v_key));
        assert_eq!(session.cpu_kv_is_empty(0), Some(true));
        assert_eq!(session.cpu_kv_is_empty(1), Some(true));
        assert_eq!(
            session
                .gpu_kv_len_tokens(0)
                .expect("layer 0 gpu kv cache should exist"),
            1
        );
        assert_eq!(
            session
                .gpu_kv_len_tokens(1)
                .expect("layer 1 gpu kv cache should exist"),
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
        assert!(session.has_mlp_block(0));
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
        assert!(session.has_mlp_block(0));
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
        assert!(session.has_tail_block());
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
        assert!(session.has_attention_block(0));
        assert!(session.has_tail_block());
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
        assert!(session.has_mlp_block(0));
        assert!(session.has_tail_block());
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
        assert!(session.has_full_last_layer_block());
        assert_eq!(session.cpu_kv_is_empty(0), Some(true));
        assert_eq!(session.cpu_kv_is_empty(1), Some(true));
        assert_eq!(
            session
                .gpu_kv_len_tokens(0)
                .expect("layer 0 gpu kv cache should exist"),
            1
        );
        assert_eq!(
            session
                .gpu_kv_len_tokens(1)
                .expect("layer 1 gpu kv cache should exist"),
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
        for layer_idx in 0..2 {
            assert_eq!(
                session.cpu_kv_is_empty(layer_idx),
                Some(true),
                "layer {layer_idx} cpu kv cache should stay empty"
            );
            assert_eq!(
                session.cpu_kv_capacities(layer_idx),
                Some((0, 0)),
                "layer {layer_idx} cpu kv cache should not be preallocated"
            );
            assert_eq!(
                session
                    .gpu_kv_len_tokens(layer_idx)
                    .expect("layer gpu kv cache should exist"),
                2,
                "layer {layer_idx} gpu kv cache should contain both decoded tokens"
            );
        }
    }

    #[test]
    fn keeps_cpu_kv_empty_across_persistent_gpu_first_generation_loop_with_gpu_embedding() {
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
            ("JENGINE_GPU_EMBEDDING", "1"),
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
            panic!(
                "gpu-first session should be selected when gpu embedding and block flags are enabled"
            );
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
        assert!(session.has_raw_f32_projection_runner(&expected_q_key));
        assert!(session.has_raw_f32_projection_runner(&expected_k_key));
        assert!(session.has_raw_f32_projection_runner(&expected_v_key));
        for layer_idx in 0..2 {
            assert_eq!(session.cpu_kv_is_empty(layer_idx), Some(true));
            assert_eq!(session.cpu_kv_capacities(layer_idx), Some((0, 0)));
            assert_eq!(
                session
                    .gpu_kv_len_tokens(layer_idx)
                    .expect("layer gpu kv cache should exist"),
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
        for layer_idx in 0..2 {
            assert_eq!(
                session.cpu_kv_is_empty(layer_idx),
                Some(true),
                "layer {layer_idx} cpu kv cache should stay empty"
            );
            assert_eq!(
                session.cpu_kv_capacities(layer_idx),
                Some((0, 0)),
                "layer {layer_idx} cpu kv cache should not be preallocated"
            );
            assert_eq!(
                session
                    .gpu_kv_len_tokens(layer_idx)
                    .expect("layer gpu kv cache should exist"),
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
