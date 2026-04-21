use crate::runtime::decode_plan::PackedDecodePlan;
use crate::runtime::gpu_decode_env::{gpu_first_use_attention_full, gpu_first_use_mlp_full};
use crate::runtime::gpu_decode_metrics::{
    DecodeMetrics, PackedAttentionStageMetrics, PackedDecodeMetrics,
    PackedGpuSessionMetrics,
    PackedDecodeValidationReport, PackedMlpStageMetrics,
    account_projection_report,
};
use crate::runtime::gpu_decode_output::{PackedDecodeResult, PackedDispatchTrace};
use crate::gpu::resident_buffer::GpuResidentBuffer;
use crate::runtime::gpu_decode_projection_state::{
    PackedProjectionCache, PreparedProjectionRunner, ResidentGpuPackedActivation,
    ResidentPackedProjection,
};
use crate::runtime::gpu_decode_scratch::PackedDecodeScratch;
use crate::runtime::gpu_decode_session_state::{
    LayerCache, PackedDecodeStepResult, allocate_layer_cache_vec,
};
use crate::runtime::gpu_decode_state::PackedResidentDecodeState;
use crate::model::tokenizer::TokenizerRuntime;
use crate::runtime::reference::{GpuFirstRunnerCache, ReferenceModel};
use crate::runtime::reference_error::ReferenceError;
use std::time::Duration;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDecodeSessionMode {
    GpuFirst,
    Legacy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedDecodeRequest {
    pub expected_tokens: usize,
    pub use_attention_qkv: bool,
    pub use_mlp_gu: bool,
    pub argmax_only: bool,
}

impl PackedDecodeRequest {
    pub fn new(
        expected_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        argmax_only: bool,
    ) -> Self {
        Self {
            expected_tokens,
            use_attention_qkv,
            use_mlp_gu,
            argmax_only,
        }
    }
}

pub struct GpuDecodeEngine<'a> {
    model: &'a ReferenceModel,
    request: PackedDecodeRequest,
    plan: PackedDecodePlan,
}

pub struct PersistentPackedDecodeSession<'a> {
    pub(crate) model: &'a ReferenceModel,
    pub(crate) cache: Vec<LayerCache>,
    pub(crate) gpu_session: PackedGpuSession<'a>,
    pub(crate) gpu_first_session: GpuFirstRunnerCache<'a>,
    pub(crate) metrics: DecodeMetrics,
    pub(crate) attention_stage_metrics: PackedAttentionStageMetrics,
    pub(crate) mlp_stage_metrics: PackedMlpStageMetrics,
    pub(crate) non_offloaded_dense_duration: std::time::Duration,
    pub(crate) resident_decode_state: PackedResidentDecodeState,
    pub(crate) next_position: usize,
    pub(crate) use_attention_qkv: bool,
    pub(crate) use_mlp_gu: bool,
    pub(crate) use_attention_full: bool,
    pub(crate) use_mlp_full: bool,
    pub(crate) argmax_only: bool,
}

pub struct GpuFirstPackedDecodeSession<'a> {
    pub(crate) inner: PersistentPackedDecodeSession<'a>,
}

pub(crate) struct PackedGpuSession<'a> {
    pub(crate) model: &'a ReferenceModel,
    pub(crate) metrics: PackedGpuSessionMetrics,
    pub(crate) dispatch_trace: Vec<PackedDispatchTrace>,
    pub(crate) scratch: PackedDecodeScratch,
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

    pub(crate) fn take_qkv_scratch(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.scratch.take_qkv()
    }

    pub(crate) fn restore_qkv_scratch(&mut self, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>) {
        self.scratch.restore_qkv(q, k, v)
    }

    pub(crate) fn take_gate_up_scratch(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.scratch.take_gate_up()
    }

    pub(crate) fn restore_gate_up_scratch(&mut self, gate: Vec<f32>, up: Vec<f32>) {
        self.scratch.restore_gate_up(gate, up)
    }

    pub(crate) fn take_mlp_scratch(&mut self) -> Vec<f32> {
        self.scratch.take_mlp()
    }

    pub(crate) fn restore_mlp_scratch(&mut self, mlp: Vec<f32>) {
        self.scratch.restore_mlp(mlp)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_dispatch_trace(
        &mut self,
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

    pub(crate) fn push_dense_stage_trace(
        &mut self,
        stage: &str,
        tensor_name: &str,
        cpu_duration: std::time::Duration,
    ) {
        self.dispatch_trace.push(PackedDispatchTrace::dense_stage(
            self.dispatch_trace.len() + 1,
            stage,
            tensor_name,
            cpu_duration,
        ));
    }

    pub(crate) fn account_projection_report(
        &mut self,
        packed: &PackedProjectionCache,
        cols: usize,
        weight_upload_duration: std::time::Duration,
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

    pub(crate) fn run_projection(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, ReferenceError> {
        let resident = self.run_projection_resident(tensor_name, rows, cols, input, "single")?;
        self.download_projection_output(resident)
    }

    pub(crate) fn run_projection_argmax(
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

    pub(crate) fn run_projection_argmax_from_packed_activation(
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
        self.dispatch_trace.push(PackedDispatchTrace::resident_stage(
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

    pub(crate) fn run_projection_resident(
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

    pub(crate) fn download_projection_output(
        &mut self,
        resident: ResidentPackedProjection,
    ) -> Result<Vec<f32>, ReferenceError> {
        let (output, download_duration) = resident
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

    pub(crate) fn prepare_projection_runner(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<PreparedProjectionRunner, ReferenceError> {
        let (packed, pack_duration, pack_cache_hit) =
            self.model.get_or_create_projection_cache(tensor_name, rows, cols)?;
        let (runner, compile_duration, weight_upload_duration, gpu_cache_hit) =
            self.model.get_or_create_projection_gpu(tensor_name, &packed)?;
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
}

impl<'a> PersistentPackedDecodeSession<'a> {
    pub(crate) fn new_with_cpu_kv_preallocation(
        model: &'a ReferenceModel,
        expected_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        argmax_only: bool,
        preallocate_cpu_kv: bool,
    ) -> Self {
        Self {
            model,
            cache: allocate_layer_cache_vec(
                model.config.num_hidden_layers,
                expected_tokens,
                model.config.num_key_value_heads * model.config.head_dim,
                preallocate_cpu_kv,
            ),
            gpu_session: PackedGpuSession::new(model),
            gpu_first_session: GpuFirstRunnerCache::new(model, expected_tokens),
            metrics: DecodeMetrics {
                prompt_tokens: 0,
                generated_tokens: 0,
                total_duration: std::time::Duration::ZERO,
                embedding_duration: std::time::Duration::ZERO,
                norm_duration: std::time::Duration::ZERO,
                qkv_duration: std::time::Duration::ZERO,
                attention_duration: std::time::Duration::ZERO,
                mlp_duration: std::time::Duration::ZERO,
                logits_duration: std::time::Duration::ZERO,
            },
            attention_stage_metrics: PackedAttentionStageMetrics::default(),
            mlp_stage_metrics: PackedMlpStageMetrics::default(),
            non_offloaded_dense_duration: std::time::Duration::ZERO,
            resident_decode_state: PackedResidentDecodeState::default(),
            next_position: 0,
            use_attention_qkv,
            use_mlp_gu,
            use_attention_full: use_attention_qkv
                && crate::runtime::gpu_decode_env::packed_use_attention_full(),
            use_mlp_full: use_mlp_gu && crate::runtime::gpu_decode_env::packed_use_mlp_full(),
            argmax_only,
        }
    }

    pub(crate) fn new(
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

    pub(crate) fn set_full_modes(&mut self, use_attention_full: bool, use_mlp_full: bool) {
        self.use_attention_full = use_attention_full;
        self.use_mlp_full = use_mlp_full;
    }

    pub fn dispatch_trace(&self) -> &[crate::runtime::gpu_decode_output::PackedDispatchTrace] {
        &self.gpu_session.dispatch_trace
    }

    pub(crate) fn next_position(&self) -> usize {
        self.next_position
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
        self.resident_decode_state = PackedResidentDecodeState::default();
        let result = self.model.forward_step_packed_decode(
            token_id,
            self.next_position,
            &mut self.cache,
            &mut self.metrics,
            &mut self.attention_stage_metrics,
            &mut self.mlp_stage_metrics,
            &mut self.non_offloaded_dense_duration,
            &mut self.resident_decode_state,
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

    pub fn finish_metrics(
        self,
        enabled_projections: String,
        total_duration: std::time::Duration,
        output_text: String,
    ) -> PackedDecodeMetrics {
        crate::runtime::gpu_decode_metrics::finish_packed_decode_metrics(
            enabled_projections,
            total_duration,
            &self.metrics,
            &self.attention_stage_metrics,
            &self.mlp_stage_metrics,
            self.non_offloaded_dense_duration,
            &self.gpu_session.metrics,
            output_text,
        )
    }

    pub fn finish_result(
        self,
        enabled_projections: String,
        total_duration: std::time::Duration,
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
        let metrics = crate::runtime::gpu_decode_metrics::finish_packed_decode_metrics(
            enabled_projections,
            total_duration,
            &decode_metrics,
            &attention_stage_metrics,
            &mlp_stage_metrics,
            non_offloaded_dense_duration,
            &gpu_session.metrics,
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

impl<'a> GpuFirstPackedDecodeSession<'a> {
    pub(crate) fn new(
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
        inner.set_full_modes(
            gpu_first_use_attention_full(use_attention_qkv),
            gpu_first_use_mlp_full(use_mlp_gu),
        );
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
        total_duration: std::time::Duration,
        output_text: String,
    ) -> PackedDecodeMetrics {
        self.inner
            .finish_metrics(enabled_projections, total_duration, output_text)
    }

    pub fn finish_result(
        self,
        enabled_projections: String,
        total_duration: std::time::Duration,
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

    pub(crate) fn has_qk_rope_runner(&self) -> bool {
        self.inner.gpu_first_session.has_qk_rope_runner()
    }

    pub(crate) fn has_raw_f32_projection_runner(&self, key: &str) -> bool {
        self.inner
            .gpu_first_session
            .has_raw_f32_projection_runner(key)
    }

    pub(crate) fn gpu_kv_len_tokens(&self, layer_idx: usize) -> Option<usize> {
        self.inner.gpu_first_session.gpu_kv_len_tokens(layer_idx)
    }

    pub(crate) fn cpu_kv_is_empty(&self, layer_idx: usize) -> Option<bool> {
        self.inner
            .cache
            .get(layer_idx)
            .map(|cache| cache.cpu_kv_is_empty())
    }

    pub(crate) fn cpu_kv_capacities(&self, layer_idx: usize) -> Option<(usize, usize)> {
        self.inner
            .cache
            .get(layer_idx)
            .map(|cache| (cache.keys_capacity(), cache.values_capacity()))
    }

    pub(crate) fn gpu_kv_snapshot_lengths(
        &self,
        layer_idx: usize,
    ) -> Result<Option<(usize, usize)>, ReferenceError> {
        self.inner
            .gpu_first_session
            .gpu_kv_snapshot_lengths(layer_idx)
    }

    pub(crate) fn has_attention_block(&self, layer_idx: usize) -> bool {
        self.inner.gpu_first_session.has_attention_block(layer_idx)
    }

    pub(crate) fn has_mlp_block(&self, layer_idx: usize) -> bool {
        self.inner.gpu_first_session.has_mlp_block(layer_idx)
    }

    pub(crate) fn has_tail_block(&self) -> bool {
        self.inner.gpu_first_session.has_tail_block()
    }

    pub(crate) fn has_full_last_layer_block(&self) -> bool {
        self.inner.gpu_first_session.has_full_last_layer_block()
    }

    pub(crate) fn embedding_lookup_output(
        &mut self,
        token_id: usize,
    ) -> Result<Option<Vec<f32>>, ReferenceError> {
        self.inner.gpu_first_session.embedding_lookup_output(token_id)
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
            Self::Legacy(session) => session.next_position(),
            Self::GpuFirst(session) => session.next_position(),
        }
    }

    pub fn is_gpu_first(&self) -> bool {
        matches!(self, Self::GpuFirst(_))
    }

    pub fn finish_metrics(
        self,
        enabled_projections: String,
        total_duration: std::time::Duration,
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
        total_duration: std::time::Duration,
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

impl<'a> GpuDecodeEngine<'a> {
    pub fn new(model: &'a ReferenceModel, request: PackedDecodeRequest) -> Self {
        let plan = PackedDecodePlan::from_env(
            request.use_attention_qkv,
            request.use_mlp_gu,
            request.argmax_only,
        );
        Self {
            model,
            request,
            plan,
        }
    }

    pub fn plan(&self) -> PackedDecodePlan {
        self.plan
    }

    pub fn session_mode(&self) -> GpuDecodeSessionMode {
        if self.plan.gpu_first_session {
            GpuDecodeSessionMode::GpuFirst
        } else {
            GpuDecodeSessionMode::Legacy
        }
    }

    pub fn prewarm(&self) -> Result<(), ReferenceError> {
        if self.model.packed_model.is_none() {
            return Ok(());
        }
        let kv_rows = self.model.config.num_key_value_heads * self.model.config.head_dim;
        self.model.prewarm_layer_projection_caches(
            &self.plan,
            self.request.use_attention_qkv,
            self.request.use_mlp_gu,
            self.plan.use_attention_full,
            self.plan.use_mlp_full,
            kv_rows,
        )?;
        self.model
            .prewarm_tail_support_caches(&self.plan, self.request.use_mlp_gu)?;
        if self.plan.gpu_first_session {
            let mut gpu_first_cache =
                GpuFirstRunnerCache::new(self.model, self.request.expected_tokens.max(1));
            gpu_first_cache.prewarm_decode_path(
                self.request.use_attention_qkv,
                self.request.use_mlp_gu,
            )?;
        }
        Ok(())
    }

    pub fn begin_packed_session(&self) -> PackedDecodeSession<'a> {
        match self.session_mode() {
            GpuDecodeSessionMode::GpuFirst => PackedDecodeSession::GpuFirst(
                GpuFirstPackedDecodeSession::new(
                    self.model,
                    self.request.expected_tokens,
                    self.request.use_attention_qkv,
                    self.request.use_mlp_gu,
                    self.request.argmax_only,
                ),
            ),
            GpuDecodeSessionMode::Legacy => PackedDecodeSession::Legacy(
                PersistentPackedDecodeSession::new(
                    self.model,
                    self.request.expected_tokens,
                    self.request.use_attention_qkv,
                    self.request.use_mlp_gu,
                    self.request.argmax_only,
                ),
            ),
        }
    }

    pub fn generate_from_token_ids(
        &self,
        tokenizer: &TokenizerRuntime,
        prompt_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<PackedDecodeResult, ReferenceError> {
        if prompt_ids.is_empty() {
            return Err(ReferenceError::Decode(
                "prompt_ids cannot be empty".to_string(),
            ));
        }
        let total_started = Instant::now();
        let mut session = self.begin_packed_session();
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
            crate::runtime::gpu_decode_env::packed_enabled_label(
                self.request.use_attention_qkv,
                self.request.use_mlp_gu,
            ),
            total_started.elapsed(),
            output_ids,
            output_text,
        ))
    }

    pub fn generate_from_prompt(
        &self,
        tokenizer: &TokenizerRuntime,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<PackedDecodeResult, ReferenceError> {
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
        GpuDecodeEngine::new(
            self.model,
            PackedDecodeRequest::new(
                prompt_ids.len() + max_new_tokens,
                self.request.use_attention_qkv,
                self.request.use_mlp_gu,
                self.request.argmax_only,
            ),
        )
        .generate_from_token_ids(tokenizer, &prompt_ids, max_new_tokens)
    }

    pub fn prefill_logits_from_token_ids(
        &self,
        prompt_ids: &[usize],
    ) -> Result<Vec<f32>, ReferenceError> {
        if prompt_ids.is_empty() {
            return Err(ReferenceError::Decode(
                "prompt_ids cannot be empty".to_string(),
            ));
        }
        let mut last_logits = Vec::new();
        let mut session = self.begin_packed_session();
        for &token_id in prompt_ids {
            last_logits = match session.push_prompt_token(token_id)? {
                PackedDecodeStepResult::Logits(logits) => logits,
                PackedDecodeStepResult::NextToken(_) => {
                    unreachable!("full-logits prefill path should not return argmax-only output")
                }
            };
        }
        Ok(last_logits)
    }

    pub fn prefill_logits_from_prompt(
        &self,
        tokenizer: &TokenizerRuntime,
        prompt: &str,
    ) -> Result<Vec<f32>, ReferenceError> {
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
        GpuDecodeEngine::new(
            self.model,
            PackedDecodeRequest::new(
                prompt_ids.len(),
                self.request.use_attention_qkv,
                self.request.use_mlp_gu,
                self.request.argmax_only,
            ),
        )
        .prefill_logits_from_token_ids(&prompt_ids)
    }

    pub fn benchmark_step_from_token_ids(
        &self,
        prompt_ids: &[usize],
    ) -> Result<PackedDecodeMetrics, ReferenceError> {
        if prompt_ids.is_empty() {
            return Err(ReferenceError::Decode(
                "prompt_ids cannot be empty".to_string(),
            ));
        }

        let total_started = Instant::now();
        let mut session = self.begin_packed_session();
        for (position, &token_id) in prompt_ids.iter().enumerate() {
            debug_assert_eq!(position, session.next_position());
            let _ = session.push_prompt_token(token_id)?;
        }

        Ok(session.finish_metrics(
            crate::runtime::gpu_decode_env::packed_enabled_label(
                self.request.use_attention_qkv,
                self.request.use_mlp_gu,
            ),
            total_started.elapsed(),
            String::new(),
        ))
    }
}

impl ReferenceModel {
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

    pub fn benchmark_packed_step_from_token_ids(
        &self,
        prompt_ids: &[usize],
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<PackedDecodeMetrics, ReferenceError> {
        GpuDecodeEngine::new(
            self,
            PackedDecodeRequest::new(prompt_ids.len(), use_attention_qkv, use_mlp_gu, true),
        )
        .benchmark_step_from_token_ids(prompt_ids)
    }

    pub fn compare_prefill_logits_against(
        &self,
        dense_reference: &ReferenceModel,
        prompt: &str,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<PackedDecodeValidationReport, ReferenceError> {
        let dense_logits = dense_reference.prefill_logits_for_variant(prompt, false, false)?;
        let packed_logits = self.prefill_logits_for_variant(prompt, use_attention_qkv, use_mlp_gu)?;
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
        GpuDecodeEngine::new(
            self,
            PackedDecodeRequest::new(1, use_attention_qkv, use_mlp_gu, true),
        )
        .generate_from_prompt(tokenizer, prompt, max_new_tokens)
    }

    pub fn generate_packed_from_token_ids(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
    ) -> Result<PackedDecodeResult, ReferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ReferenceError::Decode("tokenizer is not loaded".to_string()))?;
        GpuDecodeEngine::new(
            self,
            PackedDecodeRequest::new(
                prompt_ids.len() + max_new_tokens,
                use_attention_qkv,
                use_mlp_gu,
                true,
            ),
        )
        .generate_from_token_ids(tokenizer, prompt_ids, max_new_tokens)
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
        GpuDecodeEngine::new(
            self,
            PackedDecodeRequest::new(1, use_attention_qkv, use_mlp_gu, false),
        )
        .prefill_logits_from_prompt(tokenizer, prompt)
    }

    pub fn prewarm_packed_decode_caches(
        &self,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        _use_attention_full: bool,
        _use_mlp_full: bool,
    ) -> Result<(), ReferenceError> {
        self.prewarm_packed_decode_caches_with_expected_tokens(
            1,
            use_attention_qkv,
            use_mlp_gu,
            false,
        )
    }

    pub fn prewarm_packed_decode_caches_with_expected_tokens(
        &self,
        expected_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        argmax_only: bool,
    ) -> Result<(), ReferenceError> {
        GpuDecodeEngine::new(
            self,
            PackedDecodeRequest::new(
                expected_tokens,
                use_attention_qkv,
                use_mlp_gu,
                argmax_only,
            ),
        )
        .prewarm()
    }

    pub fn begin_packed_decode_session(
        &self,
        expected_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        argmax_only: bool,
    ) -> PackedDecodeSession<'_> {
        GpuDecodeEngine::new(
            self,
            PackedDecodeRequest::new(
                expected_tokens,
                use_attention_qkv,
                use_mlp_gu,
                argmax_only,
            ),
        )
        .begin_packed_session()
    }
}
