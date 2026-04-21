use crate::runtime::decode_plan::PackedDecodePlan;
use crate::runtime::gpu_decode_env::{gpu_first_use_attention_full, gpu_first_use_mlp_full};
use crate::runtime::gpu_decode_metrics::PackedDecodeMetrics;
use crate::runtime::gpu_decode_output::PackedDecodeResult;
use crate::runtime::gpu_decode_session_state::PackedDecodeStepResult;
use crate::runtime::reference::{
    GpuFirstPackedDecodeSession, PersistentPackedDecodeSession, ReferenceModel,
};
use crate::runtime::reference_error::ReferenceError;

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

impl<'a> PersistentPackedDecodeSession<'a> {
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
        self.model.prewarm_packed_decode_caches_internal(
            self.request.expected_tokens,
            self.request.use_attention_qkv,
            self.request.use_mlp_gu,
            self.plan.use_attention_full,
            self.plan.use_mlp_full,
        )
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
}

impl ReferenceModel {
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
