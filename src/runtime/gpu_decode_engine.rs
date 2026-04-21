use crate::runtime::decode_plan::PackedDecodePlan;
use crate::runtime::gpu_decode_env::{gpu_first_use_attention_full, gpu_first_use_mlp_full};
use crate::runtime::gpu_decode_metrics::PackedDecodeMetrics;
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
}

pub enum PackedDecodeSession<'a> {
    Legacy(PersistentPackedDecodeSession<'a>),
    GpuFirst(GpuFirstPackedDecodeSession<'a>),
}

impl<'a> PackedDecodeSession<'a> {
    pub fn push_prompt_token(
        &mut self,
        token_id: usize,
    ) -> Result<crate::runtime::gpu_decode_session_state::PackedDecodeStepResult, ReferenceError> {
        match self {
            Self::Legacy(session) => session.push_prompt_token(token_id),
            Self::GpuFirst(session) => session.push_prompt_token(token_id),
        }
    }

    pub fn push_generated_token(
        &mut self,
        token_id: usize,
    ) -> Result<crate::runtime::gpu_decode_session_state::PackedDecodeStepResult, ReferenceError> {
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
    ) -> crate::runtime::gpu_decode_output::PackedDecodeResult {
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
