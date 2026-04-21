use crate::runtime::decode_plan::PackedDecodePlan;
use crate::runtime::gpu_decode_env::{gpu_first_use_attention_full, gpu_first_use_mlp_full};
use crate::runtime::reference::{
    GpuFirstPackedDecodeSession, PackedDecodeSession, PersistentPackedDecodeSession,
    ReferenceError, ReferenceModel,
};

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
