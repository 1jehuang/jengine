use crate::runtime::decode_plan::PackedDecodePlan;
use crate::runtime::reference::{GpuFirstPackedDecodeSession, PackedDecodeSession, PersistentPackedDecodeSession, ReferenceModel};

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

pub struct GpuDecodeEngine<'a> {
    model: &'a ReferenceModel,
    request: PackedDecodeRequest,
    plan: PackedDecodePlan,
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
