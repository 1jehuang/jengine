use crate::runtime::decode_plan::PackedDecodePlan;
use crate::runtime::reference::{GpuFirstPackedDecodeSession, PackedDecodeSession, PersistentPackedDecodeSession, ReferenceModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDecodeSessionMode {
    GpuFirst,
    Legacy,
}

pub struct GpuDecodeEngine<'a> {
    model: &'a ReferenceModel,
    expected_tokens: usize,
    use_attention_qkv: bool,
    use_mlp_gu: bool,
    argmax_only: bool,
    plan: PackedDecodePlan,
}

impl<'a> GpuDecodeEngine<'a> {
    pub fn new(
        model: &'a ReferenceModel,
        expected_tokens: usize,
        use_attention_qkv: bool,
        use_mlp_gu: bool,
        argmax_only: bool,
    ) -> Self {
        let plan = PackedDecodePlan::from_env(use_attention_qkv, use_mlp_gu, argmax_only);
        Self {
            model,
            expected_tokens,
            use_attention_qkv,
            use_mlp_gu,
            argmax_only,
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
                    self.expected_tokens,
                    self.use_attention_qkv,
                    self.use_mlp_gu,
                    self.argmax_only,
                ),
            ),
            GpuDecodeSessionMode::Legacy => PackedDecodeSession::Legacy(
                PersistentPackedDecodeSession::new(
                    self.model,
                    self.expected_tokens,
                    self.use_attention_qkv,
                    self.use_mlp_gu,
                    self.argmax_only,
                ),
            ),
        }
    }
}
