use std::time::Duration;

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
