use crate::runtime::gpu_decode_metrics::PackedDecodeMetrics;
use crate::runtime::reference::DecodeMetrics;

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
