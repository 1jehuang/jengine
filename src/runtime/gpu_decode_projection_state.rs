use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use crate::gpu::pack_f16_pairs::CachedGpuPackF16PairsRunner;
use crate::gpu::packed_matvec::{CachedGpuPackedMatvecRunner, GpuPackedMatvecReport};
use crate::gpu::resident_buffer::GpuResidentBuffer;
use crate::gpu::swiglu_combined::{CachedGpuSwigluCombinedRunner, GpuSwigluCombinedReport};
use crate::gpu::swiglu_pack_f16_pairs::CachedGpuSwigluPackF16PairsRunner;
use crate::gpu::vector_add::{CachedGpuVectorAddRunner, GpuVectorAddReport};
use crate::gpu::weighted_rms_norm::{CachedGpuWeightedRmsNormRunner, GpuWeightedRmsNormReport};

#[derive(Debug, Clone)]
pub(crate) struct PackedProjectionCache {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub code_words: Vec<u32>,
    pub scales: Vec<f32>,
}

pub(crate) struct PreparedProjectionRunner {
    pub packed: Rc<PackedProjectionCache>,
    pub runner: Rc<RefCell<CachedGpuPackedMatvecRunner>>,
    pub compile_duration: Duration,
    pub weight_upload_duration: Duration,
    pub pack_cache_hit: bool,
    pub gpu_cache_hit: bool,
}

pub(crate) struct ResidentPackedProjection {
    pub tensor_name: String,
    pub operation: String,
    pub rows: usize,
    pub cols: usize,
    pub tensor: GpuResidentBuffer,
    pub prepared: PreparedProjectionRunner,
    pub report: GpuPackedMatvecReport,
}

#[allow(dead_code)]
pub(crate) struct ResidentPackedPairProjection {
    pub tensor_name: String,
    pub first_rows: usize,
    pub second_rows: usize,
    pub cols: usize,
    pub tensor: GpuResidentBuffer,
    pub activation_upload_bytes: usize,
    pub prepared: PreparedProjectionRunner,
    pub report: GpuPackedMatvecReport,
}

pub(crate) struct ResidentGpuFinalNorm {
    #[allow(dead_code)]
    pub runner: Rc<RefCell<CachedGpuWeightedRmsNormRunner>>,
    pub tensor: GpuResidentBuffer,
    pub len: usize,
    pub report: GpuWeightedRmsNormReport,
    pub compile_duration: Duration,
    pub gpu_cache_hit: bool,
}

pub(crate) struct ResidentGpuVectorAdd {
    #[allow(dead_code)]
    pub runner: Rc<RefCell<CachedGpuVectorAddRunner>>,
    pub tensor: GpuResidentBuffer,
    pub len: usize,
    pub report: GpuVectorAddReport,
    pub compile_duration: Duration,
    pub gpu_cache_hit: bool,
}

pub(crate) enum ResidentGpuPackedActivationKeepalive {
    PackF16(Rc<RefCell<CachedGpuPackF16PairsRunner>>),
    SwigluPackF16(Rc<RefCell<CachedGpuSwigluPackF16PairsRunner>>),
}

pub(crate) struct ResidentGpuPackedActivation {
    #[allow(dead_code)]
    pub keepalive: ResidentGpuPackedActivationKeepalive,
    pub tensor: GpuResidentBuffer,
    pub logical_len: usize,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub compile_duration: Duration,
    pub gpu_cache_hit: bool,
}

#[allow(dead_code)]
pub(crate) struct ResidentGpuSwigluCombined {
    #[allow(dead_code)]
    pub runner: Rc<RefCell<CachedGpuSwigluCombinedRunner>>,
    pub tensor: GpuResidentBuffer,
    pub len: usize,
    pub report: GpuSwigluCombinedReport,
    pub compile_duration: Duration,
    pub gpu_cache_hit: bool,
}
