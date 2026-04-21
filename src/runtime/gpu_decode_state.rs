use crate::runtime::gpu_decode_projection_state::ResidentGpuVectorAdd;
use ash::vk;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentHiddenState {
    Mlp { layer_idx: usize },
}

#[derive(Default)]
pub(crate) struct PackedResidentDecodeState {
    pub resident_hidden_state: Option<ResidentHiddenState>,
    pub final_hidden_gpu: Option<ResidentGpuVectorAdd>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuTailResult {
    NextToken(usize),
    Logits(Vec<f32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuTailStepReport {
    pub final_norm_gpu_duration: Duration,
    pub pack_gpu_duration: Duration,
    pub logits_gpu_duration: Duration,
    pub logits_download_duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuKvBinding {
    pub key_buffer: vk::Buffer,
    pub key_len: usize,
    pub key_buffer_size: u64,
    pub value_buffer: vk::Buffer,
    pub value_len: usize,
    pub value_buffer_size: u64,
}
