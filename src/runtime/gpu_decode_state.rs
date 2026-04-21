use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentHiddenState {
    Mlp { layer_idx: usize },
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
