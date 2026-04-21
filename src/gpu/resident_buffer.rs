use std::sync::Arc;

use ash::vk;

use crate::gpu::packed_matvec::SharedGpuPackedContext;

#[derive(Clone)]
pub struct GpuResidentBuffer {
    pub shared_context: Arc<SharedGpuPackedContext>,
    pub buffer: vk::Buffer,
    pub len: usize,
    pub buffer_size: u64,
}

impl GpuResidentBuffer {
    pub fn new(
        shared_context: Arc<SharedGpuPackedContext>,
        buffer: vk::Buffer,
        len: usize,
        buffer_size: u64,
    ) -> Self {
        Self {
            shared_context,
            buffer,
            len,
            buffer_size,
        }
    }
}
