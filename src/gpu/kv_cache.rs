use std::sync::Arc;

use ash::{Device, Instance, vk};

use crate::gpu::packed_matvec::SharedGpuPackedContext;
use crate::gpu::resident_buffer::GpuResidentBuffer;

#[derive(Debug)]
pub enum GpuKvCacheError {
    Vk(vk::Result),
    MissingMemoryType,
    Shape(String),
}

impl std::fmt::Display for GpuKvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vk(error) => write!(f, "Vulkan error: {error:?}"),
            Self::MissingMemoryType => f.write_str("missing compatible Vulkan memory type"),
            Self::Shape(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for GpuKvCacheError {}

impl From<vk::Result> for GpuKvCacheError {
    fn from(value: vk::Result) -> Self {
        Self::Vk(value)
    }
}

pub struct GpuKvCache {
    _shared_context: Arc<SharedGpuPackedContext>,
    device: Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    tokens_capacity: usize,
    kv_width: usize,
    len_tokens: usize,
    key_buffer: BufferAllocation,
    value_buffer: BufferAllocation,
}

impl GpuKvCache {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        tokens_capacity: usize,
        kv_width: usize,
    ) -> Result<Self, GpuKvCacheError> {
        if tokens_capacity == 0 || kv_width == 0 {
            return Err(GpuKvCacheError::Shape(
                "kv cache capacity and width must be non-zero".to_string(),
            ));
        }
        let instance = context.instance.clone();
        let device = context.device.clone();
        let queue = context.queue;
        let queue_family_index = context.queue_family_index;
        let physical_device = context.physical_device;
        let bytes = (tokens_capacity * kv_width * std::mem::size_of::<f32>()) as u64;
        let key_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let value_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        zero_buffer(&key_buffer, bytes as usize)?;
        zero_buffer(&value_buffer, bytes as usize)?;
        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_ci, None)? };
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        Ok(Self {
            _shared_context: context,
            device,
            queue,
            command_pool,
            fence,
            tokens_capacity,
            kv_width,
            len_tokens: 0,
            key_buffer,
            value_buffer,
        })
    }

    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), GpuKvCacheError> {
        if key.len() != self.kv_width || value.len() != self.kv_width {
            return Err(GpuKvCacheError::Shape(format!(
                "kv append expects key/value width {} but got {}/{}",
                self.kv_width,
                key.len(),
                value.len()
            )));
        }
        if self.len_tokens >= self.tokens_capacity {
            return Err(GpuKvCacheError::Shape(format!(
                "kv cache capacity {} exceeded",
                self.tokens_capacity
            )));
        }
        let offset_bytes = self.len_tokens * self.kv_width * std::mem::size_of::<f32>();
        write_f32_buffer_at(&self.key_buffer, offset_bytes, key)?;
        write_f32_buffer_at(&self.value_buffer, offset_bytes, value)?;
        self.len_tokens += 1;
        Ok(())
    }

    pub fn append_key_from_tensor_and_value_host(
        &mut self,
        key: &GpuResidentBuffer,
        value: &[f32],
    ) -> Result<(), GpuKvCacheError> {
        if key.len != self.kv_width || value.len() != self.kv_width {
            return Err(GpuKvCacheError::Shape(format!(
                "kv append expects key/value width {} but got {}/{}",
                self.kv_width,
                key.len,
                value.len()
            )));
        }
        if self.len_tokens >= self.tokens_capacity {
            return Err(GpuKvCacheError::Shape(format!(
                "kv cache capacity {} exceeded",
                self.tokens_capacity
            )));
        }
        if !Arc::ptr_eq(&self._shared_context, &key.shared_context) {
            return Err(GpuKvCacheError::Shape(
                "resident key append requires matching Vulkan context".to_string(),
            ));
        }
        let offset_bytes = self.len_tokens * self.kv_width * std::mem::size_of::<f32>();
        self.copy_into_key_slot(key.buffer, key.buffer_size, offset_bytes)?;
        write_f32_buffer_at(&self.value_buffer, offset_bytes, value)?;
        self.len_tokens += 1;
        Ok(())
    }

    pub fn len_tokens(&self) -> usize {
        self.len_tokens
    }

    pub fn kv_width(&self) -> usize {
        self.kv_width
    }

    pub fn key_buffer_handle(&self) -> vk::Buffer {
        self.key_buffer.buffer
    }

    pub fn value_buffer_handle(&self) -> vk::Buffer {
        self.value_buffer.buffer
    }

    pub fn key_buffer_size(&self) -> u64 {
        self.key_buffer.size
    }

    pub fn value_buffer_size(&self) -> u64 {
        self.value_buffer.size
    }

    pub fn snapshot_keys(&self) -> Result<Vec<f32>, GpuKvCacheError> {
        read_f32_prefix(&self.key_buffer, self.len_tokens * self.kv_width)
    }

    pub fn snapshot_values(&self) -> Result<Vec<f32>, GpuKvCacheError> {
        read_f32_prefix(&self.value_buffer, self.len_tokens * self.kv_width)
    }

    fn copy_into_key_slot(
        &self,
        source_buffer: vk::Buffer,
        source_buffer_size: u64,
        offset_bytes: usize,
    ) -> Result<(), GpuKvCacheError> {
        let byte_len = self.kv_width * std::mem::size_of::<f32>();
        let end = offset_bytes + byte_len;
        if byte_len as u64 > source_buffer_size || end as u64 > self.key_buffer.size {
            return Err(GpuKvCacheError::Shape(format!(
                "kv key copy [{}..{}) exceeds source {} or destination {}",
                offset_bytes, end, source_buffer_size, self.key_buffer.size
            )));
        }
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let copy_command = unsafe { self.device.allocate_command_buffers(&alloc)?[0] };
        unsafe {
            self.device
                .begin_command_buffer(copy_command, &vk::CommandBufferBeginInfo::default())?;
            let region = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(offset_bytes as u64)
                .size(byte_len as u64)];
            self.device
                .cmd_copy_buffer(copy_command, source_buffer, self.key_buffer.buffer, &region);
            self.device.end_command_buffer(copy_command)?;
            self.device.reset_fences(&[self.fence])?;
        }
        let submit_info =
            [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&copy_command))];
        unsafe {
            self.device
                .queue_submit(self.queue, &submit_info, self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device
                .free_command_buffers(self.command_pool, &[copy_command]);
        }
        Ok(())
    }
}

impl Drop for GpuKvCache {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            destroy_buffer(&self.device, self.key_buffer);
            destroy_buffer(&self.device, self.value_buffer);
        }
    }
}

#[derive(Clone, Copy)]
struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_ptr: *mut u8,
    size: u64,
}

fn create_host_visible_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<BufferAllocation, GpuKvCacheError> {
    let ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&ci, None)? };
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_type_index = find_memory_type(
        &memory_props,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { device.allocate_memory(&alloc, None)? };
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
    let mapped_ptr =
        unsafe { device.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())? }
            as *mut u8;
    Ok(BufferAllocation {
        buffer,
        memory,
        mapped_ptr,
        size: requirements.size,
    })
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    filter: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, GpuKvCacheError> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize].property_flags.contains(flags);
        if matches_type && has_flags {
            return Ok(i);
        }
    }
    Err(GpuKvCacheError::MissingMemoryType)
}

fn write_f32_buffer_at(
    buffer: &BufferAllocation,
    offset_bytes: usize,
    data: &[f32],
) -> Result<(), GpuKvCacheError> {
    let byte_len = std::mem::size_of_val(data);
    let end = offset_bytes + byte_len;
    if end as u64 > buffer.size {
        return Err(GpuKvCacheError::Shape(format!(
            "f32 write [{}..{}) exceeds mapped buffer size {}",
            offset_bytes, end, buffer.size
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buffer.mapped_ptr.add(offset_bytes),
            byte_len,
        );
    }
    Ok(())
}

fn read_f32_prefix(buffer: &BufferAllocation, len: usize) -> Result<Vec<f32>, GpuKvCacheError> {
    let byte_len = len * std::mem::size_of::<f32>();
    if byte_len as u64 > buffer.size {
        return Err(GpuKvCacheError::Shape(format!(
            "f32 read {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    let mut output = vec![0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.mapped_ptr,
            output.as_mut_ptr() as *mut u8,
            byte_len,
        );
    }
    Ok(output)
}

fn zero_buffer(buffer: &BufferAllocation, byte_len: usize) -> Result<(), GpuKvCacheError> {
    if byte_len as u64 > buffer.size {
        return Err(GpuKvCacheError::Shape(format!(
            "zero {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    unsafe {
        std::ptr::write_bytes(buffer.mapped_ptr, 0, byte_len);
    }
    Ok(())
}

unsafe fn destroy_buffer(device: &Device, allocation: BufferAllocation) {
    unsafe {
        device.unmap_memory(allocation.memory);
        device.destroy_buffer(allocation.buffer, None);
        device.free_memory(allocation.memory, None);
    }
}
