use ash::util::read_spv;
use ash::{Device, Instance, vk};
use std::ffi::CString;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::gpu::packed_matvec::SharedGpuPackedContext;
use crate::gpu::resident_buffer::GpuResidentBuffer;

#[derive(Debug, Clone, PartialEq)]
pub struct GpuEmbeddingLookupReport {
    pub hidden: usize,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
}

#[derive(Debug)]
pub enum GpuEmbeddingLookupError {
    Io(std::io::Error),
    Vk(vk::Result),
    Utf8(std::str::Utf8Error),
    Process(String),
    CString(std::ffi::NulError),
    Shape(String),
}

impl std::fmt::Display for GpuEmbeddingLookupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Vk(error) => write!(f, "Vulkan error: {error:?}"),
            Self::Utf8(error) => write!(f, "UTF-8 error: {error}"),
            Self::Process(error) => write!(f, "process error: {error}"),
            Self::CString(error) => write!(f, "CString error: {error}"),
            Self::Shape(message) => write!(f, "shape error: {message}"),
        }
    }
}

impl std::error::Error for GpuEmbeddingLookupError {}
impl From<std::io::Error> for GpuEmbeddingLookupError {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<vk::Result> for GpuEmbeddingLookupError {
    fn from(v: vk::Result) -> Self {
        Self::Vk(v)
    }
}
impl From<std::str::Utf8Error> for GpuEmbeddingLookupError {
    fn from(v: std::str::Utf8Error) -> Self {
        Self::Utf8(v)
    }
}
impl From<std::ffi::NulError> for GpuEmbeddingLookupError {
    fn from(v: std::ffi::NulError) -> Self {
        Self::CString(v)
    }
}

pub struct CachedGpuEmbeddingLookupRunner {
    _shared_context: Arc<SharedGpuPackedContext>,
    device: Device,
    queue: vk::Queue,
    vocab: usize,
    hidden: usize,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    _descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    embeddings_buffer: BufferAllocation,
    token_buffer: BufferAllocation,
    output_buffer: BufferAllocation,
}

impl CachedGpuEmbeddingLookupRunner {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        embedding_words: &[u32],
        vocab: usize,
        hidden: usize,
    ) -> Result<(Self, Duration), GpuEmbeddingLookupError> {
        let compile_started = Instant::now();
        let instance = context.instance.clone();
        let device = context.device.clone();
        let queue = context.queue;
        let physical_device = context.physical_device;
        let queue_family_index = context.queue_family_index;
        let shader_path = compile_shader(Path::new("shaders/embedding_lookup_f16.comp"))?;

        let embeddings_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (embedding_words.len() * std::mem::size_of::<u32>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let token_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            std::mem::size_of::<u32>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let output_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (hidden * std::mem::size_of::<f32>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        write_u32_buffer(&embeddings_buffer, embedding_words)?;
        write_u32_buffer(&token_buffer, &[0])?;
        zero_buffer(&output_buffer, hidden * std::mem::size_of::<f32>())?;

        let descriptor_layout_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let descriptor_layout_ci =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_layout_bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_layout_ci, None)? };
        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(4)];
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_ci, None)? };
        let shader_module = create_shader_module(&device, &shader_path)?;
        let entry_name = CString::new("main")?;
        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_name);
        let compute_ci = [vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(pipeline_layout)];
        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &compute_ci, None)
                .map_err(|(_, err)| GpuEmbeddingLookupError::Vk(err))?[0]
        };
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let descriptor_pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_ci, None)? };
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };
        let embeddings_info = [vk::DescriptorBufferInfo::default()
            .buffer(embeddings_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let token_info = [vk::DescriptorBufferInfo::default()
            .buffer(token_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let output_info = [vk::DescriptorBufferInfo::default()
            .buffer(output_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&embeddings_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&token_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_ci, None)? };
        let command_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&command_alloc)?[0] };
        unsafe {
            device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            let push_constants = [hidden as u32];
            let push_bytes = std::slice::from_raw_parts(push_constants.as_ptr() as *const u8, 4);
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(command_buffer, hidden.div_ceil(64) as u32, 1, 1);
            device.end_command_buffer(command_buffer)?;
        }
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        Ok((
            Self {
                _shared_context: context,
                device,
                queue,
                vocab,
                hidden,
                descriptor_set_layout,
                pipeline_layout,
                shader_module,
                pipeline,
                descriptor_pool,
                _descriptor_set: descriptor_set,
                command_pool,
                command_buffer,
                fence,
                embeddings_buffer,
                token_buffer,
                output_buffer,
            },
            compile_started.elapsed(),
        ))
    }

    pub fn run_with_output(
        &mut self,
        token_id: usize,
    ) -> Result<(Vec<f32>, GpuEmbeddingLookupReport), GpuEmbeddingLookupError> {
        let mut report = self.run_resident(token_id)?;
        let download_started = Instant::now();
        let output = read_f32_buffer(&self.output_buffer, self.hidden)?;
        report.download_duration = download_started.elapsed();
        Ok((output, report))
    }

    pub fn run_resident(
        &mut self,
        token_id: usize,
    ) -> Result<GpuEmbeddingLookupReport, GpuEmbeddingLookupError> {
        if token_id >= self.vocab {
            return Err(GpuEmbeddingLookupError::Shape(format!(
                "token_id {} out of range for vocab {}",
                token_id, self.vocab
            )));
        }
        let upload_started = Instant::now();
        write_u32_buffer(&self.token_buffer, &[token_id as u32])?;
        let upload_duration = upload_started.elapsed();
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuEmbeddingLookupReport {
            hidden: self.hidden,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
        })
    }

    pub fn shared_context(&self) -> &Arc<SharedGpuPackedContext> {
        &self._shared_context
    }

    pub fn output_buffer_handle(&self) -> vk::Buffer {
        self.output_buffer.buffer
    }

    pub fn output_buffer_size(&self) -> u64 {
        self.output_buffer.size
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn resident_output(&self) -> GpuResidentBuffer {
        GpuResidentBuffer::new(
            self.shared_context().clone(),
            self.output_buffer_handle(),
            self.hidden(),
            self.output_buffer_size(),
        )
    }

    fn submit_and_wait(&self) -> Result<Duration, GpuEmbeddingLookupError> {
        unsafe { self.device.reset_fences(&[self.fence])? };
        let submit_info = [
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer))
        ];
        let gpu_started = Instant::now();
        unsafe {
            self.device
                .queue_submit(self.queue, &submit_info, self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        Ok(gpu_started.elapsed())
    }
}

impl Drop for CachedGpuEmbeddingLookupRunner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            destroy_buffer(&self.device, self.embeddings_buffer);
            destroy_buffer(&self.device, self.token_buffer);
            destroy_buffer(&self.device, self.output_buffer);
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

fn compile_shader(path: &Path) -> Result<PathBuf, GpuEmbeddingLookupError> {
    let temp_dir = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".tmp"));
    std::fs::create_dir_all(&temp_dir)?;
    let shader_stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("embedding-lookup");
    let out = temp_dir.join(format!("{shader_stem}.spv"));
    let tmp = temp_dir.join(format!(
        "{shader_stem}-{}-{}.tmp.spv",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let output = Command::new("glslc")
        .arg(path)
        .arg("--target-spv=spv1.3")
        .arg("-o")
        .arg(&tmp)
        .output()?;
    if !output.status.success() {
        return Err(GpuEmbeddingLookupError::Process(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    std::fs::rename(&tmp, &out)?;
    Ok(out)
}

fn create_shader_module(
    device: &Device,
    path: &Path,
) -> Result<vk::ShaderModule, GpuEmbeddingLookupError> {
    let mut cursor = Cursor::new(std::fs::read(path)?);
    let code = read_spv(&mut cursor)?;
    let ci = vk::ShaderModuleCreateInfo::default().code(&code);
    Ok(unsafe { device.create_shader_module(&ci, None)? })
}

fn create_host_visible_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<BufferAllocation, GpuEmbeddingLookupError> {
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

fn destroy_buffer(device: &Device, alloc: BufferAllocation) {
    unsafe {
        device.unmap_memory(alloc.memory);
        device.destroy_buffer(alloc.buffer, None);
        device.free_memory(alloc.memory, None);
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    filter: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, GpuEmbeddingLookupError> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize].property_flags.contains(flags);
        if matches_type && has_flags {
            return Ok(i);
        }
    }
    Err(GpuEmbeddingLookupError::Shape(
        "missing compatible Vulkan memory type".to_string(),
    ))
}

fn write_u32_buffer(buffer: &BufferAllocation, data: &[u32]) -> Result<(), GpuEmbeddingLookupError> {
    let byte_len = std::mem::size_of_val(data) as u64;
    if byte_len > buffer.size {
        return Err(GpuEmbeddingLookupError::Shape(format!(
            "u32 write {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buffer.mapped_ptr,
            byte_len as usize,
        );
    }
    Ok(())
}

fn zero_buffer(buffer: &BufferAllocation, byte_len: usize) -> Result<(), GpuEmbeddingLookupError> {
    if byte_len as u64 > buffer.size {
        return Err(GpuEmbeddingLookupError::Shape(format!(
            "zero {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    unsafe {
        std::ptr::write_bytes(buffer.mapped_ptr, 0, byte_len);
    }
    Ok(())
}

fn read_f32_buffer(buffer: &BufferAllocation, len: usize) -> Result<Vec<f32>, GpuEmbeddingLookupError> {
    let byte_len = len * std::mem::size_of::<f32>();
    if byte_len as u64 > buffer.size {
        return Err(GpuEmbeddingLookupError::Shape(format!(
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
