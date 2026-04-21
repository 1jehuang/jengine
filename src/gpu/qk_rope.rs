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
pub struct GpuQkRopeReport {
    pub query_len: usize,
    pub key_len: usize,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
}

#[derive(Debug)]
pub enum GpuQkRopeError {
    Io(std::io::Error),
    Vk(vk::Result),
    Utf8(std::str::Utf8Error),
    Process(String),
    CString(std::ffi::NulError),
    Shape(String),
}

impl std::fmt::Display for GpuQkRopeError {
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

impl std::error::Error for GpuQkRopeError {}
impl From<std::io::Error> for GpuQkRopeError {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<vk::Result> for GpuQkRopeError {
    fn from(v: vk::Result) -> Self {
        Self::Vk(v)
    }
}
impl From<std::str::Utf8Error> for GpuQkRopeError {
    fn from(v: std::str::Utf8Error) -> Self {
        Self::Utf8(v)
    }
}
impl From<std::ffi::NulError> for GpuQkRopeError {
    fn from(v: std::ffi::NulError) -> Self {
        Self::CString(v)
    }
}

pub struct CachedGpuQkRopeRunner {
    _shared_context: Arc<SharedGpuPackedContext>,
    device: Device,
    queue: vk::Queue,
    query_len: usize,
    key_len: usize,
    head_dim: usize,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    q_buffer: BufferAllocation,
    k_buffer: BufferAllocation,
    q_weight_buffer: BufferAllocation,
    k_weight_buffer: BufferAllocation,
    cos_buffer: BufferAllocation,
    sin_buffer: BufferAllocation,
    q_out_buffer: BufferAllocation,
    k_out_buffer: BufferAllocation,
}

impl CachedGpuQkRopeRunner {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        num_query_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) -> Result<(Self, Duration), GpuQkRopeError> {
        let compile_started = Instant::now();
        let instance = context.instance.clone();
        let device = context.device.clone();
        let queue = context.queue;
        let physical_device = context.physical_device;
        let queue_family_index = context.queue_family_index;
        let shader_path = compile_shader(Path::new("shaders/qk_norm_rope.comp"))?;

        let query_len = num_query_heads * head_dim;
        let key_len = num_key_value_heads * head_dim;
        let q_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (query_len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let k_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (key_len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let q_weight_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (head_dim * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let k_weight_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (head_dim * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let cos_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (head_dim * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let sin_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (head_dim * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let q_out_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (query_len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let k_out_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (key_len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        zero_buffer(&q_buffer, query_len * 4)?;
        zero_buffer(&k_buffer, key_len * 4)?;
        zero_buffer(&q_weight_buffer, head_dim * 4)?;
        zero_buffer(&k_weight_buffer, head_dim * 4)?;
        zero_buffer(&cos_buffer, head_dim * 4)?;
        zero_buffer(&sin_buffer, head_dim * 4)?;
        zero_buffer(&q_out_buffer, query_len * 4)?;
        zero_buffer(&k_out_buffer, key_len * 4)?;

        let descriptor_layout_bindings = [
            binding(0),
            binding(1),
            binding(2),
            binding(3),
            binding(4),
            binding(5),
            binding(6),
            binding(7),
        ];
        let descriptor_layout_ci =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_layout_bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_layout_ci, None)? };
        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(12)];
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
                .map_err(|(_, err)| GpuQkRopeError::Vk(err))?[0]
        };
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(8)];
        let descriptor_pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_ci, None)? };
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };
        let infos = [
            buffer_info(q_buffer.buffer),
            buffer_info(k_buffer.buffer),
            buffer_info(q_weight_buffer.buffer),
            buffer_info(k_weight_buffer.buffer),
            buffer_info(cos_buffer.buffer),
            buffer_info(sin_buffer.buffer),
            buffer_info(q_out_buffer.buffer),
            buffer_info(k_out_buffer.buffer),
        ];
        let writes = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect::<Vec<_>>();
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
            let push_constants = [
                num_query_heads as u32,
                num_key_value_heads as u32,
                head_dim as u32,
            ];
            let push_bytes = std::slice::from_raw_parts(push_constants.as_ptr() as *const u8, 12);
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(
                command_buffer,
                (num_query_heads + num_key_value_heads) as u32,
                1,
                1,
            );
            let output_barriers = [
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(q_out_buffer.buffer)
                    .offset(0)
                    .size(q_out_buffer.size),
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(k_out_buffer.buffer)
                    .offset(0)
                    .size(k_out_buffer.size),
            ];
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &output_barriers,
                &[],
            );
            device.end_command_buffer(command_buffer)?;
        }
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        Ok((
            Self {
                _shared_context: context,
                device,
                queue,
                query_len,
                key_len,
                head_dim,
                descriptor_set_layout,
                pipeline_layout,
                shader_module,
                pipeline,
                descriptor_pool,
                descriptor_set,
                command_pool,
                command_buffer,
                fence,
                q_buffer,
                k_buffer,
                q_weight_buffer,
                k_weight_buffer,
                cos_buffer,
                sin_buffer,
                q_out_buffer,
                k_out_buffer,
            },
            compile_started.elapsed(),
        ))
    }

    pub fn run_resident(
        &mut self,
        q: &[f32],
        k: &[f32],
        q_weight: &[f32],
        k_weight: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<GpuQkRopeReport, GpuQkRopeError> {
        if q.len() != self.query_len
            || k.len() != self.key_len
            || q_weight.len() != self.head_dim
            || k_weight.len() != self.head_dim
            || cos.len() != self.head_dim
            || sin.len() != self.head_dim
        {
            return Err(GpuQkRopeError::Shape(
                "qk rope input shapes must match runner dimensions".to_string(),
            ));
        }
        let upload_started = Instant::now();
        self.bind_input_buffers(self.q_buffer.buffer, self.k_buffer.buffer);
        write_f32_buffer(&self.q_buffer, q)?;
        write_f32_buffer(&self.k_buffer, k)?;
        write_f32_buffer(&self.q_weight_buffer, q_weight)?;
        write_f32_buffer(&self.k_weight_buffer, k_weight)?;
        write_f32_buffer(&self.cos_buffer, cos)?;
        write_f32_buffer(&self.sin_buffer, sin)?;
        let upload_duration = upload_started.elapsed();
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuQkRopeReport {
            query_len: self.query_len,
            key_len: self.key_len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
        })
    }

    pub fn run_resident_from_tensors(
        &mut self,
        q: &GpuResidentBuffer,
        k: &GpuResidentBuffer,
        q_weight: &[f32],
        k_weight: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<GpuQkRopeReport, GpuQkRopeError> {
        if q.len != self.query_len
            || k.len != self.key_len
            || q_weight.len() != self.head_dim
            || k_weight.len() != self.head_dim
            || cos.len() != self.head_dim
            || sin.len() != self.head_dim
        {
            return Err(GpuQkRopeError::Shape(
                "qk rope tensor input shapes must match runner dimensions".to_string(),
            ));
        }
        let upload_started = Instant::now();
        if !Arc::ptr_eq(&self._shared_context, &q.shared_context)
            || !Arc::ptr_eq(&self._shared_context, &k.shared_context)
        {
            return Err(GpuQkRopeError::Shape(
                "resident qk-rope chaining requires matching Vulkan context".to_string(),
            ));
        }
        let q_byte_len = self.query_len * std::mem::size_of::<f32>();
        let k_byte_len = self.key_len * std::mem::size_of::<f32>();
        if q.buffer_size < q_byte_len as u64 || k.buffer_size < k_byte_len as u64 {
            return Err(GpuQkRopeError::Shape(
                "resident qk-rope source buffers are smaller than required".to_string(),
            ));
        }
        self.bind_input_buffers(q.buffer, k.buffer);
        write_f32_buffer(&self.q_weight_buffer, q_weight)?;
        write_f32_buffer(&self.k_weight_buffer, k_weight)?;
        write_f32_buffer(&self.cos_buffer, cos)?;
        write_f32_buffer(&self.sin_buffer, sin)?;
        let upload_duration = upload_started.elapsed();
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuQkRopeReport {
            query_len: self.query_len,
            key_len: self.key_len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
        })
    }

    pub fn run_with_output(
        &mut self,
        q: &[f32],
        k: &[f32],
        q_weight: &[f32],
        k_weight: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<((Vec<f32>, Vec<f32>), GpuQkRopeReport), GpuQkRopeError> {
        let mut report = self.run_resident(q, k, q_weight, k_weight, cos, sin)?;
        let download_started = Instant::now();
        let q_out = read_f32_buffer(&self.q_out_buffer, self.query_len)?;
        let k_out = read_f32_buffer(&self.k_out_buffer, self.key_len)?;
        report.download_duration = download_started.elapsed();
        Ok(((q_out, k_out), report))
    }

    pub fn read_query_output(&self) -> Result<(Vec<f32>, Duration), GpuQkRopeError> {
        let download_started = Instant::now();
        let q_out = read_f32_buffer(&self.q_out_buffer, self.query_len)?;
        Ok((q_out, download_started.elapsed()))
    }

    pub fn query_resident_output(&self) -> GpuResidentBuffer {
        GpuResidentBuffer::new(
            self._shared_context.clone(),
            self.q_out_buffer.buffer,
            self.query_len,
            self.q_out_buffer.size,
        )
    }

    pub fn key_resident_output(&self) -> GpuResidentBuffer {
        GpuResidentBuffer::new(
            self._shared_context.clone(),
            self.k_out_buffer.buffer,
            self.key_len,
            self.k_out_buffer.size,
        )
    }

    fn submit_and_wait(&self) -> Result<Duration, GpuQkRopeError> {
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

    fn bind_input_buffers(&self, q_buffer: vk::Buffer, k_buffer: vk::Buffer) {
        let infos = [
            buffer_info(q_buffer),
            buffer_info(k_buffer),
            buffer_info(self.q_weight_buffer.buffer),
            buffer_info(self.k_weight_buffer.buffer),
            buffer_info(self.cos_buffer.buffer),
            buffer_info(self.sin_buffer.buffer),
            buffer_info(self.q_out_buffer.buffer),
            buffer_info(self.k_out_buffer.buffer),
        ];
        let writes = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect::<Vec<_>>();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }
}

impl Drop for CachedGpuQkRopeRunner {
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
            destroy_buffer(&self.device, self.q_buffer);
            destroy_buffer(&self.device, self.k_buffer);
            destroy_buffer(&self.device, self.q_weight_buffer);
            destroy_buffer(&self.device, self.k_weight_buffer);
            destroy_buffer(&self.device, self.cos_buffer);
            destroy_buffer(&self.device, self.sin_buffer);
            destroy_buffer(&self.device, self.q_out_buffer);
            destroy_buffer(&self.device, self.k_out_buffer);
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

fn binding(index: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(index)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn buffer_info(buffer: vk::Buffer) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(vk::WHOLE_SIZE)
}

fn compile_shader(path: &Path) -> Result<PathBuf, GpuQkRopeError> {
    let temp_dir = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".tmp"));
    std::fs::create_dir_all(&temp_dir)?;
    let shader_stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("qk-rope");
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
        return Err(GpuQkRopeError::Process(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    std::fs::rename(&tmp, &out)?;
    Ok(out)
}

fn create_shader_module(device: &Device, path: &Path) -> Result<vk::ShaderModule, GpuQkRopeError> {
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
) -> Result<BufferAllocation, GpuQkRopeError> {
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
) -> Result<u32, GpuQkRopeError> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize]
            .property_flags
            .contains(flags);
        if matches_type && has_flags {
            return Ok(i);
        }
    }
    Err(GpuQkRopeError::Shape(
        "missing compatible Vulkan memory type".to_string(),
    ))
}

fn write_f32_buffer(buffer: &BufferAllocation, data: &[f32]) -> Result<(), GpuQkRopeError> {
    let byte_len = std::mem::size_of_val(data) as u64;
    if byte_len > buffer.size {
        return Err(GpuQkRopeError::Shape(format!(
            "f32 write {} bytes exceeds mapped buffer size {}",
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

fn zero_buffer(buffer: &BufferAllocation, byte_len: usize) -> Result<(), GpuQkRopeError> {
    if byte_len as u64 > buffer.size {
        return Err(GpuQkRopeError::Shape(format!(
            "zero {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    unsafe {
        std::ptr::write_bytes(buffer.mapped_ptr, 0, byte_len);
    }
    Ok(())
}

fn read_f32_buffer(buffer: &BufferAllocation, len: usize) -> Result<Vec<f32>, GpuQkRopeError> {
    let byte_len = len * std::mem::size_of::<f32>();
    if byte_len as u64 > buffer.size {
        return Err(GpuQkRopeError::Shape(format!(
            "f32 read {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    let mut output = vec![0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.mapped_ptr, output.as_mut_ptr() as *mut u8, byte_len);
    }
    Ok(output)
}

fn destroy_buffer(device: &Device, alloc: BufferAllocation) {
    unsafe {
        device.unmap_memory(alloc.memory);
        device.destroy_buffer(alloc.buffer, None);
        device.free_memory(alloc.memory, None);
    }
}
