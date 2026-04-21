use ash::util::read_spv;
use ash::{vk, Device, Instance};
use half::f16;
use std::ffi::CString;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::gpu::packed_matvec::SharedGpuPackedContext;

#[derive(Debug, Clone, PartialEq)]
pub struct GpuPackedPairSwigluPackReport {
    pub rows: usize,
    pub cols: usize,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub mismatched_words: usize,
}

#[derive(Debug)]
pub enum GpuPackedPairSwigluPackError {
    Io(std::io::Error),
    Vk(vk::Result),
    Load(ash::LoadingError),
    Utf8(std::str::Utf8Error),
    Process(String),
    CString(std::ffi::NulError),
    Shape(String),
}

impl std::fmt::Display for GpuPackedPairSwigluPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Vk(error) => write!(f, "Vulkan error: {error:?}"),
            Self::Load(error) => write!(f, "Vulkan load error: {error}"),
            Self::Utf8(error) => write!(f, "UTF-8 error: {error}"),
            Self::Process(error) => write!(f, "process error: {error}"),
            Self::CString(error) => write!(f, "CString error: {error}"),
            Self::Shape(message) => write!(f, "shape error: {message}"),
        }
    }
}
impl std::error::Error for GpuPackedPairSwigluPackError {}
impl From<std::io::Error> for GpuPackedPairSwigluPackError { fn from(v: std::io::Error) -> Self { Self::Io(v) } }
impl From<vk::Result> for GpuPackedPairSwigluPackError { fn from(v: vk::Result) -> Self { Self::Vk(v) } }
impl From<ash::LoadingError> for GpuPackedPairSwigluPackError { fn from(v: ash::LoadingError) -> Self { Self::Load(v) } }
impl From<std::str::Utf8Error> for GpuPackedPairSwigluPackError { fn from(v: std::str::Utf8Error) -> Self { Self::Utf8(v) } }
impl From<std::ffi::NulError> for GpuPackedPairSwigluPackError { fn from(v: std::ffi::NulError) -> Self { Self::CString(v) } }

pub fn run_packed_pair_swiglu_pack_with_output(
    code_words: &[u32],
    scales: &[f32],
    group_size: usize,
    rows: usize,
    cols: usize,
    input: &[f32],
    reference: Option<&[u32]>,
) -> Result<(Vec<u32>, GpuPackedPairSwigluPackReport), GpuPackedPairSwigluPackError> {
    let context = SharedGpuPackedContext::new().map_err(map_context_error)?;
    let (mut runner, compile_duration) = CachedGpuPackedPairSwigluPackRunner::new_with_context(
        context,
        code_words,
        scales,
        group_size,
        rows,
        cols,
    )?;
    let (output, mut report) = runner.run_with_output(input, reference)?;
    report.compile_duration = compile_duration;
    Ok((output, report))
}

pub struct CachedGpuPackedPairSwigluPackRunner {
    _shared_context: Arc<SharedGpuPackedContext>,
    device: Device,
    queue: vk::Queue,
    rows: usize,
    cols: usize,
    group_size: usize,
    packed_cols: usize,
    code_buffer: BufferAllocation,
    scale_buffer: BufferAllocation,
    vector_buffer: BufferAllocation,
    output_buffer: BufferAllocation,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    _descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl CachedGpuPackedPairSwigluPackRunner {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        code_words: &[u32],
        scales: &[f32],
        group_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, Duration), GpuPackedPairSwigluPackError> {
        let compile_started = Instant::now();
        let instance = context.instance.clone();
        let device = context.device.clone();
        let queue = context.queue;
        let physical_device = context.physical_device;
        let queue_family_index = context.queue_family_index;
        let packed_cols = cols.div_ceil(2);
        let packed_rows = rows.div_ceil(2);
        let shader_path = compile_shader(Path::new("shaders/packed_ternary_pair_swiglu_pack.comp"))?;

        let code_buffer = create_host_visible_buffer(&instance, &device, physical_device, (code_words.len() * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let scale_buffer = create_host_visible_buffer(&instance, &device, physical_device, (scales.len() * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let vector_buffer = create_host_visible_buffer(&instance, &device, physical_device, (packed_cols * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let output_buffer = create_host_visible_buffer(&instance, &device, physical_device, (packed_rows * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        write_u32_buffer(&code_buffer, code_words)?;
        write_f32_buffer(&scale_buffer, scales)?;
        zero_buffer(&vector_buffer, packed_cols * 4)?;
        zero_buffer(&output_buffer, packed_rows * 4)?;

        let descriptor_layout_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_count(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_count(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_count(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_count(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let descriptor_layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_layout_bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&descriptor_layout_ci, None)? };
        let push_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push_range);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_ci, None)? };
        let shader_module = create_shader_module(&device, &shader_path)?;
        let entry_name = CString::new("main")?;
        let stage_ci = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader_module).name(&entry_name);
        let compute_ci = [vk::ComputePipelineCreateInfo::default().stage(stage_ci).layout(pipeline_layout)];
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &compute_ci, None).map_err(|(_, err)| GpuPackedPairSwigluPackError::Vk(err))?[0] };
        let pool_sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(4)];
        let descriptor_pool_ci = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_ci, None)? };
        let alloc_info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&set_layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };
        let code_info = [vk::DescriptorBufferInfo::default().buffer(code_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let scale_info = [vk::DescriptorBufferInfo::default().buffer(scale_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let vector_info = [vk::DescriptorBufferInfo::default().buffer(vector_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let output_info = [vk::DescriptorBufferInfo::default().buffer(output_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&code_info),
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&scale_info),
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&vector_info),
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&output_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        let command_pool_ci = vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_ci, None)? };
        let command_alloc = vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&command_alloc)?[0] };
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let runner = Self {
            _shared_context: context,
            device,
            queue,
            rows,
            cols,
            group_size,
            packed_cols,
            code_buffer,
            scale_buffer,
            vector_buffer,
            output_buffer,
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
            descriptor_pool,
            _descriptor_set: descriptor_set,
            command_pool,
            command_buffer,
            fence,
        };
        runner.record_command_buffer()?;
        Ok((runner, compile_started.elapsed()))
    }

    pub fn run_with_output(
        &mut self,
        input: &[f32],
        reference: Option<&[u32]>,
    ) -> Result<(Vec<u32>, GpuPackedPairSwigluPackReport), GpuPackedPairSwigluPackError> {
        if input.len() != self.cols {
            return Err(GpuPackedPairSwigluPackError::Shape(format!("input len {} does not match cols {}", input.len(), self.cols)));
        }
        let vector_words = pack_f16_pairs(input);
        let upload_started = Instant::now();
        write_u32_buffer(&self.vector_buffer, &vector_words)?;
        let upload_duration = upload_started.elapsed();
        unsafe { self.device.reset_fences(&[self.fence])?; }
        let submit_info = [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer))];
        let gpu_started = Instant::now();
        unsafe {
            self.device.queue_submit(self.queue, &submit_info, self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        let gpu_duration = gpu_started.elapsed();
        let download_started = Instant::now();
        let output = read_u32_buffer(&self.output_buffer, self.rows.div_ceil(2))?;
        let download_duration = download_started.elapsed();
        let mismatched = reference.map(|r| r.iter().zip(output.iter()).filter(|(a,b)| a != b).count()).unwrap_or(0);
        Ok((output, GpuPackedPairSwigluPackReport { rows: self.rows, cols: self.cols, compile_duration: Duration::ZERO, upload_duration, gpu_duration, download_duration, mismatched_words: mismatched }))
    }

    fn record_command_buffer(&self) -> Result<(), GpuPackedPairSwigluPackError> {
        unsafe {
            self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(self.command_buffer, &vk::CommandBufferBeginInfo::default())?;
            self.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self._descriptor_set], &[]);
        }
        let push_constants = [self.rows as u32, self.cols as u32, self.group_size as u32, self.packed_cols as u32];
        let push_bytes = unsafe { std::slice::from_raw_parts(push_constants.as_ptr() as *const u8, push_constants.len() * 4) };
        unsafe {
            self.device.cmd_push_constants(self.command_buffer, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
            self.device.cmd_dispatch(self.command_buffer, self.rows.div_ceil(2).div_ceil(64) as u32, 1, 1);
            self.device.end_command_buffer(self.command_buffer)?;
        }
        Ok(())
    }
}

impl Drop for CachedGpuPackedPairSwigluPackRunner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            destroy_buffer(&self.device, self.code_buffer);
            destroy_buffer(&self.device, self.scale_buffer);
            destroy_buffer(&self.device, self.vector_buffer);
            destroy_buffer(&self.device, self.output_buffer);
        }
    }
}

#[derive(Clone, Copy)]
struct BufferAllocation { buffer: vk::Buffer, memory: vk::DeviceMemory, mapped_ptr: *mut u8, size: u64 }

fn compile_shader(path: &Path) -> Result<PathBuf, GpuPackedPairSwigluPackError> {
    let temp_dir = std::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(".tmp"));
    std::fs::create_dir_all(&temp_dir)?;
    let shader_stem = path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("packed-pair-swiglu-pack");
    let out = temp_dir.join(format!("{shader_stem}.spv"));
    let tmp = temp_dir.join(format!("{shader_stem}-{}-{}.tmp.spv", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()));
    let output = Command::new("glslc").arg(path).arg("--target-spv=spv1.3").arg("-o").arg(&tmp).output()?;
    if !output.status.success() { return Err(GpuPackedPairSwigluPackError::Process(String::from_utf8_lossy(&output.stderr).to_string())); }
    std::fs::rename(&tmp, &out)?;
    Ok(out)
}

fn create_shader_module(device: &Device, path: &Path) -> Result<vk::ShaderModule, GpuPackedPairSwigluPackError> {
    let mut cursor = Cursor::new(std::fs::read(path)?);
    let code = read_spv(&mut cursor)?;
    let ci = vk::ShaderModuleCreateInfo::default().code(&code);
    Ok(unsafe { device.create_shader_module(&ci, None)? })
}

fn create_host_visible_buffer(instance: &Instance, device: &Device, physical_device: vk::PhysicalDevice, size: u64, usage: vk::BufferUsageFlags) -> Result<BufferAllocation, GpuPackedPairSwigluPackError> {
    let ci = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&ci, None)? };
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_type_index = find_memory_type(&memory_props, requirements.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
    let alloc = vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(memory_type_index);
    let memory = unsafe { device.allocate_memory(&alloc, None)? };
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
    let mapped_ptr = unsafe { device.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())? } as *mut u8;
    Ok(BufferAllocation { buffer, memory, mapped_ptr, size: requirements.size })
}

fn find_memory_type(props: &vk::PhysicalDeviceMemoryProperties, filter: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, GpuPackedPairSwigluPackError> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize].property_flags.contains(flags);
        if matches_type && has_flags { return Ok(i); }
    }
    Err(GpuPackedPairSwigluPackError::Shape("no matching Vulkan memory type found".to_string()))
}

fn write_u32_buffer(buffer: &BufferAllocation, data: &[u32]) -> Result<(), GpuPackedPairSwigluPackError> {
    let byte_len = std::mem::size_of_val(data) as u64;
    if byte_len > buffer.size { return Err(GpuPackedPairSwigluPackError::Shape(format!("u32 write {} bytes exceeds mapped buffer size {}", byte_len, buffer.size))); }
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buffer.mapped_ptr, byte_len as usize); }
    Ok(())
}
fn write_f32_buffer(buffer: &BufferAllocation, data: &[f32]) -> Result<(), GpuPackedPairSwigluPackError> { let byte_len = std::mem::size_of_val(data) as u64; if byte_len > buffer.size { return Err(GpuPackedPairSwigluPackError::Shape(format!("f32 write {} bytes exceeds mapped buffer size {}", byte_len, buffer.size))); } unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buffer.mapped_ptr, byte_len as usize); } Ok(()) }
fn zero_buffer(buffer: &BufferAllocation, byte_len: usize) -> Result<(), GpuPackedPairSwigluPackError> { if byte_len as u64 > buffer.size { return Err(GpuPackedPairSwigluPackError::Shape(format!("zero {} bytes exceeds mapped buffer size {}", byte_len, buffer.size))); } unsafe { std::ptr::write_bytes(buffer.mapped_ptr, 0, byte_len); } Ok(()) }
fn read_u32_buffer(buffer: &BufferAllocation, len: usize) -> Result<Vec<u32>, GpuPackedPairSwigluPackError> { let mut out = vec![0u32; len]; unsafe { std::ptr::copy_nonoverlapping(buffer.mapped_ptr as *const u32, out.as_mut_ptr(), len); } Ok(out) }
fn destroy_buffer(device: &Device, buffer: BufferAllocation) { unsafe { device.unmap_memory(buffer.memory); device.destroy_buffer(buffer.buffer, None); device.free_memory(buffer.memory, None); } }
fn pack_f16_pairs(values: &[f32]) -> Vec<u32> { values.chunks(2).map(|chunk| { let a = f16::from_f32(chunk[0]).to_bits() as u32; let b = chunk.get(1).map(|v| f16::from_f32(*v).to_bits() as u32).unwrap_or(0); a | (b << 16) }).collect() }
fn map_context_error(error: crate::gpu::packed_matvec::GpuPackedMatvecError) -> GpuPackedPairSwigluPackError { match error { crate::gpu::packed_matvec::GpuPackedMatvecError::Io(err) => GpuPackedPairSwigluPackError::Io(err), crate::gpu::packed_matvec::GpuPackedMatvecError::Vk(err) => GpuPackedPairSwigluPackError::Vk(err), crate::gpu::packed_matvec::GpuPackedMatvecError::Load(err) => GpuPackedPairSwigluPackError::Load(err), crate::gpu::packed_matvec::GpuPackedMatvecError::Utf8(err) => GpuPackedPairSwigluPackError::Utf8(err), crate::gpu::packed_matvec::GpuPackedMatvecError::Process(err) => GpuPackedPairSwigluPackError::Process(err), crate::gpu::packed_matvec::GpuPackedMatvecError::CString(err) => GpuPackedPairSwigluPackError::CString(err), crate::gpu::packed_matvec::GpuPackedMatvecError::MissingDevice => GpuPackedPairSwigluPackError::Shape("no suitable Vulkan device found".to_string()), crate::gpu::packed_matvec::GpuPackedMatvecError::MissingQueue => GpuPackedPairSwigluPackError::Shape("no suitable compute queue family found".to_string()), crate::gpu::packed_matvec::GpuPackedMatvecError::Shape(msg) => GpuPackedPairSwigluPackError::Shape(msg), } }
