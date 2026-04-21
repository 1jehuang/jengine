use ash::util::read_spv;
use ash::{Device, Instance, vk};
use std::ffi::CString;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::gpu::packed_matvec::SharedGpuPackedContext;

#[derive(Debug, Clone, PartialEq)]
pub struct GpuWeightedRmsNormReport {
    pub len: usize,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

impl GpuWeightedRmsNormReport {
    pub fn summarize(&self) -> String {
        format!(
            "len={} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
            self.len,
            self.compile_duration.as_secs_f64() * 1_000.0,
            self.upload_duration.as_secs_f64() * 1_000.0,
            self.gpu_duration.as_secs_f64() * 1_000.0,
            self.download_duration.as_secs_f64() * 1_000.0,
            self.max_abs_diff,
            self.mean_abs_diff,
        )
    }
}

#[derive(Debug)]
pub enum GpuWeightedRmsNormError {
    Io(std::io::Error),
    Vk(vk::Result),
    Load(ash::LoadingError),
    Utf8(std::str::Utf8Error),
    Process(String),
    CString(std::ffi::NulError),
    Shape(String),
}

impl std::fmt::Display for GpuWeightedRmsNormError {
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

impl std::error::Error for GpuWeightedRmsNormError {}
impl From<std::io::Error> for GpuWeightedRmsNormError {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<vk::Result> for GpuWeightedRmsNormError {
    fn from(v: vk::Result) -> Self {
        Self::Vk(v)
    }
}
impl From<ash::LoadingError> for GpuWeightedRmsNormError {
    fn from(v: ash::LoadingError) -> Self {
        Self::Load(v)
    }
}
impl From<std::str::Utf8Error> for GpuWeightedRmsNormError {
    fn from(v: std::str::Utf8Error) -> Self {
        Self::Utf8(v)
    }
}
impl From<std::ffi::NulError> for GpuWeightedRmsNormError {
    fn from(v: std::ffi::NulError) -> Self {
        Self::CString(v)
    }
}

pub fn run_weighted_rms_norm_with_output(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    reference: Option<&[f32]>,
) -> Result<(Vec<f32>, GpuWeightedRmsNormReport), GpuWeightedRmsNormError> {
    let context = SharedGpuPackedContext::new().map_err(map_context_error)?;
    let (mut runner, compile_duration) =
        CachedGpuWeightedRmsNormRunner::new_with_context(context, input.len(), epsilon)?;
    let (output, mut report) = runner.run_with_output(input, weight, reference)?;
    report.compile_duration = compile_duration;
    Ok((output, report))
}

pub struct CachedGpuWeightedRmsNormRunner {
    _shared_context: Arc<SharedGpuPackedContext>,
    device: Device,
    queue: vk::Queue,
    len: usize,
    _epsilon: f32,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    _descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    input_buffer: BufferAllocation,
    weight_buffer: BufferAllocation,
    output_buffer: BufferAllocation,
}

impl CachedGpuWeightedRmsNormRunner {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        len: usize,
        epsilon: f32,
    ) -> Result<(Self, Duration), GpuWeightedRmsNormError> {
        let compile_started = Instant::now();
        let instance = context.instance.clone();
        let device = context.device.clone();
        let queue = context.queue;
        let physical_device = context.physical_device;
        let queue_family_index = context.queue_family_index;
        let shader_path = compile_shader(Path::new("shaders/weighted_rms_norm.comp"))?;

        let input_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let weight_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let output_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        zero_buffer(&input_buffer, len * 4)?;
        zero_buffer(&weight_buffer, len * 4)?;
        zero_buffer(&output_buffer, len * 4)?;

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
            .size(8)];
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
                .map_err(|(_, err)| GpuWeightedRmsNormError::Vk(err))?[0]
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
        let input_info = [vk::DescriptorBufferInfo::default()
            .buffer(input_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let weight_info = [vk::DescriptorBufferInfo::default()
            .buffer(weight_buffer.buffer)
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
                .buffer_info(&input_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&weight_info),
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
            let epsilon_bits = epsilon.to_bits();
            let push_constants = [len as u32, epsilon_bits];
            let push_bytes = std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                push_constants.len() * 4,
            );
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(command_buffer, 1, 1, 1);
            device.end_command_buffer(command_buffer)?;
        }
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        Ok((
            Self {
                _shared_context: context,
                device,
                queue,
                len,
                _epsilon: epsilon,
                descriptor_set_layout,
                pipeline_layout,
                shader_module,
                pipeline,
                descriptor_pool,
                _descriptor_set: descriptor_set,
                command_pool,
                command_buffer,
                fence,
                input_buffer,
                weight_buffer,
                output_buffer,
            },
            compile_started.elapsed(),
        ))
    }

    pub fn run_with_output(
        &mut self,
        input: &[f32],
        weight: &[f32],
        reference: Option<&[f32]>,
    ) -> Result<(Vec<f32>, GpuWeightedRmsNormReport), GpuWeightedRmsNormError> {
        let mut report = self.run_resident(input, weight)?;
        let (output, download_duration) = self.read_output()?;
        report.download_duration = download_duration;
        let (max_abs_diff, mean_abs_diff) = match reference {
            Some(reference) => compare_outputs(reference, &output),
            None => (0.0, 0.0),
        };
        report.max_abs_diff = max_abs_diff;
        report.mean_abs_diff = mean_abs_diff;
        Ok((output, report))
    }

    pub fn run_resident(
        &mut self,
        input: &[f32],
        weight: &[f32],
    ) -> Result<GpuWeightedRmsNormReport, GpuWeightedRmsNormError> {
        if input.len() != self.len || weight.len() != self.len {
            return Err(GpuWeightedRmsNormError::Shape(format!(
                "input len {} and weight len {} must both match runner len {}",
                input.len(),
                weight.len(),
                self.len
            )));
        }
        let upload_started = Instant::now();
        write_f32_buffer(&self.input_buffer, input)?;
        write_f32_buffer(&self.weight_buffer, weight)?;
        let upload_duration = upload_started.elapsed();
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuWeightedRmsNormReport {
            len: self.len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    pub fn run_resident_from_f32_buffer(
        &mut self,
        source_context: &Arc<SharedGpuPackedContext>,
        source_buffer: vk::Buffer,
        source_len: usize,
        source_buffer_size: u64,
        weight: &[f32],
    ) -> Result<GpuWeightedRmsNormReport, GpuWeightedRmsNormError> {
        if weight.len() != self.len {
            return Err(GpuWeightedRmsNormError::Shape(format!(
                "weight len {} must match runner len {}",
                weight.len(),
                self.len
            )));
        }
        let copy_duration = self.copy_input_from_buffer(
            source_context,
            source_buffer,
            source_len,
            source_buffer_size,
        )?;
        let upload_started = Instant::now();
        write_f32_buffer(&self.weight_buffer, weight)?;
        let weight_upload_duration = upload_started.elapsed();
        let upload_duration = copy_duration + weight_upload_duration;
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuWeightedRmsNormReport {
            len: self.len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    pub fn read_output(&self) -> Result<(Vec<f32>, Duration), GpuWeightedRmsNormError> {
        let download_started = Instant::now();
        let output = read_f32_buffer(&self.output_buffer, self.len)?;
        Ok((output, download_started.elapsed()))
    }

    pub fn shared_context(&self) -> &Arc<SharedGpuPackedContext> {
        &self._shared_context
    }

    pub fn output_buffer_handle(&self) -> vk::Buffer {
        self.output_buffer.buffer
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn output_buffer_size(&self) -> u64 {
        self.output_buffer.size
    }

    fn submit_and_wait(&self) -> Result<Duration, GpuWeightedRmsNormError> {
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

    fn copy_input_from_buffer(
        &self,
        source_context: &Arc<SharedGpuPackedContext>,
        source_buffer: vk::Buffer,
        source_len: usize,
        source_buffer_size: u64,
    ) -> Result<Duration, GpuWeightedRmsNormError> {
        if source_len != self.len {
            return Err(GpuWeightedRmsNormError::Shape(format!(
                "source len {} does not match destination len {}",
                source_len, self.len
            )));
        }
        if !Arc::ptr_eq(&self._shared_context, source_context) {
            return Err(GpuWeightedRmsNormError::Shape(
                "resident chaining requires runners to share the same Vulkan context".to_string(),
            ));
        }
        let byte_len = self.len * std::mem::size_of::<f32>();
        if byte_len as u64 > source_buffer_size || byte_len as u64 > self.input_buffer.size {
            return Err(GpuWeightedRmsNormError::Shape(format!(
                "copy {} bytes exceeds source {} or destination {} buffer size",
                byte_len, source_buffer_size, self.input_buffer.size
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
            let region = [vk::BufferCopy::default().size(byte_len as u64)];
            self.device.cmd_copy_buffer(
                copy_command,
                source_buffer,
                self.input_buffer.buffer,
                &region,
            );
            self.device.end_command_buffer(copy_command)?;
            self.device.reset_fences(&[self.fence])?;
        }
        let submit_info =
            [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&copy_command))];
        let started = Instant::now();
        unsafe {
            self.device
                .queue_submit(self.queue, &submit_info, self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device
                .free_command_buffers(self.command_pool, &[copy_command]);
        }
        Ok(started.elapsed())
    }
}

impl Drop for CachedGpuWeightedRmsNormRunner {
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
            destroy_buffer(&self.device, self.input_buffer);
            destroy_buffer(&self.device, self.weight_buffer);
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

fn compile_shader(path: &Path) -> Result<PathBuf, GpuWeightedRmsNormError> {
    let temp_dir = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".tmp"));
    std::fs::create_dir_all(&temp_dir)?;
    let shader_stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("weighted-rms-norm");
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
        return Err(GpuWeightedRmsNormError::Process(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    std::fs::rename(&tmp, &out)?;
    Ok(out)
}

fn create_shader_module(
    device: &Device,
    path: &Path,
) -> Result<vk::ShaderModule, GpuWeightedRmsNormError> {
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
) -> Result<BufferAllocation, GpuWeightedRmsNormError> {
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
) -> Result<u32, GpuWeightedRmsNormError> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize]
            .property_flags
            .contains(flags);
        if matches_type && has_flags {
            return Ok(i);
        }
    }
    Err(GpuWeightedRmsNormError::Shape(
        "no matching Vulkan memory type found".to_string(),
    ))
}

fn write_f32_buffer(
    buffer: &BufferAllocation,
    data: &[f32],
) -> Result<(), GpuWeightedRmsNormError> {
    let byte_len = std::mem::size_of_val(data) as u64;
    if byte_len > buffer.size {
        return Err(GpuWeightedRmsNormError::Shape(format!(
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

fn zero_buffer(buffer: &BufferAllocation, byte_len: usize) -> Result<(), GpuWeightedRmsNormError> {
    if byte_len as u64 > buffer.size {
        return Err(GpuWeightedRmsNormError::Shape(format!(
            "zero {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    unsafe {
        std::ptr::write_bytes(buffer.mapped_ptr, 0, byte_len);
    }
    Ok(())
}

fn read_f32_buffer(
    buffer: &BufferAllocation,
    len: usize,
) -> Result<Vec<f32>, GpuWeightedRmsNormError> {
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.mapped_ptr as *const f32, out.as_mut_ptr(), len);
    }
    Ok(out)
}

fn compare_outputs(reference: &[f32], output: &[f32]) -> (f32, f32) {
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;
    for (left, right) in reference.iter().zip(output.iter()) {
        let diff = (left - right).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff;
    }
    (max_abs_diff, sum_abs_diff / reference.len().max(1) as f32)
}

fn destroy_buffer(device: &Device, buffer: BufferAllocation) {
    unsafe {
        device.unmap_memory(buffer.memory);
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }
}

fn map_context_error(
    error: crate::gpu::packed_matvec::GpuPackedMatvecError,
) -> GpuWeightedRmsNormError {
    GpuWeightedRmsNormError::Shape(format!("shared gpu context init failed: {error}"))
}
