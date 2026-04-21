use crate::gpu::resident_buffer::GpuResidentBuffer;
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
pub struct GpuVectorAddReport {
    pub len: usize,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

#[derive(Debug)]
pub enum GpuVectorAddError {
    Io(std::io::Error),
    Vk(vk::Result),
    Load(ash::LoadingError),
    Utf8(std::str::Utf8Error),
    Process(String),
    CString(std::ffi::NulError),
    Shape(String),
}

impl std::fmt::Display for GpuVectorAddError {
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

impl std::error::Error for GpuVectorAddError {}
impl From<std::io::Error> for GpuVectorAddError {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<vk::Result> for GpuVectorAddError {
    fn from(v: vk::Result) -> Self {
        Self::Vk(v)
    }
}
impl From<ash::LoadingError> for GpuVectorAddError {
    fn from(v: ash::LoadingError) -> Self {
        Self::Load(v)
    }
}
impl From<std::str::Utf8Error> for GpuVectorAddError {
    fn from(v: std::str::Utf8Error) -> Self {
        Self::Utf8(v)
    }
}
impl From<std::ffi::NulError> for GpuVectorAddError {
    fn from(v: std::ffi::NulError) -> Self {
        Self::CString(v)
    }
}

pub fn run_vector_add_with_output(
    left: &[f32],
    right: &[f32],
    reference: Option<&[f32]>,
) -> Result<(Vec<f32>, GpuVectorAddReport), GpuVectorAddError> {
    let len = left.len();
    let context = SharedGpuPackedContext::new().map_err(map_context_error)?;
    let (mut runner, compile_duration) = CachedGpuVectorAddRunner::new_with_context(context, len)?;
    let (output, mut report) = runner.run_with_output(left, right, reference)?;
    report.compile_duration = compile_duration;
    Ok((output, report))
}

pub struct CachedGpuVectorAddRunner {
    _shared_context: Arc<SharedGpuPackedContext>,
    device: Device,
    queue: vk::Queue,
    len: usize,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    left_buffer: BufferAllocation,
    right_buffer: BufferAllocation,
    output_buffer: BufferAllocation,
}

impl CachedGpuVectorAddRunner {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        len: usize,
    ) -> Result<(Self, Duration), GpuVectorAddError> {
        let compile_started = Instant::now();
        let instance = context.instance.clone();
        let device = context.device.clone();
        let queue = context.queue;
        let physical_device = context.physical_device;
        let queue_family_index = context.queue_family_index;
        let shader_path = compile_shader(Path::new("shaders/vector_add.comp"))?;

        let left_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let right_buffer = create_host_visible_buffer(
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
        zero_buffer(&left_buffer, len * 4)?;
        zero_buffer(&right_buffer, len * 4)?;
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
                .map_err(|(_, err)| GpuVectorAddError::Vk(err))?[0]
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
        let left_info = [vk::DescriptorBufferInfo::default()
            .buffer(left_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let right_info = [vk::DescriptorBufferInfo::default()
            .buffer(right_buffer.buffer)
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
                .buffer_info(&left_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&right_info),
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
            let push_constants = [len as u32];
            let push_bytes = std::slice::from_raw_parts(push_constants.as_ptr() as *const u8, 4);
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(command_buffer, len.div_ceil(64) as u32, 1, 1);
            let output_barrier = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(output_buffer.buffer)
                .offset(0)
                .size(output_buffer.size)];
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &output_barrier,
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
                len,
                descriptor_set_layout,
                pipeline_layout,
                shader_module,
                pipeline,
                descriptor_pool,
                descriptor_set,
                command_pool,
                command_buffer,
                fence,
                left_buffer,
                right_buffer,
                output_buffer,
            },
            compile_started.elapsed(),
        ))
    }

    pub fn run_with_output(
        &mut self,
        left: &[f32],
        right: &[f32],
        reference: Option<&[f32]>,
    ) -> Result<(Vec<f32>, GpuVectorAddReport), GpuVectorAddError> {
        if left.len() != self.len || right.len() != self.len {
            return Err(GpuVectorAddError::Shape(format!(
                "left len {} and right len {} must both match {}",
                left.len(),
                right.len(),
                self.len
            )));
        }
        let upload_started = Instant::now();
        self.bind_internal_buffers();
        write_f32_buffer(&self.left_buffer, left)?;
        write_f32_buffer(&self.right_buffer, right)?;
        let upload_duration = upload_started.elapsed();
        let gpu_duration = self.submit_and_wait()?;
        let download_started = Instant::now();
        let output = read_f32_buffer(&self.output_buffer, self.len)?;
        let download_duration = download_started.elapsed();
        let (max_abs_diff, mean_abs_diff) = match reference {
            Some(reference) => compare_outputs(reference, &output),
            None => (0.0, 0.0),
        };
        Ok((
            output,
            GpuVectorAddReport {
                len: self.len,
                compile_duration: Duration::ZERO,
                upload_duration,
                gpu_duration,
                download_duration,
                max_abs_diff,
                mean_abs_diff,
            },
        ))
    }

    pub fn run_resident(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<GpuVectorAddReport, GpuVectorAddError> {
        if left.len() != self.len || right.len() != self.len {
            return Err(GpuVectorAddError::Shape(format!(
                "left len {} and right len {} must both match {}",
                left.len(),
                right.len(),
                self.len
            )));
        }
        let upload_started = Instant::now();
        self.bind_internal_buffers();
        write_f32_buffer(&self.left_buffer, left)?;
        write_f32_buffer(&self.right_buffer, right)?;
        let upload_duration = upload_started.elapsed();
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuVectorAddReport {
            len: self.len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    pub fn run_resident_from_left_buffer_and_host(
        &mut self,
        source_context: &Arc<SharedGpuPackedContext>,
        source_buffer: vk::Buffer,
        source_len: usize,
        source_buffer_size: u64,
        right: &[f32],
    ) -> Result<GpuVectorAddReport, GpuVectorAddError> {
        if right.len() != self.len {
            return Err(GpuVectorAddError::Shape(format!(
                "right len {} must match {}",
                right.len(),
                self.len
            )));
        }
        let upload_started = Instant::now();
        if source_len != self.len {
            return Err(GpuVectorAddError::Shape(format!(
                "source len {} does not match destination len {}",
                source_len, self.len
            )));
        }
        if !Arc::ptr_eq(&self._shared_context, source_context) {
            return Err(GpuVectorAddError::Shape(
                "resident chaining requires runners to share the same Vulkan context".to_string(),
            ));
        }
        let byte_len = self.len * std::mem::size_of::<f32>();
        if byte_len as u64 > source_buffer_size {
            return Err(GpuVectorAddError::Shape(format!(
                "source {} bytes exceeds provided buffer size {}",
                byte_len, source_buffer_size
            )));
        }
        self.bind_buffers(source_buffer, self.right_buffer.buffer);
        write_f32_buffer(&self.right_buffer, right)?;
        let upload_duration = upload_started.elapsed();
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuVectorAddReport {
            len: self.len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_resident_from_buffers(
        &mut self,
        source_context: &Arc<SharedGpuPackedContext>,
        left_buffer: vk::Buffer,
        left_len: usize,
        left_buffer_size: u64,
        right_buffer: vk::Buffer,
        right_len: usize,
        right_buffer_size: u64,
    ) -> Result<GpuVectorAddReport, GpuVectorAddError> {
        if left_len != self.len || right_len != self.len {
            return Err(GpuVectorAddError::Shape(format!(
                "source lens {}/{} do not match destination len {}",
                left_len, right_len, self.len
            )));
        }
        if !Arc::ptr_eq(&self._shared_context, source_context) {
            return Err(GpuVectorAddError::Shape(
                "resident chaining requires runners to share the same Vulkan context".to_string(),
            ));
        }
        let byte_len = self.len * std::mem::size_of::<f32>();
        if byte_len as u64 > left_buffer_size || byte_len as u64 > right_buffer_size {
            return Err(GpuVectorAddError::Shape(format!(
                "paired source {} bytes exceeds one or more source buffer sizes",
                byte_len
            )));
        }
        self.bind_buffers(left_buffer, right_buffer);
        let upload_duration = Duration::ZERO;
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuVectorAddReport {
            len: self.len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_resident_from_buffers(
        &mut self,
        source_context: &Arc<SharedGpuPackedContext>,
        left_buffer: vk::Buffer,
        left_len: usize,
        left_buffer_size: u64,
        right_buffer: vk::Buffer,
        right_len: usize,
        right_buffer_size: u64,
    ) -> Result<(), GpuVectorAddError> {
        if left_len != self.len || right_len != self.len {
            return Err(GpuVectorAddError::Shape(format!(
                "source lens {}/{} do not match destination len {}",
                left_len, right_len, self.len
            )));
        }
        if !Arc::ptr_eq(&self._shared_context, source_context) {
            return Err(GpuVectorAddError::Shape(
                "resident chaining requires runners to share the same Vulkan context".to_string(),
            ));
        }
        let byte_len = self.len * std::mem::size_of::<f32>();
        if byte_len as u64 > left_buffer_size || byte_len as u64 > right_buffer_size {
            return Err(GpuVectorAddError::Shape(format!(
                "paired source {} bytes exceeds one or more source buffer sizes",
                byte_len
            )));
        }
        self.bind_buffers(left_buffer, right_buffer);
        Ok(())
    }

    fn submit_and_wait(&self) -> Result<Duration, GpuVectorAddError> {
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

    fn bind_internal_buffers(&self) {
        self.bind_buffers(self.left_buffer.buffer, self.right_buffer.buffer);
    }

    fn bind_buffers(&self, left_buffer: vk::Buffer, right_buffer: vk::Buffer) {
        let left_info = [vk::DescriptorBufferInfo::default()
            .buffer(left_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let right_info = [vk::DescriptorBufferInfo::default()
            .buffer(right_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let output_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.output_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&left_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&right_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_info),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn shared_context(&self) -> &Arc<SharedGpuPackedContext> {
        &self._shared_context
    }

    pub fn read_output(&self) -> Result<(Vec<f32>, Duration), GpuVectorAddError> {
        let download_started = Instant::now();
        let output = read_f32_buffer(&self.output_buffer, self.len)?;
        Ok((output, download_started.elapsed()))
    }

    pub fn resident_output(&self) -> GpuResidentBuffer {
        GpuResidentBuffer::new(
            self.shared_context().clone(),
            self.output_buffer_handle(),
            self.len(),
            self.output_buffer_size(),
        )
    }

    pub fn output_buffer_handle(&self) -> vk::Buffer {
        self.output_buffer.buffer
    }
    pub fn output_buffer_size(&self) -> u64 {
        self.output_buffer.size
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn command_buffer_handle(&self) -> vk::CommandBuffer {
        self.command_buffer
    }
}

impl Drop for CachedGpuVectorAddRunner {
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
            destroy_buffer(&self.device, self.left_buffer);
            destroy_buffer(&self.device, self.right_buffer);
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

fn compile_shader(path: &Path) -> Result<PathBuf, GpuVectorAddError> {
    let temp_dir = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".tmp"));
    std::fs::create_dir_all(&temp_dir)?;
    let shader_stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("vector-add");
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
        return Err(GpuVectorAddError::Process(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    std::fs::rename(&tmp, &out)?;
    Ok(out)
}

fn create_shader_module(
    device: &Device,
    path: &Path,
) -> Result<vk::ShaderModule, GpuVectorAddError> {
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
) -> Result<BufferAllocation, GpuVectorAddError> {
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
) -> Result<u32, GpuVectorAddError> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize]
            .property_flags
            .contains(flags);
        if matches_type && has_flags {
            return Ok(i);
        }
    }
    Err(GpuVectorAddError::Shape(
        "no matching Vulkan memory type found".to_string(),
    ))
}

fn write_f32_buffer(buffer: &BufferAllocation, data: &[f32]) -> Result<(), GpuVectorAddError> {
    let byte_len = std::mem::size_of_val(data) as u64;
    if byte_len > buffer.size {
        return Err(GpuVectorAddError::Shape(format!(
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

fn zero_buffer(buffer: &BufferAllocation, byte_len: usize) -> Result<(), GpuVectorAddError> {
    if byte_len as u64 > buffer.size {
        return Err(GpuVectorAddError::Shape(format!(
            "zero {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    unsafe {
        std::ptr::write_bytes(buffer.mapped_ptr, 0, byte_len);
    }
    Ok(())
}

fn read_f32_buffer(buffer: &BufferAllocation, len: usize) -> Result<Vec<f32>, GpuVectorAddError> {
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

fn destroy_buffer(device: &Device, alloc: BufferAllocation) {
    unsafe {
        device.unmap_memory(alloc.memory);
        device.destroy_buffer(alloc.buffer, None);
        device.free_memory(alloc.memory, None);
    }
}

fn map_context_error(error: crate::gpu::packed_matvec::GpuPackedMatvecError) -> GpuVectorAddError {
    match error {
        crate::gpu::packed_matvec::GpuPackedMatvecError::Io(error) => GpuVectorAddError::Io(error),
        crate::gpu::packed_matvec::GpuPackedMatvecError::Vk(error) => GpuVectorAddError::Vk(error),
        crate::gpu::packed_matvec::GpuPackedMatvecError::Load(error) => {
            GpuVectorAddError::Load(error)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::Utf8(error) => {
            GpuVectorAddError::Utf8(error)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::Process(error) => {
            GpuVectorAddError::Process(error)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::CString(error) => {
            GpuVectorAddError::CString(error)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::Shape(message) => {
            GpuVectorAddError::Shape(message)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::MissingDevice => {
            GpuVectorAddError::Shape("missing Vulkan device".to_string())
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::MissingQueue => {
            GpuVectorAddError::Shape("missing Vulkan queue".to_string())
        }
    }
}
