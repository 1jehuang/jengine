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
pub struct GpuAttentionSingleQueryReport {
    pub query_len: usize,
    pub seq_len: usize,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

#[derive(Debug)]
pub enum GpuAttentionSingleQueryError {
    Io(std::io::Error),
    Vk(vk::Result),
    Load(ash::LoadingError),
    Utf8(std::str::Utf8Error),
    Process(String),
    CString(std::ffi::NulError),
    Shape(String),
}

impl std::fmt::Display for GpuAttentionSingleQueryError {
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

impl std::error::Error for GpuAttentionSingleQueryError {}
impl From<std::io::Error> for GpuAttentionSingleQueryError {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<vk::Result> for GpuAttentionSingleQueryError {
    fn from(v: vk::Result) -> Self {
        Self::Vk(v)
    }
}
impl From<ash::LoadingError> for GpuAttentionSingleQueryError {
    fn from(v: ash::LoadingError) -> Self {
        Self::Load(v)
    }
}
impl From<std::str::Utf8Error> for GpuAttentionSingleQueryError {
    fn from(v: std::str::Utf8Error) -> Self {
        Self::Utf8(v)
    }
}
impl From<std::ffi::NulError> for GpuAttentionSingleQueryError {
    fn from(v: std::ffi::NulError) -> Self {
        Self::CString(v)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_attention_single_query_with_output(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    reference: Option<&[f32]>,
) -> Result<(Vec<f32>, GpuAttentionSingleQueryReport), GpuAttentionSingleQueryError> {
    let context = SharedGpuPackedContext::new().map_err(map_context_error)?;
    let (mut runner, compile_duration) = CachedGpuAttentionSingleQueryRunner::new_with_context(
        context,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
    )?;
    let (output, mut report) = runner.run_with_output(query, keys, values, reference)?;
    report.compile_duration = compile_duration;
    Ok((output, report))
}

pub struct CachedGpuAttentionSingleQueryRunner {
    _shared_context: Arc<SharedGpuPackedContext>,
    device: Device,
    queue: vk::Queue,
    seq_len: usize,
    query_len: usize,
    kv_len: usize,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    _descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    query_buffer: BufferAllocation,
    keys_buffer: BufferAllocation,
    values_buffer: BufferAllocation,
    output_buffer: BufferAllocation,
}

impl CachedGpuAttentionSingleQueryRunner {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        seq_len: usize,
        num_query_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) -> Result<(Self, Duration), GpuAttentionSingleQueryError> {
        let compile_started = Instant::now();
        let instance = context.instance.clone();
        let device = context.device.clone();
        let queue = context.queue;
        let physical_device = context.physical_device;
        let queue_family_index = context.queue_family_index;
        let shader_path = compile_shader(Path::new("shaders/attention_single_query.comp"))?;

        let query_len = num_query_heads * head_dim;
        let kv_len = seq_len * num_key_value_heads * head_dim;
        let query_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (query_len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let keys_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (kv_len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let values_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (kv_len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let output_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (query_len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        zero_buffer(&query_buffer, query_len * 4)?;
        zero_buffer(&keys_buffer, kv_len * 4)?;
        zero_buffer(&values_buffer, kv_len * 4)?;
        zero_buffer(&output_buffer, query_len * 4)?;

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
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
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
            .size(16)];
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
                .map_err(|(_, err)| GpuAttentionSingleQueryError::Vk(err))?[0]
        };
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4)];
        let descriptor_pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_ci, None)? };
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };
        let query_info = [vk::DescriptorBufferInfo::default()
            .buffer(query_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let keys_info = [vk::DescriptorBufferInfo::default()
            .buffer(keys_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let values_info = [vk::DescriptorBufferInfo::default()
            .buffer(values_buffer.buffer)
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
                .buffer_info(&query_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&keys_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&values_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(3)
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
            let push_constants = [
                seq_len as u32,
                num_query_heads as u32,
                num_key_value_heads as u32,
                head_dim as u32,
            ];
            let push_bytes = std::slice::from_raw_parts(push_constants.as_ptr() as *const u8, 16);
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(command_buffer, num_query_heads as u32, 1, 1);
            device.end_command_buffer(command_buffer)?;
        }
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        Ok((
            Self {
                _shared_context: context,
                device,
                queue,
                seq_len,
                query_len,
                kv_len,
                descriptor_set_layout,
                pipeline_layout,
                shader_module,
                pipeline,
                descriptor_pool,
                _descriptor_set: descriptor_set,
                command_pool,
                command_buffer,
                fence,
                query_buffer,
                keys_buffer,
                values_buffer,
                output_buffer,
            },
            compile_started.elapsed(),
        ))
    }

    pub fn run_with_output(
        &mut self,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        reference: Option<&[f32]>,
    ) -> Result<(Vec<f32>, GpuAttentionSingleQueryReport), GpuAttentionSingleQueryError> {
        if query.len() != self.query_len || keys.len() != self.kv_len || values.len() != self.kv_len
        {
            return Err(GpuAttentionSingleQueryError::Shape(format!(
                "query len {}, keys len {}, values len {} must match {}, {}, {}",
                query.len(),
                keys.len(),
                values.len(),
                self.query_len,
                self.kv_len,
                self.kv_len
            )));
        }
        let upload_started = Instant::now();
        write_f32_buffer(&self.query_buffer, query)?;
        write_f32_buffer(&self.keys_buffer, keys)?;
        write_f32_buffer(&self.values_buffer, values)?;
        let upload_duration = upload_started.elapsed();
        let gpu_duration = self.submit_and_wait()?;
        let download_started = Instant::now();
        let output = read_f32_buffer(&self.output_buffer, self.query_len)?;
        let download_duration = download_started.elapsed();
        let (max_abs_diff, mean_abs_diff) = match reference {
            Some(reference) => compare_outputs(reference, &output),
            None => (0.0, 0.0),
        };
        Ok((
            output,
            GpuAttentionSingleQueryReport {
                query_len: self.query_len,
                seq_len: self.seq_len,
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
        query: &[f32],
        keys: &[f32],
        values: &[f32],
    ) -> Result<GpuAttentionSingleQueryReport, GpuAttentionSingleQueryError> {
        if query.len() != self.query_len || keys.len() != self.kv_len || values.len() != self.kv_len
        {
            return Err(GpuAttentionSingleQueryError::Shape(format!(
                "query len {}, keys len {}, values len {} must match {}, {}, {}",
                query.len(),
                keys.len(),
                values.len(),
                self.query_len,
                self.kv_len,
                self.kv_len
            )));
        }
        let upload_started = Instant::now();
        write_f32_buffer(&self.query_buffer, query)?;
        write_f32_buffer(&self.keys_buffer, keys)?;
        write_f32_buffer(&self.values_buffer, values)?;
        let upload_duration = upload_started.elapsed();
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuAttentionSingleQueryReport {
            query_len: self.query_len,
            seq_len: self.seq_len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    pub fn run_resident_with_query_and_kv_buffers(
        &mut self,
        query: &[f32],
        source_context: &Arc<SharedGpuPackedContext>,
        key_buffer: vk::Buffer,
        key_len: usize,
        key_buffer_size: u64,
        value_buffer: vk::Buffer,
        value_len: usize,
        value_buffer_size: u64,
    ) -> Result<GpuAttentionSingleQueryReport, GpuAttentionSingleQueryError> {
        if query.len() != self.query_len || key_len != self.kv_len || value_len != self.kv_len {
            return Err(GpuAttentionSingleQueryError::Shape(format!(
                "query len {}, key len {}, value len {} must match {}, {}, {}",
                query.len(),
                key_len,
                value_len,
                self.query_len,
                self.kv_len,
                self.kv_len
            )));
        }
        let query_upload_started = Instant::now();
        write_f32_buffer(&self.query_buffer, query)?;
        let query_upload_duration = query_upload_started.elapsed();
        let kv_copy_duration = self.copy_kv_from_buffers(
            source_context,
            key_buffer,
            key_buffer_size,
            value_buffer,
            value_buffer_size,
        )?;
        let upload_duration = query_upload_duration + kv_copy_duration;
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuAttentionSingleQueryReport {
            query_len: self.query_len,
            seq_len: self.seq_len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    pub fn run_resident_from_query_and_kv_tensors(
        &mut self,
        query: &GpuResidentBuffer,
        key: &GpuResidentBuffer,
        value: &GpuResidentBuffer,
    ) -> Result<GpuAttentionSingleQueryReport, GpuAttentionSingleQueryError> {
        if query.len != self.query_len || key.len != self.kv_len || value.len != self.kv_len {
            return Err(GpuAttentionSingleQueryError::Shape(format!(
                "query len {}, key len {}, value len {} must match {}, {}, {}",
                query.len, key.len, value.len, self.query_len, self.kv_len, self.kv_len
            )));
        }
        let query_copy_duration = self.copy_query_from_buffer(
            &query.shared_context,
            query.buffer,
            query.len,
            query.buffer_size,
        )?;
        let kv_copy_duration = self.copy_kv_from_buffers(
            &key.shared_context,
            key.buffer,
            key.buffer_size,
            value.buffer,
            value.buffer_size,
        )?;
        let upload_duration = query_copy_duration + kv_copy_duration;
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuAttentionSingleQueryReport {
            query_len: self.query_len,
            seq_len: self.seq_len,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
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
}

impl CachedGpuAttentionSingleQueryRunner {
    fn submit_and_wait(&self) -> Result<Duration, GpuAttentionSingleQueryError> {
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

    fn copy_query_from_buffer(
        &self,
        source_context: &Arc<SharedGpuPackedContext>,
        source_buffer: vk::Buffer,
        source_len: usize,
        source_buffer_size: u64,
    ) -> Result<Duration, GpuAttentionSingleQueryError> {
        if source_len != self.query_len {
            return Err(GpuAttentionSingleQueryError::Shape(format!(
                "source len {} does not match query len {}",
                source_len, self.query_len
            )));
        }
        if !Arc::ptr_eq(&self._shared_context, source_context) {
            return Err(GpuAttentionSingleQueryError::Shape(
                "resident query chaining requires matching Vulkan context".to_string(),
            ));
        }
        let byte_len = self.query_len * std::mem::size_of::<f32>();
        if byte_len as u64 > source_buffer_size || byte_len as u64 > self.query_buffer.size {
            return Err(GpuAttentionSingleQueryError::Shape(format!(
                "query copy {} bytes exceeds source {} or destination {}",
                byte_len, source_buffer_size, self.query_buffer.size
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
                self.query_buffer.buffer,
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

    fn copy_kv_from_buffers(
        &self,
        source_context: &Arc<SharedGpuPackedContext>,
        key_buffer: vk::Buffer,
        key_buffer_size: u64,
        value_buffer: vk::Buffer,
        value_buffer_size: u64,
    ) -> Result<Duration, GpuAttentionSingleQueryError> {
        if !Arc::ptr_eq(&self._shared_context, source_context) {
            return Err(GpuAttentionSingleQueryError::Shape(
                "resident KV chaining requires runners to share the same Vulkan context"
                    .to_string(),
            ));
        }
        let byte_len = self.kv_len * std::mem::size_of::<f32>();
        if byte_len as u64 > key_buffer_size
            || byte_len as u64 > value_buffer_size
            || byte_len as u64 > self.keys_buffer.size
            || byte_len as u64 > self.values_buffer.size
        {
            return Err(GpuAttentionSingleQueryError::Shape(format!(
                "kv copy {} bytes exceeds source or destination buffer sizes",
                byte_len
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
            self.device
                .cmd_copy_buffer(copy_command, key_buffer, self.keys_buffer.buffer, &region);
            self.device.cmd_copy_buffer(
                copy_command,
                value_buffer,
                self.values_buffer.buffer,
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

impl Drop for CachedGpuAttentionSingleQueryRunner {
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
            destroy_buffer(&self.device, self.query_buffer);
            destroy_buffer(&self.device, self.keys_buffer);
            destroy_buffer(&self.device, self.values_buffer);
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

fn compile_shader(path: &Path) -> Result<PathBuf, GpuAttentionSingleQueryError> {
    let temp_dir = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".tmp"));
    std::fs::create_dir_all(&temp_dir)?;
    let shader_stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("attention-single-query");
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
        return Err(GpuAttentionSingleQueryError::Process(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    std::fs::rename(&tmp, &out)?;
    Ok(out)
}

fn create_shader_module(
    device: &Device,
    path: &Path,
) -> Result<vk::ShaderModule, GpuAttentionSingleQueryError> {
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
) -> Result<BufferAllocation, GpuAttentionSingleQueryError> {
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
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32, GpuAttentionSingleQueryError> {
    for idx in 0..memory_props.memory_type_count {
        let supported = (type_bits & (1 << idx)) != 0;
        let flags = memory_props.memory_types[idx as usize].property_flags;
        if supported && flags.contains(required) {
            return Ok(idx);
        }
    }
    Err(GpuAttentionSingleQueryError::Shape(
        "no compatible Vulkan memory type found".to_string(),
    ))
}

fn zero_buffer(
    alloc: &BufferAllocation,
    len_bytes: usize,
) -> Result<(), GpuAttentionSingleQueryError> {
    unsafe {
        std::ptr::write_bytes(alloc.mapped_ptr, 0, len_bytes);
    }
    Ok(())
}

fn write_f32_buffer(
    alloc: &BufferAllocation,
    values: &[f32],
) -> Result<(), GpuAttentionSingleQueryError> {
    let byte_len = std::mem::size_of_val(values);
    if byte_len as u64 > alloc.size {
        return Err(GpuAttentionSingleQueryError::Shape(format!(
            "buffer too small: need {} bytes, have {}",
            byte_len, alloc.size
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr() as *const u8, alloc.mapped_ptr, byte_len);
    }
    Ok(())
}

fn read_f32_buffer(
    alloc: &BufferAllocation,
    len: usize,
) -> Result<Vec<f32>, GpuAttentionSingleQueryError> {
    let byte_len = len * std::mem::size_of::<f32>();
    if byte_len as u64 > alloc.size {
        return Err(GpuAttentionSingleQueryError::Shape(format!(
            "buffer too small: need {} bytes, have {}",
            byte_len, alloc.size
        )));
    }
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(alloc.mapped_ptr as *const f32, out.as_mut_ptr(), len);
    }
    Ok(out)
}

fn compare_outputs(reference: &[f32], output: &[f32]) -> (f32, f32) {
    let (max_abs_diff, sum_abs_diff) = reference.iter().zip(output.iter()).fold(
        (0.0f32, 0.0f32),
        |(max_diff, sum), (expected, actual)| {
            let diff = (expected - actual).abs();
            (max_diff.max(diff), sum + diff)
        },
    );
    (max_abs_diff, sum_abs_diff / reference.len().max(1) as f32)
}

fn map_context_error(
    error: crate::gpu::packed_matvec::GpuPackedMatvecError,
) -> GpuAttentionSingleQueryError {
    match error {
        crate::gpu::packed_matvec::GpuPackedMatvecError::Io(error) => {
            GpuAttentionSingleQueryError::Io(error)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::Vk(error) => {
            GpuAttentionSingleQueryError::Vk(error)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::Load(error) => {
            GpuAttentionSingleQueryError::Load(error)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::Utf8(error) => {
            GpuAttentionSingleQueryError::Utf8(error)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::Process(message) => {
            GpuAttentionSingleQueryError::Process(message)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::CString(error) => {
            GpuAttentionSingleQueryError::CString(error)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::Shape(message) => {
            GpuAttentionSingleQueryError::Shape(message)
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::MissingDevice => {
            GpuAttentionSingleQueryError::Shape("missing Vulkan device".to_string())
        }
        crate::gpu::packed_matvec::GpuPackedMatvecError::MissingQueue => {
            GpuAttentionSingleQueryError::Shape("missing Vulkan queue".to_string())
        }
    }
}
