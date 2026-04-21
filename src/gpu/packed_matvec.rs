use crate::gpu::resident_buffer::GpuResidentBuffer;
use ash::util::read_spv;
use ash::{Device, Entry, Instance, vk};
use half::f16;
use std::ffi::{CStr, CString};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq)]
pub struct GpuPackedMatvecReport {
    pub rows: usize,
    pub cols: usize,
    pub compile_duration: Duration,
    pub upload_duration: Duration,
    pub gpu_duration: Duration,
    pub download_duration: Duration,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

impl GpuPackedMatvecReport {
    pub fn summarize(&self) -> String {
        format!(
            "rows={} cols={} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
            self.rows,
            self.cols,
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
pub enum GpuPackedMatvecError {
    Io(std::io::Error),
    Vk(vk::Result),
    Load(ash::LoadingError),
    Utf8(std::str::Utf8Error),
    Process(String),
    CString(std::ffi::NulError),
    MissingDevice,
    MissingQueue,
    Shape(String),
}

impl std::fmt::Display for GpuPackedMatvecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Vk(error) => write!(f, "Vulkan error: {error:?}"),
            Self::Load(error) => write!(f, "Vulkan load error: {error}"),
            Self::Utf8(error) => write!(f, "UTF-8 error: {error}"),
            Self::Process(error) => write!(f, "process error: {error}"),
            Self::CString(error) => write!(f, "CString error: {error}"),
            Self::MissingDevice => write!(f, "no suitable Vulkan device found"),
            Self::MissingQueue => write!(f, "no suitable compute queue family found"),
            Self::Shape(message) => write!(f, "shape error: {message}"),
        }
    }
}

impl std::error::Error for GpuPackedMatvecError {}
impl From<std::io::Error> for GpuPackedMatvecError {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<vk::Result> for GpuPackedMatvecError {
    fn from(v: vk::Result) -> Self {
        Self::Vk(v)
    }
}
impl From<ash::LoadingError> for GpuPackedMatvecError {
    fn from(v: ash::LoadingError) -> Self {
        Self::Load(v)
    }
}
impl From<std::str::Utf8Error> for GpuPackedMatvecError {
    fn from(v: std::str::Utf8Error) -> Self {
        Self::Utf8(v)
    }
}
impl From<std::ffi::NulError> for GpuPackedMatvecError {
    fn from(v: std::ffi::NulError) -> Self {
        Self::CString(v)
    }
}

pub fn run_packed_ternary_matvec(
    code_words: &[u32],
    scales: &[f32],
    group_size: usize,
    rows: usize,
    cols: usize,
    input: &[f32],
    reference: &[f32],
) -> Result<GpuPackedMatvecReport, GpuPackedMatvecError> {
    let (_, report) = run_packed_ternary_matvec_with_output(
        code_words,
        scales,
        group_size,
        rows,
        cols,
        input,
        Some(reference),
    )?;
    Ok(report)
}

pub fn run_packed_ternary_matvec_with_output(
    code_words: &[u32],
    scales: &[f32],
    group_size: usize,
    rows: usize,
    cols: usize,
    input: &[f32],
    reference: Option<&[f32]>,
) -> Result<(Vec<f32>, GpuPackedMatvecReport), GpuPackedMatvecError> {
    let (mut runner, compile_duration) =
        CachedGpuPackedMatvecRunner::new(code_words, scales, group_size, rows, cols)?;
    let (output, mut report) = runner.run_with_output(input, reference)?;
    report.compile_duration = compile_duration;
    Ok((output, report))
}

pub fn run_packed_ternary_matvec_raw_f32_with_output(
    code_words: &[u32],
    scales: &[f32],
    group_size: usize,
    rows: usize,
    cols: usize,
    input: &[f32],
    reference: Option<&[f32]>,
) -> Result<(Vec<f32>, GpuPackedMatvecReport), GpuPackedMatvecError> {
    let (mut runner, compile_duration) =
        CachedGpuPackedMatvecRunner::new_raw_f32_input(code_words, scales, group_size, rows, cols)?;
    let (output, mut report) = runner.run_with_output(input, reference)?;
    report.compile_duration = compile_duration;
    Ok((output, report))
}

pub struct SharedGpuPackedContext {
    pub(crate) _entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) device: Device,
    pub(crate) queue: vk::Queue,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) queue_family_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedRunnerInputMode {
    PackedHalfPairs,
    RawF32,
}

impl SharedGpuPackedContext {
    pub fn new() -> Result<Arc<Self>, GpuPackedMatvecError> {
        let entry = unsafe { Entry::load()? };
        let app_name = CString::new("jengine-packed-matvec")?;
        let engine_name = CString::new("jengine")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);
        let instance_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_ci, None)? };

        let (physical_device, queue_family_index) = pick_compute_device(&instance)?;
        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let priorities = [1.0f32];
        let queue_ci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_ci = vk::DeviceCreateInfo::default().queue_create_infos(&queue_ci);
        let device = unsafe { instance.create_device(physical_device, &device_ci, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let _ = packed_shader_variant(&device_properties);
        Ok(Arc::new(Self {
            _entry: entry,
            instance,
            device,
            queue,
            physical_device,
            queue_family_index,
        }))
    }

    fn shader_choice(&self) -> (&'static str, u32) {
        let device_properties = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        packed_shader_variant(&device_properties)
    }
}

impl Drop for SharedGpuPackedContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct CachedGpuPackedMatvecRunner {
    _shared_context: Arc<SharedGpuPackedContext>,
    _instance: Instance,
    device: Device,
    queue: vk::Queue,
    rows: usize,
    cols: usize,
    group_size: usize,
    packed_cols: usize,
    input_mode: PackedRunnerInputMode,
    workgroup_size: u32,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    code_buffer: BufferAllocation,
    scale_buffer: BufferAllocation,
    vector_buffer: BufferAllocation,
    output_buffer: BufferAllocation,
}

impl CachedGpuPackedMatvecRunner {
    pub fn buffer_bytes(&self) -> usize {
        (self.code_buffer_bytes()
            + self.scale_buffer_bytes()
            + self.vector_buffer_bytes()
            + self.output_buffer_bytes()) as usize
    }

    fn code_buffer_bytes(&self) -> u64 {
        unsafe {
            self.device
                .get_buffer_memory_requirements(self.code_buffer.buffer)
                .size
        }
    }

    fn scale_buffer_bytes(&self) -> u64 {
        unsafe {
            self.device
                .get_buffer_memory_requirements(self.scale_buffer.buffer)
                .size
        }
    }

    fn vector_buffer_bytes(&self) -> u64 {
        unsafe {
            self.device
                .get_buffer_memory_requirements(self.vector_buffer.buffer)
                .size
        }
    }

    fn output_buffer_bytes(&self) -> u64 {
        unsafe {
            self.device
                .get_buffer_memory_requirements(self.output_buffer.buffer)
                .size
        }
    }

    pub fn new(
        code_words: &[u32],
        scales: &[f32],
        group_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, Duration), GpuPackedMatvecError> {
        let context = SharedGpuPackedContext::new()?;
        Self::new_with_context(context, code_words, scales, group_size, rows, cols)
    }

    pub fn new_raw_f32_input(
        code_words: &[u32],
        scales: &[f32],
        group_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, Duration), GpuPackedMatvecError> {
        let context = SharedGpuPackedContext::new()?;
        Self::new_with_context_and_input_mode(
            context,
            code_words,
            scales,
            group_size,
            rows,
            cols,
            PackedRunnerInputMode::RawF32,
        )
    }

    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        code_words: &[u32],
        scales: &[f32],
        group_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, Duration), GpuPackedMatvecError> {
        Self::new_with_context_and_input_mode(
            context,
            code_words,
            scales,
            group_size,
            rows,
            cols,
            PackedRunnerInputMode::PackedHalfPairs,
        )
    }

    pub fn new_with_context_and_input_mode(
        context: Arc<SharedGpuPackedContext>,
        code_words: &[u32],
        scales: &[f32],
        group_size: usize,
        rows: usize,
        cols: usize,
        input_mode: PackedRunnerInputMode,
    ) -> Result<(Self, Duration), GpuPackedMatvecError> {
        let compile_started = Instant::now();

        let instance = context.instance.clone();
        let device = context.device.clone();
        let queue = context.queue;
        let physical_device = context.physical_device;
        let queue_family_index = context.queue_family_index;
        let (shader_source, workgroup_size) = match input_mode {
            PackedRunnerInputMode::PackedHalfPairs => context.shader_choice(),
            PackedRunnerInputMode::RawF32 => ("shaders/packed_ternary_matvec_f32_input.comp", 64),
        };
        let shader_path = compile_shader(Path::new(shader_source))?;

        let packed_cols = cols.div_ceil(2);
        let output_bytes = rows * std::mem::size_of::<f32>();
        let vector_bytes = match input_mode {
            PackedRunnerInputMode::PackedHalfPairs => packed_cols * 4,
            PackedRunnerInputMode::RawF32 => cols * std::mem::size_of::<f32>(),
        };
        let code_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (code_words.len() * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let scale_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            (scales.len() * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let vector_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            vector_bytes as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let output_buffer = create_host_visible_buffer(
            &instance,
            &device,
            physical_device,
            output_bytes as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        write_u32_buffer(&code_buffer, code_words)?;
        write_f32_buffer(&scale_buffer, scales)?;
        zero_buffer(&vector_buffer, vector_bytes)?;
        zero_buffer(&output_buffer, output_bytes)?;

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
                .map_err(|(_, err)| GpuPackedMatvecError::Vk(err))?[0]
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4)];
        let descriptor_pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_ci, None)? };
        let set_layouts = [descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

        let code_info = [vk::DescriptorBufferInfo::default()
            .buffer(code_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let scale_info = [vk::DescriptorBufferInfo::default()
            .buffer(scale_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let vector_info = [vk::DescriptorBufferInfo::default()
            .buffer(vector_buffer.buffer)
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
                .buffer_info(&code_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&scale_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&vector_info),
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
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };

        let runner = Self {
            _shared_context: context,
            _instance: instance,
            device,
            queue,
            rows,
            cols,
            group_size,
            packed_cols,
            input_mode,
            workgroup_size,
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
            descriptor_pool,
            descriptor_set,
            command_pool,
            command_buffer,
            fence,
            code_buffer,
            scale_buffer,
            vector_buffer,
            output_buffer,
        };
        runner.record_command_buffer()?;
        Ok((runner, compile_started.elapsed()))
    }

    pub fn new_uninitialized(
        code_word_len: usize,
        scale_len: usize,
        group_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, Duration), GpuPackedMatvecError> {
        let context = SharedGpuPackedContext::new()?;
        Self::new_uninitialized_with_context(
            context,
            code_word_len,
            scale_len,
            group_size,
            rows,
            cols,
        )
    }

    pub fn new_uninitialized_with_context(
        context: Arc<SharedGpuPackedContext>,
        code_word_len: usize,
        scale_len: usize,
        group_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, Duration), GpuPackedMatvecError> {
        let zero_code_words = vec![0u32; code_word_len];
        let zero_scales = vec![0.0f32; scale_len];
        Self::new_with_context(
            context,
            &zero_code_words,
            &zero_scales,
            group_size,
            rows,
            cols,
        )
    }

    pub fn new_uninitialized_raw_f32_input_with_context(
        context: Arc<SharedGpuPackedContext>,
        code_word_len: usize,
        scale_len: usize,
        group_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, Duration), GpuPackedMatvecError> {
        let zero_code_words = vec![0u32; code_word_len];
        let zero_scales = vec![0.0f32; scale_len];
        Self::new_with_context_and_input_mode(
            context,
            &zero_code_words,
            &zero_scales,
            group_size,
            rows,
            cols,
            PackedRunnerInputMode::RawF32,
        )
    }

    pub fn update_weights(
        &mut self,
        code_words: &[u32],
        scales: &[f32],
    ) -> Result<Duration, GpuPackedMatvecError> {
        let expected_code_words = (self.code_buffer_bytes() / 4) as usize;
        let expected_scales = (self.scale_buffer_bytes() / 4) as usize;
        if code_words.len() != expected_code_words {
            return Err(GpuPackedMatvecError::Shape(format!(
                "code_words length {} does not match expected {}",
                code_words.len(),
                expected_code_words
            )));
        }
        if scales.len() != expected_scales {
            return Err(GpuPackedMatvecError::Shape(format!(
                "scales length {} does not match expected {}",
                scales.len(),
                expected_scales
            )));
        }
        let started = Instant::now();
        write_u32_buffer(&self.code_buffer, code_words)?;
        write_f32_buffer(&self.scale_buffer, scales)?;
        Ok(started.elapsed())
    }

    pub fn run_with_output(
        &mut self,
        input: &[f32],
        reference: Option<&[f32]>,
    ) -> Result<(Vec<f32>, GpuPackedMatvecReport), GpuPackedMatvecError> {
        let mut report = self.run_resident(input)?;
        let (gpu_output, download_duration) = self.read_output()?;
        report.download_duration = download_duration;
        let (max_abs_diff, mean_abs_diff) = match reference {
            Some(reference) => compare_outputs(reference, &gpu_output),
            None => (0.0, 0.0),
        };
        report.max_abs_diff = max_abs_diff;
        report.mean_abs_diff = mean_abs_diff;

        Ok((gpu_output, report))
    }

    pub fn run_with_argmax(
        &mut self,
        input: &[f32],
    ) -> Result<(usize, GpuPackedMatvecReport), GpuPackedMatvecError> {
        let mut report = self.run_resident(input)?;
        let (argmax_index, download_duration) = self.argmax_output()?;
        report.download_duration = download_duration;
        Ok((argmax_index, report))
    }

    pub fn run_with_output_from_runner(
        &mut self,
        source: &Self,
        reference: Option<&[f32]>,
    ) -> Result<(Vec<f32>, GpuPackedMatvecReport), GpuPackedMatvecError> {
        let mut report = self.run_resident_from_runner(source)?;
        let (gpu_output, download_duration) = self.read_output()?;
        report.download_duration = download_duration;
        let (max_abs_diff, mean_abs_diff) = match reference {
            Some(reference) => compare_outputs(reference, &gpu_output),
            None => (0.0, 0.0),
        };
        report.max_abs_diff = max_abs_diff;
        report.mean_abs_diff = mean_abs_diff;
        Ok((gpu_output, report))
    }

    pub fn run_resident(
        &mut self,
        input: &[f32],
    ) -> Result<GpuPackedMatvecReport, GpuPackedMatvecError> {
        let (upload_duration, gpu_duration) = self.run_without_download(input)?;
        Ok(GpuPackedMatvecReport {
            rows: self.rows,
            cols: self.cols,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    pub fn run_resident_from_runner(
        &mut self,
        source: &Self,
    ) -> Result<GpuPackedMatvecReport, GpuPackedMatvecError> {
        if self.input_mode != PackedRunnerInputMode::RawF32 {
            return Err(GpuPackedMatvecError::Shape(
                "resident chaining requires a raw-f32 input runner".to_string(),
            ));
        }
        if source.rows != self.cols {
            return Err(GpuPackedMatvecError::Shape(format!(
                "source rows {} do not match destination cols {}",
                source.rows, self.cols
            )));
        }
        if !Arc::ptr_eq(&self._shared_context, &source._shared_context) {
            return Err(GpuPackedMatvecError::Shape(
                "resident chaining requires runners to share the same Vulkan context".to_string(),
            ));
        }
        let upload_duration = self.copy_input_from_buffer(
            source.shared_context(),
            source.output_buffer.buffer,
            source.rows,
            source.output_buffer.size,
        )?;
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuPackedMatvecReport {
            rows: self.rows,
            cols: self.cols,
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
    ) -> Result<GpuPackedMatvecReport, GpuPackedMatvecError> {
        let upload_duration = self.copy_input_from_buffer(
            source_context,
            source_buffer,
            source_len,
            source_buffer_size,
        )?;
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuPackedMatvecReport {
            rows: self.rows,
            cols: self.cols,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    pub fn run_resident_from_packed_buffer(
        &mut self,
        source_context: &Arc<SharedGpuPackedContext>,
        source_buffer: vk::Buffer,
        source_packed_len: usize,
        source_buffer_size: u64,
    ) -> Result<GpuPackedMatvecReport, GpuPackedMatvecError> {
        let upload_duration = self.copy_packed_input_from_buffer(
            source_context,
            source_buffer,
            source_packed_len,
            source_buffer_size,
        )?;
        let gpu_duration = self.submit_and_wait()?;
        Ok(GpuPackedMatvecReport {
            rows: self.rows,
            cols: self.cols,
            compile_duration: Duration::ZERO,
            upload_duration,
            gpu_duration,
            download_duration: Duration::ZERO,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
        })
    }

    pub fn run_resident_from_f32_tensor(
        &mut self,
        source: &GpuResidentBuffer,
    ) -> Result<GpuPackedMatvecReport, GpuPackedMatvecError> {
        self.run_resident_from_f32_buffer(
            &source.shared_context,
            source.buffer,
            source.len,
            source.buffer_size,
        )
    }

    pub fn run_resident_from_packed_tensor(
        &mut self,
        source: &GpuResidentBuffer,
    ) -> Result<GpuPackedMatvecReport, GpuPackedMatvecError> {
        self.run_resident_from_packed_buffer(
            &source.shared_context,
            source.buffer,
            source.len,
            source.buffer_size,
        )
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

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn read_output(&self) -> Result<(Vec<f32>, Duration), GpuPackedMatvecError> {
        let download_started = Instant::now();
        let gpu_output = read_f32_buffer(&self.output_buffer, self.rows)?;
        Ok((gpu_output, download_started.elapsed()))
    }

    pub fn argmax_output(&self) -> Result<(usize, Duration), GpuPackedMatvecError> {
        let download_started = Instant::now();
        let argmax_index = argmax_f32_buffer(&self.output_buffer, self.rows)?;
        Ok((argmax_index, download_started.elapsed()))
    }

    pub fn run_without_download(
        &mut self,
        input: &[f32],
    ) -> Result<(Duration, Duration), GpuPackedMatvecError> {
        if input.len() != self.cols {
            return Err(GpuPackedMatvecError::Shape(format!(
                "input length {} does not match cols {}",
                input.len(),
                self.cols
            )));
        }
        match self.input_mode {
            PackedRunnerInputMode::PackedHalfPairs => {
                let vector_words = pack_f16_pairs(input);
                if vector_words.len() != self.packed_cols {
                    return Err(GpuPackedMatvecError::Shape(format!(
                        "packed input length {} does not match packed cols {}",
                        vector_words.len(),
                        self.packed_cols
                    )));
                }
                let upload_started = Instant::now();
                write_u32_buffer(&self.vector_buffer, &vector_words)?;
                let upload_duration = upload_started.elapsed();
                let gpu_duration = self.submit_and_wait()?;
                Ok((upload_duration, gpu_duration))
            }
            PackedRunnerInputMode::RawF32 => {
                let upload_started = Instant::now();
                write_f32_buffer(&self.vector_buffer, input)?;
                let upload_duration = upload_started.elapsed();
                let gpu_duration = self.submit_and_wait()?;
                Ok((upload_duration, gpu_duration))
            }
        }
    }

    fn submit_and_wait(&self) -> Result<Duration, GpuPackedMatvecError> {
        let submit_info = [
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer))
        ];
        unsafe {
            self.device.reset_fences(&[self.fence])?;
        }
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
    ) -> Result<Duration, GpuPackedMatvecError> {
        if self.input_mode != PackedRunnerInputMode::RawF32 {
            return Err(GpuPackedMatvecError::Shape(
                "resident chaining requires a raw-f32 input runner".to_string(),
            ));
        }
        if source_len != self.cols {
            return Err(GpuPackedMatvecError::Shape(format!(
                "source len {} does not match destination cols {}",
                source_len, self.cols
            )));
        }
        if !Arc::ptr_eq(&self._shared_context, source_context) {
            return Err(GpuPackedMatvecError::Shape(
                "resident chaining requires runners to share the same Vulkan context".to_string(),
            ));
        }
        let byte_len = self.cols * std::mem::size_of::<f32>();
        if byte_len as u64 > source_buffer_size || byte_len as u64 > self.vector_buffer.size {
            return Err(GpuPackedMatvecError::Shape(format!(
                "copy {} bytes exceeds source {} or destination {} buffer size",
                byte_len, source_buffer_size, self.vector_buffer.size
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
                self.vector_buffer.buffer,
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

    fn copy_packed_input_from_buffer(
        &self,
        source_context: &Arc<SharedGpuPackedContext>,
        source_buffer: vk::Buffer,
        source_packed_len: usize,
        source_buffer_size: u64,
    ) -> Result<Duration, GpuPackedMatvecError> {
        if self.input_mode != PackedRunnerInputMode::PackedHalfPairs {
            return Err(GpuPackedMatvecError::Shape(
                "packed resident chaining requires a packed-half-pairs input runner".to_string(),
            ));
        }
        if source_packed_len != self.packed_cols {
            return Err(GpuPackedMatvecError::Shape(format!(
                "source packed len {} does not match destination packed cols {}",
                source_packed_len, self.packed_cols
            )));
        }
        if !Arc::ptr_eq(&self._shared_context, source_context) {
            return Err(GpuPackedMatvecError::Shape(
                "resident chaining requires runners to share the same Vulkan context".to_string(),
            ));
        }
        let byte_len = self.packed_cols * std::mem::size_of::<u32>();
        if byte_len as u64 > source_buffer_size || byte_len as u64 > self.vector_buffer.size {
            return Err(GpuPackedMatvecError::Shape(format!(
                "copy {} bytes exceeds source {} or destination {} buffer size",
                byte_len, source_buffer_size, self.vector_buffer.size
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
                self.vector_buffer.buffer,
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

    fn record_command_buffer(&self) -> Result<(), GpuPackedMatvecError> {
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
        }
        let push_constants = [
            self.rows as u32,
            self.cols as u32,
            self.group_size as u32,
            self.packed_cols as u32,
        ];
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                push_constants.len() * 4,
            )
        };
        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            self.device.cmd_dispatch(
                self.command_buffer,
                self.rows.div_ceil(self.workgroup_size as usize) as u32,
                1,
                1,
            );
            self.device.end_command_buffer(self.command_buffer)?;
        }
        Ok(())
    }
}

impl Drop for CachedGpuPackedMatvecRunner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
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
            destroy_buffer(&self.device, self.code_buffer);
            destroy_buffer(&self.device, self.scale_buffer);
            destroy_buffer(&self.device, self.vector_buffer);
            destroy_buffer(&self.device, self.output_buffer);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_ptr: *mut u8,
    size: u64,
}

fn packed_shader_variant(properties: &vk::PhysicalDeviceProperties) -> (&'static str, u32) {
    match std::env::var("JENGINE_PACKED_SHADER_VARIANT")
        .ok()
        .as_deref()
    {
        Some("xe2_32") => ("shaders/packed_ternary_matvec_xe2_32.comp", 32),
        Some("xe2_subgroup_row") => ("shaders/packed_ternary_matvec_xe2_subgroup_row.comp", 1),
        Some("default") => ("shaders/packed_ternary_matvec.comp", 64),
        _ => {
            let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_str()
                .unwrap_or_default();
            if properties.vendor_id == 0x8086 && device_name.contains("(LNL)") {
                ("shaders/packed_ternary_matvec_xe2_subgroup_row.comp", 1)
            } else {
                ("shaders/packed_ternary_matvec.comp", 64)
            }
        }
    }
}

fn compile_shader(path: &Path) -> Result<PathBuf, GpuPackedMatvecError> {
    let temp_dir = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".tmp"));
    std::fs::create_dir_all(&temp_dir)?;
    let shader_stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("jengine-packed-matvec");
    let out = temp_dir.join(format!("{shader_stem}.spv"));
    if shader_cache_is_fresh(path, &out)? {
        return Ok(out);
    }
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
        return Err(GpuPackedMatvecError::Process(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    std::fs::rename(&tmp, &out)?;
    Ok(out)
}

fn shader_cache_is_fresh(source: &Path, compiled: &Path) -> Result<bool, GpuPackedMatvecError> {
    let Ok(compiled_meta) = std::fs::metadata(compiled) else {
        return Ok(false);
    };
    let source_meta = std::fs::metadata(source)?;
    let Ok(compiled_modified) = compiled_meta.modified() else {
        return Ok(false);
    };
    let Ok(source_modified) = source_meta.modified() else {
        return Ok(false);
    };
    Ok(compiled_modified >= source_modified && compiled_meta.len() > 0)
}

fn pick_compute_device(
    instance: &Instance,
) -> Result<(vk::PhysicalDevice, u32), GpuPackedMatvecError> {
    let devices = unsafe { instance.enumerate_physical_devices()? };
    for device in devices {
        let queues = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, queue) in queues.iter().enumerate() {
            if queue.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Ok((device, index as u32));
            }
        }
    }
    Err(GpuPackedMatvecError::MissingDevice)
}

fn create_shader_module(
    device: &Device,
    path: &Path,
) -> Result<vk::ShaderModule, GpuPackedMatvecError> {
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
) -> Result<BufferAllocation, GpuPackedMatvecError> {
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
) -> Result<u32, GpuPackedMatvecError> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize]
            .property_flags
            .contains(flags);
        if matches_type && has_flags {
            return Ok(i);
        }
    }
    Err(GpuPackedMatvecError::MissingQueue)
}

fn write_u32_buffer(buffer: &BufferAllocation, data: &[u32]) -> Result<(), GpuPackedMatvecError> {
    let byte_len = std::mem::size_of_val(data) as u64;
    if byte_len > buffer.size {
        return Err(GpuPackedMatvecError::Shape(format!(
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

fn write_f32_buffer(buffer: &BufferAllocation, data: &[f32]) -> Result<(), GpuPackedMatvecError> {
    let byte_len = std::mem::size_of_val(data) as u64;
    if byte_len > buffer.size {
        return Err(GpuPackedMatvecError::Shape(format!(
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

fn zero_buffer(buffer: &BufferAllocation, byte_len: usize) -> Result<(), GpuPackedMatvecError> {
    if byte_len as u64 > buffer.size {
        return Err(GpuPackedMatvecError::Shape(format!(
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
) -> Result<Vec<f32>, GpuPackedMatvecError> {
    let byte_len = len * std::mem::size_of::<f32>();
    if byte_len as u64 > buffer.size {
        return Err(GpuPackedMatvecError::Shape(format!(
            "read {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.mapped_ptr as *const f32, out.as_mut_ptr(), len);
    }
    Ok(out)
}

fn argmax_f32_buffer(buffer: &BufferAllocation, len: usize) -> Result<usize, GpuPackedMatvecError> {
    let byte_len = len * std::mem::size_of::<f32>();
    if byte_len as u64 > buffer.size {
        return Err(GpuPackedMatvecError::Shape(format!(
            "read {} bytes exceeds mapped buffer size {}",
            byte_len, buffer.size
        )));
    }
    if len == 0 {
        return Err(GpuPackedMatvecError::Shape(
            "argmax requires at least one output element".to_string(),
        ));
    }
    let values = unsafe { std::slice::from_raw_parts(buffer.mapped_ptr as *const f32, len) };
    let mut best_index = 0usize;
    let mut best_value = values[0];
    for (index, value) in values.iter().enumerate().skip(1) {
        if *value > best_value {
            best_value = *value;
            best_index = index;
        }
    }
    Ok(best_index)
}

fn destroy_buffer(device: &Device, buffer: BufferAllocation) {
    unsafe {
        device.unmap_memory(buffer.memory);
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }
}

fn pack_f16_pairs(values: &[f32]) -> Vec<u32> {
    values
        .chunks(2)
        .map(|chunk| {
            let a = f16::from_f32(chunk[0]).to_bits() as u32;
            let b = chunk
                .get(1)
                .map(|v| f16::from_f32(*v).to_bits() as u32)
                .unwrap_or(0);
            a | (b << 16)
        })
        .collect()
}

fn compare_outputs(reference: &[f32], output: &[f32]) -> (f32, f32) {
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;
    for (left, right) in reference.iter().zip(output.iter()) {
        let diff = (left - right).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff;
    }
    (max_abs_diff, sum_abs_diff / reference.len() as f32)
}
