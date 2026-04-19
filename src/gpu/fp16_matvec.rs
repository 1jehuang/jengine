use ash::util::read_spv;
use ash::{Device, Entry, Instance, vk};
use half::f16;
use std::ffi::CString;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub struct GpuMatvecReport {
    pub rows: usize,
    pub cols: usize,
    pub compile_duration: Duration,
    pub gpu_duration: Duration,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

impl GpuMatvecReport {
    pub fn summarize(&self) -> String {
        format!(
            "rows={} cols={} compile_ms={:.3} gpu_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
            self.rows,
            self.cols,
            self.compile_duration.as_secs_f64() * 1_000.0,
            self.gpu_duration.as_secs_f64() * 1_000.0,
            self.max_abs_diff,
            self.mean_abs_diff,
        )
    }
}

#[derive(Debug)]
pub enum GpuMatvecError {
    Io(std::io::Error),
    Vk(vk::Result),
    Load(ash::LoadingError),
    Utf8(std::str::Utf8Error),
    Process(String),
    CString(std::ffi::NulError),
    MissingDevice,
    MissingQueue,
}

impl std::fmt::Display for GpuMatvecError {
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
        }
    }
}

impl std::error::Error for GpuMatvecError {}
impl From<std::io::Error> for GpuMatvecError {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<vk::Result> for GpuMatvecError {
    fn from(v: vk::Result) -> Self {
        Self::Vk(v)
    }
}
impl From<ash::LoadingError> for GpuMatvecError {
    fn from(v: ash::LoadingError) -> Self {
        Self::Load(v)
    }
}
impl From<std::str::Utf8Error> for GpuMatvecError {
    fn from(v: std::str::Utf8Error) -> Self {
        Self::Utf8(v)
    }
}
impl From<std::ffi::NulError> for GpuMatvecError {
    fn from(v: std::ffi::NulError) -> Self {
        Self::CString(v)
    }
}

pub fn run_dense_fp16_matvec(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
    reference: &[f32],
) -> Result<GpuMatvecReport, GpuMatvecError> {
    let compile_started = Instant::now();
    let shader_path = compile_shader(Path::new("shaders/fp16_matvec.comp"))?;
    let compile_duration = compile_started.elapsed();

    let packed_cols = cols.div_ceil(2);
    let matrix_words = pack_f16_pairs(matrix);
    let vector_words = pack_f16_pairs(input);
    let output_bytes = rows * std::mem::size_of::<f32>();

    let entry = unsafe { Entry::load()? };
    let app_name = CString::new("jengine-fp16-matvec")?;
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
    let priorities = [1.0f32];
    let queue_ci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities)];
    let device_ci = vk::DeviceCreateInfo::default().queue_create_infos(&queue_ci);
    let device = unsafe { instance.create_device(physical_device, &device_ci, None)? };
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let matrix_buffer = create_host_visible_buffer(
        &instance,
        &device,
        physical_device,
        (matrix_words.len() * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let vector_buffer = create_host_visible_buffer(
        &instance,
        &device,
        physical_device,
        (vector_words.len() * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let output_buffer = create_host_visible_buffer(
        &instance,
        &device,
        physical_device,
        output_bytes as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    write_u32_buffer(&device, matrix_buffer.memory, &matrix_words)?;
    write_u32_buffer(&device, vector_buffer.memory, &vector_words)?;
    zero_buffer(&device, output_buffer.memory, output_bytes)?;

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
            .map_err(|(_, err)| GpuMatvecError::Vk(err))?[0]
    };

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3)];
    let descriptor_pool_ci = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(1);
    let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_ci, None)? };
    let set_layouts = [descriptor_set_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

    let matrix_info = [vk::DescriptorBufferInfo::default()
        .buffer(matrix_buffer.buffer)
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
            .buffer_info(&matrix_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&vector_info),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&output_info),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    let command_pool_ci =
        vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
    let command_pool = unsafe { device.create_command_pool(&command_pool_ci, None)? };
    let command_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&command_alloc)?[0] };

    unsafe { device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())? };
    unsafe {
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );
    }
    let push_constants = [rows as u32, cols as u32, packed_cols as u32];
    let push_bytes = unsafe {
        std::slice::from_raw_parts(
            push_constants.as_ptr() as *const u8,
            push_constants.len() * 4,
        )
    };
    unsafe {
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_bytes,
        );
        device.cmd_dispatch(command_buffer, rows.div_ceil(64) as u32, 1, 1);
        device.end_command_buffer(command_buffer)?;
    }

    let submit_info =
        [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer))];
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
    let gpu_started = Instant::now();
    unsafe {
        device.queue_submit(queue, &submit_info, fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
    }
    let gpu_duration = gpu_started.elapsed();

    let gpu_output = read_f32_buffer(&device, output_buffer.memory, rows)?;
    let (max_abs_diff, mean_abs_diff) = compare_outputs(reference, &gpu_output);

    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        destroy_buffer(&device, matrix_buffer);
        destroy_buffer(&device, vector_buffer);
        destroy_buffer(&device, output_buffer);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    let _ = std::fs::remove_file(shader_path);

    Ok(GpuMatvecReport {
        rows,
        cols,
        compile_duration,
        gpu_duration,
        max_abs_diff,
        mean_abs_diff,
    })
}

#[derive(Debug, Clone, Copy)]
struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

fn compile_shader(path: &Path) -> Result<PathBuf, GpuMatvecError> {
    let out = std::env::temp_dir().join("jengine-fp16-matvec.spv");
    let output = Command::new("glslc")
        .arg(path)
        .arg("-o")
        .arg(&out)
        .output()?;
    if !output.status.success() {
        return Err(GpuMatvecError::Process(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    let val = Command::new("spirv-val").arg(&out).output()?;
    if !val.status.success() {
        return Err(GpuMatvecError::Process(
            String::from_utf8_lossy(&val.stderr).to_string(),
        ));
    }
    Ok(out)
}

fn pick_compute_device(instance: &Instance) -> Result<(vk::PhysicalDevice, u32), GpuMatvecError> {
    let devices = unsafe { instance.enumerate_physical_devices()? };
    for device in devices {
        let queues = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, queue) in queues.iter().enumerate() {
            if queue.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Ok((device, index as u32));
            }
        }
    }
    Err(GpuMatvecError::MissingDevice)
}

fn create_shader_module(device: &Device, path: &Path) -> Result<vk::ShaderModule, GpuMatvecError> {
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
) -> Result<BufferAllocation, GpuMatvecError> {
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
    Ok(BufferAllocation { buffer, memory })
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    filter: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, GpuMatvecError> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize]
            .property_flags
            .contains(flags);
        if matches_type && has_flags {
            return Ok(i);
        }
    }
    Err(GpuMatvecError::MissingQueue)
}

fn write_u32_buffer(
    device: &Device,
    memory: vk::DeviceMemory,
    data: &[u32],
) -> Result<(), GpuMatvecError> {
    let byte_len = std::mem::size_of_val(data) as u64;
    let ptr = unsafe { device.map_memory(memory, 0, byte_len, vk::MemoryMapFlags::empty())? };
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            ptr as *mut u8,
            byte_len as usize,
        );
        device.unmap_memory(memory);
    }
    Ok(())
}

fn zero_buffer(
    device: &Device,
    memory: vk::DeviceMemory,
    byte_len: usize,
) -> Result<(), GpuMatvecError> {
    let ptr =
        unsafe { device.map_memory(memory, 0, byte_len as u64, vk::MemoryMapFlags::empty())? };
    unsafe {
        std::ptr::write_bytes(ptr, 0, byte_len);
        device.unmap_memory(memory);
    }
    Ok(())
}

fn read_f32_buffer(
    device: &Device,
    memory: vk::DeviceMemory,
    len: usize,
) -> Result<Vec<f32>, GpuMatvecError> {
    let byte_len = len * std::mem::size_of::<f32>();
    let ptr =
        unsafe { device.map_memory(memory, 0, byte_len as u64, vk::MemoryMapFlags::empty())? };
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), len);
        device.unmap_memory(memory);
    }
    Ok(out)
}

fn destroy_buffer(device: &Device, buffer: BufferAllocation) {
    unsafe {
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
