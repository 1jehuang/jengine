use ash::util::read_spv;
use ash::{Entry, Instance, vk};
use std::ffi::CString;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

type ProbeResult<T> = Result<T, Box<dyn std::error::Error>>;

struct SharedContext {
    _entry: Entry,
    instance: Instance,
    device: ash::Device,
    _queue: vk::Queue,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
}

impl SharedContext {
    fn new() -> ProbeResult<Self> {
        let entry = unsafe { Entry::load()? };
        let app_name = CString::new("jengine-shared-runner-probe")?;
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
        Ok(Self {
            _entry: entry,
            instance,
            device,
            _queue: queue,
            physical_device,
            queue_family_index,
        })
    }
}

impl Drop for SharedContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy)]
struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_ptr: *mut u8,
}

struct RunnerProbe<'a> {
    context: &'a SharedContext,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    _descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    _command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    code_buffer: BufferAllocation,
    scale_buffer: BufferAllocation,
    vector_buffer: BufferAllocation,
    output_buffer: BufferAllocation,
}

impl<'a> RunnerProbe<'a> {
    fn new(context: &'a SharedContext, rows: usize, cols: usize) -> ProbeResult<Self> {
        eprintln!("runner: compiling shader");
        let shader_path = compile_shader(Path::new("shaders/packed_ternary_matvec.comp"))?;
        eprintln!("runner: creating buffers");
        let code_bytes = 4096u64;
        let scale_bytes = 4096u64;
        eprintln!("runner: create code buffer");
        let code_buffer = create_host_visible_buffer(
            &context.instance,
            &context.device,
            context.physical_device,
            code_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        eprintln!("runner: create scale buffer");
        let scale_buffer = create_host_visible_buffer(
            &context.instance,
            &context.device,
            context.physical_device,
            scale_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        eprintln!("runner: create vector buffer");
        let vector_buffer = create_host_visible_buffer(
            &context.instance,
            &context.device,
            context.physical_device,
            cols.div_ceil(2) as u64 * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        eprintln!("runner: create output buffer");
        let output_buffer = create_host_visible_buffer(
            &context.instance,
            &context.device,
            context.physical_device,
            rows as u64 * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        eprintln!("runner: zero code");
        zero_buffer(&code_buffer, code_bytes as usize)?;
        eprintln!("runner: zero scale");
        zero_buffer(&scale_buffer, scale_bytes as usize)?;
        eprintln!("runner: zero vector");
        zero_buffer(&vector_buffer, cols.div_ceil(2) * 4)?;
        eprintln!("runner: zero output");
        zero_buffer(&output_buffer, rows * 4)?;
        eprintln!("runner: buffers ready");
        eprintln!("runner: creating descriptor set layout");

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
        let descriptor_set_layout = unsafe {
            context
                .device
                .create_descriptor_set_layout(&descriptor_layout_ci, None)?
        };
        eprintln!("runner: creating pipeline layout");
        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        let pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout(&pipeline_layout_ci, None)?
        };
        eprintln!("runner: creating shader module");
        let shader_module = create_shader_module(&context.device, &shader_path)?;
        eprintln!("runner: creating compute pipeline");
        let entry_name = CString::new("main")?;
        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_name);
        let compute_ci = [vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(pipeline_layout)];
        let pipeline = unsafe {
            context
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &compute_ci, None)
                .map_err(|(_, err)| format!("pipeline creation failed: {err:?}"))?[0]
        };
        eprintln!("runner: creating descriptor pool + set");
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4)];
        let descriptor_pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe {
            context
                .device
                .create_descriptor_pool(&descriptor_pool_ci, None)?
        };
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { context.device.allocate_descriptor_sets(&alloc_info)?[0] };
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
        unsafe { context.device.update_descriptor_sets(&writes, &[]) };
        eprintln!("runner: creating command pool + fence");
        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(context.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { context.device.create_command_pool(&command_pool_ci, None)? };
        let command_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { context.device.allocate_command_buffers(&command_alloc)?[0] };
        let fence = unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        eprintln!("runner: complete");
        Ok(Self {
            context,
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
            descriptor_pool,
            _descriptor_set: descriptor_set,
            command_pool,
            _command_buffer: command_buffer,
            fence,
            code_buffer,
            scale_buffer,
            vector_buffer,
            output_buffer,
        })
    }
}

impl<'a> Drop for RunnerProbe<'a> {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_fence(self.fence, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_shader_module(self.shader_module, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            destroy_buffer(&self.context.device, self.code_buffer);
            destroy_buffer(&self.context.device, self.scale_buffer);
            destroy_buffer(&self.context.device, self.vector_buffer);
            destroy_buffer(&self.context.device, self.output_buffer);
        }
    }
}

fn main() -> ProbeResult<()> {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "construct_two".to_string());
    let context = SharedContext::new()?;
    eprintln!("shared context initialized");
    match mode.as_str() {
        "construct_one" => {
            let _runner = RunnerProbe::new(&context, 2, 2)?;
            eprintln!("constructed one runner");
        }
        "construct_two" => {
            let _runner_a = RunnerProbe::new(&context, 2, 2)?;
            eprintln!("constructed runner A");
            let _runner_b = RunnerProbe::new(&context, 2, 2)?;
            eprintln!("constructed runner B");
        }
        "isolated_one" => {
            let isolated = SharedContext::new()?;
            let _runner = RunnerProbe::new(&isolated, 2, 2)?;
            eprintln!("constructed one isolated runner");
        }
        other => return Err(format!("unknown mode: {other}").into()),
    }
    eprintln!("probe exiting cleanly");
    Ok(())
}

fn compile_shader(path: &Path) -> ProbeResult<PathBuf> {
    let temp_dir = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".tmp"));
    std::fs::create_dir_all(&temp_dir)?;
    let shader_stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("shared-probe");
    let out = temp_dir.join(format!("{shader_stem}.spv"));
    let tmp = temp_dir.join(format!(
        "{shader_stem}-{}-{}.tmp.spv",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    let output = Command::new("glslc")
        .arg(path)
        .arg("--target-spv=spv1.3")
        .arg("-o")
        .arg(&tmp)
        .output()?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string().into());
    }
    std::fs::rename(&tmp, &out)?;
    Ok(out)
}

fn pick_compute_device(instance: &Instance) -> ProbeResult<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices()? };
    for device in devices {
        let queues = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, queue) in queues.iter().enumerate() {
            if queue.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Ok((device, index as u32));
            }
        }
    }
    Err("no compute-capable Vulkan device found".into())
}

fn create_shader_module(device: &ash::Device, path: &Path) -> ProbeResult<vk::ShaderModule> {
    let mut cursor = Cursor::new(std::fs::read(path)?);
    let code = read_spv(&mut cursor)?;
    let ci = vk::ShaderModuleCreateInfo::default().code(&code);
    Ok(unsafe { device.create_shader_module(&ci, None)? })
}

fn create_host_visible_buffer(
    instance: &Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> ProbeResult<BufferAllocation> {
    let ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    eprintln!("buffer: create_buffer size={size}");
    let buffer = unsafe { device.create_buffer(&ci, None)? };
    eprintln!("buffer: get requirements");
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    eprintln!("buffer: get memory properties");
    let memory_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    eprintln!("buffer: find memory type");
    let memory_type_index = find_memory_type(
        &memory_props,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    eprintln!("buffer: allocate memory size={}", requirements.size);
    let memory = unsafe { device.allocate_memory(&alloc, None)? };
    eprintln!("buffer: bind buffer memory");
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
    eprintln!("buffer: map memory");
    let mapped_ptr =
        unsafe { device.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())? }
            as *mut u8;
    eprintln!("buffer: allocation complete");
    Ok(BufferAllocation {
        buffer,
        memory,
        mapped_ptr,
    })
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    filter: u32,
    flags: vk::MemoryPropertyFlags,
) -> ProbeResult<u32> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize]
            .property_flags
            .contains(flags);
        if matches_type && has_flags {
            return Ok(i);
        }
    }
    Err("no matching Vulkan memory type found".into())
}

fn zero_buffer(buffer: &BufferAllocation, byte_len: usize) -> ProbeResult<()> {
    unsafe {
        std::ptr::write_bytes(buffer.mapped_ptr, 0, byte_len);
    }
    Ok(())
}

fn destroy_buffer(device: &ash::Device, buffer: BufferAllocation) {
    unsafe {
        device.unmap_memory(buffer.memory);
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }
}
