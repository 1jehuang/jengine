use ash::util::read_spv;
use ash::{Entry, Instance, vk};
use jengine::runtime::weights::WeightStore;
use std::ffi::CString;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_ptr: *mut u8,
}

struct Context {
    _entry: Entry,
    instance: Instance,
    device: ash::Device,
    queue: vk::Queue,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
}

impl Context {
    fn new() -> Result<Self> {
        let entry = unsafe { Entry::load()? };
        let app_name = CString::new("jengine-rms-norm-probe")?;
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
            queue,
            physical_device,
            queue_family_index,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct RmsNormRunner<'a> {
    context: &'a Context,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    _descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    _command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    input_buffer: BufferAllocation,
    weight_buffer: BufferAllocation,
    output_buffer: BufferAllocation,
}

impl<'a> RmsNormRunner<'a> {
    fn new(context: &'a Context, len: usize) -> Result<Self> {
        let shader_path = compile_shader(Path::new("shaders/weighted_rms_norm.comp"))?;
        let input_buffer = create_host_visible_buffer(
            &context.instance,
            &context.device,
            context.physical_device,
            (len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let weight_buffer = create_host_visible_buffer(
            &context.instance,
            &context.device,
            context.physical_device,
            (len * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let output_buffer = create_host_visible_buffer(
            &context.instance,
            &context.device,
            context.physical_device,
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
        let descriptor_set_layout = unsafe {
            context
                .device
                .create_descriptor_set_layout(&descriptor_layout_ci, None)?
        };
        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        let pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout(&pipeline_layout_ci, None)?
        };
        let shader_module = create_shader_module(&context.device, &shader_path)?;
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
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
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
        unsafe { context.device.update_descriptor_sets(&writes, &[]) };
        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(context.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { context.device.create_command_pool(&command_pool_ci, None)? };
        let command_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { context.device.allocate_command_buffers(&command_alloc)?[0] };
        unsafe {
            context.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;
            context.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            context.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            let epsilon = 1e-6f32.to_bits();
            let push_constants = [len as u32, epsilon];
            let push_bytes = std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                push_constants.len() * 4,
            );
            context.device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            context.device.cmd_dispatch(command_buffer, 1, 1, 1);
            context.device.end_command_buffer(command_buffer)?;
        }
        let fence = unsafe { context.device.create_fence(&vk::FenceCreateInfo::default(), None)? };
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
            input_buffer,
            weight_buffer,
            output_buffer,
        })
    }

    fn run(&mut self, input: &[f32], weight: &[f32]) -> Result<(Vec<f32>, f64)> {
        write_f32_buffer(&self.input_buffer, input)?;
        write_f32_buffer(&self.weight_buffer, weight)?;
        unsafe { self.context.device.reset_fences(&[self.fence])? };
        let submit_info = [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(
            &self._command_buffer,
        ))];
        let started = Instant::now();
        unsafe {
            self.context
                .device
                .queue_submit(self.context.queue, &submit_info, self.fence)?;
            self.context
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        let gpu_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let output = read_f32_buffer(&self.output_buffer, input.len())?;
        Ok((output, gpu_ms))
    }
}

impl<'a> Drop for RmsNormRunner<'a> {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_fence(self.fence, None);
            self.context.device.destroy_command_pool(self.command_pool, None);
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
            destroy_buffer(&self.context.device, self.input_buffer);
            destroy_buffer(&self.context.device, self.weight_buffer);
            destroy_buffer(&self.context.device, self.output_buffer);
        }
    }
}

fn main() -> Result<()> {
    let weights_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let tensor_name = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "model.norm.weight".to_string());

    let store = WeightStore::load_from_file(&weights_path)?;
    let weight = store.load_vector_f32(&tensor_name)?;
    let len = weight.len();
    let input = (0..len)
        .map(|i| (i % 17) as f32 * 0.03125 - 0.25)
        .collect::<Vec<_>>();

    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / len as f32;
    let scale = 1.0 / (mean_square + 1e-6).sqrt();
    let reference = input
        .iter()
        .zip(weight.iter())
        .map(|(value, gamma)| value * scale * gamma)
        .collect::<Vec<_>>();

    let context = Context::new()?;
    let mut runner = RmsNormRunner::new(&context, len)?;
    let (output, gpu_ms) = runner.run(&input, &weight)?;

    let (max_abs_diff, mean_abs_diff) = reference
        .iter()
        .zip(output.iter())
        .fold((0.0f32, 0.0f32), |(max_diff, sum), (left, right)| {
            let diff = (left - right).abs();
            (max_diff.max(diff), sum + diff)
        });
    let mean_abs_diff = mean_abs_diff / len as f32;

    println!(
        "len={} gpu_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
        len, gpu_ms, max_abs_diff, mean_abs_diff
    );
    Ok(())
}

fn compile_shader(path: &Path) -> Result<PathBuf> {
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

fn pick_compute_device(instance: &Instance) -> Result<(vk::PhysicalDevice, u32)> {
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

fn create_shader_module(device: &ash::Device, path: &Path) -> Result<vk::ShaderModule> {
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
) -> Result<BufferAllocation> {
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
    })
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    filter: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32> {
    for i in 0..props.memory_type_count {
        let matches_type = (filter & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize].property_flags.contains(flags);
        if matches_type && has_flags {
            return Ok(i);
        }
    }
    Err("no matching Vulkan memory type found".into())
}

fn write_f32_buffer(buffer: &BufferAllocation, data: &[f32]) -> Result<()> {
    let byte_len = std::mem::size_of_val(data) as u64;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buffer.mapped_ptr, byte_len as usize);
    }
    Ok(())
}

fn zero_buffer(buffer: &BufferAllocation, byte_len: usize) -> Result<()> {
    unsafe {
        std::ptr::write_bytes(buffer.mapped_ptr, 0, byte_len);
    }
    Ok(())
}

fn read_f32_buffer(buffer: &BufferAllocation, len: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.mapped_ptr as *const f32, out.as_mut_ptr(), len);
    }
    Ok(out)
}

fn destroy_buffer(device: &ash::Device, buffer: BufferAllocation) {
    unsafe {
        device.unmap_memory(buffer.memory);
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }
}
