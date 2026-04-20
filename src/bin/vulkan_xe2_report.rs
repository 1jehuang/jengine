use ash::{Entry, vk};
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};

fn main() {
    let entry = unsafe { Entry::load().expect("Vulkan loader should be available") };
    let app_name = CString::new("jengine-vulkan-xe2-report").expect("CString should build");
    let engine_name = CString::new("jengine").expect("CString should build");
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Vulkan instance should create")
    };

    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("physical devices should enumerate")
    };

    for device in devices {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_str()
            .expect("device name should decode")
            .to_string();

        let extensions = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .expect("device extensions should enumerate")
        };
        let extension_names = extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
            .map(|name| {
                name.to_str()
                    .expect("extension name should decode")
                    .to_string()
            })
            .collect::<BTreeSet<_>>();

        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut subgroup_size_control_props =
            vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup_props)
            .push_next(&mut subgroup_size_control_props);
        unsafe { instance.get_physical_device_properties2(device, &mut properties2) };

        let mut integer_dot_features = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut float16_int8_features = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut storage8_features = vk::PhysicalDevice8BitStorageFeatures::default();
        let mut storage16_features = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut subgroup_size_control_features =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut integer_dot_features)
            .push_next(&mut float16_int8_features)
            .push_next(&mut storage8_features)
            .push_next(&mut storage16_features)
            .push_next(&mut subgroup_size_control_features);
        unsafe { instance.get_physical_device_features2(device, &mut features2) };

        println!(
            "device name={} vendor=0x{:04x} device=0x{:04x} api={}.{}.{} type={:?}",
            name,
            properties.vendor_id,
            properties.device_id,
            vk::api_version_major(properties.api_version),
            vk::api_version_minor(properties.api_version),
            vk::api_version_patch(properties.api_version),
            properties.device_type,
        );
        println!(
            "  subgroup_size={} supported_stages=0x{:x} supported_ops=0x{:x} quad_ops_all_stages={}",
            subgroup_props.subgroup_size,
            subgroup_props.supported_stages.as_raw(),
            subgroup_props.supported_operations.as_raw(),
            subgroup_props.quad_operations_in_all_stages == vk::TRUE,
        );
        println!(
            "  subgroup_size_control feature={} compute_full_subgroups={} min_subgroup_size={} max_subgroup_size={} required_stages=0x{:x}",
            subgroup_size_control_features.subgroup_size_control == vk::TRUE,
            subgroup_size_control_features.compute_full_subgroups == vk::TRUE,
            subgroup_size_control_props.min_subgroup_size,
            subgroup_size_control_props.max_subgroup_size,
            subgroup_size_control_props
                .required_subgroup_size_stages
                .as_raw(),
        );
        println!(
            "  integer_dot={} float16={} int8={} storage_buffer_8bit={} storage_buffer_16bit={}",
            integer_dot_features.shader_integer_dot_product == vk::TRUE,
            float16_int8_features.shader_float16 == vk::TRUE,
            float16_int8_features.shader_int8 == vk::TRUE,
            storage8_features.storage_buffer8_bit_access == vk::TRUE,
            storage16_features.storage_buffer16_bit_access == vk::TRUE,
        );

        let interesting = [
            "VK_KHR_cooperative_matrix",
            "VK_KHR_shader_integer_dot_product",
            "VK_EXT_subgroup_size_control",
            "VK_KHR_shader_float16_int8",
            "VK_KHR_8bit_storage",
            "VK_KHR_16bit_storage",
        ];
        for extension in interesting {
            println!(
                "  extension {}={}",
                extension,
                extension_names.contains(extension)
            );
        }
    }

    unsafe { instance.destroy_instance(None) };
}
