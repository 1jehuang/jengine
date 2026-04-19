use ash::{Entry, vk};
use std::ffi::{CStr, CString};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueFamilyInfo {
    pub index: u32,
    pub queue_count: u32,
    pub supports_graphics: bool,
    pub supports_compute: bool,
    pub supports_transfer: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalDeviceInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: String,
    pub api_version: u32,
    pub driver_version: u32,
    pub queue_families: Vec<QueueFamilyInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanReport {
    pub api_version: u32,
    pub devices: Vec<PhysicalDeviceInfo>,
}

impl VulkanReport {
    pub fn summarize(&self) -> String {
        format!(
            "instance_api_version={} device_count={}",
            version_string(self.api_version),
            self.devices.len()
        )
    }
}

#[derive(Debug)]
pub enum VulkanError {
    Load(ash::LoadingError),
    Vk(vk::Result),
    Utf8(std::str::Utf8Error),
    CString(std::ffi::NulError),
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(error) => write!(f, "Vulkan loader error: {error}"),
            Self::Vk(error) => write!(f, "Vulkan error: {error:?}"),
            Self::Utf8(error) => write!(f, "UTF-8 error: {error}"),
            Self::CString(error) => write!(f, "CString error: {error}"),
        }
    }
}

impl std::error::Error for VulkanError {}

impl From<ash::LoadingError> for VulkanError {
    fn from(value: ash::LoadingError) -> Self {
        Self::Load(value)
    }
}

impl From<vk::Result> for VulkanError {
    fn from(value: vk::Result) -> Self {
        Self::Vk(value)
    }
}

impl From<std::str::Utf8Error> for VulkanError {
    fn from(value: std::str::Utf8Error) -> Self {
        Self::Utf8(value)
    }
}

impl From<std::ffi::NulError> for VulkanError {
    fn from(value: std::ffi::NulError) -> Self {
        Self::CString(value)
    }
}

pub fn collect_vulkan_report() -> Result<VulkanReport, VulkanError> {
    let entry = unsafe { Entry::load()? };
    let app_name = CString::new("jengine-vulkan-report")?;
    let engine_name = CString::new("jengine")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    let instance = unsafe { entry.create_instance(&create_info, None)? };
    let api_version = match unsafe { entry.try_enumerate_instance_version() }? {
        Some(version) => version,
        None => vk::API_VERSION_1_0,
    };

    let devices = unsafe { instance.enumerate_physical_devices()? };
    let mut device_infos = Vec::with_capacity(devices.len());
    for device in devices {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let queue_properties =
            unsafe { instance.get_physical_device_queue_family_properties(device) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_str()?
            .to_string();
        let queue_families = queue_properties
            .iter()
            .enumerate()
            .map(|(index, props)| QueueFamilyInfo {
                index: index as u32,
                queue_count: props.queue_count,
                supports_graphics: props.queue_flags.contains(vk::QueueFlags::GRAPHICS),
                supports_compute: props.queue_flags.contains(vk::QueueFlags::COMPUTE),
                supports_transfer: props.queue_flags.contains(vk::QueueFlags::TRANSFER),
            })
            .collect::<Vec<_>>();
        device_infos.push(PhysicalDeviceInfo {
            name,
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            device_type: device_type_string(properties.device_type),
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            queue_families,
        });
    }

    unsafe { instance.destroy_instance(None) };
    Ok(VulkanReport {
        api_version,
        devices: device_infos,
    })
}

fn device_type_string(device_type: vk::PhysicalDeviceType) -> String {
    match device_type {
        vk::PhysicalDeviceType::CPU => "cpu".to_string(),
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete-gpu".to_string(),
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated-gpu".to_string(),
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual-gpu".to_string(),
        _ => "other".to_string(),
    }
}

pub fn version_string(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

#[cfg(test)]
mod tests {
    use super::version_string;
    use ash::vk;

    #[test]
    fn formats_api_versions() {
        assert_eq!(
            version_string(vk::make_api_version(0, 1, 3, 281)),
            "1.3.281"
        );
    }
}
