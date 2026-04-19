use jengine::gpu::vulkan::{collect_vulkan_report, version_string};

fn main() {
    let report = collect_vulkan_report().expect("vulkan report should succeed");
    println!("{}", report.summarize());
    for device in report.devices {
        println!(
            "device name={} type={} vendor=0x{:04x} device=0x{:04x} api={} driver={} queues={}",
            device.name,
            device.device_type,
            device.vendor_id,
            device.device_id,
            version_string(device.api_version),
            device.driver_version,
            device.queue_families.len(),
        );
        for queue in device.queue_families {
            println!(
                "  queue index={} count={} graphics={} compute={} transfer={}",
                queue.index,
                queue.queue_count,
                queue.supports_graphics,
                queue.supports_compute,
                queue.supports_transfer,
            );
        }
    }
}
