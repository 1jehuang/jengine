use jengine::runtime::weights::WeightStore;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());

    let progress =
        WeightStore::download_progress(&path).expect("download progress should be available");
    let total_tensors = progress.total_tensors as u64;
    let file_bytes = progress.file_bytes;

    let packed_2bit_g128_bytes = ((file_bytes as f64) / 16.0 * 2.125).round() as u64;
    let theoretical_1_58bit_bytes = ((file_bytes as f64) / 16.0 * 1.585).round() as u64;

    println!("source_file_bytes={file_bytes}");
    println!("header_tensors={total_tensors}");
    println!("packed_g128_estimated_bytes={packed_2bit_g128_bytes}");
    println!("theoretical_1_58bit_bytes={theoretical_1_58bit_bytes}");
    println!(
        "estimated_g128_reduction_x={:.3}",
        file_bytes as f64 / packed_2bit_g128_bytes as f64
    );
    println!(
        "theoretical_reduction_x={:.3}",
        file_bytes as f64 / theoretical_1_58bit_bytes as f64
    );
}
