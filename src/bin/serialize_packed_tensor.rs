use jengine::runtime::packed::PackedTensorFile;
use jengine::runtime::repack::pack_ternary_g128;
use jengine::runtime::weights::WeightStore;
use std::time::Instant;

fn main() {
    let weights_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let tensor_name = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "model.layers.0.self_attn.q_proj.weight".to_string());
    let out_path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "/tmp/jengine-qproj.jtpk".to_string());
    let rows = std::env::args()
        .nth(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2048);
    let cols = std::env::args()
        .nth(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2048);

    let store = WeightStore::load_from_file(&weights_path).expect("weights should load");
    let values = store
        .load_vector_f32(&tensor_name)
        .expect("tensor should load");

    let started_at = Instant::now();
    let (packed, report) =
        pack_ternary_g128(&values, vec![rows, cols], 1e-3).expect("tensor should be packable");
    let pack_ms = started_at.elapsed().as_secs_f64() * 1_000.0;

    let packed_file = PackedTensorFile::new(Some(tensor_name), packed);
    let started_at = Instant::now();
    packed_file
        .write_to_path(&out_path)
        .expect("packed tensor should serialize");
    let write_ms = started_at.elapsed().as_secs_f64() * 1_000.0;

    let started_at = Instant::now();
    let loaded =
        PackedTensorFile::read_from_path(&out_path).expect("packed tensor should deserialize");
    let read_ms = started_at.elapsed().as_secs_f64() * 1_000.0;

    let out_size = std::fs::metadata(&out_path)
        .expect("packed file metadata")
        .len();
    println!("{}", report.summarize());
    println!(
        "pack_ms={:.3} write_ms={:.3} read_ms={:.3} out_bytes={} code_bytes={} scale_count={}",
        pack_ms,
        write_ms,
        read_ms,
        out_size,
        loaded.metadata.code_bytes,
        loaded.metadata.scale_count,
    );
}
