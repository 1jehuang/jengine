use jengine::runtime::packed_model::{load_packed_model_manifest, write_packed_model_artifact};

fn main() {
    let model_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let out_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/jengine-packed-model".to_string());

    let (manifest, summary) = write_packed_model_artifact(&model_root, &out_dir)
        .expect("packed model artifact should write");
    let loaded = load_packed_model_manifest(std::path::Path::new(&out_dir).join("manifest.json"))
        .expect("packed model manifest should load");

    println!("{}", summary.summarize());
    println!(
        "manifest_version={} entries={} architecture={} out_dir={}",
        loaded.version,
        loaded.entries.len(),
        loaded.architecture,
        out_dir,
    );
    assert_eq!(manifest.entries.len(), loaded.entries.len());
}
