use jengine::runtime::packed_model::validate_packed_model_artifact;

fn main() {
    let model_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let artifact_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/jengine-packed-model".to_string());

    let report = validate_packed_model_artifact(&model_root, &artifact_dir)
        .expect("packed model artifact validation should succeed");
    println!("{}", report.summarize());
}
