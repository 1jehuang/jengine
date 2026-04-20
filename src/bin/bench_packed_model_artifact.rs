use jengine::runtime::packed_model::benchmark_packed_model_artifact;

fn main() {
    let artifact_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/jengine-packed-model".to_string());

    let report = benchmark_packed_model_artifact(&artifact_dir)
        .expect("packed model artifact benchmark should succeed");
    println!("{}", report.summarize());
}
