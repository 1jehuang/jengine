use jengine::runtime::reference::ReferenceModel;

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "hello".to_string());
    let max_new_tokens = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let layer_idx = std::env::args()
        .nth(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);

    let model = ReferenceModel::load_from_root(&root).expect("reference model should load");
    let dense = model
        .generate_greedy(&prompt, max_new_tokens)
        .expect("dense decode should succeed");
    let hybrid = model
        .benchmark_hybrid_qproj_decode(&prompt, max_new_tokens, layer_idx)
        .expect("hybrid decode should succeed");

    println!(
        "dense={} output={}",
        dense.metrics.summarize(),
        dense.output_text
    );
    println!("hybrid={}", hybrid.summarize());
}
