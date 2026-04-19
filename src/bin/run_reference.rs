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

    let model = ReferenceModel::load_from_root(&root).expect("reference model should load");
    let result = model
        .generate_greedy(&prompt, max_new_tokens)
        .expect("reference generation should succeed");
    println!("{}", result.metrics.summarize());
    println!("{}", result.output_text);
}
