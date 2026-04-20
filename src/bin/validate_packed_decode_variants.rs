use jengine::runtime::reference::ReferenceModel;

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let artifact_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/home/jeremy/jengine/.artifacts/jengine-packed-model".to_string());
    let prompt_file = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "fixtures/packed_decode_regression.txt".to_string());

    let dense = ReferenceModel::load_from_root(&root).expect("dense model should load");
    let packed = ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
        .expect("packed model should load");
    let prompts = std::fs::read_to_string(&prompt_file).expect("prompt file should read");

    for (index, prompt) in prompts
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let attention = packed
            .compare_prefill_logits_against(&dense, prompt, true, false)
            .expect("attention packed validation should succeed");
        let mlp = packed
            .compare_prefill_logits_against(&dense, prompt, false, true)
            .expect("mlp packed validation should succeed");
        let combined = packed
            .compare_prefill_logits_against(&dense, prompt, true, true)
            .expect("combined packed validation should succeed");

        println!(
            "prompt_index={} prompt={:?} attention={} mlp={} combined={}",
            index,
            prompt,
            attention.summarize(),
            mlp.summarize(),
            combined.summarize(),
        );
    }
}
