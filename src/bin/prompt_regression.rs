use jengine::runtime::reference::ReferenceModel;

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let prompt_file = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "fixtures/prompt_regression.txt".to_string());
    let max_new_tokens = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);

    let prompts = std::fs::read_to_string(&prompt_file).expect("prompt file should read");
    let model = ReferenceModel::load_from_root(&root).expect("reference model should load");

    for (index, prompt) in prompts
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let result = model
            .generate_greedy(prompt, max_new_tokens)
            .expect("prompt regression generation should succeed");
        assert!(
            !result.output_text.trim().is_empty(),
            "output text should not be empty"
        );
        println!(
            "prompt_index={} prompt={:?} {} output={}",
            index,
            prompt,
            result.metrics.summarize(),
            result.output_text
        );
    }
}
