use jengine::runtime::reference::ReferenceModel;

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let prompt_ids_arg = std::env::args().nth(2).unwrap_or_else(|| "42".to_string());
    let max_new_tokens = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);

    let prompt_ids = prompt_ids_arg
        .split(',')
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .expect("prompt token ids must be usize")
        })
        .collect::<Vec<_>>();

    let model =
        ReferenceModel::load_core_from_root(&root).expect("reference core model should load");
    let result = model
        .generate_from_token_ids(&prompt_ids, max_new_tokens)
        .expect("reference generation from token ids should succeed");
    println!("{}", result.metrics.summarize());
    println!("{}", result.output_text);
}
