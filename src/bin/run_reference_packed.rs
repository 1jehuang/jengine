use jengine::runtime::reference::ReferenceModel;

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let artifact_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/home/jeremy/jengine/.artifacts/jengine-packed-model".to_string());
    let prompt = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "hello".to_string());
    let max_new_tokens = std::env::args()
        .nth(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let variant = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "combined".to_string());

    let (use_attention_qkv, use_mlp_gu) = match variant.as_str() {
        "attention" => (true, false),
        "mlp" => (false, true),
        "combined" => (true, true),
        other => panic!("unsupported variant: {other}"),
    };

    let model = ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
        .expect("packed reference model should load");
    if std::env::var_os("JENGINE_PREWARM_PACKED").is_some() {
        model
            .prewarm_packed_decode_caches(use_attention_qkv, use_mlp_gu)
            .expect("packed decode prewarm should succeed");
    }
    let result = model
        .generate_packed_greedy(&prompt, max_new_tokens, use_attention_qkv, use_mlp_gu)
        .expect("packed reference generation should succeed");
    println!("{}", result.metrics.summarize());
    println!("{}", result.output_text);
}
