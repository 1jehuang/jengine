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
    let iterations = std::env::args()
        .nth(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let variant = std::env::args()
        .nth(6)
        .unwrap_or_else(|| "combined".to_string());

    let packed = ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
        .expect("packed model should load");

    let (use_attention_qkv, use_mlp_gu) = match variant.as_str() {
        "attention" => (true, false),
        "mlp" => (false, true),
        "combined" => (true, true),
        other => panic!("unsupported variant: {other}"),
    };

    let mut total_ms = 0.0f64;
    let mut tok_s_values = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let result = packed
            .benchmark_packed_decode(&prompt, max_new_tokens, use_attention_qkv, use_mlp_gu)
            .expect("packed decode benchmark should succeed");
        let total = result.total_duration.as_secs_f64() * 1_000.0;
        let tok_s = if max_new_tokens == 0 {
            0.0
        } else {
            max_new_tokens as f64 / result.total_duration.as_secs_f64()
        };
        total_ms += total;
        tok_s_values.push(tok_s);
        println!(
            "iteration={} variant={} total_ms={:.3} tok_s={:.3} {}",
            iteration + 1,
            variant,
            total,
            tok_s,
            result.summarize(),
        );
    }
    let avg_total_ms = total_ms / iterations as f64;
    let avg_tok_s = tok_s_values.iter().sum::<f64>() / iterations as f64;
    println!(
        "avg_variant={} avg_total_ms={:.3} avg_tok_s={:.3}",
        variant, avg_total_ms, avg_tok_s
    );
}
