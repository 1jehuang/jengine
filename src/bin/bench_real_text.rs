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
        .unwrap_or(4);
    let iterations = std::env::args()
        .nth(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);

    let model = ReferenceModel::load_from_root(&root).expect("reference model should load");
    let mut total_ms = 0.0f64;
    let mut total_attention_ms = 0.0f64;
    let mut total_mlp_ms = 0.0f64;
    let mut total_qkv_ms = 0.0f64;

    for iteration in 0..iterations {
        let result = model
            .generate_greedy(&prompt, max_new_tokens)
            .expect("real text generation should succeed");
        let metrics = result.metrics;
        let total = metrics.total_duration.as_secs_f64() * 1_000.0;
        let attention = metrics.attention_duration.as_secs_f64() * 1_000.0;
        let mlp = metrics.mlp_duration.as_secs_f64() * 1_000.0;
        let qkv = metrics.qkv_duration.as_secs_f64() * 1_000.0;
        total_ms += total;
        total_attention_ms += attention;
        total_mlp_ms += mlp;
        total_qkv_ms += qkv;
        println!(
            "iteration={} {} output={}",
            iteration + 1,
            metrics.summarize(),
            result.output_text
        );
    }

    println!(
        "avg_total_ms={:.3} avg_qkv_ms={:.3} avg_attention_ms={:.3} avg_mlp_ms={:.3}",
        total_ms / iterations as f64,
        total_qkv_ms / iterations as f64,
        total_attention_ms / iterations as f64,
        total_mlp_ms / iterations as f64,
    );
}
