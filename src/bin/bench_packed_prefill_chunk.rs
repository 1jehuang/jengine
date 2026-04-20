use jengine::runtime::reference::ReferenceModel;
use std::io::{self, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

fn flush_progress(message: &str) {
    println!("{message}");
    let _ = io::stdout().flush();
}

fn run_stage<T>(name: &str, op: impl FnOnce() -> T) -> T {
    flush_progress(&format!("phase={name}:start"));
    if std::env::var_os("JENGINE_NO_HEARTBEAT").is_some() {
        let result = op();
        flush_progress(&format!("phase={name}:done"));
        return result;
    }
    let done = Arc::new(AtomicBool::new(false));
    let done_worker = done.clone();
    let stage_name = name.to_string();
    let worker = thread::spawn(move || {
        while !done_worker.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(500));
            if !done_worker.load(Ordering::Relaxed) {
                println!("phase={stage_name}:heartbeat");
                let _ = io::stdout().flush();
            }
        }
    });
    let result = op();
    done.store(true, Ordering::Relaxed);
    let _ = worker.join();
    flush_progress(&format!("phase={name}:done"));
    result
}

fn parse_variant(variant: &str) -> (bool, bool) {
    match variant {
        "attention" => (true, false),
        "mlp" => (false, true),
        "combined" => (true, true),
        other => panic!("unsupported variant: {other}"),
    }
}

fn load_hidden(path: &str) -> Vec<f32> {
    serde_json::from_str(&std::fs::read_to_string(path).expect("hidden input should read"))
        .expect("hidden input JSON should parse")
}

fn save_hidden(path: &str, hidden: &[f32]) {
    std::fs::write(
        path,
        serde_json::to_string(hidden).expect("hidden output should serialize"),
    )
    .expect("hidden output should write");
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let artifact_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/home/jeremy/jengine/.artifacts/jengine-packed-model".to_string());
    let token_id = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(42);
    let start_layer = std::env::args()
        .nth(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let end_layer = std::env::args()
        .nth(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4);
    let variant = std::env::args()
        .nth(6)
        .unwrap_or_else(|| "combined".to_string());
    let hidden_in_path = std::env::args()
        .nth(7)
        .and_then(|value| if value == "-" { None } else { Some(value) });
    let hidden_out_path = std::env::args()
        .nth(8)
        .and_then(|value| if value == "-" { None } else { Some(value) });
    let summary_out_path = std::env::args()
        .nth(9)
        .and_then(|value| if value == "-" { None } else { Some(value) });
    let include_logits = std::env::args()
        .nth(10)
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(0)
        != 0;

    let (use_attention_qkv, use_mlp_gu) = parse_variant(&variant);
    let hidden_in = hidden_in_path.as_deref().map(load_hidden);

    let packed_model = run_stage("load_packed_model", || {
        ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
            .expect("packed model should load")
    });
    let (hidden_out, metrics) = run_stage("packed_prefill_chunk", || {
        packed_model
            .benchmark_packed_prefill_chunk(
                if hidden_in.is_some() {
                    None
                } else {
                    Some(token_id)
                },
                hidden_in.as_deref(),
                start_layer,
                end_layer,
                use_attention_qkv,
                use_mlp_gu,
                include_logits,
            )
            .expect("packed prefill chunk should succeed")
    });

    if let Some(hidden_out_path) = hidden_out_path.as_deref() {
        save_hidden(hidden_out_path, &hidden_out);
    }

    let summary = format!(
        "variant={} start_layer={} end_layer={} include_logits={} {}\n",
        variant,
        start_layer,
        end_layer,
        include_logits,
        metrics.summarize(),
    );
    if let Some(summary_out_path) = summary_out_path {
        std::fs::write(&summary_out_path, &summary).expect("summary output should write");
    }
    print!("{summary}");
}
