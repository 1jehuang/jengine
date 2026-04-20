use jengine::runtime::reference::{HybridProjectionDecodeMetrics, ReferenceModel};
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

fn render(label: &str, metrics: &HybridProjectionDecodeMetrics) -> String {
    format!("variant={label} {}", metrics.summarize())
}

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
    let variant = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "qkv+gu".to_string());
    let out_path = std::env::args().nth(6);

    let model = run_stage("load_model", || {
        ReferenceModel::load_from_root(&root).expect("reference model should load")
    });
    let dense = run_stage("dense_decode", || {
        model
            .generate_greedy(&prompt, max_new_tokens)
            .expect("dense decode should succeed")
    });
    let (label, hybrid) = match variant.as_str() {
        "qkv+gu" => (
            "qkv+gu",
            run_stage("hybrid_qkv_gu", || {
                model
                    .benchmark_cached_hybrid_qkv_gu_decode(&prompt, max_new_tokens, layer_idx)
                    .expect("hybrid qkv+gu decode should succeed")
            }),
        ),
        "qkvo+gu" => (
            "qkvo+gu",
            run_stage("hybrid_qkvo_gu", || {
                model
                    .benchmark_cached_hybrid_qkvo_gu_decode(&prompt, max_new_tokens, layer_idx)
                    .expect("hybrid qkvo+gu decode should succeed")
            }),
        ),
        other => panic!("unsupported variant: {other}"),
    };

    let summary = format!(
        "dense={} output={}\n{}\n",
        dense.metrics.summarize(),
        dense.output_text,
        render(label, &hybrid)
    );
    if let Some(out_path) = out_path {
        std::fs::write(&out_path, &summary).expect("hybrid qkv+gu summary should write");
    }
    print!("{summary}");
}
