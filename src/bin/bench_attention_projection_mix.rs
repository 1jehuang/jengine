use jengine::runtime::reference::{AttentionProjectionMixMetrics, ReferenceModel};
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

fn render(label: &str, metrics: &AttentionProjectionMixMetrics) -> String {
    format!("variant={label} {}", metrics.summarize())
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let layer_idx = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let token_id = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(42);
    let out_path = std::env::args().nth(4);

    let model = run_stage("load_model", || {
        ReferenceModel::load_from_root(&root).expect("reference model should load")
    });
    let q = run_stage("variant_q", || {
        model
            .benchmark_attention_projection_mix(layer_idx, token_id, true, false, false, false)
            .expect("q-only variant should succeed")
    });
    let qkv = run_stage("variant_qkv", || {
        model
            .benchmark_attention_projection_mix(layer_idx, token_id, true, true, true, false)
            .expect("qkv variant should succeed")
    });
    let qkvo = run_stage("variant_qkvo", || {
        model
            .benchmark_attention_projection_mix(layer_idx, token_id, true, true, true, true)
            .expect("qkvo variant should succeed")
    });

    let summary = format!(
        "{}\n{}\n{}\n",
        render("q", &q),
        render("qkv", &qkv),
        render("qkvo", &qkvo)
    );
    if let Some(out_path) = out_path {
        std::fs::write(&out_path, &summary).expect("attention mix summary should write");
    }
    print!("{summary}");
}
