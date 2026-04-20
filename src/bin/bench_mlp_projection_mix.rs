use jengine::runtime::reference::{MlpProjectionMixMetrics, ReferenceModel};
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

fn render(label: &str, metrics: &MlpProjectionMixMetrics) -> String {
    format!("variant={label} {}", metrics.summarize())
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let artifact_dir = std::env::args().nth(2);
    let layer_idx = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let token_id = std::env::args()
        .nth(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(42);
    let variant = std::env::args().nth(5).unwrap_or_else(|| "all".to_string());
    let out_path = std::env::args().nth(6);

    let model = run_stage("load_model", || match artifact_dir.as_deref() {
        Some(dir) => ReferenceModel::load_from_root_with_packed_artifact(&root, dir)
            .expect("packed reference model should load"),
        None => ReferenceModel::load_from_root(&root).expect("reference model should load"),
    });
    let mut lines = Vec::new();
    if variant == "gu" || variant == "all" {
        let gu = run_stage("variant_gu", || {
            model
                .benchmark_mlp_projection_mix(layer_idx, token_id, true, true, false)
                .expect("gu variant should succeed")
        });
        lines.push(render("gu", &gu));
    }
    if variant == "gud" || variant == "all" {
        let gud = run_stage("variant_gud", || {
            model
                .benchmark_mlp_projection_mix(layer_idx, token_id, true, true, true)
                .expect("gud variant should succeed")
        });
        lines.push(render("gud", &gud));
    }

    let summary = format!("{}\n", lines.join("\n"));
    if let Some(out_path) = out_path {
        std::fs::write(&out_path, &summary).expect("mlp mix summary should write");
    }
    print!("{summary}");
}
