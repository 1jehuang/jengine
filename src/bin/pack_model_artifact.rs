use jengine::runtime::packed_model::{load_packed_model_manifest, write_packed_model_artifact};
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

fn main() {
    let model_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let out_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/jengine-packed-model".to_string());

    let (manifest, summary) = run_stage("pack_model_artifact", || {
        write_packed_model_artifact(&model_root, &out_dir)
            .expect("packed model artifact should write")
    });
    let loaded = load_packed_model_manifest(std::path::Path::new(&out_dir).join("manifest.json"))
        .expect("packed model manifest should load");

    println!("{}", summary.summarize());
    println!(
        "manifest_version={} entries={} architecture={} out_dir={}",
        loaded.version,
        loaded.entries.len(),
        loaded.architecture,
        out_dir,
    );
    assert_eq!(manifest.entries.len(), loaded.entries.len());
}
