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

fn parse_prompt_ids(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .expect("prompt token ids must be integers")
        })
        .collect()
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let artifact_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/home/jeremy/jengine/.artifacts/jengine-packed-model".to_string());
    let prompt_ids_raw = std::env::args().nth(3).unwrap_or_else(|| "42".to_string());
    let variant = std::env::args().nth(4).unwrap_or_else(|| "all".to_string());
    let out_path = std::env::args().nth(5);

    let prompt_ids = parse_prompt_ids(&prompt_ids_raw);
    assert!(
        !prompt_ids.is_empty(),
        "prompt token id list cannot be empty"
    );

    let run_dense = variant == "all" || variant == "dense";
    let dense_model = if run_dense {
        Some(run_stage("load_dense_model", || {
            ReferenceModel::load_from_root(&root).expect("dense model should load")
        }))
    } else {
        None
    };
    let packed_model = run_stage("load_packed_model", || {
        ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
            .expect("packed model should load")
    });

    let mut lines = Vec::new();
    if let Some(dense_model) = &dense_model {
        let dense = run_stage("dense_step", || {
            dense_model
                .benchmark_dense_step_from_token_ids(&prompt_ids)
                .expect("dense step should succeed")
        });
        lines.push(format!("variant=dense {}", dense.summarize()));
    }
    if variant == "attention" || variant == "all" {
        let metrics = run_stage("packed_attention_step", || {
            packed_model
                .benchmark_packed_step_from_token_ids(&prompt_ids, true, false)
                .expect("packed attention step should succeed")
        });
        lines.push(format!("variant=attention {}", metrics.summarize()));
    }
    if variant == "mlp" || variant == "all" {
        let metrics = run_stage("packed_mlp_step", || {
            packed_model
                .benchmark_packed_step_from_token_ids(&prompt_ids, false, true)
                .expect("packed mlp step should succeed")
        });
        lines.push(format!("variant=mlp {}", metrics.summarize()));
    }
    if variant == "combined" || variant == "all" {
        let metrics = run_stage("packed_combined_step", || {
            packed_model
                .benchmark_packed_step_from_token_ids(&prompt_ids, true, true)
                .expect("packed combined step should succeed")
        });
        lines.push(format!("variant=combined {}", metrics.summarize()));
    }

    let summary = format!("{}\n", lines.join("\n"));
    if let Some(out_path) = out_path {
        std::fs::write(&out_path, &summary).expect("step benchmark summary should write");
    }
    print!("{summary}");
}
