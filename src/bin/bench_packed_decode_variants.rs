use jengine::runtime::reference::{PackedDecodeMetrics, ReferenceModel};
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

fn render(label: &str, metrics: &PackedDecodeMetrics) -> String {
    format!("variant={label} {}", metrics.summarize())
}

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
    let variant = std::env::args().nth(5).unwrap_or_else(|| "all".to_string());
    let out_path = std::env::args().nth(6);

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
    if std::env::var_os("JENGINE_PREWARM_PACKED").is_some() {
        let (use_attention_qkv, use_mlp_gu) = match variant.as_str() {
            "attention" => (true, false),
            "mlp" => (false, true),
            "combined" => (true, true),
            "all" => (true, true),
            _ => (true, true),
        };
        let use_attention_full = std::env::var_os("JENGINE_PACKED_ATTENTION_FULL").is_some();
        let use_mlp_full = std::env::var_os("JENGINE_PACKED_MLP_FULL").is_some();
        run_stage("prewarm_packed", || {
            packed_model
                .prewarm_packed_decode_caches(
                    use_attention_qkv,
                    use_mlp_gu,
                    use_attention_full,
                    use_mlp_full,
                )
                .expect("packed decode prewarm should succeed")
        });
    }

    let mut lines = Vec::new();
    if let Some(dense_model) = &dense_model {
        let dense = run_stage("dense_decode", || {
            dense_model
                .generate_greedy(&prompt, max_new_tokens)
                .expect("dense decode should succeed")
        });
        lines.push(format!(
            "dense={} output={}",
            dense.metrics.summarize(),
            dense.output_text,
        ));
    }
    if variant == "attention" || variant == "all" {
        let attention_only = run_stage("packed_attention_decode", || {
            packed_model
                .benchmark_packed_decode(&prompt, max_new_tokens, true, false)
                .expect("packed attention decode should succeed")
        });
        lines.push(render("qkv", &attention_only));
    }
    if variant == "mlp" || variant == "all" {
        let mlp_only = run_stage("packed_mlp_decode", || {
            packed_model
                .benchmark_packed_decode(&prompt, max_new_tokens, false, true)
                .expect("packed mlp decode should succeed")
        });
        lines.push(render("gu", &mlp_only));
    }
    if variant == "combined" || variant == "all" {
        let combined = run_stage("packed_combined_decode", || {
            packed_model
                .benchmark_packed_decode(&prompt, max_new_tokens, true, true)
                .expect("packed combined decode should succeed")
        });
        lines.push(render("qkv+gu", &combined));
    }

    let summary = format!("{}\n", lines.join("\n"));
    if let Some(out_path) = out_path {
        std::fs::write(&out_path, &summary).expect("packed decode summary should write");
    }
    print!("{summary}");
}
