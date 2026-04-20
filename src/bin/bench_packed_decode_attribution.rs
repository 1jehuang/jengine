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
    let hardware_bandwidth_gbps = std::env::args()
        .nth(6)
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(137.0);
    let iterations = std::env::args()
        .nth(7)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let out_path = std::env::args().nth(8);

    let (use_attention_qkv, use_mlp_gu) = parse_variant(&variant);
    let model = run_stage("load_packed_model", || {
        ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
            .expect("packed model should load")
    });
    if std::env::var_os("JENGINE_PREWARM_PACKED").is_some() {
        run_stage("prewarm_packed", || {
            model
                .prewarm_packed_decode_caches(use_attention_qkv, use_mlp_gu)
                .expect("packed decode prewarm should succeed")
        });
    }
    let mut lines = Vec::with_capacity(iterations + 1);
    let mut total_ms_sum = 0.0f64;
    let mut e2e_gbps_sum = 0.0f64;
    let mut stream_window_gbps_sum = 0.0f64;
    for iteration in 0..iterations {
        let metrics = run_stage(
            &format!("packed_decode_attribution_iter_{}", iteration + 1),
            || {
                model
                    .benchmark_packed_decode(&prompt, max_new_tokens, use_attention_qkv, use_mlp_gu)
                    .expect("packed decode attribution benchmark should succeed")
            },
        );

        let generated = max_new_tokens.max(1) as f64;
        let per_generated_token_ms = metrics.total_duration.as_secs_f64() * 1_000.0 / generated;
        let per_generated_streamed_mb =
            metrics.total_streamed_bytes() as f64 / generated / 1_000_000.0;
        let e2e_pct_of_hw = if hardware_bandwidth_gbps <= f64::EPSILON {
            0.0
        } else {
            metrics.effective_end_to_end_bandwidth_gbps() / hardware_bandwidth_gbps * 100.0
        };
        let stream_window_pct_of_hw = if hardware_bandwidth_gbps <= f64::EPSILON {
            0.0
        } else {
            metrics.effective_stream_window_bandwidth_gbps() / hardware_bandwidth_gbps * 100.0
        };
        total_ms_sum += metrics.total_duration.as_secs_f64() * 1_000.0;
        e2e_gbps_sum += metrics.effective_end_to_end_bandwidth_gbps();
        stream_window_gbps_sum += metrics.effective_stream_window_bandwidth_gbps();
        lines.push(format!(
            "iteration={} variant={} prompt={} max_new_tokens={} per_generated_token_ms={:.3} per_generated_streamed_mb={:.3} hardware_gbps={:.3} e2e_pct_of_hw={:.3} stream_window_pct_of_hw={:.3} {}",
            iteration + 1,
            variant,
            prompt,
            max_new_tokens,
            per_generated_token_ms,
            per_generated_streamed_mb,
            hardware_bandwidth_gbps,
            e2e_pct_of_hw,
            stream_window_pct_of_hw,
            metrics.summarize(),
        ));
    }
    lines.push(format!(
        "avg_variant={} iterations={} avg_total_ms={:.3} avg_e2e_gbps={:.3} avg_stream_window_gbps={:.3}",
        variant,
        iterations,
        total_ms_sum / iterations as f64,
        e2e_gbps_sum / iterations as f64,
        stream_window_gbps_sum / iterations as f64,
    ));
    let summary = format!("{}\n", lines.join("\n"));

    if let Some(out_path) = out_path {
        std::fs::write(&out_path, &summary).expect("attribution summary should write");
    }
    print!("{summary}");
}
