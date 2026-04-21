use jengine::report::append_jsonl_record;
use jengine::runtime::reference::ReferenceModel;
use serde_json::json;
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

    let packed = run_stage("load_packed_model", || {
        ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
            .expect("packed model should load")
    });
    let manifest = packed.packed_model_manifest().cloned();

    let (use_attention_qkv, use_mlp_gu) = match variant.as_str() {
        "attention" => (true, false),
        "mlp" => (false, true),
        "combined" => (true, true),
        other => panic!("unsupported variant: {other}"),
    };
    if std::env::var_os("JENGINE_PREWARM_PACKED").is_some() {
        let expected_tokens = packed
            .prompt_analysis(&prompt)
            .expect("prompt analysis should succeed")
            .token_count
            + max_new_tokens;
        run_stage("prewarm_packed", || {
            packed
                .prewarm_packed_decode_caches_with_expected_tokens(
                    expected_tokens,
                    use_attention_qkv,
                    use_mlp_gu,
                    false,
                )
                .expect("packed decode prewarm should succeed")
        });
    }

    let mut total_ms = 0.0f64;
    let mut tok_s_values = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let result = run_stage(&format!("packed_decode_iter_{}", iteration + 1), || {
            packed
                .benchmark_packed_decode(&prompt, max_new_tokens, use_attention_qkv, use_mlp_gu)
                .expect("packed decode benchmark should succeed")
        });
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
        let _ = append_jsonl_record(
            "bench_packed_toks.jsonl",
            &json!({
                "kind": "iteration",
                "benchmark": "bench_packed_toks",
                "variant": variant,
                "iteration": iteration + 1,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "total_ms": total,
                "tok_s": tok_s,
                "summary": result.summarize(),
                "artifact_manifest_sha256": manifest.as_ref().and_then(|m| m.source_file_sha256.clone()),
                "artifact_created_unix_secs": manifest.as_ref().map(|m| m.created_unix_secs),
                "artifact_source_file_bytes": manifest.as_ref().map(|m| m.source_file_bytes),
            }),
        );
    }
    let avg_total_ms = total_ms / iterations as f64;
    let avg_tok_s = tok_s_values.iter().sum::<f64>() / iterations as f64;
    println!(
        "avg_variant={} avg_total_ms={:.3} avg_tok_s={:.3}",
        variant, avg_total_ms, avg_tok_s
    );
    let _ = append_jsonl_record(
        "bench_packed_toks.jsonl",
        &json!({
            "kind": "aggregate",
            "benchmark": "bench_packed_toks",
            "variant": variant,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "iterations": iterations,
            "avg_total_ms": avg_total_ms,
            "avg_tok_s": avg_tok_s,
            "artifact_manifest_sha256": manifest.as_ref().and_then(|m| m.source_file_sha256.clone()),
            "artifact_created_unix_secs": manifest.as_ref().map(|m| m.created_unix_secs),
            "artifact_source_file_bytes": manifest.as_ref().map(|m| m.source_file_bytes),
        }),
    );
}
