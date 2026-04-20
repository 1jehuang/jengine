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

fn parse_variant(variant: &str) -> (bool, bool) {
    match variant {
        "attention" => (true, false),
        "mlp" => (false, true),
        "combined" => (true, true),
        other => panic!("unsupported variant: {other}"),
    }
}

fn merge_metrics(aggregate: &mut PackedDecodeMetrics, next: &PackedDecodeMetrics) {
    aggregate.total_duration += next.total_duration;
    aggregate.embedding_duration += next.embedding_duration;
    aggregate.norm_duration += next.norm_duration;
    aggregate.qkv_duration += next.qkv_duration;
    aggregate.attention_duration += next.attention_duration;
    aggregate.mlp_duration += next.mlp_duration;
    aggregate.mlp_swiglu_duration += next.mlp_swiglu_duration;
    aggregate.mlp_down_duration += next.mlp_down_duration;
    aggregate.mlp_residual_duration += next.mlp_residual_duration;
    aggregate.logits_duration += next.logits_duration;
    aggregate.pack_duration += next.pack_duration;
    aggregate.compile_duration += next.compile_duration;
    aggregate.weight_upload_duration += next.weight_upload_duration;
    aggregate.activation_upload_duration += next.activation_upload_duration;
    aggregate.upload_duration += next.upload_duration;
    aggregate.gpu_duration += next.gpu_duration;
    aggregate.download_duration += next.download_duration;
    aggregate.non_offloaded_dense_duration += next.non_offloaded_dense_duration;
    aggregate.orchestration_duration += next.orchestration_duration;
    aggregate.pack_cache_hits += next.pack_cache_hits;
    aggregate.gpu_cache_hits += next.gpu_cache_hits;
    aggregate.dispatch_count += next.dispatch_count;
    aggregate.weight_upload_bytes += next.weight_upload_bytes;
    aggregate.activation_upload_bytes += next.activation_upload_bytes;
    aggregate.upload_bytes += next.upload_bytes;
    aggregate.download_bytes += next.download_bytes;
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
    let variant = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "combined".to_string());
    let chunk_size = std::env::args()
        .nth(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(7);
    let iterations = std::env::args()
        .nth(6)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);

    let (use_attention_qkv, use_mlp_gu) = parse_variant(&variant);
    let packed_model = run_stage("load_packed_model", || {
        ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
            .expect("packed model should load")
    });

    let layers = packed_model.config.num_hidden_layers;
    for iteration in 0..iterations {
        let mut hidden = None::<Vec<f32>>;
        let mut aggregate = None::<PackedDecodeMetrics>;
        for start_layer in (0..layers).step_by(chunk_size) {
            let end_layer = (start_layer + chunk_size).min(layers);
            let include_logits = end_layer == layers;
            let (hidden_out, metrics) = run_stage(
                &format!("iter_{}_chunk_{}_{}", iteration + 1, start_layer, end_layer),
                || {
                    packed_model
                        .benchmark_packed_prefill_chunk(
                            if hidden.is_some() {
                                None
                            } else {
                                Some(token_id)
                            },
                            hidden.as_deref(),
                            start_layer,
                            end_layer,
                            use_attention_qkv,
                            use_mlp_gu,
                            include_logits,
                        )
                        .expect("packed prefill chunk should succeed")
                },
            );
            println!(
                "iteration={} chunk_variant={} start_layer={} end_layer={} include_logits={} {}",
                iteration + 1,
                variant,
                start_layer,
                end_layer,
                include_logits,
                metrics.summarize()
            );
            hidden = Some(hidden_out);
            if let Some(aggregate_metrics) = &mut aggregate {
                merge_metrics(aggregate_metrics, &metrics);
            } else {
                aggregate = Some(metrics);
            }
        }

        let aggregate = aggregate.expect("at least one chunk should run");
        println!(
            "iteration={} aggregate_variant={} {}",
            iteration + 1,
            variant,
            aggregate.summarize()
        );
    }
}
