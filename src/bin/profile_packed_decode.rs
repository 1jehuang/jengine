use jengine::report::{timestamped_profile_path, write_json_value};
use jengine::runtime::reference::{DecodeMetrics, PackedDecodeMetrics, ReferenceModel};
use serde_json::json;
use std::path::PathBuf;
use std::time::Duration;

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn decode_metrics_json(metrics: &DecodeMetrics) -> serde_json::Value {
    json!({
        "prompt_tokens": metrics.prompt_tokens,
        "generated_tokens": metrics.generated_tokens,
        "total_ms": ms(metrics.total_duration),
        "embed_ms": ms(metrics.embedding_duration),
        "norm_ms": ms(metrics.norm_duration),
        "qkv_ms": ms(metrics.qkv_duration),
        "attention_ms": ms(metrics.attention_duration),
        "mlp_ms": ms(metrics.mlp_duration),
        "logits_ms": ms(metrics.logits_duration),
    })
}

fn packed_metrics_json(metrics: &PackedDecodeMetrics) -> serde_json::Value {
    json!({
        "enabled": metrics.enabled_projections,
        "total_ms": ms(metrics.total_duration),
        "embed_ms": ms(metrics.embedding_duration),
        "norm_ms": ms(metrics.norm_duration),
        "qkv_ms": ms(metrics.qkv_duration),
        "attention_ms": ms(metrics.attention_duration),
        "attention_query_ms": ms(metrics.attention_query_duration),
        "attention_oproj_ms": ms(metrics.attention_oproj_duration),
        "attention_residual_ms": ms(metrics.attention_residual_duration),
        "mlp_ms": ms(metrics.mlp_duration),
        "mlp_swiglu_ms": ms(metrics.mlp_swiglu_duration),
        "mlp_down_ms": ms(metrics.mlp_down_duration),
        "mlp_residual_ms": ms(metrics.mlp_residual_duration),
        "logits_ms": ms(metrics.logits_duration),
        "pack_ms": ms(metrics.pack_duration),
        "compile_ms": ms(metrics.compile_duration),
        "weight_upload_ms": ms(metrics.weight_upload_duration),
        "activation_upload_ms": ms(metrics.activation_upload_duration),
        "upload_ms": ms(metrics.upload_duration),
        "gpu_ms": ms(metrics.gpu_duration),
        "download_ms": ms(metrics.download_duration),
        "non_offloaded_dense_ms": ms(metrics.non_offloaded_dense_duration),
        "orchestration_ms": ms(metrics.orchestration_duration),
        "pack_cache_hits": metrics.pack_cache_hits,
        "gpu_cache_hits": metrics.gpu_cache_hits,
        "dispatch_count": metrics.dispatch_count,
        "weight_upload_bytes": metrics.weight_upload_bytes,
        "activation_upload_bytes": metrics.activation_upload_bytes,
        "upload_bytes": metrics.upload_bytes,
        "download_bytes": metrics.download_bytes,
        "streamed_bytes": metrics.total_streamed_bytes(),
        "e2e_gbps": metrics.effective_end_to_end_bandwidth_gbps(),
        "stream_window_gbps": metrics.effective_stream_window_bandwidth_gbps(),
    })
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
    let out_path = std::env::args().nth(6);

    let (use_attention_qkv, use_mlp_gu) = match variant.as_str() {
        "attention" => (true, false),
        "mlp" => (false, true),
        "combined" => (true, true),
        other => panic!("unsupported variant: {other}"),
    };

    let model = ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
        .expect("packed reference model should load");
    if std::env::var_os("JENGINE_PREWARM_PACKED").is_some() {
        let use_attention_full = std::env::var_os("JENGINE_PACKED_ATTENTION_FULL").is_some();
        let use_mlp_full = std::env::var_os("JENGINE_PACKED_MLP_FULL").is_some();
        model
            .prewarm_packed_decode_caches(
                use_attention_qkv,
                use_mlp_gu,
                use_attention_full,
                use_mlp_full,
            )
            .expect("packed decode prewarm should succeed");
    }

    let result = model
        .generate_packed_greedy(&prompt, max_new_tokens, use_attention_qkv, use_mlp_gu)
        .expect("packed profile run should succeed");

    let document = json!({
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "variant": variant,
        "output_text": result.output_text,
        "output_token_ids": result.output_token_ids,
        "decode_metrics": decode_metrics_json(&result.decode_metrics),
        "packed_metrics": packed_metrics_json(&result.metrics),
        "dispatch_trace": result.dispatch_trace,
    });
    let output = serde_json::to_string_pretty(&document).expect("profile JSON should serialize");

    let out_path = out_path.map(PathBuf::from).unwrap_or_else(|| {
        timestamped_profile_path("profile_packed_decode", "json")
            .expect("profile log path should build")
    });
    write_json_value(&out_path, &document).expect("profile JSON should write");
    eprintln!("profile_json_path={}", out_path.display());
    println!("{output}");
}
