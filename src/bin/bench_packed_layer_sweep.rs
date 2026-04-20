use jengine::runtime::reference::{
    AttentionProjectionMixMetrics, MlpProjectionMixMetrics, ReferenceModel,
};
use std::time::Duration;

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

    let model = ReferenceModel::load_from_root_with_packed_artifact(&root, &artifact_dir)
        .expect("packed model should load");

    let mut attn_total = Duration::ZERO;
    let mut attn_pack = Duration::ZERO;
    let mut attn_compile = Duration::ZERO;
    let mut attn_upload = Duration::ZERO;
    let mut attn_gpu = Duration::ZERO;
    let mut attn_download = Duration::ZERO;
    let mut attn_max = 0.0f32;

    let mut mlp_total = Duration::ZERO;
    let mut mlp_pack = Duration::ZERO;
    let mut mlp_compile = Duration::ZERO;
    let mut mlp_upload = Duration::ZERO;
    let mut mlp_gpu = Duration::ZERO;
    let mut mlp_download = Duration::ZERO;
    let mut mlp_max = 0.0f32;

    for layer_idx in 0..model.config.num_hidden_layers {
        let attn = model
            .benchmark_attention_projection_mix(layer_idx, token_id, true, true, true, false)
            .expect("attention sweep should succeed");
        acc_attention(
            &mut attn_total,
            &mut attn_pack,
            &mut attn_compile,
            &mut attn_upload,
            &mut attn_gpu,
            &mut attn_download,
            &mut attn_max,
            &attn,
        );

        let mlp = model
            .benchmark_mlp_projection_mix(layer_idx, token_id, true, true, false)
            .expect("mlp sweep should succeed");
        acc_mlp(
            &mut mlp_total,
            &mut mlp_pack,
            &mut mlp_compile,
            &mut mlp_upload,
            &mut mlp_gpu,
            &mut mlp_download,
            &mut mlp_max,
            &mlp,
        );
    }

    println!(
        "attention_qkv layers={} total_ms={:.3} pack_ms={:.3} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6}",
        model.config.num_hidden_layers,
        attn_total.as_secs_f64() * 1_000.0,
        attn_pack.as_secs_f64() * 1_000.0,
        attn_compile.as_secs_f64() * 1_000.0,
        attn_upload.as_secs_f64() * 1_000.0,
        attn_gpu.as_secs_f64() * 1_000.0,
        attn_download.as_secs_f64() * 1_000.0,
        attn_max,
    );
    println!(
        "mlp_gu layers={} total_ms={:.3} pack_ms={:.3} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6}",
        model.config.num_hidden_layers,
        mlp_total.as_secs_f64() * 1_000.0,
        mlp_pack.as_secs_f64() * 1_000.0,
        mlp_compile.as_secs_f64() * 1_000.0,
        mlp_upload.as_secs_f64() * 1_000.0,
        mlp_gpu.as_secs_f64() * 1_000.0,
        mlp_download.as_secs_f64() * 1_000.0,
        mlp_max,
    );
    println!(
        "combined_qkv_gu layers={} total_ms={:.3} pack_ms={:.3} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3}",
        model.config.num_hidden_layers,
        (attn_total + mlp_total).as_secs_f64() * 1_000.0,
        (attn_pack + mlp_pack).as_secs_f64() * 1_000.0,
        (attn_compile + mlp_compile).as_secs_f64() * 1_000.0,
        (attn_upload + mlp_upload).as_secs_f64() * 1_000.0,
        (attn_gpu + mlp_gpu).as_secs_f64() * 1_000.0,
        (attn_download + mlp_download).as_secs_f64() * 1_000.0,
    );
}

#[allow(clippy::too_many_arguments)]
fn acc_attention(
    total: &mut Duration,
    pack: &mut Duration,
    compile: &mut Duration,
    upload: &mut Duration,
    gpu: &mut Duration,
    download: &mut Duration,
    max_abs_diff: &mut f32,
    metrics: &AttentionProjectionMixMetrics,
) {
    *total += metrics.total_duration;
    *pack += metrics.pack_duration;
    *compile += metrics.compile_duration;
    *upload += metrics.upload_duration;
    *gpu += metrics.gpu_duration;
    *download += metrics.download_duration;
    *max_abs_diff = (*max_abs_diff).max(metrics.max_abs_diff);
}

#[allow(clippy::too_many_arguments)]
fn acc_mlp(
    total: &mut Duration,
    pack: &mut Duration,
    compile: &mut Duration,
    upload: &mut Duration,
    gpu: &mut Duration,
    download: &mut Duration,
    max_abs_diff: &mut f32,
    metrics: &MlpProjectionMixMetrics,
) {
    *total += metrics.total_duration;
    *pack += metrics.pack_duration;
    *compile += metrics.compile_duration;
    *upload += metrics.upload_duration;
    *gpu += metrics.gpu_duration;
    *download += metrics.download_duration;
    *max_abs_diff = (*max_abs_diff).max(metrics.max_abs_diff);
}
