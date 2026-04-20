use jengine::cpu::primitives::swiglu;
use jengine::gpu::swiglu::run_swiglu_with_output;

fn main() {
    let len = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(6144);
    let gate = (0..len)
        .map(|i| (i % 23) as f32 * 0.041 - 0.5)
        .collect::<Vec<_>>();
    let up = (0..len)
        .map(|i| (i % 19) as f32 * 0.037 - 0.3)
        .collect::<Vec<_>>();
    let reference = swiglu(&gate, &up);
    let (_output, report) = run_swiglu_with_output(&gate, &up, Some(&reference))
        .expect("gpu swiglu should succeed");
    println!(
        "len={} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
        report.len,
        report.compile_duration.as_secs_f64() * 1_000.0,
        report.upload_duration.as_secs_f64() * 1_000.0,
        report.gpu_duration.as_secs_f64() * 1_000.0,
        report.download_duration.as_secs_f64() * 1_000.0,
        report.max_abs_diff,
        report.mean_abs_diff,
    );
}
