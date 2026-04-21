use jengine::gpu::vector_add::run_vector_add_with_output;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let len: usize = std::env::args()
        .nth(1)
        .as_deref()
        .unwrap_or("2048")
        .parse()?;

    let left: Vec<f32> = (0..len).map(|idx| ((idx as f32) * 0.001).sin()).collect();
    let right: Vec<f32> = (0..len).map(|idx| ((idx as f32) * 0.002).cos()).collect();
    let reference: Vec<f32> = left.iter().zip(right.iter()).map(|(l, r)| l + r).collect();

    let (_output, report) = run_vector_add_with_output(&left, &right, Some(&reference))?;
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
    Ok(())
}
