use jengine::gpu::packed_matvec::SharedGpuPackedContext;
use jengine::gpu::vector_add::CachedGpuVectorAddRunner;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let len = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(2048);
    let left: Vec<f32> = (0..len).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let right: Vec<f32> = (0..len)
        .map(|i| ((len - i) as f32) * 0.0007 - 0.25)
        .collect();
    let zeros = vec![0.0f32; len];
    let reference: Vec<f32> = left.iter().zip(right.iter()).map(|(a, b)| a + b).collect();

    let context = SharedGpuPackedContext::new()?;
    let (mut left_runner, _) = CachedGpuVectorAddRunner::new_with_context(context.clone(), len)?;
    let (mut right_runner, _) = CachedGpuVectorAddRunner::new_with_context(context.clone(), len)?;
    let (mut sum_runner, compile_duration) =
        CachedGpuVectorAddRunner::new_with_context(context, len)?;

    let _ = left_runner.run_with_output(&left, &zeros, None)?;
    let _ = right_runner.run_with_output(&right, &zeros, None)?;
    let mut report = sum_runner.run_resident_from_buffers(
        left_runner.shared_context(),
        left_runner.output_buffer_handle(),
        len,
        left_runner.output_buffer_size(),
        right_runner.output_buffer_handle(),
        len,
        right_runner.output_buffer_size(),
    )?;
    let (output, download_duration) = sum_runner.read_output()?;
    report.compile_duration = compile_duration;
    report.download_duration = download_duration;

    let (max_abs_diff, mean_abs_diff) = reference
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold((0.0f32, 0.0f32), |(max_diff, sum), diff| {
            (max_diff.max(diff), sum + diff)
        });
    let mean_abs_diff = mean_abs_diff / len.max(1) as f32;

    println!(
        "len={} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} max_abs_diff={:.6} mean_abs_diff={:.6}",
        len,
        report.compile_duration.as_secs_f64() * 1_000.0,
        report.upload_duration.as_secs_f64() * 1_000.0,
        report.gpu_duration.as_secs_f64() * 1_000.0,
        report.download_duration.as_secs_f64() * 1_000.0,
        max_abs_diff,
        mean_abs_diff,
    );

    Ok(())
}
