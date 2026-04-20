use half::f16;
use jengine::cpu::primitives::swiglu;
use jengine::gpu::swiglu_pack_f16_pairs::run_swiglu_pack_f16_pairs_with_output;

fn pack_f16_pairs(values: &[f32]) -> Vec<u32> {
    values
        .chunks(2)
        .map(|chunk| {
            let a = f16::from_f32(chunk[0]).to_bits() as u32;
            let b = chunk
                .get(1)
                .map(|v| f16::from_f32(*v).to_bits() as u32)
                .unwrap_or(0);
            a | (b << 16)
        })
        .collect()
}

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
    let mut combined = gate.clone();
    combined.extend_from_slice(&up);
    let reference = pack_f16_pairs(&swiglu(&gate, &up));
    let (_output, report) = run_swiglu_pack_f16_pairs_with_output(&combined, Some(&reference))
        .expect("gpu fused swiglu pack should succeed");
    println!(
        "len={} compile_ms={:.3} upload_ms={:.3} gpu_ms={:.3} download_ms={:.3} mismatched_words={}",
        report.len,
        report.compile_duration.as_secs_f64() * 1_000.0,
        report.upload_duration.as_secs_f64() * 1_000.0,
        report.gpu_duration.as_secs_f64() * 1_000.0,
        report.download_duration.as_secs_f64() * 1_000.0,
        report.max_mismatched_words,
    );
}
