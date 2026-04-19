use criterion::{Criterion, black_box, criterion_group, criterion_main};
use jengine::cpu::primitives::{argmax, matvec, profile_primitives, rms_norm, softmax, swiglu};

fn bench_matvec(c: &mut Criterion) {
    let rows = 512;
    let cols = 1024;
    let matrix: Vec<f32> = (0..rows * cols)
        .map(|index| (index % 19) as f32 * 0.05)
        .collect();
    let vector: Vec<f32> = (0..cols).map(|index| (index % 11) as f32 * 0.03).collect();

    c.bench_function("cpu/matvec_512x1024", |b| {
        b.iter(|| matvec(black_box(&matrix), rows, cols, black_box(&vector)))
    });
}

fn bench_rms_norm(c: &mut Criterion) {
    let input: Vec<f32> = (0..4096).map(|index| (index % 17) as f32 * 0.07).collect();
    let weight: Vec<f32> = vec![1.0; input.len()];

    c.bench_function("cpu/rms_norm_4096", |b| {
        b.iter(|| rms_norm(black_box(&input), black_box(&weight), black_box(1e-6)))
    });
}

fn bench_softmax_and_argmax(c: &mut Criterion) {
    let logits: Vec<f32> = (0..4096).map(|index| index as f32 * 0.0005 - 1.0).collect();

    c.bench_function("cpu/softmax_4096", |b| {
        b.iter(|| softmax(black_box(&logits)))
    });

    let probs = softmax(&logits);
    c.bench_function("cpu/argmax_4096", |b| b.iter(|| argmax(black_box(&probs))));
}

fn bench_swiglu(c: &mut Criterion) {
    let gate: Vec<f32> = (0..4096)
        .map(|index| (index % 23) as f32 * 0.04 - 0.5)
        .collect();
    let up: Vec<f32> = (0..4096)
        .map(|index| (index % 29) as f32 * 0.03 - 0.4)
        .collect();

    c.bench_function("cpu/swiglu_4096", |b| {
        b.iter(|| swiglu(black_box(&gate), black_box(&up)))
    });
}

fn bench_profile_bundle(c: &mut Criterion) {
    c.bench_function("cpu/profile_bundle_512x1024", |b| {
        b.iter(|| profile_primitives(black_box(512), black_box(1024)))
    });
}

criterion_group!(
    cpu_primitive_benches,
    bench_matvec,
    bench_rms_norm,
    bench_softmax_and_argmax,
    bench_swiglu,
    bench_profile_bundle
);
criterion_main!(cpu_primitive_benches);
