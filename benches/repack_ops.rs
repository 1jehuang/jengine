use criterion::{Criterion, black_box, criterion_group, criterion_main};
use jengine::runtime::repack::{
    analyze_ternary_packability, pack_ternary_g128, unpack_ternary_g128,
};

fn synthetic_values() -> Vec<f32> {
    (0..(128 * 256))
        .map(|index| match index % 3 {
            0 => 0.0,
            1 => 1.5,
            _ => -1.5,
        })
        .collect()
}

fn bench_repack(c: &mut Criterion) {
    let values = synthetic_values();
    c.bench_function("repack/analyze_32k", |b| {
        b.iter(|| analyze_ternary_packability(black_box(&values)))
    });
    c.bench_function("repack/pack_32k", |b| {
        b.iter(|| pack_ternary_g128(black_box(&values), vec![128, 256], 1e-6).unwrap())
    });
    let (packed, _) = pack_ternary_g128(&values, vec![128, 256], 1e-6).unwrap();
    c.bench_function("repack/unpack_32k", |b| {
        b.iter(|| unpack_ternary_g128(black_box(&packed)).unwrap())
    });
}

criterion_group!(repack_benches, bench_repack);
criterion_main!(repack_benches);
