use criterion::{Criterion, black_box, criterion_group, criterion_main};
use jengine::runtime::repack::{matvec_packed_ternary, pack_ternary_g128};

fn synthetic_tensor() -> (Vec<f32>, Vec<f32>) {
    let values = (0..(128 * 256))
        .map(|index| match index % 3 {
            0 => 0.0,
            1 => 1.25,
            _ => -1.25,
        })
        .collect::<Vec<_>>();
    let input = (0..256)
        .map(|index| (index % 7) as f32 * 0.1 - 0.3)
        .collect::<Vec<_>>();
    (values, input)
}

fn bench_packed_matvec(c: &mut Criterion) {
    let (values, input) = synthetic_tensor();
    let (packed, _) = pack_ternary_g128(&values, vec![128, 256], 1e-6).unwrap();
    c.bench_function("packed_matvec/128x256", |b| {
        b.iter(|| matvec_packed_ternary(black_box(&packed), black_box(&input)).unwrap())
    });
}

criterion_group!(packed_matvec_benches, bench_packed_matvec);
criterion_main!(packed_matvec_benches);
