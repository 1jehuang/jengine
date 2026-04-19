use criterion::{Criterion, black_box, criterion_group, criterion_main};
use jengine::runtime::packed::PackedTensorFile;
use jengine::runtime::repack::pack_ternary_g128;
use tempfile::NamedTempFile;

fn synthetic_file() -> PackedTensorFile {
    let values = (0..(128 * 128))
        .map(|index| match index % 3 {
            0 => 0.0,
            1 => 1.0,
            _ => -1.0,
        })
        .collect::<Vec<_>>();
    let (packed, _) = pack_ternary_g128(&values, vec![128, 128], 1e-6).unwrap();
    PackedTensorFile::new(Some("synthetic.weight".to_string()), packed)
}

fn bench_packed_io(c: &mut Criterion) {
    let packed = synthetic_file();
    let temp = NamedTempFile::new().unwrap();

    c.bench_function("packed_io/write_16k", |b| {
        b.iter(|| packed.write_to_path(black_box(temp.path())).unwrap())
    });
    packed.write_to_path(temp.path()).unwrap();
    c.bench_function("packed_io/read_16k", |b| {
        b.iter(|| PackedTensorFile::read_from_path(black_box(temp.path())).unwrap())
    });
}

criterion_group!(packed_io_benches, bench_packed_io);
criterion_main!(packed_io_benches);
