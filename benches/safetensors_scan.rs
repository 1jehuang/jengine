use bytemuck::cast_slice;
use criterion::{Criterion, criterion_group, criterion_main};
use jengine::model::safetensors_scan::scan_safetensors_file;
use safetensors::tensor::{Dtype, TensorView, serialize};
use std::collections::BTreeMap;
use std::fs;
use tempfile::NamedTempFile;

fn make_fixture() -> NamedTempFile {
    let temp = NamedTempFile::new().expect("temp file should be created");

    let weights = vec![1f32; 4096];
    let scales = vec![7u16; 2048];

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "block0.weight".to_string(),
        TensorView::new(Dtype::F32, vec![64, 64], cast_slice(&weights))
            .expect("weight view should be built"),
    );
    tensors.insert(
        "block0.scale".to_string(),
        TensorView::new(Dtype::U16, vec![64, 32], cast_slice(&scales))
            .expect("scale view should be built"),
    );

    let bytes = serialize(tensors, &None).expect("serialization should succeed");
    fs::write(temp.path(), bytes).expect("fixture should be written");
    temp
}

fn bench_safetensors_scan(c: &mut Criterion) {
    let fixture = make_fixture();
    c.bench_function("safetensors/scan_inventory", |b| {
        b.iter(|| scan_safetensors_file(fixture.path()).expect("scan should succeed"))
    });
}

criterion_group!(scan_benches, bench_safetensors_scan);
criterion_main!(scan_benches);
