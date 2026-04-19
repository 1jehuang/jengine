use half::f16;
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let started_at = Instant::now();
    let file = File::open(&path).expect("model file should be readable");
    let mmap = unsafe { Mmap::map(&file).expect("model file should be memory-mapped") };
    let header_len = u64::from_le_bytes(mmap[..8].try_into().expect("header bytes")) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&mmap[8..8 + header_len]).expect("header json should parse");

    let mut available_tensors = 0usize;
    let mut eligible_tensors = 0usize;
    let mut perfect_tensors = 0usize;
    let mut total_groups = 0usize;
    let mut perfect_groups = 0usize;
    let mut worst_group_error = 0.0f32;
    let mut worst_tensor = String::new();
    let mut worst_group = 0usize;
    let mut by_prefix = BTreeMap::<String, (usize, usize)>::new();

    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info["dtype"].as_str().expect("dtype should exist");
        let offsets = info["data_offsets"]
            .as_array()
            .expect("data_offsets should exist");
        let begin = offsets[0].as_u64().expect("begin offset") as usize;
        let end = offsets[1].as_u64().expect("end offset") as usize;
        let abs_begin = 8 + header_len + begin;
        let abs_end = 8 + header_len + end;
        if abs_end > mmap.len() {
            continue;
        }
        available_tensors += 1;
        if dtype != "F16" {
            continue;
        }
        if !(name.contains("embed_tokens.weight")
            || name.contains("mlp.")
            || name.contains("self_attn.")
            || name == "model.norm.weight")
        {
            continue;
        }
        eligible_tensors += 1;
        let data = &mmap[abs_begin..abs_end];

        let mut tensor_ok = true;
        for (group_idx, chunk) in data.chunks(128 * 2).enumerate() {
            total_groups += 1;
            let mut scale = None::<f32>;
            let mut error = 0.0f32;
            for bytes in chunk.chunks_exact(2) {
                let value = f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
                let abs = value.abs();
                if abs <= 1e-8 {
                    continue;
                }
                match scale {
                    Some(existing) => {
                        error = error.max((abs - existing).abs());
                    }
                    None => scale = Some(abs),
                }
            }
            if error <= 1e-3 {
                perfect_groups += 1;
            } else {
                tensor_ok = false;
                if error > worst_group_error {
                    worst_group_error = error;
                    worst_tensor = name.clone();
                    worst_group = group_idx;
                }
            }
        }
        if tensor_ok {
            perfect_tensors += 1;
        }

        let key = if name.starts_with("model.layers.") {
            let parts = name.split('.').collect::<Vec<_>>();
            format!("layer.{}", parts[2])
        } else {
            "global".to_string()
        };
        let entry = by_prefix.entry(key).or_insert((0, 0));
        entry.0 += 1;
        if tensor_ok {
            entry.1 += 1;
        }
    }

    println!("file={path}");
    println!("available_tensors={available_tensors}");
    println!("eligible_tensors={eligible_tensors}");
    println!("perfect_tensors={perfect_tensors}");
    println!("total_groups={total_groups}");
    println!("perfect_groups={perfect_groups}");
    println!("worst_group_error={worst_group_error}");
    println!(
        "worst_tensor={}",
        if worst_tensor.is_empty() {
            "<none>"
        } else {
            &worst_tensor
        }
    );
    println!("worst_group={worst_group}");
    println!(
        "elapsed_ms={:.3}",
        started_at.elapsed().as_secs_f64() * 1_000.0
    );
    for (prefix, (seen, ok)) in by_prefix {
        println!("bucket={prefix} perfect_tensors={ok}/{seen}");
    }
}
