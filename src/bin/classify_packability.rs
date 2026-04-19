use half::f16;
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;

#[derive(Default, Debug)]
struct BucketStats {
    tensors: usize,
    perfect_tensors: usize,
    groups: usize,
    perfect_groups: usize,
    max_group_error: f32,
    sample_failure: Option<String>,
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let file = File::open(&path).expect("model file should open");
    let mmap = unsafe { Mmap::map(&file).expect("model file should mmap") };
    let header_len = u64::from_le_bytes(mmap[..8].try_into().expect("header bytes")) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&mmap[8..8 + header_len]).expect("header should parse");

    let mut buckets = BTreeMap::<String, BucketStats>::new();

    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        if info["dtype"].as_str() != Some("F16") {
            continue;
        }
        let offsets = info["data_offsets"]
            .as_array()
            .expect("data offsets should exist");
        let begin = offsets[0].as_u64().expect("begin offset") as usize;
        let end = offsets[1].as_u64().expect("end offset") as usize;
        let data = &mmap[(8 + header_len + begin)..(8 + header_len + end)];

        let bucket_name = classify_name(&name);
        let stats = buckets.entry(bucket_name).or_default();
        stats.tensors += 1;

        let mut tensor_ok = true;
        for chunk in data.chunks(128 * 2) {
            stats.groups += 1;
            let (scale, error) = analyze_group_bytes(chunk);
            let _ = scale;
            if error <= 1e-3 {
                stats.perfect_groups += 1;
            } else {
                tensor_ok = false;
                if error > stats.max_group_error {
                    stats.max_group_error = error;
                }
                if stats.sample_failure.is_none() {
                    stats.sample_failure = Some(name.clone());
                }
            }
        }
        if tensor_ok {
            stats.perfect_tensors += 1;
        }
    }

    for (bucket, stats) in buckets {
        println!(
            "bucket={} perfect_tensors={}/{} perfect_groups={}/{} max_group_error={:.6} sample_failure={}",
            bucket,
            stats.perfect_tensors,
            stats.tensors,
            stats.perfect_groups,
            stats.groups,
            stats.max_group_error,
            stats.sample_failure.unwrap_or_else(|| "<none>".to_string())
        );
    }
}

fn classify_name(name: &str) -> String {
    if name == "model.embed_tokens.weight" {
        return "embed_tokens".to_string();
    }
    if name == "model.norm.weight" {
        return "final_norm".to_string();
    }
    if name.contains("input_layernorm.weight") {
        return "input_layernorm".to_string();
    }
    if name.contains("post_attention_layernorm.weight") {
        return "post_attention_layernorm".to_string();
    }
    if name.contains("self_attn.q_norm.weight") {
        return "q_norm".to_string();
    }
    if name.contains("self_attn.k_norm.weight") {
        return "k_norm".to_string();
    }
    if name.contains("self_attn.q_proj.weight") {
        return "q_proj".to_string();
    }
    if name.contains("self_attn.k_proj.weight") {
        return "k_proj".to_string();
    }
    if name.contains("self_attn.v_proj.weight") {
        return "v_proj".to_string();
    }
    if name.contains("self_attn.o_proj.weight") {
        return "o_proj".to_string();
    }
    if name.contains("mlp.gate_proj.weight") {
        return "gate_proj".to_string();
    }
    if name.contains("mlp.up_proj.weight") {
        return "up_proj".to_string();
    }
    if name.contains("mlp.down_proj.weight") {
        return "down_proj".to_string();
    }
    "other".to_string()
}

fn analyze_group_bytes(chunk: &[u8]) -> (f32, f32) {
    let mut scale = 0.0f32;
    for bytes in chunk.chunks_exact(2) {
        let value = f16::from_le_bytes([bytes[0], bytes[1]]).to_f32().abs();
        if value > scale {
            scale = value;
        }
    }
    if scale <= 1e-8 {
        return (0.0, 0.0);
    }
    let mut max_error = 0.0f32;
    for bytes in chunk.chunks_exact(2) {
        let value = f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
        let recon = if value.abs() <= scale * 0.5 {
            0.0
        } else {
            value.signum() * scale
        };
        max_error = max_error.max((value - recon).abs());
    }
    (scale, max_error)
}
