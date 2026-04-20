use jengine::gpu::weighted_rms_norm::run_weighted_rms_norm_with_output;
use jengine::runtime::weights::WeightStore;

fn main() {
    let weights_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let tensor_name = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "model.norm.weight".to_string());

    let store = WeightStore::load_from_file(&weights_path).expect("weights should load");
    let weight = store
        .load_vector_f32(&tensor_name)
        .expect("weight tensor should load");
    let len = weight.len();
    let input = (0..len)
        .map(|i| (i % 17) as f32 * 0.03125 - 0.25)
        .collect::<Vec<_>>();

    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / len as f32;
    let scale = 1.0 / (mean_square + 1e-6).sqrt();
    let reference = input
        .iter()
        .zip(weight.iter())
        .map(|(value, gamma)| value * scale * gamma)
        .collect::<Vec<_>>();

    let (_output, report) =
        run_weighted_rms_norm_with_output(&input, &weight, 1e-6, Some(&reference))
            .expect("gpu weighted rms norm should succeed");

    println!("{}", report.summarize());
}
