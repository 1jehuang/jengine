use jengine::runtime::weights::WeightStore;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b/model.safetensors".to_string());
    let progress =
        WeightStore::download_progress(&path).expect("weight progress probe should succeed");
    println!("{}", progress.summarize());
}
