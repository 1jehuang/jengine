use jengine::runtime::reference::ReferenceModel;

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/jeremy/models/bonsai-1.7b".to_string());
    let layer_idx = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let token_id = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(42);

    let model =
        ReferenceModel::load_core_from_root(&root).expect("reference core model should load");
    let comparison = model
        .compare_qproj_dense_vs_packed(layer_idx, token_id)
        .expect("projection comparison should succeed");
    println!("{}", comparison.summarize());
}
