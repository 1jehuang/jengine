use jengine::cpu::primitives::profile_primitives;

fn main() {
    let rows = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2048);
    let cols = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2048);

    let profile = profile_primitives(rows, cols);
    println!("rows={rows} cols={cols} {}", profile.summarize());
}
