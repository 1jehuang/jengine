#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PackedDecodeScratch {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub mlp: Vec<f32>,
}
