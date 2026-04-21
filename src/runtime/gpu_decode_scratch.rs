#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PackedDecodeScratch {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub mlp: Vec<f32>,
}

impl PackedDecodeScratch {
    pub fn take_qkv(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let Self { q, k, v, .. } = self;
        (std::mem::take(q), std::mem::take(k), std::mem::take(v))
    }

    pub fn restore_qkv(&mut self, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>) {
        self.q = q;
        self.k = k;
        self.v = v;
    }

    pub fn take_gate_up(&mut self) -> (Vec<f32>, Vec<f32>) {
        let Self { gate, up, .. } = self;
        (std::mem::take(gate), std::mem::take(up))
    }

    pub fn restore_gate_up(&mut self, gate: Vec<f32>, up: Vec<f32>) {
        self.gate = gate;
        self.up = up;
    }

    pub fn take_mlp(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.mlp)
    }

    pub fn restore_mlp(&mut self, mlp: Vec<f32>) {
        self.mlp = mlp;
    }
}
