#[derive(Debug, Clone, PartialEq)]
pub enum PackedDecodeStepResult {
    Logits(Vec<f32>),
    NextToken(usize),
}

#[derive(Debug, Clone)]
pub(crate) struct LayerCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

impl LayerCache {
    pub fn with_capacity(tokens: usize, kv_width: usize) -> Self {
        let capacity = tokens * kv_width;
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    pub fn without_preallocated_cpu_kv() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }
}
