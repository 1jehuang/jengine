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

    pub fn append(&mut self, keys: &[f32], values: &[f32]) {
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn cpu_kv_is_empty(&self) -> bool {
        self.keys.is_empty() && self.values.is_empty()
    }

    pub fn keys_capacity(&self) -> usize {
        self.keys.capacity()
    }

    pub fn values_capacity(&self) -> usize {
        self.values.capacity()
    }
}
