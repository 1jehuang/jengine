#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReport {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_sequence_tokens: usize,
    pub estimated_model_fp16_bytes: usize,
    pub source_weight_bytes: usize,
    pub kv_cache_bytes_per_token_fp16: usize,
    pub kv_cache_bytes_per_token_runtime_f32: usize,
    pub kv_cache_total_bytes_fp16: usize,
    pub kv_cache_total_bytes_runtime_f32: usize,
    pub kv_cache_reserved_bytes_runtime_f32: usize,
    pub packed_cache_bytes: usize,
    pub gpu_cache_buffer_bytes: usize,
    pub activation_working_bytes: usize,
    pub staging_bytes: usize,
    pub estimated_runtime_working_set_bytes: usize,
}

impl MemoryReport {
    pub fn summarize(&self) -> String {
        format!(
            "prompt_tokens={} generated_tokens={} total_sequence_tokens={} model_fp16_bytes={} ({}) source_weight_bytes={} ({}) kv_per_token_fp16={} ({}) kv_per_token_runtime_f32={} ({}) kv_total_runtime_f32={} ({}) kv_reserved_runtime_f32={} ({}) packed_cache_bytes={} ({}) gpu_cache_buffer_bytes={} ({}) activation_working_bytes={} ({}) staging_bytes={} ({}) working_set_bytes={} ({})",
            self.prompt_tokens,
            self.generated_tokens,
            self.total_sequence_tokens,
            self.estimated_model_fp16_bytes,
            human_bytes(self.estimated_model_fp16_bytes),
            self.source_weight_bytes,
            human_bytes(self.source_weight_bytes),
            self.kv_cache_bytes_per_token_fp16,
            human_bytes(self.kv_cache_bytes_per_token_fp16),
            self.kv_cache_bytes_per_token_runtime_f32,
            human_bytes(self.kv_cache_bytes_per_token_runtime_f32),
            self.kv_cache_total_bytes_runtime_f32,
            human_bytes(self.kv_cache_total_bytes_runtime_f32),
            self.kv_cache_reserved_bytes_runtime_f32,
            human_bytes(self.kv_cache_reserved_bytes_runtime_f32),
            self.packed_cache_bytes,
            human_bytes(self.packed_cache_bytes),
            self.gpu_cache_buffer_bytes,
            human_bytes(self.gpu_cache_buffer_bytes),
            self.activation_working_bytes,
            human_bytes(self.activation_working_bytes),
            self.staging_bytes,
            human_bytes(self.staging_bytes),
            self.estimated_runtime_working_set_bytes,
            human_bytes(self.estimated_runtime_working_set_bytes),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_memory_report(
    prompt_tokens: usize,
    generated_tokens: usize,
    estimated_model_fp16_bytes: usize,
    source_weight_bytes: usize,
    kv_cache_bytes_per_token_fp16: usize,
    kv_cache_bytes_per_token_runtime_f32: usize,
    kv_cache_total_bytes_fp16: usize,
    kv_cache_total_bytes_runtime_f32: usize,
    packed_cache_bytes: usize,
    gpu_cache_buffer_bytes: usize,
    activation_working_bytes: usize,
    staging_bytes: usize,
) -> MemoryReport {
    let total_sequence_tokens = prompt_tokens + generated_tokens;
    MemoryReport {
        prompt_tokens,
        generated_tokens,
        total_sequence_tokens,
        estimated_model_fp16_bytes,
        source_weight_bytes,
        kv_cache_bytes_per_token_fp16,
        kv_cache_bytes_per_token_runtime_f32,
        kv_cache_total_bytes_fp16,
        kv_cache_total_bytes_runtime_f32,
        kv_cache_reserved_bytes_runtime_f32: kv_cache_total_bytes_runtime_f32,
        packed_cache_bytes,
        gpu_cache_buffer_bytes,
        activation_working_bytes,
        staging_bytes,
        estimated_runtime_working_set_bytes: estimated_model_fp16_bytes
            + kv_cache_total_bytes_runtime_f32
            + packed_cache_bytes
            + gpu_cache_buffer_bytes
            + activation_working_bytes,
    }
}

fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}
