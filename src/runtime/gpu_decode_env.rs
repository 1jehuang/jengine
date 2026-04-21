use crate::runtime::decode_plan::PackedDecodePlan;

pub(crate) fn packed_enabled_label(use_attention_qkv: bool, use_mlp_gu: bool) -> String {
    PackedDecodePlan::from_env(use_attention_qkv, use_mlp_gu, false).enabled_label()
}

pub(crate) fn packed_use_mlp_full() -> bool {
    std::env::var_os("JENGINE_PACKED_MLP_FULL").is_some()
}

pub(crate) fn packed_use_attention_full() -> bool {
    std::env::var_os("JENGINE_PACKED_ATTENTION_FULL").is_some()
}

pub(crate) fn packed_use_gpu_final_norm() -> bool {
    std::env::var_os("JENGINE_GPU_FINAL_NORM").is_some()
}

pub(crate) fn packed_use_gpu_swiglu_block() -> bool {
    std::env::var_os("JENGINE_GPU_SWIGLU_BLOCK").is_some()
}

pub(crate) fn packed_use_gpu_attention_block() -> bool {
    std::env::var_os("JENGINE_GPU_ATTENTION_BLOCK").is_some()
}

pub(crate) fn packed_use_gpu_embedding() -> bool {
    std::env::var_os("JENGINE_GPU_EMBEDDING").is_some()
}

pub(crate) fn packed_use_gpu_mlp_entry() -> bool {
    std::env::var_os("JENGINE_GPU_MLP_ENTRY").is_some()
}

pub(crate) fn packed_use_gpu_full_last_layer() -> bool {
    std::env::var_os("JENGINE_GPU_FULL_LAST_LAYER").is_some()
}

pub(crate) fn packed_use_gpu_tail() -> bool {
    std::env::var_os("JENGINE_GPU_TAIL").is_some()
}

pub(crate) fn packed_use_gpu_first_session() -> bool {
    packed_use_gpu_attention_block()
        || packed_use_gpu_embedding()
        || packed_use_gpu_swiglu_block()
        || packed_use_gpu_full_last_layer()
        || packed_use_gpu_tail()
}

pub(crate) fn gpu_first_use_attention_full(use_attention_qkv: bool) -> bool {
    use_attention_qkv
        && (packed_use_attention_full()
            || packed_use_gpu_attention_block()
            || packed_use_gpu_full_last_layer())
}

pub(crate) fn gpu_first_use_mlp_full(use_mlp_gu: bool) -> bool {
    use_mlp_gu
        && (packed_use_mlp_full()
            || packed_use_gpu_swiglu_block()
            || packed_use_gpu_full_last_layer()
            || packed_use_gpu_mlp_entry())
}

#[cfg(test)]
use std::sync::{Mutex, MutexGuard, OnceLock};

#[cfg(test)]
fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
fn set_env(key: &str, value: &str) {
    unsafe { std::env::set_var(key, value) }
}

#[cfg(test)]
fn remove_env(key: &str) {
    unsafe { std::env::remove_var(key) }
}

#[cfg(test)]
pub(crate) fn lock_env() -> MutexGuard<'static, ()> {
    env_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
pub(crate) struct ScopedEnvVars(Vec<(&'static str, Option<String>)>);

#[cfg(test)]
impl ScopedEnvVars {
    pub(crate) fn set(pairs: &[(&'static str, &'static str)]) -> Self {
        let previous = pairs
            .iter()
            .map(|(key, value)| {
                let old = std::env::var(key).ok();
                set_env(key, value);
                (*key, old)
            })
            .collect();
        Self(previous)
    }
}

#[cfg(test)]
impl Drop for ScopedEnvVars {
    fn drop(&mut self) {
        for (key, value) in self.0.drain(..) {
            match value {
                Some(old) => set_env(key, &old),
                None => remove_env(key),
            }
        }
    }
}
