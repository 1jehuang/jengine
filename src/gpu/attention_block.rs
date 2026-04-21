use std::sync::Arc;
use std::time::Duration;

use crate::gpu::attention_single_query::{
    CachedGpuAttentionSingleQueryRunner, GpuAttentionSingleQueryError,
};
use crate::gpu::full_last_layer_block::PackedLinearSpec;
use crate::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, GpuPackedMatvecError, PackedRunnerInputMode,
    SharedGpuPackedContext,
};
use crate::gpu::vector_add::{CachedGpuVectorAddRunner, GpuVectorAddError};

#[derive(Debug, Clone, PartialEq)]
pub struct GpuAttentionBlockReport {
    pub hidden: usize,
    pub compile_duration: Duration,
    pub attention_gpu_duration: Duration,
    pub oproj_gpu_duration: Duration,
    pub residual_add_gpu_duration: Duration,
}

#[derive(Debug)]
pub struct GpuAttentionBlockError(String);

impl std::fmt::Display for GpuAttentionBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuAttentionBlockError {}

pub struct CachedGpuAttentionBlockRunner {
    hidden: usize,
    compile_duration: Duration,
    attention_runner: CachedGpuAttentionSingleQueryRunner,
    o_proj_runner: CachedGpuPackedMatvecRunner,
    residual_seed_runner: CachedGpuVectorAddRunner,
    add_runner: CachedGpuVectorAddRunner,
}

impl CachedGpuAttentionBlockRunner {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        seq_len: usize,
        num_query_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        o_proj_spec: &PackedLinearSpec,
    ) -> Result<Self, GpuAttentionBlockError> {
        let hidden = num_query_heads * head_dim;
        let (attention_runner, attention_compile) =
            CachedGpuAttentionSingleQueryRunner::new_with_context(
                context.clone(),
                seq_len,
                num_query_heads,
                num_key_value_heads,
                head_dim,
            )
            .map_err(map_attention_error)?;
        let (o_proj_runner, oproj_compile) =
            CachedGpuPackedMatvecRunner::new_with_context_and_input_mode(
                context.clone(),
                &o_proj_spec.code_words,
                &o_proj_spec.scales,
                o_proj_spec.group_size,
                o_proj_spec.rows,
                o_proj_spec.cols,
                PackedRunnerInputMode::RawF32,
            )
            .map_err(map_packed_matvec_error)?;
        let (residual_seed_runner, residual_compile) =
            CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)
                .map_err(map_vector_add_error)?;
        let (add_runner, add_compile) = CachedGpuVectorAddRunner::new_with_context(context, hidden)
            .map_err(map_vector_add_error)?;
        Ok(Self {
            hidden,
            compile_duration: attention_compile + oproj_compile + residual_compile + add_compile,
            attention_runner,
            o_proj_runner,
            residual_seed_runner,
            add_runner,
        })
    }

    pub fn run_resident(
        &mut self,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        residual: &[f32],
    ) -> Result<GpuAttentionBlockReport, GpuAttentionBlockError> {
        if residual.len() != self.hidden {
            return Err(GpuAttentionBlockError(format!(
                "residual len {} must match hidden {}",
                residual.len(),
                self.hidden
            )));
        }
        let zeros = vec![0.0f32; self.hidden];
        let attention_report = self
            .attention_runner
            .run_with_output(query, keys, values, None)
            .map_err(map_attention_error)?
            .1;
        let oproj_report = self
            .o_proj_runner
            .run_resident_from_f32_buffer(
                self.attention_runner.shared_context(),
                self.attention_runner.output_buffer_handle(),
                self.hidden,
                self.attention_runner.output_buffer_size(),
            )
            .map_err(map_packed_matvec_error)?;
        let _ = self
            .residual_seed_runner
            .run_with_output(residual, &zeros, None)
            .map_err(map_vector_add_error)?;
        let add_report = self
            .add_runner
            .run_resident_from_buffers(
                self.o_proj_runner.shared_context(),
                self.o_proj_runner.output_buffer_handle(),
                self.hidden,
                self.o_proj_runner.output_buffer_size(),
                self.residual_seed_runner.output_buffer_handle(),
                self.hidden,
                self.residual_seed_runner.output_buffer_size(),
            )
            .map_err(map_vector_add_error)?;
        Ok(GpuAttentionBlockReport {
            hidden: self.hidden,
            compile_duration: self.compile_duration,
            attention_gpu_duration: attention_report.gpu_duration,
            oproj_gpu_duration: oproj_report.gpu_duration,
            residual_add_gpu_duration: add_report.gpu_duration,
        })
    }

    pub fn shared_context(&self) -> &Arc<SharedGpuPackedContext> {
        self.add_runner.shared_context()
    }

    pub fn output_buffer_handle(&self) -> ash::vk::Buffer {
        self.add_runner.output_buffer_handle()
    }

    pub fn output_buffer_size(&self) -> u64 {
        self.add_runner.output_buffer_size()
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }
}

fn map_attention_error(error: GpuAttentionSingleQueryError) -> GpuAttentionBlockError {
    GpuAttentionBlockError(error.to_string())
}

fn map_packed_matvec_error(error: GpuPackedMatvecError) -> GpuAttentionBlockError {
    GpuAttentionBlockError(error.to_string())
}

fn map_vector_add_error(error: GpuVectorAddError) -> GpuAttentionBlockError {
    GpuAttentionBlockError(error.to_string())
}
