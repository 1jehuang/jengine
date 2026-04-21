use std::sync::Arc;
use std::time::Duration;

use crate::gpu::full_last_layer_block::PackedLinearSpec;
use crate::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, GpuPackedMatvecError, PackedRunnerInputMode,
    SharedGpuPackedContext,
};
use crate::gpu::resident_buffer::GpuResidentBuffer;
use crate::gpu::swiglu_pack_f16_pairs::{
    CachedGpuSwigluPackF16PairsRunner, GpuSwigluPackF16PairsError,
};
use crate::gpu::vector_add::{CachedGpuVectorAddRunner, GpuVectorAddError};
use crate::gpu::weighted_rms_norm::{CachedGpuWeightedRmsNormRunner, GpuWeightedRmsNormError};

#[derive(Debug, Clone, PartialEq)]
pub struct GpuMlpBlockReport {
    pub hidden: usize,
    pub intermediate: usize,
    pub compile_duration: Duration,
    pub post_norm_gpu_duration: Duration,
    pub pair_gpu_duration: Duration,
    pub swiglu_pack_gpu_duration: Duration,
    pub down_gpu_duration: Duration,
    pub residual_add_gpu_duration: Duration,
}

#[derive(Debug)]
pub struct GpuMlpBlockError(String);

impl std::fmt::Display for GpuMlpBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuMlpBlockError {}

pub struct CachedGpuMlpBlockRunner {
    hidden: usize,
    intermediate: usize,
    compile_duration: Duration,
    post_norm_runner: CachedGpuWeightedRmsNormRunner,
    pair_runner: CachedGpuPackedMatvecRunner,
    swiglu_pack_runner: CachedGpuSwigluPackF16PairsRunner,
    down_runner: CachedGpuPackedMatvecRunner,
    residual_seed_runner: CachedGpuVectorAddRunner,
    add_runner: CachedGpuVectorAddRunner,
}

impl CachedGpuMlpBlockRunner {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        hidden: usize,
        intermediate: usize,
        epsilon: f32,
        pair_spec: &PackedLinearSpec,
        down_spec: &PackedLinearSpec,
    ) -> Result<Self, GpuMlpBlockError> {
        let (post_norm_runner, post_norm_compile) =
            CachedGpuWeightedRmsNormRunner::new_with_context(context.clone(), hidden, epsilon)
                .map_err(map_weighted_rms_norm_error)?;
        let (pair_runner, pair_compile) =
            CachedGpuPackedMatvecRunner::new_with_context_and_input_mode(
                context.clone(),
                &pair_spec.code_words,
                &pair_spec.scales,
                pair_spec.group_size,
                pair_spec.rows,
                pair_spec.cols,
                PackedRunnerInputMode::RawF32,
            )
            .map_err(map_packed_matvec_error)?;
        let (swiglu_pack_runner, swiglu_pack_compile) =
            CachedGpuSwigluPackF16PairsRunner::new_with_context(context.clone(), intermediate)
                .map_err(map_swiglu_pack_error)?;
        let (down_runner, down_compile) = CachedGpuPackedMatvecRunner::new_with_context(
            context.clone(),
            &down_spec.code_words,
            &down_spec.scales,
            down_spec.group_size,
            down_spec.rows,
            down_spec.cols,
        )
        .map_err(map_packed_matvec_error)?;
        let (residual_seed_runner, residual_compile) =
            CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)
                .map_err(map_vector_add_error)?;
        let (add_runner, add_compile) = CachedGpuVectorAddRunner::new_with_context(context, hidden)
            .map_err(map_vector_add_error)?;
        Ok(Self {
            hidden,
            intermediate,
            compile_duration: post_norm_compile
                + pair_compile
                + swiglu_pack_compile
                + down_compile
                + residual_compile
                + add_compile,
            post_norm_runner,
            pair_runner,
            swiglu_pack_runner,
            down_runner,
            residual_seed_runner,
            add_runner,
        })
    }

    pub fn run_with_host_residual(
        &mut self,
        post_attention_hidden: &[f32],
        post_norm_weight: &[f32],
    ) -> Result<GpuMlpBlockReport, GpuMlpBlockError> {
        if post_attention_hidden.len() != self.hidden || post_norm_weight.len() != self.hidden {
            return Err(GpuMlpBlockError(
                "mlp block host inputs must match hidden size".to_string(),
            ));
        }
        let zeros = vec![0.0f32; self.hidden];
        let post_norm_report = self
            .post_norm_runner
            .run_resident(post_attention_hidden, post_norm_weight)
            .map_err(map_weighted_rms_norm_error)?;
        let pair_report = self
            .pair_runner
            .run_resident_from_f32_buffer(
                self.post_norm_runner.shared_context(),
                self.post_norm_runner.output_buffer_handle(),
                self.hidden,
                self.post_norm_runner.output_buffer_size(),
            )
            .map_err(map_packed_matvec_error)?;
        let swiglu_pack_report = self
            .swiglu_pack_runner
            .run_with_output_from_buffer(
                self.pair_runner.shared_context(),
                self.pair_runner.output_buffer_handle(),
                self.intermediate * 2,
                self.pair_runner.output_buffer_size(),
            )
            .map_err(map_swiglu_pack_error)?;
        let down_report = self
            .down_runner
            .run_resident_from_packed_buffer(
                self.swiglu_pack_runner.shared_context(),
                self.swiglu_pack_runner.output_buffer_handle(),
                self.swiglu_pack_runner.packed_len(),
                self.swiglu_pack_runner.output_buffer_size(),
            )
            .map_err(map_packed_matvec_error)?;
        let _ = self
            .residual_seed_runner
            .run_resident(post_attention_hidden, &zeros)
            .map_err(map_vector_add_error)?;
        let add_report = self
            .add_runner
            .run_resident_from_buffers(
                self.down_runner.shared_context(),
                self.down_runner.output_buffer_handle(),
                self.hidden,
                self.down_runner.output_buffer_size(),
                self.residual_seed_runner.output_buffer_handle(),
                self.hidden,
                self.residual_seed_runner.output_buffer_size(),
            )
            .map_err(map_vector_add_error)?;
        Ok(GpuMlpBlockReport {
            hidden: self.hidden,
            intermediate: self.intermediate,
            compile_duration: self.compile_duration,
            post_norm_gpu_duration: post_norm_report.gpu_duration,
            pair_gpu_duration: pair_report.gpu_duration,
            swiglu_pack_gpu_duration: swiglu_pack_report.gpu_duration,
            down_gpu_duration: down_report.gpu_duration,
            residual_add_gpu_duration: add_report.gpu_duration,
        })
    }

    pub fn run_from_resident_residual(
        &mut self,
        source_context: &Arc<SharedGpuPackedContext>,
        source_buffer: ash::vk::Buffer,
        source_len: usize,
        source_buffer_size: u64,
        post_norm_weight: &[f32],
    ) -> Result<GpuMlpBlockReport, GpuMlpBlockError> {
        if source_len != self.hidden || post_norm_weight.len() != self.hidden {
            return Err(GpuMlpBlockError(
                "mlp block resident inputs must match hidden size".to_string(),
            ));
        }
        let post_norm_report = self
            .post_norm_runner
            .run_resident_from_f32_buffer(
                source_context,
                source_buffer,
                source_len,
                source_buffer_size,
                post_norm_weight,
            )
            .map_err(map_weighted_rms_norm_error)?;
        let pair_report = self
            .pair_runner
            .run_resident_from_f32_buffer(
                self.post_norm_runner.shared_context(),
                self.post_norm_runner.output_buffer_handle(),
                self.hidden,
                self.post_norm_runner.output_buffer_size(),
            )
            .map_err(map_packed_matvec_error)?;
        let swiglu_pack_report = self
            .swiglu_pack_runner
            .run_with_output_from_buffer(
                self.pair_runner.shared_context(),
                self.pair_runner.output_buffer_handle(),
                self.intermediate * 2,
                self.pair_runner.output_buffer_size(),
            )
            .map_err(map_swiglu_pack_error)?;
        let down_report = self
            .down_runner
            .run_resident_from_packed_buffer(
                self.swiglu_pack_runner.shared_context(),
                self.swiglu_pack_runner.output_buffer_handle(),
                self.swiglu_pack_runner.packed_len(),
                self.swiglu_pack_runner.output_buffer_size(),
            )
            .map_err(map_packed_matvec_error)?;
        let add_report = self
            .add_runner
            .run_resident_from_buffers(
                self.down_runner.shared_context(),
                self.down_runner.output_buffer_handle(),
                self.hidden,
                self.down_runner.output_buffer_size(),
                source_buffer,
                source_len,
                source_buffer_size,
            )
            .map_err(map_vector_add_error)?;
        Ok(GpuMlpBlockReport {
            hidden: self.hidden,
            intermediate: self.intermediate,
            compile_duration: self.compile_duration,
            post_norm_gpu_duration: post_norm_report.gpu_duration,
            pair_gpu_duration: pair_report.gpu_duration,
            swiglu_pack_gpu_duration: swiglu_pack_report.gpu_duration,
            down_gpu_duration: down_report.gpu_duration,
            residual_add_gpu_duration: add_report.gpu_duration,
        })
    }

    pub fn run_from_resident_tensor(
        &mut self,
        source: &GpuResidentBuffer,
        post_norm_weight: &[f32],
    ) -> Result<GpuMlpBlockReport, GpuMlpBlockError> {
        self.run_from_resident_residual(
            &source.shared_context,
            source.buffer,
            source.len,
            source.buffer_size,
            post_norm_weight,
        )
    }

    pub fn compile_duration(&self) -> Duration {
        self.compile_duration
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

    pub fn resident_output(&self) -> GpuResidentBuffer {
        GpuResidentBuffer::new(
            self.shared_context().clone(),
            self.output_buffer_handle(),
            self.hidden(),
            self.output_buffer_size(),
        )
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn read_output(&self) -> Result<(Vec<f32>, Duration), GpuMlpBlockError> {
        self.add_runner.read_output().map_err(map_vector_add_error)
    }
}

fn map_packed_matvec_error(error: GpuPackedMatvecError) -> GpuMlpBlockError {
    GpuMlpBlockError(error.to_string())
}

fn map_weighted_rms_norm_error(error: GpuWeightedRmsNormError) -> GpuMlpBlockError {
    GpuMlpBlockError(error.to_string())
}

fn map_vector_add_error(error: GpuVectorAddError) -> GpuMlpBlockError {
    GpuMlpBlockError(error.to_string())
}

fn map_swiglu_pack_error(error: GpuSwigluPackF16PairsError) -> GpuMlpBlockError {
    GpuMlpBlockError(error.to_string())
}
