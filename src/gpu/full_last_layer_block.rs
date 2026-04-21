use std::sync::Arc;
use std::time::Duration;

use crate::gpu::pack_f16_pairs::{CachedGpuPackF16PairsRunner, GpuPackF16PairsError};
use crate::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, GpuPackedMatvecError, PackedRunnerInputMode,
    SharedGpuPackedContext,
};
use crate::gpu::swiglu_pack_f16_pairs::{
    CachedGpuSwigluPackF16PairsRunner, GpuSwigluPackF16PairsError,
};
use crate::gpu::vector_add::{CachedGpuVectorAddRunner, GpuVectorAddError};
use crate::gpu::weighted_rms_norm::{CachedGpuWeightedRmsNormRunner, GpuWeightedRmsNormError};

#[derive(Debug, Clone)]
pub struct PackedLinearSpec {
    pub code_words: Vec<u32>,
    pub scales: Vec<f32>,
    pub group_size: usize,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuFullLastLayerReport {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub compile_duration: Duration,
    pub post_norm_gpu_duration: Duration,
    pub pair_gpu_duration: Duration,
    pub swiglu_pack_gpu_duration: Duration,
    pub down_gpu_duration: Duration,
    pub residual_add_gpu_duration: Duration,
    pub final_norm_gpu_duration: Duration,
    pub pack_gpu_duration: Duration,
    pub logits_gpu_duration: Duration,
    pub logits_download_duration: Duration,
    pub argmax_index: usize,
}

#[derive(Debug)]
pub struct GpuFullLastLayerError(String);

impl std::fmt::Display for GpuFullLastLayerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuFullLastLayerError {}

pub struct CachedGpuFullLastLayerRunner {
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    compile_duration: Duration,
    post_norm_runner: CachedGpuWeightedRmsNormRunner,
    pair_runner: CachedGpuPackedMatvecRunner,
    swiglu_pack_runner: CachedGpuSwigluPackF16PairsRunner,
    down_runner: CachedGpuPackedMatvecRunner,
    residual_seed_runner: CachedGpuVectorAddRunner,
    add_runner: CachedGpuVectorAddRunner,
    final_norm_runner: CachedGpuWeightedRmsNormRunner,
    pack_runner: CachedGpuPackF16PairsRunner,
    logits_runner: CachedGpuPackedMatvecRunner,
}

impl CachedGpuFullLastLayerRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        hidden: usize,
        intermediate: usize,
        vocab: usize,
        epsilon: f32,
        pair_spec: &PackedLinearSpec,
        down_spec: &PackedLinearSpec,
        logits_spec: &PackedLinearSpec,
    ) -> Result<Self, GpuFullLastLayerError> {
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
        let (residual_seed_runner, residual_seed_compile) =
            CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)
                .map_err(map_vector_add_error)?;
        let (add_runner, add_compile) =
            CachedGpuVectorAddRunner::new_with_context(context.clone(), hidden)
                .map_err(map_vector_add_error)?;
        let (final_norm_runner, final_norm_compile) =
            CachedGpuWeightedRmsNormRunner::new_with_context(context.clone(), hidden, epsilon)
                .map_err(map_weighted_rms_norm_error)?;
        let (pack_runner, pack_compile) =
            CachedGpuPackF16PairsRunner::new_with_context(context.clone(), hidden)
                .map_err(map_pack_f16_pairs_error)?;
        let (logits_runner, logits_compile) = CachedGpuPackedMatvecRunner::new_with_context(
            context,
            &logits_spec.code_words,
            &logits_spec.scales,
            logits_spec.group_size,
            logits_spec.rows,
            logits_spec.cols,
        )
        .map_err(map_packed_matvec_error)?;

        Ok(Self {
            hidden,
            intermediate,
            vocab,
            compile_duration: post_norm_compile
                + pair_compile
                + swiglu_pack_compile
                + down_compile
                + residual_seed_compile
                + add_compile
                + final_norm_compile
                + pack_compile
                + logits_compile,
            post_norm_runner,
            pair_runner,
            swiglu_pack_runner,
            down_runner,
            residual_seed_runner,
            add_runner,
            final_norm_runner,
            pack_runner,
            logits_runner,
        })
    }

    pub fn run_argmax(
        &mut self,
        post_attention_residual: &[f32],
        mlp_residual: &[f32],
        post_norm_weight: &[f32],
        final_norm_weight: &[f32],
    ) -> Result<GpuFullLastLayerReport, GpuFullLastLayerError> {
        if post_attention_residual.len() != self.hidden
            || mlp_residual.len() != self.hidden
            || post_norm_weight.len() != self.hidden
            || final_norm_weight.len() != self.hidden
        {
            return Err(GpuFullLastLayerError(
                "full last-layer runner received mismatched hidden-sized buffers".to_string(),
            ));
        }

        let zeros = vec![0.0f32; self.hidden];
        let post_norm_report = self
            .post_norm_runner
            .run_resident(post_attention_residual, post_norm_weight)
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
            .run_with_output(mlp_residual, &zeros, None)
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
        let final_norm_report = self
            .final_norm_runner
            .run_resident_from_f32_buffer(
                self.add_runner.shared_context(),
                self.add_runner.output_buffer_handle(),
                self.hidden,
                self.add_runner.output_buffer_size(),
                final_norm_weight,
            )
            .map_err(map_weighted_rms_norm_error)?;
        let pack_report = self
            .pack_runner
            .run_resident_from_f32_buffer(
                self.final_norm_runner.shared_context(),
                self.final_norm_runner.output_buffer_handle(),
                self.hidden,
                self.final_norm_runner.output_buffer_size(),
            )
            .map_err(map_pack_f16_pairs_error)?;
        let logits_report = self
            .logits_runner
            .run_resident_from_packed_buffer(
                self.pack_runner.shared_context(),
                self.pack_runner.output_buffer_handle(),
                self.pack_runner.packed_len(),
                self.pack_runner.output_buffer_size(),
            )
            .map_err(map_packed_matvec_error)?;
        let (argmax_index, logits_download_duration) = self
            .logits_runner
            .argmax_output()
            .map_err(map_packed_matvec_error)?;

        Ok(GpuFullLastLayerReport {
            hidden: self.hidden,
            intermediate: self.intermediate,
            vocab: self.vocab,
            compile_duration: self.compile_duration,
            post_norm_gpu_duration: post_norm_report.gpu_duration,
            pair_gpu_duration: pair_report.gpu_duration,
            swiglu_pack_gpu_duration: swiglu_pack_report.gpu_duration,
            down_gpu_duration: down_report.gpu_duration,
            residual_add_gpu_duration: add_report.gpu_duration,
            final_norm_gpu_duration: final_norm_report.gpu_duration,
            pack_gpu_duration: pack_report.gpu_duration,
            logits_gpu_duration: logits_report.gpu_duration,
            logits_download_duration,
            argmax_index,
        })
    }
}

fn map_packed_matvec_error(error: GpuPackedMatvecError) -> GpuFullLastLayerError {
    GpuFullLastLayerError(error.to_string())
}

fn map_weighted_rms_norm_error(error: GpuWeightedRmsNormError) -> GpuFullLastLayerError {
    GpuFullLastLayerError(error.to_string())
}

fn map_pack_f16_pairs_error(error: GpuPackF16PairsError) -> GpuFullLastLayerError {
    GpuFullLastLayerError(error.to_string())
}

fn map_vector_add_error(error: GpuVectorAddError) -> GpuFullLastLayerError {
    GpuFullLastLayerError(error.to_string())
}

fn map_swiglu_pack_error(error: GpuSwigluPackF16PairsError) -> GpuFullLastLayerError {
    GpuFullLastLayerError(error.to_string())
}
