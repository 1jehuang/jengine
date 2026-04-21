use std::sync::Arc;
use std::time::Duration;

use crate::gpu::full_last_layer_block::PackedLinearSpec;
use crate::gpu::pack_f16_pairs::{CachedGpuPackF16PairsRunner, GpuPackF16PairsError};
use crate::gpu::packed_matvec::{
    CachedGpuPackedMatvecRunner, GpuPackedMatvecError, SharedGpuPackedContext,
};
use crate::gpu::resident_buffer::GpuResidentBuffer;
use crate::gpu::weighted_rms_norm::{CachedGpuWeightedRmsNormRunner, GpuWeightedRmsNormError};

#[derive(Debug, Clone, PartialEq)]
pub struct GpuTailBlockReport {
    pub hidden: usize,
    pub vocab: usize,
    pub compile_duration: Duration,
    pub final_norm_gpu_duration: Duration,
    pub pack_gpu_duration: Duration,
    pub logits_gpu_duration: Duration,
    pub logits_download_duration: Duration,
    pub argmax_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuTailBlockLogitsReport {
    pub hidden: usize,
    pub vocab: usize,
    pub compile_duration: Duration,
    pub final_norm_gpu_duration: Duration,
    pub pack_gpu_duration: Duration,
    pub logits_gpu_duration: Duration,
    pub logits_download_duration: Duration,
}

#[derive(Debug)]
pub struct GpuTailBlockError(String);

impl std::fmt::Display for GpuTailBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuTailBlockError {}

pub struct CachedGpuTailBlockRunner {
    hidden: usize,
    vocab: usize,
    compile_duration: Duration,
    final_norm_runner: CachedGpuWeightedRmsNormRunner,
    pack_runner: CachedGpuPackF16PairsRunner,
    logits_runner: CachedGpuPackedMatvecRunner,
    submit_fence: ash::vk::Fence,
}

impl CachedGpuTailBlockRunner {
    pub fn new_with_context(
        context: Arc<SharedGpuPackedContext>,
        hidden: usize,
        vocab: usize,
        epsilon: f32,
        logits_spec: &PackedLinearSpec,
    ) -> Result<Self, GpuTailBlockError> {
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
        let submit_fence = unsafe {
            final_norm_runner
                .shared_context()
                .device
                .create_fence(&ash::vk::FenceCreateInfo::default(), None)
        }
        .map_err(|error| GpuTailBlockError(format!("vulkan fence creation failed: {error:?}")))?;
        Ok(Self {
            hidden,
            vocab,
            compile_duration: final_norm_compile + pack_compile + logits_compile,
            final_norm_runner,
            pack_runner,
            logits_runner,
            submit_fence,
        })
    }

    pub fn run_argmax(
        &mut self,
        hidden_input: &[f32],
        final_norm_weight: &[f32],
    ) -> Result<GpuTailBlockReport, GpuTailBlockError> {
        if hidden_input.len() != self.hidden || final_norm_weight.len() != self.hidden {
            return Err(GpuTailBlockError(
                "tail block host inputs must match hidden size".to_string(),
            ));
        }
        let final_norm_report = self
            .final_norm_runner
            .run_resident(hidden_input, final_norm_weight)
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
        Ok(GpuTailBlockReport {
            hidden: self.hidden,
            vocab: self.vocab,
            compile_duration: self.compile_duration,
            final_norm_gpu_duration: final_norm_report.gpu_duration,
            pack_gpu_duration: pack_report.gpu_duration,
            logits_gpu_duration: logits_report.gpu_duration,
            logits_download_duration,
            argmax_index,
        })
    }

    pub fn run_logits(
        &mut self,
        hidden_input: &[f32],
        final_norm_weight: &[f32],
    ) -> Result<(Vec<f32>, GpuTailBlockLogitsReport), GpuTailBlockError> {
        if hidden_input.len() != self.hidden || final_norm_weight.len() != self.hidden {
            return Err(GpuTailBlockError(
                "tail block host inputs must match hidden size".to_string(),
            ));
        }
        let final_norm_report = self
            .final_norm_runner
            .run_resident(hidden_input, final_norm_weight)
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
        let (logits, logits_download_duration) = self
            .logits_runner
            .read_output()
            .map_err(map_packed_matvec_error)?;
        Ok((
            logits,
            GpuTailBlockLogitsReport {
                hidden: self.hidden,
                vocab: self.vocab,
                compile_duration: self.compile_duration,
                final_norm_gpu_duration: final_norm_report.gpu_duration,
                pack_gpu_duration: pack_report.gpu_duration,
                logits_gpu_duration: logits_report.gpu_duration,
                logits_download_duration,
            },
        ))
    }

    pub fn run_argmax_from_resident_hidden(
        &mut self,
        source_context: &Arc<SharedGpuPackedContext>,
        source_buffer: ash::vk::Buffer,
        source_len: usize,
        source_buffer_size: u64,
        final_norm_weight: &[f32],
    ) -> Result<GpuTailBlockReport, GpuTailBlockError> {
        if source_len != self.hidden || final_norm_weight.len() != self.hidden {
            return Err(GpuTailBlockError(
                "tail block resident inputs must match hidden size".to_string(),
            ));
        }
        self.final_norm_runner
            .prepare_resident_from_f32_buffer(
                source_context,
                source_buffer,
                source_len,
                source_buffer_size,
                final_norm_weight,
            )
            .map_err(map_weighted_rms_norm_error)?;
        self.pack_runner
            .prepare_resident_from_f32_buffer(
                self.final_norm_runner.shared_context(),
                self.final_norm_runner.output_buffer_handle(),
                self.hidden,
                self.final_norm_runner.output_buffer_size(),
            )
            .map_err(map_pack_f16_pairs_error)?;
        self.logits_runner
            .prepare_resident_from_packed_buffer(
                self.pack_runner.shared_context(),
                self.pack_runner.output_buffer_handle(),
                self.pack_runner.packed_len(),
                self.pack_runner.output_buffer_size(),
            )
            .map_err(map_packed_matvec_error)?;
        let command_buffers = [
            self.final_norm_runner.command_buffer_handle(),
            self.pack_runner.command_buffer_handle(),
            self.logits_runner.command_buffer_handle(),
        ];
        let submit_info = [ash::vk::SubmitInfo::default().command_buffers(&command_buffers)];
        let gpu_started = std::time::Instant::now();
        unsafe {
            self.final_norm_runner
                .shared_context()
                .device
                .reset_fences(&[self.submit_fence])
        }
        .map_err(|error| GpuTailBlockError(format!("vulkan fence reset failed: {error:?}")))?;
        unsafe {
            self.final_norm_runner.shared_context().device.queue_submit(
                self.final_norm_runner.shared_context().queue,
                &submit_info,
                self.submit_fence,
            )
        }
        .map_err(|error| GpuTailBlockError(format!("vulkan queue submit failed: {error:?}")))?;
        unsafe {
            self.final_norm_runner.shared_context().device.wait_for_fences(
                &[self.submit_fence],
                true,
                u64::MAX,
            )
        }
        .map_err(|error| GpuTailBlockError(format!("vulkan fence wait failed: {error:?}")))?;
        let total_gpu_duration = gpu_started.elapsed();
        let (argmax_index, logits_download_duration) = self
            .logits_runner
            .argmax_output()
            .map_err(map_packed_matvec_error)?;
        Ok(GpuTailBlockReport {
            hidden: self.hidden,
            vocab: self.vocab,
            compile_duration: self.compile_duration,
            final_norm_gpu_duration: total_gpu_duration,
            pack_gpu_duration: Duration::ZERO,
            logits_gpu_duration: Duration::ZERO,
            logits_download_duration,
            argmax_index,
        })
    }

    pub fn run_logits_from_resident_hidden(
        &mut self,
        source_context: &Arc<SharedGpuPackedContext>,
        source_buffer: ash::vk::Buffer,
        source_len: usize,
        source_buffer_size: u64,
        final_norm_weight: &[f32],
    ) -> Result<(Vec<f32>, GpuTailBlockLogitsReport), GpuTailBlockError> {
        if source_len != self.hidden || final_norm_weight.len() != self.hidden {
            return Err(GpuTailBlockError(
                "tail block resident inputs must match hidden size".to_string(),
            ));
        }
        self.final_norm_runner
            .prepare_resident_from_f32_buffer(
                source_context,
                source_buffer,
                source_len,
                source_buffer_size,
                final_norm_weight,
            )
            .map_err(map_weighted_rms_norm_error)?;
        self.pack_runner
            .prepare_resident_from_f32_buffer(
                self.final_norm_runner.shared_context(),
                self.final_norm_runner.output_buffer_handle(),
                self.hidden,
                self.final_norm_runner.output_buffer_size(),
            )
            .map_err(map_pack_f16_pairs_error)?;
        self.logits_runner
            .prepare_resident_from_packed_buffer(
                self.pack_runner.shared_context(),
                self.pack_runner.output_buffer_handle(),
                self.pack_runner.packed_len(),
                self.pack_runner.output_buffer_size(),
            )
            .map_err(map_packed_matvec_error)?;
        let command_buffers = [
            self.final_norm_runner.command_buffer_handle(),
            self.pack_runner.command_buffer_handle(),
            self.logits_runner.command_buffer_handle(),
        ];
        let submit_info = [ash::vk::SubmitInfo::default().command_buffers(&command_buffers)];
        let gpu_started = std::time::Instant::now();
        unsafe {
            self.final_norm_runner
                .shared_context()
                .device
                .reset_fences(&[self.submit_fence])
        }
        .map_err(|error| GpuTailBlockError(format!("vulkan fence reset failed: {error:?}")))?;
        unsafe {
            self.final_norm_runner.shared_context().device.queue_submit(
                self.final_norm_runner.shared_context().queue,
                &submit_info,
                self.submit_fence,
            )
        }
        .map_err(|error| GpuTailBlockError(format!("vulkan queue submit failed: {error:?}")))?;
        unsafe {
            self.final_norm_runner.shared_context().device.wait_for_fences(
                &[self.submit_fence],
                true,
                u64::MAX,
            )
        }
        .map_err(|error| GpuTailBlockError(format!("vulkan fence wait failed: {error:?}")))?;
        let total_gpu_duration = gpu_started.elapsed();
        let (logits, logits_download_duration) = self
            .logits_runner
            .read_output()
            .map_err(map_packed_matvec_error)?;
        Ok((
            logits,
            GpuTailBlockLogitsReport {
                hidden: self.hidden,
                vocab: self.vocab,
                compile_duration: self.compile_duration,
                final_norm_gpu_duration: total_gpu_duration,
                pack_gpu_duration: Duration::ZERO,
                logits_gpu_duration: Duration::ZERO,
                logits_download_duration,
            },
        ))
    }

    pub fn run_argmax_from_resident_tensor(
        &mut self,
        source: &GpuResidentBuffer,
        final_norm_weight: &[f32],
    ) -> Result<GpuTailBlockReport, GpuTailBlockError> {
        self.run_argmax_from_resident_hidden(
            &source.shared_context,
            source.buffer,
            source.len,
            source.buffer_size,
            final_norm_weight,
        )
    }

    pub fn run_logits_from_resident_tensor(
        &mut self,
        source: &GpuResidentBuffer,
        final_norm_weight: &[f32],
    ) -> Result<(Vec<f32>, GpuTailBlockLogitsReport), GpuTailBlockError> {
        self.run_logits_from_resident_hidden(
            &source.shared_context,
            source.buffer,
            source.len,
            source.buffer_size,
            final_norm_weight,
        )
    }

    pub fn compile_duration(&self) -> Duration {
        self.compile_duration
    }
}

fn map_packed_matvec_error(error: GpuPackedMatvecError) -> GpuTailBlockError {
    GpuTailBlockError(error.to_string())
}

fn map_weighted_rms_norm_error(error: GpuWeightedRmsNormError) -> GpuTailBlockError {
    GpuTailBlockError(error.to_string())
}

fn map_pack_f16_pairs_error(error: GpuPackF16PairsError) -> GpuTailBlockError {
    GpuTailBlockError(error.to_string())
}

impl Drop for CachedGpuTailBlockRunner {
    fn drop(&mut self) {
        unsafe {
            self.final_norm_runner
                .shared_context()
                .device
                .destroy_fence(self.submit_fence, None);
        }
    }
}
