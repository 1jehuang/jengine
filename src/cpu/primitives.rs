use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveProfile {
    pub matvec_duration: Duration,
    pub rms_norm_duration: Duration,
    pub softmax_duration: Duration,
    pub silu_duration: Duration,
}

impl PrimitiveProfile {
    pub fn summarize(&self) -> String {
        format!(
            "matvec_ms={:.3} rms_norm_ms={:.3} softmax_ms={:.3} silu_ms={:.3}",
            self.matvec_duration.as_secs_f64() * 1_000.0,
            self.rms_norm_duration.as_secs_f64() * 1_000.0,
            self.softmax_duration.as_secs_f64() * 1_000.0,
            self.silu_duration.as_secs_f64() * 1_000.0,
        )
    }
}

pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "dot operands must have the same length"
    );
    lhs.iter().zip(rhs).map(|(left, right)| left * right).sum()
}

pub fn matvec(matrix: &[f32], rows: usize, cols: usize, vector: &[f32]) -> Vec<f32> {
    assert_eq!(
        matrix.len(),
        rows * cols,
        "matrix shape must match rows * cols"
    );
    assert_eq!(
        vector.len(),
        cols,
        "vector length must match matrix column count"
    );

    matrix
        .chunks_exact(cols)
        .map(|row| dot(row, vector))
        .collect()
}

pub fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    assert_eq!(
        input.len(),
        weight.len(),
        "rms_norm input and weight lengths must match"
    );
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (mean_square + epsilon).sqrt();

    input
        .iter()
        .zip(weight)
        .map(|(value, gamma)| value * scale * gamma)
        .collect()
}

pub fn silu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|value| value / (1.0 + (-value).exp()))
        .collect()
}

pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(
        gate.len(),
        up.len(),
        "swiglu operands must have the same length"
    );
    let activated_gate = silu(gate);
    activated_gate
        .iter()
        .zip(up)
        .map(|(left, right)| left * right)
        .collect()
}

pub fn swiglu_into(gate: &[f32], up: &[f32], out: &mut Vec<f32>) {
    assert_eq!(
        gate.len(),
        up.len(),
        "swiglu operands must have the same length"
    );
    out.clear();
    out.reserve(gate.len());
    out.extend(
        gate.iter().zip(up).map(|(gate_value, up_value)| {
            let silu_gate = *gate_value / (1.0 + (-*gate_value).exp());
            silu_gate * *up_value
        }),
    );
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|value| (value - max_logit).exp())
        .collect();
    let sum = exps.iter().sum::<f32>();
    exps.into_iter().map(|value| value / sum).collect()
}

pub fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
}

pub fn profile_primitives(rows: usize, cols: usize) -> PrimitiveProfile {
    assert!(rows > 0 && cols > 0, "rows and cols must be positive");

    let matrix: Vec<f32> = (0..rows * cols)
        .map(|index| (index % 31) as f32 * 0.03125)
        .collect();
    let vector: Vec<f32> = (0..cols)
        .map(|index| (index % 17) as f32 * 0.0625)
        .collect();
    let weight: Vec<f32> = vec![1.0; rows];
    let gate: Vec<f32> = (0..rows)
        .map(|index| (index % 13) as f32 * 0.1 - 0.5)
        .collect();
    let up: Vec<f32> = (0..rows)
        .map(|index| (index % 11) as f32 * 0.15 - 0.7)
        .collect();

    let started_at = Instant::now();
    let matvec_output = matvec(&matrix, rows, cols, &vector);
    let matvec_duration = started_at.elapsed();

    let started_at = Instant::now();
    let rms_output = rms_norm(&matvec_output, &weight, 1e-6);
    let rms_norm_duration = started_at.elapsed();

    let started_at = Instant::now();
    let _softmax_output = softmax(&rms_output);
    let softmax_duration = started_at.elapsed();

    let started_at = Instant::now();
    let _silu_output = swiglu(&gate, &up);
    let silu_duration = started_at.elapsed();

    PrimitiveProfile {
        matvec_duration,
        rms_norm_duration,
        softmax_duration,
        silu_duration,
    }
}

#[cfg(test)]
mod tests {
    use super::{argmax, dot, matvec, profile_primitives, rms_norm, silu, softmax, swiglu};

    #[test]
    fn computes_dot_product() {
        let result = dot(&[1.0, 2.0, 3.0], &[0.5, 1.5, -2.0]);
        assert!((result + 2.5).abs() < 1e-6);
    }

    #[test]
    fn computes_matrix_vector_product() {
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = [0.5, 1.0, -1.0];
        let result = matvec(&matrix, 2, 3, &vector);
        assert_eq!(result.len(), 2);
        assert!((result[0] - -0.5).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn computes_rms_norm() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0, 1.0, 1.0, 1.0];
        let result = rms_norm(&input, &weight, 1e-6);
        let expected_scale = 1.0 / ((7.5_f32 + 1e-6).sqrt());
        assert!((result[0] - (1.0 * expected_scale)).abs() < 1e-6);
        assert!((result[3] - (4.0 * expected_scale)).abs() < 1e-6);
    }

    #[test]
    fn computes_silu_and_swiglu() {
        let silu_result = silu(&[-1.0, 0.0, 1.0]);
        assert!(silu_result[0] < 0.0);
        assert_eq!(silu_result[1], 0.0);
        assert!(silu_result[2] > 0.0);

        let swiglu_result = swiglu(&[0.0, 1.0], &[2.0, 3.0]);
        assert_eq!(swiglu_result[0], 0.0);
        assert!(swiglu_result[1] > 0.0);
    }

    #[test]
    fn computes_softmax_and_argmax() {
        let result = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&result), Some(2));
    }

    #[test]
    fn profiles_primitives_and_reports_summary() {
        let profile = profile_primitives(64, 128);
        let summary = profile.summarize();

        assert!(profile.matvec_duration > Duration::ZERO);
        assert!(profile.rms_norm_duration > Duration::ZERO);
        assert!(summary.contains("matvec_ms="));
        assert!(summary.contains("softmax_ms="));
    }

    use std::time::Duration;
}
