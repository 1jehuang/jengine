use std::fmt;

pub const TERNARY_G128_GROUP_SIZE: usize = 128;

#[derive(Debug, Clone, PartialEq)]
pub struct PackedTernaryTensor {
    pub shape: Vec<usize>,
    pub original_len: usize,
    pub group_size: usize,
    pub packed_codes: Vec<u8>,
    pub scales: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackReport {
    pub elements: usize,
    pub groups: usize,
    pub perfect_groups: usize,
    pub zero_only_groups: usize,
    pub max_group_error: f32,
    pub worst_group_index: usize,
    pub estimated_packed_bytes_fp16_scales: usize,
    pub reduction_ratio_vs_fp16: f64,
}

impl PackReport {
    pub fn summarize(&self) -> String {
        format!(
            "elements={} groups={} perfect_groups={} zero_only_groups={} max_group_error={:.6} worst_group={} packed_bytes={} reduction_x={:.3}",
            self.elements,
            self.groups,
            self.perfect_groups,
            self.zero_only_groups,
            self.max_group_error,
            self.worst_group_index,
            self.estimated_packed_bytes_fp16_scales,
            self.reduction_ratio_vs_fp16,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RepackError {
    Shape(String),
    NonRepresentableGroup {
        group_index: usize,
        max_group_error: f32,
        tolerance: f32,
    },
}

impl fmt::Display for RepackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(message) => write!(f, "shape error: {message}"),
            Self::NonRepresentableGroup {
                group_index,
                max_group_error,
                tolerance,
            } => write!(
                f,
                "group {group_index} is not representable within tolerance {tolerance}; max error {max_group_error}"
            ),
        }
    }
}

impl std::error::Error for RepackError {}

pub fn analyze_ternary_packability(values: &[f32]) -> PackReport {
    let groups = values.len().div_ceil(TERNARY_G128_GROUP_SIZE);
    let mut perfect_groups = 0usize;
    let mut zero_only_groups = 0usize;
    let mut max_group_error = 0.0f32;
    let mut worst_group_index = 0usize;

    for (group_index, chunk) in values.chunks(TERNARY_G128_GROUP_SIZE).enumerate() {
        let analysis = analyze_group(chunk);
        if analysis.max_error <= 1e-6 {
            perfect_groups += 1;
        }
        if analysis.scale == 0.0 {
            zero_only_groups += 1;
        }
        if analysis.max_error > max_group_error {
            max_group_error = analysis.max_error;
            worst_group_index = group_index;
        }
    }

    let packed_codes = values.len().div_ceil(4);
    let scale_bytes = groups * 2;
    let estimated_packed_bytes_fp16_scales = packed_codes + scale_bytes;
    let fp16_bytes = values.len() * 2;
    let reduction_ratio_vs_fp16 = fp16_bytes as f64 / estimated_packed_bytes_fp16_scales as f64;

    PackReport {
        elements: values.len(),
        groups,
        perfect_groups,
        zero_only_groups,
        max_group_error,
        worst_group_index,
        estimated_packed_bytes_fp16_scales,
        reduction_ratio_vs_fp16,
    }
}

pub fn pack_ternary_g128(
    values: &[f32],
    shape: Vec<usize>,
    tolerance: f32,
) -> Result<(PackedTernaryTensor, PackReport), RepackError> {
    let expected_len = shape.iter().product::<usize>();
    if expected_len != values.len() {
        return Err(RepackError::Shape(format!(
            "shape implies {expected_len} elements but values has {}",
            values.len()
        )));
    }

    let groups = values.len().div_ceil(TERNARY_G128_GROUP_SIZE);
    let mut codes = vec![0u8; values.len().div_ceil(4)];
    let mut scales = Vec::with_capacity(groups);

    for (group_index, chunk) in values.chunks(TERNARY_G128_GROUP_SIZE).enumerate() {
        let analysis = analyze_group(chunk);
        if analysis.max_error > tolerance {
            return Err(RepackError::NonRepresentableGroup {
                group_index,
                max_group_error: analysis.max_error,
                tolerance,
            });
        }
        scales.push(analysis.scale);
        for (offset, &value) in chunk.iter().enumerate() {
            let code = quantize_to_code(value, analysis.scale);
            let element_index = group_index * TERNARY_G128_GROUP_SIZE + offset;
            set_code(&mut codes, element_index, code);
        }
    }

    let packed = PackedTernaryTensor {
        shape,
        original_len: values.len(),
        group_size: TERNARY_G128_GROUP_SIZE,
        packed_codes: codes,
        scales,
    };
    let report = analyze_ternary_packability(values);
    Ok((packed, report))
}

pub fn unpack_ternary_g128(packed: &PackedTernaryTensor) -> Result<Vec<f32>, RepackError> {
    let expected_len = packed.shape.iter().product::<usize>();
    if expected_len != packed.original_len {
        return Err(RepackError::Shape(format!(
            "shape implies {expected_len} elements but packed tensor stores {}",
            packed.original_len
        )));
    }
    let mut values = Vec::with_capacity(packed.original_len);
    for group_index in 0..packed.scales.len() {
        let scale = packed.scales[group_index];
        let group_start = group_index * packed.group_size;
        let group_end = (group_start + packed.group_size).min(packed.original_len);
        for element_index in group_start..group_end {
            let code = get_code(&packed.packed_codes, element_index);
            values.push(match code {
                0 => 0.0,
                1 => scale,
                2 => -scale,
                _ => 0.0,
            });
        }
    }
    Ok(values)
}

pub fn matvec_packed_ternary(
    packed: &PackedTernaryTensor,
    input: &[f32],
) -> Result<Vec<f32>, RepackError> {
    if packed.shape.len() != 2 {
        return Err(RepackError::Shape(
            "packed matvec requires a rank-2 tensor".to_string(),
        ));
    }
    let rows = packed.shape[0];
    let cols = packed.shape[1];
    if input.len() != cols {
        return Err(RepackError::Shape(format!(
            "input length {} does not match packed tensor columns {}",
            input.len(),
            cols
        )));
    }
    let mut output = vec![0.0f32; rows];
    for (row, out) in output.iter_mut().enumerate().take(rows) {
        let mut sum = 0.0f32;
        for (col, input_value) in input.iter().enumerate().take(cols) {
            let element_index = row * cols + col;
            let scale = packed.scales[element_index / packed.group_size];
            let value = match get_code(&packed.packed_codes, element_index) {
                0 => 0.0,
                1 => scale,
                2 => -scale,
                _ => 0.0,
            };
            sum += value * input_value;
        }
        *out = sum;
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy)]
struct GroupAnalysis {
    scale: f32,
    max_error: f32,
}

fn analyze_group(values: &[f32]) -> GroupAnalysis {
    let scale = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if scale <= 1e-8 {
        return GroupAnalysis {
            scale: 0.0,
            max_error: 0.0,
        };
    }

    let mut max_error = 0.0f32;
    for &value in values {
        let recon = if value.abs() <= scale * 0.5 {
            0.0
        } else {
            value.signum() * scale
        };
        max_error = max_error.max((value - recon).abs());
    }
    GroupAnalysis { scale, max_error }
}

fn quantize_to_code(value: f32, scale: f32) -> u8 {
    if scale <= 1e-8 || value.abs() <= scale * 0.5 {
        0
    } else if value.is_sign_positive() {
        1
    } else {
        2
    }
}

fn set_code(bytes: &mut [u8], element_index: usize, code: u8) {
    let byte_index = element_index / 4;
    let shift = (element_index % 4) * 2;
    let mask = !(0b11 << shift);
    bytes[byte_index] = (bytes[byte_index] & mask) | ((code & 0b11) << shift);
}

fn get_code(bytes: &[u8], element_index: usize) -> u8 {
    let byte_index = element_index / 4;
    let shift = (element_index % 4) * 2;
    (bytes[byte_index] >> shift) & 0b11
}

#[cfg(test)]
mod tests {
    use super::{
        TERNARY_G128_GROUP_SIZE, analyze_ternary_packability, matvec_packed_ternary,
        pack_ternary_g128, unpack_ternary_g128,
    };

    #[test]
    fn packs_and_unpacks_exact_ternary_tensor() {
        let mut values = vec![0.0f32; TERNARY_G128_GROUP_SIZE];
        for (index, value) in values.iter_mut().enumerate() {
            *value = match index % 3 {
                0 => 0.0,
                1 => 2.5,
                _ => -2.5,
            };
        }
        let (packed, report) = pack_ternary_g128(&values, vec![values.len()], 1e-6).unwrap();
        let roundtrip = unpack_ternary_g128(&packed).unwrap();

        assert_eq!(roundtrip, values);
        assert_eq!(report.groups, 1);
        assert_eq!(report.perfect_groups, 1);
        assert!(report.reduction_ratio_vs_fp16 > 7.0);
    }

    #[test]
    fn rejects_non_representable_group_in_strict_pack_mode() {
        let values = vec![1.0f32, 0.5, -1.0, 0.0];
        let error = pack_ternary_g128(&values, vec![values.len()], 1e-6).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("not representable"));
    }

    #[test]
    fn analyzes_partial_last_group() {
        let values = vec![1.0f32, -1.0, 0.0, 1.0, -1.0];
        let report = analyze_ternary_packability(&values);
        assert_eq!(report.groups, 1);
        assert_eq!(report.perfect_groups, 1);
        assert_eq!(report.zero_only_groups, 0);
    }

    #[test]
    fn computes_reference_packed_matvec() {
        let values = vec![1.0f32, 0.0, -1.0, 0.0, 1.0, -1.0];
        let input = vec![2.0f32, 3.0, 4.0];
        let (packed, _) = pack_ternary_g128(&values, vec![2, 3], 1e-6).unwrap();
        let output = matvec_packed_ternary(&packed, &input).unwrap();
        assert_eq!(output, vec![-2.0, -1.0]);
    }
}
