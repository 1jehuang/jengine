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
pub struct RowGroupPairSidecar {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub groups_per_row: usize,
    pub pair_codes: Vec<u8>,
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

pub fn embedding_lookup_packed_ternary(
    packed: &PackedTernaryTensor,
    row_index: usize,
) -> Result<Vec<f32>, RepackError> {
    if packed.shape.len() != 2 {
        return Err(RepackError::Shape(
            "packed embedding lookup requires a rank-2 tensor".to_string(),
        ));
    }
    let rows = packed.shape[0];
    let cols = packed.shape[1];
    if row_index >= rows {
        return Err(RepackError::Shape(format!(
            "row_index {row_index} out of range for {rows} rows"
        )));
    }
    let row_start = row_index * cols;
    let row_end = row_start + cols;
    let mut values = Vec::with_capacity(cols);
    for element_index in row_start..row_end {
        let scale = packed.scales[element_index / packed.group_size];
        let code = get_code(&packed.packed_codes, element_index);
        values.push(match code {
            0 => 0.0,
            1 => scale,
            2 => -scale,
            _ => 0.0,
        });
    }
    Ok(values)
}

pub fn matvec_packed_ternary_reference(
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

    if packed.group_size == TERNARY_G128_GROUP_SIZE
        && cols.is_multiple_of(packed.group_size)
        && packed.original_len == rows * cols
    {
        return matvec_packed_ternary_grouped_aligned(packed, input, rows, cols);
    }

    matvec_packed_ternary_reference(packed, input)
}

pub fn build_row_group_pair_sidecar(
    packed: &PackedTernaryTensor,
) -> Result<RowGroupPairSidecar, RepackError> {
    if packed.shape.len() != 2 {
        return Err(RepackError::Shape(
            "row-group sidecar requires a rank-2 tensor".to_string(),
        ));
    }
    let rows = packed.shape[0];
    let cols = packed.shape[1];
    if cols % packed.group_size != 0 {
        return Err(RepackError::Shape(format!(
            "row-group sidecar requires cols {cols} to be divisible by group_size {}",
            packed.group_size
        )));
    }
    if packed.group_size % 2 != 0 {
        return Err(RepackError::Shape(format!(
            "row-group sidecar requires even group_size, got {}",
            packed.group_size
        )));
    }
    let groups_per_row = cols / packed.group_size;
    let expected_groups = rows * groups_per_row;
    if packed.scales.len() != expected_groups {
        return Err(RepackError::Shape(format!(
            "packed tensor stores {} scales but row-group sidecar expects {expected_groups}",
            packed.scales.len()
        )));
    }
    if packed.original_len != rows * cols {
        return Err(RepackError::Shape(format!(
            "packed tensor stores {} elements but row-group sidecar expects {}",
            packed.original_len,
            rows * cols
        )));
    }

    let pair_codes_per_group = packed.group_size / 2;
    let mut pair_codes = Vec::with_capacity(expected_groups * pair_codes_per_group);
    for row in 0..rows {
        let row_start = row * cols;
        for group in 0..groups_per_row {
            let group_start = row_start + group * packed.group_size;
            for pair in 0..pair_codes_per_group {
                let code0 = get_code(&packed.packed_codes, group_start + pair * 2);
                let code1 = get_code(&packed.packed_codes, group_start + pair * 2 + 1);
                pair_codes.push(code0 | (code1 << 2));
            }
        }
    }

    Ok(RowGroupPairSidecar {
        rows,
        cols,
        group_size: packed.group_size,
        groups_per_row,
        pair_codes,
        scales: packed.scales.clone(),
    })
}

pub fn matvec_row_group_pair_sidecar(
    sidecar: &RowGroupPairSidecar,
    input: &[f32],
) -> Result<Vec<f32>, RepackError> {
    if input.len() != sidecar.cols {
        return Err(RepackError::Shape(format!(
            "input length {} does not match sidecar columns {}",
            input.len(),
            sidecar.cols
        )));
    }
    let expected_groups = sidecar.rows * sidecar.groups_per_row;
    if sidecar.scales.len() != expected_groups {
        return Err(RepackError::Shape(format!(
            "sidecar stores {} scales but expects {expected_groups}",
            sidecar.scales.len()
        )));
    }
    let pair_codes_per_group = sidecar.group_size / 2;
    if sidecar.pair_codes.len() != expected_groups * pair_codes_per_group {
        return Err(RepackError::Shape(format!(
            "sidecar stores {} pair codes but expects {}",
            sidecar.pair_codes.len(),
            expected_groups * pair_codes_per_group
        )));
    }

    let mut output = vec![0.0f32; sidecar.rows];
    for (row, out) in output.iter_mut().enumerate() {
        let mut row_sum = 0.0f32;
        let row_group_base = row * sidecar.groups_per_row;
        let row_pair_base = row_group_base * pair_codes_per_group;
        for group in 0..sidecar.groups_per_row {
            let scale = sidecar.scales[row_group_base + group];
            if scale == 0.0 {
                continue;
            }
            let input_group_start = group * sidecar.group_size;
            let pair_group_start = row_pair_base + group * pair_codes_per_group;
            let pair_group_end = pair_group_start + pair_codes_per_group;
            let signed_input_sum = accumulate_pair_signed_sum(
                &sidecar.pair_codes[pair_group_start..pair_group_end],
                &input[input_group_start..input_group_start + sidecar.group_size],
            );
            row_sum += scale * signed_input_sum;
        }
        *out = row_sum;
    }
    Ok(output)
}

fn matvec_packed_ternary_grouped_aligned(
    packed: &PackedTernaryTensor,
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, RepackError> {
    let groups_per_row = cols / packed.group_size;
    let bytes_per_group = packed.group_size / 4;
    let expected_groups = rows * groups_per_row;
    if packed.scales.len() != expected_groups {
        return Err(RepackError::Shape(format!(
            "packed tensor stores {} scales but aligned matvec expects {}",
            packed.scales.len(),
            expected_groups
        )));
    }
    if packed.packed_codes.len() != packed.original_len.div_ceil(4) {
        return Err(RepackError::Shape(
            "packed code length does not match original tensor length".to_string(),
        ));
    }

    let mut output = vec![0.0f32; rows];
    for (row, out) in output.iter_mut().enumerate() {
        let row_group_base = row * groups_per_row;
        let row_code_base = row * cols / 4;
        let mut row_sum = 0.0f32;
        for group in 0..groups_per_row {
            let scale = packed.scales[row_group_base + group];
            if scale == 0.0 {
                continue;
            }
            let input_group_start = group * packed.group_size;
            let input_group_end = input_group_start + packed.group_size;
            let code_group_start = row_code_base + group * bytes_per_group;
            let code_group_end = code_group_start + bytes_per_group;
            let signed_input_sum = accumulate_group_signed_sum(
                &packed.packed_codes[code_group_start..code_group_end],
                &input[input_group_start..input_group_end],
            );
            row_sum += scale * signed_input_sum;
        }
        *out = row_sum;
    }
    Ok(output)
}

fn accumulate_group_signed_sum(codes: &[u8], input: &[f32]) -> f32 {
    debug_assert_eq!(codes.len() * 4, input.len());
    let mut sum = 0.0f32;
    for (byte_index, &byte) in codes.iter().enumerate() {
        let base = byte_index * 4;
        sum += code_input_contribution(byte & 0b11, input[base]);
        sum += code_input_contribution((byte >> 2) & 0b11, input[base + 1]);
        sum += code_input_contribution((byte >> 4) & 0b11, input[base + 2]);
        sum += code_input_contribution((byte >> 6) & 0b11, input[base + 3]);
    }
    sum
}

fn accumulate_pair_signed_sum(pair_codes: &[u8], input: &[f32]) -> f32 {
    debug_assert_eq!(pair_codes.len() * 2, input.len());
    let mut sum = 0.0f32;
    for (pair_index, &pair_code) in pair_codes.iter().enumerate() {
        let base = pair_index * 2;
        sum += code_input_contribution(pair_code & 0b11, input[base]);
        sum += code_input_contribution((pair_code >> 2) & 0b11, input[base + 1]);
    }
    sum
}

#[inline]
fn code_input_contribution(code: u8, input: f32) -> f32 {
    match code {
        0 => 0.0,
        1 => input,
        2 => -input,
        _ => 0.0,
    }
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
        TERNARY_G128_GROUP_SIZE, analyze_ternary_packability, build_row_group_pair_sidecar,
        embedding_lookup_packed_ternary, matvec_packed_ternary, matvec_packed_ternary_reference,
        matvec_row_group_pair_sidecar, pack_ternary_g128, unpack_ternary_g128,
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
    fn embedding_lookup_decodes_only_target_row() {
        let values = vec![
            0.0f32, 1.5, -1.5, 0.0, 1.5, -1.5, 0.0, 1.5, -1.5, 0.0, 1.5, -1.5,
        ];
        let (packed, _) = pack_ternary_g128(&values, vec![3, 4], 1e-6).unwrap();
        let row = embedding_lookup_packed_ternary(&packed, 1).unwrap();
        assert_eq!(row, vec![1.5, -1.5, 0.0, 1.5]);
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

    #[test]
    fn optimized_matvec_matches_reference_on_aligned_groups() {
        let rows = 2;
        let cols = TERNARY_G128_GROUP_SIZE;
        let values = (0..(rows * cols))
            .map(|index| match index % 3 {
                0 => 0.0,
                1 => 1.5,
                _ => -1.5,
            })
            .collect::<Vec<_>>();
        let input = (0..cols)
            .map(|index| (index % 7) as f32 * 0.1 - 0.2)
            .collect::<Vec<_>>();
        let (packed, _) = pack_ternary_g128(&values, vec![rows, cols], 1e-6).unwrap();

        let reference = matvec_packed_ternary_reference(&packed, &input).unwrap();
        let optimized = matvec_packed_ternary(&packed, &input).unwrap();

        for (optimized_value, reference_value) in optimized.iter().zip(reference.iter()) {
            assert!((optimized_value - reference_value).abs() <= 1e-6);
        }
    }

    #[test]
    fn builds_row_group_pair_sidecar_for_aligned_rank2_tensor() {
        let rows = 2;
        let cols = TERNARY_G128_GROUP_SIZE;
        let values = (0..(rows * cols))
            .map(|index| match index % 3 {
                0 => 0.0,
                1 => 1.5,
                _ => -1.5,
            })
            .collect::<Vec<_>>();
        let (packed, _) = pack_ternary_g128(&values, vec![rows, cols], 1e-6).unwrap();

        let sidecar = build_row_group_pair_sidecar(&packed).unwrap();

        assert_eq!(sidecar.rows, rows);
        assert_eq!(sidecar.cols, cols);
        assert_eq!(sidecar.group_size, TERNARY_G128_GROUP_SIZE);
        assert_eq!(sidecar.groups_per_row, 1);
        assert_eq!(sidecar.scales, packed.scales);
        assert_eq!(sidecar.pair_codes.len(), rows * (TERNARY_G128_GROUP_SIZE / 2));
    }

    #[test]
    fn row_group_pair_sidecar_matches_reference_on_aligned_groups() {
        let rows = 3;
        let cols = TERNARY_G128_GROUP_SIZE * 2;
        let values = (0..(rows * cols))
            .map(|index| match index % 5 {
                0 => 0.0,
                1 => 1.25,
                2 => -1.25,
                3 => 1.25,
                _ => -1.25,
            })
            .collect::<Vec<_>>();
        let input = (0..cols)
            .map(|index| (index % 11) as f32 * 0.07 - 0.3)
            .collect::<Vec<_>>();
        let (packed, _) = pack_ternary_g128(&values, vec![rows, cols], 1e-6).unwrap();
        let sidecar = build_row_group_pair_sidecar(&packed).unwrap();

        let reference = matvec_packed_ternary_reference(&packed, &input).unwrap();
        let sidecar_output = matvec_row_group_pair_sidecar(&sidecar, &input).unwrap();

        for (sidecar_value, reference_value) in sidecar_output.iter().zip(reference.iter()) {
            assert!((sidecar_value - reference_value).abs() <= 1e-6);
        }
    }
}
