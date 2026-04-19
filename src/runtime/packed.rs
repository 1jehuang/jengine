use crate::runtime::repack::PackedTernaryTensor;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const MAGIC: &[u8; 8] = b"JTPK0001";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackedTensorMetadata {
    pub name: Option<String>,
    pub shape: Vec<usize>,
    pub group_size: usize,
    pub original_len: usize,
    pub code_bytes: usize,
    pub scale_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedTensorFile {
    pub metadata: PackedTensorMetadata,
    pub tensor: PackedTernaryTensor,
}

#[derive(Debug)]
pub enum PackedIoError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Format(String),
}

impl std::fmt::Display for PackedIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
            Self::Format(message) => write!(f, "format error: {message}"),
        }
    }
}

impl std::error::Error for PackedIoError {}

impl From<std::io::Error> for PackedIoError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for PackedIoError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl PackedTensorFile {
    pub fn new(name: Option<String>, tensor: PackedTernaryTensor) -> Self {
        let metadata = PackedTensorMetadata {
            name,
            shape: tensor.shape.clone(),
            group_size: tensor.group_size,
            original_len: tensor.original_len,
            code_bytes: tensor.packed_codes.len(),
            scale_count: tensor.scales.len(),
        };
        Self { metadata, tensor }
    }

    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<(), PackedIoError> {
        let metadata_bytes = serde_json::to_vec(&self.metadata)?;
        let mut bytes = Vec::with_capacity(
            MAGIC.len()
                + 8
                + metadata_bytes.len()
                + self.tensor.packed_codes.len()
                + self.tensor.scales.len() * 4,
        );
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&metadata_bytes);
        bytes.extend_from_slice(&self.tensor.packed_codes);
        for scale in &self.tensor.scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self, PackedIoError> {
        let bytes = fs::read(path)?;
        if bytes.len() < 16 {
            return Err(PackedIoError::Format("file too small".to_string()));
        }
        if &bytes[..8] != MAGIC {
            return Err(PackedIoError::Format("invalid magic".to_string()));
        }
        let metadata_len =
            u64::from_le_bytes(bytes[8..16].try_into().expect("metadata length bytes")) as usize;
        let metadata_start = 16;
        let metadata_end = metadata_start + metadata_len;
        if metadata_end > bytes.len() {
            return Err(PackedIoError::Format(
                "metadata extends past file end".to_string(),
            ));
        }
        let metadata: PackedTensorMetadata =
            serde_json::from_slice(&bytes[metadata_start..metadata_end])?;
        let codes_start = metadata_end;
        let codes_end = codes_start + metadata.code_bytes;
        if codes_end > bytes.len() {
            return Err(PackedIoError::Format(
                "codes extend past file end".to_string(),
            ));
        }
        let scales_start = codes_end;
        let scales_end = scales_start + metadata.scale_count * 4;
        if scales_end > bytes.len() {
            return Err(PackedIoError::Format(
                "scales extend past file end".to_string(),
            ));
        }
        let mut scales = Vec::with_capacity(metadata.scale_count);
        for chunk in bytes[scales_start..scales_end].chunks_exact(4) {
            scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let tensor = PackedTernaryTensor {
            shape: metadata.shape.clone(),
            original_len: metadata.original_len,
            group_size: metadata.group_size,
            packed_codes: bytes[codes_start..codes_end].to_vec(),
            scales,
        };
        Ok(Self { metadata, tensor })
    }
}

#[cfg(test)]
mod tests {
    use super::PackedTensorFile;
    use crate::runtime::repack::{pack_ternary_g128, unpack_ternary_g128};
    use tempfile::NamedTempFile;

    #[test]
    fn roundtrips_packed_tensor_file() {
        let values = (0..256)
            .map(|index| match index % 3 {
                0 => 0.0,
                1 => 1.25,
                _ => -1.25,
            })
            .collect::<Vec<_>>();
        let (packed, _) = pack_ternary_g128(&values, vec![16, 16], 1e-6).unwrap();
        let file = PackedTensorFile::new(Some("test.weight".to_string()), packed.clone());
        let temp = NamedTempFile::new().unwrap();
        file.write_to_path(temp.path()).unwrap();
        let loaded = PackedTensorFile::read_from_path(temp.path()).unwrap();
        let unpacked = unpack_ternary_g128(&loaded.tensor).unwrap();

        assert_eq!(loaded.metadata.name.as_deref(), Some("test.weight"));
        assert_eq!(loaded.metadata.code_bytes, packed.packed_codes.len());
        assert_eq!(loaded.tensor, packed);
        assert_eq!(unpacked, values);
    }
}
