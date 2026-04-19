use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum ScanError {
    Io(std::io::Error),
    SafeTensors(safetensors::SafeTensorError),
}

impl std::fmt::Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::SafeTensors(error) => write!(f, "safetensors error: {error}"),
        }
    }
}

impl std::error::Error for ScanError {}

impl From<std::io::Error> for ScanError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<safetensors::SafeTensorError> for ScanError {
    fn from(value: safetensors::SafeTensorError) -> Self {
        Self::SafeTensors(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInventoryEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorInventory {
    pub path: PathBuf,
    pub file_bytes: usize,
    pub tensor_bytes: usize,
    pub header_and_padding_bytes: usize,
    pub tensor_count: usize,
    pub scan_duration: Duration,
    pub dtypes: BTreeMap<String, usize>,
    pub tensors: Vec<TensorInventoryEntry>,
}

impl SafeTensorInventory {
    pub fn summarize(&self) -> String {
        format!(
            "path={} tensors={} tensor_bytes={} file_bytes={} overhead_bytes={} scan_ms={:.3}",
            self.path.display(),
            self.tensor_count,
            self.tensor_bytes,
            self.file_bytes,
            self.header_and_padding_bytes,
            self.scan_duration.as_secs_f64() * 1_000.0,
        )
    }
}

pub fn scan_safetensors_file(path: impl AsRef<Path>) -> Result<SafeTensorInventory, ScanError> {
    let path = path.as_ref();
    let started_at = Instant::now();
    let file = File::open(path)?;
    let file_bytes = file.metadata()?.len() as usize;
    let mmap = map_read_only(&file)?;
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut entries = Vec::new();
    let mut tensor_bytes = 0usize;
    let mut dtypes = BTreeMap::new();

    for name in tensors.names() {
        let tensor = tensors.tensor(name)?;
        let dtype = format!("{:?}", tensor.dtype());
        let shape = tensor.shape().to_vec();
        let data_bytes = tensor.data().len();
        tensor_bytes += data_bytes;
        *dtypes.entry(dtype.clone()).or_insert(0) += 1;
        entries.push(TensorInventoryEntry {
            name: name.to_owned(),
            dtype,
            shape,
            data_bytes,
        });
    }

    entries.sort_by(|left, right| left.name.cmp(&right.name));

    Ok(SafeTensorInventory {
        path: path.to_path_buf(),
        file_bytes,
        tensor_bytes,
        header_and_padding_bytes: file_bytes.saturating_sub(tensor_bytes),
        tensor_count: entries.len(),
        scan_duration: started_at.elapsed(),
        dtypes,
        tensors: entries,
    })
}

fn map_read_only(file: &File) -> Result<Mmap, std::io::Error> {
    unsafe { Mmap::map(file) }
}

#[cfg(test)]
mod tests {
    use super::scan_safetensors_file;
    use bytemuck::cast_slice;
    use safetensors::tensor::{Dtype, TensorView, serialize};
    use std::collections::BTreeMap;
    use std::fs;
    use tempfile::NamedTempFile;

    fn write_test_file() -> NamedTempFile {
        let temp = NamedTempFile::new().expect("temp file should be created");
        let a = [1f32, 2.0, 3.0, 4.0];
        let b = [5u16, 6, 7, 8, 9, 10];

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "layer.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 2], cast_slice(&a)).expect("view should be built"),
        );
        tensors.insert(
            "layer.scale".to_string(),
            TensorView::new(Dtype::U16, vec![2, 3], cast_slice(&b)).expect("view should be built"),
        );

        let bytes = serialize(tensors, &None).expect("serialization should succeed");
        fs::write(temp.path(), bytes).expect("serialized bytes should be written");
        temp
    }

    #[test]
    fn scans_tensor_inventory_and_memory_usage() {
        let temp = write_test_file();
        let inventory = scan_safetensors_file(temp.path()).expect("scan should succeed");

        assert_eq!(inventory.tensor_count, 2);
        assert_eq!(inventory.tensor_bytes, 28);
        assert!(inventory.file_bytes >= inventory.tensor_bytes);
        assert!(inventory.header_and_padding_bytes > 0);
        assert_eq!(inventory.dtypes.get("F32"), Some(&1));
        assert_eq!(inventory.dtypes.get("U16"), Some(&1));
        assert_eq!(inventory.tensors[0].name, "layer.scale");
        assert_eq!(inventory.tensors[1].name, "layer.weight");
        assert_eq!(inventory.tensors[0].shape, vec![2, 3]);
        assert_eq!(inventory.tensors[1].shape, vec![2, 2]);
    }

    #[test]
    fn summary_includes_live_analysis_fields() {
        let temp = write_test_file();
        let inventory = scan_safetensors_file(temp.path()).expect("scan should succeed");
        let summary = inventory.summarize();

        assert!(summary.contains("tensors=2"));
        assert!(summary.contains("tensor_bytes=28"));
        assert!(summary.contains("scan_ms="));
    }
}
