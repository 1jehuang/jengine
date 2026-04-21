use crate::runtime::assets::BonsaiAssetPaths;
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::tensor::Dtype;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};

type EmbeddingRowCache = RefCell<BTreeMap<(String, usize), Rc<Vec<f32>>>>;

#[derive(Debug)]
pub enum WeightError {
    Io(std::io::Error),
    SafeTensors(safetensors::SafeTensorError),
    MissingTensor(String),
    UnsupportedDtype(Dtype),
    Shape(String),
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::SafeTensors(error) => write!(f, "safetensors error: {error}"),
            Self::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            Self::UnsupportedDtype(dtype) => write!(f, "unsupported tensor dtype: {dtype:?}"),
            Self::Shape(message) => write!(f, "shape error: {message}"),
        }
    }
}

impl std::error::Error for WeightError {}

impl From<std::io::Error> for WeightError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<safetensors::SafeTensorError> for WeightError {
    fn from(value: safetensors::SafeTensorError) -> Self {
        Self::SafeTensors(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadProgress {
    pub file_bytes: u64,
    pub header_bytes: usize,
    pub total_tensors: usize,
    pub available_tensors: usize,
    pub max_complete_layer: Option<usize>,
    pub next_missing_tensor: Option<String>,
}

impl DownloadProgress {
    pub fn summarize(&self) -> String {
        format!(
            "file_bytes={} header_bytes={} available_tensors={} total_tensors={} max_complete_layer={:?} next_missing_tensor={}",
            self.file_bytes,
            self.header_bytes,
            self.available_tensors,
            self.total_tensors,
            self.max_complete_layer,
            self.next_missing_tensor
                .clone()
                .unwrap_or_else(|| "<none>".to_string()),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightProbe {
    pub tensor_count: usize,
    pub names: Vec<String>,
    pub probe_duration: Duration,
}

impl WeightProbe {
    pub fn summarize(&self) -> String {
        format!(
            "tensor_count={} first_tensor={} probe_ms={:.3}",
            self.tensor_count,
            self.names
                .first()
                .cloned()
                .unwrap_or_else(|| "<none>".to_string()),
            self.probe_duration.as_secs_f64() * 1_000.0,
        )
    }
}

enum WeightStorage {
    #[cfg(test)]
    Owned(Vec<u8>),
    Mapped(Mmap),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorIndexEntry {
    dtype: Dtype,
    shape: Vec<usize>,
    data_start: usize,
    data_end: usize,
}

pub struct WeightStore {
    storage: WeightStorage,
    tensors: BTreeMap<String, TensorIndexEntry>,
    vector_cache: RefCell<BTreeMap<String, Rc<Vec<f32>>>>,
    embedding_row_cache: EmbeddingRowCache,
}

impl WeightStore {
    pub fn download_progress(path: impl AsRef<Path>) -> Result<DownloadProgress, WeightError> {
        let path = path.as_ref();
        let file_bytes = fs::metadata(path)?.len();
        let mut file = fs::File::open(path)?;
        let mut len_buf = [0u8; 8];
        use std::io::Read;
        file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let mut header_buf = vec![0u8; header_len];
        file.read_exact(&mut header_buf)?;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&header_buf).map_err(|error| {
                WeightError::Shape(format!("invalid safetensors header: {error}"))
            })?;

        let mut available_tensors = 0usize;
        let mut total_tensors = 0usize;
        let mut max_complete_layer = None;
        let mut next_missing_tensor = None;

        for (name, info) in &header {
            if name == "__metadata__" {
                continue;
            }
            total_tensors += 1;
            let offsets = info
                .get("data_offsets")
                .and_then(|value| value.as_array())
                .ok_or_else(|| WeightError::Shape(format!("tensor {name} missing data_offsets")))?;
            let end = offsets[1].as_u64().ok_or_else(|| {
                WeightError::Shape(format!("tensor {name} has invalid end offset"))
            })?;
            let abs_end = 8u64 + header_len as u64 + end;
            if abs_end <= file_bytes {
                available_tensors += 1;
                if let Some(layer) = name.strip_prefix("model.layers.")
                    && let Some(layer_id) = layer
                        .split('.')
                        .next()
                        .and_then(|value| value.parse::<usize>().ok())
                {
                    max_complete_layer = Some(
                        max_complete_layer.map_or(layer_id, |current: usize| current.max(layer_id)),
                    );
                }
            } else if next_missing_tensor.is_none() {
                next_missing_tensor = Some(name.clone());
            }
        }

        Ok(DownloadProgress {
            file_bytes,
            header_bytes: header_len,
            total_tensors,
            available_tensors,
            max_complete_layer,
            next_missing_tensor,
        })
    }

    pub fn load_from_assets(paths: &BonsaiAssetPaths) -> Result<Self, WeightError> {
        Self::load_from_file(&paths.safetensors_path)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, WeightError> {
        let file = fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = parse_tensor_index(&mmap)?;
        Ok(Self {
            storage: WeightStorage::Mapped(mmap),
            tensors,
            vector_cache: RefCell::new(BTreeMap::new()),
            embedding_row_cache: RefCell::new(BTreeMap::new()),
        })
    }

    #[cfg(test)]
    fn from_bytes(bytes: Vec<u8>) -> Self {
        let tensors = parse_tensor_index(&bytes).expect("test fixture tensor index should build");
        Self {
            storage: WeightStorage::Owned(bytes),
            tensors,
            vector_cache: RefCell::new(BTreeMap::new()),
            embedding_row_cache: RefCell::new(BTreeMap::new()),
        }
    }

    fn as_bytes(&self) -> &[u8] {
        match &self.storage {
            #[cfg(test)]
            WeightStorage::Owned(bytes) => bytes,
            WeightStorage::Mapped(mmap) => mmap,
        }
    }

    pub fn probe(&self) -> Result<WeightProbe, WeightError> {
        let started_at = Instant::now();
        let mut names = self.tensors.keys().cloned().collect::<Vec<_>>();
        names.sort();
        Ok(WeightProbe {
            tensor_count: names.len(),
            names,
            probe_duration: started_at.elapsed(),
        })
    }

    pub fn tensor_names(&self) -> Result<Vec<String>, WeightError> {
        Ok(self.probe()?.names)
    }

    pub fn load_vector_f32(&self, name: &str) -> Result<Vec<f32>, WeightError> {
        if let Some(cached) = self.vector_cache.borrow().get(name) {
            return Ok((**cached).clone());
        }
        let tensor = self.tensor_entry(name)?;
        let values = decode_f16_tensor(self.tensor_bytes(tensor), tensor.dtype, &tensor.shape)?;
        self.vector_cache
            .borrow_mut()
            .insert(name.to_string(), Rc::new(values.clone()));
        Ok(values)
    }

    pub fn embedding_lookup(&self, name: &str, token_id: usize) -> Result<Vec<f32>, WeightError> {
        let cache_key = (name.to_string(), token_id);
        if let Some(cached) = self.embedding_row_cache.borrow().get(&cache_key) {
            return Ok((**cached).clone());
        }
        let tensor = self.tensor_entry(name)?;
        let shape = &tensor.shape;
        if shape.len() != 2 {
            return Err(WeightError::Shape(format!(
                "embedding tensor {name} must be rank-2"
            )));
        }
        let rows = shape[0];
        let cols = shape[1];
        if token_id >= rows {
            return Err(WeightError::Shape(format!(
                "token_id {token_id} out of range for {name}"
            )));
        }
        match tensor.dtype {
            Dtype::F16 => {
                let row_bytes = cols * 2;
                let start = token_id * row_bytes;
                let end = start + row_bytes;
                let row = &self.tensor_bytes(tensor)[start..end];
                let values = row
                    .chunks_exact(2)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect::<Vec<_>>();
                self.embedding_row_cache
                    .borrow_mut()
                    .insert(cache_key, Rc::new(values.clone()));
                Ok(values)
            }
            dtype => Err(WeightError::UnsupportedDtype(dtype)),
        }
    }

    pub fn embedding_lookup_u32_words(
        &self,
        name: &str,
    ) -> Result<(usize, usize, Vec<u32>), WeightError> {
        let tensor = self.tensor_entry(name)?;
        let shape = &tensor.shape;
        if shape.len() != 2 {
            return Err(WeightError::Shape(format!(
                "embedding tensor {name} must be rank-2"
            )));
        }
        let rows = shape[0];
        let cols = shape[1];
        match tensor.dtype {
            Dtype::F16 => {
                let bytes = self.tensor_bytes(tensor);
                if bytes.len() % 4 != 0 {
                    return Err(WeightError::Shape(format!(
                        "embedding tensor {name} byte length {} is not word-aligned",
                        bytes.len()
                    )));
                }
                let words = bytes
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<_>>();
                Ok((rows, cols, words))
            }
            dtype => Err(WeightError::UnsupportedDtype(dtype)),
        }
    }

    pub fn matvec_f16(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, WeightError> {
        let tensor = self.tensor_entry(name)?;
        let shape = &tensor.shape;
        if shape.len() != 2 {
            return Err(WeightError::Shape(format!(
                "matrix tensor {name} must be rank-2"
            )));
        }
        let rows = shape[0];
        let cols = shape[1];
        if cols != input.len() {
            return Err(WeightError::Shape(format!(
                "matrix tensor {name} column count {} does not match input length {}",
                cols,
                input.len()
            )));
        }
        match tensor.dtype {
            Dtype::F16 => {
                let row_bytes = cols * 2;
                let data = self.tensor_bytes(tensor);
                let output = (0..rows)
                    .into_par_iter()
                    .map(|row| {
                        let start = row * row_bytes;
                        let end = start + row_bytes;
                        let row_bytes = &data[start..end];
                        row_bytes
                            .chunks_exact(2)
                            .zip(input.iter())
                            .map(|(chunk, input_value)| {
                                f16::from_le_bytes([chunk[0], chunk[1]]).to_f32() * input_value
                            })
                            .sum::<f32>()
                    })
                    .collect();
                Ok(output)
            }
            dtype => Err(WeightError::UnsupportedDtype(dtype)),
        }
    }

    fn tensor_entry(&self, name: &str) -> Result<&TensorIndexEntry, WeightError> {
        self.tensors
            .get(name)
            .ok_or_else(|| WeightError::MissingTensor(name.to_string()))
    }

    fn tensor_bytes<'a>(&'a self, entry: &TensorIndexEntry) -> &'a [u8] {
        &self.as_bytes()[entry.data_start..entry.data_end]
    }
}

fn parse_tensor_index(bytes: &[u8]) -> Result<BTreeMap<String, TensorIndexEntry>, WeightError> {
    if bytes.len() < 8 {
        return Err(WeightError::Shape("safetensors file too small".to_string()));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("header bytes")) as usize;
    let header_start = 8;
    let header_end = header_start + header_len;
    if header_end > bytes.len() {
        return Err(WeightError::Shape(
            "safetensors header extends past file end".to_string(),
        ));
    }
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[header_start..header_end])
            .map_err(|error| WeightError::Shape(format!("invalid safetensors header: {error}")))?;

    let mut tensors = BTreeMap::new();
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = parse_dtype(
            info.get("dtype")
                .and_then(|value| value.as_str())
                .ok_or_else(|| WeightError::Shape(format!("tensor {name} missing dtype")))?,
        )?;
        let shape =
            info.get("shape")
                .and_then(|value| value.as_array())
                .ok_or_else(|| WeightError::Shape(format!("tensor {name} missing shape")))?
                .iter()
                .map(|value| {
                    value.as_u64().map(|value| value as usize).ok_or_else(|| {
                        WeightError::Shape(format!("tensor {name} has invalid shape"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(|value| value.as_array())
            .ok_or_else(|| WeightError::Shape(format!("tensor {name} missing data_offsets")))?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| WeightError::Shape(format!("tensor {name} has invalid start offset")))?
            as usize;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| WeightError::Shape(format!("tensor {name} has invalid end offset")))?
            as usize;
        let abs_start = header_end + start;
        let abs_end = header_end + end;
        if abs_end > bytes.len() || abs_start > abs_end {
            return Err(WeightError::Shape(format!(
                "tensor {name} offsets extend past file end"
            )));
        }
        tensors.insert(
            name,
            TensorIndexEntry {
                dtype,
                shape,
                data_start: abs_start,
                data_end: abs_end,
            },
        );
    }
    Ok(tensors)
}

fn parse_dtype(value: &str) -> Result<Dtype, WeightError> {
    match value {
        "F16" => Ok(Dtype::F16),
        "F32" => Ok(Dtype::F32),
        "BF16" => Ok(Dtype::BF16),
        other => Err(WeightError::Shape(format!(
            "unsupported tensor dtype string: {other}"
        ))),
    }
}

fn decode_f16_tensor(bytes: &[u8], dtype: Dtype, shape: &[usize]) -> Result<Vec<f32>, WeightError> {
    if dtype != Dtype::F16 {
        return Err(WeightError::UnsupportedDtype(dtype));
    }
    let expected_bytes = shape.iter().product::<usize>() * 2;
    if bytes.len() != expected_bytes {
        return Err(WeightError::Shape(format!(
            "tensor byte length {} does not match expected {}",
            bytes.len(),
            expected_bytes,
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::WeightStore;
    use half::f16;
    use safetensors::tensor::{Dtype, TensorView, serialize};
    use std::collections::BTreeMap;
    use std::fs;
    use tempfile::NamedTempFile;

    fn encode(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect()
    }

    fn write_fixture() -> NamedTempFile {
        let temp = NamedTempFile::new().expect("temp file should be created");
        let emb = encode(&[1.0, 2.0, 3.0, 4.0]);
        let weight = encode(&[1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        let norm = encode(&[1.0, 1.0, 1.0]);

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            TensorView::new(Dtype::F16, vec![2, 2], &emb).expect("embedding view should build"),
        );
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorView::new(Dtype::F16, vec![3], &norm).expect("norm view should build"),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorView::new(Dtype::F16, vec![3, 2], &weight).expect("weight view should build"),
        );

        let bytes = serialize(tensors, &None).expect("serialization should succeed");
        fs::write(temp.path(), bytes).expect("fixture should be written");
        temp
    }

    #[test]
    fn probes_tensor_names() {
        let fixture = write_fixture();
        let store = WeightStore::load_from_file(fixture.path()).expect("weights should load");
        let probe = store.probe().expect("probe should succeed");
        let summary = probe.summarize();

        assert_eq!(probe.tensor_count, 3);
        assert!(
            probe
                .names
                .iter()
                .any(|name| name == "model.embed_tokens.weight")
        );
        assert!(summary.contains("tensor_count=3"));
    }

    #[test]
    fn performs_embedding_lookup_and_f16_matvec() {
        let fixture = write_fixture();
        let store = WeightStore::load_from_file(fixture.path()).expect("weights should load");
        let embedding = store
            .embedding_lookup("model.embed_tokens.weight", 1)
            .expect("embedding lookup should succeed");
        let embedding_again = store
            .embedding_lookup("model.embed_tokens.weight", 1)
            .expect("cached embedding lookup should succeed");
        let output = store
            .matvec_f16("model.layers.0.self_attn.q_proj.weight", &embedding)
            .expect("matvec should succeed");
        let norm = store
            .load_vector_f32("model.layers.0.input_layernorm.weight")
            .expect("vector load should succeed");

        assert_eq!(embedding, vec![3.0, 4.0]);
        assert_eq!(embedding_again, embedding);
        assert_eq!(output.len(), 3);
        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 4.0).abs() < 1e-5);
        assert!((output[2] - 18.0).abs() < 1e-5);
        assert_eq!(norm, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn supports_owned_bytes_storage_for_small_fixtures() {
        let fixture = write_fixture();
        let bytes = fs::read(fixture.path()).expect("fixture bytes should load");
        let store = WeightStore::from_bytes(bytes);
        let probe = store.probe().expect("probe should succeed");
        assert_eq!(probe.tensor_count, 3);
    }
}
