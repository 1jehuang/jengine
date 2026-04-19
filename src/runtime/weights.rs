use crate::runtime::assets::BonsaiAssetPaths;
use half::f16;
use rayon::prelude::*;
use safetensors::tensor::{Dtype, SafeTensors};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

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

pub struct WeightStore {
    bytes: Vec<u8>,
}

impl WeightStore {
    pub fn load_from_assets(paths: &BonsaiAssetPaths) -> Result<Self, WeightError> {
        Self::load_from_file(&paths.safetensors_path)
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, WeightError> {
        let bytes = fs::read(path)?;
        Ok(Self { bytes })
    }

    pub fn probe(&self) -> Result<WeightProbe, WeightError> {
        let started_at = Instant::now();
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let mut names = tensors
            .names()
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
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
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let tensor = tensors
            .tensor(name)
            .map_err(|_| WeightError::MissingTensor(name.to_string()))?;
        decode_f16_tensor(tensor.data(), tensor.dtype(), tensor.shape())
    }

    pub fn embedding_lookup(&self, name: &str, token_id: usize) -> Result<Vec<f32>, WeightError> {
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let tensor = tensors
            .tensor(name)
            .map_err(|_| WeightError::MissingTensor(name.to_string()))?;
        let shape = tensor.shape();
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
        match tensor.dtype() {
            Dtype::F16 => {
                let row_bytes = cols * 2;
                let start = token_id * row_bytes;
                let end = start + row_bytes;
                let row = &tensor.data()[start..end];
                Ok(row
                    .chunks_exact(2)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect())
            }
            dtype => Err(WeightError::UnsupportedDtype(dtype)),
        }
    }

    pub fn matvec_f16(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, WeightError> {
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let tensor = tensors
            .tensor(name)
            .map_err(|_| WeightError::MissingTensor(name.to_string()))?;
        let shape = tensor.shape();
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
        match tensor.dtype() {
            Dtype::F16 => {
                let row_bytes = cols * 2;
                let data = tensor.data();
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
        let output = store
            .matvec_f16("model.layers.0.self_attn.q_proj.weight", &embedding)
            .expect("matvec should succeed");
        let norm = store
            .load_vector_f32("model.layers.0.input_layernorm.weight")
            .expect("vector load should succeed");

        assert_eq!(embedding, vec![3.0, 4.0]);
        assert_eq!(output.len(), 3);
        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 4.0).abs() < 1e-5);
        assert!((output[2] - 18.0).abs() < 1e-5);
        assert_eq!(norm, vec![1.0, 1.0, 1.0]);
    }
}
