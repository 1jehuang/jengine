use crate::model::config::BonsaiModelConfig;
use crate::runtime::assets::{AssetError, BonsaiAssetPaths};
use crate::runtime::packed::{PackedIoError, PackedTensorFile};
use crate::runtime::repack::{
    embedding_lookup_packed_ternary, matvec_packed_ternary, pack_ternary_g128, unpack_ternary_g128,
};
use crate::runtime::weights::{WeightError, WeightStore};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

type EmbeddingRowCache = RefCell<HashMap<(String, usize), Rc<Vec<f32>>>>;

const MANIFEST_VERSION: u32 = 1;
const PACK_TOLERANCE: f32 = 1e-3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPackSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackedModelTensorEntry {
    pub name: String,
    pub rel_path: String,
    pub shape: Vec<usize>,
    pub group_size: usize,
    pub original_len: usize,
    pub code_bytes: usize,
    pub scale_count: usize,
    pub total_file_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackedModelManifest {
    pub format: String,
    pub version: u32,
    pub created_unix_secs: u64,
    pub source_model_root: String,
    pub source_safetensors_path: String,
    pub source_file_bytes: u64,
    pub source_tensor_count: usize,
    pub architecture: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub pack_tolerance: f32,
    pub entries: Vec<PackedModelTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedModelSummary {
    pub entry_count: usize,
    pub packed_total_bytes: u64,
    pub source_file_bytes: u64,
}

impl PackedModelSummary {
    pub fn summarize(&self) -> String {
        let reduction = if self.packed_total_bytes == 0 {
            0.0
        } else {
            self.source_file_bytes as f64 / self.packed_total_bytes as f64
        };
        format!(
            "entry_count={} packed_total_bytes={} source_file_bytes={} reduction_x={:.3}",
            self.entry_count, self.packed_total_bytes, self.source_file_bytes, reduction
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedModelValidationReport {
    pub checked_entries: usize,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

impl PackedModelValidationReport {
    pub fn summarize(&self) -> String {
        format!(
            "checked_entries={} max_abs_diff={:.6} mean_abs_diff={:.6}",
            self.checked_entries, self.max_abs_diff, self.mean_abs_diff
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedModelBenchReport {
    pub manifest_load_duration: std::time::Duration,
    pub packed_total_bytes: u64,
    pub source_file_bytes: u64,
    pub entry_count: usize,
}

#[derive(Debug)]
pub struct PackedModelStore {
    pub artifact_dir: PathBuf,
    pub manifest: PackedModelManifest,
    entries_by_name: HashMap<String, PackedModelTensorEntry>,
    packed_tensor_cache: RefCell<HashMap<String, Rc<PackedTensorFile>>>,
    unpacked_cache: RefCell<HashMap<String, Rc<Vec<f32>>>>,
    embedding_row_cache: EmbeddingRowCache,
}

impl PackedModelStore {
    pub fn load_from_artifact_dir(
        artifact_dir: impl AsRef<Path>,
        config: &BonsaiModelConfig,
    ) -> Result<Self, PackedModelError> {
        let artifact_dir = artifact_dir.as_ref().to_path_buf();
        let manifest = load_packed_model_manifest(artifact_dir.join("manifest.json"))?;
        validate_packed_model_manifest(&manifest, config)?;
        let entries_by_name = manifest
            .entries
            .iter()
            .cloned()
            .map(|entry| (entry.name.clone(), entry))
            .collect::<HashMap<_, _>>();
        Ok(Self {
            artifact_dir,
            manifest,
            entries_by_name,
            packed_tensor_cache: RefCell::new(HashMap::new()),
            unpacked_cache: RefCell::new(HashMap::new()),
            embedding_row_cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.entries_by_name.contains_key(name)
    }

    pub fn load_vector_f32(&self, name: &str) -> Result<Option<Vec<f32>>, PackedModelError> {
        if let Some(cached) = self.unpacked_cache.borrow().get(name) {
            return Ok(Some((**cached).clone()));
        }
        let Some(packed) = self.load_packed_tensor_file(name)? else {
            return Ok(None);
        };
        let unpacked = unpack_ternary_g128(&packed.tensor).map_err(|error| {
            PackedModelError::Encode(format!(
                "{}: {error}",
                packed.metadata.name.as_deref().unwrap_or(name)
            ))
        })?;
        self.unpacked_cache
            .borrow_mut()
            .insert(name.to_string(), Rc::new(unpacked.clone()));
        Ok(Some(unpacked))
    }

    pub fn load_packed_tensor_file(
        &self,
        name: &str,
    ) -> Result<Option<PackedTensorFile>, PackedModelError> {
        let Some(entry) = self.entries_by_name.get(name) else {
            return Ok(None);
        };
        if let Some(cached) = self.packed_tensor_cache.borrow().get(name) {
            return Ok(Some((**cached).clone()));
        }
        let packed = PackedTensorFile::read_from_path(self.artifact_dir.join(&entry.rel_path))?;
        self.packed_tensor_cache
            .borrow_mut()
            .insert(name.to_string(), Rc::new(packed.clone()));
        Ok(Some(packed))
    }

    pub fn embedding_lookup(
        &self,
        name: &str,
        token_id: usize,
    ) -> Result<Option<Vec<f32>>, PackedModelError> {
        let Some(packed) = self.load_packed_tensor_file(name)? else {
            return Ok(None);
        };
        let tensor = packed.tensor;
        if tensor.shape.len() != 2 {
            return Err(PackedModelError::Encode(format!(
                "{name}: embedding lookup requires rank-2 tensor"
            )));
        }
        let vocab = tensor.shape[0];
        if token_id >= vocab {
            return Err(PackedModelError::Encode(format!(
                "{name}: token_id {token_id} out of range for vocab {vocab}"
            )));
        }
        Ok(Some(
            embedding_lookup_packed_ternary(&tensor, token_id)
                .map_err(|error| PackedModelError::Encode(format!("{name}: {error}")))?,
        ))
    }

    pub fn matvec_f32(
        &self,
        name: &str,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, PackedModelError> {
        let Some(packed) = self.load_packed_tensor_file(name)? else {
            return Ok(None);
        };
        let tensor = packed.tensor;
        if tensor.shape.len() != 2 {
            return Err(PackedModelError::Encode(format!(
                "{name}: matvec requires rank-2 tensor"
            )));
        }
        let cols = tensor.shape[1];
        if input.len() != cols {
            return Err(PackedModelError::Encode(format!(
                "{name}: input length {} does not match cols {cols}",
                input.len()
            )));
        }
        Ok(Some(matvec_packed_ternary(&tensor, input).map_err(
            |error| PackedModelError::Encode(format!("{name}: {error}")),
        )?))
    }

    pub fn packed_total_bytes(&self) -> usize {
        self.manifest
            .entries
            .iter()
            .map(|entry| entry.total_file_bytes as usize)
            .sum()
    }

    pub fn unpacked_cache_bytes(&self) -> usize {
        let unpacked = self
            .unpacked_cache
            .borrow()
            .values()
            .map(|values| values.len() * std::mem::size_of::<f32>())
            .sum::<usize>();
        let embedding_rows = self
            .embedding_row_cache
            .borrow()
            .values()
            .map(|values| values.len() * std::mem::size_of::<f32>())
            .sum::<usize>();
        unpacked + embedding_rows
    }
}

impl PackedModelBenchReport {
    pub fn summarize(&self) -> String {
        let reduction = if self.packed_total_bytes == 0 {
            0.0
        } else {
            self.source_file_bytes as f64 / self.packed_total_bytes as f64
        };
        format!(
            "manifest_load_ms={:.3} entry_count={} packed_total_bytes={} source_file_bytes={} reduction_x={:.3}",
            self.manifest_load_duration.as_secs_f64() * 1_000.0,
            self.entry_count,
            self.packed_total_bytes,
            self.source_file_bytes,
            reduction,
        )
    }
}

#[derive(Debug)]
pub enum PackedModelError {
    Asset(AssetError),
    Weight(WeightError),
    Io(std::io::Error),
    Json(serde_json::Error),
    PackedIo(PackedIoError),
    Encode(String),
}

impl std::fmt::Display for PackedModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Asset(error) => write!(f, "asset error: {error}"),
            Self::Weight(error) => write!(f, "weight error: {error}"),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
            Self::PackedIo(error) => write!(f, "packed I/O error: {error}"),
            Self::Encode(message) => write!(f, "encode error: {message}"),
        }
    }
}

impl std::error::Error for PackedModelError {}
impl From<AssetError> for PackedModelError {
    fn from(value: AssetError) -> Self {
        Self::Asset(value)
    }
}
impl From<WeightError> for PackedModelError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}
impl From<std::io::Error> for PackedModelError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<serde_json::Error> for PackedModelError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
impl From<PackedIoError> for PackedModelError {
    fn from(value: PackedIoError) -> Self {
        Self::PackedIo(value)
    }
}

pub fn build_packable_tensor_specs(config: &BonsaiModelConfig) -> Vec<TensorPackSpec> {
    let mut specs = Vec::new();
    specs.push(TensorPackSpec {
        name: "model.embed_tokens.weight".to_string(),
        shape: vec![config.vocab_size, config.hidden_size],
    });
    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer_idx}");
        specs.push(TensorPackSpec {
            name: format!("{prefix}.self_attn.q_proj.weight"),
            shape: vec![config.hidden_size, config.hidden_size],
        });
        specs.push(TensorPackSpec {
            name: format!("{prefix}.self_attn.k_proj.weight"),
            shape: vec![
                config.num_key_value_heads * config.head_dim,
                config.hidden_size,
            ],
        });
        specs.push(TensorPackSpec {
            name: format!("{prefix}.self_attn.v_proj.weight"),
            shape: vec![
                config.num_key_value_heads * config.head_dim,
                config.hidden_size,
            ],
        });
        specs.push(TensorPackSpec {
            name: format!("{prefix}.self_attn.o_proj.weight"),
            shape: vec![config.hidden_size, config.hidden_size],
        });
        specs.push(TensorPackSpec {
            name: format!("{prefix}.mlp.gate_proj.weight"),
            shape: vec![config.intermediate_size, config.hidden_size],
        });
        specs.push(TensorPackSpec {
            name: format!("{prefix}.mlp.up_proj.weight"),
            shape: vec![config.intermediate_size, config.hidden_size],
        });
        specs.push(TensorPackSpec {
            name: format!("{prefix}.mlp.down_proj.weight"),
            shape: vec![config.hidden_size, config.intermediate_size],
        });
    }
    specs
}

pub fn build_artifact_layout(base_dir: impl AsRef<Path>) -> (PathBuf, PathBuf) {
    let root = base_dir.as_ref().to_path_buf();
    let tensors_dir = root.join("tensors");
    let manifest_path = root.join("manifest.json");
    (tensors_dir, manifest_path)
}

pub fn write_packed_model_artifact(
    model_root: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
) -> Result<(PackedModelManifest, PackedModelSummary), PackedModelError> {
    let assets = BonsaiAssetPaths::from_root(model_root.as_ref())?;
    let config =
        serde_json::from_str::<BonsaiModelConfig>(&fs::read_to_string(&assets.config_json)?)?;
    let store = WeightStore::load_from_assets(&assets)?;
    let progress = WeightStore::download_progress(&assets.safetensors_path)?;
    let (tensors_dir, manifest_path) = build_artifact_layout(out_dir);
    fs::create_dir_all(&tensors_dir)?;

    let specs = build_packable_tensor_specs(&config);
    let mut entries = Vec::with_capacity(specs.len());
    let mut packed_total_bytes = 0u64;

    for spec in specs {
        let values = store.load_vector_f32(&spec.name)?;
        let (tensor, _) = pack_ternary_g128(&values, spec.shape.clone(), PACK_TOLERANCE)
            .map_err(|error| PackedModelError::Encode(format!("{}: {error}", spec.name)))?;
        let rel_path = format!("tensors/{}.jtpk", sanitize_name(&spec.name));
        let out_path = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&rel_path);
        let packed = PackedTensorFile::new(Some(spec.name.clone()), tensor);
        packed.write_to_path(&out_path)?;
        let total_file_bytes = fs::metadata(&out_path)?.len();
        packed_total_bytes += total_file_bytes;
        entries.push(PackedModelTensorEntry {
            name: spec.name,
            rel_path,
            shape: packed.metadata.shape.clone(),
            group_size: packed.metadata.group_size,
            original_len: packed.metadata.original_len,
            code_bytes: packed.metadata.code_bytes,
            scale_count: packed.metadata.scale_count,
            total_file_bytes,
        });
    }

    let manifest = PackedModelManifest {
        format: "jengine-packed-model".to_string(),
        version: MANIFEST_VERSION,
        created_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        source_model_root: assets.root.display().to_string(),
        source_safetensors_path: assets.safetensors_path.display().to_string(),
        source_file_bytes: progress.file_bytes,
        source_tensor_count: progress.total_tensors,
        architecture: config
            .architectures
            .first()
            .cloned()
            .unwrap_or_else(|| config.model_type.clone()),
        hidden_size: config.hidden_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        pack_tolerance: PACK_TOLERANCE,
        entries,
    };
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    let summary = PackedModelSummary {
        entry_count: manifest.entries.len(),
        packed_total_bytes,
        source_file_bytes: manifest.source_file_bytes,
    };
    Ok((manifest, summary))
}

pub fn load_packed_model_manifest(
    path: impl AsRef<Path>,
) -> Result<PackedModelManifest, PackedModelError> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

pub fn validate_packed_model_artifact(
    model_root: impl AsRef<Path>,
    artifact_dir: impl AsRef<Path>,
) -> Result<PackedModelValidationReport, PackedModelError> {
    let assets = BonsaiAssetPaths::from_root(model_root.as_ref())?;
    let store = WeightStore::load_from_assets(&assets)?;
    let manifest = load_packed_model_manifest(artifact_dir.as_ref().join("manifest.json"))?;

    let mut checked_entries = 0usize;
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;
    let mut compared_values = 0usize;

    for entry in &manifest.entries {
        let packed = PackedTensorFile::read_from_path(artifact_dir.as_ref().join(&entry.rel_path))?;
        let unpacked = unpack_ternary_g128(&packed.tensor)
            .map_err(|error| PackedModelError::Encode(format!("{}: {error}", entry.name)))?;
        let source = store.load_vector_f32(&entry.name)?;
        if unpacked.len() != source.len() {
            return Err(PackedModelError::Encode(format!(
                "{}: unpacked length {} does not match source {}",
                entry.name,
                unpacked.len(),
                source.len()
            )));
        }
        checked_entries += 1;
        for (left, right) in unpacked.iter().zip(source.iter()) {
            let diff = (left - right).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff;
            compared_values += 1;
        }
    }

    let mean_abs_diff = if compared_values == 0 {
        0.0
    } else {
        sum_abs_diff / compared_values as f32
    };
    Ok(PackedModelValidationReport {
        checked_entries,
        max_abs_diff,
        mean_abs_diff,
    })
}

pub fn benchmark_packed_model_artifact(
    artifact_dir: impl AsRef<Path>,
) -> Result<PackedModelBenchReport, PackedModelError> {
    let manifest_path = artifact_dir.as_ref().join("manifest.json");
    let started_at = std::time::Instant::now();
    let manifest = load_packed_model_manifest(&manifest_path)?;
    let manifest_load_duration = started_at.elapsed();
    let packed_total_bytes = manifest
        .entries
        .iter()
        .map(|entry| entry.total_file_bytes)
        .sum();
    Ok(PackedModelBenchReport {
        manifest_load_duration,
        packed_total_bytes,
        source_file_bytes: manifest.source_file_bytes,
        entry_count: manifest.entries.len(),
    })
}

pub fn validate_packed_model_manifest(
    manifest: &PackedModelManifest,
    config: &BonsaiModelConfig,
) -> Result<(), PackedModelError> {
    if manifest.version != MANIFEST_VERSION {
        return Err(PackedModelError::Encode(format!(
            "manifest version {} does not match expected {}",
            manifest.version, MANIFEST_VERSION
        )));
    }
    if manifest.hidden_size != config.hidden_size
        || manifest.num_hidden_layers != config.num_hidden_layers
        || manifest.num_attention_heads != config.num_attention_heads
        || manifest.num_key_value_heads != config.num_key_value_heads
        || manifest.intermediate_size != config.intermediate_size
        || manifest.vocab_size != config.vocab_size
    {
        return Err(PackedModelError::Encode(
            "manifest geometry does not match config".to_string(),
        ));
    }
    Ok(())
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        PackedModelStore, benchmark_packed_model_artifact, build_artifact_layout,
        build_packable_tensor_specs, load_packed_model_manifest, validate_packed_model_artifact,
        validate_packed_model_manifest, write_packed_model_artifact,
    };
    use crate::model::config::BonsaiModelConfig;
    use std::fs;
    use tempfile::tempdir;

    const MODEL_CONFIG_JSON: &str = include_str!("../../fixtures/bonsai_1_7b_config.json");

    #[test]
    fn builds_expected_packable_tensor_spec_count() {
        let config =
            BonsaiModelConfig::from_json_str(MODEL_CONFIG_JSON).expect("config should parse");
        let specs = build_packable_tensor_specs(&config);
        assert_eq!(specs.len(), 1 + config.num_hidden_layers * 7);
        assert!(
            specs
                .iter()
                .any(|spec| spec.name == "model.embed_tokens.weight")
        );
    }

    #[test]
    fn resolves_artifact_layout() {
        let dir = tempdir().expect("tempdir should be created");
        let (tensors_dir, manifest_path) = build_artifact_layout(dir.path());
        assert!(tensors_dir.ends_with("tensors"));
        assert!(manifest_path.ends_with("manifest.json"));
    }

    #[test]
    fn writes_and_loads_manifest_for_synthetic_model() {
        let root = tempdir().expect("tempdir should be created");
        fs::write(root.path().join("config.json"), r#"{
            "vocab_size": 4,
            "max_position_embeddings": 32,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "hidden_act": "silu",
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "rope_scaling": {"rope_type": "yarn", "factor": 1.0, "original_max_position_embeddings": 32},
            "attention_bias": false,
            "tie_word_embeddings": true,
            "architectures": ["Qwen3ForCausalLM"],
            "pad_token_id": 0,
            "eos_token_id": 3,
            "model_type": "qwen3"
        }"#).unwrap();
        fs::write(root.path().join("generation_config.json"), "{}").unwrap();
        fs::write(root.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(root.path().join("tokenizer_config.json"), "{}").unwrap();

        let mut header = serde_json::Map::new();
        header.insert(
            "model.embed_tokens.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[4,4],"data_offsets":[0,32]}),
        );
        header.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[4,4],"data_offsets":[32,64]}),
        );
        header.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[2,4],"data_offsets":[64,80]}),
        );
        header.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[2,4],"data_offsets":[80,96]}),
        );
        header.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[4,4],"data_offsets":[96,128]}),
        );
        header.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[8,4],"data_offsets":[128,192]}),
        );
        header.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[8,4],"data_offsets":[192,256]}),
        );
        header.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[4,8],"data_offsets":[256,320]}),
        );
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut safetensors = Vec::new();
        safetensors.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        safetensors.extend_from_slice(&header_bytes);
        safetensors.resize(8 + header_bytes.len() + 320, 0);
        fs::write(root.path().join("model.safetensors"), safetensors).unwrap();

        let out = tempdir().expect("tempdir should be created");
        let (manifest, summary) = write_packed_model_artifact(root.path(), out.path()).unwrap();
        let loaded = load_packed_model_manifest(out.path().join("manifest.json")).unwrap();
        let config = BonsaiModelConfig::from_json_str(
            &fs::read_to_string(root.path().join("config.json")).unwrap(),
        )
        .unwrap();
        validate_packed_model_manifest(&loaded, &config).unwrap();
        let validation = validate_packed_model_artifact(root.path(), out.path()).unwrap();
        let bench = benchmark_packed_model_artifact(out.path()).unwrap();
        assert_eq!(manifest.entries.len(), 8);
        assert_eq!(summary.entry_count, 8);
        assert_eq!(validation.checked_entries, 8);
        assert_eq!(validation.max_abs_diff, 0.0);
        assert_eq!(bench.entry_count, 8);
    }

    #[test]
    fn loads_packed_model_store_and_reuses_cached_unpack() {
        let root = tempdir().expect("tempdir should be created");
        fs::write(root.path().join("config.json"), r#"{
            "vocab_size": 4,
            "max_position_embeddings": 32,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "hidden_act": "silu",
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "rope_scaling": {"rope_type": "yarn", "factor": 1.0, "original_max_position_embeddings": 32},
            "attention_bias": false,
            "tie_word_embeddings": true,
            "architectures": ["Qwen3ForCausalLM"],
            "pad_token_id": 0,
            "eos_token_id": 3,
            "model_type": "qwen3"
        }"#).unwrap();
        fs::write(root.path().join("generation_config.json"), "{}").unwrap();
        fs::write(root.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(root.path().join("tokenizer_config.json"), "{}").unwrap();

        let mut header = serde_json::Map::new();
        header.insert(
            "model.embed_tokens.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[4,4],"data_offsets":[0,32]}),
        );
        header.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[4,4],"data_offsets":[32,64]}),
        );
        header.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[2,4],"data_offsets":[64,80]}),
        );
        header.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[2,4],"data_offsets":[80,96]}),
        );
        header.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[4,4],"data_offsets":[96,128]}),
        );
        header.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[8,4],"data_offsets":[128,192]}),
        );
        header.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[8,4],"data_offsets":[192,256]}),
        );
        header.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            serde_json::json!({"dtype":"F16","shape":[4,8],"data_offsets":[256,320]}),
        );
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut safetensors = Vec::new();
        safetensors.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        safetensors.extend_from_slice(&header_bytes);
        safetensors.resize(8 + header_bytes.len() + 320, 0);
        fs::write(root.path().join("model.safetensors"), safetensors).unwrap();

        let out = tempdir().expect("tempdir should be created");
        let (_manifest, _summary) = write_packed_model_artifact(root.path(), out.path()).unwrap();
        let config = BonsaiModelConfig::from_json_str(
            &fs::read_to_string(root.path().join("config.json")).unwrap(),
        )
        .unwrap();
        let store = PackedModelStore::load_from_artifact_dir(out.path(), &config).unwrap();
        assert!(store.has_tensor("model.embed_tokens.weight"));
        let first = store
            .load_vector_f32("model.embed_tokens.weight")
            .unwrap()
            .unwrap();
        let second = store
            .load_vector_f32("model.embed_tokens.weight")
            .unwrap()
            .unwrap();
        assert_eq!(first, second);
        assert!(store.unpacked_cache_bytes() > 0);
        let embed = store
            .embedding_lookup("model.embed_tokens.weight", 0)
            .unwrap()
            .unwrap();
        assert_eq!(embed.len(), 4);
        let after_first_embed_cache_bytes = store.unpacked_cache_bytes();
        let embed_again = store
            .embedding_lookup("model.embed_tokens.weight", 0)
            .unwrap()
            .unwrap();
        assert_eq!(embed_again, embed);
        assert_eq!(store.unpacked_cache_bytes(), after_first_embed_cache_bytes);
        let before_matvec_cache_bytes = store.unpacked_cache_bytes();
        let matvec = store
            .matvec_f32(
                "model.layers.0.self_attn.q_proj.weight",
                &[0.0, 0.0, 0.0, 0.0],
            )
            .unwrap()
            .unwrap();
        assert_eq!(matvec.len(), 4);
        assert_eq!(store.unpacked_cache_bytes(), before_matvec_cache_bytes);
    }
}
