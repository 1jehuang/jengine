use crate::runtime::assets::AssetError;
use crate::runtime::packed_model::PackedModelError;
use crate::runtime::weights::WeightError;

#[derive(Debug)]
pub enum ReferenceError {
    Asset(AssetError),
    Io(std::io::Error),
    Json(serde_json::Error),
    Tokenizer(crate::model::tokenizer::TokenizerLoadError),
    Weight(WeightError),
    PackedModel(PackedModelError),
    Decode(String),
}

impl std::fmt::Display for ReferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Asset(error) => write!(f, "asset error: {error}"),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
            Self::Tokenizer(error) => write!(f, "tokenizer error: {error}"),
            Self::Weight(error) => write!(f, "weight error: {error}"),
            Self::PackedModel(error) => write!(f, "packed model error: {error}"),
            Self::Decode(message) => write!(f, "decode error: {message}"),
        }
    }
}

impl std::error::Error for ReferenceError {}

impl From<AssetError> for ReferenceError {
    fn from(value: AssetError) -> Self {
        Self::Asset(value)
    }
}
impl From<std::io::Error> for ReferenceError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<serde_json::Error> for ReferenceError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
impl From<crate::model::tokenizer::TokenizerLoadError> for ReferenceError {
    fn from(value: crate::model::tokenizer::TokenizerLoadError) -> Self {
        Self::Tokenizer(value)
    }
}
impl From<WeightError> for ReferenceError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}
impl From<PackedModelError> for ReferenceError {
    fn from(value: PackedModelError) -> Self {
        Self::PackedModel(value)
    }
}
