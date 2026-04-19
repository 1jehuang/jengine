use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BonsaiAssetPaths {
    pub root: PathBuf,
    pub config_json: PathBuf,
    pub generation_config_json: PathBuf,
    pub tokenizer_json: PathBuf,
    pub tokenizer_config_json: PathBuf,
    pub safetensors_path: PathBuf,
}

#[derive(Debug)]
pub enum AssetError {
    MissingFile(PathBuf),
}

impl std::fmt::Display for AssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFile(path) => write!(f, "missing required asset: {}", path.display()),
        }
    }
}

impl std::error::Error for AssetError {}

impl BonsaiAssetPaths {
    pub fn from_root(root: impl AsRef<Path>) -> Result<Self, AssetError> {
        let root = root.as_ref().to_path_buf();
        let paths = Self {
            config_json: root.join("config.json"),
            generation_config_json: root.join("generation_config.json"),
            tokenizer_json: root.join("tokenizer.json"),
            tokenizer_config_json: root.join("tokenizer_config.json"),
            safetensors_path: root.join("model.safetensors"),
            root,
        };
        paths.validate()?;
        Ok(paths)
    }

    pub fn validate(&self) -> Result<(), AssetError> {
        for path in [
            &self.config_json,
            &self.generation_config_json,
            &self.tokenizer_json,
            &self.tokenizer_config_json,
            &self.safetensors_path,
        ] {
            if !path.exists() {
                return Err(AssetError::MissingFile(path.clone()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::BonsaiAssetPaths;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn resolves_and_validates_expected_asset_paths() {
        let dir = tempdir().expect("tempdir should be created");
        for file in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ] {
            fs::write(dir.path().join(file), "{}").expect("fixture file should be written");
        }

        let paths = BonsaiAssetPaths::from_root(dir.path()).expect("asset paths should validate");
        assert!(paths.safetensors_path.ends_with("model.safetensors"));
    }
}
