use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

#[derive(Debug)]
pub enum TokenizerLoadError {
    Tokenizers(tokenizers::Error),
}

impl std::fmt::Display for TokenizerLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokenizers(error) => write!(f, "tokenizer error: {error}"),
        }
    }
}

impl std::error::Error for TokenizerLoadError {}

impl From<tokenizers::Error> for TokenizerLoadError {
    fn from(value: tokenizers::Error) -> Self {
        Self::Tokenizers(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptAnalysis {
    pub prompt_bytes: usize,
    pub prompt_chars: usize,
    pub token_count: usize,
    pub encode_duration: Duration,
}

impl PromptAnalysis {
    pub fn summarize(&self) -> String {
        format!(
            "prompt_bytes={} prompt_chars={} tokens={} encode_ms={:.3}",
            self.prompt_bytes,
            self.prompt_chars,
            self.token_count,
            self.encode_duration.as_secs_f64() * 1_000.0,
        )
    }
}

pub struct TokenizerRuntime {
    source_path: PathBuf,
    tokenizer: Tokenizer,
}

impl TokenizerRuntime {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerLoadError> {
        let path = path.as_ref();
        let tokenizer = Tokenizer::from_file(path)?;

        Ok(Self {
            source_path: path.to_path_buf(),
            tokenizer,
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, TokenizerLoadError> {
        let encoding = self.tokenizer.encode(prompt, false)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerLoadError> {
        Ok(self.tokenizer.decode(token_ids, false)?)
    }

    pub fn analyze_prompt(&self, prompt: &str) -> Result<PromptAnalysis, TokenizerLoadError> {
        let started_at = Instant::now();
        let encoding = self.tokenizer.encode(prompt, false)?;

        Ok(PromptAnalysis {
            prompt_bytes: prompt.len(),
            prompt_chars: prompt.chars().count(),
            token_count: encoding.len(),
            encode_duration: started_at.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::TokenizerRuntime;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn write_test_tokenizer() -> NamedTempFile {
        let vocab = HashMap::from([
            ("[UNK]".to_string(), 0),
            ("hello".to_string(), 1),
            ("world".to_string(), 2),
        ]);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("word-level model should build");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Whitespace);

        let temp = NamedTempFile::new().expect("temp file should be created");
        tokenizer
            .save(temp.path(), false)
            .expect("tokenizer should be written");
        temp
    }

    #[test]
    fn loads_tokenizer_and_roundtrips_prompt() {
        let temp = write_test_tokenizer();
        let runtime = TokenizerRuntime::load_from_file(temp.path()).expect("tokenizer should load");
        let ids = runtime
            .encode("hello world")
            .expect("encode should succeed");
        let decoded = runtime.decode(&ids).expect("decode should succeed");

        assert_eq!(runtime.vocab_size(), 3);
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn emits_prompt_analysis_summary() {
        let temp = write_test_tokenizer();
        let runtime = TokenizerRuntime::load_from_file(temp.path()).expect("tokenizer should load");
        let analysis = runtime
            .analyze_prompt("hello world")
            .expect("analysis should succeed");
        let summary = analysis.summarize();

        assert_eq!(analysis.prompt_bytes, 11);
        assert_eq!(analysis.prompt_chars, 11);
        assert_eq!(analysis.token_count, 2);
        assert!(summary.contains("tokens=2"));
        assert!(summary.contains("encode_ms="));
    }
}
