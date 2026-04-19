use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokenizers::decoders;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers;
use tokenizers::pre_tokenizers;
use tokenizers::tokenizer::SplitDelimiterBehavior;
use tokenizers::{AddedToken, Tokenizer};

const QWEN2_PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug)]
pub enum TokenizerLoadError {
    Tokenizers(tokenizers::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for TokenizerLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokenizers(error) => write!(f, "tokenizer error: {error}"),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
        }
    }
}

impl std::error::Error for TokenizerLoadError {}

impl From<tokenizers::Error> for TokenizerLoadError {
    fn from(value: tokenizers::Error) -> Self {
        Self::Tokenizers(value)
    }
}

impl From<std::io::Error> for TokenizerLoadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for TokenizerLoadError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
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

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct AddedTokenConfig {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct QwenFallbackTokenizerConfig {
    add_prefix_space: bool,
    added_tokens_decoder: BTreeMap<String, AddedTokenConfig>,
}

impl TokenizerRuntime {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerLoadError> {
        let path = path.as_ref();
        let tokenizer = match Tokenizer::from_file(path) {
            Ok(tokenizer) => tokenizer,
            Err(_) => Self::load_qwen_fallback_from_tokenizer_json_path(path)?,
        };

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

    fn load_qwen_fallback_from_tokenizer_json_path(
        path: &Path,
    ) -> Result<Tokenizer, TokenizerLoadError> {
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let vocab_path = root.join("vocab.json");
        let merges_path = root.join("merges.txt");
        let tokenizer_config_path = root.join("tokenizer_config.json");

        let model = BPE::from_file(
            vocab_path.to_string_lossy().as_ref(),
            merges_path.to_string_lossy().as_ref(),
        )
        .continuing_subword_prefix("".to_string())
        .end_of_word_suffix("".to_string())
        .fuse_unk(false)
        .byte_fallback(false)
        .build()?;

        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(decoders::byte_level::ByteLevel::default());
        tokenizer.with_normalizer(normalizers::unicode::NFC);
        tokenizer.with_pre_tokenizer(pre_tokenizers::sequence::Sequence::new(vec![
            pre_tokenizers::split::Split::new(
                pre_tokenizers::split::SplitPattern::Regex(QWEN2_PRETOKENIZE_REGEX.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )?
            .into(),
            pre_tokenizers::byte_level::ByteLevel::new(false, true, false).into(),
        ]));

        let config = serde_json::from_str::<QwenFallbackTokenizerConfig>(&fs::read_to_string(
            tokenizer_config_path,
        )?)?;
        let mut added = Vec::new();
        for token in config.added_tokens_decoder.into_values() {
            let added_token = AddedToken::from(token.content, token.special)
                .lstrip(token.lstrip)
                .rstrip(token.rstrip)
                .single_word(token.single_word)
                .normalized(token.normalized)
                .special(token.special);
            added.push(added_token);
        }
        let _ = tokenizer.add_special_tokens(&added);
        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::TokenizerRuntime;
    use std::collections::HashMap;
    use std::fs;
    use tempfile::{NamedTempFile, tempdir};
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

    fn write_qwen_fallback_fixture() -> std::path::PathBuf {
        let dir = tempdir().expect("tempdir should be created");
        let root = dir.keep();
        fs::write(root.join("tokenizer.json"), "{ not-valid-json }")
            .expect("invalid tokenizer should be written");
        fs::write(
            root.join("vocab.json"),
            r#"{"<|endoftext|>":0,"<|im_end|>":1,"h":2,"e":3,"l":4,"o":5,"Ġ":6,"w":7,"r":8,"d":9,"he":10,"hel":11,"hell":12,"hello":13,"Ġw":14,"Ġwo":15,"Ġwor":16,"Ġworl":17,"Ġworld":18}"#,
        )
        .expect("vocab should be written");
        fs::write(
            root.join("merges.txt"),
            "#version: 0.2\nh e\nhe l\nhel l\nhell o\nĠ w\nĠw o\nĠwo r\nĠwor l\nĠworl d\n",
        )
        .expect("merges should be written");
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{"add_prefix_space":false,"added_tokens_decoder":{"0":{"content":"<|endoftext|>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true},"3":{"content":"<|im_end|>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true}}}"#,
        )
        .expect("tokenizer config should be written");
        root
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
    fn falls_back_to_qwen_bpe_assets_when_tokenizer_json_is_not_supported() {
        let root = write_qwen_fallback_fixture();
        let runtime = TokenizerRuntime::load_from_file(root.join("tokenizer.json"))
            .expect("fallback tokenizer should load");
        let ids = runtime
            .encode("hello world")
            .expect("fallback encode should succeed");
        let decoded = runtime
            .decode(&ids)
            .expect("fallback decode should succeed");

        assert_eq!(ids, vec![13, 18]);
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
