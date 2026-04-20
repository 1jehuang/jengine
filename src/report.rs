use crate::model::tokenizer::PromptAnalysis;
use crate::runtime::reference::{DecodeMetrics, MemoryReport};
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn profile_log_dir() -> PathBuf {
    std::env::var_os("JENGINE_PROFILE_LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".artifacts/profiles"))
}

pub fn ensure_profile_log_dir() -> Result<PathBuf, std::io::Error> {
    let dir = profile_log_dir();
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn timestamped_profile_path(prefix: &str, extension: &str) -> Result<PathBuf, std::io::Error> {
    let dir = ensure_profile_log_dir()?;
    let unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    Ok(dir.join(format!("{prefix}_{unix_ms}.{extension}")))
}

pub fn write_json_value(path: impl AsRef<Path>, value: &Value) -> Result<(), std::io::Error> {
    let serialized = serde_json::to_string_pretty(value).expect("JSON should serialize");
    fs::write(path, serialized)
}

pub fn append_jsonl_record(file_name: &str, value: &Value) -> Result<PathBuf, std::io::Error> {
    let path = ensure_profile_log_dir()?.join(file_name);
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    serde_json::to_writer(&mut file, value).expect("JSONL should serialize");
    writeln!(file)?;
    Ok(path)
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchIterationRecord {
    pub iteration: usize,
    pub metrics: DecodeMetrics,
    pub output_text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchReport {
    pub model_root: String,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub iterations: Vec<BenchIterationRecord>,
    pub prompt_analysis: Option<PromptAnalysis>,
    pub memory_report: Option<MemoryReport>,
}

impl BenchReport {
    pub fn average_total_ms(&self) -> f64 {
        self.average_duration_ms(|metrics| metrics.total_duration.as_secs_f64() * 1_000.0)
    }

    pub fn average_qkv_ms(&self) -> f64 {
        self.average_duration_ms(|metrics| metrics.qkv_duration.as_secs_f64() * 1_000.0)
    }

    pub fn average_attention_ms(&self) -> f64 {
        self.average_duration_ms(|metrics| metrics.attention_duration.as_secs_f64() * 1_000.0)
    }

    pub fn average_mlp_ms(&self) -> f64 {
        self.average_duration_ms(|metrics| metrics.mlp_duration.as_secs_f64() * 1_000.0)
    }

    pub fn average_logits_ms(&self) -> f64 {
        self.average_duration_ms(|metrics| metrics.logits_duration.as_secs_f64() * 1_000.0)
    }

    pub fn average_generated_tokens_per_second(&self) -> f64 {
        self.average_f64(DecodeMetrics::generated_tokens_per_second)
    }

    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Jengine benchmark report\n\n");
        out.push_str("## Run configuration\n\n");
        out.push_str(&format!("- Model root: `{}`\n", self.model_root));
        out.push_str(&format!(
            "- Prompt: `{}`\n",
            self.prompt.replace('`', "\\`")
        ));
        out.push_str(&format!("- Max new tokens: {}\n", self.max_new_tokens));
        out.push_str(&format!("- Iterations: {}\n", self.iterations.len()));
        if let Some(prompt_analysis) = &self.prompt_analysis {
            out.push_str(&format!(
                "- Prompt analysis: {}\n",
                prompt_analysis.summarize()
            ));
        }
        if let Some(memory_report) = &self.memory_report {
            out.push_str(&format!("- Memory report: {}\n", memory_report.summarize()));
        }
        out.push_str("\n## Averages\n\n");
        out.push_str("| metric | value |\n");
        out.push_str("| --- | ---: |\n");
        out.push_str(&format!(
            "| avg_total_ms | {:.3} |\n",
            self.average_total_ms()
        ));
        out.push_str(&format!("| avg_qkv_ms | {:.3} |\n", self.average_qkv_ms()));
        out.push_str(&format!(
            "| avg_attention_ms | {:.3} |\n",
            self.average_attention_ms()
        ));
        out.push_str(&format!("| avg_mlp_ms | {:.3} |\n", self.average_mlp_ms()));
        out.push_str(&format!(
            "| avg_logits_ms | {:.3} |\n",
            self.average_logits_ms()
        ));
        out.push_str(&format!(
            "| avg_generated_tok_s | {:.3} |\n",
            self.average_generated_tokens_per_second()
        ));
        out.push_str("\n## Iterations\n\n");
        out.push_str("| iteration | total_ms | qkv_ms | attention_ms | mlp_ms | logits_ms | tok_s | output |\n");
        out.push_str("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n");
        for run in &self.iterations {
            out.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | `{}` |\n",
                run.iteration,
                run.metrics.total_duration.as_secs_f64() * 1_000.0,
                run.metrics.qkv_duration.as_secs_f64() * 1_000.0,
                run.metrics.attention_duration.as_secs_f64() * 1_000.0,
                run.metrics.mlp_duration.as_secs_f64() * 1_000.0,
                run.metrics.logits_duration.as_secs_f64() * 1_000.0,
                run.metrics.generated_tokens_per_second(),
                run.output_text.replace('`', "\\`")
            ));
        }
        out
    }

    pub fn to_key_value(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("model_root={}\n", self.model_root));
        out.push_str(&format!("prompt={}\n", self.prompt.replace('\n', "\\n")));
        out.push_str(&format!("max_new_tokens={}\n", self.max_new_tokens));
        out.push_str(&format!("iterations={}\n", self.iterations.len()));
        if let Some(prompt_analysis) = &self.prompt_analysis {
            out.push_str(&format!(
                "prompt_analysis={}\n",
                prompt_analysis.summarize()
            ));
        }
        if let Some(memory_report) = &self.memory_report {
            out.push_str(&format!("memory_report={}\n", memory_report.summarize()));
        }
        out.push_str(&format!("avg_total_ms={:.3}\n", self.average_total_ms()));
        out.push_str(&format!("avg_qkv_ms={:.3}\n", self.average_qkv_ms()));
        out.push_str(&format!(
            "avg_attention_ms={:.3}\n",
            self.average_attention_ms()
        ));
        out.push_str(&format!("avg_mlp_ms={:.3}\n", self.average_mlp_ms()));
        out.push_str(&format!("avg_logits_ms={:.3}\n", self.average_logits_ms()));
        out.push_str(&format!(
            "avg_generated_tok_s={:.3}\n",
            self.average_generated_tokens_per_second()
        ));
        for run in &self.iterations {
            out.push_str(&format!(
                "iteration={} total_ms={:.3} qkv_ms={:.3} attention_ms={:.3} mlp_ms={:.3} logits_ms={:.3} tok_s={:.3} output={}\n",
                run.iteration,
                run.metrics.total_duration.as_secs_f64() * 1_000.0,
                run.metrics.qkv_duration.as_secs_f64() * 1_000.0,
                run.metrics.attention_duration.as_secs_f64() * 1_000.0,
                run.metrics.mlp_duration.as_secs_f64() * 1_000.0,
                run.metrics.logits_duration.as_secs_f64() * 1_000.0,
                run.metrics.generated_tokens_per_second(),
                run.output_text.replace('\n', "\\n")
            ));
        }
        out
    }

    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str("iteration,total_ms,qkv_ms,attention_ms,mlp_ms,logits_ms,tok_s,output\n");
        for run in &self.iterations {
            out.push_str(&format!(
                "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}\n",
                run.iteration,
                run.metrics.total_duration.as_secs_f64() * 1_000.0,
                run.metrics.qkv_duration.as_secs_f64() * 1_000.0,
                run.metrics.attention_duration.as_secs_f64() * 1_000.0,
                run.metrics.mlp_duration.as_secs_f64() * 1_000.0,
                run.metrics.logits_duration.as_secs_f64() * 1_000.0,
                run.metrics.generated_tokens_per_second(),
                csv_escape(&run.output_text)
            ));
        }
        out
    }

    pub fn write_markdown_to_path(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.to_markdown())
    }

    pub fn write_key_value_to_path(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.to_key_value())
    }

    pub fn write_csv_to_path(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.to_csv())
    }

    fn average_duration_ms(&self, map: impl Fn(&DecodeMetrics) -> f64) -> f64 {
        self.average_f64(map)
    }

    fn average_f64(&self, map: impl Fn(&DecodeMetrics) -> f64) -> f64 {
        if self.iterations.is_empty() {
            return 0.0;
        }
        self.iterations
            .iter()
            .map(|run| map(&run.metrics))
            .sum::<f64>()
            / self.iterations.len() as f64
    }
}

fn csv_escape(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\"").replace('\n', "\\n"))
}

#[cfg(test)]
mod tests {
    use super::{BenchIterationRecord, BenchReport};
    use crate::runtime::reference::DecodeMetrics;
    use std::time::Duration;

    #[test]
    fn renders_markdown_report() {
        let report = BenchReport {
            model_root: "/tmp/model".to_string(),
            prompt: "hello".to_string(),
            max_new_tokens: 2,
            iterations: vec![BenchIterationRecord {
                iteration: 1,
                metrics: DecodeMetrics {
                    prompt_tokens: 1,
                    generated_tokens: 2,
                    total_duration: Duration::from_millis(2000),
                    embedding_duration: Duration::from_millis(1),
                    norm_duration: Duration::from_millis(2),
                    qkv_duration: Duration::from_millis(3),
                    attention_duration: Duration::from_millis(4),
                    mlp_duration: Duration::from_millis(5),
                    logits_duration: Duration::from_millis(6),
                },
                output_text: "hello world".to_string(),
            }],
            prompt_analysis: None,
            memory_report: None,
        };

        let markdown = report.to_markdown();
        assert!(markdown.contains("# Jengine benchmark report"));
        assert!(markdown.contains("avg_generated_tok_s"));
        assert!(markdown.contains("hello world"));
    }

    #[test]
    fn renders_key_value_and_csv_reports() {
        let report = BenchReport {
            model_root: "/tmp/model".to_string(),
            prompt: "hello".to_string(),
            max_new_tokens: 2,
            iterations: vec![BenchIterationRecord {
                iteration: 1,
                metrics: DecodeMetrics {
                    prompt_tokens: 1,
                    generated_tokens: 2,
                    total_duration: Duration::from_millis(2000),
                    embedding_duration: Duration::from_millis(1),
                    norm_duration: Duration::from_millis(2),
                    qkv_duration: Duration::from_millis(3),
                    attention_duration: Duration::from_millis(4),
                    mlp_duration: Duration::from_millis(5),
                    logits_duration: Duration::from_millis(6),
                },
                output_text: "hello world".to_string(),
            }],
            prompt_analysis: None,
            memory_report: None,
        };

        let key_value = report.to_key_value();
        let csv = report.to_csv();
        assert!(key_value.contains("avg_total_ms=2000.000"));
        assert!(csv.contains("iteration,total_ms"));
        assert!(csv.contains("\"hello world\""));
    }
}
