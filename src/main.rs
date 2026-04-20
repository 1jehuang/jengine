use jengine::gpu::vulkan::{collect_vulkan_report, version_string};
use jengine::report::{BenchIterationRecord, BenchReport};
use jengine::runtime::packed::PackedTensorFile;
use jengine::runtime::reference::{DecodeMetrics, MemoryReport, ReferenceModel};
use jengine::runtime::repack::{analyze_ternary_packability, pack_ternary_g128};
use jengine::runtime::weights::WeightStore;

const DEFAULT_ROOT: &str = "/home/jeremy/models/bonsai-1.7b";
const DEFAULT_WEIGHTS: &str = "/home/jeremy/models/bonsai-1.7b/model.safetensors";
const DEFAULT_TENSOR: &str = "model.layers.0.self_attn.q_proj.weight";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Plain,
    KeyValue,
    Markdown,
}

#[derive(Debug, Clone, Default)]
struct BenchExportPaths {
    markdown: Option<String>,
    key_value: Option<String>,
    csv: Option<String>,
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = run_cli(&args) {
        eprintln!("error: {error}");
        eprintln!();
        eprintln!("{}", usage());
        std::process::exit(1);
    }
}

fn run_cli(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("help") | Some("--help") | Some("-h") => {
            println!("{}", usage());
            Ok(())
        }
        Some("inspect") => cmd_inspect(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("bench") => cmd_bench(&args[1..]),
        Some("profile") => cmd_profile(&args[1..]),
        Some("validate") => cmd_validate(&args[1..]),
        Some("pack") => cmd_pack(&args[1..]),
        Some(other) => Err(format!("unknown command: {other}")),
    }
}

fn cmd_inspect(args: &[String]) -> Result<(), String> {
    let (args, format) = parse_format_flag(args)?;
    let root = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_ROOT.to_string());
    let prompt = args.get(1).cloned();
    let max_new_tokens = parse_usize(args.get(2), 1, "max_new_tokens")?;

    let model = load_model(&root)?;
    let progress = WeightStore::download_progress(&model.assets.safetensors_path)
        .map_err(|error| error.to_string())?;
    let probe = model.weights.probe().map_err(|error| error.to_string())?;
    let inspection = model.config.inspect();

    let mut lines = vec![
        ("name".to_string(), jengine::name().to_string()),
        ("model_root".to_string(), root.clone()),
        (
            "architecture".to_string(),
            format!(
                "{} layers={} vocab_size={} estimated_params={} estimated_fp16_bytes={}",
                inspection.architecture,
                inspection.layers,
                inspection.vocab_size,
                inspection.estimated_parameter_count,
                inspection.estimated_fp16_bytes,
            ),
        ),
        ("weights".to_string(), progress.summarize()),
        ("probe".to_string(), probe.summarize()),
    ];
    if let Some(diagnostics) = model.tokenizer_diagnostics() {
        lines.push(("tokenizer".to_string(), diagnostics.summarize()));
    }

    if let Some(prompt) = prompt {
        let prompt_analysis = model
            .prompt_analysis(&prompt)
            .map_err(|error| error.to_string())?;
        let memory_report = model.memory_report(prompt_analysis.token_count, max_new_tokens);
        lines.push(("prompt_analysis".to_string(), prompt_analysis.summarize()));
        lines.push(("memory_report".to_string(), memory_report.summarize()));
    } else {
        lines.push((
            "memory_per_token_runtime_f32_bytes".to_string(),
            model
                .memory_report(0, 1)
                .kv_cache_bytes_per_token_runtime_f32
                .to_string(),
        ));
    }

    match collect_vulkan_report() {
        Ok(report) => {
            lines.push(("vulkan".to_string(), report.summarize()));
            for device in report.devices {
                lines.push((
                    "vulkan_device".to_string(),
                    format!(
                        "name={} type={} api={} queues={}",
                        device.name,
                        device.device_type,
                        version_string(device.api_version),
                        device.queue_families.len(),
                    ),
                ));
            }
        }
        Err(error) => lines.push(("vulkan_unavailable".to_string(), error.to_string())),
    }

    print_section(format, "Inspection", &lines);

    Ok(())
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let (args, format) = parse_format_flag(args)?;
    let root = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_ROOT.to_string());
    let prompt = args.get(1).cloned().unwrap_or_else(|| "hello".to_string());
    let max_new_tokens = parse_usize(args.get(2), 1, "max_new_tokens")?;

    let model = load_model(&root)?;
    let prompt_analysis = model
        .prompt_analysis(&prompt)
        .map_err(|error| error.to_string())?;
    let memory_report = model.memory_report(prompt_analysis.token_count, max_new_tokens);
    let result = model
        .generate_greedy(&prompt, max_new_tokens)
        .map_err(|error| error.to_string())?;

    print_run_output(
        format,
        &model,
        &prompt_analysis,
        &memory_report,
        &result.metrics,
        Some(&result.output_text),
        false,
    );
    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<(), String> {
    let (args, format, export_paths) = parse_bench_flags(args)?;
    let root = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_ROOT.to_string());
    let prompt = args.get(1).cloned().unwrap_or_else(|| "hello".to_string());
    let max_new_tokens = parse_usize(args.get(2), 3, "max_new_tokens")?;
    let iterations = parse_usize(args.get(3), 2, "iterations")?;

    let model = load_model(&root)?;
    let prompt_analysis = model
        .prompt_analysis(&prompt)
        .map_err(|error| error.to_string())?;
    let memory_report = model.memory_report(prompt_analysis.token_count, max_new_tokens);
    let mut records = Vec::with_capacity(iterations);

    print_run_header(format, &model, &prompt_analysis, &memory_report);
    for iteration in 0..iterations {
        let result = model
            .generate_greedy(&prompt, max_new_tokens)
            .map_err(|error| error.to_string())?;
        print_bench_iteration(format, iteration + 1, &result.metrics, &result.output_text);
        records.push(BenchIterationRecord {
            iteration: iteration + 1,
            metrics: result.metrics,
            output_text: result.output_text,
        });
    }

    let report = BenchReport {
        model_root: root,
        prompt,
        max_new_tokens,
        iterations: records,
        prompt_analysis: Some(prompt_analysis),
        memory_report: Some(memory_report),
    };
    print_bench_summary(format, &report);
    if let Some(path) = export_paths.markdown {
        report
            .write_markdown_to_path(&path)
            .map_err(|error| format!("failed to write markdown report to {path}: {error}"))?;
        println!("markdown_report={path}");
    }
    if let Some(path) = export_paths.key_value {
        report
            .write_key_value_to_path(&path)
            .map_err(|error| format!("failed to write key-value report to {path}: {error}"))?;
        println!("kv_report={path}");
    }
    if let Some(path) = export_paths.csv {
        report
            .write_csv_to_path(&path)
            .map_err(|error| format!("failed to write csv report to {path}: {error}"))?;
        println!("csv_report={path}");
    }
    Ok(())
}

fn cmd_profile(args: &[String]) -> Result<(), String> {
    let (args, format) = parse_format_flag(args)?;
    let root = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_ROOT.to_string());
    let prompt = args.get(1).cloned().unwrap_or_else(|| "hello".to_string());
    let max_new_tokens = parse_usize(args.get(2), 1, "max_new_tokens")?;

    let model = load_model(&root)?;
    let prompt_analysis = model
        .prompt_analysis(&prompt)
        .map_err(|error| error.to_string())?;
    let memory_report = model.memory_report(prompt_analysis.token_count, max_new_tokens);
    let result = model
        .generate_greedy(&prompt, max_new_tokens)
        .map_err(|error| error.to_string())?;

    print_run_output(
        format,
        &model,
        &prompt_analysis,
        &memory_report,
        &result.metrics,
        Some(&result.output_text),
        true,
    );
    Ok(())
}

fn cmd_validate(args: &[String]) -> Result<(), String> {
    let (args, format) = parse_format_flag(args)?;
    let root = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_ROOT.to_string());
    let prompt = args.get(1).cloned().unwrap_or_else(|| "hello".to_string());
    let max_new_tokens = parse_usize(args.get(2), 1, "max_new_tokens")?;

    let model = load_model(&root)?;
    let progress = WeightStore::download_progress(&model.assets.safetensors_path)
        .map_err(|error| error.to_string())?;
    let probe = model.weights.probe().map_err(|error| error.to_string())?;
    let prompt_analysis = model
        .prompt_analysis(&prompt)
        .map_err(|error| error.to_string())?;
    let result = model
        .generate_greedy(&prompt, max_new_tokens)
        .map_err(|error| error.to_string())?;

    let lines = vec![
        ("asset_validation".to_string(), format!("ok root={root}")),
        ("weights".to_string(), progress.summarize()),
        ("probe".to_string(), probe.summarize()),
        ("prompt_analysis".to_string(), prompt_analysis.summarize()),
        ("decode_metrics".to_string(), result.metrics.summarize()),
        (
            "validation".to_string(),
            format!("ok output={}", result.output_text),
        ),
    ];
    print_section(format, "Validation", &lines);
    Ok(())
}

fn cmd_pack(args: &[String]) -> Result<(), String> {
    let weights_path = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_WEIGHTS.to_string());
    let tensor_name = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_TENSOR.to_string());
    let rows = parse_usize(args.get(2), 2048, "rows")?;
    let cols = parse_usize(args.get(3), 2048, "cols")?;
    let out_path = args.get(4).cloned();

    let store = WeightStore::load_from_file(&weights_path).map_err(|error| error.to_string())?;
    let values = store
        .load_vector_f32(&tensor_name)
        .map_err(|error| error.to_string())?;
    let analysis = analyze_ternary_packability(&values);
    println!("analysis={}", analysis.summarize());
    let (packed, strict_report) = pack_ternary_g128(&values, vec![rows, cols], 1e-3)
        .map_err(|error| format!("strict pack failed: {error}"))?;
    println!("strict_pack={}", strict_report.summarize());
    println!(
        "packed_codes={} scale_count={} code_bytes={}",
        packed.packed_codes.len(),
        packed.scales.len(),
        packed.packed_codes.len(),
    );
    if let Some(out_path) = out_path {
        let packed_file = PackedTensorFile::new(Some(tensor_name), packed);
        packed_file
            .write_to_path(&out_path)
            .map_err(|error| format!("failed to write packed tensor: {error}"))?;
        println!("packed_file={out_path}");
    }
    Ok(())
}

fn load_model(root: &str) -> Result<ReferenceModel, String> {
    ReferenceModel::load_from_root(root).map_err(|error| error.to_string())
}

fn print_run_header(
    format: OutputFormat,
    model: &ReferenceModel,
    prompt_analysis: &jengine::model::tokenizer::PromptAnalysis,
    memory_report: &MemoryReport,
) {
    let mut lines = Vec::new();
    if let Some(diagnostics) = model.tokenizer_diagnostics() {
        lines.push(("tokenizer".to_string(), diagnostics.summarize()));
    }
    lines.push(("prompt_analysis".to_string(), prompt_analysis.summarize()));
    lines.push(("memory_report".to_string(), memory_report.summarize()));
    print_section(format, "Run", &lines);
}

fn print_run_output(
    format: OutputFormat,
    model: &ReferenceModel,
    prompt_analysis: &jengine::model::tokenizer::PromptAnalysis,
    memory_report: &MemoryReport,
    metrics: &DecodeMetrics,
    output_text: Option<&str>,
    include_phase_shares: bool,
) {
    print_run_header(format, model, prompt_analysis, memory_report);
    let mut lines = vec![
        ("metrics".to_string(), metrics.summarize()),
        (
            "generated_tok_s".to_string(),
            format!(
                "{:.3} total_sequence_tokens={}",
                metrics.generated_tokens_per_second(),
                metrics.total_sequence_tokens(),
            ),
        ),
    ];
    if include_phase_shares {
        lines.extend(phase_share_lines(metrics));
    }
    if let Some(output_text) = output_text {
        lines.push(("output".to_string(), output_text.to_string()));
    }
    print_section(format, "Decode", &lines);
}

fn phase_share_lines(metrics: &DecodeMetrics) -> Vec<(String, String)> {
    let total_ms = metrics.total_duration.as_secs_f64() * 1_000.0;
    if total_ms <= f64::EPSILON {
        return Vec::new();
    }
    let mut lines = Vec::new();
    for (name, duration) in [
        ("embed", metrics.embedding_duration),
        ("norm", metrics.norm_duration),
        ("qkv", metrics.qkv_duration),
        ("attention", metrics.attention_duration),
        ("mlp", metrics.mlp_duration),
        ("logits", metrics.logits_duration),
    ] {
        let millis = duration.as_secs_f64() * 1_000.0;
        lines.push((
            format!("phase_{name}"),
            format!(
                "ms={:.3} share_pct={:.2}",
                millis,
                millis * 100.0 / total_ms
            ),
        ));
    }
    lines
}

fn parse_format_flag(args: &[String]) -> Result<(Vec<String>, OutputFormat), String> {
    let mut format = OutputFormat::KeyValue;
    let mut positional = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == "--format" {
            let value = args
                .get(index + 1)
                .ok_or_else(|| "--format requires a value".to_string())?;
            format = parse_output_format(value)?;
            index += 2;
        } else {
            positional.push(args[index].clone());
            index += 1;
        }
    }
    Ok((positional, format))
}

fn parse_bench_flags(
    args: &[String],
) -> Result<(Vec<String>, OutputFormat, BenchExportPaths), String> {
    let mut format = OutputFormat::KeyValue;
    let mut export_paths = BenchExportPaths::default();
    let mut positional = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == "--format" {
            let value = args
                .get(index + 1)
                .ok_or_else(|| "--format requires a value".to_string())?;
            format = parse_output_format(value)?;
            index += 2;
        } else if args[index] == "--markdown" {
            let path = args
                .get(index + 1)
                .ok_or_else(|| "--markdown requires a path".to_string())?;
            export_paths.markdown = Some(path.clone());
            index += 2;
        } else if args[index] == "--kv" {
            let path = args
                .get(index + 1)
                .ok_or_else(|| "--kv requires a path".to_string())?;
            export_paths.key_value = Some(path.clone());
            index += 2;
        } else if args[index] == "--csv" {
            let path = args
                .get(index + 1)
                .ok_or_else(|| "--csv requires a path".to_string())?;
            export_paths.csv = Some(path.clone());
            index += 2;
        } else {
            positional.push(args[index].clone());
            index += 1;
        }
    }
    Ok((positional, format, export_paths))
}

fn parse_output_format(value: &str) -> Result<OutputFormat, String> {
    match value {
        "plain" => Ok(OutputFormat::Plain),
        "kv" | "key-value" => Ok(OutputFormat::KeyValue),
        "markdown" | "md" => Ok(OutputFormat::Markdown),
        other => Err(format!(
            "unsupported format `{other}`; expected plain, kv, or markdown"
        )),
    }
}

fn print_section(format: OutputFormat, title: &str, lines: &[(String, String)]) {
    match format {
        OutputFormat::KeyValue => {
            for (key, value) in lines {
                println!("{key}={value}");
            }
        }
        OutputFormat::Plain => {
            println!("{title}:");
            for (key, value) in lines {
                println!("- {key}: {value}");
            }
        }
        OutputFormat::Markdown => {
            println!("## {title}\n");
            for (key, value) in lines {
                println!("- **{key}**: `{}`", value.replace('`', "\\`"));
            }
        }
    }
}

fn print_bench_iteration(
    format: OutputFormat,
    iteration: usize,
    metrics: &DecodeMetrics,
    output_text: &str,
) {
    match format {
        OutputFormat::KeyValue => println!(
            "iteration={} {} tok_s={:.3} output={}",
            iteration,
            metrics.summarize(),
            metrics.generated_tokens_per_second(),
            output_text,
        ),
        OutputFormat::Plain => println!(
            "iteration {}: total={:.3}ms qkv={:.3}ms attention={:.3}ms mlp={:.3}ms logits={:.3}ms tok/s={:.3} output={}",
            iteration,
            metrics.total_duration.as_secs_f64() * 1_000.0,
            metrics.qkv_duration.as_secs_f64() * 1_000.0,
            metrics.attention_duration.as_secs_f64() * 1_000.0,
            metrics.mlp_duration.as_secs_f64() * 1_000.0,
            metrics.logits_duration.as_secs_f64() * 1_000.0,
            metrics.generated_tokens_per_second(),
            output_text,
        ),
        OutputFormat::Markdown => println!(
            "- iteration {}: total `{:.3}` ms, qkv `{:.3}` ms, attention `{:.3}` ms, mlp `{:.3}` ms, logits `{:.3}` ms, tok/s `{:.3}`, output `{}`",
            iteration,
            metrics.total_duration.as_secs_f64() * 1_000.0,
            metrics.qkv_duration.as_secs_f64() * 1_000.0,
            metrics.attention_duration.as_secs_f64() * 1_000.0,
            metrics.mlp_duration.as_secs_f64() * 1_000.0,
            metrics.logits_duration.as_secs_f64() * 1_000.0,
            metrics.generated_tokens_per_second(),
            output_text.replace('`', "\\`"),
        ),
    }
}

fn print_bench_summary(format: OutputFormat, report: &BenchReport) {
    match format {
        OutputFormat::KeyValue => println!(
            "avg_total_ms={:.3} avg_qkv_ms={:.3} avg_attention_ms={:.3} avg_mlp_ms={:.3} avg_logits_ms={:.3} avg_tok_s={:.3}",
            report.average_total_ms(),
            report.average_qkv_ms(),
            report.average_attention_ms(),
            report.average_mlp_ms(),
            report.average_logits_ms(),
            report.average_generated_tokens_per_second(),
        ),
        OutputFormat::Plain => println!(
            "average total={:.3}ms qkv={:.3}ms attention={:.3}ms mlp={:.3}ms logits={:.3}ms tok/s={:.3}",
            report.average_total_ms(),
            report.average_qkv_ms(),
            report.average_attention_ms(),
            report.average_mlp_ms(),
            report.average_logits_ms(),
            report.average_generated_tokens_per_second(),
        ),
        OutputFormat::Markdown => println!("{}", report.to_markdown()),
    }
}

fn parse_usize(value: Option<&String>, default: usize, label: &str) -> Result<usize, String> {
    match value {
        Some(value) => value
            .parse::<usize>()
            .map_err(|error| format!("invalid {label} `{value}`: {error}")),
        None => Ok(default),
    }
}

fn usage() -> String {
    format!(
        "{name}\n\nUSAGE:\n  jengine inspect [root] [prompt] [max_new_tokens] [--format plain|kv|markdown]\n  jengine run [root] [prompt] [max_new_tokens] [--format plain|kv|markdown]\n  jengine bench [root] [prompt] [max_new_tokens] [iterations] [--format plain|kv|markdown] [--markdown path] [--kv path] [--csv path]\n  jengine profile [root] [prompt] [max_new_tokens] [--format plain|kv|markdown]\n  jengine validate [root] [prompt] [max_new_tokens] [--format plain|kv|markdown]\n  jengine pack [weights_path] [tensor_name] [rows] [cols] [out_path]\n\nDEFAULT ROOT: {root}\nDEFAULT WEIGHTS: {weights}\nDEFAULT TENSOR: {tensor}",
        name = jengine::name(),
        root = DEFAULT_ROOT,
        weights = DEFAULT_WEIGHTS,
        tensor = DEFAULT_TENSOR,
    )
}
