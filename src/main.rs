use jengine::gpu::vulkan::{collect_vulkan_report, version_string};
use jengine::report::{BenchIterationRecord, BenchReport};
use jengine::runtime::packed::PackedTensorFile;
use jengine::runtime::reference::{DecodeMetrics, MemoryReport, ReferenceModel};
use jengine::runtime::repack::{analyze_ternary_packability, pack_ternary_g128};
use jengine::runtime::weights::WeightStore;

const DEFAULT_ROOT: &str = "/home/jeremy/models/bonsai-1.7b";
const DEFAULT_WEIGHTS: &str = "/home/jeremy/models/bonsai-1.7b/model.safetensors";
const DEFAULT_TENSOR: &str = "model.layers.0.self_attn.q_proj.weight";

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

    println!("name={}", jengine::name());
    println!("model_root={root}");
    println!(
        "architecture={} layers={} vocab_size={} estimated_params={} estimated_fp16_bytes={}",
        inspection.architecture,
        inspection.layers,
        inspection.vocab_size,
        inspection.estimated_parameter_count,
        inspection.estimated_fp16_bytes,
    );
    println!("weights={}", progress.summarize());
    println!("probe={}", probe.summarize());
    if let Some(diagnostics) = model.tokenizer_diagnostics() {
        println!("tokenizer={}", diagnostics.summarize());
    }

    if let Some(prompt) = prompt {
        let prompt_analysis = model
            .prompt_analysis(&prompt)
            .map_err(|error| error.to_string())?;
        let memory_report = model.memory_report(prompt_analysis.token_count, max_new_tokens);
        println!("prompt_analysis={}", prompt_analysis.summarize());
        println!("memory_report={}", memory_report.summarize());
    } else {
        println!(
            "memory_per_token_runtime_f32_bytes={}",
            model
                .memory_report(0, 1)
                .kv_cache_bytes_per_token_runtime_f32
        );
    }

    match collect_vulkan_report() {
        Ok(report) => {
            println!("vulkan={}", report.summarize());
            for device in report.devices {
                println!(
                    "vulkan_device name={} type={} api={} queues={}",
                    device.name,
                    device.device_type,
                    version_string(device.api_version),
                    device.queue_families.len(),
                );
            }
        }
        Err(error) => println!("vulkan_unavailable={error}"),
    }

    Ok(())
}

fn cmd_run(args: &[String]) -> Result<(), String> {
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

    print_run_header(&model, &prompt_analysis, &memory_report);
    print_decode_metrics(&result.metrics);
    println!("output={}", result.output_text);
    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<(), String> {
    let (args, markdown_path) = parse_markdown_flag(args)?;
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

    print_run_header(&model, &prompt_analysis, &memory_report);
    for iteration in 0..iterations {
        let result = model
            .generate_greedy(&prompt, max_new_tokens)
            .map_err(|error| error.to_string())?;
        println!(
            "iteration={} {} tok_s={:.3} output={}",
            iteration + 1,
            result.metrics.summarize(),
            result.metrics.generated_tokens_per_second(),
            result.output_text,
        );
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
    println!(
        "avg_total_ms={:.3} avg_qkv_ms={:.3} avg_attention_ms={:.3} avg_mlp_ms={:.3} avg_logits_ms={:.3} avg_tok_s={:.3}",
        report.average_total_ms(),
        report.average_qkv_ms(),
        report.average_attention_ms(),
        report.average_mlp_ms(),
        report.average_logits_ms(),
        report.average_generated_tokens_per_second(),
    );
    if let Some(path) = markdown_path {
        report
            .write_markdown_to_path(&path)
            .map_err(|error| format!("failed to write markdown report to {path}: {error}"))?;
        println!("markdown_report={path}");
    }
    Ok(())
}

fn cmd_profile(args: &[String]) -> Result<(), String> {
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

    print_run_header(&model, &prompt_analysis, &memory_report);
    print_decode_metrics(&result.metrics);
    print_phase_shares(&result.metrics);
    println!("output={}", result.output_text);
    Ok(())
}

fn cmd_validate(args: &[String]) -> Result<(), String> {
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

    println!("asset_validation=ok root={root}");
    println!("weights={}", progress.summarize());
    println!("probe={}", probe.summarize());
    println!("prompt_analysis={}", prompt_analysis.summarize());
    println!("decode_metrics={}", result.metrics.summarize());
    println!("validation=ok output={}", result.output_text);
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
    model: &ReferenceModel,
    prompt_analysis: &jengine::model::tokenizer::PromptAnalysis,
    memory_report: &MemoryReport,
) {
    if let Some(diagnostics) = model.tokenizer_diagnostics() {
        println!("tokenizer={}", diagnostics.summarize());
    }
    println!("prompt_analysis={}", prompt_analysis.summarize());
    println!("memory_report={}", memory_report.summarize());
}

fn print_decode_metrics(metrics: &DecodeMetrics) {
    println!("metrics={}", metrics.summarize());
    println!(
        "generated_tok_s={:.3} total_sequence_tokens={}",
        metrics.generated_tokens_per_second(),
        metrics.total_sequence_tokens(),
    );
}

fn print_phase_shares(metrics: &DecodeMetrics) {
    let total_ms = metrics.total_duration.as_secs_f64() * 1_000.0;
    if total_ms <= f64::EPSILON {
        return;
    }
    for (name, duration) in [
        ("embed", metrics.embedding_duration),
        ("norm", metrics.norm_duration),
        ("qkv", metrics.qkv_duration),
        ("attention", metrics.attention_duration),
        ("mlp", metrics.mlp_duration),
        ("logits", metrics.logits_duration),
    ] {
        let millis = duration.as_secs_f64() * 1_000.0;
        println!(
            "phase={} ms={:.3} share_pct={:.2}",
            name,
            millis,
            millis * 100.0 / total_ms,
        );
    }
}

fn parse_markdown_flag(args: &[String]) -> Result<(Vec<String>, Option<String>), String> {
    let mut markdown_path = None;
    let mut positional = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == "--markdown" {
            let path = args
                .get(index + 1)
                .ok_or_else(|| "--markdown requires a path".to_string())?;
            markdown_path = Some(path.clone());
            index += 2;
        } else {
            positional.push(args[index].clone());
            index += 1;
        }
    }
    Ok((positional, markdown_path))
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
        "{name}\n\nUSAGE:\n  jengine inspect [root] [prompt] [max_new_tokens]\n  jengine run [root] [prompt] [max_new_tokens]\n  jengine bench [root] [prompt] [max_new_tokens] [iterations] [--markdown path]\n  jengine profile [root] [prompt] [max_new_tokens]\n  jengine validate [root] [prompt] [max_new_tokens]\n  jengine pack [weights_path] [tensor_name] [rows] [cols] [out_path]\n\nDEFAULT ROOT: {root}\nDEFAULT WEIGHTS: {weights}\nDEFAULT TENSOR: {tensor}",
        name = jengine::name(),
        root = DEFAULT_ROOT,
        weights = DEFAULT_WEIGHTS,
        tensor = DEFAULT_TENSOR,
    )
}
