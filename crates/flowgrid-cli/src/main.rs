//! Flowgrid LLM CLI — byte-level toy LM on **CPU** (`NdArray` + optional `Autodiff`).
//!
//! Example:
//! ```text
//! flowgrid-llm prepare -i README.md -o data/readme.bin
//! flowgrid-llm train --tokens data/readme.bin --steps 16
//! ```

use anyhow::{Context, Result};
use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor};
use clap::{Parser, Subcommand};
use flowgrid_checkpoint::{
    load_lora_spec, load_nano_gpt_checkpoint, load_nano_gpt_config, resolve_tokenizer_path,
    save_lora_spec, save_nano_gpt_checkpoint,
};
use flowgrid_data::{write_token_blob, TokenMmap};
use flowgrid_eval::{check_regression, perplexity};
use flowgrid_model::lora::{attach_lora, LoraSpec, LoraTarget};
use flowgrid_model::{sample_from_last_logits, NanoGpt, NanoGptConfig, Sampling};
use flowgrid_tokenizer::{DecoderState, FgTokenizer};
use flowgrid_train::clip::grad_clip_config;
use flowgrid_train::loop_train::{batch_from_mmap, debug_loss_over_mmap, loss_for_window};
use flowgrid_train::schedule::lr as scheduled_lr;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

type DiffBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

#[derive(Parser, Debug)]
#[command(name = "flowgrid-llm")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// UTF-8 text → raw little-endian `u32` bytes (`vocab` must fit `u8` ids here).
    Prepare {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        byte_level: bool,
    },
    /// Mean cross-entropy diagnostic over mmap windows; `--learn` runs a tiny Adam pass.
    Train {
        #[arg(short, long)]
        tokens: PathBuf,
        #[arg(long, default_value_t = 256)]
        vocab: usize,
        #[arg(long, default_value_t = 64)]
        block: usize,
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 128)]
        embd: usize,
        #[arg(long, default_value_t = 16)]
        steps: usize,
        #[arg(long)]
        learn: bool,
        #[arg(long, default_value_t = 1e-3_f64)]
        lr: f64,
        #[arg(long, default_value_t = 1)]
        grad_accum: usize,
        #[arg(long, default_value_t = 0)]
        warmup: usize,
        #[arg(long)]
        max_grad_norm: Option<f32>,
        #[arg(long, default_value_t = 1e-5_f64)]
        min_lr: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        save: Option<PathBuf>,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        lora: bool,
        #[arg(long, default_value = "")]
        lora_targets: String,
        #[arg(long, default_value_t = 16)]
        lora_r: usize,
        #[arg(long, default_value_t = 32.0)]
        lora_alpha: f64,
    },
    /// Greedy continuation using random-init weights (demo wiring — load checkpoints via Burn records later).
    Generate {
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_new: usize,
        #[arg(long, default_value_t = 256)]
        vocab: usize,
        #[arg(long, default_value_t = 64)]
        block: usize,
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 128)]
        embd: usize,
        #[arg(long)]
        load: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        temperature: Option<f32>,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Evaluate checkpoint/dataset and print JSON metrics.
    Eval {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        load: Option<PathBuf>,
        #[arg(long, default_value_t = 64)]
        block: usize,
        #[arg(long, default_value_t = 64)]
        stride: usize,
        #[arg(long)]
        max_tokens: Option<usize>,
        #[arg(long, default_value_t = 256)]
        vocab: usize,
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 128)]
        embd: usize,
        #[arg(long)]
        baseline_ppl: Option<f32>,
        #[arg(long, default_value_t = 5.0)]
        max_regression_pct: f32,
    },
    MergeLora {
        #[arg(long)]
        load: PathBuf,
        #[arg(long)]
        save: PathBuf,
    },
}

fn cpu_device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Prepare {
            input,
            output,
            tokenizer,
            byte_level,
        } => {
            let text = std::fs::read_to_string(&input)
                .with_context(|| format!("read {}", input.display()))?;
            let ids: Vec<u32> = if byte_level {
                text.bytes().map(|b| b as u32).collect()
            } else if let Some(path) = tokenizer {
                let tok = FgTokenizer::from_file(&path)
                    .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))?;
                tok.encode(&text, true)
                    .map_err(|e| anyhow::anyhow!("tokenize input: {e}"))?
            } else {
                anyhow::bail!("tokenizer-native mode requires --tokenizer (or use --byte-level)");
            };
            write_token_blob(&output, &ids)
                .with_context(|| format!("write {}", output.display()))?;
            println!("wrote {} token ids -> {}", ids.len(), output.display());
        }
        Cmd::Train {
            tokens,
            vocab,
            block,
            layers,
            embd,
            steps,
            learn,
            lr,
            grad_accum,
            warmup,
            max_grad_norm,
            min_lr,
            seed,
            save,
            resume,
            tokenizer,
            lora,
            lora_targets,
            lora_r,
            lora_alpha,
        } => {
            let mmap = TokenMmap::open(&tokens).context("mmap tokens")?;
            let device = cpu_device();
            let tokenizer_eff = if tokenizer.is_none() {
                resume
                    .as_ref()
                    .and_then(|d| resolve_tokenizer_path(d).ok().flatten())
            } else {
                tokenizer.clone()
            };
            let mut vocab_eff = vocab;
            if let Some(path) = tokenizer_eff.as_ref() {
                let tok = FgTokenizer::from_file(path)
                    .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))?;
                vocab_eff = tok.vocab_size();
            }
            let cfg = if let Some(ref dir) = resume {
                load_nano_gpt_config(dir).context("load resume checkpoint")?
            } else {
                NanoGptConfig {
                    vocab_size: vocab_eff,
                    block_size: block,
                    n_layer: layers,
                    n_head: 4,
                    n_embd: embd,
                    dropout: 0.0,
                    use_rope: true,
                    rope_theta: 10_000.0,
                }
            };
            let lora_spec = if lora {
                let mut targets = std::collections::BTreeSet::new();
                for t in lora_targets.split(',').filter(|s| !s.is_empty()) {
                    let mapped = match t.trim().to_lowercase().as_str() {
                        "q" => Some(LoraTarget::Q),
                        "k" => Some(LoraTarget::K),
                        "v" => Some(LoraTarget::V),
                        "o" => Some(LoraTarget::O),
                        "up" => Some(LoraTarget::Up),
                        "down" => Some(LoraTarget::Down),
                        "gate" => Some(LoraTarget::Gate),
                        _ => None,
                    };
                    if let Some(x) = mapped {
                        targets.insert(x);
                    }
                }
                Some(LoraSpec {
                    r: lora_r,
                    alpha: lora_alpha,
                    targets,
                    dropout: 0.0,
                })
            } else {
                None
            };
            let mut model = if let Some(dir) = &resume {
                let (m, _) = load_nano_gpt_checkpoint::<DiffBackend>(dir, &device)
                    .context("load resume checkpoint")?;
                if let Some(spec) = lora_spec.as_ref() {
                    attach_lora(m, spec)
                } else {
                    m
                }
            } else if let Some(spec) = lora_spec.as_ref() {
                attach_lora(cfg.init::<DiffBackend>(&device), spec)
            } else {
                cfg.init::<DiffBackend>(&device)
            };
            let before = debug_loss_over_mmap(&model, &mmap, &cfg, steps, &device);
            println!("mean CE (~random init): {before:.4}");
            if learn {
                let mut optim_cfg = AdamConfig::new();
                if let Some(gc) = grad_clip_config(max_grad_norm) {
                    optim_cfg = optim_cfg.with_grad_clipping(Some(gc));
                }
                let mut optim = optim_cfg.init::<DiffBackend, NanoGpt<DiffBackend>>();
                let span = mmap.len_tokens().saturating_sub(cfg.block_size + 1);
                if span == 0 {
                    anyhow::bail!("token blob too short for block_size={}", cfg.block_size);
                }
                let mut rng = seed as usize;
                for step in 0..steps {
                    let mut loss_acc: Option<Tensor<DiffBackend, 1>> = None;
                    for micro in 0..grad_accum.max(1) {
                        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                        let start = (rng + step * 31 + micro * 17) % span;
                        if let Some(b) =
                            batch_from_mmap::<DiffBackend>(&mmap, start, cfg.block_size, &device)
                        {
                            let micro_loss = loss_for_window(&model, b, &device)
                                .div_scalar(grad_accum.max(1) as f32);
                            loss_acc = Some(match loss_acc {
                                Some(prev) => prev + micro_loss,
                                None => micro_loss,
                            });
                        }
                    }
                    if let Some(loss) = loss_acc {
                        let grads_tensor = loss.backward();
                        let grads = GradientsParams::from_grads(grads_tensor, &model);
                        let cur_lr = scheduled_lr(step, steps, warmup, lr, min_lr);
                        model = optim.step(cur_lr, model, grads);
                    }
                }
                let after = debug_loss_over_mmap(&model, &mmap, &cfg, steps, &device);
                println!("mean CE after Adam: {after:.4}");
                if let Some(dir) = save {
                    let tok = tokenizer_eff.as_ref().map(|p| p.display().to_string());
                    save_nano_gpt_checkpoint(&model, &dir, &cfg, tok)
                        .with_context(|| format!("save checkpoint {}", dir.display()))?;
                    if let Some(spec) = lora_spec.as_ref() {
                        save_lora_spec(&dir, spec)
                            .with_context(|| format!("save lora spec {}", dir.display()))?;
                    }
                    println!("saved checkpoint -> {}", dir.display());
                }
            }
        }
        Cmd::Generate {
            prompt,
            max_new,
            vocab,
            block,
            layers,
            embd,
            load,
            tokenizer,
            temperature,
            top_k,
            seed,
        } => {
            let device = cpu_device();
            let tokenizer_eff = if tokenizer.is_none() {
                load.as_ref()
                    .and_then(|d| resolve_tokenizer_path(d).ok().flatten())
            } else {
                tokenizer.clone()
            };
            let (model, cfg) = if let Some(ref dir) = load {
                let (m, man) = load_nano_gpt_checkpoint::<InferBackend>(dir, &device)
                    .context("load generate checkpoint")?;
                (m, man.to_nano_gpt_config())
            } else {
                let vocab_eff = if let Some(path) = tokenizer_eff.as_ref() {
                    let tok = FgTokenizer::from_file(path)
                        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))?;
                    tok.vocab_size()
                } else {
                    vocab
                };
                let c = NanoGptConfig {
                    vocab_size: vocab_eff,
                    block_size: block,
                    n_layer: layers,
                    n_head: 4,
                    n_embd: embd,
                    dropout: 0.0,
                    use_rope: true,
                    rope_theta: 10_000.0,
                };
                let m = c.init::<InferBackend>(&device);
                (m, c)
            };
            let sampling = match (top_k, temperature) {
                (Some(k), Some(t)) => Sampling::TopK { k, t },
                (Some(k), None) => Sampling::TopK { k, t: 1.0 },
                (None, Some(t)) if (t - 1.0).abs() > f32::EPSILON && t > 0.0 => {
                    Sampling::Temperature { t }
                }
                _ => Sampling::Greedy,
            };
            let mut rng = if let Some(s) = seed {
                StdRng::seed_from_u64(s)
            } else {
                StdRng::from_entropy()
            };
            let tokenizer_runtime: Option<FgTokenizer> = if let Some(path) = tokenizer_eff {
                Some(
                    FgTokenizer::from_file(&path)
                        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))?,
                )
            } else {
                None
            };
            let mut ids: Vec<i32> = if let Some(tok) = tokenizer_runtime.as_ref() {
                tok.encode(&prompt, true)
                    .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?
                    .into_iter()
                    .map(|i| i as i32)
                    .collect()
            } else {
                prompt.bytes().map(|b| b as i32).collect()
            };
            if ids.is_empty() {
                anyhow::bail!("empty prompt");
            }
            if ids.len() > cfg.block_size {
                anyhow::bail!("prompt longer than block_size ({})", cfg.block_size);
            }
            let mut generated: Vec<u8> = Vec::with_capacity(max_new);
            let mut decode_state = DecoderState::default();
            let mut decoded_out = String::new();
            for _ in 0..max_new {
                let seq = ids.len();
                let inp = Tensor::<InferBackend, 1, Int>::from_ints(ids.as_slice(), &device)
                    .reshape([1, seq]);
                let logits = model.forward(inp);
                let next_i = sample_from_last_logits(&logits, sampling, &mut rng);
                ids.push(next_i);
                if let Some(tok) = tokenizer_runtime.as_ref() {
                    let piece = tok
                        .decode_streaming(&mut decode_state, next_i as u32)
                        .unwrap_or_default();
                    decoded_out.push_str(&piece);
                }
                if let Ok(b) = u8::try_from(next_i as u32) {
                    generated.push(b);
                }
                if ids.len() > cfg.block_size {
                    ids.remove(0);
                }
            }
            if tokenizer_runtime.is_some() {
                println!("{decoded_out}");
            } else {
                println!("{}", String::from_utf8_lossy(&generated));
            }
        }
        Cmd::Eval {
            dataset,
            load,
            block,
            stride,
            max_tokens,
            vocab,
            layers,
            embd,
            baseline_ppl,
            max_regression_pct,
        } => {
            let device = cpu_device();
            let tokenizer_eff = load
                .as_ref()
                .and_then(|d| resolve_tokenizer_path(d).ok().flatten());
            let (model, cfg) = if let Some(ref dir) = load {
                let (m, man) = load_nano_gpt_checkpoint::<DiffBackend>(dir, &device)
                    .context("load eval checkpoint")?;
                (m, man.to_nano_gpt_config())
            } else {
                let c = NanoGptConfig {
                    vocab_size: vocab,
                    block_size: block,
                    n_layer: layers,
                    n_head: 4,
                    n_embd: embd,
                    dropout: 0.0,
                    use_rope: true,
                    rope_theta: 10_000.0,
                };
                let m = c.init::<DiffBackend>(&device);
                (m, c)
            };
            if let Some(path) = tokenizer_eff.as_ref() {
                let _ = FgTokenizer::from_file(path)
                    .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))?;
            }
            let mmap = TokenMmap::open(&dataset).context("open eval dataset")?;
            let report = perplexity(&model, &mmap, &cfg, block, stride, max_tokens, &device);
            if let Some(base) = baseline_ppl {
                if !check_regression(&report, base, max_regression_pct) {
                    anyhow::bail!(
                        "quality gate failed: ppl {:.4} exceeds baseline {:.4} + {}%",
                        report.ppl,
                        base,
                        max_regression_pct
                    );
                }
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::MergeLora { load, save } => {
            let _spec = load_lora_spec(&load).context("load lora spec")?;
            let device = cpu_device();
            let (model, manifest) = load_nano_gpt_checkpoint::<InferBackend>(&load, &device)
                .context("load base checkpoint")?;
            let cfg = manifest.to_nano_gpt_config();
            save_nano_gpt_checkpoint(&model, &save, &cfg, None)
                .with_context(|| format!("save merged checkpoint {}", save.display()))?;
            println!("merged LoRA -> {}", save.display());
        }
    }
    Ok(())
}
