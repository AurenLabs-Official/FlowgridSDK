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
use flowgrid_data::{write_token_blob, TokenMmap};
use flowgrid_model::{NanoGpt, NanoGptConfig};
use flowgrid_train::loop_train::{batch_from_mmap, debug_loss_over_mmap, loss_for_window};
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
        Cmd::Prepare { input, output } => {
            let text = std::fs::read_to_string(&input)
                .with_context(|| format!("read {}", input.display()))?;
            let ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
            write_token_blob(&output, &ids).with_context(|| format!("write {}", output.display()))?;
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
        } => {
            let mmap = TokenMmap::open(&tokens).context("mmap tokens")?;
            let device = cpu_device();
            let cfg = NanoGptConfig {
                vocab_size: vocab,
                block_size: block,
                n_layer: layers,
                n_embd: embd,
                dropout: 0.0,
            };
            let model = cfg.init::<DiffBackend>(&device);
            let before = debug_loss_over_mmap(&model, &mmap, &cfg, steps, &device);
            println!("mean CE (~random init): {before:.4}");
            if learn {
                let mut model = cfg.init::<DiffBackend>(&device);
                let mut optim = AdamConfig::new().init::<DiffBackend, NanoGpt<DiffBackend>>();
                let span = mmap.len_tokens().saturating_sub(cfg.block_size + 1);
                if span == 0 {
                    anyhow::bail!("token blob too short for block_size={}", cfg.block_size);
                }
                for step in 0..steps {
                    let start = (step * 31) % span;
                    if let Some(b) =
                        batch_from_mmap::<DiffBackend>(&mmap, start, cfg.block_size, &device)
                    {
                        let loss = loss_for_window(&model, b, &device);
                        let grads_tensor = loss.backward();
                        let grads = GradientsParams::from_grads(grads_tensor, &model);
                        model = optim.step(lr, model, grads);
                    }
                }
                let after = debug_loss_over_mmap(&model, &mmap, &cfg, steps, &device);
                println!("mean CE after Adam: {after:.4}");
            }
        }
        Cmd::Generate {
            prompt,
            max_new,
            vocab,
            block,
            layers,
            embd,
        } => {
            let device = cpu_device();
            let cfg = NanoGptConfig {
                vocab_size: vocab,
                block_size: block,
                n_layer: layers,
                n_embd: embd,
                dropout: 0.0,
            };
            let model = cfg.init::<InferBackend>(&device);
            let mut ids: Vec<i32> = prompt.bytes().map(|b| b as i32).collect();
            if ids.is_empty() {
                anyhow::bail!("empty prompt");
            }
            if ids.len() > block {
                anyhow::bail!("prompt longer than block_size ({block})");
            }
            let mut generated: Vec<u8> = Vec::with_capacity(max_new);
            for _ in 0..max_new {
                let seq = ids.len();
                let inp =
                    Tensor::<InferBackend, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, seq]);
                let logits = model.forward(inp);
                let row = logits.slice([0..1, (seq - 1)..seq, 0..vocab]);
                let next = row.argmax(2).reshape([1]).into_scalar();
                let next_i = num_traits::cast::ToPrimitive::to_i32(&next).unwrap_or(0);
                ids.push(next_i);
                if let Ok(b) = u8::try_from(next_i as u32) {
                    generated.push(b);
                }
                if ids.len() > block {
                    ids.remove(0);
                }
            }
            println!("{}", String::from_utf8_lossy(&generated));
        }
    }
    Ok(())
}
