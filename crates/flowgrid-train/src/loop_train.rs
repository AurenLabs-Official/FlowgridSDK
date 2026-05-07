//! Training utilities for [`flowgrid_model::nano_gpt::NanoGpt`].

use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::{backend::AutodiffBackend, Int, Tensor};
use flowgrid_data::sequence_bytes_to_ids;
use flowgrid_model::{NanoGpt, NanoGptConfig};
use num_traits::cast::ToPrimitive;
use std::path::Path;

use flowgrid_data::TokenMmap;

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub grad_accum: usize,
    pub max_grad_norm: Option<f32>,
    pub warmup: usize,
    pub base_lr: f64,
    pub min_lr: f64,
    pub total_steps: usize,
}

/// Language-modeling batch: `tokens` and one-step-ahead `targets` (same shape `[1, seq]`).
#[derive(Debug, Clone)]
pub struct LmBatch<B: AutodiffBackend> {
    pub tokens: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

/// Compute next-token cross-entropy for a single window (flattened positions).
pub fn loss_for_window<B: AutodiffBackend>(
    model: &NanoGpt<B>,
    batch: LmBatch<B>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let logits = model.forward(batch.tokens);
    let [b, t, v] = logits.dims();
    let logits_flat = logits.reshape([b * t, v]);
    let targets_flat = batch.targets.reshape([b * t]);
    let ce = CrossEntropyLossConfig::new().init(device);
    ce.forward(logits_flat, targets_flat)
}

/// Map a window of `seq` token ids to `(input, target)` pairs (offset by one).
pub fn window_to_batch<B: AutodiffBackend>(
    ids: &[u32],
    device: &B::Device,
) -> Option<LmBatch<B>> {
    if ids.len() < 2 {
        return None;
    }
    let seq = ids.len() - 1;
    let mut inp = Vec::with_capacity(seq);
    let mut tgt = Vec::with_capacity(seq);
    for i in 0..seq {
        inp.push(ids[i] as i32);
        tgt.push(ids[i + 1] as i32);
    }
    let tokens = Tensor::from_ints(inp.as_slice(), device).reshape([1, seq]);
    let targets = Tensor::from_ints(tgt.as_slice(), device).reshape([1, seq]);
    Some(LmBatch { tokens, targets })
}

/// Read one training window from a token mmap at `start` (length `block_size+1` ids).
pub fn batch_from_mmap<B: AutodiffBackend>(
    mmap: &TokenMmap,
    start: usize,
    block_size: usize,
    device: &B::Device,
) -> Option<LmBatch<B>> {
    let need = block_size + 1;
    if start + need > mmap.len_tokens() {
        return None;
    }
    let bytes = mmap
        .as_bytes()
        .get(start * 4..(start + need) * 4)?;
    let ids = sequence_bytes_to_ids(bytes).ok()?;
    window_to_batch(&ids, device)
}

/// Simple smoke: compute loss for a few random-ish windows (no optimizer).
pub fn debug_loss_over_mmap<B: AutodiffBackend>(
    model: &NanoGpt<B>,
    mmap: &TokenMmap,
    cfg: &NanoGptConfig,
    steps: usize,
    device: &B::Device,
) -> f32 {
    let mut acc = 0.0f32;
    let span = mmap.len_tokens().saturating_sub(cfg.block_size + 1);
    if span == 0 || steps == 0 {
        return 0.0;
    }

    for step in 0..steps {
        let start = (step * 97) % span;
        if let Some(b) = batch_from_mmap::<B>(mmap, start, cfg.block_size, device) {
            let l = loss_for_window(model, b, device);
            let v = l.into_scalar();
            acc += v.to_f32().unwrap_or(0.0);
        }
    }
    acc / steps as f32
}

/// Load a previously written token blob and return average loss (diagnostics).
pub fn debug_loss_file<B: AutodiffBackend>(
    model: &NanoGpt<B>,
    path: &Path,
    cfg: &NanoGptConfig,
    device: &B::Device,
) -> f32 {
    let mmap = TokenMmap::open(path).expect("mmap");
    let start = 0;
    if let Some(b) = batch_from_mmap::<B>(&mmap, start, cfg.block_size, device) {
        let l = loss_for_window(model, b, device);
        l.into_scalar().to_f32().unwrap_or(0.0)
    } else {
        0.0
    }
}
