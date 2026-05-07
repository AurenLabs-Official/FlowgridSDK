use burn::tensor::backend::AutodiffBackend;
use flowgrid_data::TokenMmap;
use flowgrid_model::{NanoGpt, NanoGptConfig};
use flowgrid_train::loop_train::{batch_from_mmap, loss_for_window};
use num_traits::cast::ToPrimitive;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub n_tokens: usize,
    pub n_batches: usize,
    pub mean_ce: f32,
    pub ppl: f32,
    pub tokens_per_sec: f32,
    pub peak_mem_mb: f32,
    pub dataset_len_tokens: usize,
    pub block_size: usize,
    pub stride_tokens: usize,
}

pub fn perplexity<B: AutodiffBackend>(
    model: &NanoGpt<B>,
    mmap: &TokenMmap,
    _cfg: &NanoGptConfig,
    block: usize,
    stride: usize,
    max_tokens: Option<usize>,
    device: &B::Device,
) -> EvalReport {
    let start_t = std::time::Instant::now();
    let mut acc = 0.0f32;
    let mut n_batches = 0usize;
    let mut n_tokens = 0usize;
    let limit = max_tokens.unwrap_or(usize::MAX);
    let mut start = 0usize;
    let stride = stride.max(1);
    while start + block < mmap.len_tokens() && n_tokens < limit {
        if let Some(batch) = batch_from_mmap::<B>(mmap, start, block, device) {
            let loss = loss_for_window(model, batch, device);
            let v = loss.into_scalar();
            acc += v.to_f32().unwrap_or(0.0);
            n_batches += 1;
            n_tokens += block;
        }
        start += stride;
    }
    let mean_ce = if n_batches == 0 {
        0.0
    } else {
        acc / n_batches as f32
    };
    let ppl = mean_ce.exp();
    let elapsed = start_t.elapsed().as_secs_f32().max(1e-6);
    EvalReport {
        n_tokens,
        n_batches,
        mean_ce,
        ppl,
        tokens_per_sec: n_tokens as f32 / elapsed,
        peak_mem_mb: 0.0,
        dataset_len_tokens: mmap.len_tokens(),
        block_size: block,
        stride_tokens: stride,
    }
}

pub fn check_regression(report: &EvalReport, baseline_ppl: f32, max_regression_pct: f32) -> bool {
    if baseline_ppl <= 0.0 {
        return true;
    }
    let allowed = baseline_ppl * (1.0 + max_regression_pct.max(0.0) / 100.0);
    report.ppl <= allowed
}
