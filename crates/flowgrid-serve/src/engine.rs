//! CPU `NanoGpt` inference (checkpoint + tokenizer) for the scheduler.

use std::path::Path;

use anyhow::Context;
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use flowgrid_checkpoint::{load_nano_gpt_checkpoint, resolve_tokenizer_path};
use flowgrid_model::cache::KvCache;
use flowgrid_model::lm::LmModel;
use flowgrid_model::{sample_from_last_logits, NanoGpt, Sampling};
use flowgrid_tokenizer::{DecoderState, FgTokenizer};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub type InferB = NdArray<f32>;

/// Loaded local decoder for OpenAI-shaped completions.
pub struct LocalLlm {
    pub model: NanoGpt<InferB>,
    pub tokenizer: FgTokenizer,
    pub device: burn_ndarray::NdArrayDevice,
}

impl LocalLlm {
    pub fn load_checkpoint(dir: &Path) -> anyhow::Result<Self> {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let (model, _manifest) =
            load_nano_gpt_checkpoint::<InferB>(dir, &device).context("load NanoGpt checkpoint")?;
        let tok_path = resolve_tokenizer_path(dir)
            .context("resolve tokenizer from manifest")?
            .ok_or_else(|| anyhow::anyhow!("checkpoint manifest missing tokenizer_path"))?;
        let tokenizer = FgTokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tok_path.display()))?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn from_env() -> anyhow::Result<Option<LocalLlm>> {
        let Some(dir) = std::env::var("FLOWGRID_SERVE_CHECKPOINT").ok() else {
            return Ok(None);
        };
        let p = Path::new(&dir);
        let llm = Self::load_checkpoint(p)
            .with_context(|| format!("FLOWGRID_SERVE_CHECKPOINT={}", p.display()))?;
        Ok(Some(llm))
    }

    pub fn complete(
        &self,
        prompt: &str,
        max_new: usize,
        sampling: Sampling,
        seed: u64,
    ) -> anyhow::Result<String> {
        let mut out = String::new();
        self.complete_stream(prompt, max_new, sampling, seed, |piece| out.push_str(piece))?;
        Ok(out)
    }

    /// Invokes `on_piece` with incremental decoded UTF-8 (tokenizer streaming).
    pub fn complete_stream<F: FnMut(&str)>(
        &self,
        prompt: &str,
        max_new: usize,
        sampling: Sampling,
        seed: u64,
        mut on_piece: F,
    ) -> anyhow::Result<()> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ids: Vec<i32> = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .into_iter()
            .map(|x| x as i32)
            .collect();
        if ids.is_empty() {
            ids.push(0);
        }
        let bs = self.model.block_size();
        while ids.len() > bs {
            ids.remove(0);
        }
        let mut caches: Vec<KvCache<InferB>> = (0..self.model.n_layer())
            .map(|_| KvCache::empty())
            .collect();
        let seq = ids.len();
        let inp =
            Tensor::<InferB, 1, Int>::from_ints(ids.as_slice(), &self.device).reshape([1, seq]);
        let logits = self.model.forward_step(inp, Some(&mut caches));
        let mut next = sample_from_last_logits(&logits, sampling, &mut rng);
        let mut decode_state = DecoderState::default();
        on_piece(
            &self
                .tokenizer
                .decode_streaming(&mut decode_state, next as u32)
                .unwrap_or_default(),
        );
        let n_gen = max_new.max(1);
        for _ in 1..n_gen {
            let t = Tensor::<InferB, 1, Int>::from_ints([next], &self.device).reshape([1, 1]);
            let logits = self.model.forward_step(t, Some(&mut caches));
            next = sample_from_last_logits(&logits, sampling, &mut rng);
            on_piece(
                &self
                    .tokenizer
                    .decode_streaming(&mut decode_state, next as u32)
                    .unwrap_or_default(),
            );
        }
        Ok(())
    }
}

pub fn serve_sampling_from_env() -> Sampling {
    let t = std::env::var("FLOWGRID_SERVE_TEMPERATURE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    match t {
        None | Some(0.0) => Sampling::Greedy,
        Some(x) if x < 0.0 => Sampling::Greedy,
        Some(temp) => {
            let k = std::env::var("FLOWGRID_SERVE_TOP_K")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&k| k > 0);
            match k {
                Some(k) => Sampling::TopK { k, t: temp },
                None => Sampling::Temperature { t: temp },
            }
        }
    }
}

pub fn serve_seed_from_env() -> u64 {
    std::env::var("FLOWGRID_SERVE_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}
