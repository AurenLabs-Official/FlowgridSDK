//! Local `NanoGpt` inference (checkpoint + tokenizer) for the scheduler.

use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use burn::tensor::{Int, Tensor};
use flowgrid_checkpoint::{load_nano_gpt_checkpoint, resolve_tokenizer_path};
use flowgrid_model::cache::KvCache;
use flowgrid_model::lm::LmModel;
use flowgrid_model::{sample_from_last_logits, NanoGpt, Sampling};
use flowgrid_tokenizer::{DecoderState, FgTokenizer};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::completion::CompletionMeta;

pub use crate::backend::{infer_device, InferB, InferDevice};

/// Loaded local decoder for OpenAI-shaped completions.
pub struct LocalLlm {
    pub model: NanoGpt<InferB>,
    pub tokenizer: FgTokenizer,
    pub device: InferDevice,
}

impl LocalLlm {
    pub fn load_checkpoint(dir: &Path) -> anyhow::Result<Self> {
        let device = crate::backend::infer_device();
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

    fn eos_id(&self) -> Option<u32> {
        std::env::var("FLOWGRID_SERVE_EOS_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .or_else(|| self.tokenizer.eos_id())
    }

    pub fn complete(
        &self,
        prompt: &str,
        max_new: usize,
        sampling: Sampling,
        seed: u64,
        deadline: Option<Instant>,
    ) -> anyhow::Result<(String, CompletionMeta)> {
        let mut out = String::new();
        let meta = self.complete_stream(prompt, max_new, sampling, seed, deadline, |piece| {
            out.push_str(piece)
        })?;
        Ok((out, meta))
    }

    /// Invokes `on_piece` with incremental decoded UTF-8 (tokenizer streaming).
    pub fn complete_stream<F: FnMut(&str)>(
        &self,
        prompt: &str,
        max_new: usize,
        sampling: Sampling,
        seed: u64,
        deadline: Option<Instant>,
        mut on_piece: F,
    ) -> anyhow::Result<CompletionMeta> {
        let check_deadline = || -> anyhow::Result<()> {
            if let Some(d) = deadline {
                if Instant::now() > d {
                    return Err(anyhow::anyhow!("inference timeout"));
                }
            }
            Ok(())
        };
        check_deadline()?;
        let eos = self.eos_id();
        let is_eos = |id: i32| eos.is_some_and(|e| e == id as u32);

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
        if ids.len() > bs {
            let drop = ids.len() - bs;
            ids.drain(0..drop);
        }
        let prompt_tokens = ids.len() as u32;
        let blk0 = self
            .model
            .blocks
            .first()
            .ok_or_else(|| anyhow::anyhow!("model has no transformer blocks"))?;
        let mut caches: Vec<KvCache<InferB>> = (0..self.model.n_layer())
            .map(|_| {
                KvCache::with_capacity(
                    1,
                    blk0.attn.kv_heads(),
                    bs,
                    blk0.attn.head_dim(),
                    &self.device,
                )
            })
            .collect();
        let seq = ids.len();
        let inp =
            Tensor::<InferB, 1, Int>::from_ints(ids.as_slice(), &self.device).reshape([1, seq]);
        check_deadline()?;
        let logits = self.model.forward_step(inp, Some(&mut caches));
        let mut next = sample_from_last_logits(&logits, sampling, &mut rng);
        let mut decode_state = DecoderState::default();
        let max_gen = max_new.max(1);
        let mut completion_tokens: u32 = 0;

        if is_eos(next) {
            return Ok(CompletionMeta {
                prompt_tokens,
                completion_tokens,
                finish_reason: "stop",
            });
        }
        completion_tokens += 1;
        on_piece(
            &self
                .tokenizer
                .decode_streaming(&mut decode_state, next as u32)
                .unwrap_or_default(),
        );

        for _ in 1..max_gen {
            check_deadline()?;
            let t = Tensor::<InferB, 1, Int>::from_ints([next], &self.device).reshape([1, 1]);
            let logits = self.model.forward_step(t, Some(&mut caches));
            next = sample_from_last_logits(&logits, sampling, &mut rng);
            if is_eos(next) {
                return Ok(CompletionMeta {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason: "stop",
                });
            }
            completion_tokens += 1;
            on_piece(
                &self
                    .tokenizer
                    .decode_streaming(&mut decode_state, next as u32)
                    .unwrap_or_default(),
            );
        }

        Ok(CompletionMeta {
            prompt_tokens,
            completion_tokens,
            finish_reason: "length",
        })
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

#[cfg(test)]
mod eos_completion_token_tests {
    /// Mirrors `complete_stream` completion-token policy: EOS IDs are never counted toward
    /// OpenAI-shaped `completion_tokens`.
    fn simulate(first_is_eos: bool, eos_on_extra_steps: &[bool], max_gen: usize) -> u32 {
        let mut completion_tokens = 0u32;
        if first_is_eos {
            return completion_tokens;
        }
        completion_tokens += 1;
        for &step_eos in eos_on_extra_steps
            .iter()
            .take(max_gen.saturating_sub(1))
        {
            if step_eos {
                return completion_tokens;
            }
            completion_tokens += 1;
        }
        completion_tokens
    }

    #[test]
    fn immediate_eos_zero_completion_tokens() {
        assert_eq!(simulate(true, &[], 4), 0);
    }

    #[test]
    fn eos_second_step_counts_one_non_eos_completion() {
        assert_eq!(simulate(false, &[true], 4), 1);
    }

    #[test]
    fn no_eos_hits_max_gen() {
        assert_eq!(
            simulate(false, &[false, false, false], 4),
            4,
            "max_gen generations, none EOS"
        );
    }
}
