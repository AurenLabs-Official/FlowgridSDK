//! nanoGPT-style decoder LM with plumbed causal-attention blocks and KV-cache path.

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Device, Int, Tensor};
use flowgrid_tensor::{Config, Module};

use crate::attention::{CausalSelfAttn, CausalSelfAttnConfig};
use crate::cache::{KvCache, KvCacheStack};
use crate::lm::LmModel;
use crate::mlp::Mlp;
use crate::norm::{Norm, NormConfig};

#[derive(Config, Debug)]
pub struct NanoGptConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    #[config(default = 4)]
    pub n_head: usize,
    /// GQA: KV attention heads; **`0` = multi-head (`n_head`)**. Must divide `n_head`.
    #[config(default = 0)]
    pub n_kv_head: usize,
    pub n_embd: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// RoPE in attention (recommended). When `true`, learned position embeddings are not added
    /// to the residual stream to avoid double positional encoding.
    #[config(default = true)]
    pub use_rope: bool,
    /// Base used in RoPE frequency table when `use_rope` is set.
    #[config(default = 10_000.0)]
    pub rope_theta: f32,
}

#[derive(Module, Debug)]
pub struct NanoGpt<B: Backend> {
    pub tok_emb: Embedding<B>,
    pub pos_emb: Embedding<B>,
    pub blocks: Vec<Block<B>>,
    pub head: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub attn: CausalSelfAttn<B>,
    pub mlp: Mlp<B>,
    pub pre_norm: Norm<B>,
    pub post_norm: Norm<B>,
}

impl NanoGptConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> NanoGpt<B> {
        let tok_emb = EmbeddingConfig::new(self.vocab_size, self.n_embd).init(device);
        let pos_emb = EmbeddingConfig::new(self.block_size, self.n_embd).init(device);
        let mut blocks = Vec::with_capacity(self.n_layer);
        for _ in 0..self.n_layer {
            blocks.push(Block::new(self, device));
        }
        let head = LinearConfig::new(self.n_embd, self.vocab_size).init(device);
        NanoGpt {
            tok_emb,
            pos_emb,
            blocks,
            head,
        }
    }

    /// Effective KV heads: `n_kv_head == 0` → `n_head`. **Panics** if `n_head % n_kv_head != 0`.
    pub fn resolved_n_kv_head(&self) -> usize {
        let nh = self.n_head.max(1);
        let nkv = if self.n_kv_head == 0 {
            nh
        } else {
            self.n_kv_head.max(1)
        };
        assert!(
            nh % nkv == 0,
            "n_head ({nh}) must be divisible by n_kv_head ({nkv})"
        );
        nkv
    }
}

impl<B: Backend> Block<B> {
    fn merge_lora_layers(self, device: &Device<B>) -> Self {
        Self {
            attn: self.attn.merge_lora_layers(device),
            mlp: self.mlp.merge_lora_layers(device),
            pre_norm: self.pre_norm,
            post_norm: self.post_norm,
        }
    }

    fn new(cfg: &NanoGptConfig, device: &Device<B>) -> Self {
        let n_head = cfg.n_head.max(1);
        let n_kv = cfg.resolved_n_kv_head();
        let attn_cfg = CausalSelfAttnConfig {
            n_head,
            n_kv_head: n_kv,
            head_dim: (cfg.n_embd / n_head).max(1),
            use_rope: cfg.use_rope,
            rope_theta: cfg.rope_theta,
            max_seq: cfg.block_size,
        };
        Self {
            attn: attn_cfg.init::<B>(cfg.n_embd, device),
            mlp: Mlp::new(cfg.n_embd, cfg.dropout, device),
            pre_norm: NormConfig::new(cfg.n_embd).init(device),
            post_norm: NormConfig::new(cfg.n_embd).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>, cache: Option<&mut KvCache<B>>) -> Tensor<B, 3> {
        let h = self.pre_norm.forward(x.clone());
        let x = x + self.attn.forward(h, cache);
        let h = self.post_norm.forward(x.clone());
        x + self.mlp.forward(h)
    }
}

impl<B: Backend> NanoGpt<B> {
    /// Merge all `LoraLinear` projections into their base weights (adapter contribution disabled).
    /// Use after loading a checkpoint that was trained with LoRA to export a dense-equivalent model.
    pub fn merge_lora_adapters(self, device: &Device<B>) -> Self {
        let blocks = self
            .blocks
            .into_iter()
            .map(|b| b.merge_lora_layers(device))
            .collect();
        Self {
            tok_emb: self.tok_emb,
            pos_emb: self.pos_emb,
            blocks,
            head: self.head,
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.logits_with_cache(tokens, None)
    }

    pub fn block_size(&self) -> usize {
        self.pos_emb.weight.dims()[0]
    }

    pub fn vocab_size(&self) -> usize {
        self.tok_emb.weight.dims()[0]
    }

    fn logits_with_cache(
        &self,
        tokens: Tensor<B, 2, Int>,
        mut cache: Option<&mut KvCacheStack<B>>,
    ) -> Tensor<B, 3> {
        let [batch, seq] = tokens.dims();
        let device = tokens.device();
        let use_rope = self
            .blocks
            .first()
            .map(|b| b.attn.uses_rope())
            .unwrap_or(true);
        let tok = self.tok_emb.forward(tokens);
        let mut x = if use_rope {
            tok
        } else {
            let pos_start = cache
                .as_ref()
                .and_then(|caches| caches.first())
                .and_then(|c| c.view().map(|(k, _)| k.dims()[2]))
                .unwrap_or(0);
            let pos = Tensor::arange(pos_start as i64..(pos_start + seq) as i64, &device)
                .reshape([1, seq])
                .repeat(0, batch);
            let pos_e = self.pos_emb.forward(pos);
            tok + pos_e
        };
        for (idx, blk) in self.blocks.iter().enumerate() {
            if let Some(caches) = cache.as_deref_mut() {
                while caches.len() <= idx {
                    caches.push(KvCache::with_capacity(
                        batch,
                        blk.attn.n_kv_head,
                        self.block_size(),
                        blk.attn.d_k,
                        &device,
                    ));
                }
                x = blk.forward(x, caches.get_mut(idx));
            } else {
                x = blk.forward(x, None);
            }
        }
        self.head.forward(x)
    }
}

impl<B: Backend> LmModel<B> for NanoGpt<B> {
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.logits_with_cache(tokens, None)
    }

    fn forward_step(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: Option<&mut KvCacheStack<B>>,
    ) -> Tensor<B, 3> {
        self.logits_with_cache(tokens, cache)
    }

    fn block_size(&self) -> usize {
        self.pos_emb.weight.dims()[0]
    }

    fn vocab_size(&self) -> usize {
        self.tok_emb.weight.dims()[0]
    }

    fn n_layer(&self) -> usize {
        self.blocks.len()
    }
}
