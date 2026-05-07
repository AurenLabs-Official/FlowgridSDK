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
    pub n_embd: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
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
    attn: CausalSelfAttn<B>,
    mlp: Mlp<B>,
    pre_norm: Norm<B>,
    post_norm: Norm<B>,
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
}

impl<B: Backend> Block<B> {
    fn new(cfg: &NanoGptConfig, device: &Device<B>) -> Self {
        let attn_cfg = CausalSelfAttnConfig {
            n_head: cfg.n_head.max(1),
            n_kv_head: None,
            head_dim: (cfg.n_embd / cfg.n_head.max(1)).max(1),
            use_rope: true,
            rope_theta: 10_000.0,
            max_seq: cfg.block_size,
        };
        Self {
            attn: attn_cfg.init(cfg.n_embd, device),
            mlp: Mlp::new(cfg.n_embd, cfg.dropout, device),
            pre_norm: NormConfig::new(cfg.n_embd).init(device),
            post_norm: NormConfig::new(cfg.n_embd).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>, cache: Option<&mut KvCache<B>>) -> Tensor<B, 3> {
        let h = self.pre_norm.forward(x.clone());
        let x = x + self.attn.forward(h, cache);
        let h = self.post_norm.forward(x.clone());
        self.mlp.forward(h)
    }
}

impl<B: Backend> NanoGpt<B> {
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
        let pos_start = cache
            .as_ref()
            .and_then(|caches| caches.first())
            .and_then(|c| c.view().map(|(k, _)| k.dims()[2]))
            .unwrap_or(0);
        let pos = Tensor::arange(pos_start as i64..(pos_start + seq) as i64, &device).reshape([1, seq]);
        let pos = pos.repeat(0, batch);
        let tok = self.tok_emb.forward(tokens);
        let pos_e = self.pos_emb.forward(pos);
        let mut x = tok + pos_e;
        for (idx, blk) in self.blocks.iter().enumerate() {
            if let Some(caches) = cache.as_deref_mut() {
                while caches.len() <= idx {
                    caches.push(KvCache::empty());
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
