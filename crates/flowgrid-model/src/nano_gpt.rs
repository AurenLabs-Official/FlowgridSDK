//! nanoGPT-style decoder LM using token-wise residual FFN blocks (attention/RoPE wired in later milestones).

use burn::nn::DropoutConfig;
use burn::nn::{Dropout, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::{backend::Backend, Device, Int, Tensor};
use flowgrid_tensor::{Config, Module};

#[derive(Config, Debug)]
pub struct NanoGptConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
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
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl NanoGptConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> NanoGpt<B> {
        let tok_emb = EmbeddingConfig::new(self.vocab_size, self.n_embd).init(device);
        let pos_emb = EmbeddingConfig::new(self.block_size, self.n_embd).init(device);
        let mut blocks = Vec::with_capacity(self.n_layer);
        for _ in 0..self.n_layer {
            blocks.push(Block::new(self.n_embd, self.dropout, device));
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
    fn new(n_embd: usize, dropout: f64, device: &Device<B>) -> Self {
        Block {
            fc1: LinearConfig::new(n_embd, 4 * n_embd).init(device),
            fc2: LinearConfig::new(4 * n_embd, n_embd).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = gelu(self.fc1.forward(x.clone()));
        let h = self.dropout.forward(h);
        let h = self.fc2.forward(h);
        x + self.dropout.forward(h)
    }
}

impl<B: Backend> NanoGpt<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = tokens.dims();
        let device = tokens.device();
        let pos = Tensor::arange(0..seq as i64, &device).reshape([1, seq]);
        let pos = pos.repeat(0, batch);
        let tok = self.tok_emb.forward(tokens);
        let pos_e = self.pos_emb.forward(pos);
        let mut x = tok + pos_e;
        for blk in &self.blocks {
            x = blk.forward(x);
        }
        self.head.forward(x)
    }

    pub fn block_size(&self) -> usize {
        self.pos_emb.weight.dims()[0]
    }

    pub fn vocab_size(&self) -> usize {
        self.tok_emb.weight.dims()[0]
    }
}
