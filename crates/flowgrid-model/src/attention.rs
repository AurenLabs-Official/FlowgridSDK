use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation;
use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{Config, Module};

use crate::cache::KvCache;
use crate::rope::{apply_rope_qk, rope_tables};

#[derive(Config, Debug)]
pub struct CausalSelfAttnConfig {
    pub n_head: usize,
    pub n_kv_head: Option<usize>,
    pub head_dim: usize,
    #[config(default = true)]
    pub use_rope: bool,
    #[config(default = 10_000.0)]
    pub rope_theta: f32,
    pub max_seq: usize,
}

#[derive(Module, Debug)]
pub struct CausalSelfAttn<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    n_head: usize,
    d_k: usize,
    use_rope: bool,
    max_seq: usize,
}

impl CausalSelfAttnConfig {
    pub fn init<B: Backend>(&self, n_embd: usize, device: &B::Device) -> CausalSelfAttn<B> {
        CausalSelfAttn {
            q_proj: LinearConfig::new(n_embd, n_embd).init(device),
            k_proj: LinearConfig::new(n_embd, n_embd).init(device),
            v_proj: LinearConfig::new(n_embd, n_embd).init(device),
            o_proj: LinearConfig::new(n_embd, n_embd).init(device),
            n_head: self.n_head.max(1),
            d_k: (n_embd / self.n_head.max(1)).max(1),
            use_rope: self.use_rope,
            max_seq: self.max_seq,
        }
    }
}

impl<B: Backend> CausalSelfAttn<B> {
    pub fn forward(&self, x: Tensor<B, 3>, cache: Option<&mut KvCache<B>>) -> Tensor<B, 3> {
        let [batch, seq, d_model] = x.dims();
        let q = self
            .q_proj
            .forward(x.clone())
            .reshape([batch, seq, self.n_head, self.d_k])
            .swap_dims(1, 2);
        let mut k = self
            .k_proj
            .forward(x.clone())
            .reshape([batch, seq, self.n_head, self.d_k])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([batch, seq, self.n_head, self.d_k])
            .swap_dims(1, 2);
        let mut q = q;
        if self.use_rope {
            let (cos, sin) = rope_tables::<B>(seq.min(self.max_seq.max(1)), self.d_k, &q.device());
            let out = apply_rope_qk(q, k, cos, sin);
            q = out.0;
            k = out.1;
        }
        let mut k_all = k.clone();
        let mut v_all = v.clone();
        let use_cache = cache.is_some();
        if let Some(cache) = cache {
            cache.append(k, v);
            if let Some((kc, vc)) = cache.view() {
                k_all = kc;
                v_all = vc;
            }
        }
        let mut attn_scores = q
            .matmul(k_all.transpose())
            .div_scalar((self.d_k as f32).sqrt());
        if !use_cache {
            let mask = burn::nn::attention::generate_autoregressive_mask(batch, seq, &attn_scores.device());
            attn_scores = attn_scores.mask_fill(mask.reshape([batch, 1, seq, seq]), -1.0e4);
        }
        let weights = activation::softmax(attn_scores, 3);
        let context = weights
            .matmul(v_all)
            .swap_dims(1, 2)
            .reshape([batch, seq, d_model]);
        self.o_proj.forward(context)
    }
}
