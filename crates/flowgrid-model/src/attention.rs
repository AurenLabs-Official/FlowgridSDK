use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{Config, Module};

use crate::cache::KvCache;

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
}

impl CausalSelfAttnConfig {
    pub fn init<B: Backend>(&self, n_embd: usize, device: &B::Device) -> CausalSelfAttn<B> {
        CausalSelfAttn {
            q_proj: LinearConfig::new(n_embd, n_embd).init(device),
            k_proj: LinearConfig::new(n_embd, n_embd).init(device),
            v_proj: LinearConfig::new(n_embd, n_embd).init(device),
            o_proj: LinearConfig::new(n_embd, n_embd).init(device),
        }
    }
}

impl<B: Backend> CausalSelfAttn<B> {
    /// Conservative, Burn-0.13-friendly attention path.
    ///
    /// The current implementation keeps full q/k/v projections and cache plumbed;
    /// softmax masking and full family-specific attention kernels are added in follow-up patches.
    pub fn forward(&self, x: Tensor<B, 3>, cache: Option<&mut KvCache<B>>) -> Tensor<B, 3> {
        let _q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);
        if let Some(cache) = cache {
            cache.append(k, v.clone());
        }
        self.o_proj.forward(v)
    }
}
