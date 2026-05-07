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
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    n_head: usize,
    d_k: usize,
    use_rope: bool,
    rope_theta: f32,
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
            rope_theta: self.rope_theta,
            max_seq: self.max_seq,
        }
    }
}

impl<B: Backend> CausalSelfAttn<B> {
    pub fn uses_rope(&self) -> bool {
        self.use_rope
    }

    pub fn forward(&self, x: Tensor<B, 3>, cache: Option<&mut KvCache<B>>) -> Tensor<B, 3> {
        let [batch, seq, d_model] = x.dims();
        let device = x.device();
        let past_len = cache
            .as_ref()
            .and_then(|c| c.view())
            .map(|(k, _)| k.dims()[2])
            .unwrap_or(0);

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
            let rope_seq = seq.min(self.max_seq.max(1));
            let pos_off = past_len;
            let (cos, sin) =
                rope_tables::<B>(pos_off, rope_seq, self.d_k, self.rope_theta, &device);
            let out = apply_rope_qk(q, k, cos, sin);
            q = out.0;
            k = out.1;
        }
        let mut k_all = k.clone();
        let mut v_all = v.clone();
        if let Some(cache) = cache {
            cache.append(k, v);
            if let Some((kc, vc)) = cache.view() {
                k_all = kc;
                v_all = vc;
            }
        }
        let total_keys = k_all.dims()[2];
        let mask_bias = decoder_attn_mask_bias::<B>(&device, batch, seq, past_len, total_keys);
        let attn_scores = q
            .matmul(k_all.swap_dims(2, 3))
            .div_scalar((self.d_k as f32).sqrt())
            + mask_bias;

        let weights = activation::softmax(attn_scores, 3);
        let context = weights
            .matmul(v_all)
            .swap_dims(1, 2)
            .reshape([batch, seq, d_model]);
        self.o_proj.forward(context)
    }
}

/// Large negative bias where attention must be masked out (before softmax).
fn decoder_attn_mask_bias<B: Backend>(
    device: &B::Device,
    batch: usize,
    seq: usize,
    past_len: usize,
    total_keys: usize,
) -> Tensor<B, 4> {
    let mut data = Vec::with_capacity(batch * seq * total_keys);
    for _ in 0..batch {
        for i in 0..seq {
            for kj in 0..total_keys {
                let allowed = kj < past_len + i + 1;
                data.push(if allowed { 0.0f32 } else { -1.0e4f32 });
            }
        }
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([batch, 1, seq, total_keys])
}
