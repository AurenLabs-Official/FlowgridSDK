use burn::nn::LinearConfig;
use burn::tensor::activation;
use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{Config, Module};

use crate::cache::KvCache;
use crate::lora::{LoraLinear, LoraLinearConfig};
use crate::rope::{apply_rope_qk, rope_tables};

#[derive(Config, Debug)]
pub struct CausalSelfAttnConfig {
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    #[config(default = true)]
    pub use_rope: bool,
    #[config(default = 10_000.0)]
    pub rope_theta: f32,
    pub max_seq: usize,
}

#[derive(Module, Debug)]
pub struct CausalSelfAttn<B: Backend> {
    pub q_proj: LoraLinear<B>,
    pub k_proj: LoraLinear<B>,
    pub v_proj: LoraLinear<B>,
    pub o_proj: LoraLinear<B>,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) d_k: usize,
    pub(crate) use_rope: bool,
    pub(crate) rope_theta: f32,
    pub(crate) max_seq: usize,
}

impl CausalSelfAttnConfig {
    pub fn init<B: Backend>(&self, n_embd: usize, device: &B::Device) -> CausalSelfAttn<B> {
        let disabled = LoraLinearConfig { r: 1, alpha: 0.0 };
        let proj = |n_in, n_out| disabled.init(LinearConfig::new(n_in, n_out).init(device), device);
        let nh = self.n_head.max(1);
        let n_kv = self.n_kv_head.max(1);
        let d_k = self.head_dim.max(1);
        assert!(
            nh % n_kv == 0,
            "n_head ({nh}) must be divisible by n_kv_head ({n_kv})"
        );
        let kv_dim = n_kv * d_k;
        CausalSelfAttn {
            q_proj: proj(n_embd, n_embd),
            k_proj: proj(n_embd, kv_dim),
            v_proj: proj(n_embd, kv_dim),
            o_proj: proj(n_embd, n_embd),
            n_head: nh,
            n_kv_head: n_kv,
            d_k,
            use_rope: self.use_rope,
            rope_theta: self.rope_theta,
            max_seq: self.max_seq,
        }
    }
}

impl<B: Backend> CausalSelfAttn<B> {
    /// Fold LoRA adapter weights into each projection's base; adapters are reset to zero contribution.
    pub fn merge_lora_layers(self, device: &B::Device) -> Self {
        let disabled = LoraLinearConfig { r: 1, alpha: 0.0 };
        Self {
            q_proj: disabled.init(self.q_proj.merged_linear(), device),
            k_proj: disabled.init(self.k_proj.merged_linear(), device),
            v_proj: disabled.init(self.v_proj.merged_linear(), device),
            o_proj: disabled.init(self.o_proj.merged_linear(), device),
            n_head: self.n_head,
            n_kv_head: self.n_kv_head,
            d_k: self.d_k,
            use_rope: self.use_rope,
            rope_theta: self.rope_theta,
            max_seq: self.max_seq,
        }
    }

    pub fn uses_rope(&self) -> bool {
        self.use_rope
    }

    pub fn kv_heads(&self) -> usize {
        self.n_kv_head
    }

    pub fn head_dim(&self) -> usize {
        self.d_k
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
            .reshape([batch, seq, self.n_kv_head, self.d_k])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([batch, seq, self.n_kv_head, self.d_k])
            .swap_dims(1, 2);
        let mut q = q;
        if self.use_rope {
            let pos_off = past_len;
            let (cos, sin) = rope_tables::<B>(pos_off, seq, self.d_k, self.rope_theta, &device);
            let out = apply_rope_qk(q, k, cos, sin);
            q = out.0;
            k = out.1;
        }
        let group = self.n_head / self.n_kv_head;
        let mut k_all = k.clone();
        let mut v_all = v.clone();
        if let Some(cache) = cache {
            cache.append(k, v);
            if let Some((kc, vc)) = cache.view() {
                k_all = kc;
                v_all = vc;
            }
        }
        let expand_kv = |t: Tensor<B, 4>| -> Tensor<B, 4> {
            if group == 1 {
                return t;
            }
            let [b, nk, s, dk] = t.dims();
            t.reshape([b, nk, 1, s, dk])
                .repeat(2, group)
                .reshape([b, self.n_head, s, dk])
        };
        let k_exp = expand_kv(k_all);
        let v_exp = expand_kv(v_all);
        let total_keys = k_exp.dims()[2];
        let mask_bias = decoder_attn_mask_bias::<B>(&device, batch, seq, past_len, total_keys);
        let attn_scores = q
            .matmul(k_exp.swap_dims(2, 3))
            .div_scalar((self.d_k as f32).sqrt())
            + mask_bias;

        let weights = activation::softmax(attn_scores, 3);
        let context = weights
            .matmul(v_exp)
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
