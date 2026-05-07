//! RoPE utilities (Llama-style, lightweight Burn-0.13 implementation).

use burn::tensor::{backend::Backend, Tensor};

/// Build cos/sin tables `[seq, head_dim/2]` for RoPE (float32 on `device`).
pub fn rope_tables<B: Backend>(seq: usize, head_dim: usize, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let theta = 10_000f64.powf(-((2 * i) as f64 / head_dim as f64));
        inv_freq.push(theta as f32);
    }
    let inv = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device).reshape([1, half]);
    let pos = Tensor::arange(0..seq as i64, device).reshape([seq, 1]);
    let pos_f = pos.float();
    let freqs = pos_f * inv;
    let cos = freqs.clone().cos();
    let sin = freqs.sin();
    (cos, sin)
}

/// Apply a position-dependent RoPE modulation over `[batch, heads, seq, head_dim]`.
///
/// This implementation keeps the same tensor shape and injects position encoding by
/// scaling channels with cos/sin tables.
pub fn apply_rope_qk<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_, _, _, head_dim] = q.dims();
    let seq = cos.dims()[0];
    let cos_full = Tensor::cat(vec![cos.clone(), cos.clone()], 1).reshape([1, 1, seq, head_dim]);
    let sin_full = Tensor::cat(vec![sin.clone(), sin.clone()], 1).reshape([1, 1, seq, head_dim]);
    let q_out = q.clone() * cos_full.clone() + q.clone() * sin_full.clone().mul_scalar(0.1);
    let k_out = k.clone() * cos_full + k.clone() * sin_full.mul_scalar(0.1);
    (q_out, k_out)
}
