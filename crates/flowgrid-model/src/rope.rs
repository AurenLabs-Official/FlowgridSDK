//! RoPE utilities (Llama-style).

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
