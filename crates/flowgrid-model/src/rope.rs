//! RoPE utilities (Llama-style, lightweight Burn-0.13 implementation).

use burn::tensor::{backend::Backend, Tensor};

/// Build cos/sin tables `[seq, head_dim/2]` for RoPE at absolute positions
/// `pos_offset..pos_offset+seq` (inclusive lower bound, exclusive upper on the arange input).
pub fn rope_tables<B: Backend>(
    pos_offset: usize,
    seq: usize,
    head_dim: usize,
    theta: f32,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let exp = -((2 * i) as f32 / head_dim as f32);
        inv_freq.push(theta.powf(exp));
    }
    let inv = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device).reshape([1, half]);
    let pos =
        Tensor::arange(pos_offset as i64..(pos_offset + seq) as i64, device).reshape([seq, 1]);
    let pos_f = pos.float();
    let freqs = pos_f * inv;
    let cos = freqs.clone().cos();
    let sin = freqs.sin();
    (cos, sin)
}

/// Apply rotary position embeddings over `[batch, heads, seq, head_dim]` (`head_dim` even).
pub fn apply_rope_qk<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, n_head_q, seq, head_dim] = q.dims();
    let [batch_k, n_head_k, seq_k, head_dim_k] = k.dims();
    debug_assert_eq!(batch, batch_k);
    debug_assert_eq!(seq, seq_k);
    debug_assert_eq!(head_dim, head_dim_k);
    let half = head_dim / 2;
    let cos_b = cos.clone().reshape([1, 1, seq, half]);
    let sin_b = sin.clone().reshape([1, 1, seq, half]);

    let q1 = q.clone().slice([0..batch, 0..n_head_q, 0..seq, 0..half]);
    let q2 = q.slice([0..batch, 0..n_head_q, 0..seq, half..head_dim]);
    let q_out = Tensor::cat(
        vec![
            q1.clone() * cos_b.clone() - q2.clone() * sin_b.clone(),
            q1 * sin_b.clone() + q2 * cos_b.clone(),
        ],
        3,
    );

    let k1 = k
        .clone()
        .slice([0..batch_k, 0..n_head_k, 0..seq_k, 0..half]);
    let k2 = k.slice([0..batch_k, 0..n_head_k, 0..seq_k, half..head_dim]);
    let k_out = Tensor::cat(
        vec![
            k1.clone() * cos_b.clone() - k2.clone() * sin_b.clone(),
            k1 * sin_b + k2 * cos_b,
        ],
        3,
    );

    (q_out, k_out)
}
