use std::collections::HashMap;

use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::FgResult;

use crate::hf::gpt2::tensor_f2_from_bytes;
use crate::hf_loader::{validate_expected_keys, HfArch};

/// Validate that a single-layer Llama-style export contains core projection keys (preview).
pub fn validate_llama_keys(tensors: &HashMap<String, Vec<u8>>) -> FgResult<()> {
    validate_expected_keys(tensors, HfArch::Llama)
}

pub fn expected_keys() -> Vec<&'static str> {
    vec![
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
}

/// Decode HF `model.layers.{i}.self_attn.q_proj.weight` as rank-2 **`[n_embd, n_embd]`** (standard Llama layout, F32/BF16/F16 bytes).
///
/// This is the first staged fidelity step toward full weight import; callers can assign the tensor into a compatible `Linear` / `LoraLinear` base.
pub fn decode_self_attn_q_proj<B: Backend>(
    dtype: &str,
    data: &[u8],
    n_embd: usize,
    device: &B::Device,
) -> FgResult<Tensor<B, 2>> {
    tensor_f2_from_bytes::<B>("llama.q_proj", dtype, data, &[n_embd, n_embd], device)
}

/// Decode HF `k_proj.weight` / `v_proj.weight` as rank-2 **`[n_kv_head * head_dim, n_embd]`** (Llama GQA).
pub fn decode_self_attn_kv_proj<B: Backend>(
    dtype: &str,
    data: &[u8],
    n_kv_head: usize,
    head_dim: usize,
    n_embd: usize,
    device: &B::Device,
) -> FgResult<Tensor<B, 2>> {
    let rows = (n_kv_head * head_dim).max(1);
    tensor_f2_from_bytes::<B>("llama.kv_proj", dtype, data, &[rows, n_embd], device)
}
