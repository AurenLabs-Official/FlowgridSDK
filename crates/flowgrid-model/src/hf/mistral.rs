use std::collections::HashMap;

use crate::hf_loader::{validate_expected_keys, HfArch};
use flowgrid_tensor::FgResult;

/// Validate keys for a minimal Mistral-shaped export (preview — tensor load into `NanoGpt` is WIP).
pub fn validate_mistral_keys(tensors: &HashMap<String, Vec<u8>>) -> FgResult<()> {
    validate_expected_keys(tensors, HfArch::Mistral)
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
        "model.layers.0.mlp.down_proj.weight",
    ]
}
