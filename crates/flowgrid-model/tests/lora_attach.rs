//! LoRA attach wraps targeted projections with a non-zero rank adapter.

use burn::backend::NdArray;
use flowgrid_model::lora::{attach_lora, LoraSpec, LoraTarget};
use flowgrid_model::NanoGptConfig;
use std::collections::BTreeSet;

type B = NdArray<f32>;

#[test]
fn attach_lora_changes_q_rank() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let cfg = NanoGptConfig {
        vocab_size: 32,
        block_size: 8,
        n_layer: 1,
        n_head: 2,
        n_embd: 16,
        dropout: 0.0,
        use_rope: true,
        rope_theta: 10_000.0,
    };
    let model = cfg.init::<B>(&device);
    assert_eq!(model.blocks[0].attn.q_proj.r, 1);

    let mut targets = BTreeSet::new();
    targets.insert(LoraTarget::Q);
    let spec = LoraSpec {
        r: 4,
        alpha: 8.0,
        targets,
        dropout: 0.0,
    };
    let adapted = attach_lora(model, &spec);
    assert_eq!(adapted.blocks[0].attn.q_proj.r, 4);
    assert_eq!(adapted.blocks[0].attn.k_proj.r, 1);
}

#[test]
fn attach_lora_empty_targets_is_noop() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let cfg = NanoGptConfig {
        vocab_size: 16,
        block_size: 8,
        n_layer: 1,
        n_head: 2,
        n_embd: 8,
        dropout: 0.0,
        use_rope: true,
        rope_theta: 10_000.0,
    };
    let model = cfg.init::<B>(&device);
    let spec = LoraSpec {
        r: 16,
        alpha: 32.0,
        targets: BTreeSet::new(),
        dropout: 0.0,
    };
    let adapted = attach_lora(model, &spec);
    assert_eq!(adapted.blocks[0].attn.q_proj.r, 1);
    assert_eq!(adapted.blocks[0].mlp.up.r, 1);
}
