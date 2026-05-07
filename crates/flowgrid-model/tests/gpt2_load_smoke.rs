//! Micro GPT-2-shaped tensor map → [`NanoGpt`] smoke test (no repo-sized fixture).

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use flowgrid_model::hf::gpt2::load_gpt2_into_nano_gpt;
use flowgrid_model::NanoGptConfig;
use std::collections::HashMap;

type B = NdArray<f32>;

fn f32_bytes(t: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(t.len() * 4);
    for x in t {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

#[test]
fn gpt2_weight_map_loads_and_runs_forward() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let v = 12usize;
    let p = 8usize;
    let d = 16usize;
    let nh = 1usize;
    let cfg = NanoGptConfig {
        vocab_size: v,
        block_size: p,
        n_layer: 1,
        n_head: nh,
        n_embd: d,
        dropout: 0.0,
        use_rope: false,
        rope_theta: 10_000.0,
    };
    let mut model = cfg.init::<B>(&device);

    let wte: Vec<f32> = (0..(v * d)).map(|i| (i as f32) * 1e-4).collect();
    let wpe: Vec<f32> = (0..(p * d)).map(|i| (i as f32) * 1e-4).collect();
    let lm: Vec<f32> = (0..(d * v)).map(|i| (i as f32) * 1e-4).collect();

    let c_attn_kqv: Vec<f32> = (0..(d * 3 * d)).map(|i| (i as f32) * 1e-5).collect();
    let c_attn_b: Vec<f32> = (0..(3 * d)).map(|i| (i as f32) * 1e-5).collect();
    let c_proj_w: Vec<f32> = (0..(d * d)).map(|i| (i as f32) * 1e-5).collect();
    let c_proj_b: Vec<f32> = (0..d).map(|i| (i as f32) * 1e-5).collect();
    let fc_w: Vec<f32> = (0..(4 * d * d)).map(|i| (i as f32) * 1e-5).collect();
    let fc_b: Vec<f32> = (0..(4 * d)).map(|i| (i as f32) * 1e-5).collect();
    let pj_w: Vec<f32> = (0..(d * 4 * d)).map(|i| (i as f32) * 1e-5).collect();
    let pj_b: Vec<f32> = (0..d).map(|i| (i as f32) * 1e-5).collect();

    let mut named = HashMap::new();
    named.insert(
        "transformer.wte.weight".into(),
        ("F32".into(), vec![v, d], f32_bytes(&wte)),
    );
    named.insert(
        "transformer.wpe.weight".into(),
        ("F32".into(), vec![p, d], f32_bytes(&wpe)),
    );
    named.insert(
        "lm_head.weight".into(),
        ("F32".into(), vec![d, v], f32_bytes(&lm)),
    );
    // HF Conv1D: `[nx, nf]` → `[d, 3 * d]` for fused QKV.
    named.insert(
        "transformer.h.0.attn.c_attn.weight".into(),
        ("F32".into(), vec![d, 3 * d], f32_bytes(&c_attn_kqv)),
    );
    named.insert(
        "transformer.h.0.attn.c_attn.bias".into(),
        ("F32".into(), vec![3 * d], f32_bytes(&c_attn_b)),
    );
    named.insert(
        "transformer.h.0.attn.c_proj.weight".into(),
        ("F32".into(), vec![d, d], f32_bytes(&c_proj_w)),
    );
    named.insert(
        "transformer.h.0.attn.c_proj.bias".into(),
        ("F32".into(), vec![d], f32_bytes(&c_proj_b)),
    );
    named.insert(
        "transformer.h.0.mlp.c_fc.weight".into(),
        ("F32".into(), vec![d, 4 * d], f32_bytes(&fc_w)),
    );
    named.insert(
        "transformer.h.0.mlp.c_fc.bias".into(),
        ("F32".into(), vec![4 * d], f32_bytes(&fc_b)),
    );
    named.insert(
        "transformer.h.0.mlp.c_proj.weight".into(),
        ("F32".into(), vec![4 * d, d], f32_bytes(&pj_w)),
    );
    named.insert(
        "transformer.h.0.mlp.c_proj.bias".into(),
        ("F32".into(), vec![d], f32_bytes(&pj_b)),
    );

    load_gpt2_into_nano_gpt(&mut model, &named, d, nh, &device).unwrap();
    let inp = Tensor::<B, 1, Int>::from_ints([1i32, 2, 3], &device).reshape([1, 3]);
    let logits = model.forward(inp);
    let flat = logits.into_data().convert::<f32>().value;
    let sum: f32 = flat.iter().sum();
    assert!(sum.is_finite() && sum.abs() > 1e-6);
}
