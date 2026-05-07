//! Forward pass sanity on tiny config (init is non-deterministic across Burn versions — avoid golden u64).

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use flowgrid_model::NanoGptConfig;

type B = NdArray<f32>;

#[test]
fn tiny_forward_logits_finite_and_sized() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let cfg = NanoGptConfig {
        vocab_size: 32,
        block_size: 8,
        n_layer: 1,
        n_head: 4,
        n_kv_head: 0,
        n_embd: 16,
        dropout: 0.0,
        use_rope: true,
        rope_theta: 10_000.0,
    };
    let model = cfg.init::<B>(&device);
    let inp = Tensor::<B, 1, Int>::from_ints([1i32, 2, 3, 4], &device).reshape([1, 4]);
    let logits = model.forward(inp);
    let flat = logits.into_data().convert::<f32>().value;
    assert_eq!(flat.len(), 4 * cfg.vocab_size);
    assert!(flat.iter().all(|x| x.is_finite()));
    let m: f32 = flat.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(m > 0.0);
}
