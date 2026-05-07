use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use flowgrid_model::{sample_from_last_logits, NanoGptConfig, Sampling};
use num_traits::ToPrimitive;
use rand::thread_rng;

type B = NdArray<f32>;

#[test]
fn greedy_sampler_matches_argmax() {
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
    let ids: Vec<i32> = vec![1, 2, 3];
    let inp = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, ids.len()]);
    let mut rng = thread_rng();
    let logits = model.forward(inp);
    let row = logits
        .clone()
        .slice([0..1, (ids.len() - 1)..ids.len(), 0..cfg.vocab_size]);
    let argmax = row.argmax(2).into_scalar();
    let g = sample_from_last_logits(&logits, Sampling::Greedy, &mut rng);
    assert_eq!(g, argmax.to_i32().unwrap_or(-1));
}
