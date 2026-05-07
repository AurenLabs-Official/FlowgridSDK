use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use flowgrid_model::cache::KvCacheStack;
use flowgrid_model::{LmModel, NanoGptConfig};

type B = NdArray<f32>;

#[test]
fn forward_step_matches_full_sequence_logits() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let cfg = NanoGptConfig {
        vocab_size: 64,
        block_size: 16,
        n_layer: 2,
        n_head: 4,
        n_embd: 32,
        dropout: 0.0,
        use_rope: true,
        rope_theta: 10_000.0,
    };
    let model = cfg.init::<B>(&device);
    let ids: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7];
    let full = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, ids.len()]);
    let logits_full = model.forward(full.clone());

    let mut cache: KvCacheStack<B> = Vec::new();
    for (step, id) in ids.iter().enumerate() {
        let t = Tensor::<B, 1, Int>::from_ints([*id], &device).reshape([1, 1]);
        let logits_step = model.forward_step(t, Some(&mut cache));
        let a = logits_full
            .clone()
            .slice([0..1, step..(step + 1), 0..cfg.vocab_size])
            .into_data()
            .convert::<f32>()
            .value;
        let b = logits_step
            .slice([0..1, 0..1, 0..cfg.vocab_size])
            .into_data()
            .convert::<f32>()
            .value;
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-3, "step {step}: {x} vs {y}");
        }
    }
}
