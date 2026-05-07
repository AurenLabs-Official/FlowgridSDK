use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use flowgrid_model::cache::KvCacheStack;
use flowgrid_model::{LmModel, NanoGptConfig};

type B = NdArray<f32>;

#[test]
fn forward_and_cache_step_have_close_last_token_logits() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let cfg = NanoGptConfig {
        vocab_size: 64,
        block_size: 16,
        n_layer: 2,
        n_head: 4,
        n_kv_head: 0,
        n_embd: 32,
        dropout: 0.0,
        use_rope: true,
        rope_theta: 10_000.0,
    };
    let model = cfg.init::<B>(&device);
    let ids: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
    let full = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, ids.len()]);
    let logits_full = model.forward(full);

    let mut cache: KvCacheStack<B> = Vec::new();
    let mut logits_step_last = None;
    for id in &ids {
        let t = Tensor::<B, 1, Int>::from_ints([*id], &device).reshape([1, 1]);
        logits_step_last = Some(model.forward_step(t, Some(&mut cache)));
    }
    let logits_step = logits_step_last.expect("step logits");

    let a = logits_full
        .slice([0..1, (ids.len() - 1)..ids.len(), 0..cfg.vocab_size])
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
        assert!((x - y).abs() < 1e-3, "cache mismatch: {x} vs {y}");
    }
}
