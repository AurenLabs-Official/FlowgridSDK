use burn::backend::NdArray;
use burn::tensor::Int;
use burn::tensor::Tensor;
use flowgrid_model::NanoGptConfig;

type B = NdArray<f32>;

#[test]
fn merge_lora_adapters_preserves_forward_when_adapter_zero() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let cfg = NanoGptConfig {
        vocab_size: 24,
        block_size: 8,
        n_layer: 1,
        n_head: 2,
        n_embd: 8,
        dropout: 0.0,
        use_rope: true,
        rope_theta: 10_000.0,
    };
    let model = cfg.init::<B>(&device);
    let inp = Tensor::<B, 2, Int>::from_ints([[1i32, 2, 3]], &device);
    let y0 = model.forward(inp.clone());
    let merged = model.merge_lora_adapters(&device);
    let y1 = merged.forward(inp);
    y0.to_data().assert_approx_eq(&y1.to_data(), 4);
}
