use burn::backend::NdArray;
use burn::nn::{Initializer, LinearConfig};
use burn::tensor::Tensor;
use flowgrid_model::lora::LoraLinearConfig;

type B = NdArray<f32>;

#[test]
fn merged_linear_matches_lora_forward() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let base = LinearConfig::new(4, 4)
        .with_initializer(Initializer::KaimingUniform {
            gain: 1.0,
            fan_out_only: false,
        })
        .init::<B>(&device);
    let mut layer = LoraLinearConfig::new(2, 8.0).init(base, &device);

    let a = Tensor::<B, 2>::from_floats([[0.1, -0.2], [0.3, 0.4], [0.2, -0.1], [0.0, 0.5]], &device);
    let b = Tensor::<B, 2>::from_floats([[0.2, -0.3, 0.1, 0.0], [0.4, 0.2, -0.2, 0.1]], &device);
    layer.lora_a.weight = layer.lora_a.weight.map(|_| a.clone());
    layer.lora_b.weight = layer.lora_b.weight.map(|_| b.clone());

    let x = Tensor::<B, 3>::from_floats([[[0.5, 0.1, -0.3, 0.2]]], &device);
    let y_lora = layer.forward(x.clone());
    let merged = layer.merged_linear();
    let y_merged = merged.forward(x);
    y_lora.to_data().assert_approx_eq(&y_merged.to_data(), 3);
}
