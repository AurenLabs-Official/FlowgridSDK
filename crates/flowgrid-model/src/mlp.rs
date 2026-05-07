use burn::nn::{Dropout, DropoutConfig, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::Module;

use crate::lora::{LoraLinear, LoraLinearConfig};

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub up: LoraLinear<B>,
    pub down: LoraLinear<B>,
    pub(crate) dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn new(n_embd: usize, dropout: f64, device: &B::Device) -> Self {
        let disabled = LoraLinearConfig { r: 1, alpha: 0.0 };
        Self {
            up: disabled.init(LinearConfig::new(n_embd, 4 * n_embd).init(device), device),
            down: disabled.init(LinearConfig::new(4 * n_embd, n_embd).init(device), device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = gelu(self.up.forward(x));
        let h = self.dropout.forward(h);
        self.down.forward(h)
    }
}
