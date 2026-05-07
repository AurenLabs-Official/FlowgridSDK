use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::Module;

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    up: Linear<B>,
    down: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn new(n_embd: usize, dropout: f64, device: &B::Device) -> Self {
        Self {
            up: LinearConfig::new(n_embd, 4 * n_embd).init(device),
            down: LinearConfig::new(4 * n_embd, n_embd).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = gelu(self.up.forward(x.clone()));
        let h = self.dropout.forward(h);
        let h = self.down.forward(h);
        x + self.dropout.forward(h)
    }
}
