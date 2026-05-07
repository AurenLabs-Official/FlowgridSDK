use burn::nn::{LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig};
use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{Config, Module};

#[derive(Config, Debug)]
pub struct NormConfig {
    pub d_model: usize,
    #[config(default = false)]
    pub use_rms_norm: bool,
}

#[derive(Module, Debug)]
pub enum Norm<B: Backend> {
    Layer(LayerNorm<B>),
    Rms(RmsNorm<B>),
}

impl NormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Norm<B> {
        if self.use_rms_norm {
            Norm::Rms(RmsNormConfig::new(self.d_model).init(device))
        } else {
            Norm::Layer(LayerNormConfig::new(self.d_model).init(device))
        }
    }
}

impl<B: Backend> Norm<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Layer(m) => m.forward(x),
            Self::Rms(m) => m.forward(x),
        }
    }
}
