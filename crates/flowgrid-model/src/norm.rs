use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{Config, Module};

#[derive(Config, Debug)]
pub struct NormConfig {}

#[derive(Module, Debug, Clone)]
pub struct Norm {}

impl NormConfig {
    pub fn init(&self) -> Norm {
        Norm {}
    }
}

impl Norm {
    pub fn forward<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x
    }
}
