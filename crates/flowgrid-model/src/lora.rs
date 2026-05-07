//! LoRA linear adapter (`y = base(x) + (alpha/r) * B(A(x))`).
#![allow(missing_docs)]

use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{Config, Module};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Config, Debug)]
pub struct LoraLinearConfig {
    pub r: usize,
    pub alpha: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LoraTarget {
    Q,
    K,
    V,
    O,
    Up,
    Down,
    Gate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraSpec {
    pub r: usize,
    pub alpha: f64,
    pub targets: BTreeSet<LoraTarget>,
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    pub base: Linear<B>,
    pub lora_a: Linear<B>,
    pub lora_b: Linear<B>,
    pub alpha: f64,
    pub r: usize,
}

impl LoraLinearConfig {
    pub fn init<B: Backend>(&self, base: Linear<B>, device: &B::Device) -> LoraLinear<B> {
        let dims = base.weight.val().dims();
        let in_f = dims[0];
        let out_f = dims[1];
        let lora_a = LinearConfig::new(in_f, self.r)
            .with_initializer(Initializer::KaimingUniform {
                gain: 2.0_f64.sqrt(),
                fan_out_only: false,
            })
            .init(device);
        let lora_b = LinearConfig::new(self.r, out_f)
            .with_initializer(Initializer::Zeros)
            .init(device);
        LoraLinear {
            base,
            lora_a,
            lora_b,
            alpha: self.alpha,
            r: self.r,
        }
    }
}

impl<B: Backend> LoraLinear<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y0 = self.base.forward(x.clone());
        let z = self.lora_b.forward(self.lora_a.forward(x));
        let scale = self.alpha / self.r.max(1) as f64;
        y0 + z * scale
    }
}

/// Attach LoRA adapters to a model according to target spec.
///
/// In this phase the API is stabilized while adapters remain no-op wrappers
/// for model-wide attachment (module-level selective injection follows next).
pub fn attach_lora<M>(model: M, _spec: &LoraSpec) -> M {
    model
}

/// Merge LoRA adapters into base model weights.
///
/// The current implementation is identity for non-instrumented modules.
pub fn merge_lora<M>(model: M) -> M {
    model
}
