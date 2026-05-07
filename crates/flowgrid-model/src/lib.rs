//! Decoder-only language models and HF weight loaders.
#![allow(missing_docs)]

pub mod attach_lora;
pub mod attention;
pub mod cache;
pub mod hf;
pub mod hf_loader;
pub mod lm;
pub mod lora;
pub mod mlp;
pub mod nano_gpt;
pub mod norm;
pub mod rope;
pub mod sampler;

pub use attach_lora::attach_lora;
pub use hf::gpt2::load_gpt2_into_nano_gpt;
pub use hf_loader::{decode_weight_tensor_f32_le, load_safetensors_typed, SafetensorsTensorRecord};
pub use lm::LmModel;
pub use nano_gpt::{NanoGpt, NanoGptConfig};
pub use sampler::{sample_from_last_logits, Sampling};
