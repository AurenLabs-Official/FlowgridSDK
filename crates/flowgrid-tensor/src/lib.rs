//! Tensor prelude: re-exports from **Burn** and a small error type.
//!
//! The workspace enables **CPU** (`ndarray`) by default. Opt into GPU features on this crate:
//! `wgpu`, `cuda`, `tch`, `metal`, or `candle` (maps to Burn optional backends).

pub mod obs;

pub use burn::config::Config;
pub use burn::module::Module;
pub use burn::record::FullPrecisionSettings;
pub use burn::tensor::ElementConversion;
pub use burn::tensor::{
    activation::relu,
    backend::{AutodiffBackend, Backend},
    Tensor,
};

use std::fmt;

/// Crate-local recoverable error (tokenizer/data/model layers wrap this or [`anyhow`]).
#[derive(Debug, Clone, thiserror::Error)]
pub enum FgError {
    /// User misconfiguration.
    #[error("config: {0}")]
    Config(String),
    /// I/O or parse failure.
    #[error("io: {0}")]
    Io(String),
    /// Tensor / module shape or runtime check.
    #[error("shape: {0}")]
    Shape(String),
}

impl FgError {
    pub fn config(msg: impl fmt::Display) -> Self {
        FgError::Config(msg.to_string())
    }
    pub fn io(msg: impl fmt::Display) -> Self {
        FgError::Io(msg.to_string())
    }
    pub fn shape(msg: impl fmt::Display) -> Self {
        FgError::Shape(msg.to_string())
    }
}

pub type FgResult<T> = Result<T, FgError>;
