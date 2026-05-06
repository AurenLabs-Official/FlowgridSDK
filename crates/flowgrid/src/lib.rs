//! Unified Flowgrid SDK: enable `openai` and/or `anthropic` features (both enabled by default).
//!
//! - `flowgrid::openai::*` — OpenAI HTTP API
//! - `flowgrid::anthropic::*` — Anthropic Messages API

#[cfg(feature = "openai")]
pub mod openai {
    //! Re-export of [`flowgrid_openai`].
    pub use flowgrid_openai::*;
}

#[cfg(feature = "anthropic")]
pub mod anthropic {
    //! Re-export of [`flowgrid_anthropic`].
    pub use flowgrid_anthropic::*;
}
