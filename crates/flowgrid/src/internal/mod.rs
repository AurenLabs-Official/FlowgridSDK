//! Private implementation code; the stable surface is the `flowgrid` crate root.
//!
//! **Imports:** Prefer OpenAI-specific types and helpers via `crate::internal::oai::…` (and
//! Anthropic via `crate::internal::clu::…`) instead of importing sibling modules such as
//! `crate::internal::pagination` directly, except inside barrel files like `oai.rs`.

mod client;
pub mod error;
pub mod execute_options;
pub mod resources;
mod retry_policy;
mod sse;
mod stream_types;
#[cfg(all(feature = "openai", feature = "stream-types"))]
mod stream_typing;
#[cfg(all(feature = "anthropic", feature = "stream-types"))]
mod stream_typing_clu;
mod transport;

#[cfg(feature = "opentelemetry")]
mod otel_http;

#[cfg(feature = "openai")]
mod pagination;

// `#[cfg]` for OpenAI-only helper modules below must stay in sync with the matching `pub mod …`
// shims in `oai.rs` (same predicates; `oai` is only built with `feature = "openai"`).

#[cfg(all(
    feature = "openai",
    any(feature = "files", feature = "images", feature = "audio")
))]
mod multipart;

#[cfg(all(feature = "openai", feature = "azure"))]
mod azure;

#[cfg(all(feature = "openai", feature = "webhooks"))]
mod webhooks;

#[cfg(all(feature = "openai", feature = "realtime"))]
mod realtime;

#[cfg(feature = "openai")]
pub mod oai;

#[cfg(feature = "anthropic")]
pub mod clu;
