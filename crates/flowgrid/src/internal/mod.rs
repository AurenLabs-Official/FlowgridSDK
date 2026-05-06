//! Private implementation code; the stable surface is the `flowgrid` crate root.

mod client;
mod error;
pub mod resources;
mod sse;
mod transport;

#[cfg(feature = "openai")]
mod pagination;

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
