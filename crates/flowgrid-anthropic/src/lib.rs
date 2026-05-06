//! Hand-crafted async Anthropic Claude API client (Python SDK / `api.md` parity).

#![allow(missing_docs)]
#![allow(clippy::result_large_err)]

pub mod client;
pub mod error;
pub mod resources;
mod sse;
pub mod transport;

pub use client::{Anthropic, AnthropicBuilder, WithResponse};
pub use error::{ApiError, ApiErrorKind, Error, ErrorBody, Result};
pub use transport::{ClientConfig, HttpTransport, ResponseMeta};

pub use resources::messages::{
    BoxedByteStream, CountMessageTokensRequest, CreateMessageRequest, Message, MessageTokensCount,
    MessagesClient,
};

#[cfg(feature = "batches")]
pub use resources::batches::MessageBatchesClient;

#[cfg(feature = "models")]
pub use resources::models::ModelsClient;

#[cfg(feature = "beta")]
pub use resources::beta::BetaClient;
