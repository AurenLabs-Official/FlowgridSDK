//! Hand-crafted async Anthropic Claude API client (Python SDK / `api.md` parity).

pub use crate::internal::client::clu::{Anthropic, AnthropicBuilder, WithResponse};
pub use crate::internal::error::clu::{ApiError, ApiErrorKind, Error, ErrorBody, Result};
pub use crate::internal::transport::clu::{ClientConfig, HttpTransport, ResponseMeta};

#[cfg(feature = "stream-types")]
pub use crate::internal::stream_typing_clu::{
    parse_anthropic_message_stream_json, AnthropicContentBlockDeltaEvent, AnthropicStreamLine,
    AnthropicTextDelta,
};

pub use crate::internal::resources::messages::{
    BoxedByteStream, CountMessageTokensRequest, CreateMessageRequest, Message, MessageTokensCount,
    MessagesClient,
};

#[cfg(feature = "batches")]
pub use crate::internal::resources::MessageBatchesClient;

#[cfg(feature = "models")]
pub use crate::internal::resources::models::ModelsClient;

#[cfg(feature = "beta")]
pub use crate::internal::resources::beta::BetaClient;
