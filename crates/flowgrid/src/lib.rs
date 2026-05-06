//! Unified Flowgrid SDK for OpenAI and/or Anthropic HTTP APIs.
//!
//! The **stable** surface for semver is the `pub use` items at this crate root. The `internal`
//! module tree (`internal::oai`, `internal::clu`, etc.) is private implementation detail and may
//! change in minor releases without a major version bump.
//!
//! Enable providers with Cargo features `openai` and `anthropic` (both on by default). With **both**
//! enabled, types that would clash are exported with `OpenAi*` / `Anthropic*` prefixes (for example
//! [`OpenAiError`] and [`AnthropicError`]). If you build with only one provider, [`Error`] and
//! [`Result`] refer to that provider.
//!
//! **Streaming:** chat completions, Responses API, and Anthropic messages support SSE via
//! `create_stream` on the respective clients. Decoders expose `next_event` and
//! `into_event_stream` for manual loops vs. `futures::StreamExt`. OpenAI may send a final
//! `data: [DONE]` line; parse event payloads defensively.

#![allow(missing_docs)]
#![allow(clippy::result_large_err)]

mod internal;

// ---- OpenAI + Anthropic both: prefixed collisions ----

#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::clu::WithResponse as AnthropicWithResponse;
#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::oai::WithResponse as OpenAiWithResponse;

#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::oai::ApiError as OpenAiApiError;
#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::oai::ApiErrorKind as OpenAiApiErrorKind;
#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::oai::Error as OpenAiError;
#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::oai::Result as OpenAiResult;

#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::clu::ApiError as AnthropicApiError;
#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::clu::ApiErrorKind as AnthropicApiErrorKind;
#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::clu::Error as AnthropicError;
#[cfg(all(feature = "openai", feature = "anthropic"))]
pub use internal::clu::Result as AnthropicResult;

#[cfg(feature = "openai")]
pub type OpenAiClientConfig = internal::oai::ClientConfig;
#[cfg(feature = "openai")]
pub type OpenAiHttpTransport = internal::oai::HttpTransport;
#[cfg(feature = "openai")]
pub type OpenAiResponseMeta = internal::oai::ResponseMeta;

#[cfg(feature = "anthropic")]
pub type AnthropicClientConfig = internal::clu::ClientConfig;
#[cfg(feature = "anthropic")]
pub type AnthropicHttpTransport = internal::clu::HttpTransport;
#[cfg(feature = "anthropic")]
pub type AnthropicResponseMeta = internal::clu::ResponseMeta;

#[cfg(feature = "openai")]
pub type OpenAiBoxedByteStream = internal::oai::BoxedByteStream;
#[cfg(feature = "anthropic")]
pub type AnthropicBoxedByteStream = internal::clu::BoxedByteStream;

// ---- Single provider: short names for collisions ----

#[cfg(all(feature = "openai", not(feature = "anthropic")))]
pub use internal::oai::{
    ApiError, ApiErrorKind, BoxedByteStream, ClientConfig, Error, HttpTransport, ResponseMeta,
    Result, WithResponse,
};

#[cfg(all(feature = "anthropic", not(feature = "openai")))]
pub use internal::clu::{
    ApiError, ApiErrorKind, BoxedByteStream, ClientConfig, Error, HttpTransport, ResponseMeta,
    Result, WithResponse,
};

// ---- OpenAI ----

#[cfg(feature = "openai")]
pub use internal::oai::{ClientBuilder, OpenAI};

#[cfg(feature = "openai")]
pub use internal::oai::{ErrorDetail, ErrorObject, ListPage};

#[cfg(feature = "openai")]
pub use internal::oai::{
    ChatClient, ChatCompletion, ChatCompletionChoice, ChatCompletionDeleted,
    ChatCompletionListParams, ChatCompletionMessage, ChatCompletionMessagesListParams,
    ChatCompletionsClient, Completion, CompletionChoice, CompletionsClient,
    CreateChatCompletionRequest, CreateCompletionRequest, CreateEmbeddingRequest,
    CreateEmbeddingResponse, CreateResponseRequest, Embedding, EmbeddingsClient, ResponseDeleted,
    ResponseObject, ResponsesClient,
};

#[cfg(all(feature = "openai", feature = "azure"))]
pub use internal::oai::{AzureClientBuilder, AzureOpenAI};

#[cfg(all(feature = "openai", feature = "realtime"))]
pub use internal::oai::{connect_realtime, RealtimeSocket};

#[cfg(all(feature = "openai", feature = "webhooks"))]
pub use internal::oai::WebhooksClient;

// ---- Anthropic ----

#[cfg(feature = "anthropic")]
pub use internal::clu::{Anthropic, AnthropicBuilder};

#[cfg(feature = "anthropic")]
pub use internal::clu::{
    CountMessageTokensRequest, CreateMessageRequest, Message, MessageTokensCount, MessagesClient,
};

#[cfg(feature = "anthropic")]
pub use internal::clu::ErrorBody;

#[cfg(all(feature = "anthropic", feature = "batches"))]
pub use internal::clu::MessageBatchesClient;

#[cfg(all(feature = "anthropic", feature = "models"))]
pub use internal::clu::ModelsClient;

#[cfg(all(feature = "anthropic", feature = "beta"))]
pub use internal::clu::BetaClient;
