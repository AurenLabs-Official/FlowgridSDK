//! Hand-crafted async OpenAI API client (inspired by openai-node layout).

#![allow(missing_docs)]
#![allow(clippy::result_large_err)]

pub mod client;
pub mod error;
pub mod pagination;
pub mod resources;
pub mod transport;

#[cfg(any(feature = "files", feature = "images", feature = "audio"))]
pub mod multipart;

#[cfg(feature = "azure")]
pub mod azure;

#[cfg(feature = "webhooks")]
pub mod webhooks;

#[cfg(feature = "realtime")]
pub mod realtime;

mod sse;

pub use client::{ClientBuilder, OpenAI, WithResponse};
pub use error::{ApiError, ApiErrorKind, Error, ErrorDetail, ErrorObject, Result};
pub use pagination::ListPage;
pub use transport::{ClientConfig, HttpTransport, ResponseMeta};

pub use resources::{
    BoxedByteStream, ChatClient, ChatCompletion, ChatCompletionChoice, ChatCompletionDeleted,
    ChatCompletionListParams, ChatCompletionMessage, ChatCompletionMessagesListParams,
    ChatCompletionsClient, Completion, CompletionChoice, CompletionsClient,
    CreateChatCompletionRequest, CreateCompletionRequest, CreateEmbeddingRequest,
    CreateEmbeddingResponse, CreateResponseRequest, Embedding, EmbeddingsClient, ResponseDeleted,
    ResponseObject, ResponsesClient,
};

#[cfg(feature = "azure")]
pub use azure::{AzureClientBuilder, AzureOpenAI};

#[cfg(feature = "realtime")]
pub use realtime::{connect as connect_realtime, RealtimeSocket};

#[cfg(feature = "webhooks")]
pub use webhooks::WebhooksClient;
