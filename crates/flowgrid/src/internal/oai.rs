//! Hand-crafted async OpenAI API client (inspired by openai-node layout).

pub mod pagination {
    pub use crate::internal::pagination::*;
}

#[cfg(any(feature = "files", feature = "images", feature = "audio"))]
pub mod multipart {
    pub use crate::internal::multipart::*;
}

#[cfg(feature = "azure")]
pub mod azure {
    pub use crate::internal::azure::*;
}

#[cfg(feature = "webhooks")]
pub mod webhooks {
    pub use crate::internal::webhooks::*;
}

#[cfg(feature = "realtime")]
pub mod realtime {
    pub use crate::internal::realtime::*;
}

pub use crate::internal::client::oai::{ClientBuilder, OpenAI, WithResponse};
pub use crate::internal::error::oai::{
    ApiError, ApiErrorKind, Error, ErrorDetail, ErrorObject, Result,
};
pub use crate::internal::transport::oai::{ClientConfig, HttpTransport, ResponseMeta};
pub use pagination::ListPage;

pub use crate::internal::resources::{
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
