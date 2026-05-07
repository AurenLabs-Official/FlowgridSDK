//! Hand-crafted async OpenAI API client (inspired by openai-node layout).

// `#[cfg]` on nested `pub mod` shims below must match the `mod …;` declarations in `mod.rs`
// (same predicates for multipart/azure/webhooks/realtime; this module is only built with OpenAI).

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
pub use pagination::{ListPage, ListPagesLimits};

pub use crate::internal::sse::oai::{OpenAiSseEventStream, SseEvent, SseStream};

#[cfg(feature = "stream-types")]
pub use crate::internal::stream_typing::{
    accumulate_openai_chat_visible_text_into, accumulate_openai_response_visible_text_into,
    parse_openai_chat_stream_json, parse_openai_response_stream_json, OpenAiChatChunkChoice,
    OpenAiChatCompletionChunk, OpenAiResponseStreamLine, OpenAiStreamTextLimits,
    StreamTextAccumulateError,
};

pub use crate::internal::resources::{
    BoxedByteStream, ChatClient, ChatCompletion, ChatCompletionChoice, ChatCompletionDeleted,
    ChatCompletionListParams, ChatCompletionMessage, ChatCompletionMessagesListParams,
    ChatCompletionsClient, Completion, CompletionChoice, CompletionUsage, CompletionsClient,
    CreateChatCompletionRequest, CreateCompletionRequest, CreateEmbeddingRequest,
    CreateEmbeddingResponse, CreateResponseRequest, Embedding, EmbeddingUsage, EmbeddingsClient,
    ResponseDeleted, ResponseObject, ResponseObjectUsage, ResponsesClient,
};

#[cfg(all(feature = "openai", feature = "assistants"))]
pub use crate::internal::resources::{
    Assistant, AssistantsClient, AssistantsListParams, Thread, ThreadClient, ThreadMessage,
    ThreadMessagesClient, ThreadMessagesListParams, ThreadRun, ThreadRunStep, ThreadRunsClient,
    ThreadsClient,
};

#[cfg(feature = "azure")]
pub use azure::{AzureClientBuilder, AzureOpenAI};

#[cfg(feature = "realtime")]
pub use realtime::{connect as connect_realtime, RealtimeSocket};

#[cfg(feature = "webhooks")]
pub use webhooks::WebhooksClient;
