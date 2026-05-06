#[cfg(feature = "openai")]
mod usage;

#[cfg(feature = "openai")]
pub use usage::{CompletionUsage, EmbeddingUsage, ResponseObjectUsage};
#[cfg(feature = "openai")]
mod chat;
#[cfg(feature = "openai")]
mod completions;
#[cfg(feature = "openai")]
mod embeddings;
#[cfg(feature = "openai")]
mod paths;
#[cfg(feature = "openai")]
mod responses;

#[cfg(all(feature = "openai", feature = "admin"))]
mod admin;
#[cfg(all(feature = "openai", feature = "assistants"))]
mod assistants;
#[cfg(all(feature = "openai", feature = "audio"))]
mod audio;
#[cfg(feature = "batches")]
mod batches;
#[cfg(all(feature = "openai", feature = "containers"))]
mod containers;
#[cfg(all(feature = "openai", feature = "evals"))]
mod evals;
#[cfg(all(feature = "openai", feature = "files"))]
mod files;
#[cfg(all(feature = "openai", feature = "fine_tuning"))]
mod fine_tuning;
#[cfg(all(feature = "openai", feature = "images"))]
mod images;
#[cfg(all(feature = "openai", feature = "moderations"))]
mod moderations;
#[cfg(all(feature = "openai", feature = "vector_stores"))]
mod vector_stores;

#[cfg(all(feature = "anthropic", feature = "beta"))]
pub mod beta;
#[cfg(all(feature = "anthropic", feature = "beta"))]
pub use beta::{BetaClient, BetaModel, BetaModelsListResponse};
#[cfg(feature = "anthropic")]
pub mod messages;
#[cfg(all(feature = "anthropic", feature = "models"))]
pub mod models;

#[cfg(all(feature = "openai", feature = "admin"))]
pub use admin::AdminClient;
#[cfg(all(feature = "openai", feature = "assistants"))]
pub use assistants::AssistantsClient;
#[cfg(all(feature = "openai", feature = "audio"))]
pub use audio::AudioClient;
#[cfg(all(feature = "openai", feature = "batches"))]
pub use batches::BatchesClient;
#[cfg(feature = "openai")]
pub use chat::{
    ChatClient, ChatCompletion, ChatCompletionChoice, ChatCompletionDeleted,
    ChatCompletionListParams, ChatCompletionMessage, ChatCompletionMessagesListParams,
    ChatCompletionsClient, CreateChatCompletionRequest,
};
#[cfg(feature = "openai")]
pub use completions::{Completion, CompletionChoice, CompletionsClient, CreateCompletionRequest};
#[cfg(all(feature = "openai", feature = "containers"))]
pub use containers::ContainersClient;
#[cfg(feature = "openai")]
pub use embeddings::{
    CreateEmbeddingRequest, CreateEmbeddingResponse, Embedding, EmbeddingsClient,
};
#[cfg(all(feature = "openai", feature = "evals"))]
pub use evals::EvalsClient;
#[cfg(all(feature = "openai", feature = "files"))]
pub use files::FilesClient;
#[cfg(all(feature = "openai", feature = "fine_tuning"))]
pub use fine_tuning::FineTuningClient;
#[cfg(all(feature = "openai", feature = "images"))]
pub use images::ImagesClient;
#[cfg(all(feature = "openai", feature = "moderations"))]
pub use moderations::ModerationsClient;
#[cfg(feature = "openai")]
pub use responses::{
    BoxedByteStream, CreateResponseRequest, ResponseDeleted, ResponseObject, ResponsesClient,
};
#[cfg(all(feature = "openai", feature = "vector_stores"))]
pub use vector_stores::VectorStoresClient;

#[cfg(all(feature = "anthropic", feature = "batches"))]
pub use batches::MessageBatchesClient;
