use crate::error::{Error, Result};
#[cfg(feature = "admin")]
use crate::resources::AdminClient;
#[cfg(feature = "assistants")]
use crate::resources::AssistantsClient;
#[cfg(feature = "audio")]
use crate::resources::AudioClient;
#[cfg(feature = "batches")]
use crate::resources::BatchesClient;
use crate::resources::ChatClient;
use crate::resources::CompletionsClient;
#[cfg(feature = "containers")]
use crate::resources::ContainersClient;
use crate::resources::EmbeddingsClient;
#[cfg(feature = "evals")]
use crate::resources::EvalsClient;
#[cfg(feature = "files")]
use crate::resources::FilesClient;
#[cfg(feature = "fine_tuning")]
use crate::resources::FineTuningClient;
#[cfg(feature = "images")]
use crate::resources::ImagesClient;
#[cfg(feature = "moderations")]
use crate::resources::ModerationsClient;
use crate::resources::ResponsesClient;
#[cfg(feature = "vector_stores")]
use crate::resources::VectorStoresClient;
use crate::transport::{ClientConfig, HttpTransport, ResponseMeta};
use std::time::Duration;

/// Pair of decoded body and response metadata (`withResponse()` in Node).
#[derive(Debug, Clone)]
pub struct WithResponse<T> {
    /// Decoded JSON body.
    pub data: T,
    /// Response headers / status / `x-request-id`.
    pub meta: ResponseMeta,
}

impl<T> WithResponse<T> {
    /// Map the body while preserving metadata.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> WithResponse<U> {
        WithResponse {
            data: f(self.data),
            meta: self.meta,
        }
    }
}

/// Root OpenAI client (namespace layout similar to openai-node).
#[derive(Clone)]
pub struct OpenAI {
    pub(crate) transport: HttpTransport,
}

impl OpenAI {
    /// Builder entrypoint.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Construct from environment (`OPENAI_API_KEY`, etc.).
    pub fn from_env() -> Result<Self> {
        ClientBuilder::from_env()?.build()
    }

    /// Low-level transport (advanced callers).
    pub fn transport(&self) -> &HttpTransport {
        &self.transport
    }

    /// Responses API (`client.responses` in Node).
    pub fn responses(&self) -> ResponsesClient<'_> {
        ResponsesClient::new(self)
    }

    /// Chat API (`client.chat`).
    pub fn chat(&self) -> ChatClient<'_> {
        ChatClient::new(self)
    }

    /// Legacy Completions API.
    pub fn completions(&self) -> CompletionsClient<'_> {
        CompletionsClient::new(self)
    }

    /// Embeddings API.
    pub fn embeddings(&self) -> EmbeddingsClient<'_> {
        EmbeddingsClient::new(self)
    }

    /// Files API (feature `files`).
    #[cfg(feature = "files")]
    pub fn files(&self) -> FilesClient<'_> {
        FilesClient::new(self)
    }

    /// Images API (feature `images`).
    #[cfg(feature = "images")]
    pub fn images(&self) -> ImagesClient<'_> {
        ImagesClient::new(self)
    }

    /// Audio API (feature `audio`).
    #[cfg(feature = "audio")]
    pub fn audio(&self) -> AudioClient<'_> {
        AudioClient::new(self)
    }

    /// Moderations API (feature `moderations`).
    #[cfg(feature = "moderations")]
    pub fn moderations(&self) -> ModerationsClient<'_> {
        ModerationsClient::new(self)
    }

    /// Batches API (feature `batches`).
    #[cfg(feature = "batches")]
    pub fn batches(&self) -> BatchesClient<'_> {
        BatchesClient::new(self)
    }

    /// Fine-tuning API (feature `fine_tuning`).
    #[cfg(feature = "fine_tuning")]
    pub fn fine_tuning(&self) -> FineTuningClient<'_> {
        FineTuningClient::new(self)
    }

    /// Evals API (feature `evals`).
    #[cfg(feature = "evals")]
    pub fn evals(&self) -> EvalsClient<'_> {
        EvalsClient::new(self)
    }

    /// Assistants API (feature `assistants`).
    #[cfg(feature = "assistants")]
    pub fn assistants(&self) -> AssistantsClient<'_> {
        AssistantsClient::new(self)
    }

    /// Vector stores API (feature `vector_stores`).
    #[cfg(feature = "vector_stores")]
    pub fn vector_stores(&self) -> VectorStoresClient<'_> {
        VectorStoresClient::new(self)
    }

    /// Containers API (feature `containers`).
    #[cfg(feature = "containers")]
    pub fn containers(&self) -> ContainersClient<'_> {
        ContainersClient::new(self)
    }

    /// Administration endpoints (feature `admin`).
    #[cfg(feature = "admin")]
    pub fn admin(&self) -> AdminClient<'_> {
        AdminClient::new(self)
    }

    /// Webhook helpers (feature `webhooks`).
    #[cfg(feature = "webhooks")]
    pub fn webhooks(&self) -> crate::webhooks::WebhooksClient<'_> {
        crate::webhooks::WebhooksClient::new(self)
    }
}

/// Fluent client builder.
#[derive(Clone, Debug, Default)]
pub struct ClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    org_id: Option<String>,
    project_id: Option<String>,
    timeout: Option<Duration>,
    max_retries: Option<u32>,
    user_agent_suffix: Option<String>,
    #[cfg(feature = "webhooks")]
    webhook_secret: Option<String>,
}

impl ClientBuilder {
    /// Populate defaults from environment.
    pub fn from_env() -> Result<Self> {
        let c = ClientConfig::from_env()?;
        Ok(Self {
            api_key: Some(c.api_key),
            base_url: Some(c.base_url.to_string()),
            org_id: c.org_id,
            project_id: c.project_id,
            timeout: Some(c.timeout),
            max_retries: Some(c.max_retries),
            user_agent_suffix: c.user_agent_suffix,
            #[cfg(feature = "webhooks")]
            webhook_secret: c.webhook_secret,
        })
    }

    /// API key (required unless using azure-only flows later).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override base URL (must include `/v1` path prefix like the default).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// `OpenAI-Organization` header.
    pub fn organization(mut self, org_id: impl Into<String>) -> Self {
        self.org_id = Some(org_id.into());
        self
    }

    /// `OpenAI-Project` header.
    pub fn project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Per-request timeout.
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    /// Max retries after the first attempt (default 2).
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = Some(n);
        self
    }

    /// Suffix appended to `User-Agent`.
    pub fn user_agent_suffix(mut self, s: impl Into<String>) -> Self {
        self.user_agent_suffix = Some(s.into());
        self
    }

    /// Default webhook signing secret (feature `webhooks`).
    #[cfg(feature = "webhooks")]
    pub fn webhook_secret(mut self, secret: impl Into<String>) -> Self {
        self.webhook_secret = Some(secret.into());
        self
    }

    /// Build configured [`OpenAI`].
    pub fn build(self) -> Result<OpenAI> {
        let api_key = self
            .api_key
            .ok_or_else(|| Error::Config("api_key is required".to_string()))?;
        let base = self
            .base_url
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let base_url = url::Url::parse(&base)?;
        let config = ClientConfig {
            api_key,
            base_url,
            use_api_key_header: false,
            default_query: Vec::new(),
            org_id: self.org_id,
            project_id: self.project_id,
            timeout: self.timeout.unwrap_or_else(|| Duration::from_secs(120)),
            max_retries: self.max_retries.unwrap_or(2),
            user_agent_suffix: self.user_agent_suffix,
            #[cfg(feature = "webhooks")]
            webhook_secret: self.webhook_secret,
        };
        let transport = HttpTransport::new(config)?;
        Ok(OpenAI { transport })
    }
}
