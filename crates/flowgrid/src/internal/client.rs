#[cfg(feature = "openai")]
pub mod oai {
    use crate::internal::error::oai::{Error, Result};
    #[cfg(feature = "admin")]
    use crate::internal::resources::AdminClient;
    #[cfg(feature = "assistants")]
    use crate::internal::resources::AssistantsClient;
    #[cfg(feature = "audio")]
    use crate::internal::resources::AudioClient;
    #[cfg(feature = "batches")]
    use crate::internal::resources::BatchesClient;
    use crate::internal::resources::ChatClient;
    use crate::internal::resources::CompletionsClient;
    #[cfg(feature = "containers")]
    use crate::internal::resources::ContainersClient;
    use crate::internal::resources::EmbeddingsClient;
    #[cfg(feature = "evals")]
    use crate::internal::resources::EvalsClient;
    #[cfg(feature = "files")]
    use crate::internal::resources::FilesClient;
    #[cfg(feature = "fine_tuning")]
    use crate::internal::resources::FineTuningClient;
    #[cfg(feature = "images")]
    use crate::internal::resources::ImagesClient;
    #[cfg(feature = "moderations")]
    use crate::internal::resources::ModerationsClient;
    use crate::internal::resources::ResponsesClient;
    #[cfg(feature = "assistants")]
    use crate::internal::resources::ThreadsClient;
    #[cfg(feature = "vector_stores")]
    use crate::internal::resources::VectorStoresClient;
    use crate::internal::transport::oai::{
        ClientConfig, HttpClientBuilderHook, HttpTransport, ResponseMeta, RetryIfResponseStatusFn,
    };
    use std::sync::Arc;
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

        /// Threads API for Assistants workflows (feature `assistants`); paths under `/threads`.
        #[cfg(feature = "assistants")]
        pub fn threads(&self) -> ThreadsClient<'_> {
            ThreadsClient::new(self)
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
        pub fn webhooks(&self) -> crate::internal::oai::webhooks::WebhooksClient<'_> {
            crate::internal::oai::webhooks::WebhooksClient::new(self)
        }
    }

    /// Fluent client builder.
    #[derive(Clone, Default)]
    pub struct ClientBuilder {
        api_key: Option<String>,
        base_url: Option<String>,
        org_id: Option<String>,
        project_id: Option<String>,
        timeout: Option<Duration>,
        max_retries: Option<u32>,
        user_agent_suffix: Option<String>,
        request_hook: Option<
            std::sync::Arc<
                dyn Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder + Send + Sync,
            >,
        >,
        http_client_builder_hook: Option<HttpClientBuilderHook>,
        #[cfg(feature = "rate-aware-retry")]
        rate_limit_aware_backoff: Option<bool>,
        #[cfg(feature = "webhooks")]
        webhook_secret: Option<String>,
        retry_if_response_status: Option<RetryIfResponseStatusFn>,
    }

    impl std::fmt::Debug for ClientBuilder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut d = f.debug_struct("ClientBuilder");
            d.field("api_key", &self.api_key.as_ref().map(|_| "***"));
            d.field("base_url", &self.base_url);
            d.field("org_id", &self.org_id);
            d.field("project_id", &self.project_id);
            d.field("timeout", &self.timeout);
            d.field("max_retries", &self.max_retries);
            d.field("user_agent_suffix", &self.user_agent_suffix);
            d.field(
                "request_pre_send_hook",
                &self.request_hook.as_ref().map(|_| "..."),
            );
            d.field(
                "http_client_builder_hook",
                &self.http_client_builder_hook.as_ref().map(|_| "..."),
            );
            #[cfg(feature = "rate-aware-retry")]
            d.field("rate_limit_aware_backoff", &self.rate_limit_aware_backoff);
            #[cfg(feature = "webhooks")]
            d.field(
                "webhook_secret",
                &self.webhook_secret.as_ref().map(|_| "***"),
            );
            d.field(
                "retry_if_response_status",
                &self.retry_if_response_status.as_ref().map(|_| "..."),
            );
            d.finish()
        }
    }

    impl ClientBuilder {
        /// Empty builder (same as [`Default::default`]).
        pub fn new() -> Self {
            Self::default()
        }

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
                request_hook: None,
                http_client_builder_hook: c.http_client_builder_hook,
                #[cfg(feature = "rate-aware-retry")]
                rate_limit_aware_backoff: Some(c.rate_limit_aware_backoff),
                #[cfg(feature = "webhooks")]
                webhook_secret: c.webhook_secret,
                retry_if_response_status: c.retry_if_response_status,
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

        /// Hook that runs after default headers are applied and **immediately before** the HTTP
        /// request is sent. Use for extra headers (e.g. correlation IDs). Never log API keys or
        /// request bodies here.
        pub fn request_pre_send_hook<F>(mut self, hook: F) -> Self
        where
            F: Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder + Send + Sync + 'static,
        {
            self.request_hook = Some(Arc::new(hook));
            self
        }

        /// Customize the shared [`reqwest::Client`](reqwest::Client) after the SDK applies the
        /// default per-request HTTP timeout from [`OpenAiClientConfig`](crate::OpenAiClientConfig)
        /// (field `timeout`; custom connector, CA bundle, HTTP/2, …).
        pub fn http_client_builder_hook<F>(mut self, hook: F) -> Self
        where
            F: Fn(reqwest::ClientBuilder) -> reqwest::ClientBuilder + Send + Sync + 'static,
        {
            self.http_client_builder_hook = Some(Arc::new(hook));
            self
        }

        /// Prefer rate-limit **reset** headers over raw exponential backoff when `Retry-After` is absent
        /// (feature **`rate-aware-retry`**).
        #[cfg(feature = "rate-aware-retry")]
        pub fn rate_limit_aware_backoff(mut self, enabled: bool) -> Self {
            self.rate_limit_aware_backoff = Some(enabled);
            self
        }

        /// Conservative defaults for **OpenAI-shaped** HTTP servers (LiteLLM, vLLM, etc.).
        ///
        /// You still choose `base_url`; see README *OpenAI-compatible servers* for supported surface.
        #[cfg(feature = "compat-openai")]
        pub fn openai_http_compatible_profile(mut self) -> Self {
            self.max_retries = Some(self.max_retries.unwrap_or(2).min(2));
            self
        }

        /// Default webhook signing secret (feature `webhooks`).
        #[cfg(feature = "webhooks")]
        pub fn webhook_secret(mut self, secret: impl Into<String>) -> Self {
            self.webhook_secret = Some(secret.into());
            self
        }

        /// Custom rule: return `true` to retry this **HTTP response status** (same backoff as
        /// built-ins). When unset, the default retryable-status set is used.
        pub fn retry_if_response_status<F>(mut self, f: F) -> Self
        where
            F: Fn(reqwest::StatusCode, &reqwest::header::HeaderMap) -> bool + Send + Sync + 'static,
        {
            self.retry_if_response_status = Some(Arc::new(f));
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
                request_hook: self.request_hook,
                http_client_builder_hook: self.http_client_builder_hook,
                #[cfg(feature = "webhooks")]
                webhook_secret: self.webhook_secret,
                retry_after_max: Duration::from_millis(2000),
                retry_if_response_status: self.retry_if_response_status,
                #[cfg(feature = "rate-aware-retry")]
                rate_limit_aware_backoff: self.rate_limit_aware_backoff.unwrap_or(false),
            };
            let transport = HttpTransport::new(config)?;
            Ok(OpenAI { transport })
        }
    }
}

#[cfg(feature = "anthropic")]
pub mod clu {
    use crate::internal::error::clu::{Error, Result};
    use crate::internal::transport::clu::{
        ClientConfig, HttpClientBuilderHook, HttpTransport, ResponseMeta, RetryIfResponseStatusFn,
    };
    use std::sync::Arc;
    use std::time::Duration;

    /// Response body plus HTTP metadata.
    #[derive(Debug, Clone)]
    pub struct WithResponse<T> {
        pub data: T,
        pub meta: ResponseMeta,
    }

    impl<T> WithResponse<T> {
        pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> WithResponse<U> {
            WithResponse {
                data: f(self.data),
                meta: self.meta,
            }
        }
    }

    /// Root Anthropic client.
    #[derive(Clone)]
    pub struct Anthropic {
        pub(crate) transport: HttpTransport,
    }

    impl Anthropic {
        pub fn builder() -> AnthropicBuilder {
            AnthropicBuilder::default()
        }

        pub fn from_env() -> Result<Self> {
            AnthropicBuilder::from_env()?.build()
        }

        pub fn transport(&self) -> &HttpTransport {
            &self.transport
        }

        pub fn messages(&self) -> crate::internal::resources::messages::MessagesClient<'_> {
            crate::internal::resources::messages::MessagesClient::new(self)
        }

        #[cfg(feature = "models")]
        pub fn models(&self) -> crate::internal::resources::models::ModelsClient<'_> {
            crate::internal::resources::models::ModelsClient::new(self)
        }

        #[cfg(feature = "beta")]
        pub fn beta(&self) -> crate::internal::resources::beta::BetaClient<'_> {
            crate::internal::resources::beta::BetaClient::new(self)
        }
    }

    #[derive(Clone, Default)]
    pub struct AnthropicBuilder {
        api_key: Option<String>,
        base_url: Option<String>,
        anthropic_version: Option<String>,
        anthropic_beta: Option<String>,
        timeout: Option<Duration>,
        max_retries: Option<u32>,
        user_agent_suffix: Option<String>,
        request_hook:
            Option<Arc<dyn Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder + Send + Sync>>,
        http_client_builder_hook: Option<HttpClientBuilderHook>,
        #[cfg(feature = "rate-aware-retry")]
        rate_limit_aware_backoff: Option<bool>,
        retry_if_response_status: Option<RetryIfResponseStatusFn>,
    }

    impl std::fmt::Debug for AnthropicBuilder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut d = f.debug_struct("AnthropicBuilder");
            d.field("api_key", &self.api_key.as_ref().map(|_| "***"));
            d.field("base_url", &self.base_url);
            d.field("anthropic_version", &self.anthropic_version);
            d.field("anthropic_beta", &self.anthropic_beta);
            d.field("timeout", &self.timeout);
            d.field("max_retries", &self.max_retries);
            d.field("user_agent_suffix", &self.user_agent_suffix);
            d.field(
                "request_pre_send_hook",
                &self.request_hook.as_ref().map(|_| "..."),
            );
            d.field(
                "http_client_builder_hook",
                &self.http_client_builder_hook.as_ref().map(|_| "..."),
            );
            #[cfg(feature = "rate-aware-retry")]
            d.field("rate_limit_aware_backoff", &self.rate_limit_aware_backoff);
            d.field(
                "retry_if_response_status",
                &self.retry_if_response_status.as_ref().map(|_| "..."),
            );
            d.finish()
        }
    }

    impl AnthropicBuilder {
        /// Empty builder (same as [`Default::default`]).
        pub fn new() -> Self {
            Self::default()
        }

        pub fn from_env() -> Result<Self> {
            let c = ClientConfig::from_env()?;
            Ok(Self {
                api_key: Some(c.api_key),
                base_url: Some(c.base_url.to_string()),
                anthropic_version: Some(c.anthropic_version),
                anthropic_beta: c.anthropic_beta,
                timeout: Some(c.timeout),
                max_retries: Some(c.max_retries),
                user_agent_suffix: c.user_agent_suffix,
                request_hook: None,
                http_client_builder_hook: c.http_client_builder_hook,
                #[cfg(feature = "rate-aware-retry")]
                rate_limit_aware_backoff: Some(c.rate_limit_aware_backoff),
                retry_if_response_status: c.retry_if_response_status,
            })
        }

        pub fn api_key(mut self, k: impl Into<String>) -> Self {
            self.api_key = Some(k.into());
            self
        }

        pub fn base_url(mut self, u: impl Into<String>) -> Self {
            self.base_url = Some(u.into());
            self
        }

        pub fn anthropic_version(mut self, v: impl Into<String>) -> Self {
            self.anthropic_version = Some(v.into());
            self
        }

        pub fn anthropic_beta(mut self, v: impl Into<String>) -> Self {
            self.anthropic_beta = Some(v.into());
            self
        }

        pub fn timeout(mut self, d: Duration) -> Self {
            self.timeout = Some(d);
            self
        }

        pub fn max_retries(mut self, n: u32) -> Self {
            self.max_retries = Some(n);
            self
        }

        pub fn user_agent_suffix(mut self, s: impl Into<String>) -> Self {
            self.user_agent_suffix = Some(s.into());
            self
        }

        /// Hook that runs after default Anthropic headers are applied and **immediately before**
        /// the HTTP request is sent. Do not log API keys or bodies here.
        pub fn request_pre_send_hook<F>(mut self, hook: F) -> Self
        where
            F: Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder + Send + Sync + 'static,
        {
            self.request_hook = Some(Arc::new(hook));
            self
        }

        pub fn http_client_builder_hook<F>(mut self, hook: F) -> Self
        where
            F: Fn(reqwest::ClientBuilder) -> reqwest::ClientBuilder + Send + Sync + 'static,
        {
            self.http_client_builder_hook = Some(Arc::new(hook));
            self
        }

        #[cfg(feature = "rate-aware-retry")]
        pub fn rate_limit_aware_backoff(mut self, enabled: bool) -> Self {
            self.rate_limit_aware_backoff = Some(enabled);
            self
        }

        /// Custom HTTP status retry rule (same semantics as OpenAI `ClientBuilder::retry_if_response_status`).
        pub fn retry_if_response_status<F>(mut self, f: F) -> Self
        where
            F: Fn(reqwest::StatusCode, &reqwest::header::HeaderMap) -> bool + Send + Sync + 'static,
        {
            self.retry_if_response_status = Some(Arc::new(f));
            self
        }

        pub fn build(self) -> Result<Anthropic> {
            let api_key = self
                .api_key
                .ok_or_else(|| Error::Config("api_key is required".to_string()))?;
            let base = self
                .base_url
                .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string());
            let config = ClientConfig {
                api_key,
                base_url: url::Url::parse(&base)?,
                anthropic_version: self
                    .anthropic_version
                    .unwrap_or_else(|| "2023-06-01".to_string()),
                anthropic_beta: self.anthropic_beta,
                timeout: self.timeout.unwrap_or_else(|| Duration::from_secs(120)),
                max_retries: self.max_retries.unwrap_or(2),
                user_agent_suffix: self.user_agent_suffix,
                request_hook: self.request_hook,
                http_client_builder_hook: self.http_client_builder_hook,
                retry_after_max: Duration::from_millis(2000),
                retry_if_response_status: self.retry_if_response_status,
                #[cfg(feature = "rate-aware-retry")]
                rate_limit_aware_backoff: self.rate_limit_aware_backoff.unwrap_or(false),
            };
            Ok(Anthropic {
                transport: HttpTransport::new(config)?,
            })
        }
    }
}
