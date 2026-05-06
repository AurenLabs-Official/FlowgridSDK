//! Azure OpenAI helper client (feature `azure`).
//!
//! OpenAI-only; compiled only with `openai` and `azure`.
//!
//! Azure uses the `api-key` header and typically requires an `api-version` query parameter on
//! every request. This module configures [`crate::internal::oai::OpenAI`] accordingly.

use crate::internal::client::oai::OpenAI;
use crate::internal::error::oai::{Error, Result};
use crate::internal::transport::oai::{ClientConfig, HttpTransport};
use std::sync::Arc;
use std::time::Duration;

/// Builder for Azure OpenAI (`AzureOpenAI` in Node).
#[derive(Clone, Default)]
pub struct AzureClientBuilder {
    api_key: Option<String>,
    endpoint: Option<String>,
    api_version: Option<String>,
    timeout: Option<Duration>,
    max_retries: Option<u32>,
    request_hook:
        Option<Arc<dyn Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder + Send + Sync>>,
}

impl std::fmt::Debug for AzureClientBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureClientBuilder")
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("endpoint", &self.endpoint)
            .field("api_version", &self.api_version)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field(
                "request_pre_send_hook",
                &self.request_hook.as_ref().map(|_| "..."),
            )
            .finish()
    }
}

impl AzureClientBuilder {
    /// New empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Azure `api-key` value.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Deployment base URL, for example
    /// `https://MYRESOURCE.openai.azure.com/openai/deployments/MYDEPLOYMENT`.
    pub fn endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = Some(url.into());
        self
    }

    /// `api-version` query value (defaults to `2024-02-15-preview`).
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Per-request timeout.
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    /// Retry behavior.
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = Some(n);
        self
    }

    /// Same as [`crate::internal::client::oai::ClientBuilder::request_pre_send_hook`].
    pub fn request_pre_send_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder + Send + Sync + 'static,
    {
        self.request_hook = Some(Arc::new(hook));
        self
    }

    /// Build a normal [`OpenAI`] client configured for Azure HTTP semantics.
    pub fn build(self) -> Result<OpenAI> {
        let api_key = self
            .api_key
            .ok_or_else(|| Error::Config("Azure api_key is required".to_string()))?;
        let endpoint = self
            .endpoint
            .ok_or_else(|| Error::Config("Azure endpoint is required".to_string()))?;
        let base_url = url::Url::parse(&endpoint)
            .map_err(|e| Error::Config(format!("invalid Azure endpoint: {e}")))?;
        let api_version = self
            .api_version
            .unwrap_or_else(|| "2024-02-15-preview".to_string());
        let config = ClientConfig {
            api_key,
            base_url,
            use_api_key_header: true,
            default_query: vec![("api-version".to_string(), api_version)],
            org_id: None,
            project_id: None,
            timeout: self.timeout.unwrap_or_else(|| Duration::from_secs(120)),
            max_retries: self.max_retries.unwrap_or(2),
            user_agent_suffix: Some("azure-openai".to_string()),
            request_hook: self.request_hook,
            #[cfg(feature = "webhooks")]
            webhook_secret: None,
            retry_after_max: Duration::from_millis(2000),
        };
        let transport = HttpTransport::new(config)?;
        Ok(OpenAI { transport })
    }
}

/// Type alias mirroring the Node export name.
pub type AzureOpenAI = OpenAI;
