use crate::error::{Error, Result};
use crate::transport::{ClientConfig, HttpTransport, ResponseMeta};
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

    pub fn messages(&self) -> crate::resources::messages::MessagesClient<'_> {
        crate::resources::messages::MessagesClient::new(self)
    }

    #[cfg(feature = "models")]
    pub fn models(&self) -> crate::resources::models::ModelsClient<'_> {
        crate::resources::models::ModelsClient::new(self)
    }

    #[cfg(feature = "beta")]
    pub fn beta(&self) -> crate::resources::beta::BetaClient<'_> {
        crate::resources::beta::BetaClient::new(self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnthropicBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    anthropic_version: Option<String>,
    anthropic_beta: Option<String>,
    timeout: Option<Duration>,
    max_retries: Option<u32>,
    user_agent_suffix: Option<String>,
}

impl AnthropicBuilder {
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
        };
        Ok(Anthropic {
            transport: HttpTransport::new(config)?,
        })
    }
}
