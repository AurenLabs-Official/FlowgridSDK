#[cfg(feature = "openai")]
pub mod oai {
    use crate::internal::error::oai::{ApiError, ErrorObject, Result};
    use bytes::Bytes;
    use futures::Stream;
    use futures::TryStreamExt;
    use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
    use reqwest::{Client, Method, RequestBuilder, Response, StatusCode};
    use serde::Serialize;
    use std::io;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::sleep;

    /// Client configuration (clonable, shared by `OpenAI`).
    #[derive(Clone, Debug)]
    pub struct ClientConfig {
        /// API key (`Bearer` by default; Azure uses `api-key` header).
        pub api_key: String,
        /// Base URL including `/v1` suffix by default.
        pub base_url: url::Url,
        /// When true, send `api-key` instead of `Authorization: Bearer …` (Azure OpenAI).
        pub use_api_key_header: bool,
        /// Optional default query parameters applied to every request (e.g. Azure `api-version`).
        pub default_query: Vec<(String, String)>,
        /// Optional organization id.
        pub org_id: Option<String>,
        /// Optional project id.
        pub project_id: Option<String>,
        /// HTTP timeout for individual attempts.
        pub timeout: Duration,
        /// Maximum retries after the first attempt (openai-node default: 2).
        pub max_retries: u32,
        /// Custom `User-Agent` prefix appended to the default.
        pub user_agent_suffix: Option<String>,
        /// Default webhook secret (feature `webhooks`).
        #[cfg(feature = "webhooks")]
        pub webhook_secret: Option<String>,
    }

    impl ClientConfig {
        /// Build from environment (`OPENAI_API_KEY`, optional org/project/base url).
        pub fn from_env() -> Result<Self> {
            let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                crate::internal::error::oai::Error::Config("OPENAI_API_KEY not set".to_string())
            })?;
            let base = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            let base_url = url::Url::parse(&base)?;
            Ok(Self {
                api_key,
                base_url,
                org_id: std::env::var("OPENAI_ORG_ID").ok(),
                project_id: std::env::var("OPENAI_PROJECT").ok(),
                timeout: Duration::from_secs(120),
                max_retries: 2,
                user_agent_suffix: None,
                use_api_key_header: false,
                default_query: Vec::new(),
                #[cfg(feature = "webhooks")]
                webhook_secret: std::env::var("OPENAI_WEBHOOK_SECRET").ok(),
            })
        }
    }

    /// Per-response metadata (akin to `.withResponse()` in Node).
    #[derive(Debug, Clone)]
    pub struct ResponseMeta {
        /// `x-request-id` header.
        pub request_id: Option<String>,
        /// HTTP status.
        pub status: StatusCode,
        /// Full header map.
        pub headers: HeaderMap,
    }

    fn response_meta(resp: &Response) -> ResponseMeta {
        let headers = resp.headers().clone();
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        ResponseMeta {
            request_id,
            status: resp.status(),
            headers,
        }
    }

    fn retry_status(status: StatusCode) -> bool {
        matches!(status.as_u16(), 408 | 409 | 429) || status.as_u16() >= 500
    }

    fn retry_error(err: &reqwest::Error) -> bool {
        err.is_timeout() || err.is_connect() || err.is_request()
    }

    fn backoff(attempt: u32) -> Duration {
        let ms = 50u64.saturating_mul(2u64.saturating_pow(attempt.saturating_sub(1)));
        Duration::from_millis(ms.min(2000))
    }

    /// Low-level HTTP transport with retries.
    #[derive(Clone)]
    pub struct HttpTransport {
        inner: Client,
        pub(crate) config: Arc<ClientConfig>,
    }

    impl HttpTransport {
        /// Create transport from configuration.
        pub fn new(mut config: ClientConfig) -> Result<Self> {
            let path = config.base_url.path();
            if !path.is_empty() && !path.ends_with('/') {
                config.base_url.set_path(&format!("{path}/"));
            }
            let inner = Client::builder().timeout(config.timeout).build()?;
            Ok(Self {
                inner,
                config: Arc::new(config),
            })
        }

        fn user_agent(&self) -> HeaderValue {
            let base = format!("flowgrid/{} openai rust-reqwest", env!("CARGO_PKG_VERSION"));
            let s = match &self.config.user_agent_suffix {
                Some(suffix) => format!("{base} {suffix}"),
                None => base,
            };
            HeaderValue::from_str(&s).unwrap_or_else(|_| HeaderValue::from_static("flowgrid"))
        }

        fn apply_default_headers(&self, rb: RequestBuilder) -> RequestBuilder {
            let mut rb = rb;
            if self.config.use_api_key_header {
                if let Ok(h) = HeaderValue::from_str(&self.config.api_key) {
                    rb = rb.header("api-key", h);
                }
            } else {
                let token = format!("Bearer {}", self.config.api_key);
                if let Ok(h) = HeaderValue::from_str(&token) {
                    rb = rb.header(AUTHORIZATION, h);
                }
            }
            rb = rb.header(ACCEPT, HeaderValue::from_static("application/json"));
            rb = rb.header(reqwest::header::USER_AGENT, self.user_agent());
            if let Some(ref org) = self.config.org_id {
                if let Ok(h) = HeaderValue::from_str(org) {
                    rb = rb.header("OpenAI-Organization", h);
                }
            }
            if let Some(ref proj) = self.config.project_id {
                if let Ok(h) = HeaderValue::from_str(proj) {
                    rb = rb.header("OpenAI-Project", h);
                }
            }
            rb
        }

        /// Resolve path against configured base URL.
        pub fn url(&self, path: &str) -> Result<url::Url> {
            let mut u = self
                .config
                .base_url
                .join(path.trim_start_matches('/'))
                .map_err(|e| crate::internal::error::oai::Error::Config(e.to_string()))?;
            if !self.config.default_query.is_empty() {
                {
                    let mut pairs = u.query_pairs_mut();
                    for (k, v) in &self.config.default_query {
                        pairs.append_pair(k, v);
                    }
                }
            }
            Ok(u)
        }

        async fn send_with_retries(&self, rb: RequestBuilder) -> Result<Response> {
            let max = self.config.max_retries as usize;
            let mut attempt = 0usize;
            let mut rb = rb;
            loop {
                let clone = rb.try_clone().ok_or_else(|| {
                    crate::internal::error::oai::Error::Config(
                        "request could not be cloned for retries".to_string(),
                    )
                })?;
                match rb.send().await {
                    Ok(resp) => {
                        let status = resp.status();
                        if attempt < max && retry_status(status) {
                            drop(resp);
                            sleep(backoff((attempt + 1) as u32)).await;
                            rb = clone;
                            attempt += 1;
                            continue;
                        }
                        return Ok(resp);
                    }
                    Err(e) => {
                        if attempt < max && retry_error(&e) {
                            sleep(backoff((attempt + 1) as u32)).await;
                            rb = clone;
                            attempt += 1;
                            continue;
                        }
                        return Err(e.into());
                    }
                }
            }
        }

        fn api_error_from_text(status: StatusCode, text: &str, headers: HeaderMap) -> ApiError {
            let request_id = headers
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let body: Option<ErrorObject> = serde_json::from_str(text).ok();
            ApiError {
                status,
                body,
                raw_body: Some(text.to_string()),
                request_id,
                headers,
            }
        }

        /// JSON request with body (POST/DELETE with JSON) or without (GET).
        pub async fn request_json<B: Serialize + ?Sized, T: serde::de::DeserializeOwned>(
            &self,
            method: Method,
            path: &str,
            body: Option<&B>,
        ) -> Result<(T, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self.inner.request(method, url.as_str());
            let rb = match body {
                Some(b) => self
                    .apply_default_headers(rb)
                    .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                    .json(b),
                None => self.apply_default_headers(rb),
            };
            let resp = self.send_with_retries(rb).await?;
            let meta = response_meta(&resp);
            let status = resp.status();
            let headers = meta.headers.clone();
            let text = resp.text().await?;
            if !status.is_success() {
                return Err(Self::api_error_from_text(status, &text, headers).into());
            }
            let v: T = serde_json::from_str(&text)?;
            Ok((v, meta))
        }

        /// GET JSON.
        pub async fn get_json<T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json::<serde_json::Value, T>(Method::GET, path, None)
                .await
        }

        /// GET raw bytes (e.g. file downloads, audio).
        pub async fn get_bytes(&self, path: &str) -> Result<(Vec<u8>, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self.apply_default_headers(self.inner.get(url.as_str()));
            let resp = self.send_with_retries(rb).await?;
            let meta = response_meta(&resp);
            let status = resp.status();
            let headers = meta.headers.clone();
            if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                return Err(Self::api_error_from_text(status, &text, headers).into());
            }
            let bytes = resp.bytes().await?;
            Ok((bytes.to_vec(), meta))
        }

        /// DELETE JSON (no body).
        pub async fn delete_json<T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json::<serde_json::Value, T>(Method::DELETE, path, None)
                .await
        }

        /// POST JSON.
        pub async fn post_json<B: Serialize + ?Sized, T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
            body: &B,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json(Method::POST, path, Some(body)).await
        }

        /// POST JSON returning raw bytes (e.g. `audio/speech`).
        pub async fn post_json_bytes<B: Serialize + ?Sized>(
            &self,
            path: &str,
            body: &B,
        ) -> Result<(Vec<u8>, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self
                .apply_default_headers(self.inner.request(Method::POST, url.as_str()))
                .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                .json(body);
            let resp = self.send_with_retries(rb).await?;
            let meta = response_meta(&resp);
            let status = resp.status();
            let headers = meta.headers.clone();
            if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                return Err(Self::api_error_from_text(status, &text, headers).into());
            }
            let bytes = resp.bytes().await?;
            Ok((bytes.to_vec(), meta))
        }

        /// PATCH JSON.
        pub async fn patch_json<B: Serialize + ?Sized, T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
            body: &B,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json(Method::PATCH, path, Some(body)).await
        }

        /// POST JSON returning a byte stream (SSE or binary).
        pub async fn post_stream_bytes<B: Serialize + ?Sized>(
            &self,
            path: &str,
            body: &B,
        ) -> Result<(
            impl Stream<Item = std::result::Result<Bytes, io::Error>> + Send,
            ResponseMeta,
        )> {
            let url = self.url(path)?;
            let rb = self
                .apply_default_headers(self.inner.request(Method::POST, url.as_str()))
                .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                .json(body);
            let resp = self.send_with_retries(rb).await?;
            let meta = response_meta(&resp);
            let status = resp.status();
            if !status.is_success() {
                let headers = meta.headers.clone();
                let text = resp.text().await.unwrap_or_default();
                return Err(Self::api_error_from_text(status, &text, headers).into());
            }
            let st = resp.bytes_stream().map_err(std::io::Error::other);
            Ok((st, meta))
        }

        /// POST multipart form, JSON response.
        pub async fn post_multipart_json<T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
            form: reqwest::multipart::Form,
        ) -> Result<(T, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self
                .apply_default_headers(self.inner.request(Method::POST, url.as_str()))
                .multipart(form);
            let resp = self.send_with_retries(rb).await?;
            let meta = response_meta(&resp);
            let status = resp.status();
            let headers = meta.headers.clone();
            let text = resp.text().await?;
            if !status.is_success() {
                return Err(Self::api_error_from_text(status, &text, headers).into());
            }
            let v: T = serde_json::from_str(&text)?;
            Ok((v, meta))
        }

        /// POST multipart form, raw bytes response.
        pub async fn post_multipart_bytes(
            &self,
            path: &str,
            form: reqwest::multipart::Form,
        ) -> Result<(Vec<u8>, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self
                .apply_default_headers(self.inner.request(Method::POST, url.as_str()))
                .multipart(form);
            let resp = self.send_with_retries(rb).await?;
            let meta = response_meta(&resp);
            let status = resp.status();
            let headers = meta.headers.clone();
            if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                return Err(Self::api_error_from_text(status, &text, headers).into());
            }
            let bytes = resp.bytes().await?;
            Ok((bytes.to_vec(), meta))
        }
    }
}

#[cfg(feature = "anthropic")]
pub mod clu {
    use crate::internal::error::clu::{ApiError, ErrorBody, Result};
    use bytes::Bytes;
    use futures::{Stream, TryStreamExt};
    use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
    use reqwest::{Client, Method, RequestBuilder, Response, StatusCode};
    use serde::Serialize;
    use std::io;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::sleep;

    /// Client configuration.
    #[derive(Clone, Debug)]
    pub struct ClientConfig {
        /// API key (`x-api-key`).
        pub api_key: String,
        /// Base URL (default `https://api.anthropic.com/v1`).
        pub base_url: url::Url,
        /// `anthropic-version` header (e.g. `2023-06-01`).
        pub anthropic_version: String,
        /// Optional `anthropic-beta` header (comma-separated feature flags).
        pub anthropic_beta: Option<String>,
        pub timeout: Duration,
        pub max_retries: u32,
        pub user_agent_suffix: Option<String>,
    }

    impl ClientConfig {
        /// `ANTHROPIC_API_KEY`, optional `ANTHROPIC_API_BASE`, `ANTHROPIC_VERSION`, `ANTHROPIC_BETA`.
        pub fn from_env() -> Result<Self> {
            let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                crate::internal::error::clu::Error::Config("ANTHROPIC_API_KEY not set".to_string())
            })?;
            let base = std::env::var("ANTHROPIC_API_BASE")
                .unwrap_or_else(|_| "https://api.anthropic.com/v1".to_string());
            let base_url = url::Url::parse(&base)?;
            let anthropic_version =
                std::env::var("ANTHROPIC_VERSION").unwrap_or_else(|_| "2023-06-01".to_string());
            Ok(Self {
                api_key,
                base_url,
                anthropic_version,
                anthropic_beta: std::env::var("ANTHROPIC_BETA").ok(),
                timeout: Duration::from_secs(120),
                max_retries: 2,
                user_agent_suffix: None,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct ResponseMeta {
        pub request_id: Option<String>,
        pub status: StatusCode,
        pub headers: HeaderMap,
    }

    fn response_meta(resp: &Response) -> ResponseMeta {
        let headers = resp.headers().clone();
        let request_id = headers
            .get("request-id")
            .or_else(|| headers.get("x-request-id"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        ResponseMeta {
            request_id,
            status: resp.status(),
            headers,
        }
    }

    fn retry_status(status: StatusCode) -> bool {
        matches!(status.as_u16(), 408 | 409 | 429)
            || status.as_u16() == 529
            || status.as_u16() >= 500
    }

    fn retry_error(err: &reqwest::Error) -> bool {
        err.is_timeout() || err.is_connect() || err.is_request()
    }

    fn backoff(attempt: u32) -> Duration {
        let ms = 50u64.saturating_mul(2u64.saturating_pow(attempt.saturating_sub(1)));
        Duration::from_millis(ms.min(2000))
    }

    /// HTTP transport with Anthropic headers and retries.
    #[derive(Clone)]
    pub struct HttpTransport {
        inner: Client,
        pub(crate) config: Arc<ClientConfig>,
    }

    impl HttpTransport {
        pub fn new(mut config: ClientConfig) -> Result<Self> {
            let path = config.base_url.path();
            if !path.is_empty() && !path.ends_with('/') {
                config.base_url.set_path(&format!("{path}/"));
            }
            let inner = Client::builder().timeout(config.timeout).build()?;
            Ok(Self {
                inner,
                config: Arc::new(config),
            })
        }

        fn user_agent(&self) -> HeaderValue {
            let base = format!(
                "flowgrid/{} anthropic rust-reqwest",
                env!("CARGO_PKG_VERSION")
            );
            HeaderValue::from_str(&match &self.config.user_agent_suffix {
                Some(s) => format!("{base} {s}"),
                None => base,
            })
            .unwrap_or_else(|_| HeaderValue::from_static("flowgrid"))
        }

        fn apply_default_headers(&self, rb: RequestBuilder, accept: &str) -> RequestBuilder {
            let mut rb = rb;
            if let Ok(h) = HeaderValue::from_str(&self.config.api_key) {
                rb = rb.header("x-api-key", h);
            }
            if let Ok(h) = HeaderValue::from_str(&self.config.anthropic_version) {
                rb = rb.header("anthropic-version", h);
            }
            if let Some(ref beta) = self.config.anthropic_beta {
                if let Ok(h) = HeaderValue::from_str(beta) {
                    rb = rb.header("anthropic-beta", h);
                }
            }
            if let Ok(h) = HeaderValue::from_str(accept) {
                rb = rb.header(reqwest::header::ACCEPT, h);
            }
            rb = rb.header(reqwest::header::USER_AGENT, self.user_agent());
            rb
        }

        pub fn url(&self, path: &str) -> Result<url::Url> {
            self.config
                .base_url
                .join(path.trim_start_matches('/'))
                .map_err(|e| crate::internal::error::clu::Error::Config(e.to_string()))
        }

        async fn send_with_retries(&self, rb: RequestBuilder) -> Result<Response> {
            let max = self.config.max_retries as usize;
            let mut attempt = 0usize;
            let mut rb = rb;
            loop {
                let clone = rb.try_clone().ok_or_else(|| {
                    crate::internal::error::clu::Error::Config(
                        "request could not be cloned for retries".to_string(),
                    )
                })?;
                match rb.send().await {
                    Ok(resp) => {
                        let status = resp.status();
                        if attempt < max && retry_status(status) {
                            drop(resp);
                            sleep(backoff((attempt + 1) as u32)).await;
                            rb = clone;
                            attempt += 1;
                            continue;
                        }
                        return Ok(resp);
                    }
                    Err(e) => {
                        if attempt < max && retry_error(&e) {
                            sleep(backoff((attempt + 1) as u32)).await;
                            rb = clone;
                            attempt += 1;
                            continue;
                        }
                        return Err(e.into());
                    }
                }
            }
        }

        fn api_error_from_text(status: StatusCode, text: &str, headers: HeaderMap) -> ApiError {
            let request_id = headers
                .get("request-id")
                .or_else(|| headers.get("x-request-id"))
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let body: Option<ErrorBody> = serde_json::from_str(text).ok();
            ApiError {
                status,
                body,
                raw_body: Some(text.to_string()),
                request_id,
                headers,
            }
        }

        pub async fn request_json<B: Serialize + ?Sized, T: serde::de::DeserializeOwned>(
            &self,
            method: Method,
            path: &str,
            body: Option<&B>,
            accept: &str,
        ) -> Result<(T, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self.inner.request(method, url.as_str());
            let rb = match body {
                Some(b) => self
                    .apply_default_headers(rb, accept)
                    .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                    .json(b),
                None => self.apply_default_headers(rb, accept),
            };
            let resp = self.send_with_retries(rb).await?;
            let meta = response_meta(&resp);
            let status = resp.status();
            let headers = meta.headers.clone();
            let text = resp.text().await?;
            if !status.is_success() {
                return Err(Self::api_error_from_text(status, &text, headers).into());
            }
            let v: T = serde_json::from_str(&text)?;
            Ok((v, meta))
        }

        pub async fn get_json<T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json::<serde_json::Value, T>(Method::GET, path, None, "application/json")
                .await
        }

        pub async fn delete_json<T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json::<serde_json::Value, T>(
                Method::DELETE,
                path,
                None,
                "application/json",
            )
            .await
        }

        pub async fn post_json<B: Serialize + ?Sized, T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
            body: &B,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json(Method::POST, path, Some(body), "application/json")
                .await
        }

        pub async fn post_stream_bytes<B: Serialize + ?Sized>(
            &self,
            path: &str,
            body: &B,
        ) -> Result<(
            impl Stream<Item = std::result::Result<Bytes, io::Error>> + Send,
            ResponseMeta,
        )> {
            let url = self.url(path)?;
            let rb = self
                .apply_default_headers(
                    self.inner.request(Method::POST, url.as_str()),
                    "text/event-stream",
                )
                .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                .json(body);
            let resp = self.send_with_retries(rb).await?;
            let meta = response_meta(&resp);
            let status = resp.status();
            if !status.is_success() {
                let headers = meta.headers.clone();
                let text = resp.text().await.unwrap_or_default();
                return Err(Self::api_error_from_text(status, &text, headers).into());
            }
            let st = resp.bytes_stream().map_err(std::io::Error::other);
            Ok((st, meta))
        }

        pub async fn get_bytes(&self, path: &str) -> Result<(Vec<u8>, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self.apply_default_headers(self.inner.get(url.as_str()), "*/*");
            let resp = self.send_with_retries(rb).await?;
            let meta = response_meta(&resp);
            let status = resp.status();
            let headers = meta.headers.clone();
            if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                return Err(Self::api_error_from_text(status, &text, headers).into());
            }
            let bytes = resp.bytes().await?;
            Ok((bytes.to_vec(), meta))
        }
    }
}
