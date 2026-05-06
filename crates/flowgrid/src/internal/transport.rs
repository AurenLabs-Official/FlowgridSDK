#[cfg(feature = "openai")]
pub mod oai {
    //! HTTP transport for OpenAI-compatible APIs.
    //!
    //! ## Retries
    //!
    //! Failed **responses** are retried when the status is retryable and `max_retries` has not been
    //! exhausted: **408**, **409**, **429**, and **5xx** (including **503**). Other **4xx** are not
    //! retried. Connection/timeout/request errors from the HTTP stack may also be retried.
    //! Retries use exponential backoff (50 ms × 2ⁿ, capped at **2 s**) unless the response includes
    //! [`Retry-After`](https://www.rfc-editor.org/rfc/rfc9110.html#name-retry-after), which is
    //! parsed as delta-seconds or HTTP-date and **capped** by [`ClientConfig::retry_after_max`]
    //! (default **2 s**). **`POST` / `PATCH` / `DELETE`** can be retried on transient failures—only
    //! enable high `max_retries` when duplicate side effects are acceptable.
    use crate::internal::error::oai::{ApiError, ErrorObject, Result};
    use crate::internal::error::ProviderKind;
    use crate::internal::execute_options::ExecuteOptions;
    use crate::internal::retry_policy::{body_snippet, parse_retry_after, sleep_before_retry};
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

    /// When set on [`ClientConfig`], replaces the default rule for whether an HTTP **success**
    /// response status still triggers a retry (same timing/backoff as built-in retries).
    pub type RetryIfResponseStatusFn = Arc<dyn Fn(StatusCode, &HeaderMap) -> bool + Send + Sync>;

    /// Client configuration (clonable, shared by `OpenAI`).
    #[derive(Clone)]
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
        /// Optional hook after default headers, immediately before send.
        pub request_hook: Option<Arc<dyn Fn(RequestBuilder) -> RequestBuilder + Send + Sync>>,
        /// Default webhook secret (feature `webhooks`).
        #[cfg(feature = "webhooks")]
        pub webhook_secret: Option<String>,
        /// Ceiling for delays taken from `Retry-After` while retrying (default 2 s).
        pub retry_after_max: Duration,
        /// When set, decides whether an HTTP **success** response with this status is retried (same
        /// timing/backoff as built-in retries). When `None`, the default rule applies (see module docs).
        pub retry_if_response_status: Option<RetryIfResponseStatusFn>,
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
                request_hook: None,
                retry_after_max: Duration::from_millis(2000),
                retry_if_response_status: None,
            })
        }
    }

    impl std::fmt::Debug for ClientConfig {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut d = f.debug_struct("ClientConfig");
            d.field("api_key", &"***");
            d.field("base_url", &self.base_url);
            d.field("use_api_key_header", &self.use_api_key_header);
            d.field("default_query", &self.default_query);
            d.field("org_id", &self.org_id);
            d.field("project_id", &self.project_id);
            d.field("timeout", &self.timeout);
            d.field("max_retries", &self.max_retries);
            d.field("user_agent_suffix", &self.user_agent_suffix);
            d.field("request_hook", &self.request_hook.as_ref().map(|_| "..."));
            #[cfg(feature = "webhooks")]
            d.field(
                "webhook_secret",
                &self.webhook_secret.as_ref().map(|_| "***"),
            );
            d.field("retry_after_max", &self.retry_after_max);
            d.field(
                "retry_if_response_status",
                &self.retry_if_response_status.as_ref().map(|_| "..."),
            );
            d.finish()
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
        /// Parsed `Retry-After` when present (uncommon on success).
        pub retry_after: Option<Duration>,
        /// `x-ratelimit-remaining-requests` when present.
        pub rate_limit_remaining_requests: Option<String>,
        /// `x-ratelimit-reset-requests` when present.
        pub rate_limit_reset_requests: Option<String>,
    }

    fn header_one(headers: &HeaderMap, name: &'static str) -> Option<String> {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    }

    fn response_meta(resp: &Response) -> ResponseMeta {
        let headers = resp.headers().clone();
        let request_id = header_one(&headers, "x-request-id");
        let retry_after = parse_retry_after(&headers);
        let rate_limit_remaining_requests = header_one(&headers, "x-ratelimit-remaining-requests");
        let rate_limit_reset_requests = header_one(&headers, "x-ratelimit-reset-requests");
        ResponseMeta {
            request_id,
            status: resp.status(),
            headers,
            retry_after,
            rate_limit_remaining_requests,
            rate_limit_reset_requests,
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

        fn apply_request_hook(&self, rb: RequestBuilder) -> RequestBuilder {
            match &self.config.request_hook {
                Some(h) => h(rb),
                None => rb,
            }
        }

        async fn send_with_retries(&self, rb: RequestBuilder) -> (Result<Response>, u32) {
            let max = self.config.max_retries as usize;
            let mut attempt = 0usize;
            let mut rb = rb;
            let cap = self.config.retry_after_max;
            let mut retries = 0u32;
            loop {
                let clone = rb.try_clone().ok_or_else(|| {
                    crate::internal::error::oai::Error::Config(
                        "request could not be cloned for retries".to_string(),
                    )
                });
                let clone = match clone {
                    Ok(c) => c,
                    Err(e) => return (Err(e), retries),
                };
                match rb.send().await {
                    Ok(resp) => {
                        let status = resp.status();
                        let retry_resp =
                            if let Some(p) = self.config.retry_if_response_status.as_ref() {
                                p(status, resp.headers())
                            } else {
                                retry_status(status)
                            };
                        if attempt < max && retry_resp {
                            let headers = resp.headers().clone();
                            drop(resp);
                            retries += 1;
                            let delay =
                                sleep_before_retry(&headers, (attempt + 1) as u32, backoff, cap);
                            sleep(delay).await;
                            rb = clone;
                            attempt += 1;
                            continue;
                        }
                        return (Ok(resp), retries);
                    }
                    Err(e) => {
                        if attempt < max && retry_error(&e) {
                            retries += 1;
                            sleep(backoff((attempt + 1) as u32).min(cap)).await;
                            rb = clone;
                            attempt += 1;
                            continue;
                        }
                        return (Err(e.into()), retries);
                    }
                }
            }
        }

        async fn send_traced(
            &self,
            rb: RequestBuilder,
            method: Method,
            path: &str,
            exec: &ExecuteOptions,
        ) -> Result<Response> {
            let rb = self.apply_request_hook(rb);
            let rb = if let Some(d) = exec.timeout {
                rb.timeout(d)
            } else {
                rb
            };
            #[cfg(any(feature = "tracing", feature = "opentelemetry"))]
            let start = std::time::Instant::now();
            #[cfg(feature = "tracing")]
            let (result, retry_count) = {
                use tracing::Instrument;
                let method_for_span = method.clone();
                let span = tracing::info_span!(
                    "flowgrid.http.request",
                    flowgrid.provider = "openai",
                    http.request.method = %method_for_span.as_str(),
                    flowgrid.api.path = path,
                    flowgrid.retry_count = tracing::field::Empty,
                    flowgrid.request_id = tracing::field::Empty,
                    flowgrid.ratelimit.requests.remaining = tracing::field::Empty,
                    flowgrid.ratelimit.requests.reset = tracing::field::Empty,
                );
                let span_record = span.clone();
                async move {
                    tracing::debug!(
                        target: "flowgrid_http",
                        provider = "openai",
                        method = method_for_span.as_str(),
                        path,
                        "request",
                    );
                    let (result, retry_count) = self.send_with_retries(rb).await;
                    span_record.record("flowgrid.retry_count", retry_count);
                    if let Ok(ref resp) = result {
                        let meta = response_meta(resp);
                        if let Some(ref id) = meta.request_id {
                            span_record.record("flowgrid.request_id", id.as_str());
                        }
                        if let Some(ref v) = meta.rate_limit_remaining_requests {
                            span_record.record("flowgrid.ratelimit.requests.remaining", v.as_str());
                        }
                        if let Some(ref v) = meta.rate_limit_reset_requests {
                            span_record.record("flowgrid.ratelimit.requests.reset", v.as_str());
                        }
                    }
                    (result, retry_count)
                }
                .instrument(span)
                .await
            };
            #[cfg(not(feature = "tracing"))]
            let (result, retry_count) = self.send_with_retries(rb).await;
            #[cfg(any(feature = "tracing", feature = "opentelemetry"))]
            {
                let elapsed_ms = start.elapsed().as_millis() as f64;
                #[cfg(feature = "tracing")]
                match &result {
                    Ok(resp) => {
                        tracing::debug!(
                            target: "flowgrid_http",
                            provider = "openai",
                            method = method.as_str(),
                            path,
                            status = %resp.status(),
                            elapsed_ms = elapsed_ms as u64,
                            flowgrid.retry_count = retry_count,
                            "response",
                        );
                    }
                    Err(_) => {
                        tracing::debug!(
                            target: "flowgrid_http",
                            provider = "openai",
                            method = method.as_str(),
                            path,
                            elapsed_ms = elapsed_ms as u64,
                            flowgrid.retry_count = retry_count,
                            "request_failed",
                        );
                    }
                }
                #[cfg(feature = "opentelemetry")]
                crate::internal::otel_http::record_duration_ms(
                    elapsed_ms,
                    "openai",
                    method.as_str(),
                    path,
                    result.as_ref().ok().map(|r| r.status()),
                    retry_count,
                );
            }
            #[cfg(not(any(feature = "tracing", feature = "opentelemetry")))]
            {
                let _ = (method.as_str(), path, retry_count);
            }
            result
        }

        fn api_error_from_text(status: StatusCode, text: &str, headers: HeaderMap) -> ApiError {
            let request_id = headers
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let body: Option<ErrorObject> = serde_json::from_str(text).ok();
            let retry_after = parse_retry_after(&headers);
            ApiError {
                status,
                body,
                raw_body: Some(text.to_string()),
                body_snippet: body_snippet(text),
                retry_after,
                provider: ProviderKind::OpenAi,
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
            exec: ExecuteOptions,
        ) -> Result<(T, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self.inner.request(method.clone(), url.as_str());
            let rb = match body {
                Some(b) => self
                    .apply_default_headers(rb)
                    .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                    .json(b),
                None => self.apply_default_headers(rb),
            };
            let resp = self.send_traced(rb, method, path, &exec).await?;
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
            self.request_json::<serde_json::Value, T>(
                Method::GET,
                path,
                None,
                ExecuteOptions::default(),
            )
            .await
        }

        /// GET JSON with per-call options (e.g. [`ExecuteOptions::timeout`](ExecuteOptions::timeout)).
        pub async fn get_json_with_options<T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
            exec: ExecuteOptions,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json::<serde_json::Value, T>(Method::GET, path, None, exec)
                .await
        }

        /// GET JSON with extra query pairs (appended after [`ClientConfig::default_query`](ClientConfig::default_query)).
        pub async fn get_json_query<T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
            query: &[(String, String)],
        ) -> Result<(T, ResponseMeta)> {
            let mut u = self.url(path)?;
            if !query.is_empty() {
                let mut pairs = u.query_pairs_mut();
                for (k, v) in query {
                    pairs.append_pair(k.as_str(), v.as_str());
                }
            }
            let rb = self.apply_default_headers(self.inner.get(u.as_str()));
            let resp = self
                .send_traced(rb, Method::GET, path, &ExecuteOptions::default())
                .await?;
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

        /// GET raw bytes (e.g. file downloads, audio).
        pub async fn get_bytes(&self, path: &str) -> Result<(Vec<u8>, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self.apply_default_headers(self.inner.get(url.as_str()));
            let resp = self
                .send_traced(rb, Method::GET, path, &ExecuteOptions::default())
                .await?;
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
            self.request_json::<serde_json::Value, T>(
                Method::DELETE,
                path,
                None,
                ExecuteOptions::default(),
            )
            .await
        }

        /// POST JSON.
        pub async fn post_json<B: Serialize + ?Sized, T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
            body: &B,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json(Method::POST, path, Some(body), ExecuteOptions::default())
                .await
        }

        /// POST JSON with per-call options.
        pub async fn post_json_with_options<
            B: Serialize + ?Sized,
            T: serde::de::DeserializeOwned,
        >(
            &self,
            path: &str,
            body: &B,
            exec: ExecuteOptions,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json(Method::POST, path, Some(body), exec)
                .await
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
            let resp = self
                .send_traced(rb, Method::POST, path, &ExecuteOptions::default())
                .await?;
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
            self.request_json(Method::PATCH, path, Some(body), ExecuteOptions::default())
                .await
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
            self.post_stream_bytes_with_options(path, body, ExecuteOptions::default())
                .await
        }

        /// [`Self::post_stream_bytes`] with per-call options.
        pub async fn post_stream_bytes_with_options<B: Serialize + ?Sized>(
            &self,
            path: &str,
            body: &B,
            exec: ExecuteOptions,
        ) -> Result<(
            impl Stream<Item = std::result::Result<Bytes, io::Error>> + Send,
            ResponseMeta,
        )> {
            let url = self.url(path)?;
            let rb = self
                .apply_default_headers(self.inner.request(Method::POST, url.as_str()))
                .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                .json(body);
            let resp = self.send_traced(rb, Method::POST, path, &exec).await?;
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
            let resp = self
                .send_traced(rb, Method::POST, path, &ExecuteOptions::default())
                .await?;
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
            let resp = self
                .send_traced(rb, Method::POST, path, &ExecuteOptions::default())
                .await?;
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
    //! HTTP transport for Anthropic APIs.
    //!
    //! ## Retries
    //!
    //! Retries **408**, **409**, **429**, **529**, and **5xx** when attempts remain, plus transient
    //! connection errors. Uses exponential backoff (50 ms × 2ⁿ, cap **2 s**) unless `Retry-After` is
    //! set (delta-seconds or HTTP-date), capped by [`ClientConfig::retry_after_max`]. **`POST`**
    //! may retry—tune `max_retries` if duplicate side effects are unacceptable.
    use crate::internal::error::clu::{ApiError, ErrorBody, Result};
    use crate::internal::error::ProviderKind;
    use crate::internal::execute_options::ExecuteOptions;
    use crate::internal::retry_policy::{body_snippet, parse_retry_after, sleep_before_retry};
    use bytes::Bytes;
    use futures::{Stream, TryStreamExt};
    use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
    use reqwest::{Client, Method, RequestBuilder, Response, StatusCode};
    use serde::Serialize;
    use std::io;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::sleep;

    /// When set on [`ClientConfig`], replaces the default rule for whether an HTTP **success**
    /// response status still triggers a retry.
    pub type RetryIfResponseStatusFn = Arc<dyn Fn(StatusCode, &HeaderMap) -> bool + Send + Sync>;

    /// Client configuration.
    #[derive(Clone)]
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
        pub request_hook: Option<Arc<dyn Fn(RequestBuilder) -> RequestBuilder + Send + Sync>>,
        /// Ceiling for delays taken from `Retry-After` while retrying (default 2 s).
        pub retry_after_max: Duration,
        /// When set, replaces the default rule for whether a successful HTTP response status triggers a retry.
        pub retry_if_response_status: Option<RetryIfResponseStatusFn>,
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
                request_hook: None,
                retry_after_max: Duration::from_millis(2000),
                retry_if_response_status: None,
            })
        }
    }

    impl std::fmt::Debug for ClientConfig {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ClientConfig")
                .field("api_key", &"***")
                .field("base_url", &self.base_url)
                .field("anthropic_version", &self.anthropic_version)
                .field("anthropic_beta", &self.anthropic_beta)
                .field("timeout", &self.timeout)
                .field("max_retries", &self.max_retries)
                .field("user_agent_suffix", &self.user_agent_suffix)
                .field("request_hook", &self.request_hook.as_ref().map(|_| "..."))
                .field("retry_after_max", &self.retry_after_max)
                .field(
                    "retry_if_response_status",
                    &self.retry_if_response_status.as_ref().map(|_| "..."),
                )
                .finish()
        }
    }

    #[derive(Debug, Clone)]
    pub struct ResponseMeta {
        pub request_id: Option<String>,
        pub status: StatusCode,
        pub headers: HeaderMap,
        pub retry_after: Option<Duration>,
        /// `anthropic-ratelimit-requests-remaining` when present.
        pub rate_limit_remaining_requests: Option<String>,
        /// `anthropic-ratelimit-requests-reset` when present.
        pub rate_limit_reset_requests: Option<String>,
    }

    fn header_one(headers: &HeaderMap, name: &'static str) -> Option<String> {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    }

    fn response_meta(resp: &Response) -> ResponseMeta {
        let headers = resp.headers().clone();
        let request_id =
            header_one(&headers, "request-id").or_else(|| header_one(&headers, "x-request-id"));
        let retry_after = parse_retry_after(&headers);
        let rate_limit_remaining_requests =
            header_one(&headers, "anthropic-ratelimit-requests-remaining");
        let rate_limit_reset_requests = header_one(&headers, "anthropic-ratelimit-requests-reset");
        ResponseMeta {
            request_id,
            status: resp.status(),
            headers,
            retry_after,
            rate_limit_remaining_requests,
            rate_limit_reset_requests,
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

        fn apply_request_hook(&self, rb: RequestBuilder) -> RequestBuilder {
            match &self.config.request_hook {
                Some(h) => h(rb),
                None => rb,
            }
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

        async fn send_with_retries(&self, rb: RequestBuilder) -> (Result<Response>, u32) {
            let max = self.config.max_retries as usize;
            let mut attempt = 0usize;
            let mut rb = rb;
            let cap = self.config.retry_after_max;
            let mut retries = 0u32;
            loop {
                let clone = rb.try_clone().ok_or_else(|| {
                    crate::internal::error::clu::Error::Config(
                        "request could not be cloned for retries".to_string(),
                    )
                });
                let clone = match clone {
                    Ok(c) => c,
                    Err(e) => return (Err(e), retries),
                };
                match rb.send().await {
                    Ok(resp) => {
                        let status = resp.status();
                        let retry_resp =
                            if let Some(p) = self.config.retry_if_response_status.as_ref() {
                                p(status, resp.headers())
                            } else {
                                retry_status(status)
                            };
                        if attempt < max && retry_resp {
                            let headers = resp.headers().clone();
                            drop(resp);
                            retries += 1;
                            let delay =
                                sleep_before_retry(&headers, (attempt + 1) as u32, backoff, cap);
                            sleep(delay).await;
                            rb = clone;
                            attempt += 1;
                            continue;
                        }
                        return (Ok(resp), retries);
                    }
                    Err(e) => {
                        if attempt < max && retry_error(&e) {
                            retries += 1;
                            sleep(backoff((attempt + 1) as u32).min(cap)).await;
                            rb = clone;
                            attempt += 1;
                            continue;
                        }
                        return (Err(e.into()), retries);
                    }
                }
            }
        }

        async fn send_traced(
            &self,
            rb: RequestBuilder,
            method: Method,
            path: &str,
            exec: &ExecuteOptions,
        ) -> Result<Response> {
            let rb = self.apply_request_hook(rb);
            let rb = if let Some(d) = exec.timeout {
                rb.timeout(d)
            } else {
                rb
            };
            #[cfg(any(feature = "tracing", feature = "opentelemetry"))]
            let start = std::time::Instant::now();
            #[cfg(feature = "tracing")]
            let (result, retry_count) = {
                use tracing::Instrument;
                let method_for_span = method.clone();
                let span = tracing::info_span!(
                    "flowgrid.http.request",
                    flowgrid.provider = "anthropic",
                    http.request.method = %method_for_span.as_str(),
                    flowgrid.api.path = path,
                    flowgrid.retry_count = tracing::field::Empty,
                    flowgrid.request_id = tracing::field::Empty,
                    flowgrid.ratelimit.requests.remaining = tracing::field::Empty,
                    flowgrid.ratelimit.requests.reset = tracing::field::Empty,
                );
                let span_record = span.clone();
                async move {
                    tracing::debug!(
                        target: "flowgrid_http",
                        provider = "anthropic",
                        method = method_for_span.as_str(),
                        path,
                        "request",
                    );
                    let (result, retry_count) = self.send_with_retries(rb).await;
                    span_record.record("flowgrid.retry_count", retry_count);
                    if let Ok(ref resp) = result {
                        let meta = response_meta(resp);
                        if let Some(ref id) = meta.request_id {
                            span_record.record("flowgrid.request_id", id.as_str());
                        }
                        if let Some(ref v) = meta.rate_limit_remaining_requests {
                            span_record.record("flowgrid.ratelimit.requests.remaining", v.as_str());
                        }
                        if let Some(ref v) = meta.rate_limit_reset_requests {
                            span_record.record("flowgrid.ratelimit.requests.reset", v.as_str());
                        }
                    }
                    (result, retry_count)
                }
                .instrument(span)
                .await
            };
            #[cfg(not(feature = "tracing"))]
            let (result, retry_count) = self.send_with_retries(rb).await;
            #[cfg(any(feature = "tracing", feature = "opentelemetry"))]
            {
                let elapsed_ms = start.elapsed().as_millis() as f64;
                #[cfg(feature = "tracing")]
                match &result {
                    Ok(resp) => {
                        tracing::debug!(
                            target: "flowgrid_http",
                            provider = "anthropic",
                            method = method.as_str(),
                            path,
                            status = %resp.status(),
                            elapsed_ms = elapsed_ms as u64,
                            flowgrid.retry_count = retry_count,
                            "response",
                        );
                    }
                    Err(_) => {
                        tracing::debug!(
                            target: "flowgrid_http",
                            provider = "anthropic",
                            method = method.as_str(),
                            path,
                            elapsed_ms = elapsed_ms as u64,
                            flowgrid.retry_count = retry_count,
                            "request_failed",
                        );
                    }
                }
                #[cfg(feature = "opentelemetry")]
                crate::internal::otel_http::record_duration_ms(
                    elapsed_ms,
                    "anthropic",
                    method.as_str(),
                    path,
                    result.as_ref().ok().map(|r| r.status()),
                    retry_count,
                );
            }
            #[cfg(not(any(feature = "tracing", feature = "opentelemetry")))]
            {
                let _ = (method.as_str(), path, retry_count);
            }
            result
        }

        fn api_error_from_text(status: StatusCode, text: &str, headers: HeaderMap) -> ApiError {
            let request_id = headers
                .get("request-id")
                .or_else(|| headers.get("x-request-id"))
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let body: Option<ErrorBody> = serde_json::from_str(text).ok();
            let retry_after = parse_retry_after(&headers);
            ApiError {
                status,
                body,
                raw_body: Some(text.to_string()),
                body_snippet: body_snippet(text),
                retry_after,
                provider: ProviderKind::Anthropic,
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
            exec: ExecuteOptions,
        ) -> Result<(T, ResponseMeta)> {
            let url = self.url(path)?;
            let rb = self.inner.request(method.clone(), url.as_str());
            let rb = match body {
                Some(b) => self
                    .apply_default_headers(rb, accept)
                    .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                    .json(b),
                None => self.apply_default_headers(rb, accept),
            };
            let resp = self.send_traced(rb, method, path, &exec).await?;
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
            self.request_json::<serde_json::Value, T>(
                Method::GET,
                path,
                None,
                "application/json",
                ExecuteOptions::default(),
            )
            .await
        }

        pub async fn get_json_with_options<T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
            exec: ExecuteOptions,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json::<serde_json::Value, T>(
                Method::GET,
                path,
                None,
                "application/json",
                exec,
            )
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
                ExecuteOptions::default(),
            )
            .await
        }

        pub async fn post_json<B: Serialize + ?Sized, T: serde::de::DeserializeOwned>(
            &self,
            path: &str,
            body: &B,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json(
                Method::POST,
                path,
                Some(body),
                "application/json",
                ExecuteOptions::default(),
            )
            .await
        }

        pub async fn post_json_with_options<
            B: Serialize + ?Sized,
            T: serde::de::DeserializeOwned,
        >(
            &self,
            path: &str,
            body: &B,
            exec: ExecuteOptions,
        ) -> Result<(T, ResponseMeta)> {
            self.request_json(Method::POST, path, Some(body), "application/json", exec)
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
            self.post_stream_bytes_with_options(path, body, ExecuteOptions::default())
                .await
        }

        pub async fn post_stream_bytes_with_options<B: Serialize + ?Sized>(
            &self,
            path: &str,
            body: &B,
            exec: ExecuteOptions,
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
            let resp = self.send_traced(rb, Method::POST, path, &exec).await?;
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
            let resp = self
                .send_traced(rb, Method::GET, path, &ExecuteOptions::default())
                .await?;
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
