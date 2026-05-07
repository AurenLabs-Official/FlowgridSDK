# HTTP, TLS, and proxies

`flowgrid` uses **`reqwest`**. You can tune behavior as follows:

| Layer | What controls it | Notes |
|--------|------------------|--------|
| TLS stack | Cargo features **`tls-rustls`** (default) or **`tls-native`** | Exactly one; see README. |
| Corporate proxy | Environment **`HTTP_PROXY`**, **`HTTPS_PROXY`**, **`NO_PROXY`** | Honored by `reqwest` when set in the process environment. |
| Connect / per-attempt timeout | **`ClientConfig::timeout`** (builder / from_env) | Applies to each internal HTTP attempt. |
| Single-call override | **`ExecuteOptions::timeout`** | Shorter or longer bound for one request (including stream setup). |
| Reading an SSE body | Active transport timeout + optional **`tokio::time::timeout`** around each **`next()`** | No separate SDK “stream chunk timeout” type. |
| Customize `reqwest` client | **`ClientBuilder::http_client_builder_hook`** / **`OpenAiClientConfig::http_client_builder_hook`** (same for Anthropic / Azure) | Runs **after** the SDK applies **`timeout`** on **`reqwest::ClientBuilder`**. Use for **custom connectors** (corporate HTTP(S) proxy with auth, **mTLS** client certificates, HTTP/2 tuning). **Do not** log secrets. |

## Custom `reqwest::Client` construction

Set **`http_client_builder_hook`** to adjust the shared **`Client::builder()`** before it is built. The SDK always calls **`ClientBuilder::timeout`** with **`ClientConfig::timeout`** first; your hook may chain further (**`.pool_max_idle_per_host`**, **`.http2_prior_knowledge()`**, **`.use_rustls_tls()`** is already implied by feature flags, etc.).

For **mTLS**, configure a **`reqwest`** TLS builder / connector in the hook (see the **`reqwest`** docs for your TLS backend). **Enterprise HTTP proxies** usually work via **`HTTP(S)_PROXY`**; the hook is for cases that need a bespoke connector.

Use **`request_pre_send_hook`** when you only need per-request headers.

## OpenAI-compatible HTTP endpoints

Set **`OPENAI_BASE_URL`** or **`ClientBuilder::base_url`** to any **OpenAI-shaped** API (**`…/v1/…`**). Gateways (LiteLLM, vLLM HTTP, etc.) often work **best-effort**; validate the routes you need.

Optional Cargo feature **`compat-openai`** + **`ClientBuilder::openai_http_compatible_profile()`** applies conservative defaults documented in the README; it does **not** substitute for verifying your gateway’s route support.

Azure OpenAI should use **`AzureClientBuilder`** (`api-key`, **`api-version`** query).
