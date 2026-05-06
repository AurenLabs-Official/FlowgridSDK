# HTTP, TLS, and proxies

`flowgrid` uses **`reqwest`**. You can tune behavior as follows:

| Layer | What controls it | Notes |
|--------|------------------|--------|
| TLS stack | Cargo features **`tls-rustls`** (default) or **`tls-native`** | Exactly one; see README. |
| Corporate proxy | Environment **`HTTP_PROXY`**, **`HTTPS_PROXY`**, **`NO_PROXY`** | Honored by `reqwest` when set in the process environment. |
| Connect / per-attempt timeout | **`ClientConfig::timeout`** (builder / from_env) | Applies to each internal HTTP attempt. |
| Single-call override | **`ExecuteOptions::timeout`** | Shorter or longer bound for one request (including stream setup). |
| Reading an SSE body | Active transport timeout + optional **`tokio::time::timeout`** around each **`next()`** | No separate SDK “stream chunk timeout” type. |

## Custom `reqwest::Client`

The transport builds its own **`reqwest::Client`**. There is **no** stable public injection hook today. Use **`request_pre_send_hook`** for per-request headers. Advanced TLS/proxy behavior generally follows **`reqwest`** + environment variables.

## OpenAI-compatible HTTP endpoints

Set **`OPENAI_BASE_URL`** or **`ClientBuilder::base_url`** to any **OpenAI-shaped** API (**`…/v1/…`**). Gateways (LiteLLM, vLLM HTTP, etc.) often work **best-effort**; validate the routes you need. Azure OpenAI should use **`AzureClientBuilder`** (`api-key`, **`api-version`** query).
