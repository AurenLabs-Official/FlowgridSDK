# Resilience and retries

## Built-in retry behavior

See the README sections on retries and per-call timeouts. The HTTP transports retry transient **`GET`/`POST`/…** failures when configured **`max_retries`** allows it, using exponential backoff and optional **`Retry-After`** (capped by **`retry_after_max`**).

## Custom retry predicate

Set **`retry_if_response_status`** on **`OpenAiClientConfig`** / **`AnthropicClientConfig`** via **`ClientBuilder::retry_if_response_status`** or **`AnthropicBuilder::retry_if_response_status`**. When **`Some`**, it **replaces** the built-in rule for whether a received **HTTP status** should trigger another attempt **before the response body is read**. When **`None`**, defaults apply (e.g. **408**, **429**, **5xx** as documented in the README).

## Rate limits and `Retry-After`

On **429**, the API may send **`Retry-After`**. The client parses it into wait time (capped by **`retry_after_max`**). Rate-limit headers are surfaced on response metadata and in **`tracing`** / docs for dashboards.

With Cargo feature **`rate-aware-retry`** and **`ClientBuilder::rate_limit_aware_backoff(true)`** (or the corresponding field on **`OpenAiClientConfig`** / **`AnthropicClientConfig`**), if **`Retry-After` is absent** on a retriable response, the client may derive a **single** wait from provider rate-limit **reset** headers when the **remaining** count hits zero (OpenAI **`x-ratelimit-*`**, Anthropic **`anthropic-ratelimit-*`**). This is **not** a duplicate of **`Retry-After`**: **`Retry-After` still wins** when present.

## Circuit breakers and bulkheads

**Out of scope for the core crate:** circuit breakers, bulkheads, or global concurrency limits. Implement those around your calls to the high-level clients or wrap the transport in your app.
