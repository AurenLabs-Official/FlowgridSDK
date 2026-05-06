# FlowgridSDK

Single Rust crate **`flowgrid`** with async HTTP clients for **[OpenAI](https://platform.openai.com/docs/api-reference)** and **[Anthropic](https://docs.anthropic.com/en/api/getting-started)**. Enable either or both with Cargo features (`openai`, `anthropic`; both are on by default). The public API lives at the crate root (there are no `flowgrid::openai` / `flowgrid::anthropic` modules).

Replace placeholder `repository` / `homepage` URLs in [`crates/flowgrid/Cargo.toml`](crates/flowgrid/Cargo.toml) with your Git remote before publishing to [crates.io](https://crates.io).

Anthropic’s surface follows the Python SDK’s [`api.md`](https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/api.md) (Messages, Batches, Models, beta namespaces), without using Python at runtime.

## Cookbook

Examples assume **default features** (`openai` + `anthropic` + `tls-rustls`). With both enabled, use prefixed error types such as [`OpenAiError`](https://docs.rs/flowgrid/latest/flowgrid/type.OpenAiError.html) / [`AnthropicError`](https://docs.rs/flowgrid/latest/flowgrid/type.AnthropicError.html).

### OpenAI chat (non-streaming)

```rust,ignore
use flowgrid::{ClientBuilder, CreateChatCompletionRequest};

async fn example() -> Result<(), flowgrid::OpenAiError> {
    let client = ClientBuilder::new().api_key("sk-...").build()?;
    let req = CreateChatCompletionRequest {
        model: "gpt-4o-mini".into(),
        messages: vec![CreateChatCompletionRequest::user_message("Hello")],
        stream: Some(false),
        extra: Default::default(),
    };
    let completion = client.chat().completions().create(&req).await?;
    println!("{}", completion.message_content().unwrap_or_default());
    Ok(())
}
```

### OpenAI chat (SSE streaming)

Use [`into_unpin_event_stream`](https://docs.rs/flowgrid/latest/flowgrid/) on the decoder from `create_stream`, then [`StreamExt::next`](https://docs.rs/futures/latest/futures/stream/trait.StreamExt.html) without `pin_mut`. (`into_event_stream` remains for advanced use.) OpenAI may send `data: [DONE]`; treat non-JSON lines defensively.

```rust,ignore
use flowgrid::{ClientBuilder, CreateChatCompletionRequest};
use futures::StreamExt;

async fn example() -> Result<(), flowgrid::OpenAiError> {
    let client = ClientBuilder::new().api_key("sk-...").build()?;
    let req = CreateChatCompletionRequest {
        model: "gpt-4o-mini".into(),
        messages: vec![CreateChatCompletionRequest::user_message("Hi")],
        stream: Some(true),
        extra: Default::default(),
    };
    let (sse, _meta) = client.chat().completions().create_stream(&req).await?;
    let mut events = sse.into_unpin_event_stream();
    while let Some(item) = events.next().await {
        let ev = item?;
        if ev.data.trim() == "[DONE]" {
            break;
        }
        // e.g. serde_json::from_str::<serde_json::Value>(&ev.data)
    }
    Ok(())
}
```

### Anthropic Messages (non-streaming)

```rust,ignore
use flowgrid::{AnthropicBuilder, CreateMessageRequest};

async fn example() -> Result<(), flowgrid::AnthropicError> {
    let client = AnthropicBuilder::new().api_key("sk-ant-...").build()?;
    let body = CreateMessageRequest {
        model: "claude-3-5-haiku-20241022".into(),
        max_tokens: 256,
        messages: vec![serde_json::json!({"role": "user", "content": "Hello"})],
        stream: None,
        extra: Default::default(),
    };
    let msg = client.messages().create(&body).await?;
    println!("{}", msg.text_concat().unwrap_or_default());
    Ok(())
}
```

## TLS

Defaults use **`tls-rustls`** (`reqwest` + Rustls). For system / corporate certificate stores, disable default features and enable **`tls-native`** instead. **Do not** enable both `tls-rustls` and `tls-native` (the crate will fail to compile). Disabling all default features without one of these TLS features also fails if `openai` or `anthropic` is enabled.

## Examples

From the repo root (requires API keys in the environment):

```bash
OPENAI_API_KEY=sk-... cargo run -p flowgrid --example openai_chat_stream --features openai
ANTHROPIC_API_KEY=sk-ant-... cargo run -p flowgrid --example anthropic_message --features anthropic
```

## MSRV

Rust **1.75** (see [`crates/flowgrid/Cargo.toml`](crates/flowgrid/Cargo.toml)).

## Using `flowgrid`

```toml
flowgrid = { path = "crates/flowgrid" }
# or only one provider (enable exactly one TLS backend, e.g. `tls-rustls`):
flowgrid = { path = "crates/flowgrid", default-features = false, features = ["openai", "tls-rustls"] }
```

### Both providers enabled (default)

Types that would clash are disambiguated with prefixes, for example **`OpenAiError`** / **`AnthropicError`**, **`OpenAiResult`** / **`AnthropicResult`**, **`OpenAiWithResponse`** / **`AnthropicWithResponse`**. Transport and config also have stable prefixed type aliases: **`OpenAiHttpTransport`**, **`AnthropicHttpTransport`**, **`OpenAiClientConfig`**, **`AnthropicClientConfig`**, etc.

### One provider only

Short names apply: **`Error`**, **`Result`**, **`ClientConfig`**, **`HttpTransport`**, **`WithResponse`**, etc. refer to that provider.

## Environment

| Variable | Provider | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | OpenAI | Default API key (optional if set on the builder) |
| `OPENAI_ORG_ID` | OpenAI | `OpenAI-Organization` header |
| `OPENAI_PROJECT` | OpenAI | `OpenAI-Project` header |
| `OPENAI_BASE_URL` | OpenAI | Override API base (default `https://api.openai.com/v1`; trailing slash normalized for path joins) |
| `ANTHROPIC_API_KEY` | Anthropic | Required for config-from-env helpers |
| `ANTHROPIC_API_BASE` | Anthropic | Override API base (default `https://api.anthropic.com/v1`) |
| `ANTHROPIC_VERSION` | Anthropic | `anthropic-version` header (default `2023-06-01`) |
| `ANTHROPIC_BETA` | Anthropic | Optional `anthropic-beta` flags |

## Build features

- **`minimal`**: same as the default feature set (`openai`, `anthropic`, `tls-rustls`); useful with `default-features = false`.
- **`enterprise`**: enables `tracing` and `opentelemetry` together; combine with `full` if you need every OpenAI/Anthropic submodule as well.
- OpenAI extras: `files`, `images`, `audio`, `moderations`, `batches`, `fine_tuning`, `evals`, `assistants`, `vector_stores`, `containers`, `admin`, `webhooks`, `azure`, `realtime`, `tracing`.
- Anthropic extras: `batches`, `models`, `beta` (also gated by `anthropic`).
- **`stream-types`**: optional typed parsing for streaming `data:` JSON (OpenAI chat chunks when `openai` is on; Anthropic message stream events when `anthropic` is on).
- **`opentelemetry`**: records `flowgrid.http.request.duration_ms` via the OpenTelemetry metrics API (install a meter provider in your app).
- **`full`**: enables all optional areas above **except** TLS switching and **except** `opentelemetry` (enable `opentelemetry` explicitly when needed).

Shared feature name **`batches`** turns on batch APIs for whichever provider(s) you have enabled.

## Request hook

`ClientBuilder`, `AnthropicBuilder`, and `AzureClientBuilder` support **`request_pre_send_hook`**: a closure `Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder` run after default headers are applied and immediately before the request is sent. Use it for extra headers (for example correlation IDs). Do not log secrets there.

## Tests

```bash
cargo test -p flowgrid --features full
```

`cargo test -p flowgrid --all-features` is **not** supported because it would enable both TLS stacks. CI uses the `full` feature set instead.

Ignored live tests: set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` as appropriate and run with `--ignored`.

## Retries, `Retry-After`, and headers

The HTTP transports retry transient failures up to `max_retries` (see module docs on [`HttpTransport`](https://docs.rs/flowgrid/latest/flowgrid/)). Response statuses **408**, **409**, **429**, and **5xx** are retried (Anthropic also treats **529** as retryable). Other **4xx** are surfaced as [`OpenAiApiError`](https://docs.rs/flowgrid/latest/flowgrid/struct.OpenAiApiError.html) / [`AnthropicApiError`](https://docs.rs/flowgrid/latest/flowgrid/struct.AnthropicApiError.html) without retry. When the server sends **`Retry-After`** (seconds or HTTP-date), the wait honors it but is **capped** by `retry_after_max` on [`OpenAiClientConfig`](https://docs.rs/flowgrid/latest/flowgrid/type.OpenAiClientConfig.html) / [`AnthropicClientConfig`](https://docs.rs/flowgrid/latest/flowgrid/type.AnthropicClientConfig.html) (default **2 s**, aligned with the exponential backoff ceiling). Parsed values appear on errors as `retry_after` / `body_snippet` / `provider` ([`ProviderKind`](https://docs.rs/flowgrid/latest/flowgrid/enum.ProviderKind.html)).

Successful [`OpenAiResponseMeta`](https://docs.rs/flowgrid/latest/flowgrid/type.OpenAiResponseMeta.html) / [`AnthropicResponseMeta`](https://docs.rs/flowgrid/latest/flowgrid/type.AnthropicResponseMeta.html) optionally echo `retry_after` and common rate-limit headers when the API sends them (`x-ratelimit-*` on OpenAI, `anthropic-ratelimit-*` on Anthropic).

## Per-call timeouts

[`ExecuteOptions`](https://docs.rs/flowgrid/latest/flowgrid/struct.ExecuteOptions.html) overrides the HTTP timeout for a single request (including streaming entrypoints) without building a new client. Chat and Messages expose `create_with_options`, `create_stream_with_options`, and `create_with_response_and_options`. Lower layers also provide `get_json_with_options` / `post_json_with_options` on the transports. For cancellation beyond timeouts, you can still wrap futures with `tokio::time::timeout` or your runtime’s equivalents.

## Contract fixtures

Versioned JSON under [`crates/flowgrid/tests/fixtures/contracts/`](crates/flowgrid/tests/fixtures/contracts/) guards deserialization of public response types (`ChatCompletion`, `Message`, …). **Add or update a fixture** when you introduce a new publicly relied-on field or when a provider’s JSON shape changes; keep filenames predictable (`openai_<resource>_…`, `anthropic_…`). Tests are offline (`serde_json` only).

## Releases and semver baseline

API compatibility for the crate root `pub use` surface is checked in CI with [cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks) against a committed rustdoc JSON baseline: [`crates/flowgrid/semver/baseline_rustdoc.json`](crates/flowgrid/semver/baseline_rustdoc.json). CI regenerates current rustdoc on **nightly** with `cargo rustdoc -p flowgrid --features full -Z unstable-options -- …` and passes `--current-rustdoc target/doc/flowgrid.json` so the check never enables conflicting TLS features.

When you **release** a version that changes the public API, refresh the baseline from the same nightly invocation (from the repo root, after bumping the crate version if needed), copy `target/doc/flowgrid.json` over `crates/flowgrid/semver/baseline_rustdoc.json`, and commit it together with the release PR.

Licensed under MIT OR Apache-2.0 at your option (see crate `Cargo.toml`).
