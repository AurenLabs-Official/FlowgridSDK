# FlowgridSDK

Single Rust crate **`flowgrid`** with async HTTP clients for **[OpenAI](https://platform.openai.com/docs/api-reference)** and **[Anthropic](https://docs.anthropic.com/en/api/getting-started)**. Enable either or both with Cargo features (`openai`, `anthropic`; both are on by default). The public API lives at the crate root (there are no `flowgrid::openai` / `flowgrid::anthropic` modules).

Replace placeholder `repository` / `homepage` URLs in [`crates/flowgrid/Cargo.toml`](crates/flowgrid/Cargo.toml) with your Git remote before publishing to [crates.io](https://crates.io).

Anthropic’s surface follows the Python SDK’s [`api.md`](https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/api.md) (Messages, Batches, Models, beta namespaces), without using Python at runtime.

## Cookbook

Examples assume **default features** (`openai` + `anthropic`). With both enabled, use prefixed error types such as [`OpenAiError`](https://docs.rs/flowgrid/latest/flowgrid/type.OpenAiError.html) / [`AnthropicError`](https://docs.rs/flowgrid/latest/flowgrid/type.AnthropicError.html).

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

Use [`into_event_stream`](https://docs.rs/flowgrid/latest/flowgrid/) on the decoder from `create_stream`, then [`StreamExt::next`](https://docs.rs/futures/latest/futures/stream/trait.StreamExt.html). OpenAI may send `data: [DONE]`; treat non-JSON lines defensively.

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
    let mut events = sse.into_event_stream();
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

## MSRV

Rust **1.75** (see [`crates/flowgrid/Cargo.toml`](crates/flowgrid/Cargo.toml)).

## Using `flowgrid`

```toml
flowgrid = { path = "crates/flowgrid" }
# or only one provider:
flowgrid = { path = "crates/flowgrid", default-features = false, features = ["openai"] }
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

- **`openai`** / **`anthropic`**: compile the corresponding client.
- OpenAI extras: `files`, `images`, `audio`, `moderations`, `batches`, `fine_tuning`, `evals`, `assistants`, `vector_stores`, `containers`, `admin`, `webhooks`, `azure`, `realtime`, `tracing`.
- Anthropic extras: `batches`, `models`, `beta` (also gated by `anthropic`).
- **`stream-types`**: optional typed parsing helpers for OpenAI streaming SSE payloads (requires `openai`).
- **`full`**: enables all optional areas above.

Shared feature name **`batches`** turns on batch APIs for whichever provider(s) you have enabled.

## Tests

```bash
cargo test -p flowgrid --all-features
```

Ignored live tests: set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` as appropriate and run with `--ignored`.

## License

Licensed under MIT OR Apache-2.0 at your option (see crate `Cargo.toml`).
