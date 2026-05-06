# FlowgridSDK

Single Rust crate **`flowgrid`** with async HTTP clients for **[OpenAI](https://platform.openai.com/docs/api-reference)** and **[Anthropic](https://docs.anthropic.com/en/api/getting-started)**. Enable either or both with Cargo features (`openai`, `anthropic`; both are on by default). The public API lives at the crate root (there are no `flowgrid::openai` / `flowgrid::anthropic` modules).

Anthropic’s surface follows the Python SDK’s [`api.md`](https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/api.md) (Messages, Batches, Models, beta namespaces), without using Python at runtime.

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
- **`full`**: enables all optional areas above.

Shared feature name **`batches`** turns on batch APIs for whichever provider(s) you have enabled.

## Tests

```bash
cargo test -p flowgrid --all-features
```

Ignored live tests: set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` as appropriate and run with `--ignored`.

## License

Licensed under MIT OR Apache-2.0 at your option (see crate `Cargo.toml`).
