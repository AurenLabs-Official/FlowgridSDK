# FlowgridSDK

**Same mental model. Two providers. One Rust type system.**

`flowgrid` is a **production-oriented control plane** for calling **[OpenAI](https://platform.openai.com/docs/api-reference)** and **[Anthropic](https://docs.anthropic.com/en/api/getting-started)** from async Rust—not a minimal thin wrapper, but **predictable** behavior you can operate: shared patterns for **retries** (`Retry-After`, caps), **per-call timeouts**, **structured errors** (`body_snippet`, `retry_after`, `ProviderKind`), **SSE streaming** (including `Unpin` event streams), and **response metadata** (request ids, rate-limit headers where exposed).

Enable either or both providers with Cargo features (`openai`, `anthropic`; both default on). The stable API is the **crate root** `pub use` surface (no `flowgrid::openai` / `flowgrid::anthropic` modules).

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

### OpenAI chat (SSE streaming + cooperative cancel)

Use [`stream_next_until_cancelled`](https://docs.rs/flowgrid/latest/flowgrid/fn.stream_next_until_cancelled.html) (feature **`cancel`**) together with a [`CancellationToken`](https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html) so shutdown or `tokio::select!` can stop draining events **without** waiting for the HTTP timeout. Combine with **[`ExecuteOptions::timeout`](https://docs.rs/flowgrid/latest/flowgrid/struct.ExecuteOptions.html)** for per-call network bounds.

```rust,ignore
use flowgrid::{stream_next_until_cancelled, ClientBuilder, CreateChatCompletionRequest};
use tokio_util::sync::CancellationToken;

async fn example() -> Result<(), flowgrid::OpenAiError> {
    let cancel = CancellationToken::new();
    let client = ClientBuilder::new().api_key("sk-...").build()?;
    let req = CreateChatCompletionRequest {
        model: "gpt-4o-mini".into(),
        messages: vec![CreateChatCompletionRequest::user_message("Hi")],
        stream: Some(true),
        extra: Default::default(),
    };
    let (sse, _meta) = client.chat().completions().create_stream(&req).await?;
    let mut events = sse.into_unpin_event_stream();
    loop {
        let Some(item) = stream_next_until_cancelled(&mut events, &cancel).await else {
            break; // cancelled
        };
        let ev = item?;
        if ev.data.trim() == "[DONE]" {
            break;
        }
    }
    Ok(())
}
```

For one-shot timeouts around a **single** `next()` only, `tokio::time::timeout(duration, events.next())` is fine; prefer **`cancel`** when the same token is shared across tasks (graceful shutdown).

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

## TLS, proxies, and timeouts

Defaults use **`tls-rustls`** (`reqwest` + Rustls). For system / corporate certificate stores, disable default features and enable **`tls-native`** instead. **Do not** enable both `tls-rustls` and `tls-native` (the crate will fail to compile). Disabling all default features without one of these TLS features also fails if `openai` or `anthropic` is enabled.

**Proxies:** `reqwest` honors **`HTTP_PROXY`**, **`HTTPS_PROXY`**, and **`NO_PROXY`** when set in the process environment. **Timeout layers** (client default vs [`ExecuteOptions`](https://docs.rs/flowgrid/latest/flowgrid/struct.ExecuteOptions.html) vs streaming `next()` loops) are summarized in [`docs/http.md`](docs/http.md).

## Examples

From the repo root (requires API keys in the environment):

```bash
OPENAI_API_KEY=sk-... cargo run -p flowgrid --example openai_chat_stream --features openai
OPENAI_API_KEY=sk-... OPENAI_ASSISTANT_ID=asst_... cargo run -p flowgrid --example openai_assistants_e2e --features openai,assistants
OPENAI_API_KEY=sk-... cargo run -p flowgrid --example openai_responses_stream_accumulate --features openai,stream-types
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
| `OPENAI_ASSISTANT_ID` | OpenAI | Existing assistant id for the `openai_assistants_e2e` example only |
| `ANTHROPIC_API_KEY` | Anthropic | Required for config-from-env helpers |
| `ANTHROPIC_API_BASE` | Anthropic | Override API base (default `https://api.anthropic.com/v1`) |
| `ANTHROPIC_VERSION` | Anthropic | `anthropic-version` header (default `2023-06-01`) |
| `ANTHROPIC_BETA` | Anthropic | Optional `anthropic-beta` flags |

## Build features

- **`minimal`**: same as the default feature set (`openai`, `anthropic`, `tls-rustls`); useful with `default-features = false`.
- **`enterprise`**: enables `tracing` and `opentelemetry` together; combine with `full` if you need every OpenAI/Anthropic submodule as well.
- OpenAI extras: `files`, `images`, `audio`, `moderations`, `batches`, `fine_tuning`, `evals`, `assistants`, `vector_stores`, `containers`, `admin`, `webhooks`, `azure`, `realtime`, `tracing`.
- Anthropic extras: `batches`, `models`, `beta` (also gated by `anthropic`).
- **`stream-types`**: optional typed parsing for streaming `data:` JSON (OpenAI chat chunks when `openai` is on; Anthropic message stream events when `anthropic` is on). Includes bounded helpers **`accumulate_openai_chat_visible_text_into`** / **`accumulate_openai_response_visible_text_into`** (see rustdoc).
- **`rate-aware-retry`**: optional wait hints from provider rate-limit reset headers when **`Retry-After`** is absent (see [`docs/resilience.md`](docs/resilience.md)).
- **`compat-openai`**: optional **`ClientBuilder::openai_http_compatible_profile()`** for OpenAI-shaped gateways; see **OpenAI-compatible HTTP servers** above.
- **`sse-fuzz`**: unstable **`sse_fuzz_support`** module for local **`cargo fuzz`** targets only (not semver-stable).
- **`opentelemetry`**: records `flowgrid.http.request.duration_ms` via the OpenTelemetry metrics API (install a meter provider in your app). Span/metric naming and dashboard hints are documented in [`docs/observability.md`](docs/observability.md).
- **`cancel`**: exposes [`stream_next_until_cancelled`](https://docs.rs/flowgrid/latest/flowgrid/fn.stream_next_until_cancelled.html) for cooperative shutdown while reading SSE/event streams (see Cookbook).
- **`full`**: enables all optional areas above **except** TLS switching and **except** `opentelemetry` (enable `opentelemetry` explicitly when needed).

Shared feature name **`batches`** turns on batch APIs for whichever provider(s) you have enabled.

### `full` vs `enterprise`

- **`full`**: pulls in the optional OpenAI/Anthropic surface area (Assistants, files, Azure, …) **and** enables **`tracing`** spans for outbound HTTP, but it does **not** enable the **`opentelemetry`** Cargo feature (no histogram metric emission from this crate until you add it).
- **`enterprise`**: enables **`tracing`** and **`opentelemetry`** together so `flowgrid.http.request.duration_ms` is recorded when your app installs an OpenTelemetry meter provider (see [`docs/observability.md`](docs/observability.md)).
- **Typical “all modules + metrics” setup:** keep **exactly one** TLS feature (`tls-rustls` or `tls-native`) and combine bundles explicitly, for example:

```toml
flowgrid = { version = "…", default-features = false, features = ["openai", "anthropic", "tls-rustls", "full", "enterprise"] }
```

Do **not** use `cargo ... --all-features`: it enables both TLS backends and fails to compile.

## Request hooks

`ClientBuilder`, `AnthropicBuilder`, and `AzureClientBuilder` support **`request_pre_send_hook`** for per-request headers and **`http_client_builder_hook`** to customize the shared **`reqwest::Client`** (see [`docs/http.md`](docs/http.md)).

## OpenAI-compatible HTTP servers

Many gateways expose an OpenAI-shaped **`/v1/...`** surface. **`flowgrid`** targets the official OpenAI routes used in this crate; forks may omit or diverge on **Assistants**, **Responses**, **Realtime**, etc. Enable optional feature **`compat-openai`** and call **`ClientBuilder::openai_http_compatible_profile()`** for conservative defaults; you still set **`base_url`** and should treat coverage as **best-effort** until you validate endpoints.

## Tests

```bash
cargo test -p flowgrid --features full
```

`cargo test -p flowgrid --all-features` is **not** supported because it would enable both TLS stacks. CI uses the `full` feature set instead.

On **Windows**, parallel test binaries can occasionally trigger linker error `LNK1104` (“cannot open file …exe”). If that happens, run `cargo test -p flowgrid --features full -j 1` or close programs that lock files under `target/`.

Ignored live tests: set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` as appropriate and run with `--ignored`.

## Retries, `Retry-After`, and headers

The HTTP transports retry transient failures up to `max_retries` (see rustdoc on [`OpenAiHttpTransport`](https://docs.rs/flowgrid/latest/flowgrid/type.OpenAiHttpTransport.html) / [`AnthropicHttpTransport`](https://docs.rs/flowgrid/latest/flowgrid/type.AnthropicHttpTransport.html)). Response statuses **408**, **409**, **429**, and **5xx** are retried (Anthropic also treats **529** as retryable). Other **4xx** are surfaced as [`OpenAiApiError`](https://docs.rs/flowgrid/latest/flowgrid/struct.OpenAiApiError.html) / [`AnthropicApiError`](https://docs.rs/flowgrid/latest/flowgrid/struct.AnthropicApiError.html) without retry. When the server sends **`Retry-After`** (seconds or HTTP-date), the wait honors it but is **capped** by `retry_after_max` on [`OpenAiClientConfig`](https://docs.rs/flowgrid/latest/flowgrid/type.OpenAiClientConfig.html) / [`AnthropicClientConfig`](https://docs.rs/flowgrid/latest/flowgrid/type.AnthropicClientConfig.html) (default **2 s**, aligned with the exponential backoff ceiling). Parsed values appear on errors as `retry_after` / `body_snippet` / `provider` ([`ProviderKind`](https://docs.rs/flowgrid/latest/flowgrid/enum.ProviderKind.html)).

Successful [`OpenAiResponseMeta`](https://docs.rs/flowgrid/latest/flowgrid/type.OpenAiResponseMeta.html) / [`AnthropicResponseMeta`](https://docs.rs/flowgrid/latest/flowgrid/type.AnthropicResponseMeta.html) optionally echo `retry_after` and common rate-limit headers when the API sends them (`x-ratelimit-*` on OpenAI, `anthropic-ratelimit-*` on Anthropic).

Advanced: **`retry_if_response_status`** on the builders replaces the default “retry this status” rule (see [`docs/resilience.md`](docs/resilience.md)). Circuit breakers / bulkheads stay **out of scope** for the core crate—wrap calls in your app.

## Per-call timeouts

[`ExecuteOptions`](https://docs.rs/flowgrid/latest/flowgrid/struct.ExecuteOptions.html) overrides the HTTP timeout for a single request (including streaming entrypoints) without building a new client. Chat and Messages expose `create_with_options`, `create_stream_with_options`, and `create_with_response_and_options`. Lower layers also provide `get_json_with_options` / `post_json_with_options` on the transports. For cancellation beyond timeouts, use **`tokio::select!`**, **`tokio::time::timeout`** around **`StreamExt::next`**, or the optional **`cancel`** feature with **[`stream_next_until_cancelled`](https://docs.rs/flowgrid/latest/flowgrid/fn.stream_next_until_cancelled.html)** (see Cookbook).

## Contract fixtures

Versioned JSON under [`crates/flowgrid/tests/fixtures/contracts`](crates/flowgrid/tests/fixtures/contracts) guards deserialization of public response types (`ChatCompletion`, `Message`, …).

**Naming:** `openai_<resource>_v<api_hint>_<scenario>.json` and `anthropic_<resource>_v<api_hint>_<scenario>.json` (example: `openai_chat_completion_v1_deserialize.json`). **Add or update** a fixture when you introduce a new publicly relied-on field or when a provider’s JSON shape changes. Tests are offline (`serde_json` only) and the **`contracts`** CI job runs `cargo test -p flowgrid --features full contract_` for a fast loop.

To import a capture from a proxy or logs, use [`tools/import_contract.ps1`](tools/import_contract.ps1) / [`tools/import_contract.sh`](tools/import_contract.sh) (stdin/file in → redacted JSON out); **always review** the output for secrets before committing.

## Benchmarks

```bash
cargo bench -p flowgrid --features full
```

The **`hot_path`** target exercises contract JSON deserialization and a small SSE parse through the public **`SseStream`**. Results are relative to your machine; use them for regressions, not provider latency claims.

**Draining SSE in memory:** [`try_collect_unpin`](https://docs.rs/flowgrid/latest/flowgrid/fn.try_collect_unpin.html) collects an [`Unpin`](https://docs.rs/futures/latest/futures/stream/trait.Stream.html) fallible event stream until completion (or first error). **Memory grows with the stream**—use only for bounded or trusted streams. Typed **`data:`** lines: **`stream-types`** adds chat / Responses / Anthropic parsers (see rustdoc `parse_*_stream_json`).

### Ignored live smokes (manual)

| Test / area | Env / command | Expectation |
|-------------|----------------|-------------|
| OpenAI embeddings / Azure / etc. | `OPENAI_API_KEY`, `cargo test -p flowgrid --features full -- --ignored` | Smoke only; may hit real quota. |
| Anthropic messages | `ANTHROPIC_API_KEY`, same | Same. |

Expand this table as you add `--ignored` integration tests.

## Security & privacy

- **API keys** are supplied by your app (headers / builders). The SDK does **not** phone home or send SDK-level telemetry.
- **Errors:** `Debug` / `Display` on error types may include **`body_snippet`**, `request_id`, and header-derived timing fields—treat logs as sensitive when customer data is present. Do **not** log full response bodies in hooks; see [`docs/observability.md`](docs/observability.md).
- **Trust boundaries:** your binary → `flowgrid` (configured TLS to provider) → vendor API. The crate does **not** persist prompts or completions beyond what you hold in memory in response types.
- **Secrets in memory:** API keys are held as **`String`** in config; the crate does **not** use **`zeroize`** today. Treat process memory like any other client library; prefer env vars or secret stores in production.

CI runs **`cargo deny check`** and **`cargo audit`** on Linux; configure your org’s advisory policies in [`deny.toml`](deny.toml).

## Platform support

**`wasm32-*` / edge runtimes** are **not supported** targets today: the stack assumes a full Tokio + native TLS (`reqwest`) environment. **Serde** is the supported serialization layer for provider JSON; alternate serializers are **explicit non-goals** until a concrete requirement appears (would imply a parallel type system).

## Commercial support & compatibility

**Commercial support:** placeholder—add your contact, review/audit offerings, or support SLAs here before publishing.

**Compatibility:** responses in this repo are covered by **offline contract fixtures** and **`cargo test`** with feature `full`. Live provider endpoints are exercised only where **`--ignored`** tests and manual smoke runs are documented; there is **no** guarantee of blanket parity with every beta flag or preview endpoint. Expand this matrix honestly as you add checks:

| Area | Coverage |
|------|-----------|
| OpenAI chat completions (non-stream) | Contract fixture + typed API |
| OpenAI chat completions (SSE) | Example + streaming tests where present |
| OpenAI embeddings / legacy completions / Responses (non-stream) | Contract fixtures + typed `usage` fields |
| Anthropic Messages (non-stream) | Contract fixture + typed API |
| Anthropic `beta/models` (typed list) | Contract fixture + `list_typed` / `retrieve_typed` when `beta` enabled |
| OpenAI Assistants / threads / runs (feature `assistants`) | Contract fixtures + typed resources; `OpenAI::assistants()` / `OpenAI::threads()` |
| Other submodules | Feature-gated compile + unit/integration coverage (varies) |

## Governance & contributing

- **[`CHANGELOG.md`](CHANGELOG.md)** — release notes (Keep a Changelog).
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — semver baseline rules for API edits (`crates/flowgrid/semver/baseline_rustdoc.json`).
- **[`docs/migration.md`](docs/migration.md)** — onboarding from official SDKs or raw `reqwest`.
- **[`docs/resilience.md`](docs/resilience.md)** — retries, custom status predicate, rate limits vs circuit breakers.
- **[`docs/http.md`](docs/http.md)** — TLS, proxy env, timeouts, OpenAI-compatible bases.
- **[`docs/fuzzing.md`](docs/fuzzing.md)** — optional `cargo-fuzz` notes for SSE parsing.

**Releases / semver:** API compatibility for the crate root `pub use` surface is checked in CI with [**cargo-semver-checks**](https://github.com/obi1kenobi/cargo-semver-checks) against [`crates/flowgrid/semver/baseline_rustdoc.json`](crates/flowgrid/semver/baseline_rustdoc.json). CI regenerates rustdoc on **nightly** (`cargo rustdoc -p flowgrid --features full -Z unstable-options -- …`) and passes `--current-rustdoc target/doc/flowgrid.json` so conflicting TLS features are never enabled together. When you ship an intentional API change, refresh that baseline in the **same** PR as the version bump (see CONTRIBUTING).

## Migrating

See **[`docs/migration.md`](docs/migration.md)** for configuration, errors, streaming, and TLS notes.

## Developer workflow (`just`)

Optional [**`just`**](https://github.com/casey/just) recipes mirror CI: `just fmt`, `just clippy`, `just test-full`, `just check-msrv`, `just semver-local`, `just test-contracts`, `just deny`, `just audit`.

## License

Licensed under MIT OR Apache-2.0 at your option (see crate `Cargo.toml`).
