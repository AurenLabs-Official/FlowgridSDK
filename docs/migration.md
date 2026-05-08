# Migrating to `flowgrid`

This guide sketches a move from **official first-party SDKs** or **hand-written `reqwest`** to **`flowgrid`**. Exact steps depend on how much of each provider you use.

**Toolchain:** the workspace targets **Rust 1.85+** (see root `Cargo.toml` `rust-version`). **`flowgrid` 0.2** adds public fields on **`ClientConfig`** / **`ListPage`**; update struct literals (or use `..` / builders) when upgrading from **0.1**.

## From official OpenAI / Anthropic SDKs (other languages or Rust wrappers)

1. **Map client configuration** to [`OpenAiClientConfig`](https://docs.rs/flowgrid/latest/flowgrid/type.OpenAiClientConfig.html) / [`AnthropicClientConfig`](https://docs.rs/flowgrid/latest/flowgrid/type.AnthropicClientConfig.html) (base URL, API key, optional headers). Use [`ClientBuilder`](https://docs.rs/flowgrid/latest/flowgrid/struct.ClientBuilder.html) / [`AnthropicBuilder`](https://docs.rs/flowgrid/latest/flowgrid/struct.AnthropicBuilder.html) for defaults.
2. **Errors** are structured [`OpenAiApiError`](https://docs.rs/flowgrid/latest/flowgrid/struct.OpenAiApiError.html) / [`AnthropicApiError`](https://docs.rs/flowgrid/latest/flowgrid/struct.AnthropicApiError.html) with `status`, `body_snippet`, `retry_after`, and `request_id` when headers expose it—lean on these instead of ad-hoc string matching.
3. **Retries** are centralized in the HTTP transport (see README “Retries”); tune `max_retries` and `retry_after_max` instead of per-callsite retry loops where appropriate.
4. **Streaming** uses SSE decoders with [`SseStream::next_event`](https://docs.rs/flowgrid/latest/flowgrid/struct.SseStream.html) / [`into_unpin_event_stream`](https://docs.rs/flowgrid/latest/flowgrid/struct.SseStream.html#method.into_unpin_event_stream); treat non-JSON `data:` payloads defensively the same way you would in other SDKs.
5. **OpenAI Assistants:** enable Cargo feature **`assistants`** for **`OpenAI::assistants()`** (CRUD + typed helpers) and **`OpenAI::threads()`** (threads, per-thread messages and runs, including cancel / submit tool outputs). Offline contract tests cover core JSON shapes; new provider fields land in **`extra`** maps on the typed models.

## From raw `reqwest`

1. Prefer **typed requests** under `client.chat()`, `client.messages()`, etc., instead of manual JSON maps—add `extra` fields when you need forward-compatible provider keys.
2. **Timeouts:** combine **client-level** defaults with per-call [`ExecuteOptions::timeout`](https://docs.rs/flowgrid/latest/flowgrid/struct.ExecuteOptions.html) for long calls.
3. **TLS:** enable exactly one of **`tls-rustls`** or **`tls-native`**; mixed stacks fail at compile time.

## Local LLM workflow migration (preview stack)

When moving shell scripts to `flowgrid-llm` / `flowgrid-serve`, prefer explicit runtime profiles and report artifacts:

- Use `flowgrid-llm --profile local|cloud|hybrid ...` to keep deployment intent visible.
- For reproducible tracking, persist JSON outputs with:
  - `flowgrid-llm train --run-report-out <path>`
  - `flowgrid-llm eval --run-report-out <path>`
- For split-aware eval gates, use `--split train|val|test` with `--train-frac` / `--val-frac`.

## Observability

Enable **`tracing`** / **`opentelemetry`** (see **`enterprise`**) and follow [`docs/observability.md`](observability.md) for span names and metric labels.
