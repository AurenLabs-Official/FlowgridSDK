# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Stability

The semver-stable contract is the **`pub use` items at the `flowgrid` crate root** (see crate documentation). The `internal` module tree is implementation detail and may change in minor releases.

### Added

- Cookbook examples in README (non-stream chat, streaming with `into_event_stream`, Anthropic messages).
- MSRV CI job (Rust 1.75) and expanded GitHub Actions feature-matrix for optional feature combinations.
- `tracing` spans/events on HTTP transport when the `tracing` feature is enabled (`target: flowgrid_http`).
- Wiremock SSE integration test; golden fixture for `ResponseObject` deserialization; Wiremock test for `request_pre_send_hook`.
- Optional `stream-types` Cargo feature: helpers to parse OpenAI / Anthropic streaming SSE JSON payloads from `SseEvent::data`.
- `pub(crate)` `join_path` helper for resource URL segments.
- `ClientBuilder::new`, `AnthropicBuilder::new` (aliases for `Default`).
- Cargo features **`tls-rustls`** (default) and **`tls-native`** for `reqwest` TLS (mutually exclusive).
- **`request_pre_send_hook`** on `ClientBuilder`, `AnthropicBuilder`, and `AzureClientBuilder` for last-chance header changes before send.
- Optional **`opentelemetry`** feature: histogram `flowgrid.http.request.duration_ms` on each HTTP round-trip.
- Examples `openai_chat_stream` and `anthropic_message`; CI uses `full` instead of `--all-features` (TLS conflict).
- **`cargo-semver-checks`** CI job.

### Changed

- `Debug` for client configs and builders redacts API keys and webhook secrets.
- `homepage` / `documentation` / `keywords` / `categories` in `Cargo.toml`; repository remains a publish-time placeholder (`example/flowgrid-sdk`).
- Default features now include **`tls-rustls`**; `reqwest` no longer hard-codes a TLS stack without this feature.
- **`full`** also enables **`tracing`** (not `opentelemetry`).
