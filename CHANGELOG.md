# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Observability: `tracing` span `flowgrid.http.request` with retry, request id, and rate-limit fields; OpenTelemetry metric attribute `flowgrid.retry_count`; runbook `docs/observability.md`.
- Optional **`cancel`** Cargo feature and `stream_next_until_cancelled` helper for cooperative SSE/stream shutdown.
- Contract fixture naming convention, `tools/import_contract` scripts, and Criterion `hot_path` benchmarks.
- Developer `justfile`, `docs/migration.md`, supply-chain CI (`cargo deny` / `cargo audit`), and governance docs (`CONTRIBUTING.md`, this file).
- Typed **`EmbeddingUsage`**, **`CompletionUsage`**, **`ResponseObjectUsage`** on OpenAI embedding/completion/response types; **`BetaModel`**, **`BetaModelsListResponse`**, **`list_typed`** / **`retrieve_typed`** on beta models client.
- Additional contract fixtures and tests for embeddings, completions, responses, beta models.
- **`try_collect_unpin`** helper for draining fallible **`Unpin`** SSE streams (memory grows with length).
- OpenAI Responses streaming: **`parse_openai_response_stream_json`** / **`OpenAiResponseStreamLine`** (feature **`stream-types`**); Anthropic streaming test for **`content_block_start`** lines.
- Optional **`retry_if_response_status`** on OpenAI/Anthropic **`ClientConfig`** and builders (replaces default retry-status rule when set).
- Docs: [`docs/resilience.md`](docs/resilience.md), [`docs/http.md`](docs/http.md), [`docs/fuzzing.md`](docs/fuzzing.md); README proxy/timeout/smoke-matrix/zeroize note; Azure doc link for OpenAI-compatible bases.
- OpenAI **Assistants** workflow (feature **`assistants`**): typed **`Assistant`**, **`AssistantsListParams`**, **`list_typed`** / **`create_typed`** / **`retrieve_typed`** / **`update_typed`**; **`ThreadsClient`**, **`ThreadClient`**, **`ThreadMessagesClient`**, **`ThreadRunsClient`** with typed **`Thread`**, **`ThreadMessage`**, **`ThreadRun`**; **`HttpTransport::get_json_query`** for list pagination; contract fixtures for assistants, threads, messages, runs, and assistant list pages; top-level **`ThreadsClient::update`**, **`update_typed`**, **`delete`** mirroring **`retrieve`**.

### Changed

- **`CreateEmbeddingResponse.usage`**, **`Completion.usage`**, and **`ResponseObject.usage`** are now structured types (with **`extra`** maps) instead of raw **`serde_json::Value`**.

### Fixed

- **Criterion `hot_path`:** Anthropic SSE benchmark is registered whenever feature **`anthropic`** is enabled (including alongside **`openai`** / `full`), not only when OpenAI is off.
- **`Retry-After` HTTP-date** values in the past (or equal to “now”) are ignored so retries use exponential backoff instead of a **zero** delay.
- **README:** duplicate compatibility paragraph removed.
- **`azure` module rustdoc:** link to [`docs/http.md`](docs/http.md) now points at the workspace-root file.

### Documentation

- README: security and privacy, platform limits (WASM/edge), serde stance, cancellation patterns, benchmarks, commercial/compatibility placeholders.

## [0.1.0] - YYYY-MM-DD

Initial crates.io-aligned snapshot (replace date on first publish).

[Unreleased]: https://github.com/example/flowgrid-sdk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/example/flowgrid-sdk/releases/tag/v0.1.0
