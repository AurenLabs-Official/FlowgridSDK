# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-07

Preview-only **breaking** adjustments for the local LLM workspace crates (the stable `flowgrid` HTTP SDK is unchanged except as noted in docs).

### Added

- `flowgrid-serve`: `CompletionMeta` / streaming `StreamPart` with tokenizer-backed **`usage`** and **`finish_reason`** when a checkpoint is loaded; **`FLOWGRID_SERVE_EOS_ID`** override; SSE **`data:`** error objects before stream end on inference failures.
- `flowgrid-checkpoint`: `load_manifest` warns on legacy manifests (missing `manifest_version` or non-`b3:` fingerprints); optional **`lora_schema_version`** and manifest **`lora`** sidecar pointer when saving with LoRA (`save_nano_gpt_checkpoint` gains `lora_sidecar: Option<&str>`).
- `flowgrid-device`: **`FLOWGRID_DEVICE`** parsing for preview binaries (CPU vs wgpu intent).
- `flowgrid-cli` / `flowgrid-serve`: optional Cargo feature **`gpu-wgpu`** (Burn **`Wgpu`** backend); **`FLOWGRID_DEVICE`** selects adapter when enabled.
- `flowgrid-model`: **`hf::llama::decode_self_attn_q_proj`** (F32/BF16/F16 bytes → rank-2 tensor) + Mistral re-export; unit test `llama_decode_weights`.

### Changed

- **`save_nano_gpt_checkpoint(..., tokenizer_path, lora_sidecar)`** — fourth parameter is now the LoRA sidecar relative path (e.g. `"lora.json"`), or `None`.

### Documentation

- README “OpenAI-shaped compat” table for `flowgrid-serve`; `docs/llm/overview.md` usage and manifest migration notes.

## [Unreleased]

### Added

- **LLM stack (preview):** Burn-backed crates `flowgrid-tensor`, `flowgrid-tokenizer`, `flowgrid-data`, `flowgrid-model`, `flowgrid-train`, `flowgrid-cli` (`flowgrid-llm`), `flowgrid-serve`, `flowgrid-edit`, `flowgrid-ui`; overview [`docs/llm/overview.md`](docs/llm/overview.md). Workspace pins **`bincode =2.0.0-rc.3`** for `burn` 0.13 compatibility.
- Examples: **`openai_assistants_e2e`** (Assistants thread → message → run → bounded poll) and **`openai_responses_stream_accumulate`** (Responses SSE + bounded text accumulation); README **`full` vs `enterprise`** subsection and **`docs/observability.md`** dashboard hints.
- OpenAI **cursor list helpers**: **`ListPagesLimits`**, **`ListPage::after_cursor`**, **`AssistantsClient::list_all_typed`**, **`ThreadsClient::list_all_typed`**; **`ListPage`** forward-compatible **`extra`** map; Assistants **run steps** (**`ThreadRunStep`**, **`list_steps_typed`**) with fixtures + wiremock.
- **HTTP**: **`post_empty`**, **`http_client_builder_hook`** on OpenAI / Anthropic / Azure configs and builders; `docs/http.md` updated (proxies, mTLS story).
- Streaming (**`stream-types`**): **`OpenAiStreamTextLimits`**, **`StreamTextAccumulateError`**, bounded **`accumulate_openai_*_visible_text_into`** helpers.
- Cargo features **`rate-aware-retry`**, **`compat-openai`**, **`sse-fuzz`** (+ `crates/flowgrid/fuzz` SSE target); resilience and README docs updated.
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

[Unreleased]: https://github.com/pwitt/FlowgridSDK/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pwitt/FlowgridSDK/releases/tag/v0.1.0
