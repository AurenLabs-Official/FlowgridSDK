# FlowgridSDK

Rust workspace containing async clients for major LLM HTTP APIs: **`flowgrid-openai`** for [OpenAI](https://platform.openai.com/docs/api-reference) and **`flowgrid-anthropic`** for [Anthropic](https://docs.anthropic.com/en/api/getting-started). The optional meta-crate **`flowgrid`** re-exports both behind feature flags so you can depend on a single package when you need both providers.

Spec and method surface for Anthropic follow the Python SDK’s generated [`api.md`](https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/api.md) (Messages, Batches, Models, beta namespaces), without binding the Python runtime.

## MSRV

Rust **1.75** (see `rust-version` in each crate’s `Cargo.toml`).

## Crate: `flowgrid-openai`

Located in [`crates/flowgrid-openai`](crates/flowgrid-openai).

### Environment

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Default API key (optional if set on the builder) |
| `OPENAI_ORG_ID` | `OpenAI-Organization` header |
| `OPENAI_PROJECT` | `OpenAI-Project` header |
| `OPENAI_BASE_URL` | Override API base (default `https://api.openai.com/v1`; a trailing slash is added automatically so path joining matches OpenAI paths) |

### Build features

Core APIs (Responses, Chat Completions, legacy Completions, Embeddings) are always enabled.

Optional resource areas compile when enabled; use `--features full` for everything:

- `files`, `images`, `audio`, `moderations`, `batches`, `fine_tuning`
- `evals`, `assistants`, `vector_stores`, `containers`, `admin`
- `webhooks`, `azure`, `realtime`

### Run tests

```bash
cd crates/flowgrid-openai
cargo test
cargo test --all-features
```

Live tests (optional): set `OPENAI_API_KEY` and run with `OPENAI_LIVE_TEST=1` (see integration tests).

## Crate: `flowgrid-anthropic`

Located in [`crates/flowgrid-anthropic`](crates/flowgrid-anthropic). Uses `x-api-key`, `anthropic-version` (default `2023-06-01`), and optional `anthropic-beta`; base URL defaults to `https://api.anthropic.com/v1` with the same trailing-slash normalization as OpenAI for reliable path joins.

### Environment

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Required for `ClientConfig::from_env` in `flowgrid-anthropic` |
| `ANTHROPIC_API_BASE` | Override API base (default `https://api.anthropic.com/v1`) |
| `ANTHROPIC_VERSION` | `anthropic-version` header (default `2023-06-01`) |
| `ANTHROPIC_BETA` | Optional comma-separated `anthropic-beta` feature flags |

### Build features

- `batches`, `models`, `beta` — optional API areas (thin JSON where needed); `--features full` enables all three.

### Run tests

```bash
cd crates/flowgrid-anthropic
cargo test --all-features
```

Live smoke: set `ANTHROPIC_API_KEY` (optional `ANTHROPIC_MODEL`) and run `cargo test live_messages_smoke -- --ignored --nocapture`.

## Meta-crate: `flowgrid`

[`crates/flowgrid`](crates/flowgrid) exposes `flowgrid::openai` and `flowgrid::anthropic` as re-exports. Default features include both; disable one with `default-features = false` and enable only what you need:

```toml
flowgrid = { path = "crates/flowgrid", default-features = false, features = ["openai", "anthropic"] }
```

### License

Licensed under MIT OR Apache-2.0 at your option (per-crate `LICENSE` / `Cargo.toml`).
