# FlowgridSDK

Rust workspace containing **`flowgrid-openai`**, a hand-crafted async client for the [OpenAI HTTP API](https://platform.openai.com/docs/api-reference), structured similarly to [openai-node](https://github.com/openai/openai-node).

## MSRV

Rust **1.75** (see `rust-version` in `crates/flowgrid-openai/Cargo.toml`).

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

### License

Licensed under MIT OR Apache-2.0 at your option (`flowgrid-openai` crate).
