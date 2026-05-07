# Flowgrid LLM stack (Burn, preview)

The **`flowgrid` HTTP SDK** crate is **stable** by policy; everything under `flowgrid-model`,
`flowgrid-train`, `flowgrid-checkpoint`, `flowgrid-serve`, `flowgrid-cli`, `flowgrid-edit`, and
`flowgrid-ui` is **preview** and may break across releases.

## What works today

| Area | Status |
|------|--------|
| **CPU / NdArray** decoder (`NanoGpt`), RoPE + KV-cache + cache parity tests | **Works** (preview) |
| **Checkpoint** `save_nano_gpt_checkpoint` / `load_nano_gpt_checkpoint` (Burn bincode record + `manifest.json`) | **Works** (preview) |
| **Sampler** (greedy / temperature / top-k) + CLI `generate` | **Works** (preview) |
| **GPT-2 safetensors → `NanoGpt`** (`load_gpt2_into_nano_gpt`, Conv1D `[nx, nf]` layout, `lm_head` orientation) | **Works** (preview); LayerNorm from HF still init defaults |
| **Eval** `perplexity` + `EvalReport` JSON + `flowgrid-llm eval --baseline-ppl --max-regression-pct` + `--stride` | **Works** (preview) |
| **`flowgrid-serve`** real decode when `FLOWGRID_SERVE_CHECKPOINT` set; SSE one event per streamed token; tokenizer from manifest | **Works** (preview) |
| **Studio UI** job start with **kind + command allowlist** | **Partial** |
| **LoRA** `LoraLinear` forward + `merged_linear` / tests; `attach_lora` model-wide | **Roadmap** (merge helpers work on adapter modules only) |
| **Llama / Mistral / Qwen** | **Preview**: `expected_keys` + `validate_*_keys` only; no full tensor map into `NanoGpt` yet |
| **GPU backends** | **Feature-gated** on `flowgrid-tensor` (`wgpu`, `cuda`, `tch`, `metal`, `candle`) — manual benches first |
| **ROME/MEMIT** (`flowgrid-edit`) | **Experimental** — gated until checkpoint + LoRA path is stable |

## Workspace crates

| Crate | Role |
|-------|------|
| `flowgrid` | **Stable** HTTP SDK (unchanged for LLM work) |
| `flowgrid-tensor` | Burn prelude + optional GPU features |
| `flowgrid-tokenizer` | Hugging Face `tokenizer.json` |
| `flowgrid-data` | mmap token blobs (`u32` LE) |
| `flowgrid-model` | `NanoGpt`, cache, RoPE, sampler, HF loaders (GPT-2 + key validation stubs) |
| `flowgrid-train` | CE loss, `TrainerConfig`, training loop helpers |
| `flowgrid-eval` | Perplexity + regression gate |
| `flowgrid-checkpoint` | Manifest + Burn record I/O |
| `flowgrid-cli` | `flowgrid-llm` (`prepare`, `train`, `generate`, `eval`, `merge-lora`) |
| `flowgrid-serve` | OpenAI-shaped `/v1/chat/completions` + `/v1/responses` |
| `flowgrid-edit` | ROME/MEMIT hooks (experimental) |
| `flowgrid-ui` | SQLite-backed dashboard on `9010` |

## Quickstart (CPU)

```bash
cargo build -p flowgrid-cli
cargo run -p flowgrid-cli -- prepare -i README.md -o target/readme.bin
cargo run -p flowgrid-cli -- train --tokens target/readme.bin --steps 32
cargo run -p flowgrid-cli -- generate --prompt "Hi" --max-new 16
```

### Eval quality gate

```bash
cargo run -p flowgrid-cli -- eval \
  --dataset data/eval.bin \
  --load path/to/ckpt \
  --block 64 --stride 64 \
  --baseline-ppl 12.0 --max-regression-pct 5
```

### Local OpenAI-shaped server

```bash
set FLOWGRID_SERVE_CHECKPOINT=C:\path\to\checkpoint-dir
# optional: FLOWGRID_SERVE_TOKENIZER=... when not using checkpoint manifest path
cargo run -p flowgrid-serve
# POST http://127.0.0.1:9000/v1/chat/completions
```

Optional: `FLOWGRID_SERVE_TEMPERATURE`, `FLOWGRID_SERVE_TOP_K`, `FLOWGRID_SERVE_SEED`, `FLOWGRID_SERVE_REQUEST_TIMEOUT_MS` (bounds each generation; default 10_000).

## CI

`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace` (see `.github/workflows/ci.yml`).

## Dependency pin

`burn-core 0.13` expects **`bincode 2.0.0-rc.3`**; the workspace pins `bincode` via `flowgrid-tensor`.

## Observability

Training and servers use `tracing` / `RUST_LOG`.
