# Flowgrid LLM stack (Burn, preview)

The **`flowgrid` HTTP SDK** crate is **stable** by policy; everything under `flowgrid-model`,
`flowgrid-train`, `flowgrid-checkpoint`, `flowgrid-serve`, `flowgrid-cli`, `flowgrid-edit`, and
`flowgrid-ui` is **preview** and may break across releases.

## What works today

| Area | Status |
|------|--------|
| **CPU / NdArray** decoder (`NanoGpt`), RoPE + **pre-sized KV cache** + cache parity tests; **GQA** via `n_kv_head` (must divide `n_head`) | **Works** (preview) |
| **Checkpoint** `save_nano_gpt_checkpoint` / `load_nano_gpt_checkpoint` (Burn bincode record + `manifest.json` with `manifest_version` + BLAKE3 `fingerprint` including streamed hash of `model.bin`; **`config_basis` includes resolved `n_kv_head`** → old `b3:` fingerprints change on upgrade; optional LoRA sidecar pointer `lora` + `lora_schema_version`) | **Works** (preview) |
| **Sampler** (greedy / temperature / top-k) + CLI `generate` | **Works** (preview) |
| **GPT-2 safetensors → `NanoGpt`** (`load_gpt2_into_nano_gpt`, Conv1D `[nx, nf]` layout, `lm_head` orientation) | **Works** (preview); LayerNorm from HF still init defaults |
| **Eval** `perplexity` + `EvalReport` JSON + `flowgrid-llm eval --baseline-ppl --max-regression-pct` + `--stride` | **Works** (preview) |
| **`flowgrid-serve`** real decode when `FLOWGRID_SERVE_CHECKPOINT` set; SSE one event per streamed token; tokenizer from manifest; **Bearer** or **`api-key` / `x-api-key`** auth; **token-bucket** HTTP rate limit (`FLOWGRID_SERVE_RPS` / `FLOWGRID_SERVE_BURST`) | **Works** (preview) |
| **Studio UI** job start with **kind + command allowlist** | **Partial** |
| **LoRA** `LoraLinear` forward + `merged_linear`; `attach_lora` for `NanoGpt` (wraps targeted projections; `gate` reserved for non–nano-GPT arch) | **Works** (preview) |
| **Llama / Mistral / Qwen** | **Preview**: key validation + **staged** `decode_self_attn_q_proj` and **`decode_self_attn_kv_proj`** (GQA **`[n_kv·d, n_embd]`** layout); full map into `NanoGpt` still WIP |
| **GPU backends** | **Optional** Cargo feature **`gpu-wgpu`** on `flowgrid-cli` / `flowgrid-serve` (Burn `Wgpu`); env **`FLOWGRID_DEVICE`** parsed by [`flowgrid-device`](crates/flowgrid-device) (`cpu`, `wgpu`, `wgpu:0`, …). Default builds stay **NdArray CPU** (MSRV-friendly). |
| **ROME/MEMIT** (`flowgrid-edit`) | **Experimental** — gated until checkpoint + LoRA path is stable |

## Workspace crates

| Crate | Role |
|-------|------|
| `flowgrid` | **Stable** HTTP SDK (unchanged for LLM work) |
| `flowgrid-device` | `FLOWGRID_DEVICE` parsing (no Burn dependency) |
| `flowgrid-tokenizer` | HF `tokenizer.json`; streaming decode with **incremental** `decode_streaming` |
| `flowgrid-data` | mmap token blobs (`u32` LE) |
| `flowgrid-model` | `NanoGpt` (GQA), prealloc KV cache, RoPE, sampler, HF loaders |
| `flowgrid-train` | CE loss, `TrainerConfig`, training loop helpers |
| `flowgrid-eval` | Perplexity + regression gate |
| `flowgrid-ml` | Classical ML baselines (regression/classification metrics + simple linear model) |
| `flowgrid-checkpoint` | Manifest + Burn record I/O |
| `flowgrid-cli` | `flowgrid-llm` (`prepare`, `train`, `generate`, `eval`, `merge-lora`) |
| `flowgrid-serve` | OpenAI-shaped `/v1/chat/completions` + `/v1/responses` |
| `flowgrid-edit` | ROME/MEMIT hooks (experimental) |
| `flowgrid-ui` | SQLite-backed dashboard on `9010` |

## Quickstart (CPU)

```bash
cargo build -p flowgrid-cli
cargo run -p flowgrid-cli -- prepare -i README.md -o target/readme.bin --byte-level
cargo run -p flowgrid-cli --profile local -- train --tokens target/readme.bin --steps 32 --n-head 4 --n-kv-head 0 --run-report-out target/mlops/train.json
cargo run -p flowgrid-cli -- generate --prompt "Hi" --max-new 16   # echoes prompt prefix; `--no-echo` for completion-only
```

### `FLOWGRID_DEVICE` (CLI + serve)

[`flowgrid-device`](crates/flowgrid-device) reads **`FLOWGRID_DEVICE`** (default `cpu`). GPU inference/training requires building with **`--features gpu-wgpu`**:

```bash
cargo build -p flowgrid-cli --features gpu-wgpu
cargo run -p flowgrid-cli --features gpu-wgpu -- train --tokens data.bin --steps 8
# e.g. FLOWGRID_DEVICE=wgpu  or  wgpu:0  or  integrated:0
```

Same for **`flowgrid-serve`**: `cargo build -p flowgrid-serve --features gpu-wgpu`. Without the feature, a GPU request in the env logs a **warning** and execution stays on CPU.

### Eval quality gate

```bash
cargo run -p flowgrid-cli -- eval \
  --dataset data/eval.bin \
  --load path/to/ckpt \
  --split test --train-frac 0.8 --val-frac 0.1 \
  --block 64 --stride 64 \
  --baseline-ppl 12.0 --max-regression-pct 5 \
  --run-report-out target/mlops/eval.json
```

### Local OpenAI-shaped server

```bash
set FLOWGRID_SERVE_CHECKPOINT=C:\path\to\checkpoint-dir
# optional: FLOWGRID_SERVE_TOKENIZER=... when not using checkpoint manifest path
cargo run -p flowgrid-serve
# POST http://127.0.0.1:9000/v1/chat/completions
```

Optional: `FLOWGRID_SERVE_TEMPERATURE`, `FLOWGRID_SERVE_TOP_K`, `FLOWGRID_SERVE_SEED`, `FLOWGRID_SERVE_REQUEST_TIMEOUT_MS` (bounds each generation; default 10_000), **`FLOWGRID_SERVE_RPS`** (requests/sec token bucket refill; default `32`), **`FLOWGRID_SERVE_BURST`** (max bucket tokens; defaults to **`FLOWGRID_SERVE_RPS`**), `FLOWGRID_SERVE_MAX_BODY_BYTES`.

Additional throughput/backpressure knobs:

- `FLOWGRID_SERVE_WORKERS` (scheduler workers; local checkpoint mode currently clamps to one worker for safety)
- `FLOWGRID_SERVE_STREAM_BUFFER` (per-request stream channel capacity)
- `FLOWGRID_SERVE_MAX_NEW_TOKENS` (hard cap at scheduler ingress)

Deployment profile presets are selected via **`FLOWGRID_DEPLOYMENT_PROFILE=local|cloud|hybrid`**. Profiles tune default worker/queue/rate knobs; explicit `FLOWGRID_SERVE_*` env vars still take precedence.

### Baseline loadtest + KPI capture

Use the profile matrix in `docs/loadtest-matrix.md` and generate machine-readable reports:

```bash
just kpi-serve-local
just kpi-serve-hybrid
just kpi-serve-cloud
```

All runs emit `serve_kpi_smoke_v1` JSON under `target/mlops/`.

### Usage / `finish_reason` (exact vs approximate)

| Mode | `prompt_tokens` / `completion_tokens` | `finish_reason` |
|------|----------------------------------------|-----------------|
| Local **`NanoGpt`** with checkpoint tokenizer | Encoder length of the prompt; one count per **generated** token id (**EOS is not counted** in `completion_tokens`); streamed SSE uses the same totals on the final chunk | `stop` when `<eos>` is sampled (or token id from **`FLOWGRID_SERVE_EOS_ID`**) or on scheduler failure paths; SSE inference errors emit **`event: error`** and **omit** `[DONE]`; **`length`** when `max_tokens` is reached first |
| Echo + optional `FLOWGRID_SERVE_TOKENIZER` (no checkpoint) | Prompt side from tokenizer encode when available; completion from tokenizer ids for the echoed tail | Usually `stop` |
| Pure echo (no tokenizer env) | **`~ceil(bytes/4)`** heuristic via `approx_tokens_from_text` for both sides | `stop` |

Non-streaming and streaming responses share these rules so clients do not need separate heuristics when a tokenizer-backed path is active.

### Checkpoint manifests (legacy vs current)

`flowgrid-checkpoint` writes **`manifest_version`** and a **`b3:`**-prefixed **`fingerprint`** that includes the on-disk `model.bin` digest. Loading an older `manifest.json` **without** `manifest_version` or with a **non-`b3:`** fingerprint logs a **`tracing::warn`** and still loads; you should **re-save** the directory with current tooling so fingerprints and schema version are unambiguous. Breaking manifest layout changes will bump `manifest_version` and be called out in release notes.

## CI

`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace`, plus Linux `cargo test -p flowgrid --features full` and a **Windows** job with `CARGO_TARGET_DIR=target/win-full` to catch linker locking issues (see root `README.md`).

ML reproducibility smoke is enforced in CI by running two same-seed train runs and checking report deltas (`tools/check_train_repro.py`).
Release-gate validation is codified in `tools/validate_release_gates.py` (train/eval/classical artifacts, optional KPI artifacts).

Workload template catalog and resilience governance:

- `docs/workload-templates.md`
- `docs/runtime-resilience-program.md`

Baseline authority, CI parity, and comparison reporting:

- `docs/kpi-baseline-authority.md`
- `docs/release-gates-parity.md`
- `docs/model-comparison-reporting.md`

## Dependency pin

`burn-core 0.13` expects **`bincode 2.0.0-rc.3`**; the workspace pins `bincode` via `flowgrid-tensor`.

## Observability

Training and servers use `tracing` / `RUST_LOG`.
