# Flowgrid LLM stack (Burn, preview)

Workspace crates:

| Crate | Role |
|-------|------|
| `flowgrid-tensor` | Burn prelude + backend flags (`wgpu`, `cuda`, `tch`, …) |
| `flowgrid-tokenizer` | Hugging Face `tokenizer.json` loader |
| `flowgrid-data` | mmap token blobs (`u32` LE) |
| `flowgrid-model` | nano-GPT-style LM + LoRA scaffold + safetensors helpers |
| `flowgrid-train` | CE loss helpers + checkpoint hooks + multi-GPU notes |
| `flowgrid-cli` | `flowgrid-llm` binary (`prepare`, `train`, `generate`) |
| `flowgrid-serve` | OpenAI-shaped `/v1/chat/completions` stub for wiring tests |
| `flowgrid-edit` | ROME/MEMIT hooks + quantization roadmap |
| `flowgrid-ui` | SQLite-backed REST shell on `9010` |

## Quickstart (CPU)

```bash
cargo build -p flowgrid-cli
cargo run -p flowgrid-cli -- prepare -i README.md -o target/readme.bin
cargo run -p flowgrid-cli -- train --tokens target/readme.bin --steps 32
cargo run -p flowgrid-cli -- generate --prompt "Hi" --max-new 16
```

Enable GPU backends by adding features on `flowgrid-tensor` (see crate `Cargo.toml`).

## Observability

Training CLI uses `tracing-subscriber` (`RUST_LOG=info`). HTTP servers (`flowgrid-serve`, `flowgrid-ui`) emit axum `TraceLayer` spans when `tower-http/trace` is enabled.

## Dependency pin

`burn-core 0.13` expects **`bincode 2.0.0-rc.3`** APIs; the workspace pins `bincode` via `flowgrid-tensor` so `cargo update` cannot jump to incompatible `2.0.1`.
