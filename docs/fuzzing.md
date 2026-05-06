# Fuzzing (optional)

This repository does not yet ship a **`cargo fuzz`** workspace. A high-value target is the **SSE line parser** in [`crates/flowgrid/src/internal/sse.rs`](../crates/flowgrid/src/internal/sse.rs): random chunk boundaries over a fixed byte sequence should not panic and should match golden event boundaries.

To add fuzzing locally:

1. Install [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz).
2. From `crates/flowgrid`, run `cargo fuzz init` and point the fuzz target at `SseStream::read_next_event` (or a small `pub(crate)` test helper), feeding arbitrary `Bytes` chunks.

Fuzz failures should be fixed with **parser hardening** only; treat them as **non-breaking** unless the public contract of event boundaries changes.
