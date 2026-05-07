# Fuzzing (optional)

The crate ships an internal SSE fuzz target under **`crates/flowgrid/fuzz/`** (feature **`sse-fuzz`**), exercising chunk boundaries against the same incremental decoder as production (`read_next_event`).

To fuzz locally:

1. Install [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz).
2. From **`crates/flowgrid`**, run `cargo fuzz run sse_chunks` (uses **`flowgrid`** with **`openai`**, **`tls-rustls`**, **`sse-fuzz`**).

Fuzz failures should be fixed with **parser hardening** only; treat them as **non-breaking** unless the public contract of event boundaries changes.
