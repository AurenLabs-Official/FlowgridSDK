//! Multi-GPU / data-parallel notes (Burn **libtorch** backend).
//!
//! Burn does not ship FSDP/ZeRO-3 yet; practical approaches:
//! - **Single process / multi-GPU:** enable feature `tch` on `flowgrid-tensor` and shard batches manually.
//! - **Multi-process:** spawn one `flowgrid-llm train` worker per GPU with a shared gradient server (out of scope here).
//!
//! This module intentionally stays documentation-only until collective ops stabilize upstream.
