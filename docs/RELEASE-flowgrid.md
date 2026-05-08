# Publishing `flowgrid` to crates.io (decision gate)

This repo can publish the **`flowgrid`** HTTP crate independently of preview LLM crates. The **0.2.0** release includes additive public fields (see [`CHANGELOG.md`](../CHANGELOG.md)); **`repository`** and **`homepage`** in [`crates/flowgrid/Cargo.toml`](../crates/flowgrid/Cargo.toml) must point at the real upstream before publish.

## Pre-flight (no network side effects)

```bash
cargo publish -p flowgrid --dry-run
```

Use **`--dry-run`** until the team explicitly approves a real publish.

## Tag sketch

After changelog and version are aligned on `main` (or the release branch):

- Tag: `v0.2.0` (or `flowgrid-v0.2.0` if you version tags per crate — match your org convention).
- GitHub Release: paste the **`flowgrid` 0.2.0** section from [`CHANGELOG.md`](../CHANGELOG.md), highlighting **Rust 1.85 MSRV** and **struct-literal updates** for `ClientConfig` / `ListPage`.

## Optional: `cargo publish` (requires crates.io token)

Only after maintainers approve:

```bash
cargo publish -p flowgrid
```

Do **not** automate this from CI without org policy and secret management.
