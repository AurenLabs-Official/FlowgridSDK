## Summary

<!-- What does this PR change? -->

## Checklist

- [ ] **Semver:** If the PR touches the **`flowgrid` crate root `pub use` surface** — version bump in `crates/flowgrid/Cargo.toml`, **`just semver-local`** green, and **`crates/flowgrid/semver/baseline_rustdoc.json`** updated in **this** PR ([CONTRIBUTING.md](../CONTRIBUTING.md)).
- [ ] **Changelog:** User-visible change recorded in **[CHANGELOG.md](../CHANGELOG.md)** (`[Unreleased]` or target release section).
- [ ] **`cargo deny`:** If changing `deny.toml` ignores — update **[docs/advisory-supply-chain.md](../docs/advisory-supply-chain.md)**.
- [ ] **Preview LLM crates:** If breaking — migration hint in changelog / **[docs/llm/overview.md](../docs/llm/overview.md)** when appropriate.
