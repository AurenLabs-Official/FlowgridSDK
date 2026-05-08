# Contributing

## API changes and semver

The stable surface of this crate is the **`pub use` re-exports at the crate root** (see `src/lib.rs`). Anything under `internal::` is **not** covered by semver for consumers who reach into it (path imports are discouraged).

When you change the public root API:

1. Run **`cargo +nightly rustdoc`** and [**`cargo-semver-checks`**](https://github.com/obi1kenobi/cargo-semver-checks) as in CI (or locally: `just semver-local` if you use the repo `justfile`).
2. Ensure the **`semver-local`** / CI semver job is **green** for your PR.
3. If the change is intentional for a release, update the committed baseline **`crates/flowgrid/semver/baseline_rustdoc.json`** in the **same PR** as the API change (and version bump), per README “Releases and semver baseline”.

**PR checklist (copy into description or use `.github/pull_request_template.md`):**

- [ ] Touched crate root `pub use` / semver surface? Then: version bump in [`crates/flowgrid/Cargo.toml`](../crates/flowgrid/Cargo.toml), `just semver-local` green, baseline JSON updated in same PR.
- [ ] User-visible behavior? Entry in **[`CHANGELOG.md`](../CHANGELOG.md)** under `[Unreleased]` or the release section you are cutting.

## Changelog

User-visible behavior or API updates should have an entry in **`CHANGELOG.md`** under `[Unreleased]` (or the release section you are preparing).

## Supply chain

CI runs **`cargo deny check`** (see root [`deny.toml`](deny.toml)). **[`docs/advisory-supply-chain.md`](docs/advisory-supply-chain.md)** tracks ignored transitive advisories; review quarterly. **`cargo deny`** may warn about duplicate **`windows-sys`** versions pulled by **`criterion`** (dev) vs **`tokio`**/`mio`; this is expected on Windows-heavy graphs and is tracked as **warn**-level in `deny.toml` bans, not a hard failure.

## Toward `flowgrid` 1.0 (optional, product decision)

**1.0** only makes sense if you want a **frozen** public API with explicit migration for additive fields. Until then, stay on **0.x** semver. To reduce breakage before 1.0:

- Prefer **builders** and **`Default`** over large struct literals for configs (`ClientBuilder`, `http_client_builder_hook`, etc.).
- Consider **`#[non_exhaustive]`** on public structs/enums that are extended often (requires a dedicated PR; coordinate with semver baseline + changelog).

Path imports under `internal::` remain **explicitly unstable** (see top of this file).
