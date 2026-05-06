# Contributing

## API changes and semver

The stable surface of this crate is the **`pub use` re-exports at the crate root** (see `src/lib.rs`). Anything under `internal::` is **not** covered by semver for consumers who reach into it (path imports are discouraged).

When you change the public root API:

1. Run **`cargo +nightly rustdoc`** and [**`cargo-semver-checks`**](https://github.com/obi1kenobi/cargo-semver-checks) as in CI (or locally: `just semver-local` if you use the repo `justfile`).
2. Ensure the **`semver-local`** / CI semver job is **green** for your PR.
3. If the change is intentional for a release, update the committed baseline **`crates/flowgrid/semver/baseline_rustdoc.json`** in the **same PR** as the API change (and version bump), per README “Releases and semver baseline”.

## Changelog

User-visible behavior or API updates should have an entry in **`CHANGELOG.md`** under `[Unreleased]` (or the release section you are preparing).

## Supply chain

CI runs **`cargo deny check`** (see root [`deny.toml`](deny.toml)). **`cargo deny`** may warn about duplicate **`windows-sys`** versions pulled by **`criterion`** (dev) vs **`tokio`**/`mio`; this is expected on Windows-heavy graphs and is tracked as **warn**-level in `deny.toml` bans, not a hard failure.
