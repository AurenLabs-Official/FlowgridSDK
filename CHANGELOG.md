# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Observability: `tracing` span `flowgrid.http.request` with retry, request id, and rate-limit fields; OpenTelemetry metric attribute `flowgrid.retry_count`; runbook `docs/observability.md`.
- Optional **`cancel`** Cargo feature and `stream_next_until_cancelled` helper for cooperative SSE/stream shutdown.
- Contract fixture naming convention, `tools/import_contract` scripts, and Criterion `hot_path` benchmarks.
- Developer `justfile`, `docs/migration.md`, supply-chain CI (`cargo deny` / `cargo audit`), and governance docs (`CONTRIBUTING.md`, this file).

### Documentation

- README: security and privacy, platform limits (WASM/edge), serde stance, cancellation patterns, benchmarks, commercial/compatibility placeholders.

## [0.1.0] - YYYY-MM-DD

Initial crates.io-aligned snapshot (replace date on first publish).

[Unreleased]: https://github.com/example/flowgrid-sdk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/example/flowgrid-sdk/releases/tag/v0.1.0
