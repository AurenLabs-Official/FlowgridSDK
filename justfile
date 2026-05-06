# Local tasks mirroring `.github/workflows/ci.yml` (requires `rustfmt`, `clippy`, stable + nightly Rust).

default:
    @just --list

# Format check (same as CI `fmt`).
fmt:
    cargo fmt --all -- --check

# Lint the primary feature bundle (same as CI `clippy`).
clippy:
    cargo clippy -p flowgrid --features full -- -D warnings

# Unit + integration tests (same as CI `test`).
test-full:
    cargo test -p flowgrid --features full

# Compile examples (same as CI `check examples`).
check-examples:
    cargo check -p flowgrid --examples --features full

# Minimum supported Rust version (same as CI `msrv`).
check-msrv:
    cargo +1.75 check -p flowgrid --features full

# OpenTelemetry-only smoke check (same as CI matrix helper).
check-opentelemetry:
    cargo check -p flowgrid --no-default-features --features "openai,anthropic,tls-rustls,opentelemetry"

# Regenerate rustdoc JSON on nightly and run semver-checks against the committed baseline (same as CI `semver`).
semver-local:
    cargo +nightly rustdoc -p flowgrid --features full -Z unstable-options -- -Z unstable-options --output-format json
    cargo semver-checks check-release -p flowgrid --baseline-rustdoc crates/flowgrid/semver/baseline_rustdoc.json --current-rustdoc target/doc/flowgrid.json

# Contract tests only (fast loop; optional CI job).
test-contracts:
    cargo test -p flowgrid --features full contract_

# Supply chain (requires `cargo install cargo-deny cargo-audit`; advisory DB updated at runtime).
deny:
    cargo deny check

audit:
    cargo audit
