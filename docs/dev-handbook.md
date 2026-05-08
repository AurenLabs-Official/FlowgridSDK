# Developer Handbook

This handbook is the practical day-to-day guide for contributors working in this repository.
For project policy and release notes, also see `CONTRIBUTING.md` and `CHANGELOG.md`.

## 1) Quick orientation

- Stable crate: `crates/flowgrid` (public SDK surface).
- Preview/local-LLM crates: `flowgrid-model`, `flowgrid-train`, `flowgrid-serve`, `flowgrid-cli`, `flowgrid-checkpoint`, etc.
- Classical ML baseline crate: `crates/flowgrid-ml`.
- Main docs:
  - `README.md` (top-level usage, features, compatibility).
  - `docs/http.md` (timeouts, TLS, proxies).
  - `docs/resilience.md` (retry behavior).
  - `docs/observability.md` (tracing/metrics).
  - `docs/llm/overview.md` (LLM preview stack).

## 2) Prerequisites

- Rust stable toolchain.
- Rust nightly for semver checks (`cargo +nightly rustdoc`).
- Optional tools:
  - `just` (task runner).
  - `cargo-semver-checks`.
  - `cargo-deny`.
  - `cargo-audit`.

## 3) Build and test commands

Preferred (matches CI helper tasks):

```bash
just fmt
just clippy
just test-full
just check-examples
just check-msrv
just test-contracts
just check-ml-core
```

Direct Cargo equivalents:

```bash
cargo fmt --all -- --check
cargo clippy -p flowgrid --features full -- -D warnings
cargo test -p flowgrid --features full
cargo check -p flowgrid --examples --features full
cargo +1.85 check -p flowgrid --features full
```

## 4) Feature-flag rules (important)

- Do not run `--all-features` on `flowgrid` in normal workflows.
- Exactly one TLS backend must be active:
  - `tls-rustls` or
  - `tls-native`.
- Typical full test path:

```bash
cargo test -p flowgrid --features full
```

## 5) Windows notes (`LNK1104` and locking)

If you hit linker/file-lock issues on Windows:

```powershell
$env:CARGO_TARGET_DIR = "target\\win-full"
$env:CARGO_BUILD_JOBS = "1"
cargo test -p flowgrid --features full -j 1
```

Additional mitigations:

- Avoid parallel cargo jobs in multiple terminals for the same target dir.
- Pause/close background builds temporarily.
- Retry with an isolated `CARGO_TARGET_DIR`.

## 6) Daily development flow

1. Sync branch and implement change.
2. Run crate-local checks first (fast loop), then broader checks.
3. If behavior/API changed, update docs and changelog in same PR.
4. Keep PR focused and small; separate refactors from behavior changes where possible.

Recommended loop:

```bash
cargo check -p <changed-crate>
cargo test -p <changed-crate>
cargo clippy -p <changed-crate> --all-targets -- -D warnings
```

Then before merge:

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## 7) API and semver workflow

When you change `flowgrid` public API (crate-root `pub use` surface):

1. Regenerate rustdoc JSON on nightly.
2. Run semver checks against baseline.
3. Update `crates/flowgrid/semver/baseline_rustdoc.json` in the same PR only for intentional API changes.

Use:

```bash
just semver-local
```

## 8) Documentation expectations

Update docs when behavior changes, especially:

- `README.md` for user-facing behavior and compatibility.
- `docs/http.md` for transport/timeouts/TLS/proxy behavior.
- `docs/resilience.md` for retry policy changes.
- `docs/llm/overview.md` for preview LLM runtime/config changes.
- `CHANGELOG.md` under `[Unreleased]`.

## 9) Contract fixtures and tests

- Contract fixtures live under:
  - `crates/flowgrid/tests/fixtures/contracts`
- Keep fixture naming conventions documented in `README.md`.
- Add/update fixtures when response shape or relied-on fields change.

Fast fixture loop:

```bash
just test-contracts
```

## 10) Local LLM/serve notes (preview)

Common local commands:

```bash
cargo run -p flowgrid-cli -- prepare -i README.md -o target/readme.bin --byte-level
cargo run -p flowgrid-cli --profile local -- train --tokens target/readme.bin --steps 16 --epochs 2 --batch-size 2 --n-head 4 --n-kv-head 0 --run-report-out target/mlops/train_demo.json
cargo run -p flowgrid-cli --profile local -- eval --dataset target/readme.bin --split test --train-frac 0.8 --val-frac 0.1 --run-report-out target/mlops/eval_demo.json
cargo run -p flowgrid-serve
```

Operational loops:

```bash
just golden-llm-path
just golden-classical-ml-path
just golden-multiclass-ml-path
just template-train-lora-smoke
just template-eval-val-gate
just repro-ml-smoke
just compare-train-repro-delta
just validate-release-gates
just ops-release-pack
```

Useful env vars for `flowgrid-serve`:

- `FLOWGRID_SERVE_CHECKPOINT`
- `FLOWGRID_SERVE_TEMPERATURE`
- `FLOWGRID_SERVE_TOP_K`
- `FLOWGRID_SERVE_SEED`
- `FLOWGRID_SERVE_REQUEST_TIMEOUT_MS`
- `FLOWGRID_SERVE_RPS`
- `FLOWGRID_SERVE_BURST`
- `FLOWGRID_SERVE_WORKERS`
- `FLOWGRID_SERVE_STREAM_BUFFER`
- `FLOWGRID_SERVE_MAX_NEW_TOKENS`
- `FLOWGRID_DEPLOYMENT_PROFILE` (`local`, `cloud`, `hybrid`)

See `docs/llm/overview.md` for full details and current preview constraints.
Runbook entrypoint for incidents and profile bring-up: `docs/runbook-quickstart.md`.
Roadmap governance references: `docs/kpi-baseline-authority.md`, `docs/release-gates-parity.md`, `docs/runtime-resilience-program.md`, `docs/workload-templates.md`, `docs/lifecycle-policy.md`, `docs/cycle-review-35m.md`, `docs/ops-readiness-30m.md`.

## 11) Release hygiene checklist

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] Relevant docs updated
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Semver checks completed if public API changed

