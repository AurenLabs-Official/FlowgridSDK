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
    cargo +1.85 check -p flowgrid --features full

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

# ML-core smoke loop (matches CI `ml-core-smoke`).
check-ml-core:
    cargo check -p flowgrid-data -p flowgrid-eval -p flowgrid-ml -p flowgrid-cli -p flowgrid-serve
    cargo test -p flowgrid-data -p flowgrid-eval -p flowgrid-ml

# LLM golden path with reproducible train/eval reports.
golden-llm-path:
    cargo run -p flowgrid-cli -- prepare -i README.md -o target/mlops/golden_readme.bin --byte-level
    cargo run -p flowgrid-cli --profile local -- train --tokens target/mlops/golden_readme.bin --steps 8 --epochs 2 --batch-size 2 --learn --seed 7 --run-report-out target/mlops/golden_llm_train.json
    cargo run -p flowgrid-cli --profile local -- eval --dataset target/mlops/golden_readme.bin --split test --train-frac 0.8 --val-frac 0.1 --baseline-ppl 100.0 --max-regression-pct 100.0 --run-report-out target/mlops/golden_llm_eval.json

# Classical ML golden path report.
golden-classical-ml-path:
    cargo run -p flowgrid-ml --example golden_classical_ml -- --out target/mlops/golden_classical_ml.json

golden-multiclass-ml-path:
    cargo run -p flowgrid-ml --example multiclass_golden_ml -- --out target/mlops/multiclass_classical_ml.json

# After `just repro-ml-smoke`, delta between paired train runs should be ~0 on CE fields.
compare-train-repro-delta:
    python tools/compare_run_reports.py --kind train --a target/mlops/repro_a.json --b target/mlops/repro_b.json

# UI-oriented workload template smokes.
template-train-lora-smoke:
    cargo run -p flowgrid-cli -- prepare -i README.md -o target/mlops/golden_readme.bin --byte-level
    cargo run -p flowgrid-cli -- train --tokens target/mlops/golden_readme.bin --steps 8 --vocab 256 --block 32 --layers 2 --embd 64 --lora --lora-targets q,v,o --run-report-out target/mlops/train_lora_smoke_report.json

template-eval-val-gate:
    cargo run -p flowgrid-cli -- prepare -i README.md -o target/mlops/golden_readme.bin --byte-level
    cargo run -p flowgrid-cli -- eval --dataset target/mlops/golden_readme.bin --split val --train-frac 0.8 --val-frac 0.1 --block 32 --stride 32 --run-report-out target/mlops/eval_val_gate_report.json

# Quick reproducibility gate (same seed => same quality band).
repro-ml-smoke:
    cargo run -p flowgrid-cli -- prepare -i README.md -o target/mlops/repro_readme.bin --byte-level
    cargo run -p flowgrid-cli --profile local -- train --tokens target/mlops/repro_readme.bin --steps 6 --epochs 1 --batch-size 2 --learn --seed 11 --run-report-out target/mlops/repro_a.json
    cargo run -p flowgrid-cli --profile local -- train --tokens target/mlops/repro_readme.bin --steps 6 --epochs 1 --batch-size 2 --learn --seed 11 --run-report-out target/mlops/repro_b.json
    python tools/check_train_repro.py --a target/mlops/repro_a.json --b target/mlops/repro_b.json --max-delta 1e-6

# Serve KPI smoke for baseline capture (run while `flowgrid-serve` is up).
kpi-serve-local:
    python tools/serve_kpi_smoke.py --base-url http://127.0.0.1:9000 --requests 32 --max-tokens 32 --out target/mlops/kpi_local.json

kpi-serve-hybrid:
    python tools/serve_kpi_smoke.py --base-url http://127.0.0.1:9000 --requests 64 --max-tokens 64 --out target/mlops/kpi_hybrid.json

kpi-serve-cloud:
    python tools/serve_kpi_smoke.py --base-url http://127.0.0.1:9000 --requests 128 --max-tokens 64 --out target/mlops/kpi_cloud.json

# Validate release gates from generated artifacts.
validate-release-gates:
    python tools/validate_release_gates.py

# Aggregated ops-ready artifact loop.
ops-release-pack:
    just golden-llm-path
    just golden-classical-ml-path
    just template-train-lora-smoke
    just template-eval-val-gate
    just repro-ml-smoke
    just validate-release-gates
    python tools/build_kpi_achievement_report.py --out target/mlops/kpi_achievement_report.md

# Supply chain (requires `cargo install cargo-deny cargo-audit`; advisory DB updated at runtime).
deny:
    cargo deny check

audit:
    cargo audit
