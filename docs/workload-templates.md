# Workload Templates

This catalog defines reusable baseline workloads for months 13-21.

## Purpose

- Reduce greenfield runs by using repeatable templates.
- Keep report artifacts consistent across workloads.
- Speed up first validated result for runtime and ML tracks.

## Template set

### 1) LLM tiny train baseline

- Intent: quick train smoke with report output.
- CLI shape: `flowgrid-llm train --steps 8 --run-report-out ...`
- Artifact: `target/mlops/train_tiny_report.json`

### 2) Eval split baseline

- Intent: split-aware eval run with optional checkpoint.
- CLI shape: `flowgrid-llm eval --split val --train-frac 0.8 --val-frac 0.1 ...`
- Artifact: `target/mlops/eval_smoke_report.json`

### 3) LoRA train smoke

- Intent: validate adapter fine-tune path and artifact logging.
- CLI shape: `flowgrid-llm train --lora --lora-targets q,v,o ...`
- Artifact: `target/mlops/train_lora_smoke_report.json`

### 4) Golden LLM candidate

- Intent: release-candidate train path with profile + reproducibility-friendly seed.
- CLI shape: `flowgrid-llm --profile local train ... --seed 7`
- Artifact: `target/mlops/golden_llm_candidate_train.json`

### 5) Classical ML baseline

- Intent: deterministic non-LLM baseline metrics for regression/classification.
- CLI shape: `flowgrid-ml --example golden_classical_ml -- --out ...`
- Artifact: `target/mlops/golden_classical_ml.json`

### 6) Multiclass classification baseline

- Intent: additional classical ML task type (macro-F1 over supported classes).
- CLI shape: `flowgrid-ml --example multiclass_golden_ml -- --out ...`
- Artifact: `target/mlops/multiclass_classical_ml.json`

## Adoption gate

For new workloads, choose a template first and justify deviations in PR notes.
