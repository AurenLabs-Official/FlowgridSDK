#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing artifact: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def assert_le(name, value, limit):
    if value > limit:
        raise ValueError(f"{name}={value} exceeds limit={limit}")


def assert_ge(name, value, limit):
    if value < limit:
        raise ValueError(f"{name}={value} below minimum={limit}")


def main():
    ap = argparse.ArgumentParser(description="Validate Flowgrid release gate artifacts.")
    ap.add_argument("--train", default="target/mlops/golden_llm_train.json")
    ap.add_argument("--eval", default="target/mlops/golden_llm_eval.json")
    ap.add_argument("--classical", default="target/mlops/golden_classical_ml.json")
    ap.add_argument("--kpi-local", default="target/mlops/kpi_local.json")
    ap.add_argument("--kpi-hybrid", default="target/mlops/kpi_hybrid.json")
    ap.add_argument("--kpi-cloud", default="target/mlops/kpi_cloud.json")
    ap.add_argument("--require-kpi", action="store_true")
    ap.add_argument("--max-train-after-ce", type=float, default=20.0)
    ap.add_argument("--max-eval-ppl", type=float, default=100.0)
    ap.add_argument("--min-classical-r2", type=float, default=0.95)
    ap.add_argument("--min-classical-f1", type=float, default=0.75)
    ap.add_argument("--max-kpi-error-rate", type=float, default=0.10)
    ap.add_argument("--kpi-regression-check", action="store_true")
    ap.add_argument("--kpi-current-dir", default="target/mlops")
    ap.add_argument("--kpi-baseline-dir", default="docs/ops-artifacts/baselines/latest")
    ap.add_argument("--kpi-latency-regression-pct", type=float, default=25.0)
    ap.add_argument("--kpi-throughput-regression-pct", type=float, default=20.0)
    ap.add_argument("--kpi-strict-baseline", action="store_true")
    ap.add_argument("--kpi-require-burst", action="store_true")
    args = ap.parse_args()

    train = load_json(args.train)
    eval_run = load_json(args.eval)
    classical = load_json(args.classical)

    assert_le("train.after_ce", float(train.get("after_ce", 1e9)), args.max_train_after_ce)
    eval_report = eval_run.get("report", {})
    assert_le("eval.report.ppl", float(eval_report.get("ppl", 1e9)), args.max_eval_ppl)

    reg = classical.get("regression", {})
    cls = classical.get("classification", {})
    assert_ge("classical.regression.r2", float(reg.get("r2", 0.0)), args.min_classical_r2)
    assert_ge("classical.classification.f1", float(cls.get("f1", 0.0)), args.min_classical_f1)

    kpi_paths = [args.kpi_local, args.kpi_hybrid, args.kpi_cloud]
    for p in kpi_paths:
        path = Path(p)
        if not path.exists():
            if args.require_kpi:
                raise FileNotFoundError(f"missing required KPI artifact: {path}")
            continue
        report = load_json(path)
        assert_le(f"{path.name}.error_rate", float(report.get("error_rate", 1.0)), args.max_kpi_error_rate)

    should_run_kpi_regression = args.kpi_regression_check or args.require_kpi
    if should_run_kpi_regression:
        cmd = [
            sys.executable,
            "tools/check_kpi_regression.py",
            "--current-dir",
            args.kpi_current_dir,
            "--baseline-dir",
            args.kpi_baseline_dir,
            "--latency-regression-pct",
            str(args.kpi_latency_regression_pct),
            "--throughput-regression-pct",
            str(args.kpi_throughput_regression_pct),
            "--require-current",
        ]
        if args.kpi_strict_baseline:
            cmd.append("--strict-baseline")
        if args.kpi_require_burst:
            cmd.append("--require-burst")
        subprocess.run(cmd, check=True)

    print(
        json.dumps(
            {
                "kind": "release_gate_validation_v1",
                "status": "pass",
                "require_kpi": args.require_kpi,
                "train": args.train,
                "eval": args.eval,
                "classical": args.classical,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
