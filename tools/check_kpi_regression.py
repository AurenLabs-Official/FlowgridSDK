#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


PROFILE_SHAPES: Tuple[Tuple[str, str], ...] = (
    ("local", "steady"),
    ("local", "burst"),
    ("hybrid", "steady"),
    ("hybrid", "burst"),
    ("cloud", "steady"),
    ("cloud", "burst"),
)

ERROR_RATE_GATES: Dict[Tuple[str, str], float] = {
    ("local", "steady"): 0.05,
    ("local", "burst"): 0.10,
    ("hybrid", "steady"): 0.05,
    ("hybrid", "burst"): 0.10,
    ("cloud", "steady"): 0.03,
    ("cloud", "burst"): 0.08,
}


def report_filename(profile: str, shape: str) -> str:
    suffix = "" if shape == "steady" else "_burst"
    return f"kpi_{profile}{suffix}.json"


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"missing KPI report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_pair(
    profile: str,
    shape: str,
    current: Dict[str, object],
    baseline: Dict[str, object],
    latency_regression_pct: float,
    throughput_regression_pct: float,
) -> List[str]:
    failures: List[str] = []
    name = f"{profile}/{shape}"

    current_error_rate = float(current.get("error_rate", 1.0))
    gate = ERROR_RATE_GATES[(profile, shape)]
    if current_error_rate > gate:
        failures.append(f"{name}: error_rate={current_error_rate:.4f} exceeds gate={gate:.4f}")

    current_p95 = float(current.get("latency_ms_p95", 0.0))
    baseline_p95 = float(baseline.get("latency_ms_p95", 0.0))
    if baseline_p95 > 0.0:
        max_allowed_p95 = baseline_p95 * (1.0 + (latency_regression_pct / 100.0))
        if current_p95 > max_allowed_p95:
            failures.append(
                f"{name}: latency_ms_p95={current_p95:.4f} exceeds baseline allowance={max_allowed_p95:.4f}"
            )

    current_tps = float(current.get("tokens_per_sec", 0.0))
    baseline_tps = float(baseline.get("tokens_per_sec", 0.0))
    if baseline_tps > 0.0:
        min_allowed_tps = baseline_tps * (1.0 - (throughput_regression_pct / 100.0))
        if current_tps < min_allowed_tps:
            failures.append(
                f"{name}: tokens_per_sec={current_tps:.4f} below baseline allowance={min_allowed_tps:.4f}"
            )

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Check KPI reports against baseline and gate bands.")
    ap.add_argument("--current-dir", default="target/mlops")
    ap.add_argument("--baseline-dir", default="docs/ops-artifacts/baselines/latest")
    ap.add_argument("--latency-regression-pct", type=float, default=25.0)
    ap.add_argument("--throughput-regression-pct", type=float, default=20.0)
    ap.add_argument("--strict-baseline", action="store_true")
    ap.add_argument("--require-current", action="store_true")
    ap.add_argument("--require-burst", action="store_true")
    args = ap.parse_args()

    current_dir = Path(args.current_dir)
    baseline_dir = Path(args.baseline_dir)

    failures: List[str] = []
    warnings: List[str] = []
    checked = 0

    for profile, shape in PROFILE_SHAPES:
        if shape == "burst" and not args.require_burst:
            continue

        filename = report_filename(profile, shape)
        current_path = current_dir / filename
        baseline_path = baseline_dir / filename
        scope = f"{profile}/{shape}"

        if not current_path.exists():
            if args.require_current:
                failures.append(f"{scope}: missing current report {current_path}")
            else:
                warnings.append(f"{scope}: current report missing, skipped")
            continue

        current = load_json(current_path)
        current_error_rate = float(current.get("error_rate", 1.0))
        gate = ERROR_RATE_GATES[(profile, shape)]
        if current_error_rate > gate:
            failures.append(f"{scope}: error_rate={current_error_rate:.4f} exceeds gate={gate:.4f}")

        if not baseline_path.exists():
            if args.strict_baseline:
                failures.append(f"{scope}: missing baseline report {baseline_path}")
            else:
                warnings.append(f"{scope}: baseline report missing, gate-only check applied")
            checked += 1
            continue

        baseline = load_json(baseline_path)
        failures.extend(
            evaluate_pair(
                profile=profile,
                shape=shape,
                current=current,
                baseline=baseline,
                latency_regression_pct=args.latency_regression_pct,
                throughput_regression_pct=args.throughput_regression_pct,
            )
        )
        checked += 1

    summary = {
        "kind": "kpi_regression_check_v1",
        "status": "pass" if not failures else "fail",
        "checked_reports": checked,
        "current_dir": str(current_dir),
        "baseline_dir": str(baseline_dir),
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
