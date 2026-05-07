#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_report(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data.get("before_ce", 0.0)), float(data.get("after_ce", 0.0))


def main():
    ap = argparse.ArgumentParser(description="Compare two train reports for reproducibility.")
    ap.add_argument("--a", required=True, help="first train report path")
    ap.add_argument("--b", required=True, help="second train report path")
    ap.add_argument("--max-delta", type=float, default=1e-6)
    args = ap.parse_args()

    before_a, after_a = read_report(args.a)
    before_b, after_b = read_report(args.b)
    before_delta = abs(before_a - before_b)
    after_delta = abs(after_a - after_b)
    print(
        json.dumps(
            {
                "kind": "train_repro_check_v1",
                "before_ce_delta": before_delta,
                "after_ce_delta": after_delta,
                "max_delta": args.max_delta,
            },
            indent=2,
        )
    )
    if before_delta > args.max_delta or after_delta > args.max_delta:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
