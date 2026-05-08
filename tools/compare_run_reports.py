#!/usr/bin/env python3
"""Compare two Flowgrid train or eval run-report JSON files."""

import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Compare train or eval run reports (delta).")
    ap.add_argument("--kind", choices=("train", "eval"), required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    ja = load_json(args.a)
    jb = load_json(args.b)

    if args.kind == "train":
        keys = ("before_ce", "after_ce")
        out = {
            "kind": "run_report_compare_train_v1",
            "delta_before_ce": float(jb["before_ce"]) - float(ja["before_ce"]),
            "delta_after_ce": float(jb["after_ce"]) - float(ja["after_ce"]),
        }
    else:
        ra = ja.get("report", {})
        rb = jb.get("report", {})
        out = {
            "kind": "run_report_compare_eval_v1",
            "delta_ppl": float(rb.get("ppl", 0.0)) - float(ra.get("ppl", 0.0)),
            "delta_mean_ce": float(rb.get("mean_ce", 0.0)) - float(ra.get("mean_ce", 0.0)),
        }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
