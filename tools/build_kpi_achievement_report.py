#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def kpi_line(name, value, target):
    return f"- {name}: `{value}` (target `{target}`)"


def main():
    ap = argparse.ArgumentParser(description="Build operations-ready KPI markdown report.")
    ap.add_argument("--kpi-local", default="target/mlops/kpi_local.json")
    ap.add_argument("--kpi-hybrid", default="target/mlops/kpi_hybrid.json")
    ap.add_argument("--kpi-cloud", default="target/mlops/kpi_cloud.json")
    ap.add_argument("--llm-train", default="target/mlops/golden_llm_train.json")
    ap.add_argument("--llm-eval", default="target/mlops/golden_llm_eval.json")
    ap.add_argument("--classical", default="target/mlops/golden_classical_ml.json")
    ap.add_argument("--out", default="target/mlops/kpi_achievement_report.md")
    args = ap.parse_args()

    local = read_json(args.kpi_local)
    hybrid = read_json(args.kpi_hybrid)
    cloud = read_json(args.kpi_cloud)
    llm_train = read_json(args.llm_train)
    llm_eval = read_json(args.llm_eval)
    classical = read_json(args.classical)

    lines = ["# KPI Achievement Report", ""]
    lines.append("## Runtime KPIs")
    for name, report in [("local", local), ("hybrid", hybrid), ("cloud", cloud)]:
        if report is None:
            lines.append(f"- {name}: missing report")
            continue
        lines.append(f"- {name}:")
        lines.append(kpi_line("p95 latency ms", report.get("latency_ms_p95", 0.0), "<= baseline"))
        lines.append(kpi_line("tokens/s", report.get("tokens_per_sec", 0.0), ">= baseline"))
        lines.append(kpi_line("error rate", report.get("error_rate", 1.0), "<= 0.10"))
    lines.append("")
    lines.append("## ML KPIs")
    if llm_train:
        lines.append(
            kpi_line("LLM train after_ce", llm_train.get("after_ce", "n/a"), "non-increasing vs before_ce")
        )
    else:
        lines.append("- LLM train report missing")
    if llm_eval:
        rep = llm_eval.get("report", {})
        lines.append(kpi_line("LLM eval ppl", rep.get("ppl", "n/a"), "within regression band"))
    else:
        lines.append("- LLM eval report missing")
    if classical:
        reg = classical.get("regression", {})
        cls = classical.get("classification", {})
        lines.append(kpi_line("Classical ML regression r2", reg.get("r2", "n/a"), ">= 0.95"))
        lines.append(kpi_line("Classical ML classification f1", cls.get("f1", "n/a"), ">= 0.75"))
    else:
        lines.append("- Classical ML report missing")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote KPI achievement report: {out}")


if __name__ == "__main__":
    main()
