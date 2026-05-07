#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values, p):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    idx = (len(values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def request_once(url, max_tokens, timeout_s):
    payload = {
        "model": "flowgrid-local",
        "messages": [{"role": "user", "content": "kpi smoke ping"}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url.rstrip("/") + "/v1/chat/completions",
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    ms = (time.perf_counter() - t0) * 1000.0
    parsed = json.loads(body)
    usage = parsed.get("usage", {})
    return {
        "latency_ms": ms,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def main():
    ap = argparse.ArgumentParser(description="Run lightweight serve KPI smoke.")
    ap.add_argument("--base-url", default="http://127.0.0.1:9000")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    latencies = []
    prompt_tokens = 0
    completion_tokens = 0
    errors = []
    t_start = time.perf_counter()

    for i in range(max(1, args.requests)):
        try:
            res = request_once(args.base_url, args.max_tokens, args.timeout_s)
            latencies.append(res["latency_ms"])
            prompt_tokens += res["prompt_tokens"]
            completion_tokens += res["completion_tokens"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            errors.append({"index": i, "error": str(e)})

    elapsed_s = max(time.perf_counter() - t_start, 1e-9)
    ok = len(latencies)
    total = ok + len(errors)
    total_tokens = prompt_tokens + completion_tokens
    report = {
        "kind": "serve_kpi_smoke_v1",
        "base_url": args.base_url,
        "requests_total": total,
        "requests_ok": ok,
        "requests_error": len(errors),
        "error_rate": (len(errors) / total) if total else 1.0,
        "latency_ms_p50": percentile(sorted(latencies), 0.50),
        "latency_ms_p95": percentile(sorted(latencies), 0.95),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else 0.0,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "tokens_total": total_tokens,
        "tokens_per_sec": total_tokens / elapsed_s,
        "elapsed_s": elapsed_s,
        "errors": errors[:10],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote KPI report: {out}")


if __name__ == "__main__":
    main()
