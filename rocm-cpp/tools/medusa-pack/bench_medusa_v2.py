#!/usr/bin/env python3
"""
bench_medusa_v2.py — drive lemond's /api/v1/chat/completions against the
halo-1bit-2b model with medusa_speculative ∈ {0, 2, 3, 4} and append the
resulting decode tok/s rows to /home/bcloud/claude output/medusa-bench-2026-04-25.json
under the `runs_v2` key.

Offline tooling — Rule A bans Python in the runtime path, but bench drivers
that talk HTTP to lemond are fair game.

Assumes:
- lemond is running on http://127.0.0.1:8180 with halo-1bit-2b registered.
- lemond was restarted AFTER user_models.json was repointed at the v2
  sidecar (.h1b-medusa-v2). Verify via `curl /api/v1/models | jq` —
  the `medusa` entry under halo-1bit-2b should end with `-v2`.
- Same prompt as tonight's bench: "Write 200 words about the history of
  computing.", max_tokens=200, prompt_tokens=20.

Usage:
    python bench_medusa_v2.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request

LEMOND_URL = "http://127.0.0.1:8180/api/v1/chat/completions"
MODEL_ID   = "user.halo-1bit-2b"
PROMPT     = "Write 200 words about the history of computing."
MAX_TOKENS = 200
RUNS_PER_K = 3
KS         = [0, 2, 3, 4]
BENCH_PATH = "/home/bcloud/claude output/medusa-bench-2026-04-25.json"


def one_run(k: int) -> dict:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
        "recipe_options": {"medusa_speculative": k},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        LEMOND_URL, data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())
    t1 = time.perf_counter()
    wall = t1 - t0
    completion = out.get("usage", {}).get("completion_tokens", 0)
    prompt_tok = out.get("usage", {}).get("prompt_tokens", 0)
    tps = completion / wall if wall > 0 else 0.0
    return {
        "wall_s": round(wall, 4),
        "completion_tokens": completion,
        "prompt_tokens": prompt_tok,
        "tok_per_sec": round(tps, 3),
    }


def main():
    print(f"benching {LEMOND_URL} model={MODEL_ID} K={KS} runs={RUNS_PER_K}",
          file=sys.stderr)

    runs_v2 = []
    for k in KS:
        # Warm-up (not recorded) to avoid first-call pre-fill / page-in jitter.
        try:
            one_run(k)
        except Exception as e:
            print(f"  K={k} warm-up FAILED: {e}", file=sys.stderr)
            return 1

        rows = []
        for i in range(RUNS_PER_K):
            r = one_run(k)
            print(f"  K={k} run{i}: {r['tok_per_sec']:.2f} tok/s "
                  f"({r['completion_tokens']} tok in {r['wall_s']:.3f} s)",
                  file=sys.stderr)
            rows.append(r)
        median = statistics.median(r["tok_per_sec"] for r in rows)
        runs_v2.append({
            "medusa_k": k,
            "runs": rows,
            "median_tokps": round(median, 3),
        })

    # Append under `runs_v2` (preserve the existing `runs` k=0 / k=4 rows).
    with open(BENCH_PATH, "r") as f:
        bench = json.load(f)
    base_med = next((r["median_tokps"] for r in runs_v2 if r["medusa_k"] == 0),
                    None)
    speedups = {}
    for r in runs_v2:
        if r["medusa_k"] != 0 and base_med:
            speedups[str(r["medusa_k"])] = round(r["median_tokps"] / base_med, 3)
    bench["runs_v2"] = runs_v2
    bench["runs_v2_speedups"] = speedups
    bench["sidecar_v2"] = "/home/bcloud/halo-ai/models/medusa/halo-1bit-2b.h1b-medusa-v2"
    bench["sidecar_v2_kind"] = "parrishcorcoran/MedusaBitNet-2B-4T residual MLP"
    with open(BENCH_PATH, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"appended runs_v2 to {BENCH_PATH}", file=sys.stderr)
    print(f"  speedups vs K=0: {speedups}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
