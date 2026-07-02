#!/usr/bin/env python3
"""
NPU stress test — max capacity benchmark

Measures NPU throughput under increasing concurrency and context length
to find the saturation point.

Tests:
  1. Concurrency scaling (1-16 parallel requests)
  2. Context length scaling (64-8192 tokens)
  3. Max throughput (req/s at saturation)
  4. Latency distribution (p50, p95, p99)
"""

import json
import time
import sys
import threading
import urllib.request
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

NPU_PORT = 52625
URL = f"http://localhost:{NPU_PORT}/v1/chat/completions"
MODEL = "qwen3:0.6b"

def single_request(prompt: str, max_tokens: int = 50) -> dict:
    """Send one inference request, return timing + usage."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()

    t0 = time.time()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=300)
    t1 = time.time()

    data = json.loads(resp.read())
    usage = data.get("usage", {})
    return {
        "wall_s": round(t1 - t0, 4),
        "prefill_tok_s": usage.get("prefill_speed_tps", 0),
        "decode_tok_s": usage.get("decoding_speed_tps", 0),
        "ttft_s": usage.get("prefill_duration_ttft", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "ok": True,
    }

# -----------------------------------------------------------------------
# Test 1: Concurrency scaling
# -----------------------------------------------------------------------

def test_concurrency():
    print("─" * 70)
    print("  Test 1: Concurrency Scaling")
    print("  Sending requests in parallel to find saturation point")
    print("─" * 70)
    print()
    print(f"  {'Concurrency':>12s} {'Throughput':>12s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'Avg TTFT':>10s} {'Avg tok/s':>10s}")
    print(f"  {'':>12s} {'(req/s)':>12s} {'(s)':>8s} {'(s)':>8s} {'(s)':>8s} {'(s)':>10s} {'':>10s}")
    print(f"  {'-'*68}")

    prompt_short = "Hello. Reply in one word."
    results = []

    for concurrency in [1, 2, 4, 8, 12, 16]:
        latencies = []
        ttfts = []
        tok_speeds = []
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = []
            for _ in range(concurrency * 3):  # 3 rounds per concurrency level
                futs.append(pool.submit(single_request, prompt_short, 10))

            for f in as_completed(futs):
                try:
                    r = f.result()
                    if r["ok"]:
                        latencies.append(r["wall_s"])
                        ttfts.append(r["ttft_s"])
                        if r["decode_tok_s"] > 0:
                            tok_speeds.append(r["decode_tok_s"])
                except Exception as e:
                    pass

        t1 = time.time()
        elapsed = t1 - t0
        throughput = len(latencies) / elapsed if elapsed > 0 else 0

        if latencies:
            latencies.sort()
            p50 = statistics.median(latencies)
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            avg_ttft = statistics.mean(ttfts) if ttfts else 0
            avg_tok = statistics.mean(tok_speeds) if tok_speeds else 0
        else:
            p50 = p95 = p99 = avg_ttft = avg_tok = 0

        results.append((concurrency, throughput, p50, p95, p99, avg_ttft, avg_tok))
        print(f"  {concurrency:>12d} {throughput:>12.1f} {p50:>8.3f} {p95:>8.3f} {p99:>8.3f} {avg_ttft:>10.4f} {avg_tok:>10.1f}")
        sys.stdout.flush()

    print()
    return results

# -----------------------------------------------------------------------
# Test 2: Context length scaling
# -----------------------------------------------------------------------

def test_context_length():
    print("─" * 70)
    print("  Test 2: Context Length Scaling")
    print("  Measuring prefill speed at increasing prompt sizes")
    print("─" * 70)
    print()
    print(f"  {'Context':>12s} {'Prefill':>12s} {'Decode':>12s} {'TTFT':>10s} {'Wall':>10s}")
    print(f"  {'(tokens)':>12s} {'(tok/s)':>12s} {'(tok/s)':>12s} {'(s)':>10s} {'(s)':>10s}")
    print(f"  {'-'*56}")

    contexts = [
        ("Hi", 4),
        ("What is 2+2?", 8),
        ("Write a short poem about AI.", 16),
        ("Explain what a transformer is in 3 sentences.", 32),
        ("Write a paragraph about machine learning history.", 64),
        ("Write a detailed explanation of attention mechanisms in transformers.", 128),
    ]

    for prompt, expected in contexts:
        results = []
        for _ in range(3):
            r = single_request(prompt, 50)
            if r["ok"]:
                results.append(r)

        if results:
            avg_prefill = statistics.mean([r["prefill_tok_s"] for r in results])
            avg_decode = statistics.mean([r["decode_tok_s"] for r in results])
            avg_ttft = statistics.mean([r["ttft_s"] for r in results])
            avg_wall = statistics.mean([r["wall_s"] for r in results])
            actual_tokens = results[0]["prompt_tokens"]
        else:
            avg_prefill = avg_decode = avg_ttft = avg_wall = actual_tokens = 0

        print(f"  {actual_tokens:>6d} tok  {avg_prefill:>10.1f}  {avg_decode:>10.1f}  {avg_ttft:>8.4f}  {avg_wall:>8.2f}")
        sys.stdout.flush()

    print()

# -----------------------------------------------------------------------
# Test 3: Max throughput burst
# -----------------------------------------------------------------------

def test_max_throughput():
    print("─" * 70)
    print("  Test 3: Max Throughput (Burst)")
    print("  50 concurrent requests, min tokens, measure peak req/s")
    print("─" * 70)
    print()

    prompt = "Hi. Reply: ok"
    n_requests = 50

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=50) as pool:
        futs = [pool.submit(single_request, prompt, 5) for _ in range(n_requests)]
        results = [f.result() for f in as_completed(futs)]
    t1 = time.time()

    elapsed = t1 - t0
    successful = sum(1 for r in results if r["ok"])
    latencies = sorted([r["wall_s"] for r in results if r["ok"]])

    print(f"  Requests:    {n_requests}")
    print(f"  Successful:  {successful}")
    print(f"  Duration:    {elapsed:.2f}s")
    print(f"  Throughput:  {successful/elapsed:.1f} req/s")
    if latencies:
        print(f"  p50:         {statistics.median(latencies):.3f}s")
        print(f"  p95:         {latencies[int(len(latencies)*0.95)]:.3f}s")
        print(f"  p99:         {latencies[int(len(latencies)*0.99)]:.3f}s")
        print(f"  min:         {min(latencies):.3f}s")
        print(f"  max:         {max(latencies):.3f}s")

    print()

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  NPU STRESS TEST — Max Capacity Benchmark")
    print(f"  Platform: AMD Strix Halo (Ryzen AI MAX+ 395)")
    print(f"  NPU:      XDNA 2 aie2, 8 columns")
    print(f"  Model:    {MODEL}")
    print(f"  Backend:  INT8 NPU Engine (npu_engine_all)")
    print("=" * 70)
    print()

    test_concurrency()
    test_context_length()
    test_max_throughput()

    print("=" * 70)
    print("  NPU STRESS TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
