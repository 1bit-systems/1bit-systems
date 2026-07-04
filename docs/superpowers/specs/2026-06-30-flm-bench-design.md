# flm-bench: Open-Source FastFlowLM Benchmark Suite

**Date:** 2026-06-30  
**Location:** `docs/superpowers/specs/2026-06-30-flm-bench-design.md`  
**Status:** Design Draft  

---

## 1. Purpose

A standalone, open-source (MIT) Python CLI tool `flm-bench` that benchmarks any FastFlowLM model running on AMD Ryzen AI NPU hardware. It measures end-to-end inference latency and throughput across model sizes, prompt lengths, generation lengths, context lengths, and power modes. The output is structured (JSON + CSV) for programmatic consumption and comparison across hardware and software versions.

This is not an internal script — it lives as a first-class open-source project alongside [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) with its own repo, CI, and documentation.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  flm-bench CLI                                          │
│                                                         │
│  ┌────────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │ runner.py  │  │ models.py │  │ reporters.py      │   │
│  │ server mgmt│→ │ configs   │→ │ stdout/JSON/CSV   │   │
│  │ requests   │  │ per model │  │                   │   │
│  └────────────┘  └──────────┘  └───────────────────┘   │
│       │                                                 │
│       ▼ HTTP POST /v1/chat/completions                   │
│  ┌────────────┐                                          │
│  │ flm serve  │  (subprocess or external)                │
│  └────────────┘                                          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   ./results/benchmark_<timestamp>.json
   ./results/benchmark_<timestamp>.csv
```

**Key design principles:**
- Zero dependencies beyond Python 3.10+ stdlib and `requests` (for HTTP API)
- Server lifecycle managed automatically (start `flm serve`, wait for ready, run, kill)
- Also supports `--server` flag to attach to an already-running server
- Warm-up rounds first, then measurement rounds averaged
- Fail fast: validate NPU stack before starting
- Append-only results directory

## 3. Benchmark Dimensions

| Dimension | Values | Measured by |
|---|---|---|
| **Models** | Auto-detected from installed models | `flm list --json` |
| **Prompt length** | Short (~16 tok), Medium (~256 tok), Long (~3093 tok) | Pre-filled prompt text |
| **Generation length** | 32, 128, 512 tokens | `max_tokens` parameter |
| **Context length** | Default, 4096, 8192, 16384 | `--ctx-len` server flag |
| **Power mode** | performance, balanced, powersaver | `--pmode` server flag |
| **Streaming** | true, false | `stream` API parameter |

**Note on server restarts:** Changing `--ctx-len` or `--pmode` requires restarting `flm serve`. The runner handles this automatically: for each unique (ctx-len, pmode) combination, it starts a fresh server, runs all tests with that config, then shuts down and starts the next config.

**Phase 1 scope:** All dimensions except Concurrency (sequential only).  
**Phase 2 scope:** Concurrent load testing (multiple simultaneous requests).

## 4. Per-Request Metrics

Captured directly from the FastFlowLM API response `usage` object:

| Metric | Source field | Unit |
|---|---|---|
| Time to First Token (TTFT) | `prefill_duration_ttft` | seconds |
| Prefill throughput | `prefill_speed_tps` | tokens/second |
| Decoding throughput | `decoding_speed_tps` | tokens/second |
| Decoding duration | `decoding_duration` | seconds |
| KV cache occupancy | `kv_token_occupancy_rate_percentage` | % |
| Prompt tokens | `prompt_tokens` | tokens |
| Completion tokens | `completion_tokens` | tokens |
| Total wall time | client-side `time.monotonic()` | seconds |

Plus system info collected once per benchmark run via `xrt-smi examine -f JSON` and `flm validate --json`:
- NPU firmware version
- XRT version
- CPU model & core count
- Total RAM
- Power mode

## 5. CLI Interface

```
usage: flm-bench [-h] [--models MODELS [MODELS ...]] [--all-models]
                 [--prompts {short,medium,long} [{short,medium,long} ...]]
                 [--max-tokens MAX_TOKENS [MAX_TOKENS ...]]
                 [--ctx-len CTX_LEN [CTX_LEN ...]]
                 [--pmode {performance,balanced,powersaver}]
                 [--iterations ITERATIONS]
                 [--warmup WARMUP]
                 [--output-dir OUTPUT_DIR]
                 [--server SERVER]
                 [--json] [--csv] [--table]
                 [--verbose]

Benchmark FastFlowLM models on AMD NPU hardware.

Examples:
  flm-bench --all-models --prompts short medium --max-tokens 32 128
  flm-bench --models qwen3:0.6b llama3.1:8b --iterations 5
  flm-bench --server http://localhost:52625 --models qwen3:0.6b
```

## 6. Output Format

### stdout table (human-readable)

```
╔══════════════════════════════════════════════════════════════╗
║ flm-bench results — 2026-06-30T23:15:00Z                    ║
║ System: Ryzen AI Max+ 395 | 32 cores | 122 GB               ║
║ NPU: RyzenAI-npu5 | FW 1.1.2.65 | XRT 2.21.75              ║
╠══════════════════════════════════════════════════════════════╣
║ Model        │ Prompt│ Gen│ TTFT  │ Prefill│ Decode │ KV%   ║
║              │ tok   │ tok│ (ms)  │ tok/s  │ tok/s  │       ║
╠══════════════╪═══════╪════╪═══════╪════════╪════════╪═══════╣
║ qwen3:0.6b   │   16  │ 32 │  523  │ 30.6   │ 72.4   │ 0.15% ║
║ qwen3:0.6b   │  3093 │ 32 │ 45500 │ 68.0   │ 70.1   │ 9.5%  ║
║ llama3.1:8b  │   16  │ 32 │  890  │ 18.0   │ 38.2   │ 0.02% ║
║ llama3.1:8b  │  3093 │ 32 │ 72000 │ 42.9   │ 36.5   │ 9.4%  ║
╚══════════════════════════════════════════════════════════════╝
```

### JSON results file

```json
{
  "meta": {
    "timestamp": "2026-06-30T23:15:00Z",
    "flm_bench_version": "0.1.0",
    "system": { "cpu": "AMD RYZEN AI MAX+ 395", "cores": 32, "ram_gb": 122 },
    "npu": { "device": "RyzenAI-npu5", "fw": "1.1.2.65", "xrt": "2.21.75" }
  },
  "config": {
    "models": ["qwen3:0.6b"],
    "prompts": ["short", "medium", "long"],
    "max_tokens": [32, 128],
    "ctx_len": ["default"],
    "pmode": "performance",
    "stream": false
  },
  "tests": [
    {
      "model": "qwen3:0.6b",
      "prompt_type": "short",
      "prompt_tokens": 16,
      "max_tokens": 32,
      "completion_tokens": 32,
      "prefill_duration_ttft": 0.523,
      "prefill_speed_tps": 30.6,
      "decoding_duration": 0.442,
      "decoding_speed_tps": 72.4,
      "kv_occupancy_pct": 0.15,
      "total_tokens": 48
    }
  ]
}
```

### CSV summary

```csv
model,prompt_type,prompt_tokens,max_tokens,completion_tokens,prefill_duration_ttft,prefill_speed_tps,decoding_duration,decoding_speed_tps,kv_occupancy_pct,total_tokens
qwen3:0.6b,short,16,32,32,0.523,30.6,0.442,72.4,0.15,48
qwen3:0.6b,medium,256,32,32,4.2,61.0,0.441,72.6,0.88,288
```

## 7. File Structure

```
flm-bench/
├── flm_bench/
│   ├── __init__.py           # Version string
│   ├── __main__.py           # python -m flm_bench entry point
│   ├── cli.py                # Argument parser, main dispatch
│   ├── runner.py             # Server lifecycle & HTTP request orchestration
│   ├── models.py             # Model configuration, prompt generation
│   ├── reporters.py          # Stdout table, JSON, CSV output writers
│   ├── system.py             # System info collection (xrt-smi, flm validate)
│   └── prompts.py            # Standardized prompt templates
├── results/                  # Default output directory
├── tests/
│   ├── test_runner.py
│   ├── test_models.py
│   ├── test_reporters.py
│   └── test_prompts.py
├── pyproject.toml             # PEP 621 build config
├── LICENSE                    # MIT
├── README.md
└── .github/
    └── workflows/
        └── ci.yml            # Run tests, lint with ruff
```

## 8. Prompt Templates

Three standard prompt sizes:

1. **Short (~16 tokens):** `"What is the capital of France?"`
2. **Medium (~256 tokens):** A factual question paragraph of exactly 256 tokens (verified per model). Content should be a neutral, factual description of AI hardware (to avoid triggering model biases). Example candidate: A technical description of the AMD Ryzen AI NPU architecture taken from public AMD documentation, padded/trimmed to hit 256 tokens for each model's tokenizer.
3. **Long (~3093 tokens):** The same story from FastFlowLM's own `bench_config.json` (the "Reclaimer" story, 3093 tokens in the reference tokenizer)

The exact token counts are verified by running each prompt through `flm run --prompt` once per model and reading back `prompt_tokens` from the API response.

## 9. Request Flow

```
1. Parse CLI args
2. Validate system (flm validate --json)
3. Build test matrix (Cartesian product of all dimensions)
4. Start flm serve subprocess (or attach to existing --server)
5. Wait for server ready (poll /v1/models)
6. For each test case:
   a. If model not loaded, server reloads automatically
   b. Warm-up: 1 request (discarded)
   c. Measure: N iterations (default 3), each recorded
   d. Store averaged results
7. Kill subprocess (if managed)
8. Write JSON, CSV, stdout table
9. Exit
```

## 10. Testing Strategy

- **Unit tests:** Each module tested independently with mocked HTTP/process calls
- **Integration tests:** Run against an actual `flm serve` with qwen3:0.6b (the small model)
- **CI:** `pytest` + `ruff` linting on pushes/PRs

## 11. Error Handling

| Scenario | Behavior |
|---|---|
| NPU not available / `flm validate` fails | Print error, exit code 1 |
| Server fails to start | Print stderr from subprocess, exit code 1 |
| Request times out (>60s default) | Log warning, skip test case, continue |
| Model not installed | Print available models, skip, exit code 1 at end |
| Partial results available | Write what was collected before error, non-zero exit |

## 12. Open Source Considerations

- **License:** MIT (matching FastFlowLM)
- **Hosting:** Separate GitHub repo `FastFlowLM/flm-bench` (or a `benchmark/` dir in the main FastFlowLM repo — TBD)
- **CI:** GitHub Actions — runs on push/PR, tests the benchmark suite itself
- **Documentation:** README with install instructions, example output, and a "how to interpret results" guide
- **Versioning:** Semver, independent of FastFlowLM's version

---

## Questions for Review

1. Should this live as a **separate repo** (`FastFlowLM/flm-bench`) or in a `benchmark/` directory inside the main FastFlowLM repo?
2. Should we include **quality metrics** (e.g., exact-match accuracy on 5 fixed Q&A prompts) alongside performance metrics, or strictly performance?
3. For the `long` prompt (~3093 tokens), should we use the exact text from FastFlowLM's `bench_config.json` (the "Reclaimer" story) or a different canonical text?
   - **Pros of Reclaimer story:** It's already in the FastFlowLM repo, it's the story they use for their own benchmarking, and it's a fixed canonical reference.
   - **Cons:** It's fiction with named entities; tokenization differences across models could produce slightly different token counts.
