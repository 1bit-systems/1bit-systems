# Benchmarks

Standardized, reproducible benchmark suite that auto-updates [`site/benchmarks.json`](../site/benchmarks.json) on every push.

## Usage

```bash
cmake --build build --target bench-all   # run all benchmarks, write JSON
cmake --build build --target bench-ci    # CI variant: lightweight, ~30s
```

## Output

Each benchmark writes a JSON record to `site/benchmarks.json` with:

```json
{
  "updated": "2026-07-20T21:00:00Z",
  "commit": "abc123def",
  "hardware": { "cpu": "AMD Ryzen AI Max+ 395", "npu": "XDNA 2", "gpu": "Radeon 8060S" },
  "engines": {
    "q1_gemv": {
      "tok_s": 417,
      "tflops": null,
      "status": "validated",
      "label": "Q1_0 GEMV kernel (28-layer model, 128B blocks)",
      "table": "kernel",
      "display_name": "Q1 GEMV",
      "backend": "ROCm HIP (fused kernel)"
    }
  }
}
```

Every `bench-*` target writes its result to a timestamped file under `bench/results/` and the CI run aggregates all results into `site/benchmarks.json` with the current commit hash.
