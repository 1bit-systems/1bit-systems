# Phase 1a: Parameterised GEMM Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce verified-correct NPU GEMM output at any M×K×N by parameterizing the existing MLIR-AIE / IRON xclbin pipeline and building an automated correctness test harness.

**Architecture:** The pipeline already works at specific sizes (e.g., `test_correctness.cpp` proves 1024×1024×2048). We generalize `compile_gemm.py` as a wrapper around the existing IRON Python → aiecc flow, then build a Python verification tool that: generates test vectors → compiles xclbin → calls a C runner for NPU submission → reads C output → unshuffles → compares vs CPU BF16 reference.

**Tech Stack:** Python 3.10+ (IRON/MLIR-AIE dialect), MLIR-AIE (`aie-opt`, `aiecc`, `aie-translate` from `/home/bcloud/mlir-aie/build/bin/`), XRT 2.21.75, AMD XDNA DRM driver, C (NPU runner), numpy (CPU reference)

## Global Constraints

- All MLIR compilation uses `/home/bcloud/torch2aie/.venv/bin/python` for the IRON Python dialect
- Tile sizes fixed: m=128, k=64, n=128 (baked into microkernel schedule)
- NPU device: `/dev/accel/accel0` (8 physical columns, Strix Halo)
- M divisible by m=128, K divisible by k=64, N divisible by n=128
- `(M/m)*(N/n)` must be multiple of `(4 * n_aie_cols * n_aie_rows)` = 32 (1-row) or 64 (2-row)
- All outputs to `config1/build/` — xclbin + insts.txt

---
### Task 1: `compile_gemm.py` — Parameterized xclbin Compiler Wrapper

**Files:**
- Create: `tools/compile_gemm.py` at `/home/bcloud/npu-sandbox/npu-infer/tools/compile_gemm.py`

**Interfaces:**
- Consumes: CLI args `--M`, `--K`, `--N`, `--m 128`, `--k 64`, `--n 128`, `--rows {1,2}`, `--force`
- Produces: `final_{M}x{K}x{N}_{m}x{k}x{n}.xclbin` + `insts_{M}x{K}x{N}_{m}x{k}x{n}.txt` in `config1/build/`
- Returns: `(xclbin_path, insts_path)`
- Raises: `ValueError` on invalid dims, `RuntimeError` on compile failure

**Details:**

- [ ] **Step 1: Write failing test**

```python
# tests/test_compile_gemm.py
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.compile_gemm import generate_xclbin, VALID_ROWS

def test_validate_dims():
    try:
        generate_xclbin(M=100, K=1024, N=2048)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "divisible" in str(e)

def test_valid_rows():
    assert 1 in VALID_ROWS
    assert 2 in VALID_ROWS
    assert "n1_core_bf16" in VALID_ROWS[1]
```

Run:
```bash
cd /home/bcloud/npu-sandbox/npu-infer && python -m pytest tests/test_compile_gemm.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 2: Implement `compile_gemm.py`**

```python
#!/usr/bin/env python3
"""
Parameterized GEMM xclbin compiler wrapper.
Generates MLIR via IRON Python dialect, compiles via aiecc.

Usage:
  python compile_gemm.py --M 128 --K 1024 --N 2048 --rows 1
"""

import argparse, os, subprocess, sys
from pathlib import Path

SRCDIR = Path("/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1")
BUILD_DIR = SRCDIR / "build"
AIECC = Path("/home/bcloud/mlir-aie/build/bin/aiecc")
PYTHON = Path("/home/bcloud/torch2aie/.venv/bin/python")
VALID_ROWS = {1: "n1_core_bf16", 2: "n2_core_placed"}


def generate_xclbin(M, K, N, m=128, k=64, n=128, rows=1, force=False):
    """Generate xclbin + insts for given GEMM dims. Returns (xclbin_path, insts_path)."""
    if M % m != 0:
        raise ValueError(f"M={M} not divisible by tile m={m}")
    if K % k != 0:
        raise ValueError(f"K={K} not divisible by tile k={k}")
    if N % n != 0:
        raise ValueError(f"N={N} not divisible by tile n={n}")
    if rows not in VALID_ROWS:
        raise ValueError(f"rows must be in {list(VALID_ROWS.keys())}")

    group_count = (M // m) * (N // n) // (4 * 8 * rows)
    if group_count < 1:
        raise ValueError(
            f"(M/m)*(N/n)={(M//m)*(N//n)} too small for (4*8*{rows}) ping-pong scheme"
        )

    mlir_name = f"aie_{M}x{K}x{N}_{m}x{k}x{n}.mlir"
    xclbin_name = f"final_{M}x{K}x{N}_{m}x{k}x{n}.xclbin"
    insts_name = f"insts_{M}x{K}x{N}_{m}x{k}x{n}.txt"
    mlir_path, xclbin_path, insts_path = (
        BUILD_DIR / mlir_name, BUILD_DIR / xclbin_name, BUILD_DIR / insts_name
    )
    if not force and xclbin_path.exists() and insts_path.exists():
        return (xclbin_path, insts_path)

    os.makedirs(BUILD_DIR, exist_ok=True)
    py_src = SRCDIR / f"{VALID_ROWS[rows]}.py"

    # Generate MLIR
    r = subprocess.run(
        [PYTHON, str(py_src), "-m", str(m), "-k", str(k), "-n", str(n),
         "-M", str(M), "-K", str(K), "-N", str(N)],
        capture_output=True, text=True, cwd=SRCDIR,
    )
    if r.returncode != 0:
        raise RuntimeError(f"MLIR gen failed: {r.stderr[:1000]}")
    with open(mlir_path, "w") as f:
        f.write(r.stdout)
    print(f"  [OK] MLIR: {mlir_name}")

    # Compile via aiecc
    env = os.environ.copy()
    env["PATH"] = f"/home/bcloud/mlir-aie/build/bin:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"/home/bcloud/torch2aie/toolchain/mlir_aie/python:{env.get('PYTHONPATH', '')}"
    r = subprocess.run(
        [AIECC, "--aietools=/home/bcloud/torch2aie/toolchain/aietools",
         "--alloc-scheme=basic-sequential", "--aie-generate-xclbin",
         "--no-compile-host", f"--xclbin-name={xclbin_name}",
         "--unified", "--dynamic-objFifos", "--aie-generate-npu-insts",
         f"--npu-insts-name={insts_name}", str(mlir_path)],
        capture_output=True, text=True, cwd=BUILD_DIR, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"aiecc failed:\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}")
    if not xclbin_path.exists() or not insts_path.exists():
        raise RuntimeError("xclbin/insts not produced")
    print(f"  [OK] xclbin ({xclbin_path.stat().st_size} B), insts ({insts_path.stat().st_size} B)")
    return (xclbin_path, insts_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--rows", type=int, default=1, choices=[1, 2])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    xclbin, insts = generate_xclbin(args.M, args.K, args.N, args.m, args.k, args.n, args.rows, args.force)
    print(f"xclbin: {xclbin}\ninsts:  {insts}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run test to verify it passes**

```bash
cd /home/bcloud/npu-sandbox/npu-infer && python -m pytest tests/test_compile_gemm.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 4: Run compile on known-working 1024×1024×2048**

```bash
python tools/compile_gemm.py --M 1024 --K 1024 --N 2048 --rows 1
```
Expected: xclbin and insts produced in <30 seconds.

- [ ] **Step 5: Run compile on query dimension (128×1024×4096, 2-row)**

```bash
python tools/compile_gemm.py --M 128 --K 1024 --N 4096 --rows 2
```
Expected: xclbin and insts produced.

- [ ] **Step 6: Commit**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
git add tools/compile_gemm.py tests/test_compile_gemm.py
git commit -m "feat: add parameterized GEMM xclbin compiler wrapper"
```

---

### Task 2: CPU BF16 GEMM Reference — `cpu_ref_gemm.py`

**Files:**
- Create: `tests/cpu_ref_gemm.py` at `/home/bcloud/npu-sandbox/npu-infer/tests/cpu_ref_gemm.py`

**Interfaces:**
- `gemm_reference(A_uint16, B_uint16)` → C_f32 (M×N float32): BF16 × BF16 multiply in FP64→FP32
- `generate_test_vectors(M, K, N)` → (A_uint16, B_uint16, C_f32): A=ramp per row, B=all-ones
- `relative_error(computed, reference)` → float: max relative error

- [ ] **Step 1: Write failing test**

```python
# tests/test_cpu_ref_gemm.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from cpu_ref_gemm import generate_test_vectors, gemm_reference, relative_error, f32_to_bf16, bf16_to_float
import numpy as np

def test_ramp_gemm():
    M, K, N = 4, 8, 4
    A, B, C_ref = generate_test_vectors(M, K, N)
    # A = row-based ramp (row0=1, row1=2, ...), B = all-1
    # C[m,n] = sum over K of row_m+1 = K * (row_m+1)
    assert abs(C_ref[0, 0] - K * 1.0) < 0.1
    assert abs(C_ref[1, 0] - K * 2.0) < 0.1
    assert abs(C_ref[2, 0] - K * 3.0) < 0.1

def test_bf16_roundtrip():
    orig = 1.0
    bf16_val = f32_to_bf16(orig)
    back = bf16_to_float(bf16_val)
    assert abs(back - orig) < 0.001
```

Run:
```bash
cd /home/bcloud/npu-sandbox/npu-infer && python -m pytest tests/test_cpu_ref_gemm.py -v
```
Expected: FAIL

- [ ] **Step 2: Implement `cpu_ref_gemm.py`**

```python
#!/usr/bin/env python3
"""CPU BF16/BFP16 GEMM reference for NPU correctness verification."""

import numpy as np

def f32_to_bf16(v):
    bits = np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    return np.uint16((bits + rounding_bias) >> 16)

def bf16_to_float(v):
    return float(np.frombuffer(
        np.array([v], dtype=np.uint16).tobytes() + b'\x00\x00', dtype=np.float32
    )[0])

def gemm_reference(A_uint16, B_uint16):
    """C = A × B. A,B are uint16 arrays (BF16 storage). Returns float32 C."""
    M, K = A_uint16.shape
    K2, N = B_uint16.shape
    assert K == K2
    A_f64 = np.array([[bf16_to_float(A_uint16[m, k]) for k in range(K)] for m in range(M)], dtype=np.float64)
    B_f64 = np.array([[bf16_to_float(B_uint16[k, n]) for k in range(K)] for n in range(N)], dtype=np.float64).T
    return (A_f64 @ B_f64).astype(np.float32)

def generate_test_vectors(M, K, N):
    """A=ramp-per-row (row m = m+1), B=all-1. Returns (A_bf16, B_bf16, C_ref_f32)."""
    A = np.array([[float(m + 1)] * K for m in range(M)], dtype=np.float32)
    B = np.ones((K, N), dtype=np.float32)
    A_bf16 = np.array([[f32_to_bf16(A[m, k]) for k in range(K)] for m in range(M)], dtype=np.uint16)
    B_bf16 = np.array([[f32_to_bf16(B[k, n]) for n in range(N)] for k in range(K)], dtype=np.uint16)
    C_ref = gemm_reference(A_bf16, B_bf16)
    return A_bf16, B_bf16, C_ref

def relative_error(computed, reference):
    diff = np.abs(computed.astype(np.float64) - reference.astype(np.float64))
    ref_mag = np.abs(reference.astype(np.float64))
    return float(np.max(diff / np.maximum(ref_mag, 1e-30)))
```

- [ ] **Step 3: Verify tests pass**

```bash
cd /home/bcloud/npu-sandbox/npu-infer && python -m pytest tests/test_cpu_ref_gemm.py -v
```
Expected: PASS (2)

- [ ] **Step 4: Generate reference for 128x1024x4096 (a real QKV projection dim)**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
python -c "
from tests.cpu_ref_gemm import generate_test_vectors
import numpy as np
A, B, C = generate_test_vectors(128, 1024, 4096)
np.savez_compressed('/tmp/ref_128x1024x4096.npz', A=A, B=B, C_ref=C)
print(f'C[0,0]={C[0,0]:.1f}  (expect 1024.0)')
print(f'C[1,0]={C[1,0]:.1f}  (expect 2048.0)')
"
```
Expected: values match expectations within BF16 precision.

- [ ] **Step 5: Commit**

```bash
git add tests/cpu_ref_gemm.py tests/test_cpu_ref_gemm.py
git commit -m "feat: add CPU BF16 GEMM reference implementation"
```

---

### Task 3: Python NPU Runner — `verify_gemm.py`

**Files:**
- Create: `tools/verify_gemm.py` at `/home/bcloud/npu-sandbox/npu-infer/tools/verify_gemm.py`
- Create: C source embedded in the Python module (compiled at runtime) — borrows from `npu_raw_submit.c` and `test_correctness.cpp`

**Interfaces:**
- `run_npu_gemm(xclbin_path, insts_path, A_bf16, B_bf16, M, K, N)` → `C_npu_f32` (M×N float32)
  - Compiles C runner if needed
  - Pipes A and B as raw binary to C subprocess stdin
  - Reads C output from C subprocess stdout
  - Uses `gemm_atb_layout` functions for data shuffle/unshuffle
  - Returns error count and GFLOPS measurement
- `verify(M, K, N, rows=1)` → dict with pass/fail, errors, GFLOPS, TTFT

- [ ] **Step 1: Write failing test**

```python
# tests/test_verify_gemm.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.verify_gemm import verify

def test_verify_128x1024x2048():
    # Small test that should pass quickly
    result = verify(M=128, K=1024, N=2048, rows=1)
    assert result["passed"] == True
    assert result["errors"] < 10
    assert result["gflops"] > 0
```

- [ ] **Step 2: Implement `verify_gemm.py`**

The C runner is embedded as a string template (same pattern as `npu_raw_submit.c` but pipelined via stdin/stdout). Key flow:

1. Compile C runner to temp binary (once, cached by xclbin hash)
2. Run `compile_gemm.py` to get xclbin + insts (or use cached)
3. Generate test vectors
4. Apply IRON layout transformations:
   - A: `layout_A_L1_2x1_8x8block(A_f32, M, K, 128, 64)` → per-core layout
   - B: `layout_transpose_L1_1x2_8x8block(B_f32, K, N, 64, 128)` → transposed + blocked
   - BFP16 encode: `floatToBfp16(8, K*N, B_shuffled, 0)` → block-scaled format
5. Pipe A_bf16 + B_bf16 to C runner
6. Read back C_bf16 from stdout
7. Apply inverse layout: `layout_inverse_C_L1_2x2_8x8block(C_unshuffled, M, N, 256, 128)`
8. Compare vs CPU reference: `gemm_reference(A_orig, B_orig)`

```python
import os, subprocess, tempfile, json, time
from pathlib import Path

# Will reference torch2aie layout helpers via PYTHONPATH injection
try:
    from gemm_atb_layout import (
        layout_A_L1_2x1_8x8block,
        layout_transpose_L1_1x2_8x8block,
        layout_inverse_C_L1_2x2_8x8block,
    )
except ImportError:
    import sys; sys.path.append("/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering")
    from gemm_atb_layout import (
        layout_A_L1_2x1_8x8block,
        layout_transpose_L1_1x2_8x8block,
        layout_inverse_C_L1_2x2_8x8block,
    )

def floatToBfp16(eb, n, data_f32, shift):
    """Encode float32 as BFP16 ebs8 (block-scaled, 8 elements per exponent)."""
    # Match the C++ implementation from gemm_atb_layout.h
    # For unit test where B = all 1.0, BFP16 encoding is trivial
    ...

def run_npu_gemm(xclbin_path, insts_path, A_bf16, B_bf16, M, K, N):
    """Run GEMM on NPU, return C as float32 array."""
    ...
```

- [ ] **Step 3: Integration test — run NPU on 128×1024×2048**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
python tools/verify_gemm.py --M 128 --K 1024 --N 2048 --rows 1
```

Expected output:
```
Compiling xclbin for 128x1024x2048...
  [OK] MLIR generated
  [OK] xclbin (51850 B), insts (376 B)
Running NPU GEMM...
  NPU init: 15000 us
  NPU exec: 500 us
  GFLOPS: ...
Unshuffling C...
Comparing vs CPU reference...
  Errors: 0 / 262144
  PASS
```

- [ ] **Step 4: Test at 128×1024×4096 (real QKV dimension, 2-row)**

```bash
python tools/verify_gemm.py --M 128 --K 1024 --N 4096 --rows 2
```
Expected: PASS (0 errors).

- [ ] **Step 5: Commit**

```bash
git add tools/verify_gemm.py tests/test_verify_gemm.py
git commit -m "feat: add NPU GEMM verification runner with automated correctness comparison"
```

---

### Task 4: Parameterized Sweep — `bench_gemm.py`

**Files:**
- Create: `tools/bench_gemm.py` at `/home/bcloud/npu-sandbox/npu-infer/tools/bench_gemm.py`

**Interfaces:**
- CLI: `--dims` list of MxKxN strings, `--rows 1|2`, `--iterations N`
- Produces: JSON report with per-dimension correctness + GFLOPS

**Step-by-step:**

- [ ] **Step 1: Implement `bench_gemm.py`**

```python
#!/usr/bin/env python3
"""
Benchmark parameterized GEMM kernels across multiple dimensions.
Reports correctness and performance for each.

Usage:
  python bench_gemm.py --dims 128x1024x2048 1024x1024x2048 --rows 1 --iterations 3
  python bench_gemm.py --all-qwen3-06b
"""
import argparse, json, time, sys
from pathlib import Path

# Qwen3-0.6B dimensions
QWEN3_06B_DIMS = {
    "qkv": {"M": 128, "K": 1024, "N": 3072, "rows": 2},
    "o":   {"M": 128, "K": 3072, "N": 1024, "rows": 2},
    "gu":  {"M": 128, "K": 1024, "N": 8192, "rows": 2},
    "down":{"M": 128, "K": 8192, "N": 1024, "rows": 2},
    "ff1": {"M": 128, "K": 1024, "N": 8192, "rows": 2},
    "ff2": {"M": 128, "K": 8192, "N": 1024, "rows": 2},
}

def bench_one(M, K, N, rows=1, iterations=3):
    from tools.verify_gemm import verify
    results = []
    for i in range(iterations):
        r = verify(M, K, N, rows)
        results.append(r)
    avg_gflops = sum(r["gflops"] for r in results) / len(results)
    avg_errors = sum(r["errors"] for r in results) / len(results)
    return {
        "M": M, "K": K, "N": N, "rows": rows,
        "avg_gflops": round(avg_gflops, 2),
        "avg_errors": round(avg_errors),
        "passed": all(r["passed"] for r in results),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", help="MxKxN dims", default=[])
    parser.add_argument("--rows", type=int, default=1, choices=[1, 2])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--all-qwen3-06b", action="store_true")
    parser.add_argument("--report", help="Save JSON report to path")
    args = parser.parse_args()

    dims = []
    if args.all_qwen3_06b:
        dims = [(v["M"], v["K"], v["N"], v["rows"]) for v in QWEN3_06B_DIMS.values()]
    for d in args.dims:
        parts = d.split("x")
        dims.append((int(parts[0]), int(parts[1]), int(parts[2]), args.rows))

    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": []}
    for M, K, N, rows in dims:
        print(f"\n{'='*60}")
        print(f"GEMM {M}x{K}x{N} (rows={rows})")
        try:
            r = bench_one(M, K, N, rows, args.iterations)
            report["results"].append(r)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {status}: {r['avg_gflops']} GFLOPS, {r['avg_errors']} errors")
        except Exception as e:
            print(f"  FAIL: {e}")
            report["results"].append({"M": M, "K": K, "N": N, "error": str(e)})

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {args.report}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run sweep on Qwen3-0.6B dimensions**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
python tools/bench_gemm.py --all-qwen3-06b --iterations 3 --report /tmp/gemm_bench_06b.json
```

- [ ] **Step 3: Review results**

```bash
python -m json.tool /tmp/gemm_bench_06b.json
```

- [ ] **Step 4: Commit**

```bash
git add tools/bench_gemm.py
git commit -m "feat: add parameterized GEMM benchmark sweep tool"
```

---

### Task 5: Automated Test Harness in CI Style

**Files:**
- Create: `tests/test_full_pipeline.py`

**Step-by-step:**

- [ ] **Step 1: Write comprehensive pipeline test**

```python
# tests/test_full_pipeline.py
"""End-to-end test: compile → NPU submit → verify for multiple dims."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.verify_gemm import verify

def test_minimal_128x1024x2048():
    r = verify(128, 1024, 2048, rows=1)
    assert r["passed"], f"Minimal test failed: {r}"

def test_square_512x1024x512():
    r = verify(512, 1024, 512, rows=1)
    assert r["passed"], f"Square test failed: {r}"

def test_wide_128x1024x4096():
    r = verify(128, 1024, 4096, rows=2)
    assert r["passed"], f"Wide test failed: {r}"

def test_tall_512x1024x128():
    r = verify(512, 1024, 128, rows=1)
    assert r["passed"], f"Tall test failed: {r}"
```

- [ ] **Step 2: Run full pipeline test**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
python -m pytest tests/test_full_pipeline.py -v --timeout 120
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_full_pipeline.py
git commit -m "test: add end-to-end GEMM pipeline test suite"
```

---

## Verification

After all tasks are complete:

```bash
# 1. Unit tests pass
cd /home/bcloud/npu-sandbox/npu-infer
python -m pytest tests/ -v

# 2. Can compile any valid dimension
python tools/compile_gemm.py --M 256 --K 512 --N 1024 --rows 1

# 3. Can verify a known-good dim end-to-end
python tools/verify_gemm.py --M 128 --K 1024 --N 2048 --rows 1

# 4. Can sweep Qwen3-0.6B dims
python tools/bench_gemm.py --all-qwen3-06b --iterations 1
```

Success criteria:
- All tests pass
- Zero errors for ramp-per-row test vectors (A = row_id+1, B = all-1)
- GFLOPS reported within expected range for each dimension
- Clean commit history with one commit per task
