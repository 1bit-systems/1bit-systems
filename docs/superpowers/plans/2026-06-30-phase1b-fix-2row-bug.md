# Phase 1b: Fix 2-Row GEMM Design Bug

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the n2_core_placed.py bug causing 100% correctness errors on 2-row GEMM designs, then verify constant-data and ramp-data correctness end-to-end.

**Architecture:** The 2-row design's core loop incorrectly uses `range(n_aie_rows)` (=2) instead of `range(4)` for the inner A tile acquisition loop. Each C tile (128×128) needs 4 A micro-tiles (32×64) to accumulate across the full row dimension, but the buggy code only acquires 2 — losing half the A contributions.

**Root cause:** `n2_core_placed.py` line 239: `for i in range(n_aie_rows):` should be `for i in range(4):`. All working designs (n1_core_bf16.py, n32_core_placed.py) use `range(4)`. The 2-row design was incorrectly parameterized.

**Tech Stack:** MLIR-AIE IRON Python dialect, XRT 2.21.75

---
## Global Constraints

- All MLIR compilation uses `/home/bcloud/torch2aie/.venv/bin/python`
- Tile sizes: m=128, k=64, n=128 (baked into microkernel)
- NPU device: `/dev/accel/accel0` (8 columns, Strix Halo)
- M divisible by m=128, K by k=64, N by n=128
- `(M/m)*(N/n)` must be multiple of `(4 * n_aie_cols * n_aie_rows)` = 32 (1-row) or 64 (2-row)
- All xclbin outputs to `config1/build/`

---

### Task 1: Fix Bug — n2_core_placed.py line 239

**Files:**
- Modify: `/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n2_core_placed.py:239`

**Fix:** Change `range(n_aie_rows)` to `range(4)`:

```python
                                for i in range(4):
                                    elem_in_a = A_l2l1_fifos[row].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    matmul(elem_in_a, elem_in_b, elem_out)
                                    A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
```

- [ ] **Step 1: Apply fix**

Edit `n2_core_placed.py` line 239: `for i in range(n_aie_rows):` → `for i in range(4):`

- [ ] **Step 2: Cross-check n2 vs n1 for any other differences**

```bash
cd /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1
diff -u <(sed 's/n_aie_rows/4/g; s/2 \* row/row/g; s/row \* 2/row/g; s/tiles\[2:\]/tiles[2:3]/g' n1_core_bf16.py | sed 's/range(1)/range(4)/g') n2_core_placed.py | head -100
```

(The normalization is rough, but check for structural differences beyond the range bug.)

- [ ] **Step 3: Rebuild existing 128×1024×4096 2-row xclbin**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
# Force recompile the 2-row xclbin
python tools/compile_gemm.py --M 128 --K 1024 --N 4096 --rows 2 --force
```

Expected: xclbin compiles successfully.

- [ ] **Step 4: Verify constant-data correctness on 2-row design**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
python tools/verify_gemm.py --M 128 --K 1024 --N 4096 --rows 2 --A_val 1.0 --B_val 3.0
```

Expected: `Errors: 0`, `PASS`

- [ ] **Step 5: Write 2-row constant-data test**

Add to `/home/bcloud/npu-sandbox/npu-infer/tests/test_full_pipeline.py`:

```python
def test_two_row_128x1024x4096():
    """2-row design with constant data — verifies n2_core_placed fix."""
    from tools.verify_gemm import verify
    r = verify(128, 1024, 4096, rows=2, A_val=1.0, B_val=3.0)
    assert r["passed"], f"2-row test failed: {r}"
    assert r["errors"] == 0, f"Expected 0 errors, got {r['errors']}"
```

- [ ] **Step 6: Run all tests**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
python -m pytest tests/ -v
```

Expected: 19+ passes (new 2-row test passes).

- [ ] **Step 7: Commit**

```bash
git add /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n2_core_placed.py
git add tests/test_full_pipeline.py
git commit -m "fix: n2_core_placed.py range(n_aie_rows)->range(4) for A tile acquisition loop"
```

---

### Task 2: Verify 2-row with ramp data + document status

After fixing the core loop, ramp data should still have the layout mismatch issue (since `gemm_atb_layout.h` may not match MLIR expectations). But the constant path should now work.

- [ ] **Step 1: Test ramp data with 2-row fix**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
# Run verify with ramp (default) on 2-row
python tools/verify_gemm.py --M 128 --K 1024 --N 4096 --rows 2
```

Expected: May still show errors due to host-side layout mismatch. Document result.

- [ ] **Step 2: End-to-end sweep**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
python tools/bench_gemm.py --dims 128x1024x2048 128x1024x4096 --rows 2 --iterations 1 --constant --report /tmp/phase1b_fixed.json
```

- [ ] **Step 3: Document findings**

Update `tests/test_verify_gemm.py` docstring and `tools/verify_gemm.py` block comment with the Phase 1b fix status.

- [ ] **Step 4: Commit**

```bash
git add tools/verify_gemm.py tests/test_verify_gemm.py
git commit -m "docs: update layout documentation after 2-row fix"
```

---

### Verification

```bash
# Full test suite
cd /home/bcloud/npu-sandbox/npu-infer
python -m pytest tests/ -v

# 2-row constant-data correctness
python tools/verify_gemm.py --M 128 --K 1024 --N 4096 --rows 2 --A_val 1.0 --B_val 3.0

# 1-row constant-data still works (regression)
python tools/verify_gemm.py --M 128 --K 1024 --N 2048 --rows 1 --A_val 1.0 --B_val 3.0
```

Success criteria:
- All tests pass
- `verify --M 128 --K 1024 --N 4096 --rows 2 --A_val 1.0 --B_val 3.0` → 0 errors
- Both 1-row and 2-row constant paths verified correct
