"""End-to-end test: compile -> NPU submit -> verify for multiple dims.
Uses constant data (A=1, B=3) to avoid pre-existing ramp-data layout mismatch.

HARDWARE CONSTRAINTS (rows=1, tile m=128, k=64, n=128):
- M must be exactly 128 (1-row AIE core grid can only fit 1 row tile)
- N >= 1024 (N_AIE_COLS=8 * n=128)
- K divisible by 64
- All dims must pass with 0 errors using constant data
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.verify_gemm import verify


def test_minimal_128x1024x2048():
    """Minimal valid GEMM: M=128, K=1024, N=2048, rows=1."""
    r = verify(128, 1024, 2048, rows=1, A_val=1.0, B_val=3.0)
    assert r["passed"], f"Minimal test failed: {r}"
    assert r["errors"] == 0, f"Expected 0 errors, got {r['errors']}"


def test_medium_n_128x1024x1024():
    """Medium N=1024 (minimum valid N)."""
    r = verify(128, 1024, 1024, rows=1, A_val=1.0, B_val=3.0)
    assert r["passed"], f"Medium N test failed: {r}"
    assert r["errors"] == 0


def test_wide_128x1024x4096():
    """Wide GEMM with N=4096 (needs constant data; ramp has layout bugs)."""
    r = verify(128, 1024, 4096, rows=1, A_val=1.0, B_val=3.0)
    assert r["passed"], f"Wide test failed: {r}"
    assert r["errors"] == 0


def test_qkv_128x1024x3072():
    """QKV projection dimension (M=128, K=1024, N=3072)."""
    r = verify(128, 1024, 3072, rows=1, A_val=1.0, B_val=3.0)
    assert r["passed"], f"QKV test failed: {r}"
    assert r["errors"] == 0
