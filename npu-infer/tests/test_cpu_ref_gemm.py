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
