#!/usr/bin/env python3
"""CPU BF16/BFP16 GEMM reference for NPU correctness verification."""

import numpy as np


def f32_to_bf16(v):
    bits = np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    return np.uint16((bits + rounding_bias) >> 16)


def bf16_to_float(v):
    return float(np.frombuffer(
        b'\x00\x00' + np.array([v], dtype=np.uint16).tobytes(), dtype=np.float32
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
