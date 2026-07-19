#pragma once
// rocm-cpp — Bonsai Q1_0_g128 / TQ2_0_g128 ternary GEMV C API (fp16 path).
//
// Spec: docs/wiki/Bonsai-Kernel-Spec.md. These are the PrismML Bonsai
// on-disk block layouts (g128 variants), distinct from ggml's canonical
// TQ2_0 (g256). The kernel consumes the AoS (block-interleaved) on-disk
// layout directly — no host-side SoA repack. Scales are embedded per-block
// as FP16.
//
// Both kernels target gfx1151 (RDNA 3.5, wave32). Each workgroup computes
// one output row; K_in must be a multiple of 128 (one g128 block).
//
// Status 2026-04-20: first cut of the HIP kernels. Correctness gated by
// tests/test_bonsai_gemv.cpp (differential against scalar reference).
//
// No hipBLAS, no CK, no Python.

#ifndef ROCM_CPP_BONSAI_H
#define ROCM_CPP_BONSAI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Q1_0_g128 block layout (18 bytes) ────────────────────────────────────────
//   offset  bytes  field
//   0x00    2      d : FP16       group scale
//   0x02    16     qs[16]         128 sign bits, LSB-first
// Weight reconstruction: w[i] = (qs[i/8] >> (i%8)) & 1 ? +d : -d.

// ── TQ2_0_g128 block layout (34 bytes) ───────────────────────────────────────
//   offset  bytes  field
//   0x00    32     qs[32]         128 × 2-bit codes, 4/byte LSB-first
//   0x20    2      d : FP16       group scale
// Code map: 0b00 → -d, 0b01 → 0, 0b10 → +d, 0b11 → 0 (reserved).

// Q1_0_g128 1-bit GEMV. FP32 internal accumulation; FP16 output.
//
//   packed_weights : const uint8_t*, device
//                    Row-major, row stride = (K_in / 128) * 18 bytes.
//   act_fp16       : const uint16_t*, device, length K_in (__half bit pattern).
//   out_fp16       : uint16_t*, device, length N_out (__half bit pattern).
//   N_out, K_in    : row count and input dimension. K_in must be a positive
//                    multiple of 128 (group size); otherwise launcher
//                    early-returns as a no-op.
//   stream         : hipStream_t as void* (keeps the header HIP-free).
//
// Stream-based sync only; kernel-launch failures surface at the caller's next
// synchronization boundary (via hipGetLastError).
void bonsai_q1_gemv_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

// TQ2_0_g128 ternary GEMV. Same preconditions as bonsai_q1_gemv_launch;
// weight row stride is (K_in / 128) * 34 bytes.
void bonsai_tq2_gemv_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

// Scalar reference launchers — one HIP thread per output row. Same fp32
// accumulation order as the fast kernels (ascending over blocks / weights,
// FMA into a single fp32 register), same fp16 clamp+cast epilogue. Exist
// for differential tests; do NOT benchmark.
void bonsai_q1_gemv_scalar_ref_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

void bonsai_tq2_gemv_scalar_ref_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

// ── SoA-optimized Q1_0 GEMV (4 rows/CTA) ──────────────────────────────────
// Same packed size as AoS but with SoA layout per row:
//   [d values for all blocks][qs values for all blocks]
// Convert with bonsai_q1_convert_aos_to_soa().
void bonsai_q1_gemv_soa_launch(
    const uint8_t*  packed_weights_soa,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

// Convert AoS Q1_0 weights to SoA layout in place on device.
// Allocates new device memory in *packed_soa (caller must hipFree).
void bonsai_q1_convert_aos_to_soa(
    const uint8_t* packed_aos,
    uint8_t**      packed_soa,
    int            N_out,
    int            K_in);

// ── 1024-weight block variants (higher occupancy, better LDS reuse) ──────────
// These are the same TQ2_0/Q1_0 kernels but with 1024-weight blocks instead of
// the standard 128-weight blocks. Fewer blocks per row = fewer global loads.

// TQ2_0_1024-block ternary GEMV. Same preconditions; weight row stride is
// (K_in / 1024) * 34 * 8 bytes (8×128-weight blocks per 1024-weight block).
void bonsai_tq2_1024_gemv_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

// Convert TQ2_0 AoS weights (128-weight blocks) to 1024-weight blocks on device.
// Allocates new device memory in *packed_1024 (caller must hipFree).
void bonsai_tq2_convert_to_1024(
    const uint8_t* packed_aos,
    uint8_t**      packed_1024,
    int            N_out,
    int            K_in);

// Q1_0_1024-block 1-bit GEMV. Same preconditions as bonsai_q1_gemv_launch;
// weight row stride is (K_in / 1024) * 18 * 8 bytes.
void bonsai_q1_1024_gemv_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

// Convert Q1_0 AoS weights (128-weight blocks) to 1024-weight blocks on device.
// Allocates new device memory in *packed_1024 (caller must hipFree).
void bonsai_q1_convert_to_1024(
    const uint8_t* packed_aos,
    uint8_t**      packed_1024,
    int            N_out,
    int            K_in);

#ifdef __cplusplus
}
#endif
#endif  // ROCM_CPP_BONSAI_H
