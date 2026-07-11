#pragma once
// Auto-tuner for ternary prefill GEMM kernels on gfx1151.
// One-shot benchmark at model load time, picks the fastest variant
// for the model's specific FFN shapes, then dispatches all subsequent
// calls to that variant.
//
// Variants indexed by tile geometry:
//   0  = 4i (1 wave, 64x64, vectorized A loads) — current default
//   1  = 4h (1 wave, 64x64, A+B cached in registers)
//   2  = 4k (2 waves, 64x32, LDS double-buffered)
//   3  = 4f (4 waves, 64x64, A+B in LDS)
//   4  = FP16-B (64x64, pre-decoded FP16 B weights)
//   5  = 4c (1 wave, 32x64)
//   6  = 4g (1 wave, 64x64, 4×4 accumulators)

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RCPP_PREFILL_VARIANT_4I       = 0,   // 64x64 vectorized A loads (FP16)
    RCPP_PREFILL_VARIANT_4H       = 1,   // 64x64 cached A+B (FP16)
    RCPP_PREFILL_VARIANT_4K       = 2,   // LDS ping-pong 64x32
    RCPP_PREFILL_VARIANT_4F       = 3,   // 2x2 wave 64x64 LDS
    RCPP_PREFILL_VARIANT_FP16B    = 4,   // pre-decoded FP16 B
    RCPP_PREFILL_VARIANT_4C       = 5,   // 32x64
    RCPP_PREFILL_VARIANT_4G       = 6,   // 64x64 4x4 acc
    RCPP_PREFILL_VARIANT_I8_T     = 7,   // INT8 WMMA tiled (I8 A + tiled I8 B)
    RCPP_PREFILL_VARIANT_I8_APRE  = 8,   // INT8 WMMA tiled + pre-quantized A (FASTEST: ~42 TFLOPS)
    RCPP_PREFILL_VARIANT_COUNT    = 9
} rcpp_prefill_variant_t;

// Tune prefill for a specific (M,N,K) shape. Returns the fastest variant index.
// If warmup_iters > 0, runs that many warmup iterations before timing.
// If timed_iters > 0, runs that many timed iterations (median taken).
// The variant is stored per-shape in an internal cache.
int rcpp_prefill_tune(
    const void* A_dev,
    const void* B_packed_dev,
    void* C_dev,
    int M, int N, int K,
    int warmup_iters,
    int timed_iters,
    void* stream);

// Run prefill with the best-known variant for this shape.
// Falls back to 4i if tuning hasn't been run yet.
void rcpp_prefill_dispatch(
    const void* A_dev,
    const void* B_packed_dev,
    void* C_dev,
    int M, int N, int K,
    void* stream);

// FP16-B materialization: decode pk_i4 packed weights to col-major FP16.
// Run once per weight tensor. The FP16 buffer can then be used with
// RCPP_PREFILL_VARIANT_FP16B for maximum throughput.
void rcpp_pk_i4_to_fp16(
    const void* B_packed_dev,
    void* B_fp16_dev,
    int K, int N,
    void* stream);

// Run prefill with pre-decoded FP16 B weights.
void rcpp_prefill_fp16b(
    const void* A_dev,
    const void* B_fp16_dev,
    void* C_dev,
    int M, int N, int K,
    void* stream);

#ifdef __cplusplus
}
#endif
