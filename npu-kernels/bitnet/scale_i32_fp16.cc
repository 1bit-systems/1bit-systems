//===- scale_i32_fp16.cc ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, 1bit.systems / bong-water-water-bong.
//
// Stage 3 of the BitNet-1.58 NPU pipeline on AIE2P (npu2 / Strix Halo).
// Applies per-row fp16 (bfloat16 on AIE2P) scale to the int32 matmul output
// and writes a bfloat16 output tile.
//
// BitNet-1.58 scale semantics:
//   We dequant ternary activations+weights via a single per-row scalar in
//   bfloat16 (matches halo-ai .h1b and upstream BitNet checkpoint layout).
//   One scale per output row (M dim). Columns (N dim) share the scale.
//
// Kernel contract:
//   Input:
//     int32_t          *in_tile     [M * N]      matmul accumulator tile
//     bfloat16         *scale_row   [M]          one scalar per row
//   Output:
//     bfloat16         *out_tile    [M * N]      scaled, cast to bfloat16
//   Layout: row-major; N % 16 == 0 (matches AIE2P 16-wide fp32/bf16 lanes).
//
// Why bfloat16 and not "fp16":
//   AIE2P native floating types are bfloat16 and fp32. There is no IEEE fp16
//   MAC on this core. halo-ai .h1b "fp16 scales" are stored/transported as
//   bfloat16 on the NPU side; host-side fp16<->bf16 conversion lives in the
//   loader, not here. Task spec said "fp16 scale" — we use bfloat16 to match
//   the hardware and the rest of the upstream aie2p kernels (see rms_norm).

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// 16-wide lanes for bfloat16 / fp32 / accfloat on AIE2P. Matches mm.cc's
// output tile granularity (n=8 MAC lanes, doubled in the 2x2 expansion).
static constexpr unsigned kFpLanes = 16;

// --------------------------------------------------------------------------
// Vectorized int32 -> float -> (bf16 * scale) -> bfloat16.
//
// Strategy:
//   For each row r in [0, M):
//     s       = broadcast bfloat16 scale_row[r] across 16 lanes
//     s_fp32  = to_float(s)                   // bf16 -> fp32 broadcast
//     for col block c in [0, N) step 16:
//       i32_v = load int32, 16 lanes
//       f_v   = to_float(i32_v)               // int32 -> fp32
//       p_v   = f_v * s_fp32                  // fp32 multiply
//       bf_v  = to_vector<bfloat16>(p_v)      // fp32 -> bf16 (RN)
//       store bf_v
// --------------------------------------------------------------------------
template <int M, int N>
static inline void scale_i32_to_bf16_vectorized(
    const int32_t *__restrict in_tile,
    const bfloat16 *__restrict scale_row,
    bfloat16 *__restrict out_tile) {
  event0();
  static_assert(N % kFpLanes == 0,
                "N must be a multiple of 16 for AIE2P fp lane width");
  constexpr int cols_chunks = N / kFpLanes;

  for (int r = 0; r < M; r++) {
    // Load scalar scale; broadcast to 16 fp32 lanes.
    // bfloat16 -> fp32 broadcast: broadcast then cast. We go through fp32
    // because the int32->float path lands in fp32 and the mul is cheaper
    // in fp32 than in bf16-emulated.
    const bfloat16 s_bf = scale_row[r];
    const float s_f32 = static_cast<float>(s_bf);
    aie::vector<float, kFpLanes> s_v =
        aie::broadcast<float, kFpLanes>(s_f32);

    const int32_t *__restrict row_in = in_tile + r * N;
    bfloat16 *__restrict row_out = out_tile + r * N;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int c = 0; c < cols_chunks; c++) {
      aie::vector<int32_t, kFpLanes> i32_v =
          aie::load_v<kFpLanes>(row_in + c * kFpLanes);

      // int32 -> fp32. aie::to_float lives in aie_api and is the sanctioned
      // path; aie::accum<accfloat> is an equivalent route if to_float is not
      // available for int32 on the on-box header (flag in README).
      aie::vector<float, kFpLanes> f_v =
          aie::to_float<float>(i32_v);

      // fp32 multiply by broadcast scale.
      aie::accum<accfloat, kFpLanes> acc = aie::mul(f_v, s_v);
      aie::vector<float, kFpLanes> prod_v =
          acc.template to_vector<float>();

      // fp32 -> bfloat16 (round-to-nearest-even). accfloat::to_vector<bf16>
      // is the single-instruction rounding path on AIE2P.
      aie::accum<accfloat, kFpLanes> acc_bf;
      acc_bf.from_vector(prod_v);
      aie::vector<bfloat16, kFpLanes> bf_v =
          acc_bf.template to_vector<bfloat16>();

      aie::store_v(row_out + c * kFpLanes, bf_v);
    }
  }
  event1();
}

// Scalar reference. Used by the host for numerical cross-checks. Semantics
// must match the vector path bit-for-bit modulo fp32->bf16 rounding mode.
template <int M, int N>
static inline void scale_i32_to_bf16_scalar(
    const int32_t *__restrict in_tile,
    const bfloat16 *__restrict scale_row,
    bfloat16 *__restrict out_tile) {
  event0();
  for (int r = 0; r < M; r++) {
    const float s = static_cast<float>(scale_row[r]);
    for (int c = 0; c < N; c++) {
      const float prod = static_cast<float>(in_tile[r * N + c]) * s;
      out_tile[r * N + c] = static_cast<bfloat16>(prod);
    }
  }
  event1();
}

extern "C" {

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_N
#define DIM_N 16
#endif

// Tile-shaped entry points — one call per (m-tile x n-tile) of the matmul
// output. Scale vector is m-tile-sized; the caller slices it to match the
// M-tile row range.
void scale_i32_bf16(int32_t *__restrict in_tile,
                    bfloat16 *__restrict scale_row,
                    bfloat16 *__restrict out_tile) {
  scale_i32_to_bf16_vectorized<DIM_M, DIM_N>(in_tile, scale_row, out_tile);
}

void scale_scalar_i32_bf16(int32_t *__restrict in_tile,
                           bfloat16 *__restrict scale_row,
                           bfloat16 *__restrict out_tile) {
  scale_i32_to_bf16_scalar<DIM_M, DIM_N>(in_tile, scale_row, out_tile);
}

}  // extern "C"
