// Host-side launcher wrappers for Phase 5 ternary GEMV kernels.
// CK-independent — always compiled. The actual kernel definitions are in
// kernels/ternary_gemv_phase5_dot4.hip, kernels/ternary_gemv_sherry.hip, etc.

#include "rocm_cpp/ck_gemm.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declarations of kernel launcher functions (defined in kernels/*.hip
// with extern "C" linkage). Must use extern "C" here too to match.
// ---------------------------------------------------------------------------
extern "C" void ternary_gemv_phase5_dot4_launch(const void*, const void*, float,
                                                  const void*, void*, int, int, void*);
extern "C" void ternary_gemv_phase5_halo_launch(const void*, const void*, float,
                                                  const void*, void*, int, int, void*);
extern "C" void ternary_gemv_phase5_halo_f16_launch(const void*, const void*, float,
                                                      const void*, void*, int, int, void*);
extern "C" void ternary_gemv_sherry_f16_launch(const void*, const void*, float,
                                                 const void*, void*, int, int, void*);
extern "C" void ternary_gemv_tq1_halo_f16_launch(const void*, const void*, float,
                                                   const void*, void*, int, int, void*);
// WMMA ternary GEMV — uses __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32.
// Falls back to DP4A (halo_f16) when K < 16 or K % 16 != 0.
extern "C" void ternary_gemv_wmma_launch(const void*, const void*, float,
                                          const void*, void*, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma_4x4_vec(const void*, const void*, void*,
                                                      int, int, int, void*);

// ---------------------------------------------------------------------------
// Phase 5 decode GEMV — dot4 variant (current winner)
// ---------------------------------------------------------------------------
extern "C" rcpp_status_t
rcpp_ternary_gemv(const void* packed, const void* x_i8, float x_scale,
                  const void* row_scales, void* y,
                  int M, int K, void* stream) {
    if(!packed || !x_i8 || !row_scales || !y) return RCPP_INVALID_ARG;
    if(M <= 0 || K <= 0)                      return RCPP_INVALID_ARG;
    if(K % 16 != 0)                           return RCPP_INVALID_ARG;
    ternary_gemv_phase5_dot4_launch(packed, x_i8, x_scale, row_scales, y, M, K, stream);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_ternary_gemv_halo(const void* packed, const void* x_i8, float x_scale,
                       const void* row_scales, void* y,
                       int M, int K, void* stream) {
    if(!packed || !x_i8 || !row_scales || !y) return RCPP_INVALID_ARG;
    if(M <= 0 || K <= 0)                      return RCPP_INVALID_ARG;
    if(K % 16 != 0)                           return RCPP_INVALID_ARG;
    ternary_gemv_phase5_halo_launch(packed, x_i8, x_scale, row_scales, y, M, K, stream);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_ternary_gemv_halo_f16(const void* packed, const void* x_i8, float x_scale,
                           const void* row_scales, void* y_f16,
                           int M, int K, void* stream) {
    if(!packed || !x_i8 || !row_scales || !y_f16) return RCPP_INVALID_ARG;
    if(M <= 0 || K <= 0)                          return RCPP_INVALID_ARG;
    if(K % 16 != 0)                               return RCPP_INVALID_ARG;
    ternary_gemv_phase5_halo_f16_launch(packed, x_i8, x_scale, row_scales, y_f16, M, K, stream);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_ternary_gemv_sherry_f16(const void* packed, const void* x_i8, float x_scale,
                             const void* row_scales, void* y_f16,
                             int M, int K, void* stream) {
    if(!packed || !x_i8 || !row_scales || !y_f16) return RCPP_INVALID_ARG;
    if(M <= 0 || K <= 0)                          return RCPP_INVALID_ARG;
    if(K % 32 != 0)                               return RCPP_INVALID_ARG;
    ternary_gemv_sherry_f16_launch(packed, x_i8, x_scale, row_scales, y_f16, M, K, stream);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_ternary_gemv_tq1_halo_f16(const void* packed, const void* x_i8, float x_scale,
                               const void* row_scales, void* y_f16,
                               int M, int K_padded, void* stream) {
    if(!packed || !x_i8 || !row_scales || !y_f16) return RCPP_INVALID_ARG;
    if(M <= 0 || K_padded <= 0)                   return RCPP_INVALID_ARG;
    if(K_padded % 20 != 0)                        return RCPP_INVALID_ARG;
    ternary_gemv_tq1_halo_f16_launch(packed, x_i8, x_scale, row_scales, y_f16, M, K_padded, stream);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_ternary_gemv_wmma(const void* packed, const void* x_i8, float x_scale,
                       const void* row_scales, void* y_f16,
                       int M, int K, void* stream) {
    if(!packed || !x_i8 || !row_scales || !y_f16) return RCPP_INVALID_ARG;
    if(M <= 0 || K <= 0)                          return RCPP_INVALID_ARG;
    ternary_gemv_wmma_launch(packed, x_i8, x_scale, row_scales, y_f16, M, K, stream);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_standalone_gemm(const void* A_dev, const void* B_dev_packed, void* C_dev,
                     int M, int N, int K, void* stream) {
    if(!A_dev || !B_dev_packed || !C_dev) return RCPP_INVALID_ARG;
    if(M <= 0 || N <= 0 || K <= 0)        return RCPP_INVALID_ARG;
    if(M % 64 != 0 || N % 64 != 0)        return RCPP_INVALID_ARG;
    if(K % 32 != 0)                       return RCPP_INVALID_ARG;
    rcpp_standalone_launch_wmma_4x4_vec(A_dev, B_dev_packed, C_dev, M, N, K, stream);
    return RCPP_OK;
}

// ---------------------------------------------------------------------------
// Offline packer — ternary int8 {-1, 0, +1} col-major [K, N] -> pk_i4
// WMMA-permuted bytes [K*N/2].
//
// Byte-level semantics (replacement for CK's ck::Tensor<pk_i4_t> approach):
//
//   Un-permuted byte layout, col-major B:
//     For each column n in [0,N), each K-pair (k, k+1):
//       byte[(n*K + k)/2] = (hi << 4) | lo
//     where hi = nibble(ternary[n, k    ])  -- first element, HIGH nibble
//           lo = nibble(ternary[n, k + 1])  -- second element, LOW nibble
//     (High-first ordering is required by CK_USE_PK4_LAYOUT_SHUFFLE=1.)
//
//   Nibble map compensates for CK's "n - 8" FP16 decode:
//     -1 -> 0x7  (decodes 7 - 8 = -1)
//      0 -> 0x8  (decodes 8 - 8 =  0)
//     +1 -> 0x9  (decodes 9 - 8 = +1)
//
//   Block permute (K -> [K0, N, K1] with K1 = 32):
//     dst_byte = (j * N * K1/2) + (i * K1/2) + (jj/2)
//     src_byte = (i * K/2)      + (j * K1/2) + (jj/2)
//
//   Within-8 nibble permute (01234567 -> 20643175, per upstream example):
//     For each column i, K-group of 8: read 4 bytes, permute 8 nibbles.
//
// No CK headers — pure C++17 with one std::vector.
// ---------------------------------------------------------------------------
extern "C" rcpp_status_t
rcpp_ternary_pack_pk_i4(const int8_t* ternary_host, int8_t* packed_host,
                        int K, int N) {
    if(!ternary_host || !packed_host || K <= 0 || N <= 0) return RCPP_INVALID_ARG;
    if(K % 32 != 0 || K % 8 != 0)                         return RCPP_INVALID_ARG;

    auto t_to_i4 = [](int8_t t) -> uint8_t {
        if(t == 0)  return 0x8;
        if(t >  0)  return 0x9;
        return 0x7;
    };

    constexpr int KPerBlock  = 32;
    constexpr int K1         = KPerBlock;
    constexpr int K1_bytes   = K1 / 2;

    const std::size_t nbytes = (std::size_t)K * N / 2;

    // Stage A: pack ternary -> un-permuted pk_i4 bytes (col-major [K, N]).
    std::vector<uint8_t> unpermuted(nbytes);
    for(int n = 0; n < N; ++n) {
        for(int k = 0; k < K; k += 2) {
            uint8_t hi = t_to_i4(ternary_host[(std::size_t)n * K + k    ]);
            uint8_t lo = t_to_i4(ternary_host[(std::size_t)n * K + k + 1]);
            unpermuted[((std::size_t)n * K + k) / 2] = (uint8_t)((hi << 4) | lo);
        }
    }

    // Stage B: block-reshape K -> [K0, N, K1] at K1/2-byte granularity.
    const int K0 = K / K1;
    uint8_t* dst = reinterpret_cast<uint8_t*>(packed_host);
    for(int j = 0; j < K0; ++j) {
        for(int i = 0; i < N; ++i) {
            const std::size_t dst_off =
                (std::size_t)j * N * K1_bytes + (std::size_t)i * K1_bytes;
            const std::size_t src_off =
                (std::size_t)i * (K / 2) + (std::size_t)j * K1_bytes;
            std::memcpy(dst + dst_off, unpermuted.data() + src_off, K1_bytes);
        }
    }

    // Stage C: within-8 nibble permute in place.
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < K; j += 8) {
            uint8_t* p = dst + ((std::size_t)i * K + j) / 2;
            int input[8];
            for(int kk = 0; kk < 4; ++kk) {
                int b = p[kk];
                input[kk * 2 + 0] = (b >> 4) & 0xf;
                input[kk * 2 + 1] = (b >> 0) & 0xf;
            }
            p[0] = (uint8_t)((input[2] << 4) | input[0]);
            p[1] = (uint8_t)((input[6] << 4) | input[4]);
            p[2] = (uint8_t)((input[3] << 4) | input[1]);
            p[3] = (uint8_t)((input[7] << 4) | input[5]);
        }
    }
    return RCPP_OK;
}
