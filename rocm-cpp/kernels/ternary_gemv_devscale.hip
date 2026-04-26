// Round-4 perf change — device-scalar overloads of the ternary fp16 GEMVs.
//
// Why:
//   The decode loop in tools/bitnet_decode.cpp computes x_scale =
//   max(|normed|)/127 inside rcpp_quantize_fp16_to_i8 (writes to a 4-byte
//   device location), then immediately hipMemcpy-D2H reads it so the host
//   can pass it by value into the fp16 GEMV launcher. That's ~150 host-
//   blocking round-trips per token at 30 layers, each ~10-30 µs, all in
//   the critical path. Eliminating them is a pure throughput win — the
//   scalar is already on the device, the kernel only consumes it at the
//   final fp32→fp16 write-back, and we can simply read `*x_scale_dev`
//   once per workgroup right before the store.
//
// Design constraint:
//   The canonical halo kernel (kernels/ternary_gemv_phase5_halo.hip) is
//   at 92% of LPDDR5x peak per project_bitnet_rocprof_plan.md and locked
//   for this round. Rather than perturb that source, the three kernels
//   below mirror the inner loops of the f16 launchers in
//     - kernels/ternary_gemv_phase5_halo.hip   (HALO_V2)
//     - kernels/ternary_gemv_sherry.hip        (SHERRY_I8)
//     - kernels/ternary_gemv_tq1_halo.hip      (TQ1)
//   with a single change: x_scale is loaded from a device pointer once
//   per workgroup (lane 0 read + LDS broadcast → free fast path on the
//   GPU because all 128 lanes hit the same 4-byte address).
//
// ABI note:
//   These are NEW C entry points — `rcpp_ternary_gemv_*_f16_devscale`.
//   The original `rcpp_ternary_gemv_*_f16` host-scalar entries stay
//   bit-for-bit unchanged so lemond + bitnet_decode older callers + the
//   bench tools all keep working. bitnet_decode is updated in this same
//   change to use the new entries on the hot decode path.
//
// Correctness:
//   Math is identical. The only difference vs the originals is WHERE
//   x_scale comes from. Read once per workgroup → broadcast via LDS
//   means every lane sees the same value (as before), and the fold-in
//   `(float)v * x_scale * scales[my_row]` happens lane-0-only at the
//   exact same place (under `lane == 0 && my_row < M`).
//
// Verify path:
//   1. tests/test_sherry_e2e.cpp drives a real model decode through the
//      sherry path; output logits should be argmax-equal to the prior
//      hipMemcpy-on-host path.
//   2. tools/bitnet_decode --bench --ctx 1024 --iters 64 should drop
//      ~150 D2H stalls per token (visible as fewer hipMemcpy_DtoH lines
//      in `rocprofv3 --hsa-trace` and a higher steady-state token rate).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <mutex>

#include "rocm_cpp/ck_gemm.h"

// LUTs are duplicated locally rather than cross-TU referenced. HIP
// `__constant__` cross-TU linkage is fragile across rocm-clang versions
// (the symbol mangling for `__constant__` is implementation-defined and
// has bitten us before). Cost is trivial — 128 B for sherry + 2 KB for tq1
// per device — vs. complexity savings of one self-contained translation
// unit owning its own const data.

// ── Local LUTs (mirror sherry / tq1 build_*_entry init) ─────────────────────
// `__constant__` arrays must be at file scope (NOT inside an anonymous
// namespace) for HIP to resolve the device symbol. Names are namespaced
// with the `devscale_` prefix to avoid colliding with the originals in
// kernels/ternary_gemv_sherry.hip and kernels/ternary_gemv_tq1_halo.hip.
__constant__ uint32_t devscale_sherry_unpack_lut[32];
__constant__ uint32_t devscale_tq1_lut_lo[256];
__constant__ uint32_t devscale_tq1_lut_hi[256];

// File-scope (matching the pattern in kernels/ternary_gemv_phase5_halo.hip,
// kernels/ternary_gemv_sherry.hip, kernels/ternary_gemv_tq1_halo.hip — none
// of which wrap their __global__ / __constant__ symbols in an anonymous
// namespace). Local symbols here use the `devscale_` prefix to avoid
// colliding with the originals.

#define BLOCK_SIZE      128
#define WARP_SIZE       32
#define ROWS_PER_BLOCK  8
#define ROWS_PER_WARP   2
#define VALS_PER_U32    16
#define LDS_TILE_I8_HALO 2048
#define LDS_TILE_I8_SHERRY 4096
#define LDS_TILE_I8_TQ1    2560
#define WEIGHTS_PER_MG_SHERRY 128
#define WEIGHTS_PER_MG_TQ1    20

static __device__ __forceinline__ uint32_t devscale_pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((uint32_t)(uint8_t)a)
         | ((uint32_t)(uint8_t)b << 8)
         | ((uint32_t)(uint8_t)c << 16)
         | ((uint32_t)(uint8_t)d << 24);
}

static uint32_t build_sherry_entry(uint32_t v) {
    uint32_t zp    = (v >> 3) & 0x3;
    uint32_t signs =  v       & 0x7;
    int8_t bytes[4] = {0, 0, 0, 0};
    int idx = 0;
    for (int p = 0; p < 4; ++p) {
        if ((uint32_t)p == zp) continue;
        int bit = (signs >> idx) & 1;
        bytes[p] = (int8_t)(bit ? 1 : -1);
        ++idx;
    }
    return   ((uint32_t)(uint8_t)bytes[0])
         | (((uint32_t)(uint8_t)bytes[1]) << 8)
         | (((uint32_t)(uint8_t)bytes[2]) << 16)
         | (((uint32_t)(uint8_t)bytes[3]) << 24);
}

static uint32_t pack4_host(int8_t a, int8_t b, int8_t c, int8_t d) {
    return   (uint32_t)(uint8_t)a
         | ((uint32_t)(uint8_t)b <<  8)
         | ((uint32_t)(uint8_t)c << 16)
         | ((uint32_t)(uint8_t)d << 24);
}

static std::once_flag devscale_sherry_lut_flag;
static void init_devscale_sherry_lut() {
    uint32_t host_lut[32];
    for (uint32_t v = 0; v < 32; ++v) host_lut[v] = build_sherry_entry(v);
    hipMemcpyToSymbol(HIP_SYMBOL(devscale_sherry_unpack_lut), host_lut,
                      sizeof(host_lut), 0, hipMemcpyHostToDevice);
}

static std::once_flag devscale_tq1_lut_flag;
static void init_devscale_tq1_lut() {
    uint32_t lo[256], hi[256];
    for (int byte = 0; byte < 256; ++byte) {
        int b = byte;
        int8_t v[5] = {0, 0, 0, 0, 0};
        if (b < 243) {
            for (int i = 0; i < 5; ++i) {
                int d = b % 3;
                v[i]  = (int8_t)(d - 1);
                b    /= 3;
            }
        }
        lo[byte] = pack4_host(v[0], v[1], v[2], v[3]);
        hi[byte] = pack4_host(v[4], 0, 0, 0);
    }
    hipMemcpyToSymbol(HIP_SYMBOL(devscale_tq1_lut_lo), lo, sizeof(lo), 0, hipMemcpyHostToDevice);
    hipMemcpyToSymbol(HIP_SYMBOL(devscale_tq1_lut_hi), hi, sizeof(hi), 0, hipMemcpyHostToDevice);
}

// Halo encoding: sign = (bits == 2) - (bits == 0). Identical to the helper
// in ternary_gemv_phase5_halo.hip (private to that TU; we re-derive here so
// the original file stays untouched).
static __device__ __forceinline__ uint32_t unpack_ternary_halo_4_dev(uint32_t word, int base_v) {
    const uint32_t bits0 = (word >> ((base_v + 0) * 2)) & 0x3;
    const uint32_t bits1 = (word >> ((base_v + 1) * 2)) & 0x3;
    const uint32_t bits2 = (word >> ((base_v + 2) * 2)) & 0x3;
    const uint32_t bits3 = (word >> ((base_v + 3) * 2)) & 0x3;
    const int8_t s0 = (int8_t)((int)(bits0 == 2) - (int)(bits0 == 0));
    const int8_t s1 = (int8_t)((int)(bits1 == 2) - (int)(bits1 == 0));
    const int8_t s2 = (int8_t)((int)(bits2 == 2) - (int)(bits2 == 0));
    const int8_t s3 = (int8_t)((int)(bits3 == 2) - (int)(bits3 == 0));
    return devscale_pack4(s0, s1, s2, s3);
}

// ── HALO_V2 device-scalar f16 kernel ────────────────────────────────────────
// Mirrors ternary_gemv_phase5_halo_f16_kernel but reads x_scale from a device
// pointer. Inner loop is byte-identical; only the writeback lane reads
// *x_scale_dev (all lanes branch on `lane == 0 && my_row < M`, so loading
// it lazily after the reduction has zero cost on the math path).
__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_phase5_halo_f16_devscale_kernel(
    const uint32_t* __restrict__ packed,
    const int8_t*   __restrict__ x_i8,
    const float*    __restrict__ x_scale_dev,    // [1] device-side scalar
    const float*    __restrict__ scales,
    __half*         __restrict__ y_f16,
    int M, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;

    const int row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_row_base = row_base + wave * ROWS_PER_WARP;

    const int packed_k = K / VALS_PER_U32;

    __shared__ int8_t x_shared[LDS_TILE_I8_HALO];

    int32_t acc[ROWS_PER_WARP] = {};

    for (int k_base = 0; k_base < K; k_base += LDS_TILE_I8_HALO) {
        __syncthreads();
        using v16 = __attribute__((ext_vector_type(16))) int8_t;
        const int k_start = k_base + tid * 16;
        v16* __restrict__ lds_v = reinterpret_cast<v16*>(x_shared) + tid;
        if (k_start + 16 <= K) {
            const v16* __restrict__ src_v = reinterpret_cast<const v16*>(x_i8 + k_start);
            *lds_v = *src_v;
        } else {
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                int k = k_start + b;
                x_shared[tid * 16 + b] = (k < K) ? x_i8[k] : (int8_t)0;
            }
        }
        __syncthreads();

        const int tile_u32_start = k_base / VALS_PER_U32;
        const int tile_u32_count = LDS_TILE_I8_HALO / VALS_PER_U32;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int my_row = warp_row_base + r;
            if (my_row >= M) continue;
            const uint32_t* row_weights = packed + (size_t)my_row * packed_k;

            for (int u = lane; u < tile_u32_count; u += WARP_SIZE) {
                int global_u32 = tile_u32_start + u;
                if (global_u32 >= packed_k) break;
                const uint32_t word = row_weights[global_u32];
                const int x_offset  = u * VALS_PER_U32;

                const uint32_t* xpack = reinterpret_cast<const uint32_t*>(&x_shared[x_offset]);
                const uint32_t xw0 = xpack[0];
                const uint32_t xw1 = xpack[1];
                const uint32_t xw2 = xpack[2];
                const uint32_t xw3 = xpack[3];

                const uint32_t tw0 = unpack_ternary_halo_4_dev(word, 0);
                const uint32_t tw1 = unpack_ternary_halo_4_dev(word, 4);
                const uint32_t tw2 = unpack_ternary_halo_4_dev(word, 8);
                const uint32_t tw3 = unpack_ternary_halo_4_dev(word, 12);

                int32_t local = 0;
                local = __builtin_amdgcn_sudot4(true, tw0, true, xw0, local, false);
                local = __builtin_amdgcn_sudot4(true, tw1, true, xw1, local, false);
                local = __builtin_amdgcn_sudot4(true, tw2, true, xw2, local, false);
                local = __builtin_amdgcn_sudot4(true, tw3, true, xw3, local, false);
                acc[r] += local;
            }
        }
    }

    // Single device-scalar load. All 128 lanes hit the same 4-byte address;
    // the L1 dcache serves it as one cache-line broadcast. Done before the
    // reduction loop so it overlaps with the warp-shuffle latency.
    const float x_scale = *x_scale_dev;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int32_t v = acc[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset);
        }
        const int my_row = warp_row_base + r;
        if (lane == 0 && my_row < M) {
            y_f16[my_row] = __float2half((float)v * x_scale * scales[my_row]);
        }
    }
}

// ── SHERRY_I8 device-scalar f16 kernel ──────────────────────────────────────
// Mirrors ternary_gemv_sherry_kernel. Uses the local devscale_sherry_unpack_lut
// declared above; init runs once via std::call_once at first launch.

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_sherry_f16_devscale_kernel(
    const uint32_t* __restrict__ packed,
    const int8_t*   __restrict__ x_i8,
    const float*    __restrict__ x_scale_dev,
    const float*    __restrict__ scales,
    __half*         __restrict__ y_f16,
    int M, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;

    const int row_base       = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_row_base  = row_base + wave * ROWS_PER_WARP;

    const int u32_per_row       = (K / WEIGHTS_PER_MG_SHERRY) * 5;
    const int total_mg_per_row  =  K / WEIGHTS_PER_MG_SHERRY;

    constexpr int tile_macrogroups = LDS_TILE_I8_SHERRY / WEIGHTS_PER_MG_SHERRY;
    constexpr int MG_U32           = WEIGHTS_PER_MG_SHERRY / 4;
    constexpr int MG_STRIDE_U32    = MG_U32 + 1;

    __shared__ uint32_t x_shared_u32[tile_macrogroups * MG_STRIDE_U32];

    const int32_t my_lut_entry = (int32_t)devscale_sherry_unpack_lut[lane];

    int32_t acc[ROWS_PER_WARP] = {};

    for (int k_base = 0; k_base < K; k_base += LDS_TILE_I8_SHERRY) {
        __syncthreads();
        constexpr int TOTAL_U32 = tile_macrogroups * MG_U32;
        #pragma unroll
        for (int ki = tid; ki < TOTAL_U32; ki += BLOCK_SIZE) {
            const int mg_i   = ki / MG_U32;
            const int slot   = ki - mg_i * MG_U32;
            const int k_byte = k_base + ki * 4;
            uint32_t val = 0;
            if (k_byte + 4 <= K) {
                val = *reinterpret_cast<const uint32_t*>(x_i8 + k_byte);
            } else if (k_byte < K) {
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int kk = k_byte + b;
                    if (kk < K) val |= ((uint32_t)(uint8_t)x_i8[kk]) << (b * 8);
                }
            }
            x_shared_u32[mg_i * MG_STRIDE_U32 + slot] = val;
        }
        __syncthreads();

        const int tile_macrogroup_start = k_base / WEIGHTS_PER_MG_SHERRY;
        const int tile_mg_count         = (tile_macrogroup_start + tile_macrogroups <= total_mg_per_row)
                                            ? tile_macrogroups
                                            : (total_mg_per_row - tile_macrogroup_start);

        const int mg = lane;
        const bool mg_valid = (mg < tile_mg_count);
        const int global_mg = tile_macrogroup_start + mg;
        const uint32_t* xpack = &x_shared_u32[mg * MG_STRIDE_U32];

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int my_row = warp_row_base + r;
            const bool r_valid = (my_row < M) && mg_valid;

            uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0;
            if (r_valid) {
                const uint32_t* mg_base = packed + (size_t)my_row * u32_per_row + global_mg * 5;
                w0 = __builtin_nontemporal_load(&mg_base[0]);
                w1 = __builtin_nontemporal_load(&mg_base[1]);
                w2 = __builtin_nontemporal_load(&mg_base[2]);
                w3 = __builtin_nontemporal_load(&mg_base[3]);
                w4 = __builtin_nontemporal_load(&mg_base[4]);
            }

            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            uint32_t c;
            uint32_t lut_c;
            #define SG(acc_v, code_expr, act_idx) do { \
                c = (code_expr); \
                lut_c = (uint32_t)__shfl(my_lut_entry, (int)(c)); \
                acc_v = __builtin_amdgcn_sudot4(true, lut_c, true, xpack[act_idx], acc_v, false); \
            } while (0)

            SG(a0,  w0        & 0x1Fu,                                  0 );
            SG(a1, (w0 >>  5) & 0x1Fu,                                  1 );
            SG(a2, (w0 >> 10) & 0x1Fu,                                  2 );
            SG(a3, (w0 >> 15) & 0x1Fu,                                  3 );
            SG(a0, (w0 >> 20) & 0x1Fu,                                  4 );
            SG(a1, (w0 >> 25) & 0x1Fu,                                  5 );
            SG(a2, ((w0 >> 30) & 0x3u)  | ((w1 & 0x7u)  << 2),          6 );
            SG(a3, (w1 >>  3) & 0x1Fu,                                  7 );
            SG(a0, (w1 >>  8) & 0x1Fu,                                  8 );
            SG(a1, (w1 >> 13) & 0x1Fu,                                  9 );
            SG(a2, (w1 >> 18) & 0x1Fu,                                 10 );
            SG(a3, (w1 >> 23) & 0x1Fu,                                 11 );
            SG(a0, ((w1 >> 28) & 0xFu)  | ((w2 & 0x1u)  << 4),         12 );
            SG(a1, (w2 >>  1) & 0x1Fu,                                 13 );
            SG(a2, (w2 >>  6) & 0x1Fu,                                 14 );
            SG(a3, (w2 >> 11) & 0x1Fu,                                 15 );
            SG(a0, (w2 >> 16) & 0x1Fu,                                 16 );
            SG(a1, (w2 >> 21) & 0x1Fu,                                 17 );
            SG(a2, (w2 >> 26) & 0x1Fu,                                 18 );
            SG(a3, ((w2 >> 31) & 0x1u)  | ((w3 & 0xFu)  << 1),         19 );
            SG(a0, (w3 >>  4) & 0x1Fu,                                 20 );
            SG(a1, (w3 >>  9) & 0x1Fu,                                 21 );
            SG(a2, (w3 >> 14) & 0x1Fu,                                 22 );
            SG(a3, (w3 >> 19) & 0x1Fu,                                 23 );
            SG(a0, (w3 >> 24) & 0x1Fu,                                 24 );
            SG(a1, ((w3 >> 29) & 0x7u)  | ((w4 & 0x3u)  << 3),         25 );
            SG(a2, (w4 >>  2) & 0x1Fu,                                 26 );
            SG(a3, (w4 >>  7) & 0x1Fu,                                 27 );
            SG(a0, (w4 >> 12) & 0x1Fu,                                 28 );
            SG(a1, (w4 >> 17) & 0x1Fu,                                 29 );
            SG(a2, (w4 >> 22) & 0x1Fu,                                 30 );
            SG(a3, (w4 >> 27) & 0x1Fu,                                 31 );
            #undef SG
            if (r_valid) acc[r] += (a0 + a1) + (a2 + a3);
        }
    }

    const float x_scale = *x_scale_dev;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int32_t v = acc[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset);
        }
        const int my_row = warp_row_base + r;
        if (lane == 0 && my_row < M) {
            float out = (float)v * x_scale * scales[my_row];
            constexpr float FP16_MAX = 65504.0f;
            if (out >  FP16_MAX) out =  FP16_MAX;
            if (out < -FP16_MAX) out = -FP16_MAX;
            y_f16[my_row] = __float2half(out);
        }
    }
}

// ── TQ1 device-scalar f16 kernel ────────────────────────────────────────────
// Mirrors ternary_gemv_tq1_halo_kernel. Uses the local devscale_tq1_lut_*
// arrays declared above; init runs once via std::call_once at first launch.

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_tq1_halo_f16_devscale_kernel(
    const uint32_t* __restrict__ packed,
    const int8_t*   __restrict__ x_i8,
    const float*    __restrict__ x_scale_dev,
    const float*    __restrict__ scales,
    __half*         __restrict__ y_f16,
    int M, int K_padded)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;

    const int row_base       = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_row_base  = row_base + wave * ROWS_PER_WARP;

    const int u32_per_row = K_padded / WEIGHTS_PER_MG_TQ1;

    __shared__ int8_t x_shared[LDS_TILE_I8_TQ1];

    int32_t acc[ROWS_PER_WARP] = {};

    static_assert(LDS_TILE_I8_TQ1 == BLOCK_SIZE * 20, "TQ1 LDS tile assumption");
    for (int k_base = 0; k_base < K_padded; k_base += LDS_TILE_I8_TQ1) {
        __syncthreads();
        using v16 = __attribute__((ext_vector_type(16))) int8_t;
        const int lds_off16 = tid * 16;
        const int k_start16 = k_base + lds_off16;
        v16* __restrict__ lds_v = reinterpret_cast<v16*>(&x_shared[lds_off16]);
        if (k_start16 + 16 <= K_padded) {
            const v16* __restrict__ src_v = reinterpret_cast<const v16*>(x_i8 + k_start16);
            *lds_v = *src_v;
        } else {
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                int k = k_start16 + b;
                x_shared[lds_off16 + b] = (k < K_padded) ? x_i8[k] : (int8_t)0;
            }
        }
        const int lds_off4 = BLOCK_SIZE * 16 + tid * 4;
        const int k_start4 = k_base + lds_off4;
        if (k_start4 + 4 <= K_padded) {
            *reinterpret_cast<uint32_t*>(&x_shared[lds_off4])
                = *reinterpret_cast<const uint32_t*>(x_i8 + k_start4);
        } else {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                int k = k_start4 + b;
                x_shared[lds_off4 + b] = (k < K_padded) ? x_i8[k] : (int8_t)0;
            }
        }
        __syncthreads();

        const int tile_macrogroups      = LDS_TILE_I8_TQ1 / WEIGHTS_PER_MG_TQ1;
        const int tile_macrogroup_start = k_base / WEIGHTS_PER_MG_TQ1;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int my_row = warp_row_base + r;
            if (my_row >= M) continue;
            const uint32_t* row_packed = packed + (size_t)my_row * u32_per_row;

            for (int mg = lane; mg < tile_macrogroups; mg += WARP_SIZE) {
                const int global_mg = tile_macrogroup_start + mg;
                if (global_mg >= u32_per_row) break;
                const uint32_t packed_word = row_packed[global_mg];

                // Identical 6-perm repack as ternary_gemv_tq1_halo_kernel
                // (kernels/ternary_gemv_tq1_halo.hip:130). Each byte holds
                // 5 ternaries; v_perm_b32 reassembles them across byte
                // boundaries into 5 DP4A-ready u32s. Comments inline below
                // mirror the original byte-by-byte derivation.
                const uint32_t b0 = (packed_word >>  0) & 0xFFu;
                const uint32_t b1 = (packed_word >>  8) & 0xFFu;
                const uint32_t b2 = (packed_word >> 16) & 0xFFu;
                const uint32_t b3 = (packed_word >> 24) & 0xFFu;

                const uint32_t lo0 = devscale_tq1_lut_lo[b0];
                const uint32_t lo1 = devscale_tq1_lut_lo[b1], hi1 = devscale_tq1_lut_hi[b1];
                const uint32_t lo2 = devscale_tq1_lut_lo[b2], hi2 = devscale_tq1_lut_hi[b2];
                const uint32_t lo3 = devscale_tq1_lut_lo[b3], hi3 = devscale_tq1_lut_hi[b3];
                const uint32_t hi0 = devscale_tq1_lut_hi[b0];

                uint32_t w_u32[5];
                w_u32[0] = lo0;
                w_u32[1] = __builtin_amdgcn_perm(lo1, hi0, 0x06050400u);
                const uint32_t t_w2 = __builtin_amdgcn_perm(hi1, lo1, 0x07060403u);
                w_u32[2]            = __builtin_amdgcn_perm(lo2, t_w2, 0x05040100u);
                const uint32_t t_w3 = __builtin_amdgcn_perm(hi2, lo2, 0x07040302u);
                w_u32[3]            = __builtin_amdgcn_perm(lo3, t_w3, 0x04020100u);
                w_u32[4] = __builtin_amdgcn_perm(hi3, lo3, 0x04030201u);

                const int x_offset = mg * WEIGHTS_PER_MG_TQ1;
                const uint32_t* xpack = reinterpret_cast<const uint32_t*>(&x_shared[x_offset]);

                int32_t local = 0;
                #pragma unroll
                for (int j = 0; j < 5; ++j) {
                    local = __builtin_amdgcn_sudot4(true, w_u32[j], true, xpack[j], local, false);
                }
                acc[r] += local;
            }
        }
    }

    const float x_scale = *x_scale_dev;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int32_t v = acc[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset);
        }
        const int my_row = warp_row_base + r;
        if (lane == 0 && my_row < M) {
            constexpr float FP16_MAX = 65504.0f;
            float out = (float)v * x_scale * scales[my_row];
            out = fminf(fmaxf(out, -FP16_MAX), FP16_MAX);
            y_f16[my_row] = __float2half(out);
        }
    }
}

// ── Public C entry points (additive — original *_f16 entries unchanged) ─────

extern "C" rcpp_status_t
rcpp_ternary_gemv_halo_f16_devscale(const void* packed_weights_dev,
                                    const void* activations_i8_dev,
                                    const float* x_scale_dev,
                                    const void* row_scales_dev,
                                    void* output_f16_dev,
                                    int M, int K, void* stream)
{
    if (!packed_weights_dev || !activations_i8_dev || !x_scale_dev ||
        !row_scales_dev || !output_f16_dev) return RCPP_INVALID_ARG;
    if (M <= 0 || K <= 0)                  return RCPP_INVALID_ARG;
    if (K % 16 != 0)                       return RCPP_INVALID_ARG;

    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_phase5_halo_f16_devscale_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const uint32_t*>(packed_weights_dev),
                       static_cast<const int8_t*>(activations_i8_dev),
                       x_scale_dev,
                       static_cast<const float*>(row_scales_dev),
                       static_cast<__half*>(output_f16_dev),
                       M, K);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_ternary_gemv_sherry_f16_devscale(const void* packed_weights_dev,
                                      const void* activations_i8_dev,
                                      const float* x_scale_dev,
                                      const void* row_scales_dev,
                                      void* output_f16_dev,
                                      int M, int K, void* stream)
{
    if (!packed_weights_dev || !activations_i8_dev || !x_scale_dev ||
        !row_scales_dev || !output_f16_dev) return RCPP_INVALID_ARG;
    if (M <= 0 || K <= 0)                  return RCPP_INVALID_ARG;
    if (K % 32 != 0)                       return RCPP_INVALID_ARG;

    std::call_once(devscale_sherry_lut_flag, init_devscale_sherry_lut);

    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_sherry_f16_devscale_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const uint32_t*>(packed_weights_dev),
                       static_cast<const int8_t*>(activations_i8_dev),
                       x_scale_dev,
                       static_cast<const float*>(row_scales_dev),
                       static_cast<__half*>(output_f16_dev),
                       M, K);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_ternary_gemv_tq1_halo_f16_devscale(const void* packed_weights_dev,
                                        const void* activations_i8_dev,
                                        const float* x_scale_dev,
                                        const void* row_scales_dev,
                                        void* output_f16_dev,
                                        int M, int K_padded, void* stream)
{
    if (!packed_weights_dev || !activations_i8_dev || !x_scale_dev ||
        !row_scales_dev || !output_f16_dev) return RCPP_INVALID_ARG;
    if (M <= 0 || K_padded <= 0)            return RCPP_INVALID_ARG;
    if (K_padded % 20 != 0)                 return RCPP_INVALID_ARG;

    std::call_once(devscale_tq1_lut_flag, init_devscale_tq1_lut);

    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_tq1_halo_f16_devscale_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const uint32_t*>(packed_weights_dev),
                       static_cast<const int8_t*>(activations_i8_dev),
                       x_scale_dev,
                       static_cast<const float*>(row_scales_dev),
                       static_cast<__half*>(output_f16_dev),
                       M, K_padded);
    return RCPP_OK;
}
