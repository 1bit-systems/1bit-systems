// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
// Sherry — see LICENSE-SHERRY.md and SHERRY-FILES.txt at the repo root.
// Commercial use requires a separate license.
//
// Sherry 1.25-bit ternary GEMV — halo-1bit v3 packing (Lane B of rocprof plan).
//
// Packing spec (per `project_sherry_spike.md`):
//   - Each group of 4 consecutive weights has exactly one forced-zero position.
//   - Encoded as 5 bits: zero_pos (2 bits) || signs (3 bits, MSB→LSB gives the
//     1st, 2nd, 3rd non-zero position in positional order skipping zero_pos).
//   - Row stride: K * 5 / 32 bytes. K must be a multiple of 32.
//   - Per-row FP32 scale retained (same as halo v2).
//
// Processing granularity: 128 weights per macro-group = 5 u32s = 20 bytes of
// packed data = 32 subgroups of 5 bits each. u32-aligned loads, byte-packed
// within.
//
// Each 5-bit subgroup → 4 signed bytes (one of which is 0) via a 32-entry
// __constant__ LUT populated at first launch.
//
// Output: FP16 directly (fused fp32→fp16 convert, per Lane A pattern).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <mutex>

#define BLOCK_SIZE     128
#define WARP_SIZE      32
#define ROWS_PER_BLOCK 8
#define ROWS_PER_WARP  2
#define WEIGHTS_PER_MACROGROUP 128   // 5 u32s of packed data
#define BYTES_PER_MACROGROUP   20
// LDS tile sized so tile_macrogroups (LDS_TILE/128) >= WARP_SIZE (32), else
// half the warp idles. 4096 = exactly 32 macro-groups per tile = 1 iter/lane.
#define LDS_TILE_I8    4096

// ── Unpack LUT ──────────────────────────────────────────────────────────────
// 32 entries, one per 5-bit value. Each entry = pack4(b0, b1, b2, b3) where
// bi is the signed int8 weight at position i (0 at the forced-zero position).
__constant__ uint32_t sherry_unpack_lut[32];

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

static std::once_flag sherry_lut_flag;
static void init_sherry_lut(hipStream_t stream) {
    uint32_t host_lut[32];
    for (uint32_t v = 0; v < 32; ++v) host_lut[v] = build_sherry_entry(v);
    hipMemcpyToSymbolAsync(HIP_SYMBOL(sherry_unpack_lut), host_lut,
                           sizeof(host_lut), 0, hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);
}

// ── Kernel ──────────────────────────────────────────────────────────────────
// Expert-tuned variant (P4 + P3 + P8 + P2):
//   P4: single scalar `c` variable instead of a 32-entry codes[] array
//   P3: 4-way accumulator split breaks serial sudot4 dependency chain
//   P8: __builtin_nontemporal_load on streamed weight reads (L2 bypass)
//   P2: VGPR-resident LUT via __builtin_amdgcn_ds_bpermute — each lane holds
//       lut_entry[lane] in a register, the hot loop does lane-to-lane shuffle
//       instead of ~1024 LDS loads per warp per macro-group
__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_sherry_kernel(
    const uint32_t* __restrict__ packed,   // row layout: K*5/32 bytes/row, viewed as u32s
    const int8_t*   __restrict__ x_i8,
    float                       x_scale,
    const float*    __restrict__ scales,
    __half*         __restrict__ y_f16,
    int M, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;

    const int row_base       = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_row_base  = row_base + wave * ROWS_PER_WARP;

    const int u32_per_row       = (K / WEIGHTS_PER_MACROGROUP) * 5;
    const int total_mg_per_row  =  K / WEIGHTS_PER_MACROGROUP;

    constexpr int tile_macrogroups = LDS_TILE_I8 / WEIGHTS_PER_MACROGROUP;  // 32
    constexpr int MG_U32           = WEIGHTS_PER_MACROGROUP / 4;            // 32 u32s per mg
    // Bank-conflict fix: pad each macro-group by +1 u32 in LDS.
    //   Original layout: xpack[act_idx] at (lane * MG_U32 + act_idx) u32 — bank
    //   (lane*32 + act_idx) % 32 = act_idx % 32, i.e. 32-way conflict (all
    //   lanes hit the same bank at each SG step).
    //   Padded layout:   xpack[act_idx] at (lane * (MG_U32+1) + act_idx) u32,
    //   bank (lane + act_idx) % 32 — all 32 lanes distinct → 0-way conflict.
    constexpr int MG_STRIDE_U32    = MG_U32 + 1;                            // 33

    __shared__ uint32_t x_shared_u32[tile_macrogroups * MG_STRIDE_U32];     // 32 * 33 = 1056

    // P2: VGPR-resident LUT via __shfl. Each lane L holds lut[L] in a register;
    // SG() fetches lut[c] with __shfl(my_lut_entry, c) — intra-wave lane shuffle,
    // zero LDS traffic in the hot loop. Wave32 × 32 entries = exact fit.
    // Caveat: __shfl is wave-collective. The SG block below therefore runs for
    // ALL lanes, and mg_valid / my_row bounds are handled by zeroing inputs
    // rather than early-`continue` (which would break reconvergence).
    const int32_t my_lut_entry = (int32_t)sherry_unpack_lut[lane];

    int32_t acc[ROWS_PER_WARP] = {};

    for (int k_base = 0; k_base < K; k_base += LDS_TILE_I8) {
        __syncthreads();
        // Fill the padded LDS tile. 1024 valid u32s of activation per tile
        // (32 mg × 32 u32s). Each thread handles 8 u32s = 32 bytes.
        // Mapping linear u32 index k_u32 -> padded slot (mg * MG_STRIDE_U32 + slot).
        constexpr int TOTAL_U32 = tile_macrogroups * MG_U32;                // 1024
        #pragma unroll
        for (int ki = tid; ki < TOTAL_U32; ki += BLOCK_SIZE) {
            const int mg_i   = ki / MG_U32;
            const int slot   = ki - mg_i * MG_U32;                          // ki % 32
            const int k_byte = k_base + ki * 4;
            uint32_t val = 0;
            if (k_byte + 4 <= K) {
                val = *reinterpret_cast<const uint32_t*>(x_i8 + k_byte);
            } else if (k_byte < K) {
                // Tail: byte-by-byte fill for the final partial u32.
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int kk = k_byte + b;
                    if (kk < K) val |= ((uint32_t)(uint8_t)x_i8[kk]) << (b * 8);
                }
            }
            x_shared_u32[mg_i * MG_STRIDE_U32 + slot] = val;
        }
        __syncthreads();

        const int tile_macrogroup_start = k_base / WEIGHTS_PER_MACROGROUP;
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

            // Predicate the load, NOT the SG block. Out-of-range lanes keep
            // w_i=0 → SG runs with benign inputs, preserving wave-uniform __shfl.
            uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0;
            if (r_valid) {
                const uint32_t* mg_base = packed + (size_t)my_row * u32_per_row + global_mg * 5;
                // P8: nontemporal loads — streamed weights don't need L2 retention.
                w0 = __builtin_nontemporal_load(&mg_base[0]);
                w1 = __builtin_nontemporal_load(&mg_base[1]);
                w2 = __builtin_nontemporal_load(&mg_base[2]);
                w3 = __builtin_nontemporal_load(&mg_base[3]);
                w4 = __builtin_nontemporal_load(&mg_base[4]);
            }

            // P3: 4-way accumulator split for sudot4 ILP.
            // P4: single `c` scalar reused (no codes[] array spill).
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            uint32_t c;
            uint32_t lut_c;
            #define SG(acc, code_expr, act_idx) do { \
                c = (code_expr); \
                lut_c = (uint32_t)__shfl(my_lut_entry, (int)(c)); \
                acc = __builtin_amdgcn_sudot4(true, lut_c, true, xpack[act_idx], acc, false); \
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

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int32_t v = acc[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset);
        }
        const int my_row = warp_row_base + r;
        if (lane == 0 && my_row < M) {
            // Clamp the fp32 product to the fp16 representable range BEFORE
            // __float2half. Sherry uses per-row absmean scales that can exceed
            // 50 on outlier rows (q/k/v/o after bf16 requant). In the pathological
            // case where `v * x_scale * scales[row]` exceeds 65504 (fp16 max),
            // __float2half clamps to +Inf with IEEE semantics — and downstream
            // RMSNorm (sum of squares → 1/sqrt(Inf) = 0 → 0*Inf = NaN) poisons
            // the entire residual stream. Clamping here keeps outlier rows at
            // fp16 max magnitude and preserves the rest of the forward pass.
            // Verified against halo v2 (per-tensor scale ~4): no measurable
            // accuracy change on in-range rows, NaN-safe on outliers.
            float out = (float)v * x_scale * scales[my_row];
            constexpr float FP16_MAX = 65504.0f;
            if (out >  FP16_MAX) out =  FP16_MAX;
            if (out < -FP16_MAX) out = -FP16_MAX;
            y_f16[my_row] = __float2half(out);
        }
    }
}

extern "C" void ternary_gemv_sherry_f16_launch(
    const void* packed, const void* x_i8, float x_scale,
    const void* scales, void* y_f16,
    int M, int K, void* stream)
{
    hipStream_t s = (hipStream_t)stream;
    std::call_once(sherry_lut_flag, init_sherry_lut, s);
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_sherry_kernel,
                       grid, block, 0, s,
                       static_cast<const uint32_t*>(packed),
                       static_cast<const int8_t*>(x_i8),
                       x_scale,
                       static_cast<const float*>(scales),
                       static_cast<__half*>(y_f16),
                       M, K);
}
