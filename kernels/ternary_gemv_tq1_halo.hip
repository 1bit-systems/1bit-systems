// TQ1 base-3 ternary GEMV — halo-1bit v4 packing (Lane B' of rocprof plan).
//
// Packing: 5 ternaries per byte via base-3 encoding (3^5 = 243 < 256).
//   byte = d0 + d1·3 + d2·9 + d3·27 + d4·81
//   d_i = 0 → -1, 1 → 0, 2 → +1
// Effective: 1.6 bpw. **Lossless for ternary** — any weight triplet survives
// intact. Trade vs halo v2: 20% fewer bytes for same content. Trade vs Sherry
// v3: slightly more bytes (1.6 vs 1.25 bpw) but no accuracy cost (Sherry's 3:4
// sparsity is training-time only; post-hoc breaks the model).
//
// Macro-group = 20 ternaries = 4 bytes = 1 u32. Per u32: 5 DP4As (20 weights,
// 4 per DP4A). K must be a multiple of 20; caller-side padding writes 0s.
//
// Output: fp16 (fused convert, Lane A pattern).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <mutex>

#define BLOCK_SIZE     128
#define WARP_SIZE      32
#define ROWS_PER_BLOCK 8
#define ROWS_PER_WARP  2
#define WEIGHTS_PER_MACROGROUP 20   // 4 bytes = 1 u32
#define LDS_TILE_I8    2560         // 128 macro-groups per tile = 4 per lane

// ── Base-3 decode LUT ──────────────────────────────────────────────────────
// Each of 256 possible bytes decodes to 5 int8 values. We store them packed
// as 2 u32s per entry: lo = [v0, v1, v2, v3] (DP4A-ready), hi = [v4, 0, 0, 0].
// Indices above 242 map to zeros (undefined in spec; keeps kernel sane).
__constant__ uint32_t tq1_lut_lo[256];
__constant__ uint32_t tq1_lut_hi[256];

static uint32_t pack4i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return   (uint32_t)(uint8_t)a
         | ((uint32_t)(uint8_t)b <<  8)
         | ((uint32_t)(uint8_t)c << 16)
         | ((uint32_t)(uint8_t)d << 24);
}

static std::once_flag tq1_lut_flag;
static void init_tq1_lut(hipStream_t stream) {
    uint32_t lo[256], hi[256];
    for (int byte = 0; byte < 256; ++byte) {
        int b = byte;
        int8_t v[5] = {0, 0, 0, 0, 0};
        if (b < 243) {
            for (int i = 0; i < 5; ++i) {
                int d = b % 3;   // 0, 1, 2
                v[i]  = (int8_t)(d - 1);  // -1, 0, +1
                b    /= 3;
            }
        }
        lo[byte] = pack4i8(v[0], v[1], v[2], v[3]);
        hi[byte] = pack4i8(v[4], 0, 0, 0);
    }
    hipMemcpyToSymbolAsync(HIP_SYMBOL(tq1_lut_lo), lo, sizeof(lo), 0, hipMemcpyHostToDevice, stream);
    hipMemcpyToSymbolAsync(HIP_SYMBOL(tq1_lut_hi), hi, sizeof(hi), 0, hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);
}

// ── Kernel ──────────────────────────────────────────────────────────────────
__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_tq1_halo_kernel(
    const uint32_t* __restrict__ packed,   // row layout: K/5 bytes/row viewed as u32s
    const int8_t*   __restrict__ x_i8,     // [K] activations, K padded to mult of 20
    float                       x_scale,
    const float*    __restrict__ scales,
    __half*         __restrict__ y_f16,
    int M, int K_padded)                   // K_padded is the padded K (mult of 20)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;

    const int row_base       = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_row_base  = row_base + wave * ROWS_PER_WARP;

    // u32s per row = bytes/row / 4 = (K_padded/5) / 4 = K_padded/20
    const int u32_per_row = K_padded / WEIGHTS_PER_MACROGROUP;

    __shared__ int8_t x_shared[LDS_TILE_I8];

    int32_t acc[ROWS_PER_WARP] = {};

    // LDS_TILE_I8 = 2560 = BLOCK_SIZE × 20 bytes/thread. Each thread loads
    // v16 + 4 extra bytes to cover the full tile (2048 + 512 = 2560).
    static_assert(LDS_TILE_I8 == BLOCK_SIZE * 20, "TQ1 LDS tile assumption");
    for (int k_base = 0; k_base < K_padded; k_base += LDS_TILE_I8) {
        __syncthreads();
        using v16 = __attribute__((ext_vector_type(16))) int8_t;
        const int lds_off16 = tid * 16;                          // 0, 16, 32, ... 2032
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
        // Each thread also loads 4 more bytes into the top-512-byte region.
        const int lds_off4 = BLOCK_SIZE * 16 + tid * 4;          // 2048, 2052, ... 2556
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

        const int tile_macrogroups      = LDS_TILE_I8 / WEIGHTS_PER_MACROGROUP;  // 128
        const int tile_macrogroup_start = k_base / WEIGHTS_PER_MACROGROUP;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int my_row = warp_row_base + r;
            if (my_row >= M) continue;
            const uint32_t* row_packed = packed + (size_t)my_row * u32_per_row;

            for (int mg = lane; mg < tile_macrogroups; mg += WARP_SIZE) {
                const int global_mg = tile_macrogroup_start + mg;
                if (global_mg >= u32_per_row) break;
                const uint32_t packed_word = row_packed[global_mg];

                // Decode 4 bytes × 5 ternaries = 20 values via LUT.
                // byte i's 5 values: lut_lo[byte] has [v0..v3], lut_hi[byte] has [v4, 0, 0, 0].
                const uint32_t b0 = (packed_word >>  0) & 0xFFu;
                const uint32_t b1 = (packed_word >>  8) & 0xFFu;
                const uint32_t b2 = (packed_word >> 16) & 0xFFu;
                const uint32_t b3 = (packed_word >> 24) & 0xFFu;

                // Assemble 20 int8 values as 5 u32s for DP4A using v_perm_b32
                // (single-cycle byte-shuffle). Repack cost: 6 perms total
                // (was 15+ shift/AND/OR ALU ops via the literal bit-fiddling
                // path — ~2.5× fewer hot-loop ops).
                //
                // __builtin_amdgcn_perm(a, b, mask): output byte i picks from
                //   { b.0, b.1, b.2, b.3, a.0, a.1, a.2, a.3 } at index
                //   ((mask >> (8*i)) & 7). Mask high nibble of each byte is
                //   ignored for normal selection (no sign-fill paths used).
                const uint32_t lo0 = tq1_lut_lo[b0], hi0 = tq1_lut_hi[b0];
                const uint32_t lo1 = tq1_lut_lo[b1], hi1 = tq1_lut_hi[b1];
                const uint32_t lo2 = tq1_lut_lo[b2], hi2 = tq1_lut_hi[b2];
                const uint32_t lo3 = tq1_lut_lo[b3], hi3 = tq1_lut_hi[b3];

                uint32_t w_u32[5];
                // w[0] = {b0.v0, b0.v1, b0.v2, b0.v3} = lo0
                w_u32[0] = lo0;
                // w[1] = {b0.v4, b1.v0, b1.v1, b1.v2}
                //   perm(a=lo1, b=hi0): sel 0..3 = hi0.{0,1,2,3}, sel 4..7 = lo1.{0,1,2,3}
                //   out.0 = hi0.0 -> sel 0, out.1 = lo1.0 -> sel 4,
                //   out.2 = lo1.1 -> sel 5, out.3 = lo1.2 -> sel 6
                w_u32[1] = __builtin_amdgcn_perm(lo1, hi0, 0x06050400u);
                // w[2] = {b1.v3, b1.v4, b2.v0, b2.v1}
                //   Needs 3 source u32s (lo1, hi1, lo2). Two-step:
                //   t = perm(a=hi1, b=lo1): out = {lo1.3, hi1.0, 0, 0}
                //     sel 0 = lo1.3 -> 3, sel 1 = hi1.0 -> 4
                //     sel 2,3 = 0 -> pick hi1.{2,3} which are 0 in our LUT -> 6,7
                //   w[2] = perm(a=lo2, b=t): out = {t.0, t.1, lo2.0, lo2.1}
                //     sel 0 = t.0 -> 0, sel 1 = t.1 -> 1, sel 2 = lo2.0 -> 4, sel 3 = lo2.1 -> 5
                const uint32_t t_w2 = __builtin_amdgcn_perm(hi1, lo1, 0x07060403u);
                w_u32[2]            = __builtin_amdgcn_perm(lo2, t_w2, 0x05040100u);
                // w[3] = {b2.v2, b2.v3, b2.v4, b3.v0}
                //   t = perm(a=hi2, b=lo2): {lo2.2, lo2.3, hi2.0, 0}
                //     sel 0 = lo2.2 -> 2, sel 1 = lo2.3 -> 3, sel 2 = hi2.0 -> 4, sel 3 = 0 -> 7
                //   w[3] = perm(a=lo3, b=t): {t.0, t.1, t.2, lo3.0}
                //     sel 0,1,2 = t.{0,1,2} -> 0,1,2  sel 3 = lo3.0 -> 4
                const uint32_t t_w3 = __builtin_amdgcn_perm(hi2, lo2, 0x07040302u);
                w_u32[3]            = __builtin_amdgcn_perm(lo3, t_w3, 0x04020100u);
                // w[4] = {b3.v1, b3.v2, b3.v3, b3.v4}
                //   perm(a=hi3, b=lo3): sel 0..3 from lo3, 4..7 from hi3
                //   out = {lo3.1, lo3.2, lo3.3, hi3.0} -> 1,2,3,4
                w_u32[4] = __builtin_amdgcn_perm(hi3, lo3, 0x04030201u);

                const int x_offset = mg * WEIGHTS_PER_MACROGROUP;
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

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int32_t v = acc[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset);
        }
        const int my_row = warp_row_base + r;
        if (lane == 0 && my_row < M) {
            // FP16 clamp — per-row absmean scales on post-hoc requantized
            // models can drive the product past 65504 (fp16 max); __float2half
            // would clamp to +Inf, and downstream RMSNorm squares that into
            // NaN across the residual stream. Matches the guard Sherry added.
            constexpr float FP16_MAX = 65504.0f;
            float out = (float)v * x_scale * scales[my_row];
            out = fminf(fmaxf(out, -FP16_MAX), FP16_MAX);
            y_f16[my_row] = __float2half(out);
        }
    }
}

extern "C" void ternary_gemv_tq1_halo_f16_launch(
    const void* packed, const void* x_i8, float x_scale,
    const void* scales, void* y_f16,
    int M, int K_padded, void* stream)
{
    hipStream_t s = (hipStream_t)stream;
    std::call_once(tq1_lut_flag, init_tq1_lut, s);
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_tq1_halo_kernel,
                       grid, block, 0, s,
                       static_cast<const uint32_t*>(packed),
                       static_cast<const int8_t*>(x_i8),
                       x_scale,
                       static_cast<const float*>(scales),
                       static_cast<__half*>(y_f16),
                       M, K_padded);
}
